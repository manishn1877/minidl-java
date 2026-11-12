package minidl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One served model: several worker threads share ONE job queue and answer requests in dynamic batches.
 * (The same ideas as the Java part of DJL Serving: WorkLoadManager / WorkerPool / BatchAggregator.)
 *
 * <ul>
 *  <li>Every worker owns its OWN copy of the model, because a layer remembers its inputs for backpropagation and is
 *      not safe to share between threads.</li>
 *  <li>A worker takes a job, then keeps taking more until {@code batchSize} rows are collected or {@code maxBatchDelayMs}
 *      have passed, stacks them into one matrix, runs ONE forward pass, and hands each caller its own rows.
 *      One big matrix multiplication is much faster than many small ones.</li>
 *  <li>The queue is bounded: when it is full a request is refused at once ({@link CapacityException}), so the server
 *      slows down gracefully instead of running out of memory.</li>
 *  <li>One bad request (wrong width) fails alone: its batch is retried request by request.</li>
 *  <li>Between {@code minWorkers} and {@code maxWorkers}: more workers are added when the queue grows.</li>
 * </ul>
 */
public final class ModelPool implements AutoCloseable {
    /** Settings of one served model. */
    public static final class Config {
        public String name, file;
        public int minWorkers = 1, maxWorkers = 1, batchSize = 16, maxBatchDelayMs = 5, queueSize = 100;
    }

    /** The queue is full: the caller should answer "503 busy, try again". */
    public static final class CapacityException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        public CapacityException(String m) { super(m); }
    }

    private static final class Job {
        final float[][] rows;
        final CompletableFuture<float[][]> result = new CompletableFuture<>();
        final long enqueued = System.nanoTime();
        Job(float[][] rows) { this.rows = rows; }
    }

    private final Config cfg;
    private final LinkedBlockingQueue<Job> queue;
    private final List<Worker> workers = new CopyOnWriteArrayList<>();
    private final ExecutorService scaler = Executors.newSingleThreadExecutor(r -> { Thread t = new Thread(r, "scaler-" + r.hashCode()); t.setDaemon(true); return t; });
    private final AtomicBoolean scaling = new AtomicBoolean();
    private volatile boolean closed;
    private int workerCounter;
    public final PoolStats stats = new PoolStats();

    public ModelPool(Config cfg) throws IOException {
        if (cfg.minWorkers < 1 || cfg.maxWorkers < cfg.minWorkers) throw new IllegalArgumentException("need 1 <= minWorkers <= maxWorkers");
        if (cfg.batchSize < 1 || cfg.queueSize < 1 || cfg.maxBatchDelayMs < 0) throw new IllegalArgumentException("bad batch_size / queue_size / max_batch_delay");
        this.cfg = cfg;
        this.queue = new LinkedBlockingQueue<>(cfg.queueSize);
        for (int i = 0; i < cfg.minWorkers; i++) addWorker(); // loads the model: a broken file fails here, at registration
    }

    public Config config() { return cfg; }
    public int workers() { return workers.size(); }
    public int queued() { return queue.size(); }

    /** Queues rows (one row per example) for prediction. Fails at once with CapacityException when the queue is full. */
    public CompletableFuture<float[][]> submit(float[][] rows) {
        if (closed) { CompletableFuture<float[][]> f = new CompletableFuture<>(); f.completeExceptionally(new IllegalStateException("model is being unloaded")); return f; }
        Job j = new Job(rows);
        stats.requests.incrementAndGet();
        if (!queue.offer(j)) {
            stats.rejected.incrementAndGet();
            j.result.completeExceptionally(new CapacityException("queue is full (" + cfg.queueSize + " waiting) for model " + cfg.name));
            return j.result;
        }
        if (workers.size() < cfg.maxWorkers && queue.size() > 2 * cfg.batchSize) scaleUpLater();
        return j.result;
    }

    private void scaleUpLater() {
        if (!scaling.compareAndSet(false, true)) return;
        scaler.execute(() -> {
            try { if (!closed && workers.size() < cfg.maxWorkers) addWorker(); }
            catch (Exception e) { System.out.println("could not add a worker for " + cfg.name + ": " + e); }
            finally { scaling.set(false); }
        });
    }

    /** Changes the number of workers (scale up loads more model copies, scale down stops the extra threads). */
    public synchronized void scale(int min, int max) throws IOException {
        if (min < 1 || max < min) throw new IllegalArgumentException("need 1 <= min_workers <= max_workers");
        cfg.minWorkers = min; cfg.maxWorkers = max;
        while (workers.size() < min) addWorker();
        while (workers.size() > max) { Worker w = workers.remove(workers.size() - 1); w.running = false; }
    }

    private synchronized void addWorker() throws IOException {
        Sequential model = ModelIO.loadModelBinary(cfg.file, new Random(1));
        Worker w = new Worker(model, ++workerCounter);
        workers.add(w);
        w.start();
    }

    private final class Worker extends Thread {
        final Sequential model;
        volatile boolean running = true;

        Worker(Sequential model, int n) { super("worker-" + cfg.name + "-" + n); this.model = model; setDaemon(true); }

        @Override public void run() {
            while (running && !closed) {
                try {
                    List<Job> batch = nextBatch();
                    if (batch != null) process(batch);
                } catch (InterruptedException e) { return; }
                catch (Throwable t) { System.out.println(getName() + ": " + t); }
            }
        }

        /** Waits for one job, then collects more until batchSize rows or maxBatchDelayMs. */
        private List<Job> nextBatch() throws InterruptedException {
            Job first = queue.poll(200, TimeUnit.MILLISECONDS);
            if (first == null) return null;
            List<Job> jobs = new ArrayList<>();
            jobs.add(first);
            int rows = first.rows.length;
            long deadline = System.nanoTime() + cfg.maxBatchDelayMs * 1_000_000L;
            while (rows < cfg.batchSize) {
                Job j = queue.poll();
                if (j == null) {
                    long left = deadline - System.nanoTime();
                    if (left <= 0) break;
                    j = queue.poll(left, TimeUnit.NANOSECONDS);
                    if (j == null) break;
                }
                jobs.add(j);
                rows += j.rows.length;
            }
            return jobs;
        }

        private void process(List<Job> batch) {
            long start = System.nanoTime();
            Map<Integer, List<Job>> byWidth = new LinkedHashMap<>(); // rows of different widths cannot share one matrix
            for (Job j : batch) byWidth.computeIfAbsent(j.rows.length == 0 ? 0 : j.rows[0].length, k -> new ArrayList<>()).add(j);
            for (Map.Entry<Integer, List<Job>> g : byWidth.entrySet()) {
                List<Job> jobs = g.getValue();
                try { runGroup(jobs, g.getKey(), start); }
                catch (Exception e) {
                    if (jobs.size() == 1) fail(jobs.get(0), e);
                    else for (Job j : jobs) { try { runGroup(new ArrayList<>(List.of(j)), g.getKey(), start); } catch (Exception e2) { fail(j, e2); } } // isolate the bad one
                }
            }
        }

        private void runGroup(List<Job> jobs, int width, long start) {
            int total = 0;
            for (Job j : jobs) total += j.rows.length;
            if (total == 0 || width == 0) throw new IllegalArgumentException("empty input");
            Tensor x = new Tensor(total, width);
            int r = 0;
            for (Job j : jobs) for (float[] row : j.rows) { System.arraycopy(row, 0, x.data, r * width, width); r++; }
            Tensor out = model.predict(x);
            long end = System.nanoTime();
            stats.batches.incrementAndGet();
            stats.batchRows.addAndGet(total);
            r = 0;
            for (Job j : jobs) {
                float[][] res = new float[j.rows.length][out.cols];
                for (int a = 0; a < res.length; a++) { for (int c = 0; c < out.cols; c++) res[a][c] = (float) out.get(r, c); r++; }
                stats.record((end - j.enqueued) / 1000, (start - j.enqueued) / 1000);
                j.result.complete(res);
            }
        }

        private void fail(Job j, Exception e) {
            stats.errors.incrementAndGet();
            j.result.completeExceptionally(e);
        }
    }

    @Override public void close() {
        closed = true;
        for (Worker w : workers) w.running = false;
        scaler.shutdownNow();
        Job j;
        while ((j = queue.poll()) != null) j.result.completeExceptionally(new IllegalStateException("model unloaded"));
    }

    /** Counters and latency numbers of one model (the source of /metrics). */
    public static final class PoolStats {
        public final AtomicLong requests = new AtomicLong(), errors = new AtomicLong(), rejected = new AtomicLong(), batches = new AtomicLong(), batchRows = new AtomicLong();
        /** Upper edges of the latency histogram, in microseconds. */
        public static final long[] EDGES = {1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000};
        public final AtomicLong[] bucket = new AtomicLong[EDGES.length + 1];
        public final AtomicLong latencySumMicros = new AtomicLong(), queueWaitSumMicros = new AtomicLong(), answered = new AtomicLong();
        private final long[] ring = new long[1024];
        private int ringPos, ringCount;

        PoolStats() { for (int i = 0; i < bucket.length; i++) bucket[i] = new AtomicLong(); }

        void record(long latencyMicros, long queueMicros) {
            int b = 0;
            while (b < EDGES.length && latencyMicros > EDGES[b]) b++;
            bucket[b].incrementAndGet();
            answered.incrementAndGet();
            latencySumMicros.addAndGet(latencyMicros);
            queueWaitSumMicros.addAndGet(queueMicros);
            synchronized (ring) { ring[ringPos] = latencyMicros; ringPos = (ringPos + 1) % ring.length; ringCount = Math.min(ringCount + 1, ring.length); }
        }

        /** Latency percentile (0..100) in milliseconds over the last 1024 answers; 0 if nothing was answered yet. */
        public double percentileMs(double p) {
            long[] copy;
            synchronized (ring) { copy = java.util.Arrays.copyOf(ring, ringCount); }
            if (copy.length == 0) return 0;
            java.util.Arrays.sort(copy);
            return copy[Math.min(copy.length - 1, (int) Math.ceil(p / 100.0 * copy.length) - 1 < 0 ? 0 : (int) Math.ceil(p / 100.0 * copy.length) - 1)] / 1000.0;
        }
    }
}
