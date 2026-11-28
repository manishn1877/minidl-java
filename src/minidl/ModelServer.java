package minidl;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.regex.Pattern;

/**
 * A model server in pure Java (JDK web server only), shaped after the Java side of DJL Serving.
 *
 * <pre>
 *   java -jar minidl.jar serve [--port 8080] [--store DIR] [--model name=file.bin[,batch=16][,delay=5][,workers=1][,max_workers=4][,queue=100]]...
 * </pre>
 * Without --model every *.bin file of the store folder is loaded. The API (same names as DJL Serving):
 * <pre>
 *   GET    /ping                       health
 *   GET    /models                     list the served models
 *   GET    /models/{name}              details: workers, queue, batch settings, latency percentiles
 *   POST   /models?url=f.bin&amp;name=n&amp;batch_size=16&amp;max_batch_delay=5&amp;min_workers=1&amp;max_workers=4&amp;queue_size=100   register
 *   PUT    /models/{name}?min_workers=2&amp;max_workers=4                                           scale
 *   DELETE /models/{name}              unload
 *   POST   /predictions/{name}         body {"inputs": [[...], [...]]} or {"input": [...]}  ->  {"outputs": [[...]], "argmax": [...]}
 *   GET    /metrics                    Prometheus text format
 * </pre>
 * Safety: listens on localhost only; "url" is just a file name INSIDE the model store (no paths); request bodies are limited
 * to 8 MB; a full queue answers 503; wrong input answers 400 and never disturbs other requests.
 */
public class ModelServer {
    static final int MAX_BODY = 8 << 20, MAX_ROWS = 10_000;
    static final Pattern NAME = Pattern.compile("[A-Za-z0-9_.-]{1,64}");

    private final Map<String, ModelPool> pools = new ConcurrentHashMap<>();
    private final File store;
    private HttpServer http;
    private ExecutorService httpExecutor; // the HTTP worker threads are non-daemon: stop() must shut them down, or the JVM never exits

    public ModelServer(File store) { this.store = store; }

    public static void main(String[] args) throws Exception {
        int port = 8080;
        File store = new File(".");
        List<String> specs = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port": port = Integer.parseInt(args[++i]); break;
                case "--store": store = new File(args[++i]); break;
                case "--model": specs.add(args[++i]); break;
                default: System.out.println("usage: serve [--port 8080] [--store DIR] [--model name=file.bin[,batch=16][,delay=5][,workers=1][,max_workers=4][,queue=100]]..."); return;
            }
        }
        ModelServer s = new ModelServer(store);
        if (specs.isEmpty()) {
            File[] bins = store.listFiles((d, n) -> n.endsWith(".bin") && !n.contains("-tuned"));
            if (bins != null) java.util.Arrays.sort(bins);
            if (bins != null) for (File f : bins) specs.add(f.getName().replaceAll("\\.bin$", "").replaceAll("[^A-Za-z0-9_.-]", "_") + "=" + f.getName());
        }
        for (String spec : specs) {
            try { s.register(parseSpec(spec)); System.out.println("loaded " + spec); }
            catch (Exception e) { System.out.println("skipped " + spec + ": " + e.getMessage()); }
        }
        s.start(port);
        System.out.println("MiniDL model server on http://localhost:" + s.port() + "  (" + s.pools.size() + " models; try GET /models, /ping, /metrics)");
    }

    static ModelPool.Config parseSpec(String spec) {
        String[] parts = spec.split(",");
        int eq = parts[0].indexOf('=');
        if (eq < 1) throw new IllegalArgumentException("model must look like name=file.bin");
        ModelPool.Config c = new ModelPool.Config();
        c.name = parts[0].substring(0, eq);
        c.file = parts[0].substring(eq + 1);
        for (int i = 1; i < parts.length; i++) {
            String[] kv = parts[i].split("=");
            if (kv.length != 2) throw new IllegalArgumentException("bad option '" + parts[i] + "'");
            int v = Integer.parseInt(kv[1]);
            switch (kv[0]) {
                case "batch": c.batchSize = v; break;
                case "delay": c.maxBatchDelayMs = v; break;
                case "workers": c.minWorkers = v; if (c.maxWorkers < v) c.maxWorkers = v; break;
                case "max_workers": c.maxWorkers = v; break;
                case "queue": c.queueSize = v; break;
                default: throw new IllegalArgumentException("unknown option '" + kv[0] + "'");
            }
        }
        return c;
    }

    // ------------------------------------------------------------------ registry
    /** Loads a model (its file is a name inside the store folder) and starts its workers. */
    public synchronized ModelPool register(ModelPool.Config c) throws IOException {
        if (!NAME.matcher(String.valueOf(c.name)).matches()) throw new IllegalArgumentException("model name must be 1-64 letters, digits, _ . -");
        if (pools.containsKey(c.name)) throw new IllegalStateException("model '" + c.name + "' is already registered");
        String f = String.valueOf(c.file);
        if (f.contains("/") || f.contains("\\") || f.startsWith(".") || !f.endsWith(".bin")) throw new IllegalArgumentException("url must be the name of a .bin file inside the model store (no paths)");
        File file = new File(store, f);
        if (!file.isFile()) throw new IllegalArgumentException("no such model file in the store: " + f);
        c.file = file.getPath();
        ModelPool p = new ModelPool(c);
        pools.put(c.name, p);
        return p;
    }

    public synchronized boolean unregister(String name) {
        ModelPool p = pools.remove(name);
        if (p == null) return false;
        p.close();
        return true;
    }

    public ModelPool pool(String name) { return pools.get(name); }

    public void start(int port) throws IOException {
        if (http != null) throw new IllegalStateException("server already started");
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0); // localhost only
        ExecutorService executor = Executors.newFixedThreadPool(16);
        try {
            server.createContext("/", ex -> guard(ex, this::route));
            server.setExecutor(executor);
            server.start();
        } catch (RuntimeException e) {
            executor.shutdownNow();
            server.stop(0);
            throw e;
        }
        http = server;
        httpExecutor = executor;
    }

    public int port() { return http.getAddress().getPort(); }

    /** Stops the HTTP server, releases its worker threads, and unloads every model. Safe to call twice. */
    public void stop() {
        try {
            if (http != null) http.stop(0);
            if (httpExecutor != null) shutdownAndAwait(httpExecutor);
        } finally {
            http = null;
            httpExecutor = null;
            for (ModelPool p : pools.values()) p.close();
            pools.clear();
        }
    }

    /** Shuts an executor down, waits for it to stop, and keeps the caller's interrupt flag if interrupted while waiting. */
    static void shutdownAndAwait(ExecutorService executor) {
        executor.shutdownNow();
        try {
            if (!executor.awaitTermination(10, TimeUnit.SECONDS))
                throw new IllegalStateException("HTTP worker threads did not stop within 10 s");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ------------------------------------------------------------------ HTTP
    interface Handler { void run(HttpExchange ex) throws Exception; }

    static final class HttpError extends RuntimeException {
        private static final long serialVersionUID = 1L;
        final int code;
        HttpError(int code, String m) { super(m); this.code = code; }
    }

    private void guard(HttpExchange ex, Handler h) throws IOException {
        try { h.run(ex); }
        catch (HttpError e) { send(ex, e.code, "{\"code\":" + e.code + ",\"error\":" + MiniJson.quote(e.getMessage()) + "}"); }
        catch (IllegalArgumentException e) { send(ex, 400, "{\"code\":400,\"error\":" + MiniJson.quote(e.getMessage()) + "}"); }
        catch (IllegalStateException e) { send(ex, 409, "{\"code\":409,\"error\":" + MiniJson.quote(e.getMessage()) + "}"); }
        catch (Exception e) { send(ex, 500, "{\"code\":500,\"error\":" + MiniJson.quote(String.valueOf(e)) + "}"); }
    }

    private void route(HttpExchange ex) throws Exception {
        String[] seg = ex.getRequestURI().getPath().replaceAll("^/+|/+$", "").split("/");
        String method = ex.getRequestMethod();
        Map<String, String> q = query(ex);
        if (seg.length == 1 && seg[0].equals("ping")) { send(ex, 200, "{\"status\":\"Healthy\"}"); return; }
        if (seg.length == 1 && seg[0].equals("metrics")) { send(ex, 200, "text/plain; version=0.0.4", metrics()); return; }
        if (seg.length >= 1 && seg[0].equals("predictions") && seg.length == 2) { if (!method.equals("POST")) throw new HttpError(405, "use POST"); predict(ex, seg[1]); return; }
        if (seg.length >= 1 && seg[0].equals("models")) {
            if (seg.length == 1 && method.equals("GET")) { send(ex, 200, listModels()); return; }
            if (seg.length == 1 && method.equals("POST")) {
                ModelPool.Config c = new ModelPool.Config();
                c.file = q.get("url");
                c.name = q.getOrDefault("name", c.file == null ? null : c.file.replaceAll("\\.bin$", ""));
                c.batchSize = intParam(q, "batch_size", c.batchSize);
                c.maxBatchDelayMs = intParam(q, "max_batch_delay", c.maxBatchDelayMs);
                c.queueSize = intParam(q, "queue_size", c.queueSize);
                c.minWorkers = intParam(q, "min_workers", c.minWorkers);
                c.maxWorkers = intParam(q, "max_workers", Math.max(c.minWorkers, c.maxWorkers));
                register(c);
                send(ex, 201, "{\"status\":\"Model \\\"" + c.name + "\\\" registered\"}");
                return;
            }
            if (seg.length == 2) {
                ModelPool p = pools.get(seg[1]);
                if (p == null) throw new HttpError(404, "model not found: " + seg[1]);
                if (method.equals("GET")) { send(ex, 200, describe(seg[1], p)); return; }
                if (method.equals("DELETE")) { unregister(seg[1]); send(ex, 200, "{\"status\":\"Model \\\"" + seg[1] + "\\\" unregistered\"}"); return; }
                if (method.equals("PUT")) {
                    int min = intParam(q, "min_workers", p.config().minWorkers);
                    p.scale(min, intParam(q, "max_workers", Math.max(min, p.config().maxWorkers)));
                    send(ex, 200, describe(seg[1], p));
                    return;
                }
            }
            throw new HttpError(405, "method not allowed here");
        }
        throw new HttpError(404, "not found: " + ex.getRequestURI().getPath());
    }

    private void predict(HttpExchange ex, String name) throws Exception {
        ModelPool p = pools.get(name);
        if (p == null) throw new HttpError(404, "model not found: " + name);
        byte[] raw = ex.getRequestBody().readNBytes(MAX_BODY + 1);
        if (raw.length > MAX_BODY) throw new HttpError(413, "request body is larger than " + MAX_BODY + " bytes");
        float[][] rows = parseRows(new String(raw, StandardCharsets.UTF_8));
        CompletableFuture<float[][]> f = p.submit(rows);
        float[][] out;
        try { out = f.get(30, TimeUnit.SECONDS); }
        catch (ExecutionException e) {
            Throwable c = e.getCause();
            if (c instanceof ModelPool.CapacityException) throw new HttpError(503, c.getMessage());
            if (c instanceof IllegalStateException) throw new HttpError(503, c.getMessage());
            throw new HttpError(400, "the model rejected the input: " + c.getMessage());
        } catch (TimeoutException e) { throw new HttpError(504, "no answer within 30 s"); }
        StringBuilder sb = new StringBuilder("{\"outputs\":[");
        StringBuilder am = new StringBuilder("],\"argmax\":[");
        for (int r = 0; r < out.length; r++) {
            if (r > 0) { sb.append(','); am.append(','); }
            sb.append('[');
            int best = 0;
            for (int c = 0; c < out[r].length; c++) {
                if (c > 0) sb.append(',');
                float v = out[r][c];
                sb.append(Float.isFinite(v) ? Float.toString(v) : "null");
                if (out[r][c] > out[r][best]) best = c;
            }
            sb.append(']');
            am.append(best);
        }
        send(ex, 200, sb.append(am).append("]}").toString());
    }

    /** {"inputs": [[...], ...]}, {"input": [...]}, or just an array (one row of numbers, or a list of rows). */
    static float[][] parseRows(String body) {
        Object v = MiniJson.parse(body);
        if (v instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) v;
            v = m.containsKey("inputs") ? m.get("inputs") : m.get("input");
            if (v == null) throw new IllegalArgumentException("the JSON needs an \"inputs\" (list of rows) or \"input\" (one row)");
        }
        if (!(v instanceof List) || ((List<?>) v).isEmpty()) throw new IllegalArgumentException("inputs must be a non-empty list of numbers or of rows of numbers");
        List<?> l = (List<?>) v;
        List<?> rowsList = l.get(0) instanceof List ? l : java.util.Collections.singletonList(l);
        if (rowsList.size() > MAX_ROWS) throw new IllegalArgumentException("at most " + MAX_ROWS + " rows per request");
        float[][] rows = new float[rowsList.size()][];
        for (int r = 0; r < rows.length; r++) {
            if (!(rowsList.get(r) instanceof List)) throw new IllegalArgumentException("row " + r + " is not a list of numbers");
            List<?> row = (List<?>) rowsList.get(r);
            if (row.isEmpty()) throw new IllegalArgumentException("row " + r + " is empty");
            rows[r] = new float[row.size()];
            for (int c = 0; c < row.size(); c++) {
                if (!(row.get(c) instanceof Double)) throw new IllegalArgumentException("row " + r + ", item " + c + " is not a number");
                rows[r][c] = ((Double) row.get(c)).floatValue();
            }
            if (rows[r].length != rows[0].length) throw new IllegalArgumentException("all rows must have the same length (row 0 has " + rows[0].length + ", row " + r + " has " + rows[r].length + ")");
        }
        return rows;
    }

    private String listModels() {
        StringBuilder sb = new StringBuilder("{\"models\":[");
        boolean first = true;
        for (Map.Entry<String, ModelPool> e : new java.util.TreeMap<>(pools).entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append("{\"modelName\":").append(MiniJson.quote(e.getKey())).append(",\"workers\":").append(e.getValue().workers()).append('}');
        }
        return sb.append("]}").toString();
    }

    private String describe(String name, ModelPool p) {
        ModelPool.Config c = p.config();
        ModelPool.PoolStats s = p.stats;
        return String.format("{\"modelName\":%s,\"file\":%s,\"minWorkers\":%d,\"maxWorkers\":%d,\"workers\":%d,\"batchSize\":%d,\"maxBatchDelay\":%d,\"queueSize\":%d,\"queued\":%d,"
            + "\"requests\":%d,\"errors\":%d,\"rejected\":%d,\"batches\":%d,\"avgBatchRows\":%.2f,\"latencyMs\":{\"p50\":%.3f,\"p99\":%.3f}}",
            MiniJson.quote(name), MiniJson.quote(new File(c.file).getName()), c.minWorkers, c.maxWorkers, p.workers(), c.batchSize, c.maxBatchDelayMs, c.queueSize, p.queued(),
            s.requests.get(), s.errors.get(), s.rejected.get(), s.batches.get(), s.batches.get() == 0 ? 0.0 : (double) s.batchRows.get() / s.batches.get(), s.percentileMs(50), s.percentileMs(99));
    }

    /** Prometheus text format, one set of lines per model. */
    String metrics() {
        StringBuilder sb = new StringBuilder();
        String[][] help = {{"minidl_requests_total", "counter", "Requests received"}, {"minidl_errors_total", "counter", "Requests that failed"},
            {"minidl_rejected_total", "counter", "Requests refused because the queue was full"}, {"minidl_batches_total", "counter", "Forward passes run"},
            {"minidl_batch_rows_total", "counter", "Rows in all forward passes (rows/batches = average batch size)"},
            {"minidl_queue_size", "gauge", "Requests waiting"}, {"minidl_workers", "gauge", "Worker threads"}};
        Map<String, ModelPool> sorted = new java.util.TreeMap<>(pools);
        for (String[] h : help) {
            sb.append("# HELP ").append(h[0]).append(' ').append(h[2]).append("\n# TYPE ").append(h[0]).append(' ').append(h[1]).append('\n');
            for (Map.Entry<String, ModelPool> e : sorted.entrySet()) {
                ModelPool.PoolStats s = e.getValue().stats;
                long v = h[0].equals("minidl_requests_total") ? s.requests.get() : h[0].equals("minidl_errors_total") ? s.errors.get() : h[0].equals("minidl_rejected_total") ? s.rejected.get()
                    : h[0].equals("minidl_batches_total") ? s.batches.get() : h[0].equals("minidl_batch_rows_total") ? s.batchRows.get() : h[0].equals("minidl_queue_size") ? e.getValue().queued() : e.getValue().workers();
                sb.append(h[0]).append("{model=\"").append(e.getKey()).append("\"} ").append(v).append('\n');
            }
        }
        sb.append("# HELP minidl_request_latency_seconds Time from request arrival to answer\n# TYPE minidl_request_latency_seconds histogram\n");
        for (Map.Entry<String, ModelPool> e : sorted.entrySet()) {
            ModelPool.PoolStats s = e.getValue().stats;
            long cum = 0;
            for (int i = 0; i < ModelPool.PoolStats.EDGES.length; i++) {
                cum += s.bucket[i].get();
                sb.append("minidl_request_latency_seconds_bucket{model=\"").append(e.getKey()).append("\",le=\"").append(ModelPool.PoolStats.EDGES[i] / 1e6).append("\"} ").append(cum).append('\n');
            }
            cum += s.bucket[ModelPool.PoolStats.EDGES.length].get();
            sb.append("minidl_request_latency_seconds_bucket{model=\"").append(e.getKey()).append("\",le=\"+Inf\"} ").append(cum).append('\n');
            sb.append("minidl_request_latency_seconds_sum{model=\"").append(e.getKey()).append("\"} ").append(s.latencySumMicros.get() / 1e6).append('\n');
            sb.append("minidl_request_latency_seconds_count{model=\"").append(e.getKey()).append("\"} ").append(s.answered.get()).append('\n');
        }
        return sb.toString();
    }

    private static int intParam(Map<String, String> q, String k, int dflt) {
        String v = q.get(k);
        if (v == null) return dflt;
        try { return Integer.parseInt(v); } catch (NumberFormatException e) { throw new IllegalArgumentException(k + " must be a whole number"); }
    }

    private static Map<String, String> query(HttpExchange ex) {
        Map<String, String> m = new HashMap<>();
        String q = ex.getRequestURI().getRawQuery();
        if (q != null) for (String kv : q.split("&")) {
            int i = kv.indexOf('=');
            if (i > 0) m.put(java.net.URLDecoder.decode(kv.substring(0, i), StandardCharsets.UTF_8), java.net.URLDecoder.decode(kv.substring(i + 1), StandardCharsets.UTF_8));
        }
        return m;
    }

    private static void send(HttpExchange ex, int code, String body) throws IOException { send(ex, code, "application/json", body); }

    private static void send(HttpExchange ex, int code, String type, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", type);
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.sendResponseHeaders(code, b.length);
        ex.getResponseBody().write(b);
        ex.close();
    }
}
