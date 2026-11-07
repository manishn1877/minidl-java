package minidl;

import java.util.List;
import java.util.Random;

/**
 * Speed test:  java -cp out minidl.Benchmark
 * Part 1: matrix multiply, textbook version vs the optimized one.
 * Part 2: a real training epoch (MNIST-sized network), "before" style layer vs the current Dense.
 * Part 3: convolution speed. Part 4: memory of the data. Part 5: int8 versus float32 prediction.
 */
public class Benchmark {
    public static void main(String[] args) {
        System.out.println("CPU cores: " + Runtime.getRuntime().availableProcessors());
        kernelBench();
        autogradBench();
        matmulBench();
        trainingBench();
        convBench();
        memoryBench();
        int8Bench();
    }

    static double bestMs(Runnable r, int warmup, int runs) {
        for (int i = 0; i < warmup; i++) r.run();
        double best = Double.MAX_VALUE;
        for (int i = 0; i < runs; i++) {
            long t = System.nanoTime();
            r.run();
            best = Math.min(best, (System.nanoTime() - t) / 1e6);
        }
        return best;
    }

    static void matmulBench() {
        int n = 512;
        Random rnd = new Random(1);
        Tensor a = Tensor.randn(n, n, 1, rnd), b = Tensor.randn(n, n, 1, rnd);
        double flops = 2.0 * n * n * n;
        System.out.printf("%n--- matrix multiply %dx%d . %dx%d ---%n", n, n, n, n);

        double naive = bestMs(() -> a.matmulNaive(b), 1, 3);
        System.out.printf("textbook (i-j-k, get())      %8.1f ms  %6.2f GFLOP/s%n", naive, flops / naive / 1e6);

        boolean old = Tensor.isParallel();
        Tensor.setParallel(false);
        double one = bestMs(() -> a.matmul(b), 2, 5);
        System.out.printf("optimized, 1 thread          %8.1f ms  %6.2f GFLOP/s  (%.1fx faster)%n", one, flops / one / 1e6, naive / one);

        Tensor.setParallel(true);
        double par = bestMs(() -> a.matmul(b), 2, 5);
        System.out.printf("optimized, all cores         %8.1f ms  %6.2f GFLOP/s  (%.1fx faster)%n", par, flops / par / 1e6, naive / par);
        Tensor.setParallel(old);
    }

    /** The Dense layer as it was before the speed work: makes transposed copies, extra bias copy. */
    static class OldStyleDense implements Layer {
        final Param W, b; Tensor in;
        OldStyleDense(int i, int o, Random r) { W = new Param(Tensor.randn(i, o, Math.sqrt(2.0 / i), r)); b = new Param(new Tensor(1, o)); }
        public Tensor forward(Tensor x) { in = x; return x.matmul(W.value).addRowVector(b.value); }
        public Tensor backward(Tensor g) {
            Tensor dW = in.transpose().matmul(g), db = g.sumRows();
            System.arraycopy(dW.data, 0, W.grad.data, 0, dW.data.length);
            System.arraycopy(db.data, 0, b.grad.data, 0, db.data.length);
            return g.matmul(W.value.transpose());
        }
        public List<Param> params() { return List.of(W, b); }
    }

    static void trainingBench() {
        int n = 6400, in = 784, hidden = 128, classes = 10, batch = 64;
        Random rnd = new Random(2);
        Tensor x = Tensor.randn(n, in, 1, rnd), y = new Tensor(n, classes);
        for (int i = 0; i < n; i++) y.set(i, rnd.nextInt(classes), 1);
        System.out.printf("%n--- one training epoch: %d samples, 784 -> 128 -> 10, batch %d ---%n", n, batch);

        boolean old = Tensor.isParallel();

        Sequential before = new Sequential()
            .add(new OldStyleDense(in, hidden, rnd)).add(new Activations.ReLU())
            .add(new OldStyleDense(hidden, classes, rnd));
        Tensor.setParallel(false);
        double tb = bestMs(() -> before.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001), 1, batch, 0, 0), 1, 3);
        System.out.printf("before (transpose copies, 1 thread)    %8.0f ms/epoch%n", tb);

        Sequential after = new Sequential()
            .add(new Dense(in, hidden, rnd)).add(new Activations.ReLU())
            .add(new Dense(hidden, classes, rnd));
        double ta1 = bestMs(() -> after.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001), 1, batch, 0, 0), 1, 3);
        System.out.printf("now    (fused matmul, 1 thread)        %8.0f ms/epoch  (%.2fx faster)%n", ta1, tb / ta1);

        Tensor.setParallel(true);
        double ta = bestMs(() -> after.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001), 1, batch, 0, 0), 1, 3);
        System.out.printf("now    (fused matmul, all cores)       %8.0f ms/epoch  (%.2fx faster)%n", ta, tb / ta);
        Tensor.setParallel(old);
    }

    static void convBench() {
        int n = 640;
        Random rnd = new Random(3);
        Tensor x = Tensor.randn(n, 784, 1, rnd), y = new Tensor(n, 10);
        for (int i = 0; i < n; i++) y.set(i, rnd.nextInt(10), 1);
        Sequential cnn = MnistDemo.cnn(rnd);
        System.out.printf("%n--- small CNN (%,d weights): one training epoch on %d images, batch 64 ---%n", cnn.paramCount(), n);
        double ms = bestMs(() -> cnn.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001), 1, 64, 0, 0), 1, 3);
        System.out.printf("%.0f ms per %d images  ->  about %.1f s for 54,000 images%n", ms, n, ms * 54000 / n / 1000);
    }

    static long usedHeap() {
        for (int i = 0; i < 3; i++) System.gc();
        Runtime r = Runtime.getRuntime();
        return r.totalMemory() - r.freeMemory();
    }

    static void memoryBench() {
        int n = 60000, dim = 784;
        System.out.printf("%n--- memory to hold %d MNIST-sized images (%d pixels each) ---%n", n, dim);
        long base = usedHeap();
        byte[] bytes = new byte[n * dim];
        long asBytes = usedHeap() - base;
        float[] floats = new float[n * dim];
        long asFloat = usedHeap() - base - asBytes;
        System.out.printf("as double (previous version, arithmetic)  %6.0f MB%n", n * (long) dim * 8 / 1e6);
        System.out.printf("as float32 (measured)                     %6.0f MB%n", asFloat / 1e6);
        System.out.printf("as bytes   (measured, ByteSource)         %6.0f MB%n", asBytes / 1e6);
        if (bytes.length + floats.length == 0) System.out.println(); // keep both arrays alive until here
    }

    static void int8Bench() {
        int n = 10000;
        Random rnd = new Random(4);
        Sequential f = new Sequential()
            .add(new Dense(784, 512, rnd)).add(new Activations.ReLU())
            .add(new Dense(512, 512, rnd)).add(new Activations.ReLU())
            .add(new Dense(512, 10, rnd));
        Sequential q = Quantizer.quantize(f);
        Tensor x = Tensor.randn(n, 784, 1, rnd);
        System.out.printf("%n--- prediction, 784 -> 512 -> 512 -> 10, %d inputs ---%n", n);
        double tf = bestMs(() -> f.predict(x), 1, 3), tq = bestMs(() -> q.predict(x), 1, 3);
        Tensor a = f.predict(x), b = q.predict(x);
        double maxDiff = 0, maxAbs = 0;
        for (int i = 0; i < a.data.length; i++) { maxDiff = Math.max(maxDiff, Math.abs(a.data[i] - b.data[i])); maxAbs = Math.max(maxAbs, Math.abs(a.data[i])); }
        int same = 0;
        for (int i = 0; i < n; i++) if (a.argmaxRow(i) == b.argmaxRow(i)) same++;
        System.out.printf("float32 weights  %7.0f KB   %6.0f ms%n", f.weightBytes() / 1024.0, tf);
        System.out.printf("int8 weights     %7.0f KB   %6.0f ms   (%.1fx less memory, %.2fx speed)%n", q.weightBytes() / 1024.0, tq, (double) f.weightBytes() / q.weightBytes(), tf / tq);
        System.out.printf("same predicted class on %.1f%% of inputs; biggest output difference %.4f (outputs up to %.2f)%n", 100.0 * same / n, maxDiff, maxAbs);
    }

    /** Portable Java loops versus the Vector API kernels (needs --add-modules jdk.incubator.vector to show the second). */
    static void kernelBench() {
        System.out.println("\n=== kernels: " + Tensor.kernelName() + " ===");
        if (!Tensor.vectorAvailable()) {
            System.out.println("(Vector API not enabled. Run with:  java --add-modules jdk.incubator.vector -cp out minidl.Benchmark)");
            return;
        }
        Random rnd = new Random(8);
        System.out.println("matrix multiply:");
        for (int n : new int[]{256, 512, 1024}) {
            Tensor a = Tensor.randn(n, n, 1, rnd), b = Tensor.randn(n, n, 1, rnd);
            double flops = 2.0 * n * n * n;
            Tensor.setVector(false);
            double ps = bestMs(() -> a.matmul(b), 2, 5);
            Tensor.setVector(true);
            double pv = bestMs(() -> a.matmul(b), 6, 9);
            System.out.printf("  %4d x %4d   portable %8.1f ms (%5.1f GFLOP/s)   vector %8.1f ms (%5.1f GFLOP/s)   %.1fx%n", n, n, ps, flops / ps / 1e6, pv, flops / pv / 1e6, ps / pv);
        }

        System.out.println("whole training epochs (1 thread):");
        int n = 6400;
        Tensor x = Tensor.randn(n, 784, 1, rnd), y = new Tensor(n, 10);
        for (int i = 0; i < n; i++) y.set(i, rnd.nextInt(10), 1);
        Sequential mlp = MnistDemo.mlp(rnd);
        timeBoth("plain network 784-128-64-10, 6,400 samples", () -> mlp.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001), 1, 64, 0, 0));
        Sequential cnn = MnistDemo.cnn(rnd);
        Tensor xc = x.sliceRows(0, 640), yc = y.sliceRows(0, 640);
        timeBoth("CNN, 640 images", () -> cnn.fit(xc, yc, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001), 1, 64, 0, 0));
        Sequential tf = CharTransformerDemo.buildModel(65, rnd);
        Tensor ids = new Tensor(320, 32), tgt = new Tensor(320 * 32, 65);
        for (int i = 0; i < ids.data.length; i++) { ids.data[i] = rnd.nextInt(65); tgt.data[i * 65 + rnd.nextInt(65)] = 1f; }
        DataSource tfData = new DataSource() {
            public int size() { return 320; }
            public Tensor features(int[] idx, int f, int t) { return ids.selectRows(idx, f, t); }
            public Tensor targets(int[] idx, int f, int t) {
                Tensor o = new Tensor((t - f) * 32, 65);
                for (int r = f; r < t; r++) System.arraycopy(tgt.data, idx[r] * 32 * 65, o.data, (r - f) * 32 * 65, 32 * 65);
                return o;
            }
        };
        timeBoth("transformer, 10 steps of 32 sequences", () -> new Trainer().epochs(1).batchSize(32).fit(tf, tfData, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001)));
        Tensor.setVector(true);
    }

    private static void timeBoth(String name, Runnable r) {
        boolean oldPar = Tensor.isParallel();
        Tensor.setParallel(false);
        Tensor.setVector(false);
        double p = bestMs(r, 1, 3);
        Tensor.setVector(true);
        double v = bestMs(r, 2, 3);
        Tensor.setParallel(oldPar);
        System.out.printf("  %-46s portable %7.0f ms   vector %7.0f ms   %.2fx%n", name, p, v, p / v);
    }

    /** Hand-written layers versus the same layers written with autograd: speed and memory. */
    static void autogradBench() {
        System.out.println("\n=== autograd layers versus hand-written layers ===");
        Random rnd = new Random(12);
        int n = 3200;
        Tensor x = Tensor.randn(n, 784, 1, rnd), y = new Tensor(n, 10);
        for (int i = 0; i < n; i++) y.set(i, rnd.nextInt(10), 1);
        Layer r1 = new AutoLayer("AutoReLU", java.util.List.of(), (v, p) -> v.relu()), r2 = new AutoLayer("AutoReLU", java.util.List.of(), (v, p) -> v.relu());
        Sequential hand = new Sequential().add(new Dense(784, 128, rnd)).add(new Activations.ReLU()).add(new Dense(128, 10, rnd));
        Sequential auto = new Sequential().add(AutoLayers.dense(784, 128, rnd)).add(r1).add(AutoLayers.dense(128, 10, rnd));
        double th = bestMs(() -> hand.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001), 1, 64, 0, 0), 1, 3);
        double ta = bestMs(() -> auto.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001), 1, 64, 0, 0), 1, 3);
        System.out.printf("plain network, 3,200 samples/epoch   hand-written %6.0f ms   autograd %6.0f ms   (autograd %.2fx slower)%n", th, ta, ta / th);

        Sequential tfHand = CharTransformerDemo.buildModel(65, rnd), tfAuto = CharTransformerDemo.buildAutoModel(65, rnd);
        int T = 32, B = 8;
        Tensor ids = new Tensor(B, T), tg = new Tensor(B * T, 65);
        for (int i = 0; i < ids.data.length; i++) { ids.data[i] = rnd.nextInt(65); tg.data[i * 65 + rnd.nextInt(65)] = 1f; }
        Losses.SoftmaxCrossEntropy ce = new Losses.SoftmaxCrossEntropy();
        Optimizers.Adam o1 = new Optimizers.Adam(0.001), o2 = new Optimizers.Adam(0.001);
        double sh = bestMs(() -> tfHand.trainStep(ids, tg, ce, o1), 2, 5), sa = bestMs(() -> tfAuto.trainStep(ids, tg, ce, o2), 2, 5);
        System.out.printf("transformer, one step of 8 sequences  hand-written %6.0f ms   autograd %6.0f ms   (autograd %.2fx slower)%n", sh, sa, sa / sh);

        // memory kept between forward and backward
        System.out.println("memory kept between forward and backward (batch 1024, 784-512-512-10):");
        for (int mode = 0; mode < 2; mode++) {
            Sequential m = mode == 0
                ? new Sequential().add(new Dense(784, 512, rnd)).add(new Activations.ReLU()).add(new Dense(512, 512, rnd)).add(new Activations.ReLU()).add(new Dense(512, 10, rnd))
                : new Sequential().add(AutoLayers.dense(784, 512, rnd)).add(new AutoLayer("AutoReLU", java.util.List.of(), (v, p) -> v.relu())).add(AutoLayers.dense(512, 512, rnd))
                      .add(new AutoLayer("AutoReLU", java.util.List.of(), (v, p) -> v.relu())).add(AutoLayers.dense(512, 10, rnd));
            Tensor xb = Tensor.randn(1024, 784, 1, rnd);
            m.setTraining(true);
            long base = usedHeap();
            Tensor out = m.forward(xb);
            long held = usedHeap() - base;
            System.out.printf("  %-14s %7.2f MB%n", mode == 0 ? "hand-written" : "autograd", held / 1e6);
            if (out.rows < 0) System.out.println(out);
        }
    }
}
