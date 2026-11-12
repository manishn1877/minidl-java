package minidl;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

/** Tiny test runner (no JUnit needed). Run: java -cp out minidl.Tests */
public class Tests {
    static int passed = 0, failed = 0;
    static final double GC_TOL = 5e-3;      // float32 gradient-check tolerance (real bugs give 0.1+)
    static final double GC_TOL_DEEP = 2e-2; // deeper nets: float32 rounding noise adds up with depth

    static void check(String name, boolean ok) {
        System.out.println((ok ? "PASS  " : "FAIL  ") + name);
        if (ok) passed++; else failed++;
    }

    static double maxDiff(Tensor a, Tensor b) {
        double d = 0;
        for (int i = 0; i < a.data.length; i++) d = Math.max(d, Math.abs(a.data[i] - b.data[i]));
        return d;
    }

    public static void main(String[] args) throws Exception {
        matmul();
        fusedMatmul();
        parallelMatmul();
        transpose();
        softmax();
        gradients();
        lossValues();
        optimizers();
        weightDecayAndClip();
        schedules();
        normalizer();
        saveLoad();
        dropout();
        batchNormBehaviour();
        earlyStopping();
        csv();
        mnistFormat();
        metrics();
        convolution();
        pooling();
        byteData();
        quantization();
        binaryFiles();
        layerNormAndResidual();
        avgPoolAndReshape();
        transformerParts();
        autograd();
        augmentation();
        moreQuantization();
        mappedAndChar();
        kernelAgreement();
        convOptions();
        halfPrecision();
        mixedPrecisionTraining();
        ndTensor();
        shapeInference();
        autogradLayers();
        cifarAndColour();
        handwriting();
        letters();
        hostileFiles();
        alnum();
        segmentation();
        modelChoice();
        serving();
        phase4();
        javaOnly();
        System.out.printf("%n%d passed, %d failed%n", passed, failed);
        if (failed > 0) System.exit(1);
    }

    // ------------------------------------------------------------------ tensor

    static void matmul() {
        Tensor a = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        Tensor b = Tensor.of(new double[][]{{5, 6}, {7, 8}});
        Tensor c = a.matmul(b);
        check("matmul [[1,2],[3,4]].[[5,6],[7,8]] = [[19,22],[43,50]]",
              c.get(0, 0) == 19 && c.get(0, 1) == 22 && c.get(1, 0) == 43 && c.get(1, 1) == 50);
    }

    static void fusedMatmul() {
        Random r = new Random(4);
        Tensor a = Tensor.randn(17, 13, 1, r), b = Tensor.randn(13, 9, 1, r);
        check("optimized matmul equals textbook matmul", maxDiff(a.matmul(b), a.matmulNaive(b)) < 1e-4);

        Tensor p = Tensor.randn(17, 5, 1, r), q = Tensor.randn(17, 9, 1, r);
        check("matmulTA equals transpose().matmul()", maxDiff(p.matmulTA(q), p.transpose().matmul(q)) < 1e-4);

        Tensor s = Tensor.randn(17, 6, 1, r), t = Tensor.randn(11, 6, 1, r);
        check("matmulTB equals matmul(transpose())", maxDiff(s.matmulTB(t), s.matmul(t.transpose())) < 1e-4);

        boolean threw = false;
        try { a.matmul(a); } catch (IllegalArgumentException e) { threw = true; }
        check("shape mismatch gives a clear error", threw);
    }

    static void parallelMatmul() {
        Random r = new Random(8);
        Tensor a = Tensor.randn(220, 150, 1, r), b = Tensor.randn(150, 130, 1, r), c = Tensor.randn(220, 130, 1, r);
        boolean old = Tensor.isParallel();
        Tensor.setParallel(false);
        Tensor s1 = a.matmul(b), s2 = a.matmulTA(c), s3 = c.matmulTB(b);
        Tensor.setParallel(true);
        Tensor p1 = a.matmul(b), p2 = a.matmulTA(c), p3 = c.matmulTB(b);
        Tensor.setParallel(old);
        check("multi-thread results identical to single-thread",
              maxDiff(s1, p1) == 0 && maxDiff(s2, p2) == 0 && maxDiff(s3, p3) == 0);
    }

    static void transpose() {
        Tensor a = Tensor.of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor t = a.transpose();
        check("transpose shape and values", t.rows == 3 && t.cols == 2 && t.get(2, 1) == 6 && t.get(0, 1) == 4);
    }

    static void softmax() {
        Tensor z = Tensor.of(new double[][]{{2, 1, 0.1}, {1000, 1001, 1002}});
        Tensor p = Losses.SoftmaxCrossEntropy.softmax(z);
        boolean ok = true;
        for (int i = 0; i < p.rows; i++) {
            double s = 0;
            for (int j = 0; j < p.cols; j++) { s += p.get(i, j); if (Double.isNaN(p.get(i, j))) ok = false; }
            if (Math.abs(s - 1) > 1e-6) ok = false;
        }
        check("softmax rows sum to 1, no NaN even for huge scores", ok);
    }

    // ------------------------------------------------------------------ backprop

    static void gradients() {
        Random r = new Random(3);
        Tensor x = Tensor.randn(6, 4, 1.0, r);
        Tensor y = Tensor.randn(6, 2, 1.0, r);
        Tensor oneHot = new Tensor(6, 2);
        for (int i = 0; i < 6; i++) oneHot.set(i, i % 2, 1.0);
        Tensor bin = new Tensor(6, 1);
        for (int i = 0; i < 6; i++) bin.set(i, 0, i % 2);

        Sequential tanh = new Sequential().add(new Dense(4, 5, r)).add(new Activations.Tanh()).add(new Dense(5, 2, r));
        check("gradcheck Dense+Tanh+MSE", GradCheck.maxRelativeError(tanh, new Losses.MSE(), x, y) < GC_TOL);

        Sequential relu = new Sequential().add(new Dense(4, 5, r)).add(new Activations.ReLU()).add(new Dense(5, 2, r));
        check("gradcheck Dense+ReLU+MSE", GradCheck.maxRelativeError(relu, new Losses.MSE(), x, y) < GC_TOL);

        Sequential leaky = new Sequential().add(new Dense(4, 5, r)).add(new Activations.LeakyReLU()).add(new Dense(5, 2, r));
        check("gradcheck Dense+LeakyReLU+MSE", GradCheck.maxRelativeError(leaky, new Losses.MSE(), x, y) < GC_TOL);

        Sequential gelu = new Sequential().add(new Dense(4, 5, r)).add(new Activations.GELU()).add(new Dense(5, 2, r));
        check("gradcheck Dense+GELU+MSE", GradCheck.maxRelativeError(gelu, new Losses.MSE(), x, y) < GC_TOL);

        Sequential sig = new Sequential().add(new Dense(4, 5, r)).add(new Activations.Sigmoid()).add(new Dense(5, 2, r)).add(new Activations.Sigmoid());
        check("gradcheck Sigmoid layers+MSE", GradCheck.maxRelativeError(sig, new Losses.MSE(), x, y) < GC_TOL);

        Sequential bn = new Sequential().add(new Dense(4, 5, r)).add(new BatchNorm(5)).add(new Activations.Tanh()).add(new Dense(5, 2, r));
        check("gradcheck BatchNorm (training mode)", GradCheck.maxRelativeError(bn, new Losses.MSE(), x, y) < GC_TOL);

        Sequential ce = new Sequential().add(new Dense(4, 5, r)).add(new Activations.Tanh()).add(new Dense(5, 2, r));
        check("gradcheck SoftmaxCrossEntropy", GradCheck.maxRelativeError(ce, new Losses.SoftmaxCrossEntropy(), x, oneHot) < GC_TOL);

        Sequential bce = new Sequential().add(new Dense(4, 5, r)).add(new Activations.Tanh()).add(new Dense(5, 1, r));
        check("gradcheck BCEWithLogits", GradCheck.maxRelativeError(bce, new Losses.BCEWithLogits(), x, bin) < GC_TOL);

        Sequential mae = new Sequential().add(new Dense(4, 5, r)).add(new Activations.Tanh()).add(new Dense(5, 2, r));
        check("gradcheck MAE", GradCheck.maxRelativeError(mae, new Losses.MAE(), x, y) < GC_TOL);
    }

    static void lossValues() {
        double mae = new Losses.MAE().forward(Tensor.of(new double[][]{{1, 2}}), Tensor.of(new double[][]{{0, 4}}));
        check("MAE([1,2] vs [0,4]) = 1.5", Math.abs(mae - 1.5) < 1e-12);
        double bce = new Losses.BCEWithLogits().forward(Tensor.of(new double[][]{{0}}), Tensor.of(new double[][]{{1}}));
        check("BCE(score 0, truth 1) = ln 2", Math.abs(bce - Math.log(2)) < 1e-12);
        double big = new Losses.BCEWithLogits().forward(Tensor.of(new double[][]{{1000}}), Tensor.of(new double[][]{{0}}));
        check("BCE stays finite for huge scores", Double.isFinite(big) && Math.abs(big - 1000) < 1e-9);
    }

    // ------------------------------------------------------------------ training

    static Tensor[] line() { // y = 2x + 1
        Tensor x = new Tensor(20, 1), y = new Tensor(20, 1);
        for (int i = 0; i < 20; i++) { double v = -1 + i / 9.5; x.set(i, 0, v); y.set(i, 0, 2 * v + 1); }
        return new Tensor[]{x, y};
    }

    static void learns(String name, Optimizer opt, int epochs) {
        Tensor[] d = line();
        Sequential m = new Sequential().add(new Dense(1, 1, new Random(1)));
        m.fit(d[0], d[1], new Losses.MSE(), opt, epochs, 20, 0, 0);
        double l = new Losses.MSE().forward(m.predict(d[0]), d[1]);
        check(name + " learns y=2x+1 (loss " + String.format("%.2e", l) + ")", l < 1e-4);
    }

    static void optimizers() {
        learns("SGD", new Optimizers.SGD(0.1), 300);
        learns("SGD+momentum", new Optimizers.SGD(0.05, 0.9), 300);
        learns("Adam", new Optimizers.Adam(0.05), 400);
        learns("AdamW", new Optimizers.Adam(0.05, 1e-6), 400);
        learns("RMSProp", new Optimizers.RMSProp(0.01), 800);
    }

    static void weightDecayAndClip() {
        Param w = new Param(Tensor.of(new double[][]{{1.0}}), true);
        Param bias = new Param(Tensor.of(new double[][]{{1.0}}), false);
        new Optimizers.SGD(0.1, 0.0, 0.5).step(List.of(w, bias)); // gradients are 0
        check("SGD weight decay shrinks weights (1.0 -> 0.95), leaves biases alone",
              Math.abs(w.value.data[0] - 0.95) < 1e-6 && bias.value.data[0] == 1.0);

        Param w2 = new Param(Tensor.of(new double[][]{{1.0}}), true);
        new Optimizers.Adam(0.01, 0.1).step(List.of(w2));
        check("AdamW decay applies directly to the weight (1.0 -> 0.999)", Math.abs(w2.value.data[0] - 0.999) < 1e-6);

        Param g = new Param(new Tensor(1, 2));
        g.grad.data[0] = 3; g.grad.data[1] = 4;
        double before = GradClip.clipGlobalNorm(List.of(g), 1.0);
        check("gradient clipping: norm 5 -> 1, direction kept",
              Math.abs(before - 5) < 1e-6 && Math.abs(g.grad.data[0] - 0.6) < 1e-6 && Math.abs(g.grad.data[1] - 0.8) < 1e-6);
    }

    static void schedules() {
        Schedule s = Schedule.stepDecay(0.1, 10, 0.5);
        check("step decay: 0.1, 0.05 after 10 epochs, 0.025 after 20",
              Math.abs(s.rate(0, 100) - 0.1) < 1e-12 && Math.abs(s.rate(10, 100) - 0.05) < 1e-12 && Math.abs(s.rate(25, 100) - 0.025) < 1e-12);
        Schedule c = Schedule.cosine(0.1, 0.001);
        check("cosine: starts at max, ends at min, halfway in between",
              Math.abs(c.rate(0, 101) - 0.1) < 1e-12 && Math.abs(c.rate(100, 101) - 0.001) < 1e-12 && Math.abs(c.rate(50, 101) - 0.0505) < 1e-12);
    }

    static void earlyStopping() {
        Random r = new Random(5);
        Tensor x = new Tensor(30, 1), y = new Tensor(30, 1), vx = new Tensor(30, 1), vy = new Tensor(30, 1);
        for (int i = 0; i < 30; i++) {
            double a = r.nextDouble() * 2 - 1, b = r.nextDouble() * 2 - 1;
            x.set(i, 0, a); y.set(i, 0, a + r.nextGaussian() * 0.4);
            vx.set(i, 0, b); vy.set(i, 0, b + r.nextGaussian() * 0.4);
        }
        Sequential m = new Sequential().add(new Dense(1, 64, r)).add(new Activations.Tanh())
            .add(new Dense(64, 64, r)).add(new Activations.Tanh()).add(new Dense(64, 1, r));
        Trainer.History h = new Trainer().epochs(3000).batchSize(30).validation(vx, vy).patience(40)
            .fit(m, x, y, new Losses.MSE(), new Optimizers.Adam(0.01));
        check("early stopping triggers on an overfitting model (stopped at epoch " + h.epochsRun + ")",
              h.stoppedEarly && h.epochsRun < 3000);
        double finalVal = new Losses.MSE().forward(m.predict(vx), vy);
        check("best weights are restored (final validation loss = best seen)", Math.abs(finalVal - h.bestValLoss) < 1e-9);
    }

    // ------------------------------------------------------------------ helpers

    static void normalizer() throws Exception {
        Tensor x = Tensor.randn(200, 3, 5.0, new Random(5)).map(v -> v + 100);
        Normalizer n = new Normalizer().fit(x);
        Tensor z = n.transform(x);
        double mean = 0, var = 0;
        for (int i = 0; i < z.rows; i++) mean += z.get(i, 0);
        mean /= z.rows;
        for (int i = 0; i < z.rows; i++) var += Math.pow(z.get(i, 0) - mean, 2);
        var /= z.rows;
        check("normalizer gives mean~0 and std~1", Math.abs(mean) < 1e-5 && Math.abs(Math.sqrt(var) - 1) < 1e-5);
        check("normalizer inverse gets original numbers back", maxDiff(x, n.inverse(z)) < 1e-3);

        File f = File.createTempFile("minidl-norm", ".txt");
        n.save(f.getPath());
        Normalizer back = Normalizer.load(f.getPath());
        f.delete();
        check("normalizer save/load gives identical results", maxDiff(n.transform(x), back.transform(x)) == 0.0);
    }

    static void saveLoad() throws Exception {
        Random r = new Random(1);
        Sequential a = new Sequential().add(new Dense(3, 8, r)).add(new BatchNorm(8)).add(new Activations.GELU())
            .add(new Dropout(0.2, r)).add(new Dense(8, 2, r));
        Tensor x = Tensor.randn(16, 3, 1.0, new Random(9)), y = Tensor.randn(16, 2, 1.0, new Random(10));
        a.fit(x, y, new Losses.MSE(), new Optimizers.Adam(0.01), 20, 8, 0, 0); // BatchNorm running stats now non-trivial

        File f = File.createTempFile("minidl", ".txt");
        ModelIO.save(a, f.getPath());

        Sequential b = ModelIO.loadModel(f.getPath(), new Random(99));
        check("loadModel rebuilds the architecture from the file", b.layers().size() == 5 && b.paramCount() == a.paramCount());
        check("save then loadModel gives identical predictions (incl. BatchNorm running stats)",
              maxDiff(a.predict(x), b.predict(x)) == 0.0);

        Sequential wrong = new Sequential().add(new Dense(3, 4, r)).add(new Dense(4, 2, r));
        boolean threw = false;
        try { ModelIO.load(wrong, f.getPath()); } catch (IOException e) { threw = true; }
        check("loading into a different architecture gives a clear error", threw);
        f.delete();
    }

    static void dropout() {
        Tensor x = new Tensor(1000, 10);
        java.util.Arrays.fill(x.data, 1f);
        Dropout d = new Dropout(0.5, new Random(1));
        d.setTraining(false);
        check("dropout does nothing when predicting", d.forward(x) == x);
        d.setTraining(true);
        Tensor out = d.forward(x);
        double mean = 0; int zeros = 0;
        for (double v : out.data) { mean += v; if (v == 0) zeros++; }
        mean /= out.data.length;
        check("dropout (p=0.5) zeros about half, keeps average ~1", Math.abs(mean - 1) < 0.05 && zeros > 4000 && zeros < 6000);
        Tensor g = d.backward(x);
        boolean same = true;
        for (int i = 0; i < g.data.length; i++) if ((g.data[i] == 0) != (out.data[i] == 0)) same = false;
        check("dropout blocks gradient where neuron was off", same);
    }

    static void batchNormBehaviour() {
        BatchNorm bn = new BatchNorm(2);
        Tensor x = Tensor.randn(500, 2, 3.0, new Random(2)).map(v -> v + 10);
        Tensor out = bn.forward(x);
        double m0 = 0, v0 = 0;
        for (int i = 0; i < out.rows; i++) m0 += out.get(i, 0);
        m0 /= out.rows;
        for (int i = 0; i < out.rows; i++) v0 += Math.pow(out.get(i, 0) - m0, 2);
        v0 /= out.rows;
        check("BatchNorm output has mean~0, std~1 while training", Math.abs(m0) < 1e-5 && Math.abs(Math.sqrt(v0) - 1) < 1e-3);
        for (int i = 0; i < 200; i++) bn.forward(x); // let running averages settle
        bn.setTraining(false);
        Tensor ev = bn.forward(x);
        double me = 0;
        for (int i = 0; i < ev.rows; i++) me += ev.get(i, 0);
        check("BatchNorm uses running averages when predicting", Math.abs(me / ev.rows) < 0.05);
    }

    static void csv() throws Exception {
        File f = File.createTempFile("minidl", ".csv");
        try (PrintWriter w = new PrintWriter(f)) { w.println("a,b,label"); w.println("1,2,cat"); w.println("3,4,dog"); w.println(""); w.println("5,6,cat"); }
        Data.Dataset d = Data.loadCsvClassification(f.getPath(), true);
        check("CSV classification: features, sorted class names, one-hot labels",
              d.x.rows == 3 && d.x.cols == 2 && d.x.get(2, 1) == 6 && d.classNames[0].equals("cat") && d.classNames[1].equals("dog")
              && d.y.get(0, 0) == 1 && d.y.get(1, 1) == 1);

        try (PrintWriter w = new PrintWriter(f)) { w.println("1,2,3"); w.println("4,5,6"); }
        Tensor[] xy = Data.loadCsvXY(f.getPath(), false, 1);
        check("CSV numeric: last column becomes y", xy[0].cols == 2 && xy[1].get(1, 0) == 6 && xy[0].get(1, 0) == 4);

        try (PrintWriter w = new PrintWriter(f)) { w.println("1,2,oops"); }
        boolean threw = false;
        try { Data.loadCsvXY(f.getPath(), false, 1); } catch (IOException e) { threw = e.getMessage().contains("oops"); }
        check("CSV bad value gives an error with line and value", threw);
        f.delete();

        Tensor x = new Tensor(100, 1), y = new Tensor(100, 1);
        for (int i = 0; i < 100; i++) { x.set(i, 0, i); y.set(i, 0, i); }
        Tensor[] s = Data.trainTestSplit(x, y, 0.2, 3);
        double sum = 0;
        for (double v : s[0].data) sum += v;
        for (double v : s[2].data) sum += v;
        check("train/test split: 80/20, no row lost or duplicated", s[0].rows == 80 && s[2].rows == 20 && sum == 4950);
    }

    static void mnistFormat() throws Exception {
        File img = File.createTempFile("minidl-img", ".gz"), lab = File.createTempFile("minidl-lab", ".idx");
        try (DataOutputStream o = new DataOutputStream(new GZIPOutputStream(new FileOutputStream(img)))) {
            o.writeInt(2051); o.writeInt(2); o.writeInt(2); o.writeInt(2);
            o.write(new byte[]{0, (byte) 255, 51, 102, (byte) 255, 0, 0, 0});
        }
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(lab))) {
            o.writeInt(2049); o.writeInt(2); o.write(7); o.write(3);
        }
        Tensor x = MnistLoader.loadImages(img.getPath());
        Tensor y = MnistLoader.loadLabels(lab.getPath());
        check("MNIST loader: gzip images scaled to 0..1, one-hot labels",
              x.rows == 2 && x.cols == 4 && x.get(0, 1) == 1.0 && Math.abs(x.get(0, 2) - 0.2) < 1e-6 && y.cols == 10 && y.get(0, 7) == 1 && y.get(1, 3) == 1);
        img.delete(); lab.delete();
    }

    static void metrics() {
        Tensor p = Tensor.of(new double[][]{{1}, {2}, {3}}), t = Tensor.of(new double[][]{{1}, {2}, {3}});
        check("metrics: perfect prediction gives MAE 0, RMSE 0, R2 1",
              Metrics.mae(p, t) == 0 && Metrics.rmse(p, t) == 0 && Metrics.r2(p, t) == 1.0);
        Tensor scores = Tensor.of(new double[][]{{0.9, 0.1}, {0.2, 0.8}, {0.7, 0.3}});
        Tensor truth = Tensor.of(new double[][]{{1, 0}, {0, 1}, {0, 1}});
        int[][] cm = Metrics.confusionMatrix(scores, truth);
        check("confusion matrix counts [truth][predicted]", cm[0][0] == 1 && cm[1][1] == 1 && cm[1][0] == 1 && cm[0][1] == 0);
    }

    // ------------------------------------------------------------------ new: convolution, pooling, memory, int8

    /** Plain 6-loop convolution to compare against (no tricks). */
    static Tensor naiveConv(Tensor x, Tensor W, Tensor b, int inC, int H, int Wd, int outC, int k, int stride, int pad) {
        int oh = (H + 2 * pad - k) / stride + 1, ow = (Wd + 2 * pad - k) / stride + 1;
        Tensor out = new Tensor(x.rows, outC * oh * ow);
        for (int n = 0; n < x.rows; n++)
            for (int oc = 0; oc < outC; oc++)
                for (int oy = 0; oy < oh; oy++)
                    for (int ox = 0; ox < ow; ox++) {
                        double s = b.data[oc];
                        for (int ci = 0; ci < inC; ci++)
                            for (int ky = 0; ky < k; ky++)
                                for (int kx = 0; kx < k; kx++) {
                                    int iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
                                    if (iy < 0 || iy >= H || ix < 0 || ix >= Wd) continue;
                                    s += x.data[n * inC * H * Wd + (ci * H + iy) * Wd + ix] * W.data[oc * inC * k * k + (ci * k + ky) * k + kx];
                                }
                        out.data[n * outC * oh * ow + (oc * oh + oy) * ow + ox] = (float) s;
                    }
        return out;
    }

    static void convolution() {
        Random r = new Random(6);
        int[][] cfg = {{2, 7, 6, 3, 3, 1, 1}, {1, 8, 8, 2, 3, 2, 0}, {3, 6, 6, 4, 2, 1, 0}, {1, 5, 5, 2, 5, 1, 2}};
        boolean allMatch = true;
        for (int[] c : cfg) {
            Conv2D conv = new Conv2D(c[0], c[1], c[2], c[3], c[4], c[5], c[6], r);
            Tensor x = Tensor.randn(3, c[0] * c[1] * c[2], 1, r);
            Tensor want = naiveConv(x, conv.params().get(0).value, conv.params().get(1).value, c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
            Tensor got = conv.forward(x);
            if (got.rows != want.rows || got.cols != want.cols || maxDiff(got, want) > 1e-4) allMatch = false;
        }
        check("Conv2D matches a plain 6-loop convolution (4 shapes: padding, stride, 2x2 and 5x5 filters)", allMatch);

        Tensor x = Tensor.randn(4, 2 * 5 * 5, 1, r), y = Tensor.randn(4, 3, 1, r);
        Sequential c1 = new Sequential().add(new Conv2D(2, 5, 5, 3, 3, 1, 1, r)).add(new Activations.Tanh()).add(new Dense(3 * 5 * 5, 3, r));
        check("gradcheck Conv2D (stride 1, padding 1) + Tanh + Dense", GradCheck.maxRelativeError(c1, new Losses.MSE(), x, y) < GC_TOL);

        Tensor x2 = Tensor.randn(4, 1 * 6 * 6, 1, r);
        Sequential c2 = new Sequential().add(new Conv2D(1, 6, 6, 2, 3, 2, 0, r)).add(new Activations.Tanh()).add(new Dense(2 * 2 * 2, 3, r));
        check("gradcheck Conv2D (stride 2, no padding)", GradCheck.maxRelativeError(c2, new Losses.MSE(), x2, y) < GC_TOL);

        Tensor x3 = Tensor.randn(4, 1 * 8 * 8, 1, r);
        Sequential c3 = new Sequential().add(new Conv2D(1, 8, 8, 2, 3, 1, 0, r)).add(new Activations.Tanh()).add(new Conv2D(2, 6, 6, 2, 3, 1, 0, r))
            .add(new Activations.Tanh()).add(new Dense(2 * 4 * 4, 3, r));
        check("gradcheck two stacked Conv2D layers (input gradient flows through the second)", GradCheck.maxRelativeError(c3, new Losses.MSE(), x3, y) < GC_TOL);

        boolean old = Tensor.isParallel();
        Conv2D cv = new Conv2D(2, 7, 7, 4, 3, 1, 1, new Random(1));
        Tensor xb = Tensor.randn(9, 2 * 7 * 7, 1, new Random(2)), gb = Tensor.randn(9, 4 * 7 * 7, 1, new Random(3));
        Tensor.setParallel(false);
        Tensor o1 = cv.forward(xb); Tensor d1 = cv.backward(gb); Tensor gw1 = cv.params().get(0).grad.copy();
        Tensor.setParallel(true);
        Tensor o2 = cv.forward(xb); Tensor d2 = cv.backward(gb); Tensor gw2 = cv.params().get(0).grad.copy();
        Tensor.setParallel(old);
        check("Conv2D: multi-thread results match single-thread", maxDiff(o1, o2) < 1e-5 && maxDiff(d1, d2) < 1e-4 && maxDiff(gw1, gw2) < 1e-3);
    }

    static void pooling() {
        MaxPool2D p = new MaxPool2D(1, 4, 4, 2);
        Tensor x = Tensor.of(new double[][]{{1, 2, 5, 3,  4, 9, 1, 1,  0, 0, 7, 2,  8, 1, 3, 6}});
        Tensor y = p.forward(x);
        check("MaxPool2D keeps the biggest of each 2x2 square: [9,5,8,7]",
              y.cols == 4 && y.get(0, 0) == 9 && y.get(0, 1) == 5 && y.get(0, 2) == 8 && y.get(0, 3) == 7);
        Tensor g = p.backward(Tensor.of(new double[][]{{10, 20, 30, 40}}));
        check("MaxPool2D sends the blame only to the winner (and stores just 1 byte per window)",
              g.get(0, 5) == 10 && g.get(0, 2) == 20 && g.get(0, 12) == 30 && g.get(0, 10) == 40 && g.get(0, 0) == 0);

        Random r = new Random(2); // max-pool is not smooth: a near-tie inside the nudge size can fool the checker (rare, seed 11 does it)
        Tensor xin = Tensor.randn(3, 1 * 6 * 6, 1, r), t = Tensor.randn(3, 2, 1, r);
        Sequential m = new Sequential().add(new Conv2D(1, 6, 6, 2, 3, 1, 0, r)).add(new Activations.Tanh()).add(new MaxPool2D(2, 4, 4, 2)).add(new Dense(2 * 2 * 2, 2, r));
        check("gradcheck Conv2D + Tanh + MaxPool2D + Dense", GradCheck.maxRelativeError(m, new Losses.MSE(), xin, t) < GC_TOL);
    }

    static void byteData() {
        byte[] px = new byte[]{0, (byte) 255, 51, 102, (byte) 255, 0, 0, 0};
        byte[] lab = new byte[]{1, 0};
        ByteSource s = new ByteSource(px, 4, lab, 2);
        int[] idx = {0, 1};
        Tensor f = s.features(idx, 0, 2), t = s.targets(idx, 0, 2);
        check("ByteSource: bytes become floats 0..1, labels become one-hot",
              f.get(0, 1) == 1.0 && Math.abs(f.get(0, 2) - 0.2) < 1e-6 && t.get(0, 1) == 1 && t.get(1, 0) == 1);
        ByteSource v = s.view(1, 2);
        check("ByteSource.view shares the same bytes (no copy) and has the right size", v.size() == 1 && v.bytes() == s.bytes() && v.features(new int[]{0}, 0, 1).get(0, 0) == 1.0);
        check("memory: 1 byte per value (8 values + 2 labels = 10 bytes), a float Tensor of the same data is 32 bytes",
              s.bytes() == 10 && new Tensor(2, 4).bytes() == 32);

        // train from a ByteSource: 2 classes, 4 "pixels"
        Random r = new Random(2);
        int n = 300;
        byte[] p2 = new byte[n * 4], l2 = new byte[n];
        for (int i = 0; i < n; i++) {
            int c = i % 2; l2[i] = (byte) c;
            for (int j = 0; j < 4; j++) {
                boolean on = (j % 2 == c);
                p2[i * 4 + j] = (byte) (on ? 180 + r.nextInt(70) : r.nextInt(70));
            }
        }
        ByteSource data = new ByteSource(p2, 4, l2, 2);
        Sequential m = new Sequential().add(new Dense(4, 8, r)).add(new Activations.ReLU()).add(new Dense(8, 2, r));
        new Trainer().epochs(30).batchSize(30).seed(1).validation(data.view(240, 300))
            .fit(m, data.view(0, 240), new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.05));
        double acc = m.accuracy(data.view(240, 300));
        check("Trainer learns from a ByteSource (accuracy " + acc + "% on held-out rows)", acc >= 99.0);
    }

    static void quantization() {
        Random r = new Random(7);
        Sequential f = new Sequential().add(new Dense(200, 128, r)).add(new Activations.ReLU()).add(new Dense(128, 10, r));
        Tensor x = Tensor.randn(50, 200, 1, r);
        Sequential q = Quantizer.quantize(f);
        Tensor a = f.predict(x), b = q.predict(x);
        double maxAbs = 0;
        for (float v : a.data) maxAbs = Math.max(maxAbs, Math.abs(v));
        check("int8 predictions stay close to float (max difference " + String.format("%.4f", maxDiff(a, b)) + " on outputs up to " + String.format("%.2f", maxAbs) + ")",
              maxDiff(a, b) < 0.03 * maxAbs);
        double ratio = (double) f.weightBytes() / q.weightBytes();
        check("int8 model uses about 4x less memory (" + String.format("%.2f", ratio) + "x)", ratio > 3.7);
        boolean threw = false;
        try { q.backward(new Tensor(50, 10)); } catch (UnsupportedOperationException e) { threw = true; }
        check("quantized layers refuse to train, with a clear error", threw);
    }

    static void binaryFiles() throws Exception {
        Random r = new Random(5);
        Sequential cnn = new Sequential()
            .add(new Conv2D(1, 8, 8, 3, 3, 1, 0, r)).add(new Activations.ReLU()).add(new MaxPool2D(3, 6, 6, 2))
            .add(new Dense(3 * 3 * 3, 8, r)).add(new BatchNorm(8)).add(new Activations.ReLU()).add(new Dense(8, 4, r));
        Tensor x = Tensor.randn(20, 64, 1, r), y = Tensor.randn(20, 4, 1, r);
        cnn.fit(x, y, new Losses.MSE(), new Optimizers.Adam(0.01), 5, 10, 0, 0);

        File bin = File.createTempFile("minidl", ".bin"), txt = File.createTempFile("minidl", ".txt");
        ModelIO.saveBinary(cnn, bin.getPath());
        ModelIO.save(cnn, txt.getPath());
        Sequential fromBin = ModelIO.loadModelBinary(bin.getPath(), new Random(1));
        Sequential fromTxt = ModelIO.loadModel(txt.getPath(), new Random(1));
        check("CNN (Conv2D, MaxPool2D, BatchNorm) round-trips through the binary file with identical predictions", maxDiff(cnn.predict(x), fromBin.predict(x)) == 0.0);
        check("CNN round-trips through the text file with identical predictions", maxDiff(cnn.predict(x), fromTxt.predict(x)) == 0.0);
        check("binary file is smaller than the text file (" + bin.length() + " vs " + txt.length() + " bytes)", bin.length() < txt.length() / 2);

        Sequential q = Quantizer.quantize(cnn);
        File qf = File.createTempFile("minidl-q", ".bin");
        ModelIO.saveBinary(q, qf.getPath());
        Sequential qLoaded = ModelIO.loadModelBinary(qf.getPath(), new Random(1));
        check("int8 model round-trips through the binary file with identical predictions", maxDiff(q.predict(x), qLoaded.predict(x)) == 0.0);
        bin.delete(); txt.delete(); qf.delete();
    }

    // ------------------------------------------------------------------ round 3: LayerNorm, Residual, pooling

    static void layerNormAndResidual() throws Exception {
        Random r = new Random(21);
        LayerNorm ln = new LayerNorm(6);
        Tensor x = Tensor.randn(5, 6, 3.0, r).map(v -> v + 10);
        Tensor y = ln.forward(x);
        boolean rowsOk = true;
        for (int i = 0; i < y.rows; i++) {
            double m = 0, v = 0;
            for (int j = 0; j < 6; j++) m += y.get(i, j);
            m /= 6;
            for (int j = 0; j < 6; j++) v += Math.pow(y.get(i, j) - m, 2);
            v /= 6;
            if (Math.abs(m) > 1e-4 || Math.abs(Math.sqrt(v) - 1) > 1e-3) rowsOk = false;
        }
        check("LayerNorm: every ROW has mean 0 and std 1", rowsOk);
        Tensor one = ln.forward(x.selectRows(new int[]{2}, 0, 1));
        boolean same = true;
        for (int j = 0; j < 6; j++) if (Math.abs(one.get(0, j) - y.get(2, j)) > 1e-6) same = false;
        check("LayerNorm: a row gives the same result alone or inside a batch", same);

        Tensor xin = Tensor.randn(4, 5, 1, r), t = Tensor.randn(4, 3, 1, r);
        Sequential m1 = new Sequential().add(new Dense(5, 6, r)).add(new LayerNorm(6)).add(new Activations.Tanh()).add(new Dense(6, 3, r));
        check("gradcheck LayerNorm", GradCheck.maxRelativeError(m1, new Losses.MSE(), xin, t) < GC_TOL);

        Dense inner = new Dense(5, 5, r);
        Residual res = new Residual(inner);
        Tensor out = res.forward(xin);
        Tensor expect = xin.add(inner.forward(xin));
        check("Residual output = input + f(input)", maxDiff(out, expect) < 1e-6);
        Sequential m2 = new Sequential().add(new Dense(5, 5, r)).add(new Residual(new Dense(5, 5, r), new Activations.Tanh(), new Dense(5, 5, r))).add(new Dense(5, 3, r));
        check("gradcheck Residual block (gradient flows through the shortcut and through f)", GradCheck.maxRelativeError(m2, new Losses.MSE(), xin, t) < GC_TOL_DEEP);
        boolean threw = false;
        try { new Residual(new Dense(5, 4, r)).forward(xin); } catch (IllegalArgumentException e) { threw = true; }
        check("Residual refuses a block that changes the shape", threw);
    }

    static void avgPoolAndReshape() {
        AvgPool2D p = new AvgPool2D(1, 4, 4, 2);
        Tensor x = Tensor.of(new double[][]{{1, 2, 5, 3,  4, 9, 1, 1,  0, 0, 7, 2,  8, 1, 3, 6}});
        Tensor y = p.forward(x);
        check("AvgPool2D averages each 2x2 square: [4, 2.5, 2.25, 4.5]",
              Math.abs(y.get(0, 0) - 4) < 1e-6 && Math.abs(y.get(0, 1) - 2.5) < 1e-6 && Math.abs(y.get(0, 2) - 2.25) < 1e-6 && Math.abs(y.get(0, 3) - 4.5) < 1e-6);
        Random r = new Random(4);
        Tensor xin = Tensor.randn(3, 36, 1, r), t = Tensor.randn(3, 2, 1, r);
        Sequential m = new Sequential().add(new Conv2D(1, 6, 6, 2, 3, 1, 0, r)).add(new Activations.Tanh()).add(new AvgPool2D(2, 4, 4, 2)).add(new Dense(8, 2, r));
        check("gradcheck Conv2D + AvgPool2D + Dense", GradCheck.maxRelativeError(m, new Losses.MSE(), xin, t) < GC_TOL);
        Reshape rs = new Reshape(4);
        Tensor z = rs.forward(new Tensor(2, 12));
        check("Reshape re-cuts rows without copying: (2x12) -> (6x4) and back", z.rows == 6 && z.cols == 4 && rs.backward(z).cols == 12);
    }

    // ------------------------------------------------------------------ round 3: transformer parts

    static Sequential miniTransformer(int V, int T, int D, int H, Random r) {
        return new Sequential().add(new Embedding(V, D, T, r))
            .add(new Residual(new TokenWise(T, D, new LayerNorm(D)), new CausalSelfAttention(T, D, H, r)))
            .add(new Residual(new TokenWise(T, D, new LayerNorm(D), new Dense(D, 2 * D, r), new Activations.GELU(), new Dense(2 * D, D, r))))
            .add(new TokenWise(T, D, new LayerNorm(D), new Dense(D, V, r)))
            .add(new Reshape(V));
    }

    static Tensor naiveAttention(Tensor x, Tensor Wqkv, Tensor bqkv, Tensor Wo, Tensor bo, int T, int D, int H) {
        int dh = D / H;
        Tensor out = new Tensor(x.rows, T * D);
        for (int n = 0; n < x.rows; n++) {
            double[][] qkv = new double[T][3 * D];
            for (int t = 0; t < T; t++) for (int c = 0; c < 3 * D; c++) {
                double s = bqkv.data[c];
                for (int d = 0; d < D; d++) s += x.data[n * T * D + t * D + d] * Wqkv.data[d * 3 * D + c];
                qkv[t][c] = s;
            }
            double[][] o = new double[T][D];
            for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
                double[] sc = new double[i + 1];
                double mx = Double.NEGATIVE_INFINITY;
                for (int j = 0; j <= i; j++) {
                    double s = 0;
                    for (int d = 0; d < dh; d++) s += qkv[i][h * dh + d] * qkv[j][D + h * dh + d];
                    sc[j] = s / Math.sqrt(dh);
                    mx = Math.max(mx, sc[j]);
                }
                double sum = 0;
                for (int j = 0; j <= i; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
                for (int j = 0; j <= i; j++) for (int d = 0; d < dh; d++) o[i][h * dh + d] += sc[j] / sum * qkv[j][2 * D + h * dh + d];
            }
            for (int t = 0; t < T; t++) for (int c = 0; c < D; c++) {
                double s = bo.data[c];
                for (int d = 0; d < D; d++) s += o[t][d] * Wo.data[d * D + c];
                out.data[n * T * D + t * D + c] = (float) s;
            }
        }
        return out;
    }

    static void transformerParts() {
        Random r = new Random(31);
        int V = 7, T = 5, D = 8, H = 2, N = 3;

        Embedding emb = new Embedding(V, D, T, r);
        Tensor ids = new Tensor(2, T);
        for (int i = 0; i < ids.data.length; i++) ids.data[i] = r.nextInt(V);
        Tensor e = emb.forward(ids);
        Tensor tok = emb.params().get(0).value, pos = emb.params().get(1).value;
        boolean embOk = true;
        for (int n = 0; n < 2; n++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
            if (Math.abs(e.data[(n * T + t) * D + d] - (tok.data[(int) ids.data[n * T + t] * D + d] + pos.data[t * D + d])) > 1e-6) embOk = false;
        check("Embedding output = token vector + position vector", embOk);
        boolean threw = false;
        try { emb.forward(Tensor.of(new double[][]{{0, 1, 2, 3, 99}})); } catch (IllegalArgumentException ex) { threw = true; }
        check("Embedding rejects an id outside the vocabulary", threw);

        CausalSelfAttention att = new CausalSelfAttention(T, D, H, r);
        for (Param p : att.params()) for (int i = 0; i < p.value.data.length; i++) if (p.value.data[i] == 0f) p.value.data[i] = (float) (r.nextGaussian() * 0.3);
        Tensor x = Tensor.randn(N, T * D, 1, r);
        Tensor got = att.forward(x);
        Tensor want = naiveAttention(x, att.params().get(0).value, att.params().get(1).value, att.params().get(2).value, att.params().get(3).value, T, D, H);
        check("CausalSelfAttention matches a plain double-precision reference (max difference " + String.format("%.1e", maxDiff(got, want)) + ")", maxDiff(got, want) < 1e-4);

        Sequential tf = miniTransformer(V, T, D, H, r);
        Tensor tin = new Tensor(N, T), targets = new Tensor(N * T, V);
        for (int i = 0; i < tin.data.length; i++) { tin.data[i] = r.nextInt(V); targets.data[i * V + r.nextInt(V)] = 1f; }
        check("gradcheck the whole mini transformer (embedding, attention, LayerNorm, MLP, residuals, output)",
              GradCheck.maxRelativeError(tf, new Losses.SoftmaxCrossEntropy(), tin, targets) < GC_TOL);

        Tensor a = tin.copy(), b = tin.copy();
        b.data[T - 1] = (b.data[T - 1] + 1) % V; // change ONLY the last character of sequence 0
        Tensor ya = tf.predict(a), yb = tf.predict(b);
        double early = 0, late = 0;
        for (int t = 0; t < T - 1; t++) for (int v = 0; v < V; v++) early = Math.max(early, Math.abs(ya.data[t * V + v] - yb.data[t * V + v]));
        for (int v = 0; v < V; v++) late = Math.max(late, Math.abs(ya.data[(T - 1) * V + v] - yb.data[(T - 1) * V + v]));
        check("causal: changing the LAST character cannot change earlier predictions (" + String.format("%.1e", early) + ") but does change the last (" + String.format("%.3f", late) + ")", early < 1e-6 && late > 1e-4);

        // learns a simple rule from text: in "abcdef abcdef ..." the next character always follows the alphabet
        Random r2 = new Random(5);
        CharSource rule = CharSource.fromText("abcdef".repeat(60), 6);
        Sequential m = miniTransformer(6, 6, 16, 2, r2);
        new Trainer().epochs(40).batchSize(32).maxBatchesPerEpoch(8).seed(1).fit(m, rule, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.01));
        double loss = m.evalLoss(rule.view(0, 120), new Losses.SoftmaxCrossEntropy());
        check("transformer learns a text rule (loss " + String.format("%.3f", loss) + ", random guessing is 1.792)", loss < 0.2);

        Sequential q = Quantizer.quantize(tf);
        boolean hasQ = q.layers().toString().length() > 0;
        StringBuilder specs = new StringBuilder();
        for (Layer l : q.layers()) specs.append(l.spec()).append(' ');
        check("Quantizer also converts the Dense layers INSIDE Residual / TokenWise blocks", specs.toString().contains("QDense") && hasQ);
        Tensor fp = tf.predict(tin), qp = q.predict(tin);
        double mx = 0;
        for (float v : fp.data) mx = Math.max(mx, Math.abs(v));
        check("quantized transformer predictions stay close (max difference " + String.format("%.4f", maxDiff(fp, qp)) + " on outputs up to " + String.format("%.2f", mx) + ")", maxDiff(fp, qp) < 0.05 * Math.max(mx, 1));

        // nested specs survive a save and load, text and binary
        try {
            File txt = File.createTempFile("minidl-tf", ".txt"), bin = File.createTempFile("minidl-tf", ".bin");
            ModelIO.save(tf, txt.getPath());
            ModelIO.saveBinary(tf, bin.getPath());
            Sequential fromTxt = ModelIO.loadModel(txt.getPath(), new Random(9)), fromBin = ModelIO.loadModelBinary(bin.getPath(), new Random(9));
            check("transformer (nested Residual / TokenWise) round-trips through text and binary files with identical predictions",
                  maxDiff(tf.predict(tin), fromTxt.predict(tin)) == 0.0 && maxDiff(tf.predict(tin), fromBin.predict(tin)) == 0.0);
            txt.delete(); bin.delete();
        } catch (Exception ex) { check("transformer save/load: " + ex, false); }

        // Trainer step limit
        final int[] calls = {0};
        DataSource counting = new DataSource() {
            public int size() { return 100; }
            public Tensor features(int[] idx, int from, int to) { calls[0]++; return new Tensor(to - from, 2); }
            public Tensor targets(int[] idx, int from, int to) { return new Tensor(to - from, 1); }
        };
        new Trainer().epochs(2).batchSize(10).maxBatchesPerEpoch(3).fit(new Sequential().add(new Dense(2, 1, r)), counting, new Losses.MSE(), new Optimizers.SGD(0.01));
        check("Trainer.maxBatchesPerEpoch limits batches (2 epochs x 3 = 6 batches, got " + calls[0] + ")", calls[0] == 6);
    }

    // ------------------------------------------------------------------ round 3: autograd

    static void autograd() {
        Random r = new Random(41);
        Tensor x = Tensor.randn(6, 3, 1, r), y = Tensor.randn(6, 2, 1, r);

        Sequential layerModel = new Sequential().add(new Dense(3, 2, r));
        Tensor W = layerModel.params().get(0).value.copy(), B = layerModel.params().get(1).value.copy();
        Losses.MSE mse = new Losses.MSE();
        layerModel.trainStep(x, y, mse, new Optimizers.SGD(0.0)); // lr 0: computes gradients, changes nothing
        Var w = new Var(W), b = new Var(B);
        Var loss = new Var(x).matmul(w).addRow(b).mseLoss(y);
        loss.backward();
        check("autograd Dense+MSE gives the same gradients as the hand-written layer backward",
              maxDiff(w.grad, layerModel.params().get(0).grad) < 1e-6 && maxDiff(b.grad, layerModel.params().get(1).grad) < 1e-6);

        Tensor oneHot = new Tensor(6, 3);
        for (int i = 0; i < 6; i++) oneHot.set(i, i % 3, 1);
        Sequential net = new Sequential().add(new Dense(3, 5, r)).add(new Activations.Tanh()).add(new Dense(5, 3, r));
        Tensor W1 = net.params().get(0).value.copy(), B1 = net.params().get(1).value.copy(), W2 = net.params().get(2).value.copy(), B2 = net.params().get(3).value.copy();
        net.trainStep(x, oneHot, new Losses.SoftmaxCrossEntropy(), new Optimizers.SGD(0.0));
        Var w1 = new Var(W1), b1 = new Var(B1), w2 = new Var(W2), b2 = new Var(B2);
        Var l2 = new Var(x).matmul(w1).addRow(b1).tanh().matmul(w2).addRow(b2).softmaxCrossEntropy(oneHot);
        l2.backward();
        check("autograd 2-layer net (tanh + softmax cross entropy) matches the layer gradients for all 4 weight sets",
              maxDiff(w1.grad, net.params().get(0).grad) < 1e-5 && maxDiff(b1.grad, net.params().get(1).grad) < 1e-5
              && maxDiff(w2.grad, net.params().get(2).grad) < 1e-5 && maxDiff(b2.grad, net.params().get(3).grad) < 1e-5);

        // a Var used twice (x*x): gradient must add up from both uses
        Var a = new Var(Tensor.of(new double[][]{{3}}));
        Var sq = a.mul(a);
        sq.backward();
        check("autograd: a value used twice gets both gradients added (d(x*x)/dx = 2x = 6)", Math.abs(a.grad.data[0] - 6) < 1e-6);

        // trains XOR with nothing but forward maths
        Random r3 = new Random(3);
        Tensor xs = Tensor.of(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}}), ys = Tensor.of(new double[][]{{0}, {1}, {1}, {0}});
        Var p1 = new Var(Tensor.randn(2, 8, 1, r3)), q1 = new Var(new Tensor(1, 8)), p2 = new Var(Tensor.randn(8, 1, 1, r3)), q2 = new Var(new Tensor(1, 1));
        Var[] ps = {p1, q1, p2, q2};
        double last = 1;
        for (int step = 0; step < 1500; step++) {
            for (Var v : ps) v.zeroGrad();
            Var l = new Var(xs).matmul(p1).addRow(q1).tanh().matmul(p2).addRow(q2).sigmoid().mseLoss(ys);
            l.backward();
            last = l.value.data[0];
            for (Var v : ps) for (int i = 0; i < v.value.data.length; i++) v.value.data[i] -= 0.5f * v.grad.data[i];
        }
        check("autograd trains XOR using only forward maths (final loss " + String.format("%.4f", last) + ")", last < 0.02);

        boolean threw = false;
        try { new Var(new Tensor(2, 2)).backward(); } catch (IllegalStateException e) { threw = true; }
        check("backward() on a non-scalar gives a clear error", threw);
    }

    // ------------------------------------------------------------------ round 3: data augmentation

    static void augmentation() {
        int h = 5, w = 5;
        float[] img = new float[h * w];
        img[2 * w + 2] = 1f;
        float[] out = new float[h * w];
        AugmentedSource.transform(img, 0, out, 0, h, w, 0, 0, 0);
        check("augmentation: no rotation and no shift leaves the image unchanged", maxDiff(new Tensor(1, 25, img), new Tensor(1, 25, out)) < 1e-6);
        AugmentedSource.transform(img, 0, out, 0, h, w, 0, 1, 0);
        check("augmentation: shift of 1 pixel to the right moves the dot from (2,2) to (2,3)", Math.abs(out[2 * w + 3] - 1) < 1e-6 && Math.abs(out[2 * w + 2]) < 1e-6);

        float[] bar = new float[h * w];
        for (int y = 0; y < h; y++) bar[y * w + 2] = 1f;
        AugmentedSource.transform(bar, 0, out, 0, h, w, Math.PI / 2, 0, 0);
        boolean horizontal = true;
        for (int x = 0; x < w; x++) if (Math.abs(out[2 * w + x] - 1) > 1e-5) horizontal = false;
        for (int y = 0; y < h; y++) if (y != 2) for (int x = 0; x < w; x++) if (Math.abs(out[y * w + x]) > 1e-5) horizontal = false;
        check("augmentation: rotating a vertical bar by 90 degrees gives a horizontal bar", horizontal);

        byte[] px = new byte[3 * 784], lab = {0, 1, 2};
        for (int i = 0; i < px.length; i++) px[i] = (byte) ((i * 7) % 256);
        ByteSource base = new ByteSource(px, 784, lab, 3);
        int[] idx = {0, 1, 2};
        Tensor plain = base.features(idx, 0, 3);
        Tensor same = new AugmentedSource(base, 28, 28, 0, 0, 1).features(idx, 0, 3);
        Tensor moved = new AugmentedSource(base, 28, 28, 2, 10, 1).features(idx, 0, 3);
        Tensor again = new AugmentedSource(base, 28, 28, 2, 10, 1).features(idx, 0, 3);
        check("AugmentedSource: zero shift/rotation = original; real augmentation changes pixels; same seed = same result; targets untouched",
              maxDiff(plain, same) < 1e-5 && maxDiff(plain, moved) > 0.01 && maxDiff(moved, again) == 0.0
              && maxDiff(base.targets(idx, 0, 3), new AugmentedSource(base, 28, 28, 2, 10, 1).targets(idx, 0, 3)) == 0.0);
    }

    // ------------------------------------------------------------------ round 3: int4, bfloat16, int8 convolution

    static void moreQuantization() throws Exception {
        Random r = new Random(51);
        Sequential f = new Sequential().add(new Dense(512, 512, r)).add(new Activations.ReLU()).add(new Dense(512, 10, r));
        Tensor x = Tensor.randn(40, 512, 1, r);
        Tensor a = f.predict(x);
        double rmsOut = 0;
        for (float v : a.data) rmsOut += v * v;
        rmsOut = Math.sqrt(rmsOut / a.data.length);

        Sequential q4 = Quantizer.quantizeInt4(f, 32);
        Tensor b4 = q4.predict(x);
        double rms4 = 0;
        for (int i = 0; i < a.data.length; i++) rms4 += Math.pow(a.data[i] - b4.data[i], 2);
        rms4 = Math.sqrt(rms4 / a.data.length) / rmsOut;
        double ratio4 = (double) f.weightBytes() / q4.weightBytes();
        check("int4 (group 32) output error is " + String.format("%.1f%%", 100 * rms4) + " of the signal, and the model is " + String.format("%.1fx", ratio4) + " smaller", rms4 < 0.12 && ratio4 > 5.0);

        Sequential qb = Quantizer.toBF16(f);
        Tensor bb = qb.predict(x);
        double ratioB = (double) f.weightBytes() / qb.weightBytes();
        check("bfloat16 weights: output within 1% (" + String.format("%.4f", maxDiff(a, bb)) + ") and " + String.format("%.2fx", ratioB) + " smaller",
              maxDiff(a, bb) < 0.01 * rmsOut * 5 && ratioB > 1.95);
        check("bfloat16 conversion: 1.0 stays 1.0, pi is within 0.4%",
              BF16Dense.fromBF16(BF16Dense.toBF16(1.0f)) == 1.0f && Math.abs(BF16Dense.fromBF16(BF16Dense.toBF16(3.14159265f)) - 3.14159265) / 3.14159265 < 0.004);

        // odd output width + group bigger than the layer
        Sequential odd = new Sequential().add(new Dense(10, 3, r));
        Tensor xo = Tensor.randn(4, 10, 1, r);
        Sequential odd4 = Quantizer.quantizeInt4(odd, 64);
        check("int4 handles an odd number of outputs and a short last group", odd4.predict(xo).cols == 3 && maxDiff(odd.predict(xo), odd4.predict(xo)) < 0.5);

        Sequential cnn = new Sequential().add(new Conv2D(2, 8, 8, 4, 3, 1, 1, r)).add(new Activations.ReLU()).add(new Dense(4 * 64, 5, r));
        Tensor xc = Tensor.randn(6, 2 * 64, 1, r);
        Sequential qc = Quantizer.quantize(cnn);
        double mc = 0;
        for (float v : cnn.predict(xc).data) mc = Math.max(mc, Math.abs(v));
        check("int8 convolution (QConv2D) output stays close to float (" + String.format("%.4f", maxDiff(cnn.predict(xc), qc.predict(xc))) + " on outputs up to " + String.format("%.2f", mc) + ")",
              qc.layers().get(0).spec().startsWith("QConv2D") && maxDiff(cnn.predict(xc), qc.predict(xc)) < 0.03 * mc);

        File f4 = File.createTempFile("minidl-q4", ".bin"), fb = File.createTempFile("minidl-qb", ".bin"), fc = File.createTempFile("minidl-qc", ".bin");
        ModelIO.saveBinary(q4, f4.getPath());
        ModelIO.saveBinary(qb, fb.getPath());
        ModelIO.saveBinary(qc, fc.getPath());
        check("int4, bfloat16 and int8-conv models round-trip through binary files with identical predictions",
              maxDiff(b4, ModelIO.loadModelBinary(f4.getPath(), new Random(1)).predict(x)) == 0.0
              && maxDiff(bb, ModelIO.loadModelBinary(fb.getPath(), new Random(1)).predict(x)) == 0.0
              && maxDiff(qc.predict(xc), ModelIO.loadModelBinary(fc.getPath(), new Random(1)).predict(xc)) == 0.0);
        f4.delete(); fb.delete(); fc.delete();
    }

    // ------------------------------------------------------------------ round 3: memory-mapped data, text data

    static void mappedAndChar() throws Exception {
        File img = File.createTempFile("minidl-map-img", ".idx"), lab = File.createTempFile("minidl-map-lab", ".idx");
        int n = 5, dim = 4;
        byte[] px = new byte[n * dim], lb = new byte[n];
        for (int i = 0; i < px.length; i++) px[i] = (byte) (i * 11 + 3);
        for (int i = 0; i < n; i++) lb[i] = (byte) (i % 3);
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(img))) { o.writeInt(2051); o.writeInt(n); o.writeInt(2); o.writeInt(2); o.write(px); }
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(lab))) { o.writeInt(2049); o.writeInt(n); o.write(lb); }
        MappedSource ms = MappedSource.open(img.getPath(), lab.getPath());
        ByteSource bs = new ByteSource(px, dim, lb, 10);
        int[] idx = {4, 0, 2};
        boolean same = ms.size() == 5 && maxDiff(ms.features(idx, 0, 3), bs.features(idx, 0, 3)) == 0.0 && maxDiff(ms.targets(idx, 0, 3), bs.targets(idx, 0, 3)) == 0.0;
        MappedSource v = ms.view(2, 5);
        same = same && v.size() == 3 && maxDiff(v.features(new int[]{0}, 0, 1), bs.features(new int[]{2}, 0, 1)) == 0.0;
        check("MappedSource (data stays on disk) gives exactly the same batches as ByteSource, also through view()", same);
        img.delete(); lab.delete();

        CharSource cs = CharSource.fromText("abcabcabcabc", 3);
        int[] ix = {0, 1};
        Tensor f = cs.features(ix, 0, 2), t = cs.targets(ix, 0, 2);
        check("CharSource: windows of ids, vocabulary sorted ('abc' -> 0,1,2)", cs.vocabSize() == 3 && cs.size() == 8 && f.get(0, 0) == 0 && f.get(0, 2) == 2 && f.get(1, 0) == 1 && cs.id('c') == 2);
        check("CharSource: the target at every position is the NEXT character (a b c -> b c a)",
              t.rows == 6 && t.cols == 3 && t.get(0, 1) == 1 && t.get(1, 2) == 1 && t.get(2, 0) == 1 && t.get(3, 2) == 1);
        check("CharSource.view shares the text without copying", cs.view(4, 8).size() == 4 && cs.view(4, 8).features(new int[]{0}, 0, 1).get(0, 0) == 1);
    }

    // ------------------------------------------------------------------ round 4: fast kernels, dilated + grouped convolution

    static void kernelAgreement() {
        Random r = new Random(61);
        int[][] shapes = {{1, 1, 1}, {3, 5, 7}, {5, 17, 33}, {37, 300, 70}, {13, 9, 129}, {64, 784, 128}, {10, 64, 10}, {4, 32, 32}, {9, 20, 100}, {70, 33, 17}};
        boolean old = Tensor.isVector(), oldPar = Tensor.isParallel();
        double worstPortable = 0, worstVector = 0, worstTA = 0, worstTB = 0;
        for (int[] sh : shapes) {
            int M = sh[0], K = sh[1], N = sh[2];
            Tensor a = Tensor.randn(M, K, 1, r), b = Tensor.randn(K, N, 1, r);
            Tensor ta = Tensor.randn(M, N, 1, r);      // for A^T . B   (A = a: M x K, B = ta: M x N)  -> K x N
            Tensor tb = Tensor.randn(N, K, 1, r);      // for A . B^T   (A = a: M x K, B = tb: N x K)  -> M x N
            Tensor want = a.matmulNaive(b);
            Tensor wantTA = a.transpose().matmulNaive(ta), wantTB = a.matmulNaive(tb.transpose());
            Tensor.setVector(false);
            worstPortable = Math.max(worstPortable, maxDiff(a.matmul(b), want) / (1 + K * 0.01));
            if (Tensor.vectorAvailable()) {
                Tensor.setVector(true);
                Tensor.setParallel(false);
                worstVector = Math.max(worstVector, maxDiff(a.matmul(b), want) / (1 + K * 0.01));
                worstTA = Math.max(worstTA, maxDiff(a.matmulTA(ta), wantTA) / (1 + M * 0.01));
                worstTB = Math.max(worstTB, maxDiff(a.matmulTB(tb), wantTB) / (1 + K * 0.01));
                Tensor.setParallel(true);
                worstVector = Math.max(worstVector, maxDiff(a.matmul(b), want) / (1 + K * 0.01));
            }
        }
        Tensor.setVector(old); Tensor.setParallel(oldPar);
        check("portable kernels match the textbook multiply on 10 awkward shapes (worst " + String.format("%.1e", worstPortable) + ")", worstPortable < 1e-4);
        if (Tensor.vectorAvailable())
            check("Vector API kernels (multiply, A^T.B and A.B^T, 1 and many threads) match the textbook multiply on the same shapes (worst " + String.format("%.1e", Math.max(worstVector, Math.max(worstTA, worstTB))) + ")",
                  worstVector < 1e-4 && worstTA < 1e-4 && worstTB < 1e-4);
        else
            check("Vector API kernels: skipped, not enabled in this run (start Java with --add-modules jdk.incubator.vector to test them)", true);
    }

    /** Plain convolution with dilation and groups, to compare against. W is (outC x cinPerGroup*k*k). */
    static Tensor naiveConvG(Tensor x, Tensor W, Tensor b, int inC, int H, int Wd, int outC, int k, int stride, int pad, int dil, int groups) {
        int span = dil * (k - 1) + 1, oh = (H + 2 * pad - span) / stride + 1, ow = (Wd + 2 * pad - span) / stride + 1;
        int cing = inC / groups, coutg = outC / groups, Kg = cing * k * k;
        Tensor out = new Tensor(x.rows, outC * oh * ow);
        for (int n = 0; n < x.rows; n++)
            for (int oc = 0; oc < outC; oc++) {
                int g = oc / coutg;
                for (int oy = 0; oy < oh; oy++)
                    for (int ox = 0; ox < ow; ox++) {
                        double s = b.data[oc];
                        for (int cl = 0; cl < cing; cl++)
                            for (int ky = 0; ky < k; ky++)
                                for (int kx = 0; kx < k; kx++) {
                                    int iy = oy * stride - pad + ky * dil, ix = ox * stride - pad + kx * dil;
                                    if (iy < 0 || iy >= H || ix < 0 || ix >= Wd) continue;
                                    s += x.data[n * inC * H * Wd + ((g * cing + cl) * H + iy) * Wd + ix] * W.data[oc * Kg + (cl * k + ky) * k + kx];
                                }
                        out.data[n * outC * oh * ow + (oc * oh + oy) * ow + ox] = (float) s;
                    }
            }
        return out;
    }

    static void convOptions() throws Exception {
        Random r = new Random(71);
        int[][] cfg = {{2, 9, 8, 4, 3, 1, 2, 2, 1}, {4, 8, 8, 6, 3, 1, 1, 1, 2}, {6, 7, 7, 6, 3, 1, 1, 1, 6}, {4, 10, 9, 4, 3, 2, 2, 2, 2}, {3, 6, 6, 9, 1, 1, 0, 1, 3}};
        boolean allMatch = true;
        for (int[] c : cfg) {
            Conv2D conv = new Conv2D(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], r);
            Tensor x = Tensor.randn(3, c[0] * c[1] * c[2], 1, r);
            Tensor want = naiveConvG(x, conv.params().get(0).value, conv.params().get(1).value, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
            Tensor got = conv.forward(x);
            if (got.rows != want.rows || got.cols != want.cols || maxDiff(got, want) > 1e-4) allMatch = false;
        }
        check("Conv2D with dilation and groups matches a plain loop (5 settings: dilated, grouped, depthwise, strided+both, 1x1 grouped)", allMatch);

        Tensor x = Tensor.randn(4, 4 * 6 * 6, 1, r), y = Tensor.randn(4, 3, 1, r);
        Sequential grouped = new Sequential().add(new Conv2D(4, 6, 6, 4, 3, 1, 1, 1, 2, r)).add(new Activations.Tanh()).add(new Dense(4 * 36, 3, r));
        check("gradcheck grouped Conv2D (groups 2)", GradCheck.maxRelativeError(grouped, new Losses.MSE(), x, y) < GC_TOL);
        Sequential dilated = new Sequential().add(new Conv2D(4, 6, 6, 3, 3, 1, 2, 2, 1, r)).add(new Activations.Tanh()).add(new Dense(3 * 36, 3, r));
        check("gradcheck dilated Conv2D (dilation 2)", GradCheck.maxRelativeError(dilated, new Losses.MSE(), x, y) < GC_TOL);
        Sequential depthwise = new Sequential().add(new Conv2D(4, 6, 6, 4, 3, 1, 1, 1, 4, r)).add(new Activations.Tanh())
            .add(new Conv2D(4, 6, 6, 5, 1, 1, 0, r)).add(new Activations.Tanh()).add(new Dense(5 * 36, 3, r));
        check("gradcheck depthwise-separable block (depthwise 3x3 groups=channels, then 1x1 pointwise)", GradCheck.maxRelativeError(depthwise, new Losses.MSE(), x, y) < GC_TOL_DEEP);

        int weightsGrouped = new Conv2D(8, 8, 8, 8, 3, 1, 1, 1, 2, r).params().get(0).value.data.length;
        int weightsPlain = new Conv2D(8, 8, 8, 8, 3, 1, 1, r).params().get(0).value.data.length;
        check("groups cut the weights: 8->8 channels, 3x3: plain " + weightsPlain + ", groups=2 " + weightsGrouped + ", depthwise " + new Conv2D(8, 8, 8, 8, 3, 1, 1, 1, 8, r).params().get(0).value.data.length,
              weightsGrouped * 2 == weightsPlain);
        boolean threw = false;
        try { new Conv2D(5, 8, 8, 4, 3, 1, 1, 1, 2, r); } catch (IllegalArgumentException e) { threw = e.getMessage().contains("divisible"); }
        check("groups must divide the channels (clear error)", threw);

        File txt = File.createTempFile("minidl-cv", ".txt"), bin = File.createTempFile("minidl-cv", ".bin");
        ModelIO.save(grouped, txt.getPath());
        ModelIO.saveBinary(grouped, bin.getPath());
        boolean ok = maxDiff(grouped.predict(x), ModelIO.loadModel(txt.getPath(), new Random(2)).predict(x)) == 0.0
            && maxDiff(grouped.predict(x), ModelIO.loadModelBinary(bin.getPath(), new Random(2)).predict(x)) == 0.0
            && grouped.layers().get(0).spec().endsWith(" 1 2");
        Sequential q = Quantizer.quantize(grouped);
        ok = ok && q.layers().get(0).spec().startsWith("Conv2D") && q.layers().get(2).spec().startsWith("QDense");
        check("grouped convolution saves/loads (text + binary), and the quantizer leaves it in float32 while still shrinking the Dense layer", ok);
        txt.delete(); bin.delete();

        // vector vs portable on a convolution with a real size
        if (Tensor.vectorAvailable()) {
            boolean oldV = Tensor.isVector();
            Conv2D cv = new Conv2D(8, 13, 13, 16, 3, 1, 0, new Random(5));
            Tensor xb = Tensor.randn(6, 8 * 13 * 13, 1, new Random(6)), gb = Tensor.randn(6, 16 * 11 * 11, 1, new Random(7));
            Tensor.setVector(false);
            Tensor o1 = cv.forward(xb), d1 = cv.backward(gb), w1 = cv.params().get(0).grad.copy();
            Tensor.setVector(true);
            Tensor o2 = cv.forward(xb), d2 = cv.backward(gb), w2 = cv.params().get(0).grad.copy();
            Tensor.setVector(oldV);
            check("Conv2D gives the same forward, input-gradient and weight-gradient with Vector API kernels as with portable ones",
                  maxDiff(o1, o2) < 1e-3 && maxDiff(d1, d2) < 1e-3 && maxDiff(w1, w2) < 5e-3);
        } else {
            check("Conv2D vector-vs-portable comparison: skipped, Vector API not enabled in this run", true);
        }
    }

    // ------------------------------------------------------------------ round 4: 16-bit numbers and mixed-precision training

    static void halfPrecision() {
        int bad = 0, nan = 0;
        for (int h = 0; h < 65536; h++) {
            float f = Half.fromFP16((short) h);
            if (Float.isNaN(f)) { nan++; if (!Float.isNaN(Half.fromFP16(Half.toFP16(f)))) bad++; continue; }
            if (Half.toFP16(f) != (short) h) bad++;
        }
        check("FP16: all 65,536 bit patterns survive half -> float -> half (" + nan + " of them NaN)", bad == 0);

        check("FP16 known values: 1 = 0x3C00, -2 = 0xC000, 65504 = 0x7BFF (the biggest), 2^-14 = 0x0400, 2^-24 = 0x0001 (the smallest)",
              Half.toFP16(1f) == 0x3C00 && Half.toFP16(-2f) == (short) 0xC000 && Half.toFP16(65504f) == 0x7BFF
              && Half.toFP16((float) Math.pow(2, -14)) == 0x0400 && Half.toFP16((float) Math.pow(2, -24)) == 0x0001);
        check("FP16 limits: 65520 rounds to infinity, 1e9 is infinity, 1e-9 and 2^-25 become 0",
              Half.toFP16(65520f) == 0x7C00 && Half.toFP16(1e9f) == 0x7C00 && Half.toFP16(1e-9f) == 0 && Half.toFP16((float) Math.pow(2, -25)) == 0);

        Random r = new Random(81);
        int wrong = 0;
        for (int t = 0; t < 20000; t++) {
            float x = (float) (r.nextGaussian() * Math.pow(2, r.nextInt(30) - 16));
            if (Math.abs(x) > 60000) continue;
            short h = Half.toFP16(x);
            double err = Math.abs(x - Half.fromFP16(h));
            double up = Math.abs(x - Half.fromFP16((short) (h + 1))), down = Math.abs(x - Half.fromFP16((short) (h - 1)));
            boolean negative = x < 0; // bit patterns run away from zero for negatives too, so +1/-1 still are the two neighbours
            if (err > up + 1e-12 || err > down + 1e-12) wrong++;
        }
        check("FP16 rounds to the NEAREST half-precision number (20,000 random values)", wrong == 0);

        check("BF16 known values: 1 = 0x3F80; a tie rounds to even (1 + 2^-8 -> 1); infinity and NaN survive",
              Half.toBF16(1f) == 0x3F80 && Half.fromBF16(Half.toBF16(1f + (float) Math.pow(2, -8))) == 1f
              && Float.isInfinite(Half.fromBF16(Half.toBF16(Float.POSITIVE_INFINITY))) && Float.isNaN(Half.fromBF16(Half.toBF16(Float.NaN))));
        double worstRel = 0;
        for (int t = 0; t < 20000; t++) {
            float x = (float) (r.nextGaussian() * Math.pow(2, r.nextInt(60) - 30));
            if (x == 0) continue;
            worstRel = Math.max(worstRel, Math.abs((Half.round(x, Precision.BF16) - x) / x));
        }
        check("BF16 keeps about 3 digits at ANY size: worst relative error " + String.format("%.4f", worstRel) + " (limit 2^-8 = 0.0039)", worstRel <= 0.0039063);

        float tiny = 1e-8f;
        check("underflow: 1e-8 becomes 0 in FP16 but survives in BF16 (same range as float32)",
              Half.round(tiny, Precision.FP16) == 0f && Math.abs(Half.round(tiny, Precision.BF16) - tiny) / tiny < 0.004);
        float scaled = Half.round(tiny * 65536f, Precision.FP16) / 65536f;
        check("loss scaling rescues it: 1e-8 x 65536 survives FP16, and dividing by 65536 gives back " + scaled, Math.abs(scaled - tiny) / tiny < 0.01);
        check("overflow: 70000 becomes infinity in FP16 but is fine in BF16", Float.isInfinite(Half.round(70000f, Precision.FP16)) && !Float.isInfinite(Half.round(70000f, Precision.BF16)));

        LossScaler d = LossScaler.dynamic(8, 3);
        d.update(true);
        boolean halved = d.scale() == 4 && d.skippedSteps() == 1;
        d.update(false); d.update(false); d.update(false);
        boolean doubled = d.scale() == 8;
        LossScaler none = LossScaler.none(); none.update(true);
        check("LossScaler: halves on overflow, doubles after 3 good steps, 'none' never changes", halved && doubled && none.scale() == 1.0);
    }

    static void mixedPrecisionTraining() throws Exception {
        // 1. a gradient that FP16 loses without scaling and keeps with it
        Sequential lone = new Sequential().add(new Dense(1, 1, new Random(1)));
        Param w = lone.params().get(0);
        w.value.data[0] = 0f; lone.params().get(1).value.data[0] = 0f;
        Tensor x1 = Tensor.of(new double[][]{{1}}), y1 = Tensor.of(new double[][]{{-1e-8}}); // loss gradient = 2e-8
        double[] got = new double[4];
        Precision[] ps = {Precision.FP16, Precision.FP16, Precision.FP16, Precision.BF16};
        double[] scales = {1, 1024, 65536, 1};
        for (int i = 0; i < 4; i++) {
            lone.setPrecision(ps[i], LossScaler.fixed(scales[i]));
            lone.trainStep(x1, y1, new Losses.MSE(), null);
            got[i] = w.grad.data[0];
        }
        check("FP16 without loss scaling loses a 2e-8 gradient (got " + got[0] + "); scale 1024 keeps it (" + String.format("%.2e", got[1]) + "); scale 65536 keeps it (" + String.format("%.2e", got[2]) + "); BF16 keeps it with no scaling",
              got[0] == 0.0 && Math.abs(got[1] - 2e-8) / 2e-8 < 0.2 && Math.abs(got[2] - 2e-8) / 2e-8 < 0.01 && Math.abs(got[3] - 2e-8) / 2e-8 < 0.01);

        // 2. all three precisions learn a real (if small) problem
        Random r = new Random(91);
        int n = 240;
        Tensor x = new Tensor(n, 2), y = new Tensor(n, 2);
        for (int i = 0; i < n; i++) {
            int c = i % 2;
            x.set(i, 0, r.nextGaussian() * 0.6 + (c == 0 ? -1.2 : 1.2)); x.set(i, 1, r.nextGaussian() * 0.6 + (c == 0 ? 1.0 : -1.0));
            y.set(i, c, 1);
        }
        StringBuilder accs = new StringBuilder();
        boolean allLearn = true;
        for (Precision p : Precision.values()) {
            Sequential m = new Sequential().add(new Dense(2, 16, new Random(3))).add(new Activations.ReLU()).add(new Dense(16, 2, new Random(4)));
            if (p != Precision.FP32) m.setPrecision(p);
            m.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.01), 60, 32, 0, 0);
            double acc = m.accuracy(x, y);
            accs.append(p).append(' ').append(String.format("%.1f%%", acc)).append("  ");
            if (acc < 95) allLearn = false;
        }
        check("training in FP32, FP16 (dynamic loss scaling) and BF16 all learn the problem: " + accs.toString().trim(), allLearn);

        // 3. master weights stay full precision
        Sequential mm = new Sequential().add(new Dense(2, 16, new Random(3))).add(new Activations.ReLU()).add(new Dense(16, 2, new Random(4)));
        mm.setPrecision(Precision.FP16);
        mm.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.01), 5, 32, 0, 0);
        int total = 0, exact16 = 0;
        for (Param p : mm.params()) for (float v : p.value.data) { total++; if (Half.round(v, Precision.FP16) == v) exact16++; }
        check("weights stay full precision (32-bit master copy): only " + exact16 + " of " + total + " are exactly representable in FP16", exact16 < total / 2);

        // 4. overflow: step is skipped, weights untouched, scale halves
        Sequential ov = new Sequential().add(new Dense(2, 4, new Random(5))).add(new Activations.ReLU()).add(new Dense(4, 2, new Random(6)));
        LossScaler sc = LossScaler.dynamic(1e9, 100);
        ov.setPrecision(Precision.FP16, sc);
        List<float[]> before = ov.snapshot();
        ov.trainStep(x.sliceRows(0, 8), y.sliceRows(0, 8), new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.1));
        boolean same = true;
        List<float[]> after = ov.snapshot();
        for (int i = 0; i < before.size(); i++) if (!java.util.Arrays.equals(before.get(i), after.get(i))) same = false;
        check("FP16 overflow (scale 1e9): step skipped, weights unchanged, scale halved to " + sc.scale() + ", skipped steps " + sc.skippedSteps(),
              ov.lastStepOverflowed() && same && sc.scale() == 5e8 && sc.skippedSteps() == 1);

        // 5. 16-bit storage of what backward needs gives EXACTLY the same gradients when inputs are already 16-bit values
        Random rr = new Random(95);
        Dense d = new Dense(12, 5, rr);
        Tensor xd = Tensor.randn(9, 12, 1, rr), gd = Tensor.randn(9, 5, 1, rr);
        Half.roundInPlace(xd.data, Precision.BF16);
        d.forward(xd); d.backward(gd);
        Tensor g32 = d.params().get(0).grad.copy();
        d.setActivationStorage(Precision.BF16);
        d.forward(xd); d.backward(gd);
        boolean denseSame = maxDiff(g32, d.params().get(0).grad) == 0.0;
        Conv2D cv = new Conv2D(2, 6, 6, 3, 3, 1, 1, rr);
        Tensor xc = Tensor.randn(4, 72, 1, rr), gc = Tensor.randn(4, 3 * 36, 1, rr);
        Half.roundInPlace(xc.data, Precision.FP16);
        cv.forward(xc); cv.backward(gc);
        Tensor c32 = cv.params().get(0).grad.copy();
        cv.setActivationStorage(Precision.FP16);
        cv.forward(xc); cv.backward(gc);
        check("storing Dense and Conv2D inputs in 16 bits gives exactly the same gradients (inputs that are already 16-bit values)", denseSame && maxDiff(c32, cv.params().get(0).grad) == 0.0);

        // 6. the 1-bit ReLU mask
        Activations.ReLU relu = new Activations.ReLU();
        Tensor xr = new Tensor(1, 70), gr = new Tensor(1, 70);
        for (int i = 0; i < 70; i++) { xr.data[i] = i % 3 == 0 ? -1f : (i % 3 == 1 ? 0f : 2f); gr.data[i] = i + 1; }
        Tensor yr = relu.forward(xr), dr = relu.backward(gr);
        boolean maskOk = true;
        for (int i = 0; i < 70; i++) {
            boolean pos = xr.data[i] > 0;
            if (yr.data[i] != (pos ? xr.data[i] : 0f) || dr.data[i] != (pos ? gr.data[i] : 0f)) maskOk = false;
        }
        Activations.LeakyReLU leaky = new Activations.LeakyReLU(0.1);
        Tensor yl = leaky.forward(xr), dl = leaky.backward(gr);
        for (int i = 0; i < 70; i++) {
            boolean pos = xr.data[i] > 0;
            if (Math.abs(yl.data[i] - (pos ? xr.data[i] : 0.1f * xr.data[i])) > 1e-6 || Math.abs(dl.data[i] - (pos ? gr.data[i] : 0.1f * gr.data[i])) > 1e-5) maskOk = false;
        }
        check("ReLU and LeakyReLU keep just 1 bit per number (70 numbers = 2 words) and still give exact results", maskOk);

        // 7. whole transformer (embedding first layer, residuals) trains one mixed-precision step without breaking token ids
        Random rt = new Random(97);
        Sequential tf = miniTransformer(7, 5, 8, 2, rt);
        Tensor ids = new Tensor(3, 5), tg = new Tensor(15, 7);
        for (int i = 0; i < ids.data.length; i++) { ids.data[i] = rt.nextInt(7); tg.data[i * 7 + rt.nextInt(7)] = 1f; }
        tf.setPrecision(Precision.BF16);
        double l = tf.trainStep(ids, tg, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.01));
        check("a transformer (Embedding, Residual, TokenWise, attention) takes a BF16 training step with a finite loss (" + String.format("%.3f", l) + ")", Double.isFinite(l) && !tf.lastStepOverflowed());
    }

    // ------------------------------------------------------------------ round 4: N-dimensional tensors and shape inference

    static void ndTensor() {
        Random r = new Random(101);
        NDTensor a = new NDTensor(2, 3, 4);
        for (int i = 0; i < a.data.length; i++) a.data[i] = i;
        check("NDTensor indexing: element [1][2][3] of a 2x3x4 tensor is number 23", a.get(1, 2, 3) == 23 && a.rank() == 3 && a.size() == 24);

        NDTensor v = a.reshape(4, -1);
        a.data[5] = 99;
        check("reshape is a VIEW: same numbers, no copy (changing the original changes the view); -1 is worked out (2x3x4 -> 4x6)",
              v.dim(0) == 4 && v.dim(1) == 6 && v.data == a.data && v.get(0, 5) == 99);
        boolean threw = false;
        try { a.reshape(5, 5); } catch (IllegalArgumentException e) { threw = true; }
        check("reshape to a size that does not fit gives a clear error", threw);

        Tensor m = a.toMatrix(1);
        Tensor m2 = a.toMatrix(2);
        check("toMatrix: first dimension = rows (2x12), first two = rows (6x4); both share the numbers", m.rows == 2 && m.cols == 12 && m2.rows == 6 && m2.cols == 4 && m.data == a.data);

        NDTensor p = a.permute(2, 0, 1);
        boolean permOk = p.dim(0) == 4 && p.dim(1) == 2 && p.dim(2) == 3;
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 4; k++) if (p.get(k, i, j) != a.get(i, j, k)) permOk = false;
        NDTensor t = a.transpose(1, 2);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 4; k++) if (t.get(i, k, j) != a.get(i, j, k)) permOk = false;
        check("permute (2,0,1) and transpose(1,2) put every element where the indices say", permOk);

        NDTensor col = new NDTensor(new float[]{1, 2, 3}, 3, 1), row = new NDTensor(new float[]{10, 20, 30, 40}, 1, 4);
        NDTensor sum = col.add(row);
        check("broadcasting [3,1] + [1,4] -> [3,4]: element (2,3) = 3 + 40 = 43", sum.dim(0) == 3 && sum.dim(1) == 4 && sum.get(2, 3) == 43 && sum.get(0, 0) == 11);
        NDTensor bias = new NDTensor(new float[]{1, 2, 3, 4}, 4);
        NDTensor withBias = a.add(bias);
        NDTensor scaled = new NDTensor(2, 1, 4);
        java.util.Arrays.fill(scaled.data, 2f);
        NDTensor prod = a.mul(scaled).mul(new NDTensor(new float[]{1, 10, 100}, 1, 3, 1));
        check("broadcasting a [4] bias over [2,3,4], and [2,1,4] * [1,3,1]", withBias.get(1, 2, 3) == a.get(1, 2, 3) + 4 && prod.get(1, 2, 3) == a.get(1, 2, 3) * 2 * 100);
        threw = false;
        try { new NDTensor(3).add(new NDTensor(4)); } catch (IllegalArgumentException e) { threw = e.getMessage().contains("cannot broadcast"); }
        check("incompatible shapes ([3] + [4]) give a clear error", threw);

        NDTensor g = new NDTensor(2, 3, 4);
        for (int i = 0; i < g.data.length; i++) g.data[i] = (float) r.nextGaussian();
        NDTensor s1 = g.sum(1, false), s1k = g.sum(1, true), mean2 = g.mean(2, false), mx = g.max(0, false);
        double want = 0;
        for (int j = 0; j < 3; j++) want += g.get(1, j, 2);
        double wantMax = Math.max(g.get(0, 2, 3), g.get(1, 2, 3));
        check("sum / mean / max along an axis (shapes [2,4], [2,1,4], [2,3], [3,4] and the values)",
              s1.rank() == 2 && s1.dim(0) == 2 && s1.dim(1) == 4 && Math.abs(s1.get(1, 2) - want) < 1e-5 && s1k.rank() == 3 && s1k.dim(1) == 1
              && mean2.dim(1) == 3 && Math.abs(mean2.get(0, 1) * 4 - (g.get(0, 1, 0) + g.get(0, 1, 1) + g.get(0, 1, 2) + g.get(0, 1, 3))) < 1e-5 && mx.get(2, 3) == wantMax);

        NDTensor sm = g.softmax(2);
        boolean smOk = true;
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) { double z = 0; for (int k = 0; k < 4; k++) z += sm.get(i, j, k); if (Math.abs(z - 1) > 1e-5) smOk = false; }
        Tensor flat = g.toMatrix(2);
        Tensor ref = Losses.SoftmaxCrossEntropy.softmax(flat);
        check("softmax along the last axis: every row adds up to 1 and equals the 2D softmax", smOk && maxDiff(sm.toMatrix(2), ref) < 1e-6);

        NDTensor left = g.slice(1, 0, 2), right = g.slice(1, 2, 3);
        NDTensor joined = NDTensor.concat(1, left, right);
        boolean sliceOk = left.dim(1) == 2 && right.dim(1) == 1 && joined.dim(1) == 3;
        for (int i = 0; i < g.data.length; i++) if (joined.data[i] != g.data[i]) sliceOk = false;
        check("slice along an axis, then concat the pieces back, gives the original", sliceOk);

        // batched matmul versus plain loops
        NDTensor x = new NDTensor(2, 3, 5, 4), wgt = new NDTensor(4, 6);
        for (int i = 0; i < x.data.length; i++) x.data[i] = (float) r.nextGaussian();
        for (int i = 0; i < wgt.data.length; i++) wgt.data[i] = (float) r.nextGaussian();
        NDTensor y = x.matmul(wgt);
        double worst = 0;
        for (int b1 = 0; b1 < 2; b1++) for (int b2 = 0; b2 < 3; b2++) for (int i = 0; i < 5; i++) for (int j = 0; j < 6; j++) {
            double sacc = 0;
            for (int k = 0; k < 4; k++) sacc += x.get(b1, b2, i, k) * wgt.get(k, j);
            worst = Math.max(worst, Math.abs(y.get(b1, b2, i, j) - sacc));
        }
        NDTensor q = new NDTensor(3, 5, 4), k2 = new NDTensor(3, 4, 7);
        for (int i = 0; i < q.data.length; i++) q.data[i] = (float) r.nextGaussian();
        for (int i = 0; i < k2.data.length; i++) k2.data[i] = (float) r.nextGaussian();
        NDTensor qk = q.matmul(k2);
        for (int b = 0; b < 3; b++) for (int i = 0; i < 5; i++) for (int j = 0; j < 7; j++) {
            double sacc = 0;
            for (int k = 0; k < 4; k++) sacc += q.get(b, i, k) * k2.get(b, k, j);
            worst = Math.max(worst, Math.abs(qk.get(b, i, j) - sacc));
        }
        check("batched matmul: [2,3,5,4].[4,6] (shared weights) and [3,5,4].[3,4,7] (one per batch) match plain loops (worst " + String.format("%.1e", worst) + "), shapes " + java.util.Arrays.toString(y.shape) + " and " + java.util.Arrays.toString(qk.shape),
              worst < 1e-4 && y.rank() == 4 && y.dim(3) == 6 && qk.dim(2) == 7);

        // images straight into a model: [N, 1, 8, 8]
        Random rr = new Random(5);
        Sequential cnn = new Sequential().add(new Conv2D(1, 8, 8, 2, 3, 1, 0, rr)).add(new Activations.ReLU()).add(new MaxPool2D(2, 6, 6, 2)).add(new Dense(18, 3, rr));
        NDTensor imgs = new NDTensor(4, 1, 8, 8);
        for (int i = 0; i < imgs.data.length; i++) imgs.data[i] = (float) rr.nextGaussian();
        Tensor viaND = cnn.predict(imgs), viaMatrix = cnn.predict(imgs.toMatrix(1));
        check("Sequential.predict(NDTensor [N,1,8,8]) gives the same answer as the flattened matrix, and Tensor.nd() is a view",
              maxDiff(viaND, viaMatrix) == 0.0 && viaND.rows == 4 && viaND.cols == 3 && imgs.toMatrix(1).nd(4, 1, 8, 8).data == imgs.data);
    }

    static void shapeInference() {
        Random r = new Random(111);
        Sequential cnn = new Sequential()
            .add(new Conv2D(1, 28, 28, 8, 3, 1, 0, r)).add(new Activations.ReLU()).add(new MaxPool2D(8, 26, 26, 2))
            .add(new Conv2D(8, 13, 13, 16, 3, 1, 0, r)).add(new Activations.ReLU()).add(new MaxPool2D(16, 11, 11, 2))
            .add(new Dense(400, 64, r)).add(new Activations.ReLU()).add(new Dense(64, 10, r));
        List<int[]> sh = cnn.shapes(1, 28, 28);
        check("shape inference follows a CNN: [1,28,28] -> conv [8,26,26] -> pool [8,13,13] -> conv [16,11,11] -> pool [16,5,5] -> Dense [64] -> [10]",
              java.util.Arrays.equals(sh.get(0), new int[]{8, 26, 26}) && java.util.Arrays.equals(sh.get(2), new int[]{8, 13, 13}) && java.util.Arrays.equals(sh.get(5), new int[]{16, 5, 5})
              && java.util.Arrays.equals(sh.get(6), new int[]{64}) && java.util.Arrays.equals(sh.get(sh.size() - 1), new int[]{10}));
        check("shape inference also accepts the flattened input [784]", java.util.Arrays.equals(cnn.shapes(784).get(sh.size() - 1), new int[]{10}));

        Sequential wrong = new Sequential().add(new Conv2D(1, 28, 28, 8, 3, 1, 0, r)).add(new MaxPool2D(8, 26, 26, 2)).add(new Dense(1000, 10, r));
        String msg = "";
        try { wrong.shapes(1, 28, 28); } catch (IllegalArgumentException e) { msg = e.getMessage(); }
        check("a wrong Dense size is caught with the layer number and both sizes: \"" + msg + "\"", msg.startsWith("layer 3 (Dense 1000 10)") && msg.contains("1000") && msg.contains("[8, 13, 13]") && msg.contains("1352"));

        String summary = cnn.summary(1, 28, 28);
        check("summary(inputShape) prints a shape for every layer and the weight total", summary.contains("[8, 26, 26]") && summary.contains("[16, 5, 5]") && summary.contains("27562"));

        Sequential tf = miniTransformer(7, 5, 8, 2, r);
        List<int[]> tsh = tf.shapes(5);
        check("shape inference follows a transformer: ids [5] -> embedding [5,8] -> blocks keep [5,8] -> head [5,7] -> rows [5,7]",
              java.util.Arrays.equals(tsh.get(0), new int[]{5, 8}) && java.util.Arrays.equals(tsh.get(1), new int[]{5, 8}) && java.util.Arrays.equals(tsh.get(2), new int[]{5, 8})
              && java.util.Arrays.equals(tsh.get(3), new int[]{5, 7}) && java.util.Arrays.equals(tsh.get(4), new int[]{5, 7}));
        boolean threw = false;
        try { tf.shapes(6); } catch (IllegalArgumentException e) { threw = e.getMessage().startsWith("layer 1 (Embedding"); }
        check("a wrong sequence length is caught at the Embedding layer", threw);

        Sequential badRes = new Sequential().add(new Dense(4, 6, r)).add(new Residual(new Dense(6, 5, r)));
        threw = false;
        try { badRes.shapes(4); } catch (IllegalArgumentException e) { threw = e.getMessage().contains("Residual block must keep the size"); }
        check("a Residual block that changes the size is caught before any data is run", threw);

        Sequential q = Quantizer.quantize(cnn);
        check("quantized models keep their shapes", java.util.Arrays.equals(q.shapes(1, 28, 28).get(q.layers().size() - 1), new int[]{10}));
    }

    // ------------------------------------------------------------------ round 4: layers written with autograd

    /** Largest difference between autograd's gradient and a nudge-and-measure gradient for one expression. */
    static double varCheck(Function<Var, Var> f, Tensor x0) {
        Var xv = new Var(x0.copy());
        Var out = f.apply(xv);
        Tensor target = new Tensor(out.value.rows, out.value.cols);
        for (int i = 0; i < target.data.length; i++) target.data[i] = 0.3f * (i % 5) - 0.4f;
        out.mseLoss(target).backward();
        Tensor analytic = xv.grad;
        double eps = 3e-3, worst = 0;
        for (int i = 0; i < x0.data.length; i++) {
            Tensor up = x0.copy(), down = x0.copy();
            up.data[i] += (float) eps; down.data[i] -= (float) eps;
            double lu = f.apply(new Var(up)).mseLoss(target).value.data[0], ld = f.apply(new Var(down)).mseLoss(target).value.data[0];
            double numeric = (lu - ld) / (2 * eps), a = analytic.data[i];
            worst = Math.max(worst, Math.abs(a - numeric) / Math.max(1e-2, Math.abs(a) + Math.abs(numeric)));
        }
        return worst;
    }

    static void copyParams(Layer from, Layer to) {
        for (int i = 0; i < from.params().size(); i++)
            System.arraycopy(from.params().get(i).value.data, 0, to.params().get(i).value.data, 0, from.params().get(i).value.data.length);
    }

    static double gradDiff(Layer a, Layer b) {
        double worst = 0;
        for (int i = 0; i < a.params().size(); i++) {
            Tensor ga = a.params().get(i).grad, gb = b.params().get(i).grad;
            double scale = 1e-3;
            for (float v : ga.data) scale = Math.max(scale, Math.abs(v));
            worst = Math.max(worst, maxDiff(ga, gb) / scale);
        }
        return worst;
    }

    static Sequential autoTransformer(int V, int T, int D, int H, Random r) {
        return new Sequential().add(new Embedding(V, D, T, r))
            .add(new Residual(new TokenWise(T, D, AutoLayers.layerNorm(D)), AutoLayers.attention(T, D, H, r)))
            .add(new Residual(new TokenWise(T, D, AutoLayers.layerNorm(D), AutoLayers.dense(D, 2 * D, r), AutoLayers.gelu(), AutoLayers.dense(2 * D, D, r))))
            .add(new TokenWise(T, D, AutoLayers.layerNorm(D), AutoLayers.dense(D, V, r)))
            .add(new Reshape(V));
    }

    static void autogradLayers() throws Exception {
        Random r = new Random(121);
        Tensor pos = new Tensor(4, 5);
        for (int i = 0; i < pos.data.length; i++) pos.data[i] = 0.5f + r.nextFloat() * 1.5f; // positive numbers for log / sqrt / 1/x

        java.util.LinkedHashMap<String, Function<Var, Var>> ops = new java.util.LinkedHashMap<>();
        ops.put("neg", Var::neg); ops.put("shift", v -> v.shift(1.5)); ops.put("exp", Var::exp); ops.put("log", Var::log);
        ops.put("sqrt", Var::sqrt); ops.put("inv", Var::inv); ops.put("div", v -> v.div(v.shift(3)));
        ops.put("rowSum", Var::rowSum); ops.put("rowMean", Var::rowMean); ops.put("rowSoftmax", Var::rowSoftmax);
        ops.put("transpose", Var::transpose); ops.put("matmul", v -> v.matmul(v.transpose()));
        ops.put("addCol", v -> v.addCol(v.sliceCols(0, 1))); ops.put("subCol", v -> v.subCol(v.sliceCols(1, 2)));
        ops.put("mulCol", v -> v.mulCol(v.sliceCols(0, 1))); ops.put("divCol", v -> v.divCol(v.sliceCols(2, 3).shift(1)));
        ops.put("mulRow", v -> v.mulRow(v.sliceRows(0, 1))); ops.put("sliceCols", v -> v.sliceCols(1, 4)); ops.put("sliceRows", v -> v.sliceRows(1, 3));
        ops.put("concatCols", v -> Var.concatCols(List.of(v, v.scale(2)))); ops.put("concatRows", v -> Var.concatRows(List.of(v, v.tanh())));
        ops.put("reshape", v -> v.reshape(2, 10)); ops.put("sigmoid", Var::sigmoid); ops.put("tanh", Var::tanh);
        double worst = 0; String worstName = "";
        for (java.util.Map.Entry<String, Function<Var, Var>> e : ops.entrySet()) {
            double d = varCheck(e.getValue(), pos);
            if (d > worst) { worst = d; worstName = e.getKey(); }
        }
        check("autograd: all " + ops.size() + " operations match a nudge-and-measure gradient (worst " + String.format("%.1e", worst) + " in '" + worstName + "')", worst < 1e-2);

        // Dense
        Dense hd = new Dense(6, 4, r);
        Layer ad = AutoLayers.dense(6, 4, new Random(7));
        copyParams(hd, ad);
        Tensor x = Tensor.randn(5, 6, 1, r), g = Tensor.randn(5, 4, 1, r);
        Tensor hy = hd.forward(x), ay = ad.forward(x);
        Tensor hdx = hd.backward(g), adx = ad.backward(g);
        check("AutoDense (one line of forward maths) = Dense: same output (" + String.format("%.0e", maxDiff(hy, ay)) + "), same weight gradients and input gradient",
              maxDiff(hy, ay) < 1e-5 && gradDiff(hd, ad) < 1e-4 && maxDiff(hdx, adx) < 1e-4);

        // LayerNorm
        LayerNorm hl = new LayerNorm(8);
        Layer al = AutoLayers.layerNorm(8);
        for (int i = 0; i < 8; i++) { hl.params().get(0).value.data[i] = 0.5f + r.nextFloat(); hl.params().get(1).value.data[i] = (float) r.nextGaussian(); }
        copyParams(hl, al);
        Tensor xl = Tensor.randn(6, 8, 2, r), gl = Tensor.randn(6, 8, 1, r);
        Tensor ly1 = hl.forward(xl), ly2 = al.forward(xl);
        Tensor ldx1 = hl.backward(gl), ldx2 = al.backward(gl);
        check("AutoLayerNorm = LayerNorm: same output, same gamma/beta gradients, same input gradient",
              maxDiff(ly1, ly2) < 1e-4 && gradDiff(hl, al) < 1e-3 && maxDiff(ldx1, ldx2) < 1e-3);

        // GELU
        Activations.GELU hg = new Activations.GELU();
        Layer ag = AutoLayers.gelu();
        Tensor xg = Tensor.randn(4, 9, 2, r), gg = Tensor.randn(4, 9, 1, r);
        check("AutoGELU (built from tanh) = GELU: same output and input gradient",
              maxDiff(hg.forward(xg), ag.forward(xg)) < 1e-5 && maxDiff(hg.backward(gg), ag.backward(gg)) < 1e-4);

        // Attention
        int T = 5, D = 8, H = 2, N = 3;
        CausalSelfAttention ha = new CausalSelfAttention(T, D, H, r);
        Layer aa = AutoLayers.attention(T, D, H, new Random(9));
        for (Param p : ha.params()) for (int i = 0; i < p.value.data.length; i++) if (p.value.data[i] == 0f) p.value.data[i] = (float) (r.nextGaussian() * 0.3);
        copyParams(ha, aa);
        Tensor xa = Tensor.randn(N, T * D, 1, r), ga = Tensor.randn(N, T * D, 1, r);
        Tensor ay1 = ha.forward(xa), ay2 = aa.forward(xa);
        Tensor adx1 = ha.backward(ga), adx2 = aa.backward(ga);
        check("AutoAttention (multi-head causal attention in forward maths only) = CausalSelfAttention: output (" + String.format("%.0e", maxDiff(ay1, ay2)) + "), all 4 weight gradients and the input gradient agree",
              maxDiff(ay1, ay2) < 1e-4 && gradDiff(ha, aa) < 2e-3 && maxDiff(adx1, adx2) < 2e-3);

        // a whole network of autograd layers trains
        Tensor xs = Tensor.of(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}}), ys = Tensor.of(new double[][]{{0}, {1}, {1}, {0}});
        Random rx = new Random(3);
        Layer tanhL = new AutoLayer("AutoTanh", List.of(), (v, p) -> v.tanh()), sigL = new AutoLayer("AutoSigmoid", List.of(), (v, p) -> v.sigmoid());
        Sequential xor = new Sequential().add(AutoLayers.dense(2, 8, rx)).add(tanhL).add(AutoLayers.dense(8, 1, rx)).add(sigL);
        xor.fit(xs, ys, new Losses.MSE(), new Optimizers.Adam(0.05), 400, 4, 0, 0);
        double xl2 = new Losses.MSE().forward(xor.predict(xs), ys);
        check("a model made ONLY of autograd layers trains with the normal Trainer and Adam: XOR loss " + String.format("%.4f", xl2), xl2 < 0.02);

        // mixing hand-written and autograd layers
        Sequential mixed = new Sequential().add(new Dense(5, 6, r)).add(AutoLayers.layerNorm(6)).add(AutoLayers.gelu()).add(new Dense(6, 3, r));
        check("gradcheck a model that mixes hand-written Dense with AutoLayerNorm and AutoGELU",
              GradCheck.maxRelativeError(mixed, new Losses.MSE(), Tensor.randn(4, 5, 1, r), Tensor.randn(4, 3, 1, r)) < GC_TOL_DEEP);

        // whole transformer, hand-written vs autograd
        int V = 7;
        Sequential hand = miniTransformer(V, T, D, H, new Random(131)), auto = autoTransformer(V, T, D, H, new Random(132));
        for (int i = 0; i < hand.params().size(); i++) {
            Param ph = hand.params().get(i), pa = auto.params().get(i);
            if (ph.value.data.length != pa.value.data.length) throw new IllegalStateException("parameter lists differ at " + i);
            System.arraycopy(ph.value.data, 0, pa.value.data, 0, ph.value.data.length);
        }
        Tensor ids = new Tensor(3, T), tg = new Tensor(3 * T, V);
        for (int i = 0; i < ids.data.length; i++) { ids.data[i] = r.nextInt(V); tg.data[i * V + r.nextInt(V)] = 1f; }
        Losses.SoftmaxCrossEntropy ce1 = new Losses.SoftmaxCrossEntropy(), ce2 = new Losses.SoftmaxCrossEntropy();
        for (Param p : hand.params()) p.zeroGrad();
        for (Param p : auto.params()) p.zeroGrad();
        Tensor o1 = hand.forward(ids), o2 = auto.forward(ids);
        double l1 = ce1.forward(o1, tg), l2 = ce2.forward(o2, tg);
        hand.backward(ce1.backward());
        auto.backward(ce2.backward());
        double worstG = 0;
        for (int i = 0; i < hand.params().size(); i++) {
            Tensor a1 = hand.params().get(i).grad, a2 = auto.params().get(i).grad;
            double sc = 1e-3;
            for (float v : a1.data) sc = Math.max(sc, Math.abs(v));
            worstG = Math.max(worstG, maxDiff(a1, a2) / sc);
        }
        check("a transformer built ENTIRELY from autograd layers matches the hand-written transformer: loss " + String.format("%.5f vs %.5f", l1, l2) + ", all " + hand.params().size() + " weight gradients agree (worst " + String.format("%.1e", worstG) + " of the gradient size)",
              Math.abs(l1 - l2) < 1e-4 && maxDiff(o1, o2) < 1e-3 && worstG < 5e-3);

        // saving and loading autograd layers, including through the registry
        File txt = File.createTempFile("minidl-auto", ".txt"), bin = File.createTempFile("minidl-auto", ".bin");
        ModelIO.save(auto, txt.getPath());
        ModelIO.saveBinary(auto, bin.getPath());
        Sequential fromTxt = ModelIO.loadModel(txt.getPath(), new Random(1)), fromBin = ModelIO.loadModelBinary(bin.getPath(), new Random(1));
        check("autograd layers save and load (text and binary) and rebuild through the layer registry with identical predictions",
              maxDiff(auto.predict(ids), fromTxt.predict(ids)) == 0.0 && maxDiff(auto.predict(ids), fromBin.predict(ids)) == 0.0);
        txt.delete(); bin.delete();

        File t2 = File.createTempFile("minidl-custom", ".txt");
        ModelIO.save(new Sequential().add(new AutoLayer("MyOwnLayer 3", List.of(), (v, p) -> v)), t2.getPath());
        String msg = "";
        try { ModelIO.loadModel(t2.getPath(), new Random(1)); } catch (IOException e) { msg = e.getMessage(); }
        check("an unknown custom layer gives a clear message that tells you about ModelIO.register", msg.contains("MyOwnLayer") && msg.contains("register"));
        ModelIO.register("MyOwnLayer", (w, rnd) -> new AutoLayer("MyOwnLayer " + w[1], List.of(), (v, p) -> v));
        check("after ModelIO.register(...) the same file loads", ModelIO.loadModel(t2.getPath(), new Random(1)).layers().size() == 1);
        t2.delete();
    }

    // ------------------------------------------------------------------ round 4: CIFAR-10 format and colour augmentation

    static void cifarAndColour() throws Exception {
        File f = File.createTempFile("minidl-cifar", ".bin");
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(f))) {
            for (int rec = 0; rec < 3; rec++) {
                o.write(new int[]{6, 0, 9}[rec]);
                for (int i = 0; i < 3072; i++) o.write(i < 1024 ? 255 : (i < 2048 ? rec * 100 : 51));   // red plane, green plane, blue plane
            }
        }
        ByteSource src = CifarLoader.load(f.getPath(), f.getPath());
        int[] idx = {0, 1, 2, 3, 4, 5};
        Tensor x = src.features(idx, 0, 6), y = src.targets(idx, 0, 6);
        check("CifarLoader reads the official binary format (label byte + 1024 red + 1024 green + 1024 blue) from two files: 6 images of 3072 numbers",
              src.size() == 6 && x.cols == 3072 && x.get(1, 0) == 1.0 && Math.abs(x.get(1, 1024) - 100f / 255f) < 1e-6 && Math.abs(x.get(2, 2048) - 0.2) < 1e-6 && Math.abs(x.get(0, 1500) - 0.0) < 1e-6
              && Math.abs(x.get(2, 1500) - 200f / 255f) < 1e-6 && y.get(0, 6) == 1 && y.get(2, 9) == 1 && y.get(4, 0) == 1 && y.get(5, 9) == 1);
        File bad = File.createTempFile("minidl-cifar-bad", ".bin");
        try (FileOutputStream o = new FileOutputStream(bad)) { o.write(new byte[100]); }
        String msg = "";
        try { CifarLoader.load(bad.getPath()); } catch (IOException e) { msg = e.getMessage(); }
        check("a file that is not CIFAR-10 sized gives a clear error", msg.contains("not a CIFAR-10 binary file"));
        f.delete(); bad.delete();

        float[] plane = {1, 2, 3, 4, 5, 6};
        AugmentedSource.mirrorRows(plane, 0, 2, 3);
        check("mirror flips each row left-right ([1,2,3 / 4,5,6] -> [3,2,1 / 6,5,4])", java.util.Arrays.equals(plane, new float[]{3, 2, 1, 6, 5, 4}));

        // 3-channel images: every channel gets the SAME move
        int h = 6, w = 6;
        byte[] px = new byte[40 * 3 * h * w], lab = new byte[40];
        for (int i = 0; i < 40; i++) for (int ch = 0; ch < 3; ch++) px[i * 3 * h * w + ch * h * w + 1 * w + 0] = (byte) (100 * (ch + 1)); // a dot at (row 1, col 0) in each channel
        ByteSource base = new ByteSource(px, 3 * h * w, lab, 2);
        int[] all = new int[40];
        for (int i = 0; i < 40; i++) all[i] = i;
        Tensor out = new AugmentedSource(base, 3, h, w, 0, 0, true, 3).features(all, 0, 40);
        int flipped = 0, plain = 0; boolean together = true;
        for (int i = 0; i < 40; i++) {
            double r0 = out.get(i, 1 * w + 0), g0 = out.get(i, h * w + 1 * w + 0), r5 = out.get(i, 1 * w + 5), g5 = out.get(i, h * w + 1 * w + 5), b5 = out.get(i, 2 * h * w + 1 * w + 5);
            if (r0 > 0 && g0 > 0 && r5 == 0 && g5 == 0) plain++;
            else if (r5 > 0 && g5 > 0 && b5 > 0 && r0 == 0 && g0 == 0) flipped++;
            else together = false;
        }
        check("colour augmentation with flips: about half the images are mirrored (" + flipped + " of 40) and all 3 channels always move together", flipped > 8 && plain > 8 && flipped + plain == 40 && together);
        Tensor shifted = new AugmentedSource(base, 3, h, w, 2, 0, false, 9).features(all, 0, 40);
        boolean sameMove = true;
        for (int i = 0; i < 40; i++) {
            int posR = -1, posG = -1, posB = -1;
            for (int p = 0; p < h * w; p++) {
                if (shifted.get(i, p) > 0.1f && posR < 0) posR = p;
                if (shifted.get(i, h * w + p) > 0.1f && posG < 0) posG = p;
                if (shifted.get(i, 2 * h * w + p) > 0.1f && posB < 0) posB = p;
            }
            if (posR != posG || posG != posB) sameMove = false;
        }
        check("shifted colour images: the red, green and blue dots end up at the same place", sameMove);
    }

    // ------------------------------------------------------------------ handwriting app
    static void handwriting() {
        int w = 120, h = 80;
        float[] ink = new float[w * h];
        for (int y = 5; y < 25; y++) for (int x = 90; x < 100; x++) ink[y * w + x] = 1f; // a thin bar in the top-right corner
        float[] px = DigitPreprocess.toMnist(ink, w, h);
        double mass = 0, cx = 0, cy = 0; float max = 0;
        for (int i = 0; i < 784; i++) { mass += px[i]; cx += px[i] * (i % 28); cy += px[i] * (i / 28); max = Math.max(max, px[i]); }
        check("preprocess: a shape in the corner ends up centred in the 28x28 picture (centre of mass 13.5 +/- 0.6)",
            Math.abs(cx / mass - 13.5) < 0.6 && Math.abs(cy / mass - 13.5) < 0.6);
        check("preprocess: strongest pixel is 1.0 and the digit fits the 20 px box", max == 1f && px[0] == 0 && px[27] == 0 && px[27 * 28] == 0);
        check("preprocess: a blank page gives null", DigitPreprocess.toMnist(new float[w * h], w, h) == null);

        float[] two = new float[w * h];
        for (int y = 10; y < 60; y++) { for (int x = 10; x < 18; x++) two[y * w + x] = 1f; for (int x = 70; x < 90; x++) two[y * w + x] = 1f; }
        two[2 * w + 50] = 1f; // one speck of dust between them
        List<DigitPreprocess.Piece> parts = DigitPreprocess.segment(two, w, h);
        check("segment: two digits with a gap are found left to right, dust ignored (" + parts.size() + ")",
            parts.size() == 2 && parts.get(0).x < parts.get(1).x && parts.get(0).w == 8 && parts.get(1).w == 20);
        float[] touch = new float[w * h];
        for (int y = 10; y < 60; y++) for (int x = 10; x < 40; x++) touch[y * w + x] = 1f;
        check("segment: one connected blob stays one digit", DigitPreprocess.segment(touch, w, h).size() == 1);

        float[] sh = DigitRecognizer.shift(px, 1, 0);
        boolean moved = true;
        for (int y = 0; y < 28; y++) for (int x = 1; x < 28; x++) if (sh[y * 28 + x] != px[y * 28 + x - 1]) moved = false;
        check("recognizer: shift moves the picture one pixel right", moved);

        Random rnd = new Random(3);
        Sequential m = new Sequential().add(new Dense(784, 16, rnd)).add(new Activations.ReLU()).add(new Dense(16, 10, rnd));
        DigitRecognizer rec = new DigitRecognizer(m);
        DigitRecognizer.Result before = rec.predict(px, false);
        double sum = 0; for (double p : before.probs) sum += p;
        check("recognizer: probabilities add up to 1 (with and without test-time augmentation)",
            Math.abs(sum - 1) < 1e-6 && Math.abs(java.util.Arrays.stream(rec.predict(px, true).probs).sum() - 1) < 1e-6);
        int target = (before.label + 3) % 10;
        for (int i = 0; i < 6; i++) rec.learn(px, target);
        DigitRecognizer.Result after = rec.predict(px, false);
        check("learn: after 6 corrections the model says " + target + " (p " + String.format("%.2f", before.probs[target]) + " -> " + String.format("%.2f", after.probs[target]) + ")",
            after.label == target && after.probs[target] > before.probs[target] && rec.learnedCount() == 6);
        rec.reset();
        DigitRecognizer.Result back = rec.predict(px, false);
        check("reset: the original answers come back exactly", back.label == before.label && Math.abs(back.probs[target] - before.probs[target]) < 1e-9 && rec.learnedCount() == 0);
        boolean bad = false;
        try { rec.learn(px, 10); } catch (IllegalArgumentException e) { bad = true; }
        check("learn: a label outside 0..9 gives a clear error", bad);
    }

    static void letters() throws Exception {
        File img = File.createTempFile("emn-img", ".idx"), lab = File.createTempFile("emn-lab", ".idx");
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(img))) {
            o.writeInt(2051); o.writeInt(2); o.writeInt(28); o.writeInt(28);
            byte[] a = new byte[784], b = new byte[784];
            a[5 * 28 + 2] = (byte) 200;   // stored sideways: row 5, column 2
            b[0] = 1;
            o.write(a); o.write(b);
        }
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(lab))) { o.writeInt(2049); o.writeInt(2); o.write(new byte[]{1, 26}); }
        ByteSource src = EmnistLoader.letters(img.getPath(), lab.getPath(), 10);
        Tensor x = src.features(new int[]{0, 1}, 0, 2), y = src.targets(new int[]{0, 1}, 0, 2);
        check("EMNIST: pictures are turned back (stored row 5, col 2 -> row 2, col 5)", Math.abs(x.get(0, 2 * 28 + 5) - 200 / 255.0) < 1e-6 && x.get(0, 5 * 28 + 2) == 0);
        check("EMNIST: labels 1..26 become classes 0..25 (A = 0, Z = 25), 26 columns", y.cols == 26 && y.get(0, 0) == 1 && y.get(1, 25) == 1);
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(lab))) { o.writeInt(2049); o.writeInt(2); o.write(new byte[]{0, 5}); }
        String msg = "";
        try { EmnistLoader.letters(img.getPath(), lab.getPath(), 10); } catch (IOException e) { msg = e.getMessage(); }
        check("EMNIST: a label 0 (not a letter) gives a clear error", msg.contains("outside"));
        img.delete(); lab.delete();

        int w = 100, h = 60;
        float[] ink = new float[w * h];
        for (int yy = 10; yy < 50; yy++) for (int xx = 40; xx < 50; xx++) ink[yy * w + xx] = 1f; // a tall bar
        float[] mn = DigitPreprocess.toMnist(ink, w, h), em = DigitPreprocess.whole(ink, w, h, DigitPreprocess.Style.EMNIST).pixels;
        int rowsMn = 0, rowsEm = 0;
        for (int r = 0; r < 28; r++) { boolean a = false, b = false; for (int c = 0; c < 28; c++) { a |= mn[r * 28 + c] > 0.1f; b |= em[r * 28 + c] > 0.1f; } if (a) rowsMn++; if (b) rowsEm++; }
        check("EMNIST style fills a 24 px box, MNIST style 20 px (" + rowsEm + " vs " + rowsMn + " rows)", rowsEm == 24 && rowsMn == 20);

        Random rnd = new Random(4);
        Sequential m = new Sequential().add(new Dense(784, 12, rnd)).add(new Activations.ReLU()).add(new Dense(12, 26, rnd));
        DigitRecognizer rec = new DigitRecognizer(m);
        check("recognizer: a 26-output model is a letters model (A-Z, EMNIST preprocessing)", rec.classes == 26 && rec.alphabet.equals("ABCDEFGHIJKLMNOPQRSTUVWXYZ") && rec.style() == DigitPreprocess.Style.EMNIST);
        for (int i = 0; i < 8; i++) rec.learn(em, 17); // R
        check("letters: after 8 corrections the model says R", rec.predict(em, false).label == 17);
        boolean bad = false;
        try { rec.learn(em, 26); } catch (IllegalArgumentException e) { bad = true; }
        check("letters: class 26 gives a clear error", bad);
    }

    /** A damaged or hostile model file must give an IOException with a clear message, never a crash. */
    static void hostileFiles() throws Exception {
        String[][] cases = {
            {"huge layer", "Dense 2000000000 2000000000"},
            {"not a number", "Dense abc 4"},
            {"missing size", "Dense 4"},
            {"negative size", "Dense -5 4"},
            {"unknown layer", "Teleport 3"},
        };
        for (String[] c : cases) {
            File f = File.createTempFile("hostile", ".bin");
            try (DataOutputStream o = new DataOutputStream(new FileOutputStream(f))) { o.writeInt(0x4D444C33); o.writeInt(1); o.writeUTF(c[1]); }
            String got = "no exception";
            try { ModelIO.loadModelBinary(f.getPath(), new Random(1)); } catch (IOException e) { got = "IOException"; } catch (Throwable t) { got = t.getClass().getSimpleName(); }
            check("damaged file (" + c[0] + ") gives a clear IOException, got " + got, got.equals("IOException"));
            f.delete();
        }
        File f = File.createTempFile("hostile", ".bin");
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(f))) { o.writeInt(0x4D444C33); o.writeInt(2000000000); }
        String got = "no exception";
        try { ModelIO.loadModelBinary(f.getPath(), new Random(1)); } catch (IOException e) { got = "IOException"; } catch (Throwable t) { got = t.getClass().getSimpleName(); }
        check("damaged file (2 billion layers) gives a clear IOException, got " + got, got.equals("IOException"));
        f.delete();
    }

    static void alnum() {
        byte[] dpx = new byte[2 * 784], lpx = new byte[2 * 784];
        for (int y = 4; y < 24; y++) for (int x = 12; x < 16; x++) { dpx[y * 28 + x] = (byte) 255; dpx[784 + y * 28 + x] = (byte) 255; } // two small "1" bars, 20 px tall
        lpx[0] = 1; lpx[784] = 1;
        ByteSource digits = new ByteSource(dpx, 784, new byte[]{3, 7}, 10), letters = new ByteSource(lpx, 784, new byte[]{0, 25}, 26);
        ByteSource both = Alnum.combine(digits, letters);
        int[] idx = {0, 1, 2, 3};
        Tensor y = both.targets(idx, 0, 4), x = both.features(idx, 0, 4);
        boolean[] seen = new boolean[36];
        for (int i = 0; i < 4; i++) for (int c = 0; c < 36; c++) if (y.get(i, c) == 1) seen[c] = true;
        check("alnum: 4 pictures, 36 columns, classes 3 and 7 (digits), 10 (A) and 35 (Z) all present",
            both.size() == 4 && y.cols == 36 && seen[3] && seen[7] && seen[10] && seen[35]);
        int tall = 0;
        for (int i = 0; i < 4; i++) { if (y.get(i, 3) + y.get(i, 7) != 1) continue; int rows = 0; for (int r = 0; r < 28; r++) { boolean a = false; for (int c = 0; c < 28; c++) a |= x.get(i, r * 28 + c) > 0.1; if (a) rows++; } if (rows == 24) tall++; }
        check("alnum: the digits are re-made in the 24 px EMNIST style (20 px bars became 24 px)", tall == 2);
        Random rnd = new Random(2);
        DigitRecognizer rec = new DigitRecognizer(new Sequential().add(new Dense(784, 8, rnd)).add(new Activations.ReLU()).add(new Dense(8, 36, rnd)));
        check("recognizer: a 36-output model reads 0-9 then A-Z with EMNIST preprocessing", rec.alphabet.equals("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") && rec.style() == DigitPreprocess.Style.EMNIST);
    }

    // ---- drawing helpers for the segmentation tests: a tall bar is a "1", a ring is a "0"
    static void bar(float[] ink, int w, int x0, int y0, int bw, int bh, double slant) {
        for (int y = 0; y < bh; y++) { int sh = (int) Math.round(slant * (bh - y)); for (int x = 0; x < bw; x++) ink[(y0 + y) * w + x0 + x + sh] = 1f; }
    }

    static void ring(float[] ink, int w, int x0, int y0, int rw, int rh, double thick) {
        double cx = x0 + rw / 2.0, cy = y0 + rh / 2.0, rx = rw / 2.0, ry = rh / 2.0;
        for (int y = y0; y < y0 + rh; y++) for (int x = x0; x < x0 + rw; x++) {
            double d = Math.pow((x + 0.5 - cx) / rx, 2) + Math.pow((y + 0.5 - cy) / ry, 2);
            if (d <= 1.0 && d >= Math.pow(1 - thick, 2)) ink[y * w + x] = 1f;
        }
    }

    /** A stand-in for the model: how much a picture looks like a bar ("1") or a ring ("0"), 0..1 (cosine similarity). */
    static DigitPreprocess.Scorer barOrRing(int w, int h) {
        float[] b = new float[w * h], r = new float[w * h];
        bar(b, w, 10, 10, 10, 100, 0); ring(r, w, 10, 10, 90, 100, 0.2);
        final float[] tb = DigitPreprocess.whole(b, w, h, DigitPreprocess.Style.EMNIST).pixels, tr = DigitPreprocess.whole(r, w, h, DigitPreprocess.Style.EMNIST).pixels;
        return px -> Math.max(cos(px, tb), cos(px, tr));
    }

    static double cos(float[] a, float[] b) {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return na == 0 || nb == 0 ? 0 : d / Math.sqrt(na * nb);
    }

    static void segmentation() {
        final int w = 720, h = 320;
        DigitPreprocess.Style E = DigitPreprocess.Style.EMNIST;
        // "1" and "0" only 4 px apart: the old rule (gap of 18 px) read them as ONE shape
        float[] a = new float[w * h];
        bar(a, w, 100, 100, 12, 150, 0); ring(a, w, 116, 100, 110, 150, 0.2);
        List<DigitPreprocess.Piece> p = DigitPreprocess.segment(a, w, h, E);
        check("segment: '1' and '0' 4 px apart are two characters, left to right (" + p.size() + ")", p.size() == 2 && p.get(0).x < p.get(1).x && p.get(0).w == 12);
        // "110"
        float[] b = new float[w * h];
        bar(b, w, 100, 100, 12, 150, 0); bar(b, w, 125, 100, 12, 150, 0); ring(b, w, 142, 100, 110, 150, 0.2);
        check("segment: '110' with 4 to 5 px gaps is three characters (" + DigitPreprocess.segment(b, w, h, E).size() + ")", DigitPreprocess.segment(b, w, h, E).size() == 3);
        // slanted "1" whose columns overlap the ring's columns but whose ink does not touch it: each piece keeps ONLY its own ink
        float[] c = new float[w * h];
        bar(c, w, 100, 100, 12, 150, 0.0); ring(c, w, 124, 100, 110, 150, 0.2);
        float[] lone = new float[w * h]; bar(lone, w, 100, 100, 12, 150, 0.0);
        List<DigitPreprocess.Piece> pc = DigitPreprocess.segment(c, w, h, E);
        check("segment: each character gets only its own ink (left piece equals a lone bar)", pc.size() == 2 && java.util.Arrays.equals(pc.get(0).pixels, DigitPreprocess.whole(lone, w, h, E).pixels));
        float[] sl = new float[w * h];
        ring(sl, w, 160, 100, 110, 150, 0.2); bar(sl, w, 120, 100, 10, 150, 0.12); // slanted bar leans toward the ring: its columns reach x = 148 .. 158, inside the ring's box? no: ring starts at 160
        List<DigitPreprocess.Piece> ps = DigitPreprocess.segment(sl, w, h, E);
        check("segment: a slanted '1' leaning toward a '0' is still separate (" + ps.size() + ")", ps.size() == 2 && ps.get(0).x < ps.get(1).x);
        // dot on top of a bar (like the dot of an i) joins it
        float[] d = new float[w * h];
        bar(d, w, 100, 140, 12, 110, 0); bar(d, w, 100, 100, 12, 14, 0);
        check("segment: a dot above a stem is ONE character", DigitPreprocess.segment(d, w, h, E).size() == 1);
        // a broken "0": two arcs drawn as 2 strokes (gap 3 px); the model-check joins them, without a model they stay 2
        float[] arcs = new float[w * h];
        ring(arcs, w, 100, 100, 110, 150, 0.2);
        for (int y = 100; y < 250; y++) for (int x = 150; x < 154; x++) arcs[y * w + x] = 0; // cut through the middle: left arc | right arc
        DigitPreprocess.Scorer ringOnly = barOrRing(w, h);
        int without = DigitPreprocess.segment(arcs, w, h, E).size();
        check("segment: a ring cut in two arcs gives 2 blobs without the model (" + without + ")", without == 2);
        // touching "1" + "0" (the bar's edge kisses the ring): cut with the model's help
        float[] t = new float[w * h];
        bar(t, w, 100, 100, 12, 150, 0); ring(t, w, 112, 100, 150, 150, 0.2);
        List<DigitPreprocess.Piece> pt = DigitPreprocess.segment(t, w, h, E, ringOnly);
        check("segment: touching '1' + '0' are cut into 2, left to right, with a note (" + pt.size() + ")",
            pt.size() == 2 && pt.get(0).x < pt.get(1).x && pt.get(0).note != null && pt.get(0).w <= 14);
        check("segment: touching '1' + '0' stay one without a model (it cannot know)", DigitPreprocess.segment(t, w, h, E).size() == 1);
        // a single ring and a single "8" (two touching rings) must NOT be cut
        float[] z = new float[w * h]; ring(z, w, 100, 100, 150, 150, 0.2);
        check("segment: a lone '0' is not cut", DigitPreprocess.segment(z, w, h, E, ringOnly).size() == 1);
        float[] e8 = new float[w * h]; ring(e8, w, 100, 100, 80, 80, 0.25); ring(e8, w, 98, 172, 84, 80, 0.25);
        check("segment: a lone '8' is not cut", DigitPreprocess.segment(e8, w, h, E, ringOnly).size() == 1);
        // many characters keep their left-to-right order
        float[] many = new float[w * h];
        int x = 30; for (int i = 0; i < 6; i++) { ring(many, w, x, 100, 80, 120, 0.25); x += 95; }
        List<DigitPreprocess.Piece> pm = DigitPreprocess.segment(many, w, h, E, ringOnly);
        boolean ordered = pm.size() == 6;
        for (int i = 1; i < pm.size(); i++) ordered &= pm.get(i).x > pm.get(i - 1).x;
        check("segment: 6 characters, 15 px apart, come out in order (" + pm.size() + ")", ordered);
        // the same character drawn small/large and in different places gives (nearly) the same 28 x 28 input
        float[] big = new float[w * h], small = new float[w * h];
        ring(big, w, 300, 40, 160, 240, 0.2); ring(small, w, 50, 200, 40, 60, 0.2);
        double sim = cos(DigitPreprocess.whole(big, w, h, E).pixels, DigitPreprocess.whole(small, w, h, E).pixels);
        check("preprocess: the same shape drawn big and small gives (nearly) the same input, cosine " + String.format("%.3f", sim), sim > 0.95);
    }

    /** Class numbers -> characters, and which model the server picks. */
    static void modelChoice() throws Exception {
        check("alphabet: class 0..9 are '0'..'9', 10 is 'A', 35 is 'Z'", Alnum.CHARS.charAt(0) == '0' && Alnum.CHARS.charAt(9) == '9' && Alnum.CHARS.charAt(10) == 'A' && Alnum.CHARS.charAt(35) == 'Z' && Alnum.CHARS.length() == 36);
        check("alphabet: EMNIST letters class 0 is 'A', 25 is 'Z'", EmnistLoader.LETTERS.charAt(0) == 'A' && EmnistLoader.LETTERS.charAt(25) == 'Z' && EmnistLoader.LETTERS.length() == 26);
        Random rnd = new Random(9);
        HandwritingServer.slots.clear();
        int[] sizes = {10, 26, 36};
        for (int n : sizes) {
            File f = File.createTempFile("m" + n, ".bin");
            ModelIO.saveBinary(new Sequential().add(new Dense(784, 6, rnd)).add(new Activations.ReLU()).add(new Dense(6, n, rnd)), f.getPath());
            HandwritingServer.addSlot(f.getPath());
            f.delete();
        }
        check("server: models are filed by size: 36 = all, 26 = letters, 10 = digits (" + HandwritingServer.slots.keySet() + ")",
            HandwritingServer.slots.keySet().equals(new java.util.LinkedHashSet<>(java.util.Arrays.asList("digits", "letters", "all"))));
        HandwritingServer.Slot all = HandwritingServer.slots.get("all"), dig = HandwritingServer.slots.get("digits");
        float[] px = new float[784];
        double[] pr = new double[36]; pr[10] = 1;
        DigitPreprocess.Piece piece = DigitPreprocess.whole(blobImage(), 100, 100, DigitPreprocess.Style.EMNIST);
        String j36 = HandwritingServer.digitJson(all, piece, resultFor(pr));
        check("server: class 10 of the 36-class model is shown as 'A'", j36.contains("\"ch\":\"A\"") && j36.contains("\"label\":10"));
        pr = new double[36]; pr[35] = 1;
        check("server: class 35 is shown as 'Z'", HandwritingServer.digitJson(all, piece, resultFor(pr)).contains("\"ch\":\"Z\""));
        double[] p10 = new double[10]; p10[7] = 1;
        check("server: class 7 of the digits model is shown as '7'", HandwritingServer.digitJson(dig, piece, resultFor(p10)).contains("\"ch\":\"7\""));
        HandwritingServer.slots.clear();
    }

    static float[] blobImage() { float[] f = new float[100 * 100]; for (int y = 20; y < 80; y++) for (int x = 40; x < 60; x++) f[y * 100 + x] = 1f; return f; }

    static DigitRecognizer.Result resultFor(double[] probs) {
        try { java.lang.reflect.Constructor<DigitRecognizer.Result> c = DigitRecognizer.Result.class.getDeclaredConstructor(double[].class); c.setAccessible(true); return c.newInstance((Object) probs); }
        catch (Exception e) { throw new RuntimeException(e); }
    }

    // ---------------------------------------------------------------- serving (DJL-style model server)
    static File tinyModelDir() throws Exception {
        File dir = java.nio.file.Files.createTempDirectory("minidl-store").toFile();
        Random rnd = new Random(5);
        ModelIO.saveBinary(new Sequential().add(new Dense(4, 8, rnd)).add(new Activations.ReLU()).add(new Dense(8, 3, rnd)), new File(dir, "tiny.bin").getPath());
        ModelIO.saveBinary(new Sequential().add(new Dense(2000, 2000, rnd)).add(new Activations.ReLU()).add(new Dense(2000, 2, rnd)), new File(dir, "slow.bin").getPath());
        return dir;
    }

    static float[] direct(File model, float[] row) throws Exception {
        Sequential m = ModelIO.loadModelBinary(model.getPath(), new Random(1));
        Tensor x = new Tensor(1, row.length);
        System.arraycopy(row, 0, x.data, 0, row.length);
        Tensor o = m.predict(x);
        float[] r = new float[o.cols];
        for (int c = 0; c < o.cols; c++) r[c] = (float) o.get(0, c);
        return r;
    }

    static void serving() throws Exception {
        File dir = tinyModelDir(), tiny = new File(dir, "tiny.bin");
        Random rnd = new Random(8);

        // 1. dynamic batching: 8 single-row requests that arrive together are answered by (about) ONE forward pass, each caller gets ITS row
        ModelPool.Config c = new ModelPool.Config();
        c.name = "tiny"; c.file = tiny.getPath(); c.batchSize = 8; c.maxBatchDelayMs = 300;
        ModelPool pool = new ModelPool(c);
        List<float[]> in = new ArrayList<>();
        List<java.util.concurrent.CompletableFuture<float[][]>> fs = new ArrayList<>();
        for (int i = 0; i < 8; i++) { float[] row = new float[4]; for (int k = 0; k < 4; k++) row[k] = (float) rnd.nextGaussian(); in.add(row); fs.add(pool.submit(new float[][]{row})); }
        boolean same = true;
        for (int i = 0; i < 8; i++) { float[] got = fs.get(i).get(10, java.util.concurrent.TimeUnit.SECONDS)[0], want = direct(tiny, in.get(i)); for (int k = 0; k < 3; k++) same &= Math.abs(got[k] - want[k]) < 1e-5; }
        check("serving: 8 requests are answered in " + pool.stats.batches.get() + " forward pass(es), each caller gets its own correct row", pool.stats.batches.get() <= 2 && same && pool.stats.batchRows.get() == 8);

        // 2. one bad request fails alone: wrong width is refused, the good ones in the same moment are fine
        java.util.concurrent.CompletableFuture<float[][]> good1 = pool.submit(new float[][]{in.get(0)}), bad = pool.submit(new float[][]{new float[5]}), good2 = pool.submit(new float[][]{in.get(1)});
        boolean badFailed = false;
        try { bad.get(10, java.util.concurrent.TimeUnit.SECONDS); } catch (java.util.concurrent.ExecutionException e) { badFailed = true; }
        check("serving: a request with the wrong width fails alone, the others still succeed", badFailed && good1.get(10, java.util.concurrent.TimeUnit.SECONDS).length == 1 && good2.get(10, java.util.concurrent.TimeUnit.SECONDS).length == 1);

        // 3. scaling
        pool.scale(3, 3);
        boolean up = pool.workers() == 3;
        pool.scale(1, 1);
        check("serving: scale up to 3 workers and back down to 1", up && pool.workers() == 1);
        pool.close();

        // 4. a full queue refuses at once (no waiting, no memory growth)
        ModelPool.Config sc = new ModelPool.Config();
        sc.name = "slow"; sc.file = new File(dir, "slow.bin").getPath(); sc.batchSize = 1; sc.maxBatchDelayMs = 0; sc.queueSize = 1;
        ModelPool slow = new ModelPool(sc);
        List<java.util.concurrent.CompletableFuture<float[][]>> many = new ArrayList<>();
        for (int i = 0; i < 100; i++) many.add(slow.submit(new float[][]{new float[2000]}));
        int ok = 0, refused = 0;
        for (java.util.concurrent.CompletableFuture<float[][]> f : many) {
            try { f.get(60, java.util.concurrent.TimeUnit.SECONDS); ok++; }
            catch (java.util.concurrent.ExecutionException e) { if (e.getCause() instanceof ModelPool.CapacityException) refused++; }
        }
        check("serving: queue of 1 with a slow model: " + ok + " answered, " + refused + " refused at once, none lost", refused > 0 && ok >= 1 && ok + refused == 100 && slow.stats.rejected.get() == refused);
        slow.close();

        // 5. request parsing
        check("serving: JSON inputs: rows, one row, bare array", ModelServer.parseRows("{\"inputs\":[[1,2],[3,4]]}").length == 2 && ModelServer.parseRows("{\"input\":[1,2,3]}")[0].length == 3 && ModelServer.parseRows("[[0.5,-1e2]]")[0][1] == -100f);
        int bad2 = 0;
        for (String b : new String[]{"{\"inputs\":[[1,2],[3]]}", "{\"inputs\":[]}", "{\"inputs\":[[1,\"x\"]]}", "{\"foo\":1}", "[1,2", "not json", "{\"inputs\":[[1,2]]} trailing"}) {
            try { ModelServer.parseRows(b); } catch (IllegalArgumentException e) { bad2++; }
        }
        check("serving: 7 kinds of broken input are refused with a clear error (" + bad2 + "/7)", bad2 == 7);

        // 6. the HTTP API, end to end
        ModelServer srv = new ModelServer(dir);
        srv.start(0);
        try {
            String base = "http://127.0.0.1:" + srv.port();
            java.net.http.HttpClient hc = java.net.http.HttpClient.newHttpClient();
            java.util.function.BiFunction<String, String[], String[]> call = (method, a) -> { // a = {path, body or null}
                try {
                    java.net.http.HttpRequest.Builder b = java.net.http.HttpRequest.newBuilder(java.net.URI.create(base + a[0]));
                    b.method(method, a[1] == null ? java.net.http.HttpRequest.BodyPublishers.noBody() : java.net.http.HttpRequest.BodyPublishers.ofString(a[1]));
                    java.net.http.HttpResponse<String> r = hc.send(b.build(), java.net.http.HttpResponse.BodyHandlers.ofString());
                    return new String[]{String.valueOf(r.statusCode()), r.body()};
                } catch (Exception e) { throw new RuntimeException(e); }
            };
            check("http: GET /ping says Healthy", call.apply("GET", new String[]{"/ping", null})[1].contains("Healthy"));
            check("http: register a model -> 201", call.apply("POST", new String[]{"/models?url=tiny.bin&name=tiny&batch_size=4&max_batch_delay=2", null})[0].equals("201"));
            check("http: registering the same name again -> 409", call.apply("POST", new String[]{"/models?url=tiny.bin&name=tiny", null})[0].equals("409"));
            int refusedPaths = 0;
            for (String u : new String[]{"../tiny.bin", "%2Fetc%2Fpasswd", "..%2Ftiny.bin", "sub%5Ctiny.bin", "missing.bin", ".hidden.bin", "tiny.txt"})
                if (call.apply("POST", new String[]{"/models?url=" + u + "&name=evil", null})[0].equals("400")) refusedPaths++;
            check("http: paths outside the model store, missing files and wrong types are refused with 400 (" + refusedPaths + "/7)", refusedPaths == 7);
            check("http: a bad model name is refused", call.apply("POST", new String[]{"/models?url=tiny.bin&name=a%2Fb", null})[0].equals("400"));
            float[] r0 = {0.5f, -1f, 2f, 0.25f}, r1 = {1f, 1f, -1f, 0f};
            String[] resp = call.apply("POST", new String[]{"/predictions/tiny", "{\"inputs\":[[0.5,-1,2,0.25],[1,1,-1,0]]}"});
            java.util.Map<?, ?> js = (java.util.Map<?, ?>) MiniJson.parse(resp[1]);
            List<?> outs = (List<?>) js.get("outputs");
            boolean eq = resp[0].equals("200");
            float[][] want = {direct(tiny, r0), direct(tiny, r1)};
            for (int r = 0; eq && r < 2; r++) for (int k = 0; k < 3; k++) eq &= Math.abs(((Number) ((List<?>) outs.get(r)).get(k)).doubleValue() - want[r][k]) < 1e-5;
            check("http: POST /predictions/tiny returns the same numbers as the model run directly, plus argmax", eq && ((List<?>) js.get("argmax")).size() == 2);
            check("http: wrong width -> 400 (not 500)", call.apply("POST", new String[]{"/predictions/tiny", "{\"input\":[1,2,3]}"})[0].equals("400"));
            check("http: broken JSON -> 400", call.apply("POST", new String[]{"/predictions/tiny", "{oops"})[0].equals("400"));
            check("http: GET on predictions -> 405, unknown model -> 404", call.apply("GET", new String[]{"/predictions/tiny", null})[0].equals("405") && call.apply("POST", new String[]{"/predictions/nope", "{\"input\":[1]}"})[0].equals("404"));
            // 30 requests at once from 30 threads, every answer must be the right one
            java.util.concurrent.atomic.AtomicInteger good = new java.util.concurrent.atomic.AtomicInteger();
            List<Thread> ts = new ArrayList<>();
            for (int t = 0; t < 30; t++) {
                final int seed = t;
                Thread th = new Thread(() -> {
                    try {
                        float[] row = {seed, -seed / 2f, 1f, 0.1f * seed};
                        String[] rs = call.apply("POST", new String[]{"/predictions/tiny", "{\"input\":[" + row[0] + "," + row[1] + "," + row[2] + "," + row[3] + "]}"});
                        float[] w = direct(tiny, row);
                        List<?> o = (List<?>) ((List<?>) ((java.util.Map<?, ?>) MiniJson.parse(rs[1])).get("outputs")).get(0);
                        boolean okk = rs[0].equals("200");
                        for (int k = 0; k < 3; k++) okk &= Math.abs(((Number) o.get(k)).doubleValue() - w[k]) < 1e-4;
                        if (okk) good.incrementAndGet();
                    } catch (Exception e) { /* counted as not good */ }
                });
                ts.add(th); th.start();
            }
            for (Thread th : ts) th.join();
            check("http: 30 parallel requests, all answered correctly (" + good.get() + "/30)", good.get() == 30);
            String info = call.apply("GET", new String[]{"/models/tiny", null})[1];
            java.util.Map<?, ?> im = (java.util.Map<?, ?>) MiniJson.parse(info);
            check("http: /models/tiny reports the request count (1 + 1 wrong width + 30 parallel = 32) and latency percentiles", ((Number) im.get("requests")).intValue() == 32 && im.get("latencyMs") instanceof java.util.Map);
            String met = call.apply("GET", new String[]{"/metrics", null})[1];
            check("http: /metrics is Prometheus text with counters, gauges and a latency histogram", met.contains("minidl_requests_total{model=\"tiny\"}") && met.contains("# TYPE minidl_request_latency_seconds histogram") && met.contains("le=\"+Inf\"") && met.contains("minidl_workers{model=\"tiny\"} 1"));
            check("http: scale with PUT", call.apply("PUT", new String[]{"/models/tiny?min_workers=2&max_workers=2", null})[0].equals("200") && srv.pool("tiny").workers() == 2);
            check("http: list shows the model", call.apply("GET", new String[]{"/models", null})[1].contains("\"modelName\":\"tiny\""));
            check("http: DELETE unloads it; then predict -> 404", call.apply("DELETE", new String[]{"/models/tiny", null})[0].equals("200") && call.apply("POST", new String[]{"/predictions/tiny", "{\"input\":[1,2,3,4]}"})[0].equals("404"));
        } finally { srv.stop(); }
    }

    // ---------------------------------------------------------------- phase 4: BatchNorm2D, LSTM, ResNet, BPE tokenizer
    static void phase4() throws Exception {
        Random r = new Random(21);
        Tensor x = Tensor.randn(4, 16, 1.0, r), y = Tensor.randn(4, 2, 1.0, r);

        Sequential bn = new Sequential().add(new Conv2D(1, 4, 4, 2, 3, 1, 1, r)).add(new BatchNorm2D(2, 4, 4)).add(new Activations.Tanh()).add(new Dense(32, 2, r));
        check("gradcheck BatchNorm2D (training mode, per-channel statistics)", GradCheck.maxRelativeError(bn, new Losses.MSE(), x, y) < GC_TOL_DEEP);
        BatchNorm2D b2 = new BatchNorm2D(3, 2, 2);
        Tensor img = Tensor.randn(5, 12, 3.0, r);
        for (int i = 0; i < img.data.length; i++) img.data[i] += (i % 12) / 4 * 10f; // different mean per channel
        Tensor o2 = b2.forward(img);
        double worst = 0;
        for (int c = 0; c < 3; c++) {
            double m = 0, v = 0; int cnt = 0;
            for (int i = 0; i < 5; i++) for (int p = 0; p < 4; p++) { m += o2.data[i * 12 + c * 4 + p]; cnt++; }
            m /= cnt;
            for (int i = 0; i < 5; i++) for (int p = 0; p < 4; p++) { double d = o2.data[i * 12 + c * 4 + p] - m; v += d * d; }
            worst = Math.max(worst, Math.max(Math.abs(m), Math.abs(Math.sqrt(v / cnt) - 1)));
        }
        check("BatchNorm2D: every channel ends with mean 0 and std 1 over (batch, y, x), worst deviation " + String.format("%.4f", worst), worst < 1e-3);

        // LSTM
        Sequential l1 = new Sequential().add(new LSTM(3, 2, 4, false, r)).add(new Dense(4, 2, r));
        Tensor xs = Tensor.randn(4, 6, 1.0, r);
        check("gradcheck LSTM (last hidden state, BPTT over 3 steps)", GradCheck.maxRelativeError(l1, new Losses.MSE(), xs, y) < GC_TOL_DEEP);
        Sequential l2 = new Sequential().add(new LSTM(3, 2, 3, true, r)).add(new Dense(9, 2, r));
        check("gradcheck LSTM (return every step)", GradCheck.maxRelativeError(l2, new Losses.MSE(), xs, y) < GC_TOL_DEEP);
        // a task that needs memory: is the FIRST number of a 6-step sequence positive? (the last steps are noise)
        int N = 600, T = 6;
        Tensor tx = Tensor.randn(N, T, 1.0, r), ty = new Tensor(N, 2);
        for (int i = 0; i < N; i++) ty.data[i * 2 + (tx.data[i * T] > 0 ? 1 : 0)] = 1f;
        Sequential mem = new Sequential().add(new LSTM(T, 1, 12, false, new Random(3))).add(new Dense(12, 2, new Random(4)));
        mem.fit(tx, ty, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.02), 80, 32, 0, 0);
        Tensor vx = Tensor.randn(300, T, 1.0, new Random(99)), vy = new Tensor(300, 2);
        for (int i = 0; i < 300; i++) vy.data[i * 2 + (vx.data[i * T] > 0 ? 1 : 0)] = 1f;
        double acc = mem.accuracy(vx, vy);
        check("LSTM learns to remember: sign of the first of 6 numbers, accuracy on new data " + String.format("%.1f%%", acc), acc > 90);

        // ResNet. The gradient check uses tanh (smooth): ReLU has kinks that fool finite differences (not a bug in backward).
        Sequential smooth = new Sequential().add(new Conv2D(1, 8, 8, 2, 3, 1, 1, r)).add(new BatchNorm2D(2, 8, 8)).add(new Activations.Tanh())
            .add(new Residual(new Conv2D(2, 8, 8, 2, 3, 1, 1, r), new BatchNorm2D(2, 8, 8), new Activations.Tanh(), new Conv2D(2, 8, 8, 2, 3, 1, 1, r), new BatchNorm2D(2, 8, 8))).add(new Activations.Tanh())
            .add(new Conv2D(2, 8, 8, 4, 3, 2, 1, r)).add(new BatchNorm2D(4, 4, 4)).add(new Activations.Tanh()).add(new AvgPool2D(4, 4, 4, 4)).add(new Dense(4, 3, r));
        Tensor rx = Tensor.randn(6, 64, 1.0, r), ry = Tensor.randn(6, 3, 1.0, r);
        check("gradcheck ResNet-style stack (Conv, BatchNorm2D, Residual, stride 2, AvgPool, Dense)", GradCheck.maxRelativeError(smooth, new Losses.MSE(), rx, ry) < GC_TOL_DEEP);
        Sequential rn = ResNet.build(1, 8, 8, 4, 1, 3, new Random(31));
        Tensor tx8 = Tensor.randn(16, 64, 1.0, r), ty8 = Tensor.randn(16, 3, 1.0, r);
        double rl0 = new Losses.MSE().forward(rn.predict(tx8), ty8);
        rn.fit(tx8, ty8, new Losses.MSE(), new Optimizers.Adam(0.01), 80, 8, 0, 0);
        rn.setTraining(false);
        double rl1 = new Losses.MSE().forward(rn.predict(tx8), ty8);
        check("ResNet.build (ReLU) trains: loss on 16 examples " + String.format("%.3f -> %.3f", rl0, rl1), rl1 < 0.5 * rl0);
        File f = File.createTempFile("resnet", ".bin");
        Tensor rx2 = Tensor.randn(3, 64, 1.0, r);
        Tensor before = rn.predict(rx2);
        ModelIO.saveBinary(rn, f.getPath());
        Tensor after = ModelIO.loadModelBinary(f.getPath(), new Random(1)).predict(rx2);
        double diff = 0;
        for (int i = 0; i < before.data.length; i++) diff = Math.max(diff, Math.abs(before.data[i] - after.data[i]));
        check("ResNet save + load gives identical predictions (BatchNorm2D running stats kept), max diff " + diff, diff < 1e-6);
        f.delete();
        boolean refused = false;
        try { ResNet.build(1, 10, 10, 2, 1, 3, r); } catch (IllegalArgumentException e) { refused = true; }
        check("ResNet refuses a picture size that is not divisible by 4", refused);

        // BPE tokenizer
        String text = "the cat sat on the mat. the cat ate the rat. a cat and a rat sat on the mat.\nthe end of the story.";
        BpeTokenizer tk = BpeTokenizer.train(text, 60);
        int[] ids = tk.encode(text);
        check("BPE: decode(encode(text)) gives the text back exactly", tk.decode(ids).equals(text));
        check("BPE: merges make the text shorter than one token per character (" + ids.length + " tokens for " + text.length() + " characters), vocabulary " + tk.vocabSize(), ids.length < text.length() * 0.6 && tk.vocabSize() <= 60);
        check("BPE: a frequent word is ONE token", tk.encode(" the").length == 1 && tk.encode(" cat").length == 1);
        int[] unseen = tk.encode("the Zebra");
        check("BPE: an unseen character becomes the unknown token, the rest still works", java.util.Arrays.stream(unseen).anyMatch(v -> v == BpeTokenizer.UNKNOWN) && tk.decode(unseen).startsWith("the "));
        File tf = File.createTempFile("bpe", ".txt");
        tk.save(tf.getPath());
        BpeTokenizer tk2 = BpeTokenizer.load(tf.getPath());
        check("BPE: save + load gives the same tokens", java.util.Arrays.equals(tk2.encode(text), ids) && tk2.vocabSize() == tk.vocabSize());
        tf.delete();
        boolean badId = false;
        try { tk.decode(new int[]{9999}); } catch (IllegalArgumentException e) { badId = true; }
        check("BPE: an id outside the vocabulary gives a clear error", badId);
    }

    // ---------------------------------------------------------------- Java-only tools: SafeTensors, Plot, TokenSource, web server safety
    static void javaOnly() throws Exception {
        // 1. a safetensors file built BY HAND from the format description (header length, JSON header, little-endian floats)
        String hdr = "{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"n\":{\"dtype\":\"I64\",\"shape\":[],\"data_offsets\":[16,24]}}";
        while (hdr.length() % 8 != 0) hdr += " ";
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(8 + hdr.length() + 24).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        bb.putLong(hdr.length()).put(hdr.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        bb.putFloat(1f).putFloat(2f).putFloat(3f).putFloat(4f).putLong(7);
        File hand = File.createTempFile("hand", ".safetensors");
        java.nio.file.Files.write(hand.toPath(), bb.array());
        java.util.Map<String, SafeTensors.Entry> got = SafeTensors.read(hand.getPath());
        float[] a = got.get("a").floats();
        check("safetensors: a file written by hand from the format description is read correctly (shape, values, I64 kept, metadata ignored)",
            got.size() == 2 && java.util.Arrays.equals(got.get("a").shape, new long[]{2, 2}) && a[0] == 1f && a[3] == 4f && got.get("n").raw.length == 8);
        byte[] bad = bb.array().clone();
        bad[8 + hdr.indexOf("[0,16]") + 4] = '9'; // offsets now point outside the file
        java.nio.file.Files.write(hand.toPath(), bad);
        String why = "";
        try { SafeTensors.read(hand.getPath()); } catch (IOException e) { why = e.getMessage(); }
        check("safetensors: offsets that point outside the file give a clear error", why.contains("outside") || why.contains("does not fit"));
        java.nio.file.Files.write(hand.toPath(), new byte[]{1, 2, 3});
        why = "";
        try { SafeTensors.read(hand.getPath()); } catch (IOException e) { why = e.getMessage(); }
        check("safetensors: a too short file gives a clear error", why.contains("too short"));
        hand.delete();

        // 2. model <-> PyTorch layout round trip
        Random r = new Random(14);
        Sequential src = new Sequential().add(new Conv2D(1, 4, 4, 2, 3, 1, 0, r)).add(new BatchNorm2D(2, 2, 2)).add(new Activations.ReLU()).add(new Dense(8, 3, r));
        Tensor warm = Tensor.randn(6, 16, 1.0, r);
        src.predict(warm); src.forward(warm); // moves the BatchNorm running statistics away from 0/1
        src.setTraining(false);
        Tensor want = src.predict(warm);
        File st = File.createTempFile("model", ".safetensors");
        SafeTensors.exportModel(src, st.getPath());
        java.util.Map<String, SafeTensors.Entry> ex = SafeTensors.read(st.getPath());
        Tensor dw = src.layers().get(3).params().get(0).value; // MiniDL Dense weight (in=8, out=3)
        float[] exported = ex.get("3.weight").floats();
        check("safetensors export: Dense weight is (out, in) = [3, 8] and transposed, Conv2D weight is [2, 1, 3, 3], BatchNorm has the 5 PyTorch tensors",
            java.util.Arrays.equals(ex.get("3.weight").shape, new long[]{3, 8}) && exported[2 * 8 + 5] == dw.data[5 * 3 + 2] && java.util.Arrays.equals(ex.get("0.weight").shape, new long[]{2, 1, 3, 3})
            && ex.containsKey("1.running_mean") && ex.containsKey("1.running_var") && ex.containsKey("1.num_batches_tracked") && ex.get("1.num_batches_tracked").dtype.equals("I64"));
        Sequential dst = new Sequential().add(new Conv2D(1, 4, 4, 2, 3, 1, 0, new Random(99))).add(new BatchNorm2D(2, 2, 2)).add(new Activations.ReLU()).add(new Dense(8, 3, new Random(98)));
        dst.setTraining(false);
        SafeTensors.importModel(dst, st.getPath());
        Tensor have = dst.predict(warm);
        double d = 0;
        for (int i = 0; i < want.data.length; i++) d = Math.max(d, Math.abs(want.data[i] - have.data[i]));
        check("safetensors: export then import into a freshly initialised model gives identical predictions, max diff " + d, d < 1e-6);
        Sequential wrongShape = new Sequential().add(new Conv2D(1, 4, 4, 2, 3, 1, 0, r)).add(new BatchNorm2D(2, 2, 2)).add(new Activations.ReLU()).add(new Dense(8, 4, r));
        why = "";
        try { SafeTensors.importModel(wrongShape, st.getPath()); } catch (IOException e) { why = e.getMessage(); }
        check("safetensors: importing into a model with different sizes names the tensor and both shapes", why.contains("3.weight") && why.contains("shape"));
        Sequential lstm = new Sequential().add(new LSTM(3, 2, 4, false, r));
        why = "";
        try { SafeTensors.exportModel(lstm, st.getPath()); } catch (IOException e) { why = e.getMessage(); }
        check("safetensors: a layer without a mapping says so instead of writing wrong weights", why.contains("LSTM"));
        st.delete();

        // 3. Plot: SVG is valid XML with the right number of curves and points
        String log = "train 54000\nepoch    1  loss 0.42741  val 0.07573\nnoise\nepoch    2  loss 0.18905  val 0.06626\nepoch    3  loss 0.14079  val 0.05357\n";
        Plot.Series ps = Plot.parseLog("run", log);
        String svg = Plot.svg(java.util.Arrays.asList(ps, ps), "t <&> \"x\"");
        org.w3c.dom.Document doc = javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(new java.io.ByteArrayInputStream(svg.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
        check("plot: 3 epochs parsed from a log with other lines; SVG is well-formed XML with 2 charts x 2 curves x 3 points",
            ps.epoch.size() == 3 && doc.getElementsByTagName("polyline").getLength() == 4 && doc.getElementsByTagName("circle").getLength() == 12);

        // 4. TokenSource windows
        TokenSource ts = TokenSource.of(new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10, 3);
        Tensor tf = ts.features(new int[]{0, 2}, 0, 2), tt = ts.targets(new int[]{0, 2}, 0, 2);
        check("TokenSource: 10 tokens, window 3 -> 6 examples; example 2 reads 2,3,4 and must predict 3,4,5", ts.size() == 6 && tf.get(1, 0) == 2 && tf.get(1, 2) == 4 && tt.get(3 + 0, 3) == 1 && tt.get(3 + 2, 5) == 1 && tt.rows == 6);
        TokenSource view = ts.view(2, 5);
        check("TokenSource: a view starts at the right place", view.size() == 3 && view.features(new int[]{0}, 0, 1).get(0, 0) == 2);

        // 5. web server: which websites may call it, and the safety header
        File dir = File.createTempFile("webm", "");
        dir.delete(); dir.mkdir();
        File tiny = new File(dir, "mnist-cnn.bin");
        ModelIO.saveBinary(new Sequential().add(new Dense(784, 6, r)).add(new Activations.ReLU()).add(new Dense(6, 10, r)), tiny.getPath());
        HandwritingServer.slots.clear();
        HandwritingServer.addSlot(tiny.getPath());
        HandwritingServer.defaultKey = "digits";
        com.sun.net.httpserver.HttpServer web = HandwritingServer.startHttp(0);
        try {
            String base = "http://127.0.0.1:" + web.getAddress().getPort();
            java.net.http.HttpClient hc = java.net.http.HttpClient.newHttpClient();
            final String ok = "https://manishn1877.github.io", evil = "https://evil.example";
            java.util.function.Function<String[], java.net.http.HttpResponse<String>> call = q -> { // method, path, origin|null, withHeader(1/0), body|null
                try {
                    java.net.http.HttpRequest.Builder b = java.net.http.HttpRequest.newBuilder(java.net.URI.create(base + q[1]));
                    if (q[2] != null) b.header("Origin", q[2]);
                    if (q[3].equals("1")) b.header("X-MiniDL", "1");
                    b.method(q[0], q[4] == null ? java.net.http.HttpRequest.BodyPublishers.noBody() : java.net.http.HttpRequest.BodyPublishers.ofByteArray(new byte[16 * 16]));
                    return hc.send(b.build(), java.net.http.HttpResponse.BodyHandlers.ofString());
                } catch (Exception e) { throw new RuntimeException(e); }
            };
            java.net.http.HttpResponse<String> g1 = call.apply(new String[]{"GET", "/api/info", ok, "0", null});
            check("web: an allowed website (the GitHub Pages one) gets the cross-origin headers, including the private-network one",
                g1.statusCode() == 200 && ok.equals(g1.headers().firstValue("Access-Control-Allow-Origin").orElse("")) && g1.headers().firstValue("Access-Control-Allow-Private-Network").orElse("").equals("true"));
            java.net.http.HttpResponse<String> g2 = call.apply(new String[]{"GET", "/api/info", evil, "0", null});
            check("web: any other website gets NO cross-origin permission", g2.statusCode() == 200 && g2.headers().firstValue("Access-Control-Allow-Origin").isEmpty());
            check("web: preflight (OPTIONS) is answered 204 for the allowed website and 403 for others",
                call.apply(new String[]{"OPTIONS", "/api/predict", ok, "0", null}).statusCode() == 204 && call.apply(new String[]{"OPTIONS", "/api/predict", evil, "0", null}).statusCode() == 403);
            check("web: a POST without the X-MiniDL header is refused (403) for learn, save and reset too",
                call.apply(new String[]{"POST", "/api/predict?w=16&h=16", null, "0", "x"}).statusCode() == 403 && call.apply(new String[]{"POST", "/api/learn?w=16&h=16&label=1", evil, "0", "x"}).statusCode() == 403
                && call.apply(new String[]{"POST", "/api/save", evil, "0", null}).statusCode() == 403 && call.apply(new String[]{"POST", "/api/reset", evil, "0", null}).statusCode() == 403);
            java.net.http.HttpResponse<String> p = call.apply(new String[]{"POST", "/api/predict?w=16&h=16&mode=single", ok, "1", "x"});
            check("web: a POST with the header works (a blank canvas gives no digits)", p.statusCode() == 200 && p.body().contains("\"digits\":[]"));
        } finally { web.stop(0); HandwritingServer.slots.clear(); tiny.delete(); dir.delete(); }
    }
}
