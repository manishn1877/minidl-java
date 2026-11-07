package minidl;

import java.io.File;
import java.util.List;
import java.util.Random;

/**
 * 16-bit (mixed precision) training on real MNIST digits.
 *   java -Xmx256m -cp out minidl.MixedPrecisionDemo <mnist-folder> [epochs=2]
 *
 * Part 1: how much of the gradient survives in each format (one batch).
 * Part 2: train the same network from scratch in FP32, BF16, FP16 without loss scaling, FP16 with dynamic loss scaling.
 */
public class MixedPrecisionDemo {
    static Sequential net(Random rnd) { // no dropout, so gradients can be compared exactly
        return new Sequential()
            .add(new Dense(784, 128, rnd)).add(new Activations.ReLU())
            .add(new Dense(128, 64, rnd)).add(new Activations.ReLU())
            .add(new Dense(64, 10, rnd));
    }

    public static void main(String[] args) throws Exception {
        String dir = args[0];
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 2;
        ByteSource all = MnistLoader.loadSource(find(dir, "train-images-idx3-ubyte"), find(dir, "train-labels-idx1-ubyte"), 60000);
        ByteSource test = MnistLoader.loadSource(find(dir, "t10k-images-idx3-ubyte"), find(dir, "t10k-labels-idx1-ubyte"), Integer.MAX_VALUE);
        ByteSource train = all.view(0, 54000), val = all.view(54000, 60000);

        // ---------------- part 1: gradient fidelity
        Sequential m = net(new Random(1));
        new Trainer().epochs(3).batchSize(64).seed(0).fit(m, train, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001)); // a trained network has small gradients
        Losses.SoftmaxCrossEntropy ce = new Losses.SoftmaxCrossEntropy();
        Object[][] cfg = {
            {"BF16, no scaling", Precision.BF16, 1.0},
            {"FP16, no scaling", Precision.FP16, 1.0},
            {"FP16, loss scale 1024", Precision.FP16, 1024.0},
            {"FP16, loss scale 65536", Precision.FP16, 65536.0},
            {"FP16, loss scale 16777216", Precision.FP16, 16777216.0},
        };
        for (int batch : new int[]{64, 1024}) {
            int[] idx = new int[batch];
            for (int i = 0; i < batch; i++) idx[i] = i;
            Tensor xb = train.features(idx, 0, batch), yb = train.targets(idx, 0, batch);
            m.setPrecision(Precision.FP32);
            m.trainStep(xb, yb, ce, new Optimizers.SGD(0.0)); // learning rate 0: only computes gradients
            float[][] ref = copyGrads(m.params());
            long nonzero = 0, total = 0;
            for (float[] a : ref) for (float v : a) { total++; if (v != 0f) nonzero++; }
            System.out.printf("%nPart 1: gradients of one batch of %d digits (network trained 3 epochs, %,d weights, %.1f%% of FP32 gradients are non-zero)%n", batch, total, 100.0 * nonzero / total);
            System.out.printf("%-34s %14s %16s %12s%n", "format", "lost to zero", "relative error", "overflow");
            for (Object[] c : cfg) {
                m.setPrecision((Precision) c[1], LossScaler.fixed((Double) c[2]));
                m.trainStep(xb, yb, ce, new Optimizers.SGD(0.0));
                boolean over = m.lastStepOverflowed();
                float[][] g = copyGrads(m.params());
                long lost = 0; double num = 0, den = 0;
                for (int k = 0; k < g.length; k++)
                    for (int i = 0; i < g[k].length; i++) {
                        if (ref[k][i] != 0f && g[k][i] == 0f) lost++;
                        double d = g[k][i] - ref[k][i];
                        num += d * d; den += (double) ref[k][i] * ref[k][i];
                    }
                System.out.printf("%-34s %13.1f%% %16s %12s%n", c[0], 100.0 * lost / nonzero, over ? "-" : String.format("%.4f", Math.sqrt(num / den)), over ? "YES (skipped)" : "no");
            }
        }
        if (args.length > 2 && args[2].equals("part1")) return;

        // ---------------- part 2: training
        System.out.printf("%nPart 2: training the same network from scratch, %d epochs, batch 64, Adam 0.001%n", epochs);
        System.out.printf("%-34s %10s %12s %14s %12s%n", "setting", "test acc", "s / epoch", "skipped steps", "final scale");
        Object[][] runs = {
            {"FP32 (normal)", Precision.FP32, null},
            {"BF16", Precision.BF16, LossScaler.none()},
            {"FP16, no loss scaling", Precision.FP16, LossScaler.none()},
            {"FP16, dynamic loss scaling", Precision.FP16, LossScaler.dynamic()},
        };
        for (Object[] r : runs) {
            Sequential mm = net(new Random(1));
            Precision p = (Precision) r[1];
            if (p != Precision.FP32) mm.setPrecision(p, (LossScaler) r[2]);
            long t0 = System.nanoTime();
            new Trainer().epochs(epochs).batchSize(64).seed(0).fit(mm, train, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.001));
            double secs = (System.nanoTime() - t0) / 1e9 / epochs;
            LossScaler sc = mm.lossScaler();
            System.out.printf("%-34s %9.2f%% %12.1f %14d %12s%n", r[0], mm.accuracy(test), secs,
                p == Precision.FP32 ? 0 : sc.skippedSteps(), p == Precision.FP16 ? String.format("%.0f", sc.scale()) : "-");
        }
    }

    static float[][] copyGrads(List<Param> ps) {
        float[][] g = new float[ps.size()][];
        for (int i = 0; i < g.length; i++) g[i] = ps.get(i).grad.data.clone();
        return g;
    }

    private static String find(String dir, String name) {
        File gz = new File(dir, name + ".gz");
        return (gz.exists() ? gz : new File(dir, name)).getPath();
    }
}
