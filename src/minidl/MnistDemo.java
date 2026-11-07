package minidl;

import java.io.File;
import java.util.Random;

/**
 * Handwritten digits (MNIST).
 *   java -cp out minidl.MnistDemo  <folder>  [epochs=10]  [mlp|cnn=cnn]  [trainLimit=60000]  [aug] [mmap]
 *   aug   = train on randomly shifted + rotated digits (data augmentation)
 *   mmap  = keep the image files ON DISK (memory-mapped); almost no heap memory is used for data
 * Files (either .gz or unzipped): train-images-idx3-ubyte, train-labels-idx1-ubyte,
 *                                  t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte
 *
 * Memory: images stay as ONE BYTE per pixel (47 MB for all 60,000 training digits) and become
 * floats one batch at a time. This runs fine with:  java -Xmx256m ...
 */
public class MnistDemo {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("usage: java -cp out minidl.MnistDemo <mnist-folder> [epochs=10] [mlp|cnn=cnn] [trainLimit=60000] [aug] [mmap]");
            return;
        }
        String dir = args[0];
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        String kind = args.length > 2 ? args[2] : "cnn";
        int limit = args.length > 3 ? Integer.parseInt(args[3]) : 60000;
        boolean aug = false, mmap = false;
        for (int i = 4; i < args.length; i++) { if (args[i].equals("aug")) aug = true; if (args[i].equals("mmap")) mmap = true; }

        DataSource train, val, test;
        int nVal, nTrain;
        if (mmap) {
            String tmp = new File(dir, "unzipped").getPath();
            MappedSource all = MappedSource.open(plain(dir, "train-images-idx3-ubyte", tmp), plain(dir, "train-labels-idx1-ubyte", tmp));
            MappedSource tst = MappedSource.open(plain(dir, "t10k-images-idx3-ubyte", tmp), plain(dir, "t10k-labels-idx1-ubyte", tmp));
            int n = Math.min(limit, all.size());
            nVal = n / 10; nTrain = n - nVal;
            train = all.view(0, nTrain); val = all.view(nTrain, n); test = tst;
            System.out.printf("train %d, validation %d, test %d   (images stay ON DISK, memory-mapped; the Java heap holds only the current batch)%n", nTrain, nVal, test.size());
        } else {
            ByteSource all = MnistLoader.loadSource(find(dir, "train-images-idx3-ubyte"), find(dir, "train-labels-idx1-ubyte"), limit);
            ByteSource tst = MnistLoader.loadSource(find(dir, "t10k-images-idx3-ubyte"), find(dir, "t10k-labels-idx1-ubyte"), Integer.MAX_VALUE);
            nVal = all.size() / 10; nTrain = all.size() - nVal;
            train = all.view(0, nTrain); val = all.view(nTrain, all.size()); test = tst; // views: no copying
            System.out.printf("train %d, validation %d, test %d   (image data in memory: %.1f MB as bytes; as double it would be %.0f MB)%n",
                nTrain, nVal, test.size(), (all.bytes() + tst.bytes()) / 1e6, (all.size() + tst.size()) * 784 * 8.0 / 1e6);
        }

        Random rnd = new Random(1);
        Sequential m = kind.equals("mlp") ? mlp(rnd) : cnn(rnd);
        System.out.print(m.summary());
        System.out.printf("weights memory: %.1f KB (float32)%n", m.weightBytes() / 1024.0);

        DataSource trainSrc = aug ? new AugmentedSource(train, 28, 28, 2, 10, 7) : train; // shift up to 2 px, rotate up to 10 degrees
        if (aug) System.out.println("data augmentation ON (random shift up to 2 px, rotation up to 10 degrees)");
        long t0 = System.nanoTime();
        Trainer.History h = new Trainer().epochs(epochs).batchSize(64).seed(0).printEvery(1)
            .validation(val).patience(aug ? 6 : 3)
            .schedule(Schedule.cosine(0.002, 0.0001))
            .fit(m, trainSrc, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.002));
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.printf("trained %d epochs in %.1f s (%.1f s/epoch)%n", h.epochsRun, secs, secs / h.epochsRun);

        long t1 = System.nanoTime();
        double acc = m.accuracy(test);
        double infSecs = (System.nanoTime() - t1) / 1e9;
        System.out.printf("TEST accuracy on %d unseen digits: %.2f%%   (%.2f s for all test digits)%n", test.size(), acc, infSecs);

        int[][] cm = confusion(m, test);
        System.out.print(Metrics.formatConfusion(cm));
        System.out.print(Metrics.classificationReport(cm));

        quantTable(m, test);

        String tag = kind + (aug ? "-aug" : "");
        ModelIO.saveBinary(m, "mnist-" + tag + ".bin");
        ModelIO.saveBinary(Quantizer.quantize(m), "mnist-" + tag + "-int8.bin");
        System.out.printf("files: mnist-%s.bin %d KB, mnist-%s-int8.bin %d KB%n",
            tag, new File("mnist-" + tag + ".bin").length() / 1024, tag, new File("mnist-" + tag + "-int8.bin").length() / 1024);
    }

    /** Float32 versus every shrink option: memory, accuracy and prediction time on the test digits. */
    static void quantTable(Sequential m, DataSource test) {
        m.accuracy(test); // warm-up pass, so the first row is not slowed by the JIT compiler
        System.out.printf("%n%-28s %10s %10s %10s%n", "model", "weights KB", "accuracy", "time (s)");
        row("float32", m, test);
        row("bfloat16 (Dense)", Quantizer.toBF16(m), test);
        row("int8 (Dense + Conv)", Quantizer.quantize(m), test);
        row("int4 group 32 (Dense)", Quantizer.quantizeInt4(m, 32), test);
    }

    private static void row(String name, Sequential q, DataSource test) {
        long t = System.nanoTime();
        double acc = q.accuracy(test);
        System.out.printf("%-28s %10.1f %9.2f%% %10.2f%n", name, q.weightBytes() / 1024.0, acc, (System.nanoTime() - t) / 1e9);
    }

    /** 784 -> 128 -> 64 -> 10, plain Dense layers. */
    static Sequential mlp(Random rnd) {
        return new Sequential()
            .add(new Dense(784, 128, rnd)).add(new Activations.ReLU())
            .add(new Dropout(0.1, rnd))
            .add(new Dense(128, 64, rnd)).add(new Activations.ReLU())
            .add(new Dense(64, 10, rnd));
    }

    /** Two small convolution stages, then Dense. Fewer weights than the MLP, but sees the image's shape. */
    static Sequential cnn(Random rnd) {
        return new Sequential()
            .add(new Conv2D(1, 28, 28, 8, 3, 1, 0, rnd)).add(new Activations.ReLU())   // 8 x 26 x 26
            .add(new MaxPool2D(8, 26, 26, 2))                                           // 8 x 13 x 13
            .add(new Conv2D(8, 13, 13, 16, 3, 1, 0, rnd)).add(new Activations.ReLU())  // 16 x 11 x 11
            .add(new MaxPool2D(16, 11, 11, 2))                                          // 16 x 5 x 5 = 400
            .add(new Dense(400, 64, rnd)).add(new Activations.ReLU())
            .add(new Dropout(0.2, rnd))
            .add(new Dense(64, 10, rnd));
    }

    static int[][] confusion(Sequential m, DataSource src) {
        m.setTraining(false);
        int n = src.size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        int[][] cm = new int[10][10];
        for (int s = 0; s < n; s += 1000) {
            int e = Math.min(s + 1000, n);
            Tensor out = m.forward(src.features(idx, s, e)), truth = src.targets(idx, s, e);
            for (int i = 0; i < e - s; i++) cm[truth.argmaxRow(i)][out.argmaxRow(i)]++;
        }
        return cm;
    }

    private static String find(String dir, String name) {
        File gz = new File(dir, name + ".gz");
        return (gz.exists() ? gz : new File(dir, name)).getPath();
    }

    /** Path of an UNZIPPED copy of the file (unzips once if needed). */
    private static String plain(String dir, String name, String tmpDir) throws java.io.IOException {
        String p = find(dir, name);
        return p.endsWith(".gz") ? MnistLoader.gunzip(p, tmpDir) : p;
    }
}
