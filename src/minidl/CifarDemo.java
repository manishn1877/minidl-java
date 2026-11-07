package minidl;

import java.io.File;
import java.util.Random;

/**
 * CIFAR-10 colour photos (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).
 *   java -Xmx512m -cp out minidl.CifarDemo <folder with data_batch_1..5.bin and test_batch.bin> [epochs=8] [aug] [resnet] [limit=50000]
 *   aug = random shifts + flips;  resnet = BatchNorm2D and residual blocks instead of the plain stack
 *   Long runs can be done in pieces: epochs=2 total=6 from=2 resume=cifar-ckpt.bin  (2 more epochs of a 6-epoch schedule, starting at epoch 3,
 *   from the saved weights). Test accuracy is printed at the end of every piece.
 * (If you only have the PNG sheets from the npm package tfjs-cifar10, convert them first with minidl.CifarConvert.)
 */
public class CifarDemo {
    /** 3 x 32 x 32 -> three conv + pool stages -> 64 x 4 x 4 -> Dense. */
    static Sequential cnn(Random rnd) {
        return new Sequential()
            .add(new Conv2D(3, 32, 32, 16, 3, 1, 1, rnd)).add(new Activations.ReLU()).add(new MaxPool2D(16, 32, 32, 2))
            .add(new Conv2D(16, 16, 16, 32, 3, 1, 1, rnd)).add(new Activations.ReLU()).add(new MaxPool2D(32, 16, 16, 2))
            .add(new Conv2D(32, 8, 8, 64, 3, 1, 1, rnd)).add(new Activations.ReLU()).add(new MaxPool2D(64, 8, 8, 2))
            .add(new Dense(64 * 4 * 4, 128, rnd)).add(new Activations.ReLU())
            .add(new Dropout(0.3, rnd))
            .add(new Dense(128, 10, rnd));
    }

    /** Convolution + BatchNorm2D + ReLU stages with two residual blocks (skip connections). */
    static Sequential resnet(Random rnd) {
        return new Sequential()
            .add(new Conv2D(3, 32, 32, 16, 3, 1, 1, rnd)).add(new BatchNorm2D(16, 32, 32)).add(new Activations.ReLU()).add(new MaxPool2D(16, 32, 32, 2))   // 16 x 16 x 16
            .add(new Residual(new Conv2D(16, 16, 16, 16, 3, 1, 1, rnd), new BatchNorm2D(16, 16, 16), new Activations.ReLU(),
                              new Conv2D(16, 16, 16, 16, 3, 1, 1, rnd), new BatchNorm2D(16, 16, 16))).add(new Activations.ReLU())
            .add(new Conv2D(16, 16, 16, 32, 3, 1, 1, rnd)).add(new BatchNorm2D(32, 16, 16)).add(new Activations.ReLU()).add(new MaxPool2D(32, 16, 16, 2))   // 32 x 8 x 8
            .add(new Residual(new Conv2D(32, 8, 8, 32, 3, 1, 1, rnd), new BatchNorm2D(32, 8, 8), new Activations.ReLU(),
                              new Conv2D(32, 8, 8, 32, 3, 1, 1, rnd), new BatchNorm2D(32, 8, 8))).add(new Activations.ReLU())
            .add(new MaxPool2D(32, 8, 8, 2))                                                                                                              // 32 x 4 x 4 = 512
            .add(new Dense(512, 10, rnd));
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) { System.out.println("usage: java -cp out minidl.CifarDemo <cifar-folder> [epochs=8] [aug] [resnet] [limit=50000]"); return; }
        String dir = args[0];
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        boolean aug = false, res = false;
        int limit = 50000, total = 0, from = 0;
        String resume = null;
        for (int i = 2; i < args.length; i++) {
            if (args[i].equals("aug")) aug = true;
            else if (args[i].equals("resnet")) res = true;
            else if (args[i].startsWith("limit=")) limit = Integer.parseInt(args[i].substring(6));
            else if (args[i].startsWith("total=")) total = Integer.parseInt(args[i].substring(6));
            else if (args[i].startsWith("from=")) from = Integer.parseInt(args[i].substring(5));
            else if (args[i].startsWith("resume=")) resume = args[i].substring(7);
        }

        String[] files = new String[5];
        for (int i = 0; i < 5; i++) files[i] = new File(dir, "data_batch_" + (i + 1) + ".bin").getPath();
        ByteSource all = CifarLoader.load(files), tst = CifarLoader.load(new File(dir, "test_batch.bin").getPath());
        int n = Math.min(limit, all.size()), nVal = Math.max(1, n / 10), nTrain = n - nVal;
        ByteSource train = all.view(0, nTrain), val = all.view(nTrain, n);
        System.out.printf("train %d, validation %d, test %d  (images kept as bytes: %.0f MB; as double they would be %.0f MB)%n",
            nTrain, nVal, tst.size(), (all.bytes() + tst.bytes()) / 1e6, (all.size() + tst.size()) * 3072 * 8.0 / 1e6);

        Sequential m = resume != null ? ModelIO.loadModelBinary(resume, new Random(1)) : (res ? resnet(new Random(1)) : cnn(new Random(1)));
        if (resume != null) System.out.println("continuing from the saved weights in " + resume + " at epoch " + from);
        if (res) System.out.println("model: BatchNorm2D + residual blocks");
        System.out.print(m.summary(3, 32, 32));        // shapes are checked here, before any training
        DataSource trainSrc = aug ? new AugmentedSource(train, 3, 32, 32, 3, 0, true, 5 + from) : train;
        if (aug) System.out.println("data augmentation ON (random shift up to 3 px + random left-right flip)");

        long t0 = System.nanoTime();
        Trainer.History h = new Trainer().epochs(epochs).batchSize(64).seed(from).printEvery(1).validation(val)
            .schedule(Schedule.cosine(0.002, 0.0001))
            .continueSchedule(from, total > 0 ? total : epochs).checkpoint(res ? "cifar-resnet-ckpt.bin" : "cifar-cnn-ckpt.bin")
            .fit(m, trainSrc, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.002));
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.printf("trained %d epochs in %.0f s (%.1f s/epoch)%n", h.epochsRun, secs, secs / h.epochsRun);
        double acc = m.accuracy(tst);
        System.out.printf("TEST accuracy on %d unseen photos: %.2f%%   (random guessing: 10%%)%n", tst.size(), acc);

        m.setTraining(false);
        int[] idx = new int[tst.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        int[][] cm = new int[10][10];
        for (int s = 0; s < idx.length; s += 200) {
            int e = Math.min(s + 200, idx.length);
            Tensor out = m.forward(tst.features(idx, s, e)), truth = tst.targets(idx, s, e);
            for (int i = 0; i < e - s; i++) cm[truth.argmaxRow(i)][out.argmaxRow(i)]++;
        }
        System.out.printf("%-12s %9s%n", "class", "recall");
        for (int k = 0; k < 10; k++) {
            int tot = 0;
            for (int j = 0; j < 10; j++) tot += cm[k][j];
            System.out.printf("%-12s %8.1f%%%n", CifarLoader.CLASS_NAMES[k], 100.0 * cm[k][k] / tot);
        }
        int bestI = 0, bestJ = 1, worst = -1;
        for (int i = 0; i < 10; i++) for (int j = 0; j < 10; j++) if (i != j && cm[i][j] > worst) { worst = cm[i][j]; bestI = i; bestJ = j; }
        System.out.printf("most common mistake: %d real %s read as %s%n", worst, CifarLoader.CLASS_NAMES[bestI], CifarLoader.CLASS_NAMES[bestJ]);
        ModelIO.saveBinary(m, res ? "cifar-resnet.bin" : "cifar-cnn.bin");
    }
}
