package minidl;

import java.io.File;
import java.util.Random;

/**
 * Handwritten LETTERS (EMNIST). Run get-emnist.sh first.
 *   java -cp out minidl.EmnistDemo &lt;emnist-folder&gt; [epochs=8] [trainLimit=124800] [aug]
 * Saves emnist-letters-cnn.bin (and an int8 copy). The web app (./run-web.sh letters) uses the first one.
 * Letters are harder than digits (26 classes, "I" vs "l", "O" vs "Q", upper/lower case merged): expect about 90 to 93%.
 */
public class EmnistDemo {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) { System.out.println("usage: java -cp out minidl.EmnistDemo <emnist-folder> [epochs=8] [trainLimit=124800] [aug]"); return; }
        String dir = args[0];
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        int limit = args.length > 2 ? Integer.parseInt(args[2]) : Integer.MAX_VALUE;
        boolean aug = args.length > 3 && args[3].equals("aug");

        ByteSource all = EmnistLoader.lettersFrom(dir, "train", limit), test = EmnistLoader.lettersFrom(dir, "test", Integer.MAX_VALUE);
        int nVal = all.size() / 10, nTrain = all.size() - nVal;
        DataSource train = all.view(0, nTrain), val = all.view(nTrain, all.size());
        System.out.printf("train %d, validation %d, test %d letters%n", nTrain, nVal, test.size());

        Sequential m = cnn(new Random(1));
        System.out.print(m.summary());
        DataSource trainSrc = aug ? new AugmentedSource(train, 28, 28, 2, 10, 7) : train;
        long t0 = System.nanoTime();
        Trainer.History h = new Trainer().epochs(epochs).batchSize(64).seed(0).printEvery(1)
            .validation(val).patience(aug ? 6 : 3).schedule(Schedule.cosine(0.002, 0.0001))
            .fit(m, trainSrc, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.002));
        System.out.printf("trained %d epochs in %.0f s%n", h.epochsRun, (System.nanoTime() - t0) / 1e9);
        System.out.printf("TEST accuracy on %d unseen letters: %.2f%%  (random guessing: 3.85%%)%n", test.size(), m.accuracy(test));

        ModelIO.saveBinary(m, "emnist-letters-cnn.bin");
        ModelIO.saveBinary(Quantizer.quantize(m), "emnist-letters-cnn-int8.bin");
        System.out.printf("files: emnist-letters-cnn.bin %d KB, emnist-letters-cnn-int8.bin %d KB%n",
            new File("emnist-letters-cnn.bin").length() / 1024, new File("emnist-letters-cnn-int8.bin").length() / 1024);
    }

    /** Two convolution stages (16 and 32 filters), then Dense 128 -> 26. A bit wider than the digit model: 26 classes need more room. */
    static Sequential cnn(Random rnd) { return cnn(rnd, 26); }

    static Sequential cnn(Random rnd, int classes) {
        return new Sequential()
            .add(new Conv2D(1, 28, 28, 16, 3, 1, 0, rnd)).add(new Activations.ReLU())   // 16 x 26 x 26
            .add(new MaxPool2D(16, 26, 26, 2))                                           // 16 x 13 x 13
            .add(new Conv2D(16, 13, 13, 32, 3, 1, 0, rnd)).add(new Activations.ReLU())  // 32 x 11 x 11
            .add(new MaxPool2D(32, 11, 11, 2))                                           // 32 x 5 x 5 = 800
            .add(new Dense(800, 128, rnd)).add(new Activations.ReLU())
            .add(new Dropout(0.25, rnd))
            .add(new Dense(128, classes, rnd));
    }
}
