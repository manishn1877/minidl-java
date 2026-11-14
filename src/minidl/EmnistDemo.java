package minidl;

import java.io.File;
import java.util.Random;

/**
 * Handwritten LETTERS (EMNIST). Run get-emnist.sh first.
 *   java -cp out minidl.EmnistDemo &lt;emnist-folder&gt; [epochs=8] [trainLimit=124800] [aug] [letters|case|full]
 * Saves emnist-letters-cnn.bin (and an int8 copy). The web app (./run-web.sh letters) uses the first one.
 * Letters are harder than digits (26 classes, "I" vs "l", "O" vs "Q", upper/lower case merged): expect about 90 to 93%.
 */
public class EmnistDemo {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) { System.out.println("usage: java -cp out minidl.EmnistDemo <emnist-folder> [epochs=8] [trainLimit=124800] [aug] [letters|case|full]\n  letters = A-Z (26, capital and small merged)   case = A-Z and a-z (52)   full = 0-9, A-Z, a-z (62, needs the byclass files)"); return; }
        String dir = args[0];
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        int limit = args.length > 2 ? Integer.parseInt(args[2]) : Integer.MAX_VALUE;
        boolean aug = false;
        String kind = "letters";
        for (int i = 3; i < args.length; i++) { if (args[i].equals("aug")) aug = true; else kind = args[i]; }
        if (!kind.equals("letters") && !kind.equals("case") && !kind.equals("full")) throw new IllegalArgumentException("last argument must be letters, case or full");
        int classes = kind.equals("letters") ? 26 : kind.equals("case") ? 52 : 62;
        if (classes > 26 && limit == Integer.MAX_VALUE) limit = 200_000; // byclass has 697,932 training pictures: use the first 200,000 unless asked
        ByteSource all = classes == 26 ? EmnistLoader.lettersFrom(dir, "train", limit) : EmnistLoader.byClass(dir, "train", limit, classes == 52);
        ByteSource test = classes == 26 ? EmnistLoader.lettersFrom(dir, "test", Integer.MAX_VALUE) : EmnistLoader.byClass(dir, "test", 40_000, classes == 52);
        int nVal = all.size() / 10, nTrain = all.size() - nVal;
        DataSource train = all.view(0, nTrain), val = all.view(nTrain, all.size());
        System.out.printf("train %d, validation %d, test %d letters%n", nTrain, nVal, test.size());

        Sequential m = cnn(new Random(1), classes);
        System.out.print(m.summary());
        DataSource trainSrc = aug ? new AugmentedSource(train, 28, 28, 2, 10, 7) : train;
        long t0 = System.nanoTime();
        Trainer.History h = new Trainer().epochs(epochs).batchSize(64).seed(0).printEvery(1)
            .validation(val).patience(aug ? 6 : 3).schedule(Schedule.cosine(0.002, 0.0001))
            .fit(m, trainSrc, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.002));
        System.out.printf("trained %d epochs in %.0f s%n", h.epochsRun, (System.nanoTime() - t0) / 1e9);
        System.out.printf("TEST accuracy on %d unseen pictures: %.2f%%  (random guessing: %.2f%%)%n", test.size(), m.accuracy(test), 100.0 / classes);

        String stem = "emnist-" + kind + "-cnn";
        ModelIO.saveBinary(m, stem + ".bin");
        ModelIO.saveBinary(Quantizer.quantize(m), stem + "-int8.bin");
        System.out.printf("files: %s.bin %d KB, %s-int8.bin %d KB%n", stem, new File(stem + ".bin").length() / 1024, stem, new File(stem + "-int8.bin").length() / 1024);
    }

    /** Two convolution stages (16 and 32 filters), then Dense 128 -> 26. A bit wider than the digit model: 26 classes need more room. */
    static Sequential cnn(Random rnd) { return cnn(rnd, 26); }

    static Sequential cnn(Random rnd, int classes) {
        return new Sequential()
            .add(new Conv2D(1, 28, 28, 16, 3, 1, 0, rnd)).add(new Activations.ReLU())   // 16 x 26 x 26
            .add(new MaxPool2D(16, 26, 26, 2))                                           // 16 x 13 x 13
            .add(new Conv2D(16, 13, 13, 32, 3, 1, 0, rnd)).add(new Activations.ReLU())  // 32 x 11 x 11
            .add(new MaxPool2D(32, 11, 11, 2))                                           // 32 x 5 x 5 = 800
            .add(new Dense(800, classes > 40 ? 192 : 128, rnd)).add(new Activations.ReLU())
            .add(new Dropout(0.25, rnd))
            .add(new Dense(classes > 40 ? 192 : 128, classes, rnd));
    }
}
