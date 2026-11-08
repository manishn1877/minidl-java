package minidl;

import java.io.File;
import java.util.Random;

/**
 * ONE model for digits and letters (36 classes: 0-9, A-Z). Needs both data sets (get-mnist.sh, get-emnist.sh).
 *  java -Xmx1g -cp out minidl.AlnumDemo &lt;mnist-folder&gt; &lt;emnist-folder&gt; [epochs=8] [mnistLimit=60000] [aug]
 * Saves alnum-cnn.bin. The web app uses it with  ./run-web.sh all
 * Some pairs are impossible to tell apart on their own, even for people: 0 and O, 1 and I, 2 and Z, 5 and S, 8 and B.
 * The model can only guess from the shape and the habit of the writer, so expect more mix-ups there than in the two separate models.
 */
public class AlnumDemo {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) { System.out.println("usage: java -Xmx1g -cp out minidl.AlnumDemo <mnist-folder> <emnist-folder> [epochs=8] [mnistLimit=60000] [aug]"); return; }
        int epochs = args.length > 2 ? Integer.parseInt(args[2]) : 8;
        int limit = args.length > 3 ? Integer.parseInt(args[3]) : 60000;
        boolean aug = args.length > 4 && args[4].equals("aug");

        ByteSource all = Alnum.load(args[0], args[1], "train", limit), test = Alnum.load(args[0], args[1], "test", Integer.MAX_VALUE);
        int nVal = all.size() / 10, nTrain = all.size() - nVal;
        DataSource train = all.view(0, nTrain), val = all.view(nTrain, all.size());
        System.out.printf("train %d, validation %d, test %d pictures (digits + letters, 36 classes)%n", nTrain, nVal, test.size());

        Sequential m = EmnistDemo.cnn(new Random(1), 36);
        System.out.print(m.summary());
        DataSource trainSrc = aug ? new AugmentedSource(train, 28, 28, 2, 10, 7) : train;
        long t0 = System.nanoTime();
        Trainer.History h = new Trainer().epochs(epochs).batchSize(64).seed(0).printEvery(1)
            .validation(val).patience(aug ? 6 : 3).schedule(Schedule.cosine(0.002, 0.0001))
            .fit(m, trainSrc, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.002));
        System.out.printf("trained %d epochs in %.0f s%n", h.epochsRun, (System.nanoTime() - t0) / 1e9);
        System.out.printf("TEST accuracy on %d unseen pictures: %.2f%%  (random guessing: 2.78%%)%n", test.size(), m.accuracy(test));

        ModelIO.saveBinary(m, "alnum-cnn.bin");
        ModelIO.saveBinary(Quantizer.quantize(m), "alnum-cnn-int8.bin");
        System.out.printf("files: alnum-cnn.bin %d KB, alnum-cnn-int8.bin %d KB%n", new File("alnum-cnn.bin").length() / 1024, new File("alnum-cnn-int8.bin").length() / 1024);
    }
}
