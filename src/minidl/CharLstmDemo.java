package minidl;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * The same next-character task as CharTransformerDemo, with an LSTM instead of attention.
 *   java -Xmx512m -cp out minidl.CharLstmDemo  <text file>  [steps=1500]
 */
public class CharLstmDemo {
    static final int T = CharTransformerDemo.T, D = 64, HIDDEN = 128, BATCH = 32;

    /** Embedding -> LSTM -> Dense per character -> one row of scores per character. */
    static Sequential buildModel(int V, Random rnd) {
        return new Sequential()
            .add(new Embedding(V, D, T, rnd))
            .add(new LSTM(T, D, HIDDEN, rnd))
            .add(new TokenWise(T, HIDDEN, new Dense(HIDDEN, V, rnd)))
            .add(new Reshape(V));
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) { System.out.println("usage: java -cp out minidl.CharLstmDemo <text file> [steps=1500]"); return; }
        String text = Files.readString(Path.of(args[0]));
        int steps = args.length > 1 ? Integer.parseInt(args[1]) : 1500;
        CharSource all = CharSource.fromText(text, T);
        int V = all.vocabSize(), n = all.size();
        CharSource train = all.view(0, n - 2000 - T - 1), val = all.view(n - 640, n);
        System.out.printf("text: %,d characters, %d different. Random guessing: loss %.2f (%.2f bits per character)%n", text.length(), V, Math.log(V), Math.log(V) / Math.log(2));

        Random rnd = new Random(1);
        Sequential m = buildModel(V, rnd);
        System.out.print(m.summary(T));
        int perEpoch = 50;
        long t0 = System.nanoTime();
        Trainer.History h = new Trainer().epochs(steps / perEpoch).maxBatchesPerEpoch(perEpoch).batchSize(BATCH).seed(0).printEvery(2)
            .validation(val).clipNorm(1.0)
            .schedule(Schedule.cosine(0.003, 0.0003))
            .fit(m, train, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.003));
        double secs = (System.nanoTime() - t0) / 1e9;
        double last = h.valLoss.get(h.valLoss.size() - 1);
        System.out.printf("trained %d steps in %.0f s (%.2f s/step). Validation loss %.3f = %.2f bits per character (random guessing: %.2f)%n",
            h.epochsRun * perEpoch, secs, secs / (h.epochsRun * perEpoch), last, last / Math.log(2), Math.log(V) / Math.log(2));
        Random g = new Random(3);
        System.out.println("\n----- sample (temperature 0.8) -----");
        System.out.println(CharTransformerDemo.generate(m, all, "ROMEO:\n", 300, 0.8, g));
    }
}
