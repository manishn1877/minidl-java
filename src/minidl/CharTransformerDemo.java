package minidl;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * A tiny GPT-style language model, built from the library's own layers, that learns to write
 * text one character at a time.
 *   java -Xmx512m -cp out minidl.CharTransformerDemo  &lt;text file&gt;  [steps=1500]
 * Try it with the Tiny Shakespeare file (about 1.1 MB of plays).
 */
public class CharTransformerDemo {
    static int T = 32;                  // context window (characters or tokens): option --context
    static boolean TIE = false;         // option --tie: the output layer shares its weights with the input embedding (weight tying)
    static double DROPOUT = 0.0;        // option --dropout: randomly switches off this share of the numbers while training (0 = off)
    static final int D = 64, HEADS = 4, LAYERS = 2, BATCH = 32;

    /** Embedding -> (attention block, MLP block) x LAYERS -> LayerNorm -> Dense -> one row per character. */
    static Sequential buildModel(int V, Random rnd) {
        Embedding emb = new Embedding(V, D, T, rnd);
        Sequential m = new Sequential().add(emb);
        for (int l = 0; l < LAYERS; l++) {
            if (DROPOUT > 0) {
                m.add(new Residual(new TokenWise(T, D, new LayerNorm(D)), new CausalSelfAttention(T, D, HEADS, rnd), new Dropout(DROPOUT, rnd)));
                m.add(new Residual(new TokenWise(T, D, new LayerNorm(D), new Dense(D, 4 * D, rnd), new Activations.GELU(), new Dense(4 * D, D, rnd)), new Dropout(DROPOUT, rnd)));
            } else {
                m.add(new Residual(new TokenWise(T, D, new LayerNorm(D)), new CausalSelfAttention(T, D, HEADS, rnd)));
                m.add(new Residual(new TokenWise(T, D, new LayerNorm(D), new Dense(D, 4 * D, rnd), new Activations.GELU(), new Dense(4 * D, D, rnd))));
            }
        }
        m.add(new TokenWise(T, D, new LayerNorm(D), TIE ? new TiedOutput(emb) : new Dense(D, V, rnd)));
        m.add(new Reshape(V));
        return m;
    }

    /** The same model, but every attention / LayerNorm / Dense / GELU is an autograd layer (forward maths only). */
    static Sequential buildAutoModel(int V, Random rnd) {
        Sequential m = new Sequential().add(new Embedding(V, D, T, rnd));
        for (int l = 0; l < LAYERS; l++) {
            m.add(new Residual(new TokenWise(T, D, AutoLayers.layerNorm(D)), AutoLayers.attention(T, D, HEADS, rnd)));
            m.add(new Residual(new TokenWise(T, D, AutoLayers.layerNorm(D), AutoLayers.dense(D, 4 * D, rnd), AutoLayers.gelu(), AutoLayers.dense(4 * D, D, rnd))));
        }
        m.add(new TokenWise(T, D, AutoLayers.layerNorm(D), AutoLayers.dense(D, V, rnd)));
        m.add(new Reshape(V));
        return m;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("usage: java -cp out minidl.CharTransformerDemo <text file> [steps=1500] [--bpe VOCAB] [--context T] [--dropout P] [--tie]\n"
                + "  --bpe VOCAB   read sub-word tokens (byte-pair encoding, vocabulary of VOCAB) instead of single characters\n"
                + "  --context T   how many characters/tokens the model sees at once (default 32)\n"
                + "  --tie         the output layer shares its weights with the input embedding (fewer weights)\n"
                + "  --dropout P   switch off a share P (for example 0.1) of the numbers while training, against over-fitting");
            return;
        }
        String text = Files.readString(Path.of(args[0]));
        int steps = 1500, bpeVocab = 0;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--bpe": bpeVocab = Integer.parseInt(args[++i]); break;
                case "--context": T = Integer.parseInt(args[++i]); break;
                case "--dropout": DROPOUT = Double.parseDouble(args[++i]); break;
                case "--tie": TIE = true; break;
                default: steps = Integer.parseInt(args[i]);
            }
        }
        if (bpeVocab > 0) { runBpe(text, steps, bpeVocab); return; }

        CharSource all = CharSource.fromText(text, T);
        int V = all.vocabSize(), n = all.size();
        CharSource train = all.view(0, n - 2000 - T - 1), val = all.view(n - 640, n); // validation windows come from the last part of the text
        System.out.printf("text: %,d characters, %d different characters. Guessing at random would give loss %.2f (%.2f bits per character)%n",
            text.length(), V, Math.log(V), Math.log(V) / Math.log(2));

        Random rnd = new Random(1);
        Sequential m = buildModel(V, rnd);
        System.out.print(m.summary());
        System.out.printf("batch %d sequences x %d characters, %d training steps%n", BATCH, T, steps);

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
        System.out.println(generate(m, all, "ROMEO:\n", 300, 0.8, g));
        System.out.println("\n----- sample (temperature 1.0) -----");
        System.out.println(generate(m, all, "KING:\n", 300, 1.0, g));

        ModelIO.saveBinary(m, "char-transformer.bin");
        Sequential q = Quantizer.quantize(m);
        System.out.printf("%nweights: float32 %.0f KB, int8 for the Dense layers %.0f KB (attention weights stay float32)%n", m.weightBytes() / 1024.0, q.weightBytes() / 1024.0);
    }

    /** The same model over sub-word tokens: a BPE vocabulary is learned from the text, then the transformer predicts the next TOKEN. */
    static void runBpe(String text, int steps, int vocab) throws Exception {
        long t0 = System.nanoTime();
        BpeTokenizer tok = BpeTokenizer.train(text, vocab);
        int[] ids = tok.encode(text);
        System.out.printf("BPE: vocabulary %d tokens learned in %.1f s; the text is %,d characters = %,d tokens (%.2f characters per token)%n",
            tok.vocabSize(), (System.nanoTime() - t0) / 1e9, text.length(), ids.length, (double) text.length() / ids.length);
        int V = tok.vocabSize();
        TokenSource all = TokenSource.of(ids, V, T);
        int n = all.size();
        TokenSource train = all.view(0, n - 1500 - T - 1), val = all.view(n - 640, n);
        Sequential m = buildModel(V, new Random(1));
        System.out.printf("random guessing would give loss %.2f; batch %d sequences x %d tokens (about %d characters of context), %d steps, dropout %.2f%n",
            Math.log(V), BATCH, T, (int) (T * (double) text.length() / ids.length), steps, DROPOUT);
        int perEpoch = 50;
        t0 = System.nanoTime();
        Trainer.History h = new Trainer().epochs(steps / perEpoch).maxBatchesPerEpoch(perEpoch).batchSize(BATCH).seed(0).printEvery(2)
            .validation(val).clipNorm(1.0).schedule(Schedule.cosine(0.003, 0.0003))
            .fit(m, train, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.003));
        double secs = (System.nanoTime() - t0) / 1e9, last = h.valLoss.get(h.valLoss.size() - 1);
        System.out.printf("trained %d steps in %.0f s (%.2f s/step). Validation loss %.3f per token (random guessing: %.2f)%n", h.epochsRun * perEpoch, secs, secs / (h.epochsRun * perEpoch), last, Math.log(V));
        Random g = new Random(3);
        System.out.println("\n----- sample (temperature 0.8) -----\n" + generateTokens(m, tok, "ROMEO:\n", 80, 0.8, g));
        ModelIO.saveBinary(m, "bpe-transformer.bin");
        tok.save("bpe-tokenizer.txt");
        System.out.println("\nsaved bpe-transformer.bin and bpe-tokenizer.txt");
    }

    static String generateTokens(Sequential m, BpeTokenizer tok, String prompt, int count, double temperature, Random rnd) {
        int V = tok.vocabSize();
        java.util.List<Integer> ids = new java.util.ArrayList<>();
        for (int v : tok.encode(prompt)) ids.add(v);
        int pad = tok.encode(" ").length > 0 ? tok.encode(" ")[0] : 0;
        for (int step = 0; step < count; step++) {
            float[] ctx = new float[T];
            for (int t = 0; t < T; t++) { int pos = ids.size() - T + t; ctx[t] = pos >= 0 ? ids.get(pos) : pad; }
            Tensor out = m.predict(new Tensor(1, T, ctx));
            int base = (T - 1) * V;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < V; j++) max = Math.max(max, out.data[base + j]);
            double[] p = new double[V];
            double sum = 0;
            for (int j = 0; j < V; j++) { p[j] = Math.exp((out.data[base + j] - max) / temperature); sum += p[j]; }
            double r = rnd.nextDouble() * sum, acc = 0;
            int pick = V - 1;
            for (int j = 0; j < V; j++) { acc += p[j]; if (r <= acc) { pick = j; break; } }
            ids.add(pick);
        }
        int[] all = new int[ids.size()];
        for (int i = 0; i < all.length; i++) all[i] = ids.get(i);
        return tok.decode(all);
    }

    /** Writes `count` characters after `prompt`, sampling each next character from the model's predicted percentages. */
    static String generate(Sequential m, CharSource src, String prompt, int count, double temperature, Random rnd) {
        int V = src.vocabSize();
        StringBuilder sb = new StringBuilder(prompt);
        int pad = src.id(' ');
        for (int step = 0; step < count; step++) {
            float[] ctx = new float[T];
            for (int t = 0; t < T; t++) {
                int pos = sb.length() - T + t;
                ctx[t] = pos >= 0 ? src.id(sb.charAt(pos)) : pad;
            }
            Tensor out = m.predict(new Tensor(1, T, ctx));
            int base = (T - 1) * V; // prediction made at the last character
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < V; j++) max = Math.max(max, out.data[base + j]);
            double[] p = new double[V];
            double sum = 0;
            for (int j = 0; j < V; j++) { p[j] = Math.exp((out.data[base + j] - max) / temperature); sum += p[j]; }
            double r = rnd.nextDouble() * sum, acc = 0;
            int pick = V - 1;
            for (int j = 0; j < V; j++) { acc += p[j]; if (r <= acc) { pick = j; break; } }
            sb.append(src.charOf(pick));
        }
        return sb.toString();
    }
}
