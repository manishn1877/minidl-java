package minidl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A trained 784 -> 10 digit model plus the extras a real app needs:
 *  - predict(): probabilities, with optional test-time augmentation (the answer is averaged over the
 *    digit nudged 1 pixel in 4 directions, which makes it steadier on shaky handwriting)
 *  - learn():   "you got that wrong, it is a 7". A few small gradient steps on that one drawing,
 *    mixed with earlier corrections and (optionally) real MNIST digits so it does not forget the rest.
 *  - reset():   go back to the weights the model started with.
 * All methods are synchronized, so a web server can call them from several threads.
 */
public class DigitRecognizer {
    public static final int DIM = DigitPreprocess.SIZE * DigitPreprocess.SIZE;
    public static final String DIGITS = "0123456789", LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    /** What the model can answer with (10 outputs = digits, 26 = letters, 36 = digits and letters, 52 = capital and small letters, 62 = digits and both cases) and how to prepare a drawing for it. */
    public final String alphabet;
    public final int classes;

    public static final class Result {
        public final double[] probs;
        public final int label;
        public final double confidence, entropyBits;
        Result(double[] p) {
            probs = p;
            int best = 0; double h = 0;
            for (int i = 0; i < p.length; i++) { if (p[i] > p[best]) best = i; if (p[i] > 1e-12) h -= p[i] * Math.log(p[i]) / Math.log(2); }
            label = best; confidence = p[best]; entropyBits = h;
        }
    }

    private final Sequential model;
    private final List<float[]> original;
    private final List<float[]> memX = new ArrayList<>();
    private final List<Integer> memY = new ArrayList<>();
    private final Random rnd = new Random(5);
    private DataSource replay;
    private int learned;

    public DigitRecognizer(Sequential model) {
        this.model = model;
        this.original = model.snapshot();
        this.classes = model.predict(new Tensor(1, DIM)).cols;
        this.alphabet = classes == 62 ? EmnistLoader.FULL : classes == 52 ? EmnistLoader.CASED : classes == 36 ? Alnum.CHARS : classes == 26 ? LETTERS : DIGITS.substring(0, Math.min(classes, 10));
    }

    public DigitPreprocess.Style style() { return classes >= 26 ? DigitPreprocess.Style.EMNIST : DigitPreprocess.Style.MNIST; }

    /** Real labelled digits that are mixed into every correction step (stops the model forgetting). */
    public synchronized DigitRecognizer replayFrom(DataSource src) { this.replay = src; return this; }

    public synchronized Result predict(float[] px, boolean tta) {
        List<float[]> views = new ArrayList<>();
        views.add(px);
        if (tta) for (int[] d : new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) views.add(shift(px, d[0], d[1]));
        Tensor x = new Tensor(views.size(), DIM);
        for (int i = 0; i < views.size(); i++) System.arraycopy(views.get(i), 0, x.data, i * DIM, DIM);
        Tensor out = model.predict(x);
        double[] p = new double[out.cols];
        for (int r = 0; r < out.rows; r++) {
            double max = Double.NEGATIVE_INFINITY, sum = 0;
            for (int c = 0; c < out.cols; c++) max = Math.max(max, out.get(r, c));
            double[] e = new double[out.cols];
            for (int c = 0; c < out.cols; c++) { e[c] = Math.exp(out.get(r, c) - max); sum += e[c]; }
            for (int c = 0; c < out.cols; c++) p[c] += e[c] / sum / out.rows;
        }
        return new Result(p);
    }

    /** Teach the model that px is `label`. Returns the loss of the last step. */
    public synchronized double learn(float[] px, int label) {
        if (label < 0 || label >= classes) throw new IllegalArgumentException("label must be 0.." + (classes - 1));
        memX.add(px); memY.add(label);
        if (memX.size() > 300) { memX.remove(0); memY.remove(0); }
        double loss = 0;
        for (int step = 0; step < 4; step++) {
            int nMem = Math.min(memX.size(), 15), nRep = replay == null ? 0 : 16;
            Tensor x = new Tensor(1 + nMem + nRep, DIM), y = new Tensor(1 + nMem + nRep, classes);
            put(x, y, 0, px, label); // the new correction is always in the batch
            for (int i = 0; i < nMem; i++) { int j = rnd.nextInt(memX.size()); put(x, y, 1 + i, memX.get(j), memY.get(j)); }
            if (nRep > 0) {
                int[] idx = new int[nRep];
                for (int i = 0; i < nRep; i++) idx[i] = rnd.nextInt(replay.size());
                Tensor fx = replay.features(idx, 0, nRep), fy = replay.targets(idx, 0, nRep);
                System.arraycopy(fx.data, 0, x.data, (1 + nMem) * DIM, nRep * DIM);
                System.arraycopy(fy.data, 0, y.data, (1 + nMem) * classes, nRep * classes);
            }
            loss = model.trainStep(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.SGD(0.02));
        }
        learned++;
        return loss;
    }

    public synchronized void reset() { model.restore(original); memX.clear(); memY.clear(); learned = 0; }
    public synchronized int learnedCount() { return learned; }
    public synchronized double accuracy(DataSource test) { return model.accuracy(test); }
    public synchronized String summary() { return model.summary(); }
    public synchronized int paramCount() { return model.paramCount(); }
    public synchronized long weightBytes() { return model.weightBytes(); }
    public synchronized void save(String path) throws java.io.IOException { ModelIO.saveBinary(model, path); }

    private void put(Tensor x, Tensor y, int row, float[] px, int label) {
        System.arraycopy(px, 0, x.data, row * DIM, DIM);
        y.data[row * classes + label] = 1f;
    }

    /** The 28 x 28 picture moved dx right and dy down (empty paper fills the gap). */
    static float[] shift(float[] px, int dx, int dy) {
        int n = DigitPreprocess.SIZE;
        float[] out = new float[DIM];
        for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) {
            int sx = x - dx, sy = y - dy;
            if (sx >= 0 && sx < n && sy >= 0 && sy < n) out[y * n + x] = px[sy * n + sx];
        }
        return out;
    }
}
