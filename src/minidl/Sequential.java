package minidl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** A model = layers in a row. Data flows forward, blame flows backward. */
public class Sequential {
    private static final int EVAL_CHUNK = 256; // rows per prediction chunk: small enough for convolution networks in little memory
    private final List<Layer> layers = new ArrayList<>();

    // ---- mixed-precision training (see setPrecision)
    private Precision precision = Precision.FP32;
    private LossScaler scaler = LossScaler.none();
    private float[][] masters;          // full-precision copies of the weights
    private boolean lastOverflow = false;

    public Sequential add(Layer l) {
        if (layers.isEmpty()) l.setComputeInputGrad(false); // nothing is before it to receive the blame
        layers.add(l);
        return this;
    }

    public List<Layer> layers() { return Collections.unmodifiableList(layers); }

    public Tensor forward(Tensor x) {
        for (Layer l : layers) x = l.forward(x);
        return x;
    }

    /** Use this when you just want answers (turns Dropout off, BatchNorm uses running averages). */
    public Tensor predict(Tensor x) {
        setTraining(false);
        return forward(x);
    }

    public void setTraining(boolean training) {
        for (Layer l : layers) l.setTraining(training);
    }

    public void backward(Tensor grad) {
        for (int i = layers.size() - 1; i >= 0; i--) grad = layers.get(i).backward(grad);
    }

    public List<Param> params() {
        List<Param> all = new ArrayList<>();
        for (Layer l : layers) all.addAll(l.params());
        return all;
    }

    /** Non-learned numbers (BatchNorm running averages). */
    public List<Tensor> state() {
        List<Tensor> all = new ArrayList<>();
        for (Layer l : layers) all.addAll(l.state());
        return all;
    }

    public int paramCount() {
        int n = 0;
        for (Param p : params()) n += p.value.data.length;
        return n;
    }

    /**
     * Turns on 16-bit (mixed precision) training.
     *   - Weights, activations and gradients are rounded to the 16-bit format, like a real 16-bit run.
     *   - A full 32-bit "master" copy of the weights is kept, and that is what the optimizer updates.
     *   - Layers keep what backward needs in 16 bits (Dense, Conv2D), which really saves memory.
     *   - The scaler multiplies the loss so small gradients survive in FP16 (BF16 does not need it).
     * Note: Java has no 16-bit arithmetic, so the maths itself still runs in float32 on rounded numbers.
     * The numbers (and their overflow / underflow behaviour) are what a 16-bit chip would see; the speed is not.
     */
    public Sequential setPrecision(Precision p, LossScaler s) {
        this.precision = p;
        this.scaler = s == null ? LossScaler.none() : s;
        for (Layer l : layers) l.setActivationStorage(p);
        masters = null;
        return this;
    }

    public Sequential setPrecision(Precision p) { return setPrecision(p, p == Precision.FP16 ? LossScaler.dynamic() : LossScaler.none()); }

    public Precision precision() { return precision; }
    public LossScaler lossScaler() { return scaler; }
    /** True if the last mixed-precision step hit an infinity/NaN and was skipped. */
    public boolean lastStepOverflowed() { return lastOverflow; }

    /** One training step on one batch. Returns the loss. */
    public double trainStep(Tensor x, Tensor y, Loss loss, Optimizer opt) {
        return trainStep(x, y, loss, opt, 0);
    }

    /** clipNorm > 0 turns on gradient clipping. */
    public double trainStep(Tensor x, Tensor y, Loss loss, Optimizer opt, double clipNorm) {
        if (precision != Precision.FP32) return trainStepMixed(x, y, loss, opt, clipNorm);
        setTraining(true);
        List<Param> ps = params();
        for (Param p : ps) p.zeroGrad();
        double l = loss.forward(forward(x), y);
        backward(loss.backward());
        if (clipNorm > 0) GradClip.clipGlobalNorm(ps, clipNorm);
        opt.step(ps);
        return l;
    }

    private double trainStepMixed(Tensor x, Tensor y, Loss loss, Optimizer opt, double clipNorm) {
        setTraining(true);
        List<Param> ps = params();
        if (masters == null || masters.length != ps.size()) {
            masters = new float[ps.size()][];
            for (int k = 0; k < masters.length; k++) masters[k] = new float[ps.get(k).value.data.length];
        }
        for (int k = 0; k < ps.size(); k++) {            // 1. keep the exact weights, compute with 16-bit weights
            Param p = ps.get(k);
            System.arraycopy(p.value.data, 0, masters[k], 0, masters[k].length);
            Half.roundInPlace(p.value.data, precision);
            p.zeroGrad();
        }
        Tensor h = x.copy();                              // 2. forward: every layer's output is rounded to 16 bits
        if (!(layers.get(0) instanceof Embedding)) Half.roundInPlace(h.data, precision); // token ids must stay exact
        for (Layer l : layers) { h = l.forward(h); Half.roundInPlace(h.data, precision); }
        double l = loss.forward(h, y);                    // the loss itself is computed in 32 bits
        double s = scaler.scale();
        Tensor g = loss.backward();
        g.scaleInPlace(s);                                // 3. backward starts from a SCALED loss so small gradients survive
        for (int i = layers.size() - 1; i >= 0 && g != null; i--) {
            Half.roundInPlace(g.data, precision);
            g = layers.get(i).backward(g);
        }
        for (int k = 0; k < ps.size(); k++) {             // 4. back to the exact weights; gradients stored in 16 bits, then unscaled in 32
            Param p = ps.get(k);
            System.arraycopy(masters[k], 0, p.value.data, 0, masters[k].length);
        }
        boolean overflow = !Double.isFinite(l);
        float inv = (float) (1.0 / s);
        for (Param p : ps) {
            Half.roundInPlace(p.grad.data, precision);
            for (int i = 0; i < p.grad.data.length; i++) {
                float v = p.grad.data[i];
                if (Float.isNaN(v) || Float.isInfinite(v)) { overflow = true; break; }
                p.grad.data[i] = v * inv;
            }
            if (overflow) break;
        }
        lastOverflow = overflow;
        if (!overflow) {                                  // 5. update the master weights (skipped if anything overflowed)
            if (clipNorm > 0) GradClip.clipGlobalNorm(ps, clipNorm);
            if (opt != null) opt.step(ps);
        }
        scaler.update(overflow);
        return l;
    }

    /** Simple training. For validation, early stopping, schedules use {@link Trainer}. */
    public void fit(Tensor x, Tensor y, Loss loss, Optimizer opt,
                    int epochs, int batchSize, long seed, int printEvery) {
        new Trainer().epochs(epochs).batchSize(batchSize).seed(seed).printEvery(printEvery).fit(this, x, y, loss, opt);
    }

    /** % of rows where the highest score is the correct class. */
    public double accuracy(Tensor x, Tensor oneHot) {
        Tensor out = predict(x);
        int ok = 0;
        for (int i = 0; i < x.rows; i++)
            if (out.argmaxRow(i) == oneHot.argmaxRow(i)) ok++;
        return 100.0 * ok / x.rows;
    }

    /** Average loss over a whole DataSource, 256 rows at a time (so big data never has to fit in memory as tensors). */
    public double evalLoss(DataSource src, Loss loss) {
        setTraining(false);
        int n = src.size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        double total = 0;
        for (int s = 0; s < n; s += EVAL_CHUNK) {
            int e = Math.min(s + EVAL_CHUNK, n);
            total += loss.forward(forward(src.features(idx, s, e)), src.targets(idx, s, e)) * (e - s);
        }
        return total / n;
    }

    /** % correct over a whole DataSource, 256 rows at a time. */
    public double accuracy(DataSource src) {
        setTraining(false);
        int n = src.size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        int ok = 0;
        for (int s = 0; s < n; s += EVAL_CHUNK) {
            int e = Math.min(s + EVAL_CHUNK, n);
            Tensor out = forward(src.features(idx, s, e)), truth = src.targets(idx, s, e);
            for (int i = 0; i < e - s; i++) if (out.argmaxRow(i) == truth.argmaxRow(i)) ok++;
        }
        return 100.0 * ok / n;
    }

    /** Memory used by this model's learned numbers (weights + running averages), in bytes. */
    public long weightBytes() {
        long b = 0;
        for (Layer l : layers) b += l.weightBytes();
        return b;
    }

    /** Copy of all weights + state. Used to remember the best moment during training. */
    public List<float[]> snapshot() {
        List<float[]> snap = new ArrayList<>();
        for (Param p : params()) snap.add(p.value.data.clone());
        for (Tensor t : state()) snap.add(t.data.clone());
        return snap;
    }

    public void restore(List<float[]> snap) {
        int k = 0;
        for (Param p : params()) { float[] s = snap.get(k++); System.arraycopy(s, 0, p.value.data, 0, s.length); }
        for (Tensor t : state()) { float[] s = snap.get(k++); System.arraycopy(s, 0, t.data, 0, s.length); }
    }

    /** One sample in -> the shape after every layer. Fails with a clear message naming the layer that does not fit. */
    public List<int[]> shapes(int... inputShape) {
        List<int[]> out = new ArrayList<>();
        int[] s = inputShape;
        for (int i = 0; i < layers.size(); i++) {
            try { s = layers.get(i).outputShape(s); }
            catch (IllegalArgumentException e) { throw new IllegalArgumentException("layer " + (i + 1) + " (" + layers.get(i).spec() + "): " + e.getMessage()); }
            out.add(s);
        }
        return out;
    }

    /** Predict straight from an N-dimensional tensor whose FIRST dimension is the batch, e.g. [N, 1, 28, 28]. */
    public Tensor predict(NDTensor x) { return predict(x.toMatrix(1)); }

    /** Table of layers, the shape each one outputs for ONE sample of the given input shape, and weight counts. Checks the shapes on the way. */
    public String summary(int... inputShape) {
        List<int[]> sh = shapes(inputShape);
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-4s %-30s %-18s %10s%n", "#", "Layer", "Output shape", "Weights"));
        sb.append(String.format("%-4s %-30s %-18s %10s%n", "", "input", java.util.Arrays.toString(inputShape), ""));
        for (int i = 0; i < layers.size(); i++) {
            int n = 0;
            for (Param p : layers.get(i).params()) n += p.value.data.length;
            String spec = layers.get(i).spec();
            if (spec.length() > 30) spec = spec.substring(0, 27) + "...";
            sb.append(String.format("%-4d %-30s %-18s %10d%n", i + 1, spec, java.util.Arrays.toString(sh.get(i)), n));
        }
        sb.append(String.format("%-4s %-30s %-18s %10d%n", "", "TOTAL", "", paramCount()));
        return sb.toString();
    }

    /** Pretty table of layers and how many weights each has. */
    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-24s %10s%n", "Layer", "Weights"));
        for (Layer l : layers) {
            int n = 0;
            for (Param p : l.params()) n += p.value.data.length;
            sb.append(String.format("%-24s %10d%n", l.spec(), n));
        }
        sb.append(String.format("%-24s %10d%n", "TOTAL", paramCount()));
        return sb.toString();
    }
}
