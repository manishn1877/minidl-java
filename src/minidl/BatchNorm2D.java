package minidl;

import java.util.List;

/**
 * Batch normalization for pictures. Like {@link BatchNorm}, but the numbers are normalized PER CHANNEL: all the pixels of
 * channel c (over the whole batch and over every y, x) share one mean and one variance, and one scale and shift.
 * Data layout is the same as Conv2D: every row is one image, flattened as [channel][y][x].
 * Why: it keeps the numbers between layers in a friendly range, so deep convolutional networks (ResNet) train fast and stably.
 */
public class BatchNorm2D implements Layer {
    private static final double EPS = 1e-5;

    private final int channels, h, w, size;
    private final double momentum;
    private final Param gamma, beta;
    private final Tensor runMean, runVar;
    private boolean training = true;

    private Tensor xhat;
    private double[] invStd;
    private boolean usedBatchStats;

    public BatchNorm2D(int channels, int h, int w) { this(channels, h, w, 0.9); }

    public BatchNorm2D(int channels, int h, int w, double momentum) {
        if (channels < 1 || h < 1 || w < 1) throw new IllegalArgumentException("bad BatchNorm2D size");
        this.channels = channels; this.h = h; this.w = w; this.size = channels * h * w; this.momentum = momentum;
        Tensor g = new Tensor(1, channels);
        java.util.Arrays.fill(g.data, 1f);
        gamma = new Param(g, false);
        beta = new Param(new Tensor(1, channels), false);
        runMean = new Tensor(1, channels);
        runVar = new Tensor(1, channels);
        java.util.Arrays.fill(runVar.data, 1f);
    }

    @Override public void setTraining(boolean training) { this.training = training; }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != size) throw new IllegalArgumentException("BatchNorm2D expects " + size + " numbers per image, got " + x.cols);
        int n = x.rows, hw = h * w;
        double count = (double) n * hw;
        double[] mean = new double[channels], var = new double[channels];
        usedBatchStats = training && count > 1;
        if (usedBatchStats) {
            for (int i = 0; i < n; i++) for (int c = 0; c < channels; c++) { int base = i * size + c * hw; for (int p = 0; p < hw; p++) mean[c] += x.data[base + p]; }
            for (int c = 0; c < channels; c++) mean[c] /= count;
            for (int i = 0; i < n; i++) for (int c = 0; c < channels; c++) { int base = i * size + c * hw; for (int p = 0; p < hw; p++) { double d = x.data[base + p] - mean[c]; var[c] += d * d; } }
            for (int c = 0; c < channels; c++) {
                var[c] /= count;
                runMean.data[c] = (float) (momentum * runMean.data[c] + (1 - momentum) * mean[c]);
                runVar.data[c] = (float) (momentum * runVar.data[c] + (1 - momentum) * var[c] * count / (count - 1.0));
            }
        } else {
            for (int c = 0; c < channels; c++) { mean[c] = runMean.data[c]; var[c] = runVar.data[c]; }
        }
        invStd = new double[channels];
        for (int c = 0; c < channels; c++) invStd[c] = 1.0 / Math.sqrt(var[c] + EPS);
        xhat = new Tensor(n, size);
        Tensor out = new Tensor(n, size);
        for (int i = 0; i < n; i++) for (int c = 0; c < channels; c++) {
            int base = i * size + c * hw;
            for (int p = 0; p < hw; p++) {
                float v = (float) ((x.data[base + p] - mean[c]) * invStd[c]);
                xhat.data[base + p] = v;
                out.data[base + p] = gamma.value.data[c] * v + beta.value.data[c];
            }
        }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        int n = g.rows, hw = h * w;
        double count = (double) n * hw;
        double[] dgamma = new double[channels], dbeta = new double[channels];
        for (int i = 0; i < n; i++) for (int c = 0; c < channels; c++) {
            int base = i * size + c * hw;
            for (int p = 0; p < hw; p++) { dbeta[c] += g.data[base + p]; dgamma[c] += g.data[base + p] * xhat.data[base + p]; }
        }
        for (int c = 0; c < channels; c++) { beta.grad.data[c] = (float) dbeta[c]; gamma.grad.data[c] = (float) dgamma[c]; }
        Tensor dx = new Tensor(n, size);
        for (int i = 0; i < n; i++) for (int c = 0; c < channels; c++) {
            int base = i * size + c * hw;
            for (int p = 0; p < hw; p++) {
                double dxh = g.data[base + p] * gamma.value.data[c];
                dx.data[base + p] = usedBatchStats
                    ? (float) (invStd[c] / count * (count * dxh - gamma.value.data[c] * dbeta[c] - xhat.data[base + p] * gamma.value.data[c] * dgamma[c]))
                    : (float) (dxh * invStd[c]);
            }
        }
        return dx;
    }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != size) throw new IllegalArgumentException("BatchNorm2D expects " + size + " numbers per image but gets " + java.util.Arrays.toString(in));
        return in;
    }

    @Override public List<Param> params() { return List.of(gamma, beta); }
    @Override public List<Tensor> state() { return List.of(runMean, runVar); }
    @Override public String spec() { return "BatchNorm2D " + channels + " " + h + " " + w + " " + momentum; }
}
