package minidl;

import java.util.List;

/**
 * Batch normalization for convolution outputs. Every CHANNEL is normalized on its own, using all the numbers that channel
 * has across the whole batch AND across every pixel position (so 32 images of 16 x 16 give 8,192 numbers per channel),
 * then rescaled with a learned gamma and beta per channel.
 * While predicting it uses running averages collected during training.
 * (BatchNorm, the other class, is for Dense outputs and normalizes every FEATURE instead.)
 */
public class BatchNorm2D implements Layer {
    private static final double EPS = 1e-5;

    private final int C, HW, size;
    private final double momentum;
    private final String shapeText;
    private final Param gamma, beta;
    private final Tensor runMean, runVar;
    private boolean training = true;

    private Tensor xhat;
    private double[] invStd;
    private boolean usedBatchStats;

    public BatchNorm2D(int channels, int height, int width) { this(channels, height, width, 0.9); }

    public BatchNorm2D(int channels, int height, int width, double momentum) {
        this.C = channels; this.HW = height * width; this.size = channels * height * width;
        this.momentum = momentum;
        this.shapeText = channels + " " + height + " " + width;
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
        int n = x.rows;
        long m = (long) n * HW;
        double[] mean = new double[C], var = new double[C];
        usedBatchStats = training && m > 1;
        if (usedBatchStats) {
            for (int c = 0; c < C; c++) {
                double s = 0;
                for (int i = 0; i < n; i++) { int base = i * size + c * HW; for (int p = 0; p < HW; p++) s += x.data[base + p]; }
                mean[c] = s / m;
                double v = 0;
                for (int i = 0; i < n; i++) { int base = i * size + c * HW; for (int p = 0; p < HW; p++) { double d = x.data[base + p] - mean[c]; v += d * d; } }
                var[c] = v / m;
                runMean.data[c] = (float) (momentum * runMean.data[c] + (1 - momentum) * mean[c]);
                runVar.data[c] = (float) (momentum * runVar.data[c] + (1 - momentum) * var[c] * m / (m - 1.0));
            }
        } else {
            for (int c = 0; c < C; c++) { mean[c] = runMean.data[c]; var[c] = runVar.data[c]; }
        }
        invStd = new double[C];
        for (int c = 0; c < C; c++) invStd[c] = 1.0 / Math.sqrt(var[c] + EPS);
        xhat = training ? new Tensor(n, size) : null;   // only backward needs it; when just predicting, do not keep a second copy
        Tensor out = new Tensor(n, size);
        for (int c = 0; c < C; c++) {
            final float mu = (float) mean[c], is = (float) invStd[c];
            final float a = gamma.value.data[c] * is, b = beta.value.data[c] - mu * a;   // out = x * a + b
            for (int i = 0; i < n; i++) {
                int base = i * size + c * HW;
                if (xhat != null) for (int p = 0; p < HW; p++) { float xv = x.data[base + p]; xhat.data[base + p] = (xv - mu) * is; out.data[base + p] = xv * a + b; }
                else for (int p = 0; p < HW; p++) out.data[base + p] = x.data[base + p] * a + b;
            }
        }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        int n = g.rows;
        long m = (long) n * HW;
        Tensor dx = new Tensor(n, size);
        for (int c = 0; c < C; c++) {
            double sg = 0, sgx = 0;
            for (int i = 0; i < n; i++) {
                int base = i * size + c * HW;
                for (int p = 0; p < HW; p++) { sg += g.data[base + p]; sgx += g.data[base + p] * xhat.data[base + p]; }
            }
            beta.grad.data[c] = (float) sg;
            gamma.grad.data[c] = (float) sgx;
            double gm = gamma.value.data[c];
            // dx = invStd * gm * g  -  invStd/m * (gm*sg)  -  invStd/m * (gm*sgx) * xhat      (all per channel constants)
            final float A = (float) (invStd[c] * gm);
            if (!usedBatchStats) { // fixed statistics: just a scale
                for (int i = 0; i < n; i++) { int base = i * size + c * HW; for (int p = 0; p < HW; p++) dx.data[base + p] = g.data[base + p] * A; }
                continue;
            }
            final float K0 = (float) (-invStd[c] / m * gm * sg), K1 = (float) (-invStd[c] / m * gm * sgx);
            for (int i = 0; i < n; i++) {
                int base = i * size + c * HW;
                for (int p = 0; p < HW; p++) dx.data[base + p] = A * g.data[base + p] + K1 * xhat.data[base + p] + K0;
            }
        }
        return dx;
    }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != size) throw new IllegalArgumentException("BatchNorm2D expects " + size + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return in;
    }

    @Override public List<Param> params() { return List.of(gamma, beta); }
    @Override public List<Tensor> state() { return List.of(runMean, runVar); }
    @Override public String spec() { return "BatchNorm2D " + shapeText + " " + momentum; }
}
