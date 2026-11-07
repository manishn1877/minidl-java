package minidl;

import java.util.List;

/**
 * Batch normalization (for Dense outputs).
 * While training: rescales each feature using the average and spread of the CURRENT batch,
 * then lets the model re-shift it with two learned numbers (gamma, beta).
 * While predicting: uses running averages collected during training.
 * Makes deep networks train faster and more steadily.
 */
public class BatchNorm implements Layer {
    private static final double EPS = 1e-5;

    private final int features;
    private final double momentum;
    private final Param gamma, beta;
    private final Tensor runMean, runVar;
    private boolean training = true;

    // remembered for backward
    private Tensor xhat;
    private double[] invStd;
    private boolean usedBatchStats;

    public BatchNorm(int features) { this(features, 0.9); }

    public BatchNorm(int features, double momentum) {
        this.features = features;
        this.momentum = momentum;
        Tensor g = new Tensor(1, features);
        java.util.Arrays.fill(g.data, 1f);
        gamma = new Param(g, false);
        beta = new Param(new Tensor(1, features), false);
        runMean = new Tensor(1, features);
        runVar = new Tensor(1, features);
        java.util.Arrays.fill(runVar.data, 1f);
    }

    @Override public void setTraining(boolean training) { this.training = training; }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != features) throw new IllegalArgumentException("BatchNorm expects " + features + " columns, got " + x.cols);
        int n = x.rows, f = features;
        double[] mean = new double[f], var = new double[f];
        usedBatchStats = training && n > 1;

        if (usedBatchStats) {
            for (int i = 0; i < n; i++) for (int j = 0; j < f; j++) mean[j] += x.data[i * f + j];
            for (int j = 0; j < f; j++) mean[j] /= n;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < f; j++) { double d = x.data[i * f + j] - mean[j]; var[j] += d * d; }
            for (int j = 0; j < f; j++) var[j] /= n;
            for (int j = 0; j < f; j++) {
                runMean.data[j] = (float) (momentum * runMean.data[j] + (1 - momentum) * mean[j]);
                runVar.data[j] = (float) (momentum * runVar.data[j] + (1 - momentum) * var[j] * n / (n - 1.0));
            }
        } else {
            for (int j = 0; j < f; j++) { mean[j] = runMean.data[j]; var[j] = runVar.data[j]; }
        }

        invStd = new double[f];
        for (int j = 0; j < f; j++) invStd[j] = 1.0 / Math.sqrt(var[j] + EPS);
        xhat = training ? new Tensor(n, f) : null;      // only backward needs it
        Tensor out = new Tensor(n, f);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < f; j++) {
                float h = (float) ((x.data[i * f + j] - mean[j]) * invStd[j]);
                if (xhat != null) xhat.data[i * f + j] = h;
                out.data[i * f + j] = gamma.value.data[j] * h + beta.value.data[j];
            }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        int n = g.rows, f = features;
        double[] dgamma = new double[f], dbeta = new double[f];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < f; j++) {
                dbeta[j] += g.data[i * f + j];
                dgamma[j] += g.data[i * f + j] * xhat.data[i * f + j];
            }
        for (int j = 0; j < f; j++) { beta.grad.data[j] = (float) dbeta[j]; gamma.grad.data[j] = (float) dgamma[j]; }

        Tensor dx = new Tensor(n, f);
        if (!usedBatchStats) { // statistics were fixed numbers, so it is just a scale
            for (int i = 0; i < n; i++)
                for (int j = 0; j < f; j++)
                    dx.data[i * f + j] = (float) (g.data[i * f + j] * gamma.value.data[j] * invStd[j]);
            return dx;
        }
        double[] s1 = new double[f], s2 = new double[f]; // sum(dxhat), sum(dxhat * xhat)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < f; j++) {
                double dxh = g.data[i * f + j] * gamma.value.data[j];
                s1[j] += dxh;
                s2[j] += dxh * xhat.data[i * f + j];
            }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < f; j++) {
                double dxh = g.data[i * f + j] * gamma.value.data[j];
                dx.data[i * f + j] = (float) (invStd[j] / n * (n * dxh - s1[j] - xhat.data[i * f + j] * s2[j]));
            }
        return dx;
    }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != features) throw new IllegalArgumentException("BatchNorm expects " + features + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return in;
    }

    @Override public List<Param> params() { return List.of(gamma, beta); }
    @Override public List<Tensor> state() { return List.of(runMean, runVar); }
    @Override public String spec() { return "BatchNorm " + features + " " + momentum; }
}
