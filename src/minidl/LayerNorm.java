package minidl;

import java.util.List;

/**
 * Layer normalization: every ROW is rescaled to mean 0 and spread 1 across its own features,
 * then re-scaled with learned gamma / beta.
 * Unlike BatchNorm it does not look at other rows, so it behaves the same for a batch of 1 and
 * in training or prediction. This is the normalization used inside transformers.
 */
public class LayerNorm implements Layer {
    private static final double EPS = 1e-5;
    private final int features;
    private final Param gamma, beta;
    private Tensor xhat;
    private double[] invStd;

    public LayerNorm(int features) {
        this.features = features;
        Tensor g = new Tensor(1, features);
        java.util.Arrays.fill(g.data, 1f);
        gamma = new Param(g, false);
        beta = new Param(new Tensor(1, features), false);
    }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != features) throw new IllegalArgumentException("LayerNorm expects " + features + " columns, got " + x.cols);
        int n = x.rows, f = features;
        xhat = new Tensor(n, f);
        invStd = new double[n];
        Tensor out = new Tensor(n, f);
        for (int i = 0; i < n; i++) {
            double mean = 0;
            for (int j = 0; j < f; j++) mean += x.data[i * f + j];
            mean /= f;
            double var = 0;
            for (int j = 0; j < f; j++) { double d = x.data[i * f + j] - mean; var += d * d; }
            var /= f;
            invStd[i] = 1.0 / Math.sqrt(var + EPS);
            for (int j = 0; j < f; j++) {
                float h = (float) ((x.data[i * f + j] - mean) * invStd[i]);
                xhat.data[i * f + j] = h;
                out.data[i * f + j] = gamma.value.data[j] * h + beta.value.data[j];
            }
        }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        int n = g.rows, f = features;
        double[] dgamma = new double[f], dbeta = new double[f];
        Tensor dx = new Tensor(n, f);
        for (int i = 0; i < n; i++) {
            double s1 = 0, s2 = 0;
            for (int j = 0; j < f; j++) {
                double gj = g.data[i * f + j], xh = xhat.data[i * f + j];
                dbeta[j] += gj;
                dgamma[j] += gj * xh;
                double dxh = gj * gamma.value.data[j];
                s1 += dxh;
                s2 += dxh * xh;
            }
            for (int j = 0; j < f; j++) {
                double dxh = g.data[i * f + j] * gamma.value.data[j];
                dx.data[i * f + j] = (float) (invStd[i] / f * (f * dxh - s1 - xhat.data[i * f + j] * s2));
            }
        }
        for (int j = 0; j < f; j++) { gamma.grad.data[j] = (float) dgamma[j]; beta.grad.data[j] = (float) dbeta[j]; }
        return dx;
    }

    @Override public int[] outputShape(int[] in) {
        if (in.length == 0 || in[in.length - 1] != features) throw new IllegalArgumentException("LayerNorm works on the last dimension, which must be " + features + ", but the shape is " + java.util.Arrays.toString(in));
        return in;
    }

    @Override public List<Param> params() { return List.of(gamma, beta); }
    @Override public String spec() { return "LayerNorm " + features; }
}
