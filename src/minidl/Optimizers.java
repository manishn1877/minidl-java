package minidl;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class Optimizers {
    private Optimizers() {}

    /** SGD (+ optional momentum, + optional L2 weight decay): weight -= lr * gradient. */
    public static class SGD implements Optimizer {
        private double lr;
        private final double momentum, weightDecay;
        private final Map<Param, float[]> velocity = new HashMap<>();

        public SGD(double lr) { this(lr, 0.0, 0.0); }
        public SGD(double lr, double momentum) { this(lr, momentum, 0.0); }
        public SGD(double lr, double momentum, double weightDecay) {
            this.lr = lr; this.momentum = momentum; this.weightDecay = weightDecay;
        }

        @Override public void setLearningRate(double lr) { this.lr = lr; }
        @Override public double getLearningRate() { return lr; }

        @Override public void step(List<Param> params) {
            float mom = (float) momentum, rate = (float) lr;
            for (Param p : params) {
                float[] v = velocity.computeIfAbsent(p, k -> new float[p.value.data.length]);
                float wd = p.decay ? (float) weightDecay : 0f;
                for (int i = 0; i < v.length; i++) {
                    float g = p.grad.data[i] + wd * p.value.data[i]; // L2: pull weights toward 0
                    v[i] = mom * v[i] - rate * g;
                    p.value.data[i] += v[i];
                }
            }
        }
    }

    /**
     * Adam: gives every weight its own adaptive step size. Usually trains faster.
     * If weightDecay > 0 this is AdamW (the decay is applied directly to the weight, not mixed into the gradient).
     */
    public static class Adam implements Optimizer {
        private double lr;
        private final double beta1, beta2, eps, weightDecay;
        private final Map<Param, float[][]> state = new HashMap<>(); // [m, v]  (32-bit, half the memory of 64-bit)
        private int t = 0;

        public Adam(double lr) { this(lr, 0.9, 0.999, 1e-8, 0.0); }
        public Adam(double lr, double weightDecay) { this(lr, 0.9, 0.999, 1e-8, weightDecay); }
        public Adam(double lr, double beta1, double beta2, double eps, double weightDecay) {
            this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps; this.weightDecay = weightDecay;
        }

        @Override public void setLearningRate(double lr) { this.lr = lr; }
        @Override public double getLearningRate() { return lr; }

        @Override public void step(List<Param> params) {
            t++;
            float c1 = (float) (1 - Math.pow(beta1, t)), c2 = (float) (1 - Math.pow(beta2, t)); // fix early bias toward 0
            float b1 = (float) beta1, b2 = (float) beta2, rate = (float) lr, e = (float) eps;
            for (Param p : params) {
                int n = p.value.data.length;
                float[][] s = state.computeIfAbsent(p, k -> new float[][]{new float[n], new float[n]});
                float[] m = s[0], v = s[1];
                float decay = p.decay ? (float) (lr * weightDecay) : 0f;
                for (int i = 0; i < n; i++) {
                    float g = p.grad.data[i];
                    m[i] = b1 * m[i] + (1 - b1) * g;
                    v[i] = b2 * v[i] + (1 - b2) * g * g;
                    p.value.data[i] -= rate * (m[i] / c1) / ((float) Math.sqrt(v[i] / c2) + e) + decay * p.value.data[i];
                }
            }
        }
    }

    /** RMSProp: divides each step by a running average of recent gradient sizes. */
    public static class RMSProp implements Optimizer {
        private double lr;
        private final double rho, eps;
        private final Map<Param, float[]> cache = new HashMap<>();

        public RMSProp(double lr) { this(lr, 0.9, 1e-8); }
        public RMSProp(double lr, double rho, double eps) { this.lr = lr; this.rho = rho; this.eps = eps; }

        @Override public void setLearningRate(double lr) { this.lr = lr; }
        @Override public double getLearningRate() { return lr; }

        @Override public void step(List<Param> params) {
            float r = (float) rho, rate = (float) lr, e = (float) eps;
            for (Param p : params) {
                float[] c = cache.computeIfAbsent(p, k -> new float[p.value.data.length]);
                for (int i = 0; i < c.length; i++) {
                    float g = p.grad.data[i];
                    c[i] = r * c[i] + (1 - r) * g * g;
                    p.value.data[i] -= rate * g / ((float) Math.sqrt(c[i]) + e);
                }
            }
        }
    }
}
