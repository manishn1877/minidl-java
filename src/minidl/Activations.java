package minidl;

/** Activation layers: they bend the straight lines so the network can learn curves. */
public final class Activations {
    private Activations() {}

    /**
     * ReLU. Backward only needs to know WHICH inputs were positive, so we keep one BIT per number
     * (a 32x smaller footprint than keeping the input) instead of the whole input tensor.
     */
    public static class ReLU implements Layer {
        private long[] mask = new long[0];

        @Override public Tensor forward(Tensor x) {
            int n = x.data.length, words = (n + 63) >>> 6;
            if (mask.length < words) mask = new long[words];
            Tensor out = new Tensor(x.rows, x.cols);
            for (int w = 0; w < words; w++) {
                int base = w << 6, end = Math.min(n, base + 64);
                long bits = 0;
                for (int i = base; i < end; i++) {
                    float v = x.data[i];
                    if (v > 0) { out.data[i] = v; bits |= 1L << (i - base); }
                }
                mask[w] = bits;
            }
            return out;
        }

        @Override public Tensor backward(Tensor g) {
            Tensor out = new Tensor(g.rows, g.cols);
            for (int i = 0; i < g.data.length; i++) if (((mask[i >>> 6] >>> (i & 63)) & 1L) != 0) out.data[i] = g.data[i];
            return out;
        }
    }

    /** Like ReLU, but negative numbers leak through a little, so neurons never fully "die". Also keeps just a 1-bit sign mask. */
    public static class LeakyReLU implements Layer {
        private final float slope;
        private long[] mask = new long[0];
        public LeakyReLU() { this(0.01); }
        public LeakyReLU(double slope) { this.slope = (float) slope; }

        @Override public Tensor forward(Tensor x) {
            int n = x.data.length, words = (n + 63) >>> 6;
            if (mask.length < words) mask = new long[words];
            Tensor out = new Tensor(x.rows, x.cols);
            for (int w = 0; w < words; w++) {
                int base = w << 6, end = Math.min(n, base + 64);
                long bits = 0;
                for (int i = base; i < end; i++) {
                    float v = x.data[i];
                    if (v > 0) { out.data[i] = v; bits |= 1L << (i - base); } else out.data[i] = slope * v;
                }
                mask[w] = bits;
            }
            return out;
        }

        @Override public Tensor backward(Tensor g) {
            Tensor out = new Tensor(g.rows, g.cols);
            for (int i = 0; i < g.data.length; i++) out.data[i] = ((mask[i >>> 6] >>> (i & 63)) & 1L) != 0 ? g.data[i] : slope * g.data[i];
            return out;
        }

        @Override public String spec() { return "LeakyReLU " + slope; }
    }

    public static class Sigmoid implements Layer {
        private Tensor out;
        @Override public Tensor forward(Tensor x) { out = x.map(v -> 1.0 / (1.0 + Math.exp(-v))); return out; }
        @Override public Tensor backward(Tensor g) {
            Tensor d = new Tensor(g.rows, g.cols);
            for (int i = 0; i < g.data.length; i++) d.data[i] = g.data[i] * out.data[i] * (1 - out.data[i]);
            return d;
        }
    }

    public static class Tanh implements Layer {
        private Tensor out;
        @Override public Tensor forward(Tensor x) { out = x.map(Math::tanh); return out; }
        @Override public Tensor backward(Tensor g) {
            Tensor d = new Tensor(g.rows, g.cols);
            for (int i = 0; i < g.data.length; i++) d.data[i] = g.data[i] * (1 - out.data[i] * out.data[i]);
            return d;
        }
    }

    /** GELU: a smooth ReLU. Used inside transformers (the family of models behind modern chatbots). */
    public static class GELU implements Layer {
        private static final double C = Math.sqrt(2.0 / Math.PI);
        private Tensor in;
        @Override public Tensor forward(Tensor x) {
            in = x;
            return x.map(v -> 0.5 * v * (1 + Math.tanh(C * (v + 0.044715 * v * v * v))));
        }
        @Override public Tensor backward(Tensor g) {
            Tensor d = new Tensor(g.rows, g.cols);
            for (int i = 0; i < g.data.length; i++) {
                double x = in.data[i];
                double t = Math.tanh(C * (x + 0.044715 * x * x * x));
                double slope = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C * (1 + 3 * 0.044715 * x * x);
                d.data[i] = (float) (g.data[i] * slope);
            }
            return d;
        }
    }
}
