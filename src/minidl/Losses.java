package minidl;

public final class Losses {
    private Losses() {}

    /** Mean squared error: good for predicting numbers (regression). */
    public static class MSE implements Loss {
        private Tensor pred, target;
        @Override public double forward(Tensor p, Tensor t) {
            pred = p; target = t;
            double s = 0;
            for (int i = 0; i < p.data.length; i++) { double d = (double) p.data[i] - t.data[i]; s += d * d; }
            return s / p.data.length;
        }
        @Override public Tensor backward() {
            return pred.sub(target).scale(2.0 / pred.data.length);
        }
    }

    /** Mean absolute error: like MSE but big mistakes hurt less, so outliers disturb it less. */
    public static class MAE implements Loss {
        private Tensor pred, target;
        @Override public double forward(Tensor p, Tensor t) {
            pred = p; target = t;
            double s = 0;
            for (int i = 0; i < p.data.length; i++) s += Math.abs(p.data[i] - t.data[i]);
            return s / p.data.length;
        }
        @Override public Tensor backward() {
            Tensor g = new Tensor(pred.rows, pred.cols);
            double n = pred.data.length;
            for (int i = 0; i < g.data.length; i++) g.data[i] = (float) (Math.signum(pred.data[i] - target.data[i]) / n);
            return g;
        }
    }

    /**
     * Binary cross entropy that takes raw scores (logits). For yes/no questions (spam / not spam).
     * Target is 0 or 1 in a (batch x 1) tensor. Do NOT put a Sigmoid layer before it.
     * To get a probability afterwards: 1 / (1 + exp(-score)).
     */
    public static class BCEWithLogits implements Loss {
        private Tensor logits, target;
        @Override public double forward(Tensor z, Tensor t) {
            logits = z; target = t;
            double s = 0;
            for (int i = 0; i < z.data.length; i++) {
                double v = z.data[i];
                s += Math.max(v, 0) - v * t.data[i] + Math.log1p(Math.exp(-Math.abs(v))); // stable form
            }
            return s / z.data.length;
        }
        @Override public Tensor backward() {
            Tensor g = new Tensor(logits.rows, logits.cols);
            double n = logits.data.length;
            for (int i = 0; i < g.data.length; i++)
                g.data[i] = (float) ((1.0 / (1.0 + Math.exp(-logits.data[i])) - target.data[i]) / n);
            return g;
        }
    }

    /** Softmax + cross entropy: for picking 1 class out of many. Takes raw scores (logits). Target = one-hot. */
    public static class SoftmaxCrossEntropy implements Loss {
        private Tensor probs, target;
        @Override public double forward(Tensor logits, Tensor t) {
            target = t;
            probs = softmax(logits);
            double s = 0;
            for (int i = 0; i < probs.data.length; i++)
                if (t.data[i] > 0) s -= t.data[i] * Math.log(probs.data[i] + 1e-12);
            return s / logits.rows;
        }
        @Override public Tensor backward() {
            // the famous neat result: (softmax - target) / batch
            return probs.sub(target).scale(1.0 / probs.rows);
        }

        public static Tensor softmax(Tensor z) {
            Tensor out = new Tensor(z.rows, z.cols);
            for (int i = 0; i < z.rows; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < z.cols; j++) max = Math.max(max, z.get(i, j));
                double sum = 0;
                for (int j = 0; j < z.cols; j++) {
                    double e = Math.exp(z.get(i, j) - max); // minus max = no overflow
                    out.data[i * z.cols + j] = (float) e;
                    sum += e;
                }
                for (int j = 0; j < z.cols; j++) out.data[i * z.cols + j] = (float) (out.data[i * z.cols + j] / sum);
            }
            return out;
        }
    }
}
