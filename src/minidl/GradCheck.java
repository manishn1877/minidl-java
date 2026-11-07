package minidl;

/**
 * Proof that backprop is right: compare our gradient with a "nudge and measure" gradient.
 * (f(w+eps) - f(w-eps)) / 2eps
 *
 * The library computes in 32-bit floats, which carry about 7 digits. A nudge of 1e-5 would be
 * lost in rounding, so the nudge here is bigger (3e-3). Expect errors around 1e-4 to 1e-3 for
 * correct code; a real bug gives errors of 0.1 or more. (The earlier 64-bit version of this
 * library reached 1e-8 with the same formulas.)
 */
public final class GradCheck {
    private GradCheck() {}

    public static double maxRelativeError(Sequential model, Loss loss, Tensor x, Tensor y) {
        for (Param p : model.params()) p.zeroGrad();
        loss.forward(model.forward(x), y);
        model.backward(loss.backward());

        double eps = 3e-3, worst = 0;
        for (Param p : model.params()) {
            for (int i = 0; i < p.value.data.length; i++) {
                double analytic = p.grad.data[i];
                float orig = p.value.data[i];
                p.value.data[i] = orig + (float) eps;
                double up = loss.forward(model.forward(x), y);
                p.value.data[i] = orig - (float) eps;
                double down = loss.forward(model.forward(x), y);
                p.value.data[i] = orig;
                double numeric = (up - down) / (2 * eps);
                // relative error for big gradients, absolute error for tiny ones (a gradient that is truly 0,
                // like a Dense bias right before BatchNorm, would otherwise divide noise by noise)
                double rel = Math.abs(analytic - numeric) / Math.max(1e-2, Math.abs(analytic) + Math.abs(numeric));
                worst = Math.max(worst, rel);
            }
        }
        return worst;
    }
}
