package minidl;

import java.util.List;

/** Gradient clipping: if the total gradient is huge, shrink it. Stops one bad batch from wrecking the weights. */
public final class GradClip {
    private GradClip() {}

    /** Returns the gradient size BEFORE clipping. */
    public static double clipGlobalNorm(List<Param> params, double maxNorm) {
        double sum = 0;
        for (Param p : params) for (float g : p.grad.data) sum += (double) g * g;
        double norm = Math.sqrt(sum);
        if (norm > maxNorm && norm > 0) {
            double scale = maxNorm / norm;
            for (Param p : params) p.grad.scaleInPlace(scale);
        }
        return norm;
    }
}
