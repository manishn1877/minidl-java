package minidl;

/**
 * Loss scaling for 16-bit training.
 *
 * Problem: in FP16 the smallest number is about 6e-8, and gradients are often smaller than that, so they silently become 0
 * and the weights stop learning. Fix: multiply the loss by a big number S before backprop, so every gradient is S times bigger
 * and survives, then divide by S again before updating the weights (in 32-bit, where small numbers are safe).
 *
 * Dynamic scaling picks S by itself: start big; if a gradient overflows (becomes infinity) skip that step and halve S;
 * after many good steps in a row, try doubling S again.
 */
public final class LossScaler {
    private double scale;
    private final boolean dynamic;
    private final int growthInterval;
    private int goodSteps = 0, skipped = 0;

    private LossScaler(double scale, boolean dynamic, int growthInterval) {
        this.scale = scale; this.dynamic = dynamic; this.growthInterval = growthInterval;
    }

    /** No scaling at all (S = 1). */
    public static LossScaler none() { return new LossScaler(1.0, false, 0); }
    /** Always use the same S. */
    public static LossScaler fixed(double s) { return new LossScaler(s, false, 0); }
    /** Start at 65536, halve on overflow, double after `growthInterval` good steps. */
    public static LossScaler dynamic() { return dynamic(65536.0, 200); }
    public static LossScaler dynamic(double initial, int growthInterval) { return new LossScaler(initial, true, growthInterval); }

    public double scale() { return scale; }
    public int skippedSteps() { return skipped; }

    /** Call after every step. `overflow` = a gradient became infinity or NaN. */
    public void update(boolean overflow) {
        if (overflow) {
            skipped++;
            if (dynamic) { scale = Math.max(1.0, scale / 2); goodSteps = 0; }
        } else if (dynamic && ++goodSteps >= growthInterval) {
            scale = Math.min(scale * 2, 16777216.0);
            goodSteps = 0;
        }
    }
}
