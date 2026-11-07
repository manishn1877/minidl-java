package minidl;

import java.util.Random;

/**
 * Data augmentation: every time a batch is requested, each image is randomly shifted and
 * slightly rotated. The model never sees exactly the same picture twice, so it cannot just
 * memorize the training set and has to learn what a "7" really looks like.
 *
 * Costs almost no memory: the original bytes stay untouched, only the current batch is changed.
 * Use it for TRAINING data only. Validation and test data stay unchanged.
 */
public class AugmentedSource implements DataSource {
    private final DataSource base;
    private final int c, h, w, maxShift;
    private final boolean flip;
    private final double maxRotRad;
    private final Random rnd;

    /** maxShift in pixels, maxRotDegrees in degrees. Images are h x w with 1 channel. */
    public AugmentedSource(DataSource base, int h, int w, int maxShift, double maxRotDegrees, long seed) {
        this(base, 1, h, w, maxShift, maxRotDegrees, false, seed);
    }

    /**
     * Colour version: every image has `channels` planes of h x w, stored as [channel][y][x].
     * All channels of one image move together. flip = also mirror left-right half of the time
     * (fine for photos of objects, wrong for digits and text).
     */
    public AugmentedSource(DataSource base, int channels, int h, int w, int maxShift, double maxRotDegrees, boolean flip, long seed) {
        this.base = base; this.c = channels; this.h = h; this.w = w; this.maxShift = maxShift; this.flip = flip;
        this.maxRotRad = Math.toRadians(maxRotDegrees);
        this.rnd = new Random(seed);
    }

    @Override public int size() { return base.size(); }

    @Override public Tensor features(int[] idx, int from, int to) {
        Tensor x = base.features(idx, from, to);
        Tensor out = new Tensor(x.rows, x.cols);
        for (int i = 0; i < x.rows; i++) {
            double angle = (rnd.nextDouble() * 2 - 1) * maxRotRad;
            double dx = maxShift == 0 ? 0 : rnd.nextInt(2 * maxShift + 1) - maxShift;
            double dy = maxShift == 0 ? 0 : rnd.nextInt(2 * maxShift + 1) - maxShift;
            boolean mirror = flip && rnd.nextBoolean();
            for (int ch = 0; ch < c; ch++) {
                int off = i * x.cols + ch * h * w;
                if (mirror) mirrorRows(x.data, off, h, w);
                transform(x.data, off, out.data, off, h, w, angle, dx, dy);
            }
        }
        return out;
    }

    @Override public Tensor targets(int[] idx, int from, int to) { return base.targets(idx, from, to); }

    /** Mirrors one h x w plane left-right, in place. */
    static void mirrorRows(float[] a, int off, int h, int w) {
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w / 2; x++) {
                int p = off + y * w + x, q = off + y * w + (w - 1 - x);
                float t = a[p]; a[p] = a[q]; a[q] = t;
            }
    }

    /**
     * Rotates an h x w image by `angle` (radians) around its centre, then moves it by (dx, dy) pixels.
     * Works backwards: for every NEW pixel, find where it came from in the old image and blend the 4
     * nearest old pixels (bilinear). Outside the picture counts as 0 (black).
     */
    public static void transform(float[] src, int so, float[] dst, int dOff, int h, int w, double angle, double dx, double dy) {
        double cy = (h - 1) / 2.0, cx = (w - 1) / 2.0;
        double cos = Math.cos(angle), sin = Math.sin(angle);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                double py = y - dy - cy, px = x - dx - cx;
                double sy = cos * py + sin * px + cy;   // inverse rotation
                double sx = -sin * py + cos * px + cx;
                int y0 = (int) Math.floor(sy), x0 = (int) Math.floor(sx);
                double fy = sy - y0, fx = sx - x0;
                double v = 0;
                for (int oy = 0; oy <= 1; oy++)
                    for (int ox = 0; ox <= 1; ox++) {
                        int yy = y0 + oy, xx = x0 + ox;
                        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                        v += src[so + yy * w + xx] * (oy == 0 ? 1 - fy : fy) * (ox == 0 ? 1 - fx : fx);
                    }
                dst[dOff + y * w + x] = (float) v;
            }
    }
}
