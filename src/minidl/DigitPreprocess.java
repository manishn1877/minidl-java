package minidl;

import java.util.ArrayList;
import java.util.List;

/**
 * Turns a drawing (any size) into what the MNIST model expects, the same way MNIST was made:
 * crop to the ink, scale the longer side to 20 px (keeping the shape), then slide it so its
 * "centre of mass" sits in the middle of a 28 x 28 picture. Without this step a digit drawn
 * small, big or off-centre is read badly, even by a 99% model.
 *
 * Input:  ink[] with values 0..1, where 1 = ink (so black pen on white paper is flipped first).
 */
public final class DigitPreprocess {
    private DigitPreprocess() {}

    public static final int SIZE = 28, BOX = 20;

    /**
     * MNIST digits: longer side 20 px, centred by centre of mass.
     * EMNIST letters: the letter fills a 24 px box, centred by its bounding box (this follows the EMNIST paper's
     * description; compare "what the model sees" with real EMNIST pictures if letters are read badly).
     */
    public enum Style { MNIST, EMNIST }
    static final float INK = 0.1f; // below this a pixel counts as empty paper

    /** One digit found in a drawing: where it is, and its 28 x 28 pixels (784 floats, 0..1). */
    public static final class Piece {
        public final int x, y, w, h;
        public final float[] pixels;
        Piece(int x, int y, int w, int h, float[] pixels) { this.x = x; this.y = y; this.w = w; this.h = h; this.pixels = pixels; }
    }

    /** The whole drawing treated as ONE digit. Null if the page is blank. */
    public static Piece whole(float[] ink, int w, int h) { return whole(ink, w, h, Style.MNIST); }

    public static Piece whole(float[] ink, int w, int h, Style st) {
        int[] b = bbox(ink, w, h, 0, w);
        return b == null ? null : piece(ink, w, b, st);
    }

    /** Same as whole(), but only the 784 pixels. */
    public static float[] toMnist(float[] ink, int w, int h) {
        Piece p = whole(ink, w, h);
        return p == null ? null : p.pixels;
    }

    /**
     * Several digits side by side ("2024"): columns with ink form groups, and a gap of empty columns
     * starts a new digit. Returns the digits left to right. Specks of noise are ignored.
     * (Limit: digits that touch or overlap in the same columns are read as one.)
     */
    public static List<Piece> segment(float[] ink, int w, int h) { return segment(ink, w, h, Style.MNIST); }

    public static List<Piece> segment(float[] ink, int w, int h, Style st) {
        boolean[] col = new boolean[w];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) if (ink[y * w + x] > INK) col[x] = true;
        int gap = Math.max(3, w / 40);
        List<int[]> runs = new ArrayList<>();
        int start = -1, last = -1;
        for (int x = 0; x < w; x++) {
            if (!col[x]) continue;
            if (start >= 0 && x - last > gap) { runs.add(new int[]{start, last}); start = x; }
            else if (start < 0) start = x;
            last = x;
        }
        if (start >= 0) runs.add(new int[]{start, last});
        List<Piece> out = new ArrayList<>();
        int speck = Math.max(4, Math.min(w, h) / 25);
        for (int[] r : runs) {
            int[] b = bbox(ink, w, h, r[0], r[1] + 1);
            if (b == null || (b[2] - b[0] + 1 < speck && b[3] - b[1] + 1 < speck)) continue;
            out.add(piece(ink, w, b, st));
        }
        return out;
    }

    /** {minX, minY, maxX, maxY} of the ink inside columns [x0, x1), or null when there is none. */
    static int[] bbox(float[] ink, int w, int h, int x0, int x1) {
        int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE, maxX = -1, maxY = -1;
        for (int y = 0; y < h; y++)
            for (int x = x0; x < x1; x++)
                if (ink[y * w + x] > INK) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
        return maxX < 0 ? null : new int[]{minX, minY, maxX, maxY};
    }

    private static Piece piece(float[] ink, int w, int[] b, Style st) {
        return new Piece(b[0], b[1], b[2] - b[0] + 1, b[3] - b[1] + 1, normalize(ink, w, b, st));
    }

    static float[] normalize(float[] ink, int w, int[] b, Style st) {
        final int BOX = st == Style.EMNIST ? 24 : DigitPreprocess.BOX;
        int bw = b[2] - b[0] + 1, bh = b[3] - b[1] + 1;
        double s = (double) BOX / Math.max(bw, bh);
        int nw = Math.max(1, (int) Math.round(bw * s)), nh = Math.max(1, (int) Math.round(bh * s));
        float[] small = new float[nw * nh];
        float peak = 0;
        for (int dy = 0; dy < nh; dy++) {
            int sy0 = b[1] + (int) Math.floor(dy * (double) bh / nh);
            int sy1 = Math.max(sy0 + 1, b[1] + (int) Math.ceil((dy + 1) * (double) bh / nh));
            for (int dx = 0; dx < nw; dx++) {
                int sx0 = b[0] + (int) Math.floor(dx * (double) bw / nw);
                int sx1 = Math.max(sx0 + 1, b[0] + (int) Math.ceil((dx + 1) * (double) bw / nw));
                double sum = 0; int n = 0;
                for (int y = sy0; y < sy1 && y <= b[3]; y++) for (int x = sx0; x < sx1 && x <= b[2]; x++) { sum += Math.min(1f, ink[y * w + x]); n++; }
                float v = n == 0 ? 0 : (float) (sum / n);
                small[dy * nw + dx] = v;
                peak = Math.max(peak, v);
            }
        }
        if (peak > 0) for (int i = 0; i < small.length; i++) small[i] /= peak; // darkest stroke = full ink, like MNIST
        int ox = (SIZE - nw) / 2, oy = (SIZE - nh) / 2;
        double mass = 0, cx = 0, cy = 0;
        for (int j = 0; j < nh; j++) for (int i = 0; i < nw; i++) { float v = small[j * nw + i]; mass += v; cx += v * (ox + i); cy += v * (oy + j); }
        int shiftX = 0, shiftY = 0;
        if (mass > 0 && st == Style.MNIST) { shiftX = (int) Math.round((SIZE - 1) / 2.0 - cx / mass); shiftY = (int) Math.round((SIZE - 1) / 2.0 - cy / mass); }
        float[] out = new float[SIZE * SIZE];
        for (int j = 0; j < nh; j++)
            for (int i = 0; i < nw; i++) {
                int x = ox + i + shiftX, y = oy + j + shiftY;
                if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) out[y * SIZE + x] = small[j * nw + i];
            }
        return out;
    }
}
