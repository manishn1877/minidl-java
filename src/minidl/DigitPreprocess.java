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
    /** A shape is only cut when the model is LESS sure than this about it as a whole (unless it is far too wide for one character). */
    /** A cut must go through a real pinch: the ink in the cut column is at most this fraction of the smaller peak on its two sides. */
    static final double VALLEY = Double.parseDouble(System.getProperty("minidl.valley", "0.6"));
    static final double WIDE_DIGIT = Double.parseDouble(System.getProperty("minidl.wide", "1.3"));
    static final double SPLIT_GATE = Double.parseDouble(System.getProperty("minidl.splitGate", "0.95"));

    /** One digit found in a drawing: where it is, and its 28 x 28 pixels (784 floats, 0..1). */
    public static final class Piece {
        public final int x, y, w, h;
        public final float[] pixels;
        /** Why this piece is not a plain blob (e.g. "split a joined shape ..."), or null. Shown as a warning in the page. */
        public final String note;
        Piece(int x, int y, int w, int h, float[] pixels) { this(x, y, w, h, pixels, null); }
        Piece(int x, int y, int w, int h, float[] pixels, String note) { this.x = x; this.y = y; this.w = w; this.h = h; this.pixels = pixels; this.note = note; }
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

    /** Asks the model how sure it is (0..1) that a 28 x 28 picture is ONE character. Used to check uncertain splits and merges. */
    public interface Scorer { double confidence(float[] pixels); }

    public static List<Piece> segment(float[] ink, int w, int h) { return segment(ink, w, h, Style.MNIST, null); }

    public static List<Piece> segment(float[] ink, int w, int h, Style st) { return segment(ink, w, h, st, null); }

    /**
     * Several characters side by side ("2310", "HELLO"), found WITHOUT needing a wide gap:
     *  1. connected components: every separate blob of ink is a candidate (so "1" and "0" are two blobs even
     *     when only 2 px apart, and even when slanted so that their columns overlap)
     *  2. small blobs stacked on a bigger one (the dot of an i, a loose top bar) join it; specks of dust are dropped
     *  3. each character gets its own mask, so a neighbour's ink inside its bounding box is never copied into it
     *  4. with a scorer (the model), two safeguards: a SHAPE THAT IS TOO WIDE for one character is cut at the best
     *     valley of the column profile if both halves are read with high confidence (touching "10"), and two
     *     neighbouring blobs are joined if together they are read much better than apart (an "4" drawn with 2 strokes)
     * Returns the characters left to right. Piece.note says when step 4 changed something.
     * Limits: characters that touch AND are written slanted into each other may be cut in the wrong place.
     */
    public static List<Piece> segment(float[] ink, int w, int h, Style st, Scorer sc) {
        int[] lab = new int[w * h], stack = new int[w * h];
        List<int[]> comps = new ArrayList<>(); // {minX, minY, maxX, maxY, area}, id = index + 1
        for (int s0 = 0; s0 < w * h; s0++) {
            if (ink[s0] <= INK || lab[s0] != 0) continue;
            int id = comps.size() + 1, sp = 0;
            int[] c = {w, h, -1, -1, 0};
            stack[sp++] = s0; lab[s0] = id;
            while (sp > 0) {
                int pnt = stack[--sp], x = pnt % w, y = pnt / w;
                c[0] = Math.min(c[0], x); c[1] = Math.min(c[1], y); c[2] = Math.max(c[2], x); c[3] = Math.max(c[3], y); c[4]++;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    int q = ny * w + nx;
                    if (ink[q] > INK && lab[q] == 0) { lab[q] = id; stack[sp++] = q; }
                }
            }
            comps.add(c);
        }
        int speck = Math.max(4, Math.min(w, h) / 25);
        List<Group> groups = new ArrayList<>();
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < comps.size(); i++) {
            int[] c = comps.get(i);
            if ((c[2] - c[0] + 1 < speck && c[3] - c[1] + 1 < speck) || c[4] < 3) continue; // dust
            order.add(i);
        }
        order.sort((p1, p2) -> Integer.compare(comps.get(p1)[0], comps.get(p2)[0]));
        for (int i : order) {
            int[] c = comps.get(i);
            Group home = null;
            for (Group g : groups) if (g.joins(c)) { home = g; break; }
            if (home == null) { home = new Group(); groups.add(home); }
            home.add(i + 1, c);
        }
        groups.sort((g1, g2) -> Integer.compare(g1.minX, g2.minX));
        if (groups.isEmpty()) return new ArrayList<>();
        int[] hs = new int[groups.size()];
        for (int i = 0; i < hs.length; i++) hs[i] = groups.get(i).maxY - groups.get(i).minY + 1;
        java.util.Arrays.sort(hs);
        int refH = hs[hs.length / 2]; // typical character height on this page

        if (sc != null) { // join blobs that are really one character drawn in several strokes
            for (int i = 0; i + 1 < groups.size(); i++) {
                Group a = groups.get(i), b = groups.get(i + 1);
                int gap = b.minX - a.maxX - 1;
                if (gap > 0.2 * refH || Math.max(a.maxX, b.maxX) - Math.min(a.minX, b.minX) + 1 > 0.95 * refH) continue;
                Group u = new Group(); u.addAll(a); u.addAll(b);
                double ca = sc.confidence(normalize(mask(ink, lab, w, a, comps.size()), a.w(), a.tight(), st)), cb = sc.confidence(normalize(mask(ink, lab, w, b, comps.size()), b.w(), b.tight(), st));
                double cu = sc.confidence(normalize(mask(ink, lab, w, u, comps.size()), u.w(), u.tight(), st));
                if (cu >= 0.9 && cu > (ca + cb) / 2 + 0.15 && Math.min(ca, cb) < 0.8) { groups.set(i, u); groups.remove(i + 1); i--; }
            }
        }
        List<Piece> out = new ArrayList<>();
        for (Group g : groups) emit(mask(ink, lab, w, g, comps.size()), g.w(), g.maxY - g.minY + 1, g.minX, g.minY, refH, st, sc, 0, out, null);
        return out;
    }

    /** Blobs that belong to one character. */
    private static final class Group {
        int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE, maxX = -1, maxY = -1;
        final List<Integer> ids = new ArrayList<>();
        void add(int id, int[] c) { ids.add(id); minX = Math.min(minX, c[0]); minY = Math.min(minY, c[1]); maxX = Math.max(maxX, c[2]); maxY = Math.max(maxY, c[3]); }
        void addAll(Group o) { ids.addAll(o.ids); minX = Math.min(minX, o.minX); minY = Math.min(minY, o.minY); maxX = Math.max(maxX, o.maxX); maxY = Math.max(maxY, o.maxY); }
        int w() { return maxX - minX + 1; }
        int[] tight() { return new int[]{0, 0, w() - 1, maxY - minY}; }
        /** A blob joins this group when it sits ON TOP of it (dot, loose bar), or hides almost completely behind its columns. */
        boolean joins(int[] c) {
            int ov = Math.min(maxX, c[2]) - Math.max(minX, c[0]) + 1;
            if (ov <= 0) return false;
            int narrow = Math.min(maxX - minX + 1, c[2] - c[0] + 1);
            int vOv = Math.min(maxY, c[3]) - Math.max(minY, c[1]) + 1, small = Math.min(maxY - minY + 1, c[3] - c[1] + 1);
            boolean stacked = vOv < 0.3 * small;
            return stacked ? ov >= 0.5 * narrow : ov >= 0.9 * narrow;
        }
    }

    /** The group's ink only (other blobs inside its bounding box are left out), as a picture of the group's bounding box. */
    private static float[] mask(float[] ink, int[] lab, int w, Group g, int nComps) {
        boolean[] mine = new boolean[nComps + 1];
        for (int id : g.ids) mine[id] = true;
        int gw = g.w(), gh = g.maxY - g.minY + 1;
        float[] sub = new float[gw * gh];
        for (int y = 0; y < gh; y++) for (int x = 0; x < gw; x++) { int p = (g.minY + y) * w + g.minX + x; if (mine[lab[p]]) sub[y * gw + x] = ink[p]; }
        return sub;
    }

    /** Adds one character; a shape that is much wider than a character is checked for a touching pair and cut if the model agrees. */
    private static void emit(float[] sub, int gw, int gh, int ox, int oy, int refH, Style st, Scorer sc, int depth, List<Piece> out, String note) {
        float[] px = normalize(sub, gw, new int[]{0, 0, gw - 1, gh - 1}, st);
        double aspect = (double) gw / Math.max(gh, refH);
        // Only a shape that is as wide as it is tall AND that the model is not sure about is a candidate. (A model is often
        // 100% sure about meaningless halves: half of a "0" looks like a "1". So "both halves are confident" alone proves nothing.)
        // A shape much wider than any single character (digits 1.3 x its height, letters 1.7 x: W and M are wide) is cut even when the model is sure.
        final double wideAspect = st == Style.MNIST ? WIDE_DIGIT : 1.7;
        boolean veryWide = aspect >= wideAspect;
        double whole = sc != null && depth < 6 && aspect >= 0.95 ? sc.confidence(px) : 1.0;
        if (sc != null && depth < 6 && aspect >= 0.95 && (veryWide || whole < SPLIT_GATE)) {
            int[] col = new int[gw];
            for (int y = 0; y < gh; y++) for (int x = 0; x < gw; x++) if (sub[y * gw + x] > INK) col[x]++;
            List<Integer> cand = new ArrayList<>();
            for (int c = Math.max(3, (int) Math.ceil(0.04 * gw)); c <= Math.min(gw - 3, (int) (0.96 * gw)); c++) if (col[c] <= col[c - 1] && col[c] <= col[c + 1]) cand.add(c);
            final int mid = gw / 2;
            cand.sort((p1, p2) -> col[p1] != col[p2] ? Integer.compare(col[p1], col[p2]) : Integer.compare(Math.abs(p1 - mid), Math.abs(p2 - mid)));
            List<Integer> tried = new ArrayList<>();
            double bestAvg = -1; int bestCut = -1; double bestL = 0, bestR = 0;
            for (int c : cand) {
                boolean near = false;
                for (int t : tried) if (Math.abs(t - c) < Math.max(2, 0.12 * gw)) near = true;
                if (near || tried.size() >= 4) continue;
                tried.add(c);
                int peakL = 0, peakR = 0;
                for (int x = 0; x < c; x++) peakL = Math.max(peakL, col[x]);
                for (int x = c; x < gw; x++) peakR = Math.max(peakR, col[x]);
                if (col[c] > VALLEY * Math.min(peakL, peakR)) continue; // no pinch here: not a place where two characters meet
                Crop l = cropCols(sub, gw, gh, 0, c), r = cropCols(sub, gw, gh, c, gw);
                if (l == null || r == null) continue;
                double cl = sc.confidence(normalize(l.px, l.w, new int[]{0, 0, l.w - 1, l.h - 1}, st));
                double cr = sc.confidence(normalize(r.px, r.w, new int[]{0, 0, r.w - 1, r.h - 1}, st));
                // a part that is itself wide may hold several characters: it is cut again below, so it does not need to be confident now
                double el = l.w >= 0.95 * Math.max(l.h, refH) ? Math.max(cl, 0.9) : cl, er = r.w >= 0.95 * Math.max(r.h, refH) ? Math.max(cr, 0.9) : cr;
                double avg = (el + er) / 2;
                if (Math.min(l.w, r.w) >= Math.max(4, 0.06 * refH) && Math.min(el, er) >= 0.75 && avg >= 0.9 && (veryWide || avg >= whole + 0.1) && avg > bestAvg) { bestAvg = avg; bestCut = c; bestL = cl; bestR = cr; }
            }
            if (bestCut > 0) {
                String n = String.format("split a joined shape (whole %.2f, parts %.2f and %.2f)", whole, bestL, bestR);
                Crop l = cropCols(sub, gw, gh, 0, bestCut), r = cropCols(sub, gw, gh, bestCut, gw);
                emit(l.px, l.w, l.h, ox + l.x, oy + l.y, refH, st, sc, depth + 1, out, n);
                emit(r.px, r.w, r.h, ox + r.x, oy + r.y, refH, st, sc, depth + 1, out, n);
                return;
            }
        }
        out.add(new Piece(ox, oy, gw, gh, px, note));
    }

    /** A cut-out piece of a picture: its pixels, size, and where its top-left corner was. */
    private static final class Crop { float[] px; int w, h, x, y; }

    /** Columns [x0, x1) of sub, trimmed to their ink (tight box), or null if empty. */
    private static Crop cropCols(float[] sub, int gw, int gh, int x0, int x1) {
        int[] b = bbox(sub, gw, gh, x0, x1);
        if (b == null) return null;
        Crop c = new Crop();
        c.w = b[2] - b[0] + 1; c.h = b[3] - b[1] + 1; c.x = b[0]; c.y = b[1];
        c.px = new float[c.w * c.h];
        for (int y = 0; y < c.h; y++) for (int x = 0; x < c.w; x++) c.px[y * c.w + x] = sub[(b[1] + y) * gw + b[0] + x];
        return c;
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
