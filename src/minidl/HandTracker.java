package minidl;

import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Finds hands and fingertips in camera pictures, in plain Java (no OpenCV, no libraries). The steps follow the classic
 * skin-colour pipeline:
 * <ol>
 *  <li><b>Skin colour.</b> Before calibration a general skin rule in YCrCb colour space. After {@link #calibrate} (the user covers the
 *      9 squares with the palm) a hue/saturation histogram of YOUR skin in YOUR light is built, and every pixel is scored by
 *      looking it up (histogram back-projection), then smoothed over a small disc.</li>
 *  <li><b>Clean-up.</b> Threshold, then erode and dilate (removes specks, fills pin holes).</li>
 *  <li><b>Blobs.</b> Connected components; the biggest ones (up to 2 hands) that are large enough.</li>
 *  <li><b>Palm.</b> Distance transform: the centre of the biggest circle inside the blob is the palm centre, its radius the palm size.</li>
 *  <li><b>Contour, convex hull, convexity defects.</b> The outline is traced; the hull is the "rubber band" around it; the
 *      defects are the valleys between fingers.</li>
 *  <li><b>Fingertips.</b> Hull corners that stick out far beyond the palm circle (and are not on the picture border) are fingertips;
 *      close ones are merged. Number of fingertips = number of fingers shown (0 = fist, 5 = open hand).</li>
 *  <li><b>Tracking.</b> Hands are matched between pictures by position; the pointing fingertip is smoothed by a Kalman filter,
 *      which also gives its speed and keeps guessing for a few pictures if the finger is briefly lost.</li>
 * </ol>
 * What keeps faces and the room out (added after real-camera testing showed colour alone is not enough):
 * <ul>
 *  <li><b>Learn background:</b> pictures taken WITHOUT your hands (your face may stay in) become a reference; afterwards only what is new
 *      (your hand) can be a hand. Walls, wooden furniture and your resting face are background.</li>
 *  <li><b>Wave to start:</b> without a background, a skin-coloured blob must MOVE before it counts as a new hand (a still face or wall never does);
 *      once found, the hand is tracked even when held still.</li>
 *  <li><b>Not a hand:</b> blobs with eye/mouth-like holes (faces), blobs bigger than 30% of the picture, blobs touching 3 edges (walls).</li>
 *  <li><b>Hand zone:</b> only a rectangle you choose counts.</li>
 * </ul>
 * Every rejected blob is reported with its reason, so you can see what was ignored and why.
 * Limits: it is colour based. A face or an arm in the picture is skin too (keep them out, wear long sleeves), strong coloured light
 * or a wooden/skin-coloured background confuses it, and the finger count is geometry, not a trained model: it knows how many fingers
 * are out, not WHICH ones. Thumbs-up, "rock" and the OK sign need a trained landmark model and are not recognised.
 */
public final class HandTracker {
    public static final String[] GESTURES = {"FIST", "POINT", "PEACE", "THREE", "FOUR", "OPEN PALM"};

    /** One hand in one picture. All coordinates are pixels of the (possibly scaled) picture of size Result.w x Result.h. */
    public static final class Hand {
        public int slot, area, fingers;
        public String gesture = "";
        public double cx, cy, palmR, pointerX, pointerY, speed;
        public boolean hasPointer, coasting;
        public int[][] contour = new int[0][], hull = new int[0][], tips = new int[0][], valleys = new int[0][];
    }

    /** A skin-coloured blob that was NOT counted as a hand, with the reason. */
    public static final class Rejected {
        public final int x0, y0, x1, y1, area;
        public final String reason;
        Rejected(int x0, int y0, int x1, int y1, int area, String reason) { this.x0 = x0; this.y0 = y0; this.x1 = x1; this.y1 = y1; this.area = area; this.reason = reason; }
    }

    public static final class Result {
        public int w, h;
        public boolean calibrated;
        public double ms;
        public final List<Hand> hands = new ArrayList<>();
        public final List<Rejected> rejected = new ArrayList<>();
        public boolean background;
        public byte[] mask; // 1 = skin, after clean-up
    }

    private static final int MAX_W = 320, H_BINS = 30, S_BINS = 32;
    // cleanup settings (the OpenCV tutorial works at 640 x 480; this works at 320 x 240, so everything is about half as big)
    static final int ERODE = Integer.getInteger("minidl.hand.erode", 1), SMOOTH_DIV = Integer.getInteger("minidl.hand.smooth", 80), THRESHOLD = Integer.getInteger("minidl.hand.thr", 40);
    /** A skin blob must have moved before it can become a NEW hand (when no background was learned). Tracked hands may rest. */
    public volatile boolean requireMotion = true;
    /** Reject blobs with two or more eye/mouth-like holes (faces). */
    public volatile boolean rejectFaces = true;
    static final double MAX_AREA = Double.parseDouble(System.getProperty("minidl.hand.maxarea", "0.30")), FACE_MIN_AREA = 0.025, MOVE_FRACTION = 0.03;
    static final int MOTION_THR = 14, HOLE_MIN = 8, BG_MIN = 30, BG_GROW = 3;
    static final double MIN_PALM = 8; // pixels (at 320 wide): a hand has a palm at least this big; a sliver or an edge does not
    private final List<int[]> bgFaces = new ArrayList<>(); // boxes of faces seen in the learned background: minX, minY, maxX, maxY
    static final double BG_K = 4.0;

    private double[][] hist;            // [hue bin][saturation bin], 0..255; null = not calibrated
    // background model: per pixel mean colour (r, g, b) and spread
    private double[] bgSum, bgSq;
    private float[] bgMean, bgStd;
    private int bgFrames, bgW, bgH;
    private boolean bgReady;
    private int[] prevGray;
    private final Slot[] slots = {new Slot(), new Slot()};
    private double lastTime = -1;

    /** Per-hand tracking memory. */
    private static final class Slot {
        final Kalman1D kx = new Kalman1D(4e4, 6), ky = new Kalman1D(4e4, 6);
        double cx, cy;
        boolean active;
        int lost;
    }

    public boolean calibrated() { return hist != null; }

    public synchronized void reset() {
        // (the learned background is kept: it has its own button)
        hist = null;
        lastTime = -1;
        prevGray = null;
        for (Slot s : slots) { s.kx.reset(); s.ky.reset(); s.active = false; s.lost = 0; }
    }

    // ------------------------------------------------------------------ background
    public synchronized boolean backgroundReady() { return bgReady; }

    /** Start collecting a new background (forgets the old one). */
    public synchronized void beginBackground() { bgSum = bgSq = null; bgMean = bgStd = null; bgFrames = 0; bgReady = false; }

    /** Adds one picture WITHOUT hands in it. Returns how many pictures were added so far. */
    public synchronized int addBackground(BufferedImage frame) {
        Object[] p = pixels(frame);
        int[] px = (int[]) p[0];
        int w = (Integer) p[1], h = (Integer) p[2];
        if (bgSum == null || bgW != w || bgH != h) { bgSum = new double[w * h * 3]; bgSq = new double[w * h * 3]; bgW = w; bgH = h; bgFrames = 0; bgReady = false; }
        for (int i = 0; i < px.length; i++) {
            int[] c = {px[i] >> 16 & 255, px[i] >> 8 & 255, px[i] & 255};
            for (int k = 0; k < 3; k++) { bgSum[i * 3 + k] += c[k]; bgSq[i * 3 + k] += (double) c[k] * c[k]; }
        }
        return ++bgFrames;
    }

    /** Makes the collected pictures the background. Needs at least 3. */
    public synchronized void finishBackground() {
        if (bgSum == null || bgFrames < 3) throw new IllegalStateException("need at least 3 background pictures, have " + bgFrames);
        int n = bgW * bgH;
        bgMean = new float[n * 3];
        bgStd = new float[n];
        for (int i = 0; i < n; i++) {
            double worst = 0;
            for (int k = 0; k < 3; k++) {
                double m = bgSum[i * 3 + k] / bgFrames, v = Math.max(0, bgSq[i * 3 + k] / bgFrames - m * m);
                bgMean[i * 3 + k] = (float) m;
                worst = Math.max(worst, Math.sqrt(v));
            }
            bgStd[i] = (float) worst;
        }
        bgSum = bgSq = null;
        bgReady = true;
        findBackgroundFaces();
    }

    /** Looks for faces (big skin blobs with eye/mouth holes) in the background picture, so a head that moves is not mistaken for a hand. */
    private void findBackgroundFaces() {
        bgFaces.clear();
        if (!bgReady) return;
        int w = bgW, h = bgH;
        int[] px = new int[w * h];
        for (int i = 0; i < px.length; i++) px[i] = (Math.round(bgMean[i * 3]) << 16) | (Math.round(bgMean[i * 3 + 1]) << 8) | Math.round(bgMean[i * 3 + 2]);
        byte[] m = dilate(erode(skinMask(px, w, h), w, h, ERODE), w, h, ERODE);
        List<int[]> blobs = components(m, w, h, Math.max(40, (int) (FACE_MIN_AREA * w * h)));
        int[] label = lastLabels;
        for (int[] b : blobs) if (b[1] <= MAX_AREA * w * h && countHoles(label, b[0], b[4], b[5], b[6], b[7], w) >= 2) bgFaces.add(new int[]{b[4], b[5], b[6], b[7]});
    }

    public synchronized void clearBackground() { bgSum = bgSq = null; bgMean = null; bgStd = null; bgFrames = 0; bgReady = false; bgFaces.clear(); }

    /** Pixels that differ from the learned background (grown a little so the whole hand edge is included). Also lets the background follow slow light changes. */
    private byte[] foreground(int[] px, int w, int h) {
        byte[] fg = new byte[w * h];
        for (int i = 0; i < fg.length; i++) {
            int r = px[i] >> 16 & 255, g = px[i] >> 8 & 255, b = px[i] & 255;
            double d = Math.max(Math.abs(r - bgMean[i * 3]), Math.max(Math.abs(g - bgMean[i * 3 + 1]), Math.abs(b - bgMean[i * 3 + 2])));
            fg[i] = (byte) (d > Math.max(BG_MIN, BG_K * bgStd[i]) ? 1 : 0);
        }
        fg = dilate(erode(fg, w, h, 1), w, h, BG_GROW);
        for (int i = 0; i < fg.length; i++) if (fg[i] == 0) { // slow drift (light, exposure) where nothing new is
            bgMean[i * 3] += 0.02f * ((px[i] >> 16 & 255) - bgMean[i * 3]);
            bgMean[i * 3 + 1] += 0.02f * ((px[i] >> 8 & 255) - bgMean[i * 3 + 1]);
            bgMean[i * 3 + 2] += 0.02f * ((px[i] & 255) - bgMean[i * 3 + 2]);
        }
        return fg;
    }

    /** True where the picture changed since the previous one (a global brightness change is ignored). Null for the first picture. */
    private boolean[] motion(int[] px, int w, int h) {
        int[] g = new int[w * h];
        for (int i = 0; i < g.length; i++) g[i] = ((px[i] >> 16 & 255) * 299 + (px[i] >> 8 & 255) * 587 + (px[i] & 255) * 114) / 1000;
        boolean[] mv = null;
        if (prevGray != null && prevGray.length == g.length) {
            long sum = 0;
            for (int i = 0; i < g.length; i++) sum += g[i] - prevGray[i];
            double shift = sum / (double) g.length;
            mv = new boolean[g.length];
            for (int i = 0; i < g.length; i++) mv[i] = Math.abs(g[i] - prevGray[i] - shift) > MOTION_THR;
        }
        prevGray = g;
        return mv;
    }

    /** The 9 small squares the user covers with the palm: {x, y, size} in pixels of a w x h picture (same layout as the OpenCV tutorial). */
    public static int[][] calibrationSquares(int w, int h) {
        int size = Math.max(6, (int) Math.round(0.035 * Math.min(w, h)));
        int[][] q = new int[9][];
        int k = 0;
        for (int row = 0; row < 3; row++) for (int col = 0; col < 3; col++) {
            int cx = (int) Math.round((9 + col) / 20.0 * w), cy = (int) Math.round((6 + 3 * row) / 20.0 * h);
            q[k++] = new int[]{Math.min(w - size, Math.max(0, cx - size / 2)), Math.min(h - size, Math.max(0, cy - size / 2)), size};
        }
        return q;
    }

    // ------------------------------------------------------------------ pictures
    /** The picture as ints 0xRRGGBB, scaled down to at most 320 wide. Returns {pixels, w, h} packed in an Object[]. */
    private static Object[] pixels(BufferedImage src) {
        int w = src.getWidth(), h = src.getHeight();
        if (w > MAX_W) { h = Math.max(1, (int) Math.round((double) h * MAX_W / w)); w = MAX_W; }
        BufferedImage img = src;
        if (w != src.getWidth() || src.getType() != BufferedImage.TYPE_INT_RGB) {
            img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D g = img.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(src, 0, 0, w, h, null);
            g.dispose();
        }
        return new Object[]{img.getRGB(0, 0, w, h, null, 0, w), w, h};
    }

    /** OpenCV-style HSV: hue 0..179, saturation and value 0..255. */
    static void hsv(int rgb, int[] out) {
        int r = rgb >> 16 & 255, g = rgb >> 8 & 255, b = rgb & 255;
        int mx = Math.max(r, Math.max(g, b)), mn = Math.min(r, Math.min(g, b)), d = mx - mn;
        double hh = 0;
        if (d > 0) {
            if (mx == r) hh = 60.0 * (((g - b) / (double) d) % 6);
            else if (mx == g) hh = 60.0 * ((b - r) / (double) d + 2);
            else hh = 60.0 * ((r - g) / (double) d + 4);
            if (hh < 0) hh += 360;
        }
        out[0] = (int) Math.min(179, Math.round(hh / 2));
        out[1] = mx == 0 ? 0 : 255 * d / mx;
        out[2] = mx;
    }

    private static boolean skinRule(int rgb) { // a general skin range in YCrCb
        int r = rgb >> 16 & 255, g = rgb >> 8 & 255, b = rgb & 255;
        double y = 0.299 * r + 0.587 * g + 0.114 * b, cr = (r - y) * 0.713 + 128, cb = (b - y) * 0.564 + 128;
        return y > 40 && cr >= 133 && cr <= 173 && cb >= 77 && cb <= 127;
    }

    // ------------------------------------------------------------------ calibration
    /** Builds the skin histogram from the 9 squares (the user's palm covers them). Returns how many pixels were sampled. */
    public synchronized int calibrate(BufferedImage frame) {
        Object[] p = pixels(frame);
        int[] px = (int[]) p[0];
        int w = (Integer) p[1], h = (Integer) p[2];
        double[][] hs = new double[H_BINS][S_BINS];
        int[] t = new int[3];
        int n = 0;
        for (int[] sq : calibrationSquares(w, h))
            for (int y = sq[1]; y < sq[1] + sq[2]; y++) for (int x = sq[0]; x < sq[0] + sq[2]; x++) {
                hsv(px[y * w + x], t);
                hs[t[0] * H_BINS / 180][Math.min(S_BINS - 1, t[1] * S_BINS / 256)]++;
                n++;
            }
        if (n == 0) throw new IllegalArgumentException("nothing to sample");
        double[][] sm = new double[H_BINS][S_BINS]; // blur by one bin so that neighbouring colours count a little too
        double mx = 0;
        for (int a = 0; a < H_BINS; a++) for (int b = 0; b < S_BINS; b++) {
            double s = 0;
            for (int da = -1; da <= 1; da++) for (int db = -1; db <= 1; db++) {
                int aa = a + da, bb = b + db;
                if (aa < 0 || bb < 0 || aa >= H_BINS || bb >= S_BINS) continue;
                s += hs[aa][bb] * (da == 0 && db == 0 ? 2.0 : 1.0);
            }
            sm[a][b] = s; mx = Math.max(mx, s);
        }
        for (double[] row : sm) for (int b = 0; b < S_BINS; b++) row[b] = 255 * row[b] / mx;
        hist = sm;
        findBackgroundFaces(); // the skin colour changed: look for faces again
        return n;
    }

    // ------------------------------------------------------------------ the pipeline
    /** @param timeSec any increasing clock in seconds (used for speed and the Kalman filter) */
    public synchronized Result process(BufferedImage frame, double timeSec) { return process(frame, timeSec, null); }

    /**
     * @param zone optional hand zone {x0, y0, x1, y1} as fractions 0..1 of the picture: only skin inside it counts
     */
    public synchronized Result process(BufferedImage frame, double timeSec, double[] zone) {
        long t0 = System.nanoTime();
        Object[] p = pixels(frame);
        int[] px = (int[]) p[0];
        final int w = (Integer) p[1], h = (Integer) p[2];
        Result res = new Result();
        res.w = w; res.h = h; res.calibrated = hist != null;
        double dt = lastTime < 0 ? 1.0 / 15 : Math.max(1e-3, Math.min(0.5, timeSec - lastTime));
        lastTime = timeSec;

        byte[] mask = skinMask(px, w, h);
        if (zone != null && zone.length == 4) {
            int x0 = (int) Math.round(zone[0] * w), y0 = (int) Math.round(zone[1] * h), x1 = (int) Math.round(zone[2] * w), y1 = (int) Math.round(zone[3] * h);
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) if (x < x0 || x >= x1 || y < y0 || y >= y1) mask[y * w + x] = 0;
        }
        res.background = bgReady && bgW == w && bgH == h;
        if (res.background) { // only what is NEW compared with the learned background can be a hand
            byte[] fg = foreground(px, w, h);
            for (int i = 0; i < mask.length; i++) if (fg[i] == 0) mask[i] = 0;
        }
        mask = erode(mask, w, h, ERODE);
        mask = dilate(mask, w, h, ERODE);
        res.mask = mask;
        boolean[] moved = motion(px, w, h);

        List<int[]> blobs = components(mask, w, h, Math.max(40, (int) (0.012 * w * h))); // {label, area, cx*1000, cy*1000, minX, minY, maxX, maxY}
        blobs.sort((a, b) -> Integer.compare(b[1], a[1]));
        int[] label = lastLabels;
        List<Hand> found = new ArrayList<>();
        for (int[] blob : blobs) {
            if (found.size() >= 2) break;
            String why = notAHand(label, blob, w, h, moved, res.background);
            if (why != null) { res.rejected.add(new Rejected(blob[4], blob[5], blob[6], blob[7], blob[1], why)); continue; }
            Hand hd = analyse(label, blob, w, h);
            if (hd == null) continue;
            if (hd.palmR < MIN_PALM) { res.rejected.add(new Rejected(blob[4], blob[5], blob[6], blob[7], blob[1], "too thin: no palm inside (a sliver or an edge, not a hand)")); continue; }
            found.add(hd);
        }
        track(found, dt, w);
        res.hands.addAll(found);
        res.ms = (System.nanoTime() - t0) / 1e6;
        return res;
    }

    private byte[] skinMask(int[] px, int w, int h) {
        byte[] m = new byte[w * h];
        if (hist == null) {
            for (int i = 0; i < m.length; i++) m[i] = (byte) (skinRule(px[i]) ? 1 : 0);
            return m;
        }
        int[] score = new int[w * h];
        int[] t = new int[3];
        for (int i = 0; i < score.length; i++) {
            hsv(px[i], t);
            score[i] = t[2] < 30 ? 0 : (int) hist[t[0] * H_BINS / 180][Math.min(S_BINS - 1, t[1] * S_BINS / 256)];
        }
        int r = Math.max(1, w / SMOOTH_DIV);                       // smooth over a small square, like the disc filter of the tutorial
        long[] sum = new long[(w + 1) * (h + 1)];
        for (int y = 0; y < h; y++) { long row = 0; for (int x = 0; x < w; x++) { row += score[y * w + x]; sum[(y + 1) * (w + 1) + x + 1] = sum[y * (w + 1) + x + 1] + row; } }
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            int x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1), y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
            long s = sum[y1 * (w + 1) + x1] - sum[y0 * (w + 1) + x1] - sum[y1 * (w + 1) + x0] + sum[y0 * (w + 1) + x0];
            m[y * w + x] = (byte) (s / ((long) (x1 - x0) * (y1 - y0)) >= THRESHOLD ? 1 : 0);
        }
        return m;
    }

    static byte[] erode(byte[] m, int w, int h, int times) {
        for (int t = 0; t < times; t++) {
            byte[] o = new byte[m.length];
            for (int y = 1; y < h - 1; y++) for (int x = 1; x < w - 1; x++) {
                int i = y * w + x;
                if (m[i] != 0 && m[i - 1] != 0 && m[i + 1] != 0 && m[i - w] != 0 && m[i + w] != 0 && m[i - w - 1] != 0 && m[i - w + 1] != 0 && m[i + w - 1] != 0 && m[i + w + 1] != 0) o[i] = 1;
            }
            m = o;
        }
        return m;
    }

    static byte[] dilate(byte[] m, int w, int h, int times) {
        for (int t = 0; t < times; t++) {
            byte[] o = new byte[m.length];
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
                if (m[y * w + x] != 0) { for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { int xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < w && yy < h) o[yy * w + xx] = 1; } }
            }
            m = o;
        }
        return m;
    }

    private int[] lastLabels;

    /** 8-connected components of the mask, those with at least minArea pixels; the label picture is left in lastLabels. */
    private List<int[]> components(byte[] m, int w, int h, int minArea) {
        int[] label = new int[w * h], stack = new int[w * h];
        List<int[]> out = new ArrayList<>();
        int next = 0;
        for (int s0 = 0; s0 < w * h; s0++) {
            if (m[s0] == 0 || label[s0] != 0) continue;
            int id = ++next, sp = 0, area = 0, minX = w, minY = h, maxX = 0, maxY = 0;
            long sx = 0, sy = 0;
            stack[sp++] = s0; label[s0] = id;
            while (sp > 0) {
                int q = stack[--sp], x = q % w, y = q / w;
                area++; sx += x; sy += y;
                minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    int nq = ny * w + nx;
                    if (m[nq] != 0 && label[nq] == 0) { label[nq] = id; stack[sp++] = nq; }
                }
            }
            if (area >= minArea) out.add(new int[]{id, area, (int) (1000.0 * sx / area), (int) (1000.0 * sy / area), minX, minY, maxX, maxY});
        }
        lastLabels = label;
        return out;
    }

    // ------------------------------------------------------------------ is this skin-coloured blob really a hand?
    /** Null if the blob may be a hand, else the reason it is not. */
    private String notAHand(int[] label, int[] blob, int w, int h, boolean[] moved, boolean background) {
        int area = blob[1];
        if (area > MAX_AREA * w * h) return "too big: over " + Math.round(MAX_AREA * 100) + "% of the picture (a wall or the background?)";
        int edges = (blob[4] <= 0 ? 1 : 0) + (blob[5] <= 0 ? 1 : 0) + (blob[6] >= w - 1 ? 1 : 0) + (blob[7] >= h - 1 ? 1 : 0);
        if (edges >= 3) return "touches " + edges + " edges of the picture (a wall or the background?)";
        if (rejectFaces && background) {
            double bx = blob[2] / 1000.0, by = blob[3] / 1000.0;
            for (int[] f : bgFaces) { // the head moved: its new edge is "new skin" whose CENTRE sits at the old face (a hand reaching toward the chin has its centre far away)
                double mx = 0.25 * (f[2] - f[0]), my = 0.25 * (f[3] - f[1]);
                if (bx >= f[0] - mx && bx <= f[2] + mx && by >= f[1] - my && by <= f[3] + my) return "at the face seen in the background (the head moved?)";
            }
        }
        if (rejectFaces && area >= FACE_MIN_AREA * w * h) {
            int holes = countHoles(label, blob[0], blob[4], blob[5], blob[6], blob[7], w);
            if (holes >= 2) return "face-like: " + holes + " holes inside (eyes, mouth)";
        }
        if (requireMotion && !background) {
            double cx = blob[2] / 1000.0, cy = blob[3] / 1000.0;
            boolean tracked = false;
            for (Slot sl : slots) if (sl.active && Math.hypot(sl.cx - cx, sl.cy - cy) < 0.3 * w) tracked = true;
            if (!tracked) {
                int mv = 0;
                if (moved != null) for (int y = Math.max(0, blob[5] - 4); y <= Math.min(h - 1, blob[7] + 4); y++) for (int x = Math.max(0, blob[4] - 4); x <= Math.min(w - 1, blob[6] + 4); x++) if (moved[y * w + x]) mv++;
                if (mv < MOVE_FRACTION * area) return "still: it has not moved (wave your hand to start, or learn the background)";
            }
        }
        return null;
    }

    /** Number of enclosed non-blob regions (at least HOLE_MIN pixels) inside the blob's box: eyes and mouth make a face, a hand has none. */
    static int countHoles(int[] label, int id, int minX, int minY, int maxX, int maxY, int w) {
        int bw = maxX - minX + 1, bh = maxY - minY + 1;
        byte[] state = new byte[bw * bh]; // 0 = unknown, 1 = reachable from the box border (outside), 2 = already counted as a hole
        int[] stack = new int[bw * bh];
        int sp = 0;
        for (int y = 0; y < bh; y++) for (int x = 0; x < bw; x++) {
            boolean border = x == 0 || y == 0 || x == bw - 1 || y == bh - 1;
            if (border && label[(minY + y) * w + minX + x] != id && state[y * bw + x] == 0) { state[y * bw + x] = 1; stack[sp++] = y * bw + x; }
        }
        while (sp > 0) {
            int q = stack[--sp], x = q % bw, y = q / bw;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue;
                int nq = ny * bw + nx;
                if (state[nq] == 0 && label[(minY + ny) * w + minX + nx] != id) { state[nq] = 1; stack[sp++] = nq; }
            }
        }
        int holes = 0;
        for (int start = 0; start < state.length; start++) {
            if (state[start] != 0 || label[(minY + start / bw) * w + minX + start % bw] == id) continue;
            int area = 0;
            sp = 0; stack[sp++] = start; state[start] = 2;
            while (sp > 0) {
                int q = stack[--sp], x = q % bw, y = q / bw;
                area++;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue;
                    int nq = ny * bw + nx;
                    if (state[nq] == 0 && label[(minY + ny) * w + minX + nx] != id) { state[nq] = 2; stack[sp++] = nq; }
                }
            }
            if (area >= HOLE_MIN) holes++;
        }
        return holes;
    }

    // ------------------------------------------------------------------ one hand
    private Hand analyse(int[] label, int[] blob, int w, int h) {
        int id = blob[0];
        Hand hd = new Hand();
        hd.area = blob[1];
        hd.cx = blob[2] / 1000.0; hd.cy = blob[3] / 1000.0;

        // palm: biggest inscribed circle (chamfer 3-4 distance transform)
        int[] dt = new int[w * h];
        final int INF = 1 << 28;
        for (int i = 0; i < dt.length; i++) dt[i] = label[i] == id ? INF : 0;
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            int i = y * w + x;
            if (dt[i] == 0) continue;
            int v = dt[i];
            v = Math.min(v, (x > 0 ? dt[i - 1] : 0) + 3);
            v = Math.min(v, (y > 0 ? dt[i - w] : 0) + 3);
            v = Math.min(v, (x > 0 && y > 0 ? dt[i - w - 1] : 0) + 4);
            v = Math.min(v, (x < w - 1 && y > 0 ? dt[i - w + 1] : 0) + 4);
            dt[i] = v;
        }
        int best = 0, bx = (int) hd.cx, by = (int) hd.cy;
        for (int y = h - 1; y >= 0; y--) for (int x = w - 1; x >= 0; x--) {
            int i = y * w + x;
            if (dt[i] == 0) continue;
            int v = dt[i];
            v = Math.min(v, (x < w - 1 ? dt[i + 1] : 0) + 3);
            v = Math.min(v, (y < h - 1 ? dt[i + w] : 0) + 3);
            v = Math.min(v, (x < w - 1 && y < h - 1 ? dt[i + w + 1] : 0) + 4);
            v = Math.min(v, (x > 0 && y < h - 1 ? dt[i + w - 1] : 0) + 4);
            dt[i] = v;
            double dc = Math.hypot(x - hd.cx, y - hd.cy);
            if (v > best || (v == best && dc < Math.hypot(bx - hd.cx, by - hd.cy))) { best = v; bx = x; by = y; }
        }
        hd.palmR = Math.max(3, best / 3.0);
        double pcx = bx, pcy = by;

        int[][] contour = trace(label, id, w, h);
        if (contour.length < 12) return null;
        // keep about 150 points for drawing, but work with the full outline
        int step = Math.max(1, contour.length / 150);
        hd.contour = new int[(contour.length + step - 1) / step][];
        for (int i = 0, k = 0; i < contour.length; i += step) hd.contour[k++] = contour[i];

        int[] hullIdx = convexHull(contour);
        hd.hull = new int[hullIdx.length][];
        for (int i = 0; i < hullIdx.length; i++) hd.hull[i] = contour[hullIdx[i]];

        // valleys between fingers: convexity defects that are deep enough
        List<int[]> val = new ArrayList<>();
        for (int[] d : defects(contour, hullIdx)) if (d[3] / 256.0 > 0.45 * hd.palmR) val.add(contour[d[2]]);
        hd.valleys = val.toArray(new int[0][]);

        // fingertips: hull corners far beyond the palm circle, not on the picture border, merged when close together
        List<int[]> cand = new ArrayList<>();
        for (int[] c : hd.hull) {
            boolean border = c[0] <= 1 || c[1] <= 1 || c[0] >= w - 2 || c[1] >= h - 2;
            if (!border && Math.hypot(c[0] - pcx, c[1] - pcy) >= 1.55 * hd.palmR && narrowAtTip(label, id, w, h, c[0], c[1], pcx, pcy, hd.palmR)) cand.add(c);
        }
        cand.sort((a, b) -> Double.compare(Math.hypot(b[0] - pcx, b[1] - pcy), Math.hypot(a[0] - pcx, a[1] - pcy)));
        List<int[]> tips = new ArrayList<>();
        for (int[] c : cand) {
            boolean near = false;
            for (int[] t : tips) if (Math.hypot(t[0] - c[0], t[1] - c[1]) < 0.9 * hd.palmR) near = true;
            if (!near && tips.size() < 5) tips.add(c);
        }
        hd.tips = tips.toArray(new int[0][]);
        hd.fingers = tips.size();
        hd.gesture = GESTURES[Math.min(5, hd.fingers)];
        if (!tips.isEmpty()) { hd.pointerX = tips.get(0)[0]; hd.pointerY = tips.get(0)[1]; hd.hasPointer = true; } // the tip that sticks out farthest = the pointing finger
        hd.cx = pcx; hd.cy = pcy; // the palm centre is steadier than the blob centre
        return hd;
    }

    /**
     * A finger is NARROW near its tip. A little way back from the tip (0.6 palm radii, toward the palm centre) the blob is measured across;
     * a finger is thinner than the palm circle, a forehead, a wrist or a forearm is wider.
     */
    static boolean narrowAtTip(int[] label, int id, int w, int h, int tx, int ty, double cx, double cy, double palmR) {
        double dx = cx - tx, dy = cy - ty, len = Math.hypot(dx, dy);
        if (len < 1e-6) return true;
        dx /= len; dy /= len;
        double px = tx + dx * 0.6 * palmR, py = ty + dy * 0.6 * palmR, nx = -dy, ny = dx;
        int width = 0;
        for (int dir = -1; dir <= 1; dir += 2)
            for (double t = 0; t < 3 * palmR; t += 0.5) {
                int x = (int) Math.round(px + dir * nx * t), y = (int) Math.round(py + dir * ny * t);
                if (x < 0 || y < 0 || x >= w || y >= h || label[y * w + x] != id) break;
                width++;
            }
        return width * 0.5 <= 0.95 * palmR;
    }

    /** Outer outline of one blob, in order (Moore neighbour tracing, Jacob's stopping rule). */
    static int[][] trace(int[] label, int id, int w, int h) {
        int sx = -1, sy = -1;
        outer:
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) if (label[y * w + x] == id) { sx = x; sy = y; break outer; }
        if (sx < 0) return new int[0][];
        final int[] DX = {1, 1, 0, -1, -1, -1, 0, 1}, DY = {0, 1, 1, 1, 0, -1, -1, -1}; // clockwise, starting east
        List<int[]> pts = new ArrayList<>();
        int x = sx, y = sy, dir = 6; // we arrived from the north (nothing above the first pixel)
        int startDir = -1, guard = 0, limit = 8 * (w + h) + 4 * w * h / 4;
        pts.add(new int[]{x, y});
        while (guard++ < limit) {
            boolean moved = false;
            for (int k = 0; k < 8; k++) {
                int d = (dir + 6 + k) % 8; // begin the search just after the pixel we came from
                int nx = x + DX[d], ny = y + DY[d];
                if (nx >= 0 && ny >= 0 && nx < w && ny < h && label[ny * w + nx] == id) {
                    x = nx; y = ny; dir = d; moved = true;
                    break;
                }
            }
            if (!moved) break; // a single pixel
            if (x == sx && y == sy) { if (startDir == -1) startDir = dir; else if (startDir == dir) break; if (pts.size() > 2 && startDir != -1) break; }
            pts.add(new int[]{x, y});
        }
        return pts.toArray(new int[0][]);
    }

    /** Indices (into the contour, ascending) of the convex hull corners (Andrew's monotone chain). */
    static int[] convexHull(int[][] pts) {
        int n = pts.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> pts[a][0] != pts[b][0] ? Integer.compare(pts[a][0], pts[b][0]) : Integer.compare(pts[a][1], pts[b][1]));
        int[] hull = new int[2 * n];
        int k = 0;
        for (int i = 0; i < n; i++) { while (k >= 2 && cross(pts[hull[k - 2]], pts[hull[k - 1]], pts[idx[i]]) <= 0) k--; hull[k++] = idx[i]; }
        for (int i = n - 2, t = k + 1; i >= 0; i--) { while (k >= t && cross(pts[hull[k - 2]], pts[hull[k - 1]], pts[idx[i]]) <= 0) k--; hull[k++] = idx[i]; }
        int[] out = Arrays.copyOf(hull, Math.max(1, k - 1));
        // remove repeated indices (a point on two chains) and sort along the contour
        return Arrays.stream(out).distinct().sorted().toArray();
    }

    private static long cross(int[] o, int[] a, int[] b) { return (long) (a[0] - o[0]) * (b[1] - o[1]) - (long) (a[1] - o[1]) * (b[0] - o[0]); }

    /** Convexity defects: for every hull edge the contour point farthest from it. Each entry: {start, end, far, depth*256}. */
    static List<int[]> defects(int[][] c, int[] hull) {
        List<int[]> out = new ArrayList<>();
        int n = c.length;
        for (int i = 0; i < hull.length; i++) {
            int a = hull[i], b = hull[(i + 1) % hull.length];
            if (a == b) continue;
            double ax = c[a][0], ay = c[a][1], bx = c[b][0], by = c[b][1], len = Math.hypot(bx - ax, by - ay);
            if (len < 1e-9) continue;
            double best = 0;
            int far = -1;
            for (int j = (a + 1) % n; j != b; j = (j + 1) % n) {
                double d = Math.abs((bx - ax) * (ay - c[j][1]) - (ax - c[j][0]) * (by - ay)) / len;
                if (d > best) { best = d; far = j; }
            }
            if (far >= 0 && best > 1.0) out.add(new int[]{a, b, far, (int) Math.round(best * 256)});
        }
        return out;
    }

    // ------------------------------------------------------------------ tracking between pictures
    private void track(List<Hand> found, double dt, int w) {
        boolean[] used = new boolean[slots.length];
        found.sort((a, b) -> Double.compare(a.cx, b.cx)); // left hand of the picture first, so a hand keeps its slot
        for (Hand hd : found) {
            int bestSlot = -1;
            double bestD = 0.3 * w;
            for (int s = 0; s < slots.length; s++) {
                if (used[s] || !slots[s].active) continue;
                double d = Math.hypot(slots[s].cx - hd.cx, slots[s].cy - hd.cy);
                if (d < bestD) { bestD = d; bestSlot = s; }
            }
            if (bestSlot < 0) for (int s = 0; s < slots.length; s++) if (!used[s] && !slots[s].active) { bestSlot = s; break; }
            if (bestSlot < 0) bestSlot = used[0] ? 1 : 0;
            used[bestSlot] = true;
            Slot sl = slots[bestSlot];
            if (!sl.active) { sl.kx.reset(); sl.ky.reset(); }
            sl.active = true; sl.lost = 0; sl.cx = hd.cx; sl.cy = hd.cy;
            hd.slot = bestSlot;
            if (hd.hasPointer) {
                hd.pointerX = sl.kx.update(hd.pointerX, dt);
                hd.pointerY = sl.ky.update(hd.pointerY, dt);
                hd.speed = Math.hypot(sl.kx.velocity(), sl.ky.velocity());
            } else if (sl.kx.started() && hd.fingers == 0) { sl.kx.reset(); sl.ky.reset(); }
        }
        for (int s = 0; s < slots.length; s++) {
            if (used[s]) continue;
            slots[s].lost++;
            if (slots[s].lost > 8) { slots[s].active = false; slots[s].kx.reset(); slots[s].ky.reset(); }
        }
    }
}
