package minidl;

/**
 * Capital or small? A model with both cases (52 or 62 classes) cannot always tell from the SHAPE: c C, k K, o O, p P, s S, u U, v V, w W, x X, z Z
 * look the same, only the size differs, and the training pictures were all scaled to one size. So for these letters the case is
 * decided from the SIZE of the letter compared with the tallest character already written on the line (refH):
 * clearly smaller (under 68% of it) means small, otherwise capital. With no other character to compare to, the model's own guess is kept and the
 * answer is marked "amb" (case could not be decided), so the page can offer to flip it.
 * Modes: "auto" (the rule above), "upper" (only capitals and digits are allowed), "lower" (only small letters and digits).
 */
public final class LetterCase {
    private LetterCase() {}

    /** Letters whose capital and small form have the same shape. */
    public static final String SIZE_AMBIGUOUS = "CKOPSUVWXZ";
    public static final double SMALL_BELOW = 0.68;

    /** The probabilities after the case rule, the chosen class, and whether the case is a guess. */
    public static final class Outcome {
        public final double[] probs;
        public final int label;
        public final boolean amb;
        public final String note;
        Outcome(double[] probs, int label, boolean amb, String note) { this.probs = probs; this.label = label; this.amb = amb; this.note = note; }
    }

    public static boolean hasBothCases(String alphabet) { return alphabet.indexOf('a') >= 0 && alphabet.indexOf('A') >= 0; }

    private static int argmax(double[] p) { int b = 0; for (int i = 1; i < p.length; i++) if (p[i] > p[b]) b = i; return b; }

    private static double[] normalize(double[] p) {
        double s = 0;
        for (double v : p) s += v;
        double[] o = new double[p.length];
        for (int i = 0; i < p.length; i++) o[i] = s > 0 ? p[i] / s : 1.0 / p.length;
        return o;
    }

    /** boxH = height of this letter, refH = height of the tallest character on the line (0 = unknown). */
    public static Outcome apply(String alphabet, double[] probs, String mode, double boxH, double refH) {
        if (!hasBothCases(alphabet) || probs.length != alphabet.length()) return new Outcome(probs, argmax(probs), false, null);
        double[] p = probs.clone();
        if ("upper".equals(mode) || "lower".equals(mode)) {
            for (int i = 0; i < p.length; i++) {
                char c = alphabet.charAt(i);
                if ("upper".equals(mode) ? Character.isLowerCase(c) : Character.isUpperCase(c)) p[i] = 0;
            }
            p = normalize(p);
            return new Outcome(p, argmax(p), false, "case fixed to " + mode + "case");
        }
        int top = argmax(p);
        char ch = alphabet.charAt(top);
        if (!Character.isLetter(ch) || SIZE_AMBIGUOUS.indexOf(Character.toUpperCase(ch)) < 0) return new Outcome(p, top, false, null);
        int up = alphabet.indexOf(Character.toUpperCase(ch)), lo = alphabet.indexOf(Character.toLowerCase(ch));
        if (refH <= 0 || boxH <= 0) return new Outcome(p, top, true, null);
        boolean small = boxH < SMALL_BELOW * refH;
        double merged = p[up] + p[lo];
        p[up] = small ? 0 : merged;
        p[lo] = small ? merged : 0;
        String note = String.format(java.util.Locale.ROOT, "case chosen by size: %d%% of the tallest letter, so %s", Math.round(100 * boxH / refH), small ? "small" : "capital");
        return new Outcome(p, small ? lo : up, true, note);
    }
}
