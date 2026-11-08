package minidl;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * One data set with digits AND letters: 36 classes, 0-9 then A-Z.
 * MNIST digits (classes 0-9) and EMNIST letters (classes 10-35) are put together and shuffled.
 *
 * The two data sets were prepared differently (digits: 20 px, centred by centre of mass; letters: 24 px,
 * centred by bounding box). A model has to see ONE style, so every MNIST digit is re-made in the EMNIST
 * style here, with the same code the web app uses on your drawing (DigitPreprocess, Style.EMNIST).
 */
public final class Alnum {
    private Alnum() {}

    public static final String CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    /** Loads {@code split} ("train" or "test") from the two folders. maxDigits limits how many MNIST digits are used. */
    public static ByteSource load(String mnistDir, String emnistDir, String split, int maxDigits) throws IOException {
        boolean train = split.equals("train");
        String pre = train ? "train" : "t10k";
        ByteSource digits = MnistLoader.loadSource(find(mnistDir, pre + "-images-idx3-ubyte"), find(mnistDir, pre + "-labels-idx1-ubyte"), maxDigits);
        return combine(digits, EmnistLoader.lettersFrom(emnistDir, split, Integer.MAX_VALUE));
    }

    /** digits (10 classes) + letters (26 classes) -> one shuffled 36-class source. Digits are re-styled to EMNIST. */
    public static ByteSource combine(ByteSource digits, ByteSource letters) {
        byte[] dp = digits.copyPixels(), dl = digits.copyLabels(), lp = letters.copyPixels(), ll = letters.copyLabels();
        final int dim = 784, nd = dl.length, nl = ll.length, n = nd + nl;
        float[] ink = new float[dim];
        for (int i = 0; i < nd; i++) { // MNIST style -> EMNIST style, in place
            for (int k = 0; k < dim; k++) ink[k] = (dp[i * dim + k] & 0xFF) / 255f;
            DigitPreprocess.Piece p = DigitPreprocess.whole(ink, 28, 28, DigitPreprocess.Style.EMNIST);
            for (int k = 0; k < dim; k++) dp[i * dim + k] = p == null ? 0 : (byte) Math.round(p.pixels[k] * 255);
        }
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Random rnd = new Random(42);
        for (int i = n - 1; i > 0; i--) { int j = rnd.nextInt(i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
        byte[] px = new byte[n * dim], lab = new byte[n];
        for (int i = 0; i < n; i++) {
            int s = order[i];
            if (s < nd) { System.arraycopy(dp, s * dim, px, i * dim, dim); lab[i] = dl[s]; }
            else { System.arraycopy(lp, (s - nd) * dim, px, i * dim, dim); lab[i] = (byte) (ll[s - nd] + 10); }
        }
        return new ByteSource(px, dim, lab, 36);
    }

    public static boolean exists(String mnistDir, String emnistDir) {
        return new File(find(mnistDir, "train-images-idx3-ubyte")).exists() && EmnistLoader.exists(emnistDir);
    }

    private static String find(String dir, String name) {
        File gz = new File(dir, name + ".gz");
        return (gz.exists() ? gz : new File(dir, name)).getPath();
    }
}
