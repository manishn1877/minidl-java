package minidl;

import java.io.File;
import java.io.IOException;

/**
 * EMNIST "letters": 124,800 training and 20,800 test pictures of handwritten A-Z (26 classes).
 * Same IDX file format as MNIST, with two differences this class takes care of:
 *   - every picture is stored turned on its side (transposed), so we turn it back
 *   - labels are 1..26, we use 0..25 (0 = A)
 * Upper and lower case share one class (a "c" and a "C" are both class C).
 * Files: emnist-letters-{train,test}-{images-idx3,labels-idx1}-ubyte  (.gz or unzipped). See get-emnist.sh.
 */
public final class EmnistLoader {
    private EmnistLoader() {}

    public static final String LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    /** Capital and small letters, in the order of the EMNIST "byclass" labels 10..61 (A-Z, then a-z). */
    public static final String CASED = LETTERS + "abcdefghijklmnopqrstuvwxyz";
    /** All 62 classes of the EMNIST "byclass" split: digits, capitals, small letters. */
    public static final String FULL = "0123456789" + CASED;

    public static ByteSource letters(String imagesPath, String labelsPath, int maxCount) throws IOException {
        return MnistLoader.loadSource(imagesPath, labelsPath, maxCount, 26, 1, true);
    }

    /** The folder's train or test pair ("train" / "test"), or throws a clear error when files are missing. */
    public static ByteSource lettersFrom(String dir, String split, int maxCount) throws IOException {
        return letters(find(dir, "emnist-letters-" + split + "-images-idx3-ubyte"), find(dir, "emnist-letters-" + split + "-labels-idx1-ubyte"), maxCount);
    }

    /**
     * The EMNIST "byclass" split: 62 classes, labels 0-9 = digits, 10-35 = A-Z, 36-61 = a-z (capital and small letters are DIFFERENT classes).
     * Pictures are stored sideways like the other splits (turned upright here). With lettersOnly only the 52 letters are returned,
     * numbered 0-25 = A-Z and 26-51 = a-z. maxCount limits how many pictures are READ (before filtering).
     * Many letters look the same in capital and small (c C, o O, s S, v V, w W, x X, z Z): the pictures are all scaled to the same size,
     * so only the writer's habit and the size relative to other letters can tell them apart.
     */
    public static ByteSource byClass(String dir, String split, int maxCount, boolean lettersOnly) throws IOException {
        ByteSource all = MnistLoader.loadSource(find(dir, "emnist-byclass-" + split + "-images-idx3-ubyte"), find(dir, "emnist-byclass-" + split + "-labels-idx1-ubyte"), maxCount, 62, 0, true);
        if (!lettersOnly) return all;
        byte[] px = all.copyPixels(), lab = all.copyLabels();
        int keep = 0;
        for (byte b : lab) if (b >= 10) keep++;
        byte[] px2 = new byte[keep * 784], lab2 = new byte[keep];
        for (int i = 0, k = 0; i < lab.length; i++) {
            if (lab[i] < 10) continue;
            System.arraycopy(px, i * 784, px2, k * 784, 784);
            lab2[k++] = (byte) (lab[i] - 10);
        }
        return new ByteSource(px2, 784, lab2, 52);
    }

    public static boolean existsByClass(String dir) {
        return new File(find(dir, "emnist-byclass-train-images-idx3-ubyte")).exists();
    }

    public static boolean exists(String dir) {
        return new File(find(dir, "emnist-letters-train-images-idx3-ubyte")).exists();
    }

    private static String find(String dir, String name) {
        File gz = new File(dir, name + ".gz");
        return (gz.exists() ? gz : new File(dir, name)).getPath();
    }
}
