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

    public static ByteSource letters(String imagesPath, String labelsPath, int maxCount) throws IOException {
        return MnistLoader.loadSource(imagesPath, labelsPath, maxCount, 26, 1, true);
    }

    /** The folder's train or test pair ("train" / "test"), or throws a clear error when files are missing. */
    public static ByteSource lettersFrom(String dir, String split, int maxCount) throws IOException {
        return letters(find(dir, "emnist-letters-" + split + "-images-idx3-ubyte"), find(dir, "emnist-letters-" + split + "-labels-idx1-ubyte"), maxCount);
    }

    public static boolean exists(String dir) {
        return new File(find(dir, "emnist-letters-train-images-idx3-ubyte")).exists();
    }

    private static String find(String dir, String name) {
        File gz = new File(dir, name + ".gz");
        return (gz.exists() ? gz : new File(dir, name)).getPath();
    }
}
