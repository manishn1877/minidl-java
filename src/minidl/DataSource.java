package minidl;

/**
 * Where training data comes from. The Trainer asks for one small batch at a time, so the
 * full dataset never has to exist as big float tensors. See ByteSource (8 bits per pixel).
 */
public interface DataSource {
    int size();
    /** Rows idx[from..to) as a (to-from x features) tensor. */
    Tensor features(int[] idx, int from, int to);
    /** Matching answers (one-hot for classes, or numbers for regression). */
    Tensor targets(int[] idx, int from, int to);
}
