package minidl;

/**
 * Images (or any 0..255 data) stored as ONE BYTE per value, plus labels as one byte each.
 * Turned into floats (0..1) only for the batch being used right now.
 *
 * MNIST train set:  as double = 376 MB,  as float = 188 MB,  as bytes = 47 MB.
 *
 * view(from, to) shares the same byte arrays (no copy), so splitting train/validation is free.
 */
public class ByteSource implements DataSource {
    private final byte[] pixels, labels;
    private final int dim, classes, offset, count;

    public ByteSource(byte[] pixels, int dim, byte[] labels, int classes) {
        this(pixels, dim, labels, classes, 0, labels.length);
        if (pixels.length != (long) labels.length * dim) throw new IllegalArgumentException("pixels.length must be labels.length * dim");
    }

    private ByteSource(byte[] pixels, int dim, byte[] labels, int classes, int offset, int count) {
        this.pixels = pixels; this.dim = dim; this.labels = labels; this.classes = classes;
        this.offset = offset; this.count = count;
    }

    /** Rows [from, to) of this source. No data is copied. */
    public ByteSource view(int from, int to) {
        if (from < 0 || to > count || from > to) throw new IllegalArgumentException("bad view range");
        return new ByteSource(pixels, dim, labels, classes, offset + from, to - from);
    }

    @Override public int size() { return count; }

    @Override public Tensor features(int[] idx, int from, int to) {
        Tensor out = new Tensor(to - from, dim);
        final float scale = 1f / 255f;
        for (int r = from; r < to; r++) {
            int base = (offset + idx[r]) * dim, ob = (r - from) * dim;
            for (int j = 0; j < dim; j++) out.data[ob + j] = (pixels[base + j] & 0xFF) * scale;
        }
        return out;
    }

    @Override public Tensor targets(int[] idx, int from, int to) {
        Tensor out = new Tensor(to - from, classes);
        for (int r = from; r < to; r++) out.data[(r - from) * classes + (labels[offset + idx[r]] & 0xFF)] = 1f;
        return out;
    }

    /** Memory held by the byte arrays (shared by all views). */
    public long bytes() { return (long) pixels.length + labels.length; }
}
