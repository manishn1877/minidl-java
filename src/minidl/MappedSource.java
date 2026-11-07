package minidl;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * MNIST-format data that stays ON DISK. The file is "memory-mapped": the operating system loads the
 * parts we touch into its own cache and drops them when memory is needed. The Java heap holds almost nothing.
 * This is how data bigger than RAM is handled. Needs the UNZIPPED idx files (see MnistLoader.gunzip).
 * One mapped file can be at most 2 GB.
 */
public class MappedSource implements DataSource {
    private final MappedByteBuffer pix, lab;
    private final int dim, classes, offset, count, pixBase, labBase;

    private MappedSource(MappedByteBuffer pix, MappedByteBuffer lab, int dim, int classes, int offset, int count, int pixBase, int labBase) {
        this.pix = pix; this.lab = lab; this.dim = dim; this.classes = classes;
        this.offset = offset; this.count = count; this.pixBase = pixBase; this.labBase = labBase;
    }

    public static MappedSource open(String imagesPath, String labelsPath) throws IOException {
        MappedByteBuffer pb, lb;
        try (FileChannel c = FileChannel.open(Path.of(imagesPath), StandardOpenOption.READ)) { pb = c.map(FileChannel.MapMode.READ_ONLY, 0, c.size()); }
        try (FileChannel c = FileChannel.open(Path.of(labelsPath), StandardOpenOption.READ)) { lb = c.map(FileChannel.MapMode.READ_ONLY, 0, c.size()); }
        if (pb.getInt(0) != 2051) throw new IOException("not an (unzipped) MNIST image file");
        if (lb.getInt(0) != 2049) throw new IOException("not an (unzipped) MNIST label file");
        int n = pb.getInt(4), dim = pb.getInt(8) * pb.getInt(12);
        if (lb.getInt(4) != n) throw new IOException("image and label counts differ");
        return new MappedSource(pb, lb, dim, 10, 0, n, 16, 8);
    }

    public MappedSource view(int from, int to) {
        if (from < 0 || to > count || from > to) throw new IllegalArgumentException("bad view range");
        return new MappedSource(pix, lab, dim, classes, offset + from, to - from, pixBase, labBase);
    }

    @Override public int size() { return count; }

    @Override public Tensor features(int[] idx, int from, int to) {
        Tensor out = new Tensor(to - from, dim);
        final float scale = 1f / 255f;
        for (int r = from; r < to; r++) {
            int base = pixBase + (offset + idx[r]) * dim, ob = (r - from) * dim;
            for (int j = 0; j < dim; j++) out.data[ob + j] = (pix.get(base + j) & 0xFF) * scale;
        }
        return out;
    }

    @Override public Tensor targets(int[] idx, int from, int to) {
        Tensor out = new Tensor(to - from, classes);
        for (int r = from; r < to; r++) out.data[(r - from) * classes + (lab.get(labBase + offset + idx[r]) & 0xFF)] = 1f;
        return out;
    }
}
