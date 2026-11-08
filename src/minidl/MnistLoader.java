package minidl;

import java.io.*;
import java.util.zip.GZIPInputStream;

/**
 * Reads the MNIST handwritten digit files (IDX format). Works with .gz or unzipped files.
 * Images come back as (n x 784) with pixels scaled to 0..1. Labels come back one-hot (n x 10).
 */
public final class MnistLoader {
    private MnistLoader() {}

    public static Tensor loadImages(String path) throws IOException { return loadImages(path, Integer.MAX_VALUE); }

    public static Tensor loadImages(String path, int maxCount) throws IOException {
        try (DataInputStream in = open(path)) {
            int magic = in.readInt();
            if (magic != 2051) throw new IOException("not an MNIST image file (magic " + magic + ")");
            int n = Math.min(in.readInt(), maxCount), rows = in.readInt(), cols = in.readInt();
            int size = rows * cols;
            Tensor x = new Tensor(n, size);
            byte[] buf = new byte[size];
            for (int i = 0; i < n; i++) {
                in.readFully(buf);
                for (int j = 0; j < size; j++) x.data[i * size + j] = (buf[j] & 0xFF) / 255f;
            }
            return x;
        }
    }

    public static int[] loadLabelIds(String path) throws IOException { return loadLabelIds(path, Integer.MAX_VALUE); }

    public static int[] loadLabelIds(String path, int maxCount) throws IOException {
        try (DataInputStream in = open(path)) {
            int magic = in.readInt();
            if (magic != 2049) throw new IOException("not an MNIST label file (magic " + magic + ")");
            int n = Math.min(in.readInt(), maxCount);
            int[] ids = new int[n];
            for (int i = 0; i < n; i++) ids[i] = in.readUnsignedByte();
            return ids;
        }
    }

    public static Tensor loadLabels(String path) throws IOException { return loadLabels(path, Integer.MAX_VALUE); }

    public static Tensor loadLabels(String path, int maxCount) throws IOException {
        return Data.oneHot(loadLabelIds(path, maxCount), 10);
    }


    /**
     * Loads images + labels as raw bytes (1 byte per pixel). Use this for big data:
     * 8x smaller than doubles, 4x smaller than floats. Pair it with the Trainer (it converts one batch at a time).
     */
    public static ByteSource loadSource(String imagesPath, String labelsPath, int maxCount) throws IOException {
        return loadSource(imagesPath, labelsPath, maxCount, 10, 0, false);
    }

    /**
     * Same, for other IDX data sets. classes = number of answers, labelShift is subtracted from every label
     * (EMNIST letters are numbered 1..26, we want 0..25), transpose = EMNIST stores every picture turned on its side.
     */
    public static ByteSource loadSource(String imagesPath, String labelsPath, int maxCount, int classes, int labelShift, boolean transpose) throws IOException {
        byte[] pixels; int dim, n, side = 0;
        try (DataInputStream in = open(imagesPath)) {
            int magic = in.readInt();
            if (magic != 2051) throw new IOException("not an MNIST image file (magic " + magic + ")");
            n = Math.min(in.readInt(), maxCount);
            int rows = in.readInt(), cols = in.readInt();
            dim = rows * cols;
            if (transpose && rows != cols) throw new IOException("can only transpose square pictures");
            side = rows;
            pixels = new byte[n * dim];
            in.readFully(pixels);
        }
        if (transpose) {
            byte[] t = new byte[side * side];
            for (int i = 0; i < n; i++) {
                int base = i * dim;
                for (int y = 0; y < side; y++) for (int x = 0; x < side; x++) t[y * side + x] = pixels[base + x * side + y];
                System.arraycopy(t, 0, pixels, base, dim);
            }
        }
        byte[] labels = new byte[n];
        try (DataInputStream in = open(labelsPath)) {
            int magic = in.readInt();
            if (magic != 2049) throw new IOException("not an MNIST label file (magic " + magic + ")");
            in.readInt();
            in.readFully(labels);
        }
        for (int i = 0; i < n; i++) {
            int v = (labels[i] & 0xFF) - labelShift;
            if (v < 0 || v >= classes) throw new IOException("label " + (v + labelShift) + " is outside 0.." + (classes - 1 + labelShift));
            labels[i] = (byte) v;
        }
        return new ByteSource(pixels, dim, labels, classes);
    }


    /** Unzips a .gz file into outDir (if not done already) and returns the path of the plain file. Needed for MappedSource. */
    public static String gunzip(String gzPath, String outDir) throws IOException {
        File gz = new File(gzPath);
        File out = new File(outDir, gz.getName().replaceAll("\\.gz$", ""));
        if (out.exists() && out.length() > 0) return out.getPath();
        out.getParentFile().mkdirs();
        try (InputStream in = new GZIPInputStream(new BufferedInputStream(new FileInputStream(gz)));
             OutputStream o = new BufferedOutputStream(new FileOutputStream(out))) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) o.write(buf, 0, n);
        }
        return out.getPath();
    }

    private static DataInputStream open(String path) throws IOException {
        InputStream raw = new BufferedInputStream(new FileInputStream(path), 1 << 16);
        if (path.endsWith(".gz")) raw = new BufferedInputStream(new GZIPInputStream(raw), 1 << 16);
        return new DataInputStream(raw);
    }
}
