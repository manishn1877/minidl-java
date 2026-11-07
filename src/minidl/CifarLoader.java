package minidl;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Reads CIFAR-10 colour images in the official binary format: every record is 1 label byte followed by
 * 3072 bytes (1024 red, 1024 green, 1024 blue pixels of a 32 x 32 image).
 * Images stay as ONE BYTE per number (see ByteSource); one image is a row [channel][y][x] of 3072 numbers,
 * which is exactly the layout Conv2D expects (3 channels, 32 x 32).
 */
public final class CifarLoader {
    private CifarLoader() {}

    public static final int IMAGE_SIZE = 3 * 32 * 32;
    public static final String[] CLASS_NAMES = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

    /** Loads one or more batch files (data_batch_1.bin ... or test_batch.bin) into one source. */
    public static ByteSource load(String... files) throws IOException {
        int total = 0;
        long[] sizes = new long[files.length];
        for (int i = 0; i < files.length; i++) {
            sizes[i] = new java.io.File(files[i]).length();
            if (sizes[i] % (1 + IMAGE_SIZE) != 0) throw new IOException(files[i] + " is not a CIFAR-10 binary file (size " + sizes[i] + " is not a multiple of " + (1 + IMAGE_SIZE) + ")");
            total += (int) (sizes[i] / (1 + IMAGE_SIZE));
        }
        byte[] pixels = new byte[total * IMAGE_SIZE], labels = new byte[total];
        int n = 0;
        for (String f : files) {
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f), 1 << 16))) {
                int count = (int) (new java.io.File(f).length() / (1 + IMAGE_SIZE));
                for (int i = 0; i < count; i++, n++) {
                    int label = in.readUnsignedByte();
                    if (label > 9) throw new IOException(f + ": label " + label + " is not in 0..9");
                    labels[n] = (byte) label;
                    in.readFully(pixels, n * IMAGE_SIZE, IMAGE_SIZE);
                }
            }
        }
        return new ByteSource(pixels, IMAGE_SIZE, labels, 10);
    }
}
