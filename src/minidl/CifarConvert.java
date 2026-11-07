package minidl;

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/**
 * Converts the CIFAR-10 copy that ships inside the npm package "tfjs-cifar10" (PNG sheets + label json) into the
 * official binary files that CifarLoader reads.
 *   java -Xmx1g -cp out minidl.CifarConvert <unpacked package folder> <output folder>
 * In each PNG, row r (1024 pixels wide) is image r: pixel k of the row is pixel (k / 32, k % 32) of the image, as R, G, B.
 * Needs the JDK's image support (java.desktop), which every full JDK has.
 */
public class CifarConvert {
    public static void main(String[] a) throws Exception {
        Path in = Path.of(a[0]), out = Path.of(a[1]);
        Files.createDirectories(out);
        int[] train = labels(in.resolve("train_lables.json")), test = labels(in.resolve("test_lables.json"));
        for (int b = 1; b <= 5; b++) convert(in.resolve("data_batch_" + b + ".png"), train, (b - 1) * 10000, out.resolve("data_batch_" + b + ".bin"));
        convert(in.resolve("test_batch.png"), test, 0, out.resolve("test_batch.bin"));
        System.out.println("wrote 5 training batches and the test batch to " + out);
    }

    static int[] labels(Path json) throws IOException {
        String[] t = Files.readString(json).replaceAll("[\\[\\]\\s]", "").split(",");
        int[] l = new int[t.length];
        for (int i = 0; i < l.length; i++) l[i] = Integer.parseInt(t[i]);
        return l;
    }

    static void convert(Path png, int[] labels, int labelOffset, Path bin) throws IOException {
        BufferedImage img = ImageIO.read(png.toFile());
        if (img.getWidth() != 1024 || img.getHeight() != 10000) throw new IOException(png + ": expected 1024 x 10000 pixels");
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(bin.toFile()), 1 << 16))) {
            byte[] rec = new byte[1 + 3072];
            for (int r = 0; r < 10000; r++) {
                rec[0] = (byte) labels[labelOffset + r];
                for (int k = 0; k < 1024; k++) {
                    int rgb = img.getRGB(k, r);
                    rec[1 + k] = (byte) (rgb >> 16);          // red plane
                    rec[1 + 1024 + k] = (byte) (rgb >> 8);    // green plane
                    rec[1 + 2048 + k] = (byte) rgb;           // blue plane
                }
                o.write(rec);
            }
        }
    }
}
