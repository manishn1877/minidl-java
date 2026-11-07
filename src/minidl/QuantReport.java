package minidl;

import java.util.Random;

/** Loads a saved binary MNIST model and prints the shrink table.  java -cp out minidl.QuantReport model.bin <mnist-folder> */
public class QuantReport {
    public static void main(String[] a) throws Exception {
        Sequential m = ModelIO.loadModelBinary(a[0], new Random(1));
        String dir = a[1];
        java.io.File gz = new java.io.File(dir, "t10k-images-idx3-ubyte.gz");
        String img = gz.exists() ? gz.getPath() : new java.io.File(dir, "t10k-images-idx3-ubyte").getPath();
        String lab = new java.io.File(dir, "t10k-labels-idx1-ubyte" + (gz.exists() ? ".gz" : "")).getPath();
        ByteSource test = MnistLoader.loadSource(img, lab, Integer.MAX_VALUE);
        System.out.print(m.summary());
        MnistDemo.quantTable(m, test);
    }
}
