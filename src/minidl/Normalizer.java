package minidl;

import java.io.*;

/**
 * Puts every column on the same scale (mean 0, spread 1).
 * Without this, a column like "salary" (50000) drowns out "age" (25).
 */
public class Normalizer {
    private double[] mean, std;

    public Normalizer fit(Tensor x) {
        mean = new double[x.cols];
        std = new double[x.cols];
        for (int j = 0; j < x.cols; j++) {
            double s = 0;
            for (int i = 0; i < x.rows; i++) s += x.get(i, j);
            mean[j] = s / x.rows;
            double v = 0;
            for (int i = 0; i < x.rows; i++) v += Math.pow(x.get(i, j) - mean[j], 2);
            std[j] = Math.sqrt(v / x.rows);
            if (std[j] < 1e-8) std[j] = 1.0; // constant column, avoid dividing by 0
        }
        return this;
    }

    public Tensor transform(Tensor x) {
        Tensor out = new Tensor(x.rows, x.cols);
        for (int i = 0; i < x.rows; i++)
            for (int j = 0; j < x.cols; j++)
                out.set(i, j, (x.get(i, j) - mean[j]) / std[j]); // maths in double, stored as float
        return out;
    }

    /** Go back to the original units (e.g. turn the model's answer back into minutes). */
    public Tensor inverse(Tensor x) {
        Tensor out = new Tensor(x.rows, x.cols);
        for (int i = 0; i < x.rows; i++)
            for (int j = 0; j < x.cols; j++)
                out.set(i, j, x.get(i, j) * std[j] + mean[j]);
        return out;
    }

    /** Save the learned mean/std. You need them again when the model is used later. */
    public void save(String path) throws IOException {
        try (PrintWriter w = new PrintWriter(new FileWriter(path))) {
            w.println("NORMALIZER " + mean.length);
            w.println(join(mean));
            w.println(join(std));
        }
    }

    public static Normalizer load(String path) throws IOException {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String[] head = r.readLine().trim().split(" ");
            if (!head[0].equals("NORMALIZER")) throw new IOException("not a Normalizer file");
            int n = Integer.parseInt(head[1]);
            Normalizer z = new Normalizer();
            z.mean = parse(r.readLine(), n);
            z.std = parse(r.readLine(), n);
            return z;
        }
    }

    private static String join(double[] v) {
        StringBuilder sb = new StringBuilder();
        for (double d : v) sb.append(d).append(' ');
        return sb.toString().trim();
    }

    private static double[] parse(String line, int n) throws IOException {
        String[] t = line.trim().split(" ");
        if (t.length != n) throw new IOException("corrupt Normalizer file");
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = Double.parseDouble(t[i]);
        return v;
    }
}
