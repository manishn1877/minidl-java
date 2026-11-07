package minidl;

import java.io.*;
import java.util.*;

/** Loading and preparing data: CSV files, one-hot labels, train/test split. */
public final class Data {
    private Data() {}

    /** x + y + class names (for text labels like "cat", "dog"). */
    public static class Dataset {
        public final Tensor x, y;
        public final String[] classNames;
        Dataset(Tensor x, Tensor y, String[] classNames) { this.x = x; this.y = y; this.classNames = classNames; }
    }

    /**
     * Numeric CSV. The LAST `labelCols` columns are the answer (y), the rest are features (x).
     * Plain comma separated, no quotes. Returns {x, y}.
     */
    public static Tensor[] loadCsvXY(String path, boolean hasHeader, int labelCols) throws IOException {
        List<double[]> rows = readNumericRows(path, hasHeader);
        if (rows.isEmpty()) throw new IOException("no data rows in " + path);
        int cols = rows.get(0).length;
        if (labelCols <= 0 || labelCols >= cols) throw new IOException("labelCols must be between 1 and " + (cols - 1));
        int fc = cols - labelCols;
        Tensor x = new Tensor(rows.size(), fc), y = new Tensor(rows.size(), labelCols);
        for (int i = 0; i < rows.size(); i++) {
            double[] r = rows.get(i);
            for (int j = 0; j < fc; j++) x.data[i * fc + j] = (float) r[j];
            for (int j = 0; j < labelCols; j++) y.data[i * labelCols + j] = (float) r[fc + j];
        }
        return new Tensor[]{x, y};
    }

    /**
     * Classification CSV: numeric feature columns, and the LAST column is a text label.
     * Class names are sorted alphabetically; y comes back one-hot.
     */
    public static Dataset loadCsvClassification(String path, boolean hasHeader) throws IOException {
        List<double[]> feats = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String line; int lineNo = 0; boolean headerSkipped = !hasHeader;
            while ((line = r.readLine()) != null) {
                lineNo++;
                line = line.trim();
                if (line.isEmpty()) continue;
                if (!headerSkipped) { headerSkipped = true; continue; }
                String[] t = line.split(",");
                double[] f = new double[t.length - 1];
                for (int j = 0; j < f.length; j++) f[j] = parse(t[j], lineNo, j + 1);
                feats.add(f);
                labels.add(t[t.length - 1].trim());
            }
        }
        if (feats.isEmpty()) throw new IOException("no data rows in " + path);
        TreeSet<String> names = new TreeSet<>(labels);
        String[] classNames = names.toArray(new String[0]);
        Map<String, Integer> id = new HashMap<>();
        for (int i = 0; i < classNames.length; i++) id.put(classNames[i], i);

        int fc = feats.get(0).length;
        Tensor x = new Tensor(feats.size(), fc), y = new Tensor(feats.size(), classNames.length);
        for (int i = 0; i < feats.size(); i++) {
            for (int j = 0; j < fc; j++) x.data[i * fc + j] = (float) feats.get(i)[j];
            y.set(i, id.get(labels.get(i)), 1.0);
        }
        return new Dataset(x, y, classNames);
    }

    /** [2, 0, 1] with 3 classes -> [[0,0,1],[1,0,0],[0,1,0]] */
    public static Tensor oneHot(int[] labels, int classes) {
        Tensor y = new Tensor(labels.length, classes);
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] < 0 || labels[i] >= classes) throw new IllegalArgumentException("label " + labels[i] + " out of range");
            y.set(i, labels[i], 1.0);
        }
        return y;
    }

    /** A (n x 1) column of class numbers -> one-hot. */
    public static Tensor oneHotColumn(Tensor col, int classes) {
        int[] l = new int[col.rows];
        for (int i = 0; i < l.length; i++) l[i] = (int) Math.round(col.get(i, 0));
        return oneHot(l, classes);
    }

    /** Shuffles, then splits. Returns {xTrain, yTrain, xTest, yTest}. */
    public static Tensor[] trainTestSplit(Tensor x, Tensor y, double testFraction, long seed) {
        if (testFraction <= 0 || testFraction >= 1) throw new IllegalArgumentException("testFraction must be in (0,1)");
        int n = x.rows;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Random rnd = new Random(seed);
        for (int i = n - 1; i > 0; i--) { int j = rnd.nextInt(i + 1); int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        int nTest = Math.max(1, (int) Math.round(n * testFraction));
        return new Tensor[]{
            x.selectRows(idx, nTest, n), y.selectRows(idx, nTest, n),
            x.selectRows(idx, 0, nTest), y.selectRows(idx, 0, nTest)};
    }

    private static List<double[]> readNumericRows(String path, boolean hasHeader) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String line; int lineNo = 0; boolean headerSkipped = !hasHeader;
            while ((line = r.readLine()) != null) {
                lineNo++;
                line = line.trim();
                if (line.isEmpty()) continue;
                if (!headerSkipped) { headerSkipped = true; continue; }
                String[] t = line.split(",");
                if (!rows.isEmpty() && t.length != rows.get(0).length)
                    throw new IOException("line " + lineNo + ": expected " + rows.get(0).length + " columns but found " + t.length);
                double[] v = new double[t.length];
                for (int j = 0; j < t.length; j++) v[j] = parse(t[j], lineNo, j + 1);
                rows.add(v);
            }
        }
        return rows;
    }

    private static double parse(String s, int line, int col) throws IOException {
        try { return Double.parseDouble(s.trim()); }
        catch (NumberFormatException e) { throw new IOException("line " + line + ", column " + col + ": '" + s.trim() + "' is not a number"); }
    }
}
