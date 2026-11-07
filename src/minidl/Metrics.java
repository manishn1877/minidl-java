package minidl;

/** Ways to measure how good a model is. */
public final class Metrics {
    private Metrics() {}

    /** Mean absolute error: "on average we are off by this much" (same units as your target). */
    public static double mae(Tensor pred, Tensor target) {
        double s = 0;
        for (int i = 0; i < pred.data.length; i++) s += Math.abs(pred.data[i] - target.data[i]);
        return s / pred.data.length;
    }

    /** Root mean squared error: like MAE but punishes big misses more. */
    public static double rmse(Tensor pred, Tensor target) {
        double s = 0;
        for (int i = 0; i < pred.data.length; i++) { double d = pred.data[i] - target.data[i]; s += d * d; }
        return Math.sqrt(s / pred.data.length);
    }

    /** R squared: 1.0 = perfect, 0.0 = no better than always guessing the average. */
    public static double r2(Tensor pred, Tensor target) {
        double mean = 0;
        for (float v : target.data) mean += v;
        mean /= target.data.length;
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < pred.data.length; i++) {
            ssRes += Math.pow(target.data[i] - pred.data[i], 2);
            ssTot += Math.pow(target.data[i] - mean, 2);
        }
        return ssTot == 0 ? 0 : 1 - ssRes / ssTot;
    }

    /** cm[truth][predicted] = how many rows. Both tensors are scores/one-hot (argmax per row). */
    public static int[][] confusionMatrix(Tensor scores, Tensor oneHot) {
        int c = oneHot.cols;
        int[][] cm = new int[c][c];
        for (int i = 0; i < scores.rows; i++) cm[oneHot.argmaxRow(i)][scores.argmaxRow(i)]++;
        return cm;
    }

    public static String formatConfusion(int[][] cm) {
        StringBuilder sb = new StringBuilder("truth \\ predicted\n      ");
        for (int j = 0; j < cm.length; j++) sb.append(String.format("%5d", j));
        sb.append('\n');
        for (int i = 0; i < cm.length; i++) {
            sb.append(String.format("%5d ", i));
            for (int j = 0; j < cm.length; j++) sb.append(String.format("%5d", cm[i][j]));
            sb.append('\n');
        }
        return sb.toString();
    }

    /** Per class: precision (when I say X, how often am I right), recall (of all real X, how many did I find), F1. */
    public static String classificationReport(int[][] cm) {
        StringBuilder sb = new StringBuilder(String.format("%-6s %9s %9s %9s %8s%n", "class", "precision", "recall", "f1", "count"));
        for (int k = 0; k < cm.length; k++) {
            int tp = cm[k][k], fp = 0, fn = 0;
            for (int i = 0; i < cm.length; i++) { if (i != k) { fp += cm[i][k]; fn += cm[k][i]; } }
            double p = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
            double r = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
            double f1 = p + r == 0 ? 0 : 2 * p * r / (p + r);
            sb.append(String.format("%-6d %9.3f %9.3f %9.3f %8d%n", k, p, r, f1, tp + fn));
        }
        return sb.toString();
    }
}
