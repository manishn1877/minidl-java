package minidl;

/** Data that already lives in two Tensors. */
public class TensorSource implements DataSource {
    private final Tensor x, y;
    private final int targetRowsPerSample;

    public TensorSource(Tensor x, Tensor y) { this(x, y, 1); }

    /**
     * For sequence models: every sample (row of x) has `targetRowsPerSample` rows of targets in y
     * (one answer per position), stored one sample after the other.
     */
    public TensorSource(Tensor x, Tensor y, int targetRowsPerSample) {
        if (targetRowsPerSample < 1 || (long) x.rows * targetRowsPerSample != y.rows)
            throw new IllegalArgumentException("x and y need the same number of rows" + (targetRowsPerSample > 1 ? " times " + targetRowsPerSample : ""));
        this.x = x; this.y = y; this.targetRowsPerSample = targetRowsPerSample;
    }

    @Override public int size() { return x.rows; }
    @Override public Tensor features(int[] idx, int from, int to) { return x.selectRows(idx, from, to); }

    @Override public Tensor targets(int[] idx, int from, int to) {
        if (targetRowsPerSample == 1) return y.selectRows(idx, from, to);
        int k = targetRowsPerSample, cols = y.cols;
        Tensor out = new Tensor((to - from) * k, cols);
        for (int r = from; r < to; r++) System.arraycopy(y.data, idx[r] * k * cols, out.data, (r - from) * k * cols, k * cols);
        return out;
    }
}
