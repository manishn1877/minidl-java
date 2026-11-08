package minidl;

/** Data that already lives in two Tensors. */
public class TensorSource implements DataSource {
    private final Tensor x, y;
    public TensorSource(Tensor x, Tensor y) {
        if (x.rows != y.rows) throw new IllegalArgumentException("x and y need the same number of rows");
        this.x = x; this.y = y;
    }
    @Override public int size() { return x.rows; }
    @Override public Tensor features(int[] idx, int from, int to) { return x.selectRows(idx, from, to); }
    @Override public Tensor targets(int[] idx, int from, int to) { return y.selectRows(idx, from, to); }
}
