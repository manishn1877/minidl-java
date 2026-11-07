package minidl;

/** Re-cuts the same numbers into rows of `cols` numbers (no copying). Example: (N x T*V) -> (N*T x V). */
public class Reshape implements Layer {
    private final int cols;
    private int inCols = -1;

    public Reshape(int cols) { this.cols = cols; }

    @Override public Tensor forward(Tensor x) {
        if ((long) x.rows * x.cols % cols != 0) throw new IllegalArgumentException("cannot cut " + x.shape() + " into rows of " + cols);
        inCols = x.cols;
        return new Tensor(x.rows * x.cols / cols, cols, x.data);
    }

    @Override public Tensor backward(Tensor g) {
        return new Tensor(g.rows * g.cols / inCols, inCols, g.data);
    }

    @Override public int[] outputShape(int[] in) {
        int total = NDTensor.product(in);
        if (total % cols != 0) throw new IllegalArgumentException("Reshape(" + cols + ") cannot cut " + java.util.Arrays.toString(in) + " = " + total + " numbers into rows of " + cols);
        return new int[]{total / cols, cols};
    }

    @Override public String spec() { return "Reshape " + cols; }
}
