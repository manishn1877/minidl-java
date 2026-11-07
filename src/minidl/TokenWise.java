package minidl;

import java.util.ArrayList;
import java.util.List;

/**
 * Applies layers to every TOKEN (word / character position) separately.
 * A sequence model keeps one row per sequence: (N x T*D). TokenWise looks at the same numbers as
 * (N*T x D), runs ordinary layers (Dense, LayerNorm, ...) on every token, and goes back.
 * No data is copied: both views share the same array.
 */
public class TokenWise implements Layer {
    private final int T, inD;
    private final List<Layer> inner;
    private int outD = -1;

    public TokenWise(int T, int inD, List<Layer> inner) { this.T = T; this.inD = inD; this.inner = new ArrayList<>(inner); }
    public TokenWise(int T, int inD, Layer... layers) { this(T, inD, List.of(layers)); }

    List<Layer> inner() { return inner; }
    int steps() { return T; }
    int inDim() { return inD; }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != T * inD) throw new IllegalArgumentException("TokenWise expects " + T * inD + " numbers per sequence, got " + x.cols);
        Tensor h = new Tensor(x.rows * T, inD, x.data);
        for (Layer l : inner) h = l.forward(h);
        outD = h.cols;
        return new Tensor(x.rows, T * outD, h.data);
    }

    @Override public Tensor backward(Tensor g) {
        Tensor gv = new Tensor(g.rows * T, outD, g.data);
        for (int i = inner.size() - 1; i >= 0; i--) gv = inner.get(i).backward(gv);
        return new Tensor(g.rows, T * inD, gv.data);
    }

    @Override public void setComputeInputGrad(boolean on) { /* inner layers decide */ }
    @Override public void setTraining(boolean t) { for (Layer l : inner) l.setTraining(t); }
    @Override public void setActivationStorage(Precision p) { for (Layer l : inner) l.setActivationStorage(p); }

    @Override public List<Param> params() {
        List<Param> all = new ArrayList<>();
        for (Layer l : inner) all.addAll(l.params());
        return all;
    }

    @Override public List<Tensor> state() {
        List<Tensor> all = new ArrayList<>();
        for (Layer l : inner) all.addAll(l.state());
        return all;
    }

    @Override public long weightBytes() { long b = 0; for (Layer l : inner) b += l.weightBytes(); return b; }
    @Override public void writeExtra(java.io.DataOutput o) throws java.io.IOException { for (Layer l : inner) l.writeExtra(o); }
    @Override public void readExtra(java.io.DataInput i) throws java.io.IOException { for (Layer l : inner) l.readExtra(i); }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != T * inD) throw new IllegalArgumentException("TokenWise expects " + T + " tokens x " + inD + " = " + T * inD + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        int[] s = {inD};
        for (Layer l : inner) s = l.outputShape(s);
        int[] out = new int[s.length + 1];
        out[0] = T;
        System.arraycopy(s, 0, out, 1, s.length);
        return out;
    }

    @Override public String spec() {
        StringBuilder sb = new StringBuilder("TokenWise " + T + " " + inD + " [");
        for (int i = 0; i < inner.size(); i++) sb.append(i > 0 ? " | " : "").append(inner.get(i).spec());
        return sb.append("]").toString();
    }
}
