package minidl;

import java.util.ArrayList;
import java.util.List;

/**
 * Skip connection: output = input + f(input), where f is a small stack of layers.
 * The input is added back on, so the blame (gradient) has a straight shortcut to earlier layers.
 * That is what lets networks with dozens of layers train (ResNets, transformers).
 * f must give back the same number of columns it received.
 */
public class Residual implements Layer {
    private final List<Layer> inner;

    public Residual(List<Layer> inner) { this.inner = new ArrayList<>(inner); }
    public Residual(Layer... layers) { this(List.of(layers)); }

    List<Layer> inner() { return inner; }

    @Override public Tensor forward(Tensor x) {
        Tensor h = x;
        for (Layer l : inner) h = l.forward(h);
        if (h.rows != x.rows || h.cols != x.cols)
            throw new IllegalArgumentException("Residual block must keep the shape: got " + h.shape() + " from " + x.shape());
        return x.add(h);
    }

    @Override public Tensor backward(Tensor g) {
        Tensor gi = g;
        for (int i = inner.size() - 1; i >= 0; i--) gi = inner.get(i).backward(gi);
        return g.add(gi); // the shortcut carries g straight through, plus whatever came back through f
    }

    @Override public void setComputeInputGrad(boolean on) { /* always needed: the shortcut passes it on */ }
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
        int[] s = in;
        for (Layer l : inner) s = l.outputShape(s);
        if (NDTensor.product(s) != NDTensor.product(in)) throw new IllegalArgumentException("Residual block must keep the size: it turns " + java.util.Arrays.toString(in) + " into " + java.util.Arrays.toString(s));
        return in;
    }

    @Override public String spec() {
        StringBuilder sb = new StringBuilder("Residual [");
        for (int i = 0; i < inner.size(); i++) sb.append(i > 0 ? " | " : "").append(inner.get(i).spec());
        return sb.append("]").toString();
    }
}
