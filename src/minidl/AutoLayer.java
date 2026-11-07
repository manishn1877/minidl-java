package minidl;

import java.util.ArrayList;
import java.util.List;

/**
 * A layer defined ONLY by its forward maths, written with autograd (Var). There is no hand-written backward():
 * the gradients come from the recorded expression. It plugs into Sequential, Trainer, optimizers, saving,
 * quantizing... like any other layer.
 *
 *   Layer relu2 = new AutoLayer("Square", List.of(), (x, p) -> x.mul(x));   // a whole new layer in one line
 *
 * Compared with the hand-written layers it is simpler to write and to trust, but slower and hungrier for memory,
 * because every intermediate result of the expression is kept until backward (see the benchmark).
 */
public class AutoLayer implements Layer {
    /** The forward maths. `params` are the layer's weights as Vars (they share the numbers of the Param objects). */
    public interface Fn { Var apply(Var x, List<Var> params); }

    private final String name;
    private final List<Param> params;
    private final Fn fn;
    private boolean inputGrad = true;
    private Var xVar, outVar;
    private List<Var> pVars;

    public AutoLayer(String name, List<Param> params, Fn fn) {
        this.name = name; this.params = new ArrayList<>(params); this.fn = fn;
    }

    @Override public Tensor forward(Tensor x) {
        xVar = new Var(x);
        pVars = new ArrayList<>();
        for (Param p : params) pVars.add(new Var(p.value)); // same Tensor object: when the optimizer changes the weights, the layer sees it
        outVar = fn.apply(xVar, pVars);
        return outVar.value;
    }

    @Override public Tensor backward(Tensor g) {
        outVar.backward(g);
        for (int i = 0; i < params.size(); i++) {
            Tensor gr = pVars.get(i).grad;
            float[] dst = params.get(i).grad.data;
            if (gr == null) java.util.Arrays.fill(dst, 0f);
            else System.arraycopy(gr.data, 0, dst, 0, dst.length);
        }
        Tensor dx = inputGrad ? (xVar.grad != null ? xVar.grad : new Tensor(xVar.value.rows, xVar.value.cols)) : null;
        outVar = null; xVar = null; pVars = null; // free the recorded expression
        return dx;
    }

    @Override public void setComputeInputGrad(boolean on) { inputGrad = on; }
    @Override public List<Param> params() { return params; }
    @Override public String spec() { return name; }
}
