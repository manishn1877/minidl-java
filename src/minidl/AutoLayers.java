package minidl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Layers written with autograd. Each one is a few lines of FORWARD maths; the backward pass comes for free.
 * They give the same results and gradients as the hand-written Dense, LayerNorm, GELU and CausalSelfAttention
 * (the tests check that), and they are the easiest place to start when you invent a new layer.
 */
public final class AutoLayers {
    private AutoLayers() {}

    static {
        ModelIO.register("AutoDense", (t, rnd) -> dense(Integer.parseInt(t[1]), Integer.parseInt(t[2]), rnd));
        ModelIO.register("AutoLayerNorm", (t, rnd) -> layerNorm(Integer.parseInt(t[1])));
        ModelIO.register("AutoGELU", (t, rnd) -> gelu());
        ModelIO.register("AutoAttention", (t, rnd) -> attention(Integer.parseInt(t[1]), Integer.parseInt(t[2]), Integer.parseInt(t[3]), rnd));
    }

    /** out = x . W + b */
    public static Layer dense(int in, int out, Random rnd) {
        Param W = new Param(Tensor.randn(in, out, Math.sqrt(2.0 / in), rnd), true), b = new Param(new Tensor(1, out), false);
        return new AutoLayer("AutoDense " + in + " " + out, List.of(W, b), (x, p) -> x.matmul(p.get(0)).addRow(p.get(1)));
    }

    /** Each row: subtract its mean, divide by its spread, then scale and shift. */
    public static Layer layerNorm(int features) {
        Tensor ones = new Tensor(1, features);
        java.util.Arrays.fill(ones.data, 1f);
        Param gamma = new Param(ones, false), beta = new Param(new Tensor(1, features), false);
        return new AutoLayer("AutoLayerNorm " + features, List.of(gamma, beta), (x, p) -> {
            Var centered = x.subCol(x.rowMean());
            Var std = centered.mul(centered).rowMean().shift(1e-5).sqrt();
            return centered.divCol(std).mulRow(p.get(0)).addRow(p.get(1));
        });
    }

    /** GELU built from tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))). */
    public static Layer gelu() {
        final double c = Math.sqrt(2.0 / Math.PI);
        return new AutoLayer("AutoGELU", List.of(), (x, p) -> {
            Var cubed = x.mul(x).mul(x);
            Var t = x.add(cubed.scale(0.044715)).scale(c).tanh();
            return x.mul(t.shift(1)).scale(0.5);
        });
    }

    /**
     * Multi-head causal self-attention, written out step by step for every sequence and head:
     * scores = Q . K^T / sqrt(dh) (+ a mask that hides later tokens), softmax, then mix the values.
     * Input and output: (N x T*D). Same weights layout as CausalSelfAttention.
     */
    public static Layer attention(int T, int D, int H, Random rnd) {
        if (D % H != 0) throw new IllegalArgumentException("dim must be divisible by heads");
        final int dh = D / H;
        double std = Math.sqrt(1.0 / D);
        Param Wqkv = new Param(Tensor.randn(D, 3 * D, std, rnd), true), bqkv = new Param(new Tensor(1, 3 * D), false);
        Param Wo = new Param(Tensor.randn(D, D, std, rnd), true), bo = new Param(new Tensor(1, D), false);
        final Tensor mask = new Tensor(T, T);                       // 0 where a token may look, -1e9 where it may not (later tokens)
        for (int i = 0; i < T; i++) for (int j = i + 1; j < T; j++) mask.data[i * T + j] = -1e9f;
        return new AutoLayer("AutoAttention " + T + " " + D + " " + H, List.of(Wqkv, bqkv, Wo, bo), (x, p) -> {
            int n = x.value.rows;
            Var qkv = x.reshape(n * T, D).matmul(p.get(0)).addRow(p.get(1));     // all tokens at once
            List<Var> samples = new ArrayList<>();
            for (int s = 0; s < n; s++) {
                Var rows = qkv.sliceRows(s * T, (s + 1) * T);                      // this sequence: T x 3D
                List<Var> heads = new ArrayList<>();
                for (int h = 0; h < H; h++) {
                    Var q = rows.sliceCols(h * dh, (h + 1) * dh);
                    Var k = rows.sliceCols(D + h * dh, D + (h + 1) * dh);
                    Var v = rows.sliceCols(2 * D + h * dh, 2 * D + (h + 1) * dh);
                    Var scores = q.matmul(k.transpose()).scale(1.0 / Math.sqrt(dh)).add(new Var(mask));
                    heads.add(scores.rowSoftmax().matmul(v));
                }
                samples.add(Var.concatCols(heads));                                // T x D
            }
            return Var.concatRows(samples).matmul(p.get(2)).addRow(p.get(3)).reshape(n, T * D);
        });
    }
}
