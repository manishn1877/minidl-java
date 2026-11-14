package minidl;

import java.util.List;
import java.util.Random;

/**
 * Turns token numbers (letters / words as ids) into vectors the network can work with.
 * Each id has its own learned vector (a row of `tok`), and each position 0..T-1 has one too (`pos`),
 * so the model knows the ORDER of the tokens. Output = tok[id] + pos[position].
 *
 * Input:  (N x T), each number is a whole-number id.   Output: (N x T*D).
 * Must be the FIRST layer of a model (ids have no gradient to pass back).
 */
public class Embedding implements Layer {
    private final int V, D, T;
    private final Param tok, pos;
    private Tensor ids;
    private TiedOutput tied; // an output layer that shares the token table (weight tying), or null

    public Embedding(int vocab, int dim, int steps, Random rnd) {
        this.V = vocab; this.D = dim; this.T = steps;
        tok = new Param(Tensor.randn(vocab, dim, 0.1, rnd), false);
        pos = new Param(Tensor.randn(steps, dim, 0.1, rnd), false);
    }

    int vocab() { return V; }
    int dim() { return D; }
    Tensor tokenTable() { return tok.value; }
    void tie(TiedOutput t) { this.tied = t; }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != T) throw new IllegalArgumentException("Embedding expects " + T + " ids per row, got " + x.cols);
        ids = x;
        Tensor out = new Tensor(x.rows, T * D);
        for (int n = 0; n < x.rows; n++)
            for (int t = 0; t < T; t++) {
                int id = (int) x.data[n * T + t];
                if (id < 0 || id >= V) throw new IllegalArgumentException("token id " + id + " is outside 0.." + (V - 1));
                int o = (n * T + t) * D, a = id * D, b = t * D;
                for (int d = 0; d < D; d++) out.data[o + d] = tok.value.data[a + d] + pos.value.data[b + d];
            }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        java.util.Arrays.fill(tok.grad.data, 0f);
        java.util.Arrays.fill(pos.grad.data, 0f);
        for (int n = 0; n < g.rows; n++)
            for (int t = 0; t < T; t++) {
                int id = (int) ids.data[n * T + t];
                int o = (n * T + t) * D, a = id * D, b = t * D;
                for (int d = 0; d < D; d++) { tok.grad.data[a + d] += g.data[o + d]; pos.grad.data[b + d] += g.data[o + d]; }
            }
        if (tied != null && tied.gE != null) for (int i = 0; i < tok.grad.data.length; i++) tok.grad.data[i] += tied.gE[i]; // the output layer's share (its backward ran first)
        return null; // nothing before the embedding needs a gradient
    }

    @Override public List<Param> params() { return List.of(tok, pos); }
    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != T) throw new IllegalArgumentException("Embedding expects " + T + " token ids per sample but gets " + java.util.Arrays.toString(in));
        return new int[]{T, D};
    }

    @Override public String spec() { return "Embedding " + V + " " + D + " " + T; }
}
