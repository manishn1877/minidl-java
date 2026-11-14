package minidl;

import java.util.Arrays;

/**
 * The output layer of a language model that SHARES its weights with the input {@link Embedding} ("weight tying").
 * The embedding turns token ids into vectors (a V x D table E). Instead of learning a second V x D table for the output, the
 * score of token v is the dot product of the final vector with the SAME row E[v]: logits = x . E^T.
 * Benefits: V x D fewer weights (for a 30,000-token vocabulary and D = 64 that is almost 2 million), and tokens that mean similar things
 * get similar input AND output vectors, which usually trains better. There is no bias.
 *
 * How the sharing works here: the Embedding owns the table (so the optimizer updates it once). This layer reads it, and in backward it
 * keeps its share of the gradient for E; the Embedding adds that share to its own after it has cleared its gradient.
 * When a model is loaded from a file, ModelIO calls {@link #bindTo} to connect the two again.
 */
public final class TiedOutput implements Layer {
    private final int D, V;
    private Embedding emb;
    private Tensor x;
    float[] gE; // this layer's share of the gradient of the embedding table (V x D), refreshed by every backward

    public TiedOutput(Embedding emb) {
        this.emb = emb;
        this.V = emb.vocab();
        this.D = emb.dim();
        emb.tie(this);
    }

    /** For loading from a file: the layer is first built without its table (size from the description), then connected. */
    TiedOutput(int vocab, int dim) { this.V = vocab; this.D = dim; }

    void bindTo(Embedding e) {
        if (e.vocab() != V || e.dim() != D) throw new IllegalArgumentException("the embedding is " + e.vocab() + " x " + e.dim() + " but the tied output expects " + V + " x " + D);
        emb = e;
        e.tie(this);
    }

    private Tensor table() {
        if (emb == null) throw new IllegalStateException("TiedOutput is not connected to an Embedding (load the whole model with ModelIO)");
        return emb.tokenTable();
    }

    @Override public Tensor forward(Tensor in) {
        if (in.cols != D) throw new IllegalArgumentException("TiedOutput expects " + D + " numbers per row, got " + in.cols);
        x = in;
        float[] e = table().data;
        Tensor out = new Tensor(in.rows, V);
        for (int n = 0; n < in.rows; n++) {
            int xo = n * D, oo = n * V;
            for (int v = 0; v < V; v++) {
                double s = 0;
                int eo = v * D;
                for (int d = 0; d < D; d++) s += in.data[xo + d] * e[eo + d];
                out.data[oo + v] = (float) s;
            }
        }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        float[] e = table().data;
        int n = g.rows;
        Tensor dx = new Tensor(n, D);
        gE = new float[V * D];
        double[] acc = new double[V * D];
        for (int r = 0; r < n; r++) {
            int go = r * V, xo = r * D;
            for (int v = 0; v < V; v++) {
                float gv = g.data[go + v];
                if (gv == 0f) continue;
                int eo = v * D;
                for (int d = 0; d < D; d++) { dx.data[xo + d] += gv * e[eo + d]; acc[eo + d] += gv * x.data[xo + d]; }
            }
        }
        for (int i = 0; i < acc.length; i++) gE[i] = (float) acc[i];
        return dx;
    }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != D) throw new IllegalArgumentException("TiedOutput expects " + D + " numbers but gets " + Arrays.toString(in));
        return new int[]{V};
    }

    @Override public String spec() { return "TiedOutput " + V + " " + D; }
}
