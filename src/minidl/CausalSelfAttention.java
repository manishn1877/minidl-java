package minidl;

import java.util.List;
import java.util.Random;

/**
 * Multi-head causal self-attention: the core of a transformer.
 *
 * Every token asks a question (query), every token offers a label (key) and some content (value).
 * A token compares its query with the keys of all EARLIER tokens (and itself), turns the scores into
 * percentages (softmax), and takes that mix of their values. "Causal" means no peeking at later tokens,
 * which is what lets the model learn to predict the next letter or word.
 * Several "heads" do this side by side, each on its own slice of the numbers.
 *
 * Input / output shape: (N x T*D)  (N sequences, T tokens, D numbers per token).
 *
 * Memory saving: the T x T attention table is never stored. Backward rebuilds it from the stored
 * queries and keys (the same idea the FlashAttention method is built on).
 */
public class CausalSelfAttention implements Layer {
    private final int T, D, H, dh;
    private final Param Wqkv, bqkv, Wo, bo;
    private Tensor input, qkv, att; // remembered for backward

    public CausalSelfAttention(int steps, int dim, int heads, Random rnd) {
        if (dim % heads != 0) throw new IllegalArgumentException("dim must be divisible by heads");
        this.T = steps; this.D = dim; this.H = heads; this.dh = dim / heads;
        double std = Math.sqrt(1.0 / dim);
        Wqkv = new Param(Tensor.randn(dim, 3 * dim, std, rnd), true);
        bqkv = new Param(new Tensor(1, 3 * dim), false);
        Wo = new Param(Tensor.randn(dim, dim, std, rnd), true);
        bo = new Param(new Tensor(1, dim), false);
    }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != T * D) throw new IllegalArgumentException("attention expects " + T * D + " numbers per sequence, got " + x.cols);
        input = x;
        int N = x.rows, NT = N * T;
        Tensor xv = new Tensor(NT, D, x.data);
        qkv = xv.matmul(Wqkv.value);
        qkv.addRowVectorInPlace(bqkv.value);
        att = new Tensor(NT, D);
        final float scale = (float) (1.0 / Math.sqrt(dh));
        Tensor.forRows(N, (long) N * H * T * T * dh, n -> {
            float[] p = new float[T * T];
            for (int h = 0; h < H; h++) {
                probs(n, h, scale, p);
                for (int i = 0; i < T; i++) {
                    int ob = (n * T + i) * D + h * dh;
                    for (int j = 0; j <= i; j++) {
                        float pj = p[i * T + j];
                        int vb = (n * T + j) * 3 * D + 2 * D + h * dh;
                        for (int d = 0; d < dh; d++) att.data[ob + d] += pj * qkv.data[vb + d];
                    }
                }
            }
        });
        Tensor out = att.matmul(Wo.value);
        out.addRowVectorInPlace(bo.value);
        return new Tensor(N, T * D, out.data);
    }

    /** Attention percentages for sample n, head h: p[i*T + j] for j <= i (0 for later tokens). */
    private void probs(int n, int h, float scale, float[] p) {
        for (int i = 0; i < T; i++) {
            int qb = (n * T + i) * 3 * D + h * dh;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j <= i; j++) {
                int kb = (n * T + j) * 3 * D + D + h * dh;
                float s = 0;
                for (int d = 0; d < dh; d++) s += qkv.data[qb + d] * qkv.data[kb + d];
                s *= scale;
                p[i * T + j] = s;
                if (s > max) max = s;
            }
            float sum = 0;
            for (int j = 0; j <= i; j++) { float e = (float) Math.exp(p[i * T + j] - max); p[i * T + j] = e; sum += e; }
            for (int j = 0; j <= i; j++) p[i * T + j] /= sum;
            for (int j = i + 1; j < T; j++) p[i * T + j] = 0f;
        }
    }

    @Override public Tensor backward(Tensor g) {
        int N = g.rows, NT = N * T;
        Tensor gv = new Tensor(NT, D, g.data);
        att.matmulTAInto(gv, Wo.grad);               // dWo
        gv.sumRowsInto(bo.grad);                      // dbo
        Tensor dO = gv.matmulTB(Wo.value);            // blame on the mixed values
        Tensor dqkv = new Tensor(NT, 3 * D);
        final float scale = (float) (1.0 / Math.sqrt(dh));
        Tensor.forRows(N, (long) N * H * T * T * dh * 2, n -> {
            float[] p = new float[T * T], dp = new float[T];
            for (int h = 0; h < H; h++) {
                probs(n, h, scale, p); // rebuilt, not stored
                for (int i = 0; i < T; i++) {
                    int ob = (n * T + i) * D + h * dh;
                    float dot = 0;
                    for (int j = 0; j <= i; j++) {
                        int vb = (n * T + j) * 3 * D + 2 * D + h * dh;
                        float s = 0;
                        for (int d = 0; d < dh; d++) s += dO.data[ob + d] * qkv.data[vb + d];
                        dp[j] = s;
                        dot += p[i * T + j] * s;
                        int dvb = (n * T + j) * 3 * D + 2 * D + h * dh;     // dV[j] += p[i][j] * dO[i]
                        float pj = p[i * T + j];
                        for (int d = 0; d < dh; d++) dqkv.data[dvb + d] += pj * dO.data[ob + d];
                    }
                    int qb = (n * T + i) * 3 * D + h * dh;
                    for (int j = 0; j <= i; j++) {
                        float ds = p[i * T + j] * (dp[j] - dot) * scale;   // softmax backward, then undo the scaling
                        int kb = (n * T + j) * 3 * D + D + h * dh;
                        for (int d = 0; d < dh; d++) {
                            dqkv.data[qb + d] += ds * qkv.data[kb + d];    // dQ[i]
                            dqkv.data[kb + d] += ds * qkv.data[qb + d];    // dK[j]
                        }
                    }
                }
            }
        });
        Tensor xv = new Tensor(NT, D, input.data);
        xv.matmulTAInto(dqkv, Wqkv.grad);
        dqkv.sumRowsInto(bqkv.grad);
        Tensor dx = dqkv.matmulTB(Wqkv.value);
        return new Tensor(N, T * D, dx.data);
    }

    @Override public List<Param> params() { return List.of(Wqkv, bqkv, Wo, bo); }
    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != T * D) throw new IllegalArgumentException("CausalSelfAttention expects " + T + " tokens x " + D + " = " + T * D + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return new int[]{T, D};
    }

    @Override public String spec() { return "CausalSelfAttention " + T + " " + D + " " + H; }
}
