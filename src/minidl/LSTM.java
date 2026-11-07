package minidl;

import java.util.List;
import java.util.Random;

/**
 * LSTM (long short-term memory): a recurrent layer that reads a sequence one step at a time and carries a memory along.
 *
 * At every step it makes four "gates" from the current input x and the previous output h:
 *     i = sigmoid(...)   how much NEW information to write into the memory
 *     f = sigmoid(...)   how much of the OLD memory to keep
 *     g = tanh(...)      the new candidate information
 *     o = sigmoid(...)   how much of the memory to show as output
 *     memory c = f * (old memory) + i * g          output h = o * tanh(c)
 * Because the memory is mostly copied forward (f close to 1), the blame (gradient) can travel back through many steps.
 *
 * Input: (N x T*D)  (N sequences of T steps with D numbers each).   Output: (N x T*H), the output h at every step.
 * Backward is "backpropagation through time": the same rules, walked from the last step to the first.
 */
public class LSTM implements Layer {
    private final int T, D, H;
    private final Param Wx, Wh, b;
    private boolean inputGrad = true;

    // remembered for backward
    private Tensor input;
    private float[] gates;   // T x N x 4H : the four gates after their activation (i, f, g, o)
    private float[] cells;   // (T+1) x N x H : memory c_0 (zeros) .. c_T
    private float[] hids;    // (T+1) x N x H : output h_0 (zeros) .. h_T

    public LSTM(int steps, int dim, int hidden, Random rnd) {
        this.T = steps; this.D = dim; this.H = hidden;
        double std = Math.sqrt(1.0 / (dim + hidden));
        Wx = new Param(Tensor.randn(dim, 4 * hidden, std, rnd), true);
        Wh = new Param(Tensor.randn(hidden, 4 * hidden, std, rnd), true);
        b = new Param(new Tensor(1, 4 * hidden), false);
        for (int k = 0; k < hidden; k++) b.value.data[hidden + k] = 1f; // start with "keep the memory" (forget bias 1)
    }

    private static float sigmoid(double z) { return (float) (1.0 / (1.0 + Math.exp(-z))); }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != T * D) throw new IllegalArgumentException("LSTM expects " + T + " steps x " + D + " = " + T * D + " numbers per sequence, got " + x.cols);
        input = x;
        int N = x.rows, H4 = 4 * H;
        Tensor xv = new Tensor(N * T, D, x.data);
        Tensor pre = xv.matmul(Wx.value);                // the input part of every gate at every step, in one big multiply
        pre.addRowVectorInPlace(b.value);
        gates = new float[T * N * H4];
        cells = new float[(T + 1) * N * H];
        hids = new float[(T + 1) * N * H];
        Tensor out = new Tensor(N, T * H);
        for (int t = 0; t < T; t++) {
            Tensor hp = new Tensor(N, H);
            System.arraycopy(hids, t * N * H, hp.data, 0, N * H);
            Tensor rec = hp.matmul(Wh.value);            // the previous-output part
            for (int n = 0; n < N; n++) {
                int pb = (n * T + t) * H4, rb = n * H4, gb = (t * N + n) * H4;
                for (int k = 0; k < H; k++) {
                    float gi = sigmoid(pre.data[pb + k] + rec.data[rb + k]);
                    float gf = sigmoid(pre.data[pb + H + k] + rec.data[rb + H + k]);
                    float gg = (float) Math.tanh(pre.data[pb + 2 * H + k] + rec.data[rb + 2 * H + k]);
                    float go = sigmoid(pre.data[pb + 3 * H + k] + rec.data[rb + 3 * H + k]);
                    gates[gb + k] = gi; gates[gb + H + k] = gf; gates[gb + 2 * H + k] = gg; gates[gb + 3 * H + k] = go;
                    float c = gf * cells[(t * N + n) * H + k] + gi * gg;
                    cells[((t + 1) * N + n) * H + k] = c;
                    float h = go * (float) Math.tanh(c);
                    hids[((t + 1) * N + n) * H + k] = h;
                    out.data[n * T * H + t * H + k] = h;
                }
            }
        }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        int N = g.rows, H4 = 4 * H;
        float[] dZ = new float[N * T * H4];              // blame on the pre-activation of every gate, row order (n, t)
        float[] dhNext = new float[N * H], dcNext = new float[N * H];
        java.util.Arrays.fill(Wh.grad.data, 0f);
        for (int t = T - 1; t >= 0; t--) {
            Tensor dz = new Tensor(N, H4);
            for (int n = 0; n < N; n++) {
                int gb = (t * N + n) * H4;
                for (int k = 0; k < H; k++) {
                    float gi = gates[gb + k], gf = gates[gb + H + k], gg = gates[gb + 2 * H + k], go = gates[gb + 3 * H + k];
                    float c = cells[((t + 1) * N + n) * H + k], cprev = cells[(t * N + n) * H + k];
                    float tc = (float) Math.tanh(c);
                    float dh = g.data[n * T * H + t * H + k] + dhNext[n * H + k];
                    float dO = dh * tc;
                    float dc = dh * go * (1 - tc * tc) + dcNext[n * H + k];
                    dcNext[n * H + k] = dc * gf;
                    int zb = n * H4;
                    dz.data[zb + k] = dc * gg * gi * (1 - gi);                 // input gate
                    dz.data[zb + H + k] = dc * cprev * gf * (1 - gf);          // forget gate
                    dz.data[zb + 2 * H + k] = dc * gi * (1 - gg * gg);         // candidate
                    dz.data[zb + 3 * H + k] = dO * go * (1 - go);              // output gate
                }
                System.arraycopy(dz.data, n * H4, dZ, (n * T + t) * H4, H4);
            }
            Tensor hp = new Tensor(N, H);
            System.arraycopy(hids, t * N * H, hp.data, 0, N * H);
            Wh.grad.addInPlace(hp.matmulTA(dz));          // dWh += h_prev^T . dz
            dhNext = dz.matmulTB(Wh.value).data;          // blame passed to the previous step's output
        }
        Tensor dZt = new Tensor(N * T, H4, dZ);
        Tensor xv = new Tensor(N * T, D, input.data);
        xv.matmulTAInto(dZt, Wx.grad);
        dZt.sumRowsInto(b.grad);
        if (!inputGrad) return null;
        Tensor dx = dZt.matmulTB(Wx.value);
        return new Tensor(N, T * D, dx.data);
    }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != T * D) throw new IllegalArgumentException("LSTM expects " + T + " steps x " + D + " = " + T * D + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return new int[]{T, H};
    }

    @Override public void setComputeInputGrad(boolean on) { inputGrad = on; }
    @Override public List<Param> params() { return List.of(Wx, Wh, b); }
    @Override public String spec() { return "LSTM " + T + " " + D + " " + H; }
}
