package minidl;

import java.util.List;
import java.util.Random;

/**
 * Long Short-Term Memory layer: reads a sequence one step at a time and keeps a memory ("cell") that it can write to,
 * keep, or read from through three gates. This lets a model use what it saw many steps ago.
 *
 * Input row  = the whole sequence, flattened as [step][feature]  (T steps of D numbers, so T*D numbers)
 * Output row = the last hidden state (H numbers), or with returnSequences the hidden state of every step (T*H numbers).
 *
 * Per step:  z = x Wx + h Wh + b ;  i,f,o = sigmoid ; g = tanh ;  c = f*c + i*g ;  h = o * tanh(c)
 * (gate order in the weights: input, forget, candidate, output). The forget-gate bias starts at 1, so memory is kept by default.
 * Backward is full backpropagation through time (BPTT), checked against numerical gradients in the tests.
 */
public class LSTM implements Layer {
    private final int T, D, H;
    private final boolean seq;
    private final Param Wx, Wh, b;

    // remembered for backward, all in double: [sample][step]...
    private float[] xin;
    private double[][] gate;   // [n * T][4H] after the activation functions
    private double[][] cell;   // [n * (T+1)][H]
    private double[][] hid;    // [n * (T+1)][H]
    private double[][] tanhC;  // [n * T][H]
    private int n;

    public LSTM(int steps, int features, int hidden, boolean returnSequences, Random rnd) {
        if (steps < 1 || features < 1 || hidden < 1) throw new IllegalArgumentException("bad LSTM size");
        T = steps; D = features; H = hidden; seq = returnSequences;
        Wx = new Param(Tensor.randn(D, 4 * H, Math.sqrt(1.0 / D), rnd), true);
        Wh = new Param(Tensor.randn(H, 4 * H, Math.sqrt(1.0 / H), rnd), true);
        Tensor bias = new Tensor(1, 4 * H);
        for (int j = H; j < 2 * H; j++) bias.data[j] = 1f; // forget gate: remember by default
        b = new Param(bias, false);
    }

    private static double sig(double v) { return 1.0 / (1.0 + Math.exp(-v)); }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != T * D) throw new IllegalArgumentException("LSTM expects " + T + " steps x " + D + " = " + T * D + " numbers per sequence, got " + x.cols);
        n = x.rows;
        xin = x.data;
        gate = new double[n * T][4 * H];
        cell = new double[n * (T + 1)][H];
        hid = new double[n * (T + 1)][H];
        tanhC = new double[n * T][H];
        Tensor out = new Tensor(n, seq ? T * H : H);
        double[] z = new double[4 * H];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < T; t++) {
                double[] hp = hid[i * (T + 1) + t], cp = cell[i * (T + 1) + t];
                for (int j = 0; j < 4 * H; j++) z[j] = b.value.data[j];
                for (int d = 0; d < D; d++) { double xv = x.data[i * T * D + t * D + d]; int row = d * 4 * H; for (int j = 0; j < 4 * H; j++) z[j] += xv * Wx.value.data[row + j]; }
                for (int k = 0; k < H; k++) { double hv = hp[k]; int row = k * 4 * H; for (int j = 0; j < 4 * H; j++) z[j] += hv * Wh.value.data[row + j]; }
                double[] gt = gate[i * T + t], cn = cell[i * (T + 1) + t + 1], hn = hid[i * (T + 1) + t + 1], tc = tanhC[i * T + t];
                for (int k = 0; k < H; k++) {
                    double ig = sig(z[k]), fg = sig(z[H + k]), gg = Math.tanh(z[2 * H + k]), og = sig(z[3 * H + k]);
                    gt[k] = ig; gt[H + k] = fg; gt[2 * H + k] = gg; gt[3 * H + k] = og;
                    cn[k] = fg * cp[k] + ig * gg;
                    tc[k] = Math.tanh(cn[k]);
                    hn[k] = og * tc[k];
                    if (seq) out.data[i * T * H + t * H + k] = (float) hn[k];
                }
            }
            if (!seq) for (int k = 0; k < H; k++) out.data[i * H + k] = (float) hid[i * (T + 1) + T][k];
        }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        float[] dWx = new float[Wx.value.data.length], dWh = new float[Wh.value.data.length], db = new float[4 * H];
        double[] aWx = new double[dWx.length], aWh = new double[dWh.length], ab = new double[4 * H];
        Tensor dx = new Tensor(n, T * D);
        double[] dh = new double[H], dc = new double[H], dz = new double[4 * H], dhPrev = new double[H];
        for (int i = 0; i < n; i++) {
            java.util.Arrays.fill(dh, 0); java.util.Arrays.fill(dc, 0);
            for (int t = T - 1; t >= 0; t--) {
                double[] gt = gate[i * T + t], cp = cell[i * (T + 1) + t], hp = hid[i * (T + 1) + t], tc = tanhC[i * T + t];
                for (int k = 0; k < H; k++) {
                    double dht = dh[k] + (seq ? g.data[i * T * H + t * H + k] : (t == T - 1 ? g.data[i * H + k] : 0));
                    double ig = gt[k], fg = gt[H + k], gg = gt[2 * H + k], og = gt[3 * H + k];
                    double dct = dc[k] + dht * og * (1 - tc[k] * tc[k]);
                    dz[k] = dct * gg * ig * (1 - ig);              // input gate
                    dz[H + k] = dct * cp[k] * fg * (1 - fg);       // forget gate
                    dz[2 * H + k] = dct * ig * (1 - gg * gg);      // candidate
                    dz[3 * H + k] = dht * tc[k] * og * (1 - og);   // output gate
                    dc[k] = dct * fg;                              // goes to step t-1
                }
                java.util.Arrays.fill(dhPrev, 0);
                for (int j = 0; j < 4 * H; j++) ab[j] += dz[j];
                for (int d = 0; d < D; d++) {
                    double xv = xin[i * T * D + t * D + d], acc = 0;
                    int row = d * 4 * H;
                    for (int j = 0; j < 4 * H; j++) { aWx[row + j] += xv * dz[j]; acc += Wx.value.data[row + j] * dz[j]; }
                    dx.data[i * T * D + t * D + d] = (float) acc;
                }
                for (int k = 0; k < H; k++) {
                    int row = k * 4 * H; double acc = 0;
                    for (int j = 0; j < 4 * H; j++) { aWh[row + j] += hp[k] * dz[j]; acc += Wh.value.data[row + j] * dz[j]; }
                    dhPrev[k] = acc;
                }
                System.arraycopy(dhPrev, 0, dh, 0, H);
            }
        }
        for (int j = 0; j < dWx.length; j++) Wx.grad.data[j] = (float) aWx[j];
        for (int j = 0; j < dWh.length; j++) Wh.grad.data[j] = (float) aWh[j];
        for (int j = 0; j < 4 * H; j++) b.grad.data[j] = (float) ab[j];
        return dx;
    }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != T * D) throw new IllegalArgumentException("LSTM expects " + T * D + " numbers per sequence but gets " + java.util.Arrays.toString(in));
        return new int[]{seq ? T * H : H};
    }

    @Override public List<Param> params() { return List.of(Wx, Wh, b); }
    @Override public String spec() { return "LSTM " + T + " " + D + " " + H + " " + (seq ? 1 : 0); }
}
