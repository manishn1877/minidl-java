package minidl;

/**
 * Convolution with int8 filter weights (one scale per filter). Prediction only.
 * Same idea as QuantizedDense: store each weight as a whole number -127..127 and multiply back by the scale.
 */
public class QuantizedConv2D implements Layer {
    private final int inC, inH, inW, outC, k, stride, pad, outH, outW, K, P, inSize, outSize;
    private final byte[] w;      // outC x K
    private final float[] scale; // one per filter
    private final float[] bias;

    public QuantizedConv2D(int inC, int inH, int inW, int outC, int k, int stride, int pad) {
        this.inC = inC; this.inH = inH; this.inW = inW; this.outC = outC; this.k = k; this.stride = stride; this.pad = pad;
        this.outH = (inH + 2 * pad - k) / stride + 1;
        this.outW = (inW + 2 * pad - k) / stride + 1;
        this.K = inC * k * k; this.P = outH * outW;
        this.inSize = inC * inH * inW; this.outSize = outC * P;
        this.w = new byte[outC * K];
        this.scale = new float[outC];
        this.bias = new float[outC];
    }

    public static QuantizedConv2D from(Conv2D c) {
        if (!c.isPlain()) throw new IllegalArgumentException("only plain convolutions (dilation 1, groups 1) can be quantized");
        int[] g = c.geometry();
        QuantizedConv2D q = new QuantizedConv2D(g[0], g[1], g[2], g[3], g[4], g[5], g[6]);
        Tensor W = c.params().get(0).value, b = c.params().get(1).value;
        for (int f = 0; f < q.outC; f++) {
            float max = 0;
            for (int j = 0; j < q.K; j++) max = Math.max(max, Math.abs(W.data[f * q.K + j]));
            q.scale[f] = max == 0 ? 1f : max / 127f;
            q.bias[f] = b.data[f];
            for (int j = 0; j < q.K; j++) {
                int v = Math.round(W.data[f * q.K + j] / q.scale[f]);
                q.w[f * q.K + j] = (byte) Math.max(-127, Math.min(127, v));
            }
        }
        return q;
    }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != inSize) throw new IllegalArgumentException("QuantizedConv2D expects " + inSize + " numbers per image, got " + x.cols);
        Tensor out = new Tensor(x.rows, outSize);
        final float[] wf = new float[outC * K];        // the filters as floats (the int8 scale folded in), for this call only
        for (int c = 0; c < outC; c++) for (int kk = 0; kk < K; kk++) wf[c * K + kk] = w[c * K + kk] * scale[c];
        Tensor.forRows(x.rows, (long) x.rows * outC * P * K, i -> {
            float[] col = new float[K * P];
            Conv2D.im2col(x.data, i * inSize, col, 0, inC, inH, inW, k, stride, pad, 1, outH, outW);
            Tensor.gemmNN(wf, 0, K, col, 0, P, out.data, i * outSize, 0, outC);   // the same fast multiply as the float convolution
            for (int c = 0; c < outC; c++) {
                int ob = i * outSize + c * P;
                float bc = bias[c];
                for (int p = 0; p < P; p++) out.data[ob + p] += bc;
            }
        });
        return out;
    }

    @Override public Tensor backward(Tensor g) { throw new UnsupportedOperationException("QuantizedConv2D is for prediction only."); }
    @Override public long weightBytes() { return w.length + 4L * (scale.length + bias.length); }
    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != inSize) throw new IllegalArgumentException("QuantizedConv2D expects " + inSize + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return new int[]{outC, outH, outW};
    }

    @Override public String spec() { return "QConv2D " + inC + " " + inH + " " + inW + " " + outC + " " + k + " " + stride + " " + pad; }

    @Override public void writeExtra(java.io.DataOutput o) throws java.io.IOException {
        o.write(w);
        for (float s : scale) o.writeFloat(s);
        for (float b : bias) o.writeFloat(b);
    }

    @Override public void readExtra(java.io.DataInput i) throws java.io.IOException {
        i.readFully(w);
        for (int j = 0; j < outC; j++) scale[j] = i.readFloat();
        for (int j = 0; j < outC; j++) bias[j] = i.readFloat();
    }
}
