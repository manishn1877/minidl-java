package minidl;

/**
 * Dense layer whose weights are stored as 8-bit integers (int8) instead of 32-bit floats.
 * INFERENCE ONLY (no training): train in float, then shrink the model with {@link Quantizer}.
 *
 * How: every output column j gets one scale s_j = (biggest |weight| in that column) / 127.
 * Each weight is stored as round(w / s_j), a whole number from -127 to 127 (1 byte).
 * While predicting: out = (x . q) * s_j + bias. (Weights are int8, inputs and results stay float.)
 *
 * This is "weight-only int8 quantization", the same idea AI companies use to fit big models into less memory.
 */
public class QuantizedDense implements Layer {
    private final int in, out;
    private final byte[] w;      // in x out, row-major
    private final float[] scale; // one per output column
    private final float[] bias;

    public QuantizedDense(int in, int out) {
        this.in = in; this.out = out;
        this.w = new byte[in * out];
        this.scale = new float[out];
        this.bias = new float[out];
    }

    public static QuantizedDense from(Dense d) {
        Tensor W = d.params().get(0).value, b = d.params().get(1).value;
        QuantizedDense q = new QuantizedDense(W.rows, W.cols);
        for (int j = 0; j < W.cols; j++) {
            float max = 0;
            for (int k = 0; k < W.rows; k++) max = Math.max(max, Math.abs(W.data[k * W.cols + j]));
            q.scale[j] = max == 0 ? 1f : max / 127f;
            q.bias[j] = b.data[j];
        }
        for (int k = 0; k < W.rows; k++)
            for (int j = 0; j < W.cols; j++) {
                int v = Math.round(W.data[k * W.cols + j] / q.scale[j]);
                q.w[k * W.cols + j] = (byte) Math.max(-127, Math.min(127, v));
            }
        return q;
    }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != in) throw new IllegalArgumentException("QuantizedDense expects " + in + " columns, got " + x.cols);
        // Turn the int8 weights back into floats ONCE per call (a temporary copy of this one layer, reused by the next call),
        // then use the same fast multiply as the float layers. The stored weights stay 1 byte each.
        float[] wf = Tensor.scratchB(in * out);
        for (int k = 0; k < in; k++) {
            int base = k * out;
            for (int j = 0; j < out; j++) wf[base + j] = w[base + j] * scale[j];
        }
        Tensor y = new Tensor(x.rows, out);
        Tensor.matmulRaw(x.data, 0, x.rows, in, wf, 0, out, y.data, 0);
        for (int i = 0; i < x.rows; i++) {
            int base = i * out;
            for (int j = 0; j < out; j++) y.data[base + j] += bias[j];
        }
        return y;
    }

    @Override public Tensor backward(Tensor g) {
        throw new UnsupportedOperationException("QuantizedDense is for prediction only. Train a float model, then quantize it.");
    }

    @Override public long weightBytes() { return w.length + 4L * (scale.length + bias.length); }
    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != this.in) throw new IllegalArgumentException("QuantizedDense expects " + this.in + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return new int[]{out};
    }

    @Override public String spec() { return "QDense " + in + " " + out; }

    @Override public void writeExtra(java.io.DataOutput o) throws java.io.IOException {
        o.write(w);
        for (float s : scale) o.writeFloat(s);
        for (float b : bias) o.writeFloat(b);
    }

    @Override public void readExtra(java.io.DataInput i) throws java.io.IOException {
        i.readFully(w);
        for (int j = 0; j < out; j++) scale[j] = i.readFloat();
        for (int j = 0; j < out; j++) bias[j] = i.readFloat();
    }
}
