package minidl;

/**
 * Dense layer with 4-bit weights: every weight is a whole number from -7 to 7, and TWO weights share one byte.
 * That is 8x smaller than float32. Prediction only.
 *
 * 4 bits are very coarse, so one scale for a whole column would lose too much. Instead every
 * `group` rows of a column get their own scale ("group-wise" quantization, the way many
 * language-model weights are shrunk).
 */
public class QuantizedDense4 implements Layer {
    private final int in, out, outPad, half, group, groups;
    private final byte[] packed; // in x (outPad/2): column j even -> low nibble, odd -> high nibble (value + 8)
    private final float[] scale; // groups x outPad
    private final float[] bias;  // out

    public QuantizedDense4(int in, int out, int group) {
        this.in = in; this.out = out; this.group = group;
        this.outPad = out + (out & 1);
        this.half = outPad / 2;
        this.groups = (in + group - 1) / group;
        this.packed = new byte[in * half];
        this.scale = new float[groups * outPad];
        this.bias = new float[out];
    }

    public static QuantizedDense4 from(Dense d, int group) {
        Tensor W = d.params().get(0).value, b = d.params().get(1).value;
        QuantizedDense4 q = new QuantizedDense4(W.rows, W.cols, group);
        for (int g = 0; g < q.groups; g++)
            for (int j = 0; j < W.cols; j++) {
                float max = 0;
                for (int k = g * group; k < Math.min(W.rows, (g + 1) * group); k++) max = Math.max(max, Math.abs(W.data[k * W.cols + j]));
                q.scale[g * q.outPad + j] = max == 0 ? 1f : max / 7f;
            }
        for (int k = 0; k < W.rows; k++)
            for (int j = 0; j < W.cols; j++) {
                int v = Math.round(W.data[k * W.cols + j] / q.scale[(k / group) * q.outPad + j]);
                int nib = Math.max(-7, Math.min(7, v)) + 8;
                int idx = k * q.half + j / 2;
                q.packed[idx] = (byte) ((j & 1) == 0 ? (q.packed[idx] & 0xF0) | nib : (q.packed[idx] & 0x0F) | (nib << 4));
            }
        System.arraycopy(b.data, 0, q.bias, 0, W.cols);
        return q;
    }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != in) throw new IllegalArgumentException("QuantizedDense4 expects " + in + " columns, got " + x.cols);
        Tensor y = new Tensor(x.rows, out);
        final float[] xd = x.data, yd = y.data;
        Tensor.forRows(x.rows, (long) x.rows * in * out, i -> {
            float[] tmp = new float[outPad], acc = new float[outPad];
            int xi = i * in;
            for (int g = 0; g < groups; g++) {
                java.util.Arrays.fill(tmp, 0f);
                int kEnd = Math.min(in, (g + 1) * group);
                for (int k = g * group; k < kEnd; k++) {
                    float xv = xd[xi + k];
                    int base = k * half;
                    for (int jj = 0; jj < half; jj++) {
                        int bt = packed[base + jj];
                        tmp[2 * jj] += xv * ((bt & 0xF) - 8);
                        tmp[2 * jj + 1] += xv * (((bt >> 4) & 0xF) - 8);
                    }
                }
                int sb = g * outPad;
                for (int j = 0; j < outPad; j++) acc[j] += tmp[j] * scale[sb + j];
            }
            for (int j = 0; j < out; j++) yd[i * out + j] = acc[j] + bias[j];
        });
        return y;
    }

    @Override public Tensor backward(Tensor g) { throw new UnsupportedOperationException("QuantizedDense4 is for prediction only."); }
    @Override public long weightBytes() { return packed.length + 4L * (scale.length + bias.length); }
    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != this.in) throw new IllegalArgumentException("QuantizedDense4 expects " + this.in + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return new int[]{out};
    }

    @Override public String spec() { return "QDense4 " + in + " " + out + " " + group; }

    @Override public void writeExtra(java.io.DataOutput o) throws java.io.IOException {
        o.write(packed);
        for (float s : scale) o.writeFloat(s);
        for (float b : bias) o.writeFloat(b);
    }

    @Override public void readExtra(java.io.DataInput i) throws java.io.IOException {
        i.readFully(packed);
        for (int j = 0; j < scale.length; j++) scale[j] = i.readFloat();
        for (int j = 0; j < out; j++) bias[j] = i.readFloat();
    }
}
