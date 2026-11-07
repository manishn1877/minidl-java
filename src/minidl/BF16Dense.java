package minidl;

/**
 * Dense layer with weights stored in bfloat16: the first 16 bits of a float32 (same range, only 8 bits
 * of precision instead of 24). Half the memory of float32, and no scales to manage.
 * bfloat16 is the 16-bit format used for training and running large AI models on modern chips.
 * Prediction only here.
 */
public class BF16Dense implements Layer {
    private final int in, out;
    private final short[] w;   // in x out
    private final float[] bias;

    public BF16Dense(int in, int out) { this.in = in; this.out = out; this.w = new short[in * out]; this.bias = new float[out]; }

    /** Round to nearest even, keep the top 16 bits. */
    public static short toBF16(float f) { return Half.toBF16(f); }

    public static float fromBF16(short s) { return Half.fromBF16(s); }

    public static BF16Dense from(Dense d) {
        Tensor W = d.params().get(0).value, b = d.params().get(1).value;
        BF16Dense q = new BF16Dense(W.rows, W.cols);
        for (int i = 0; i < W.data.length; i++) q.w[i] = toBF16(W.data[i]);
        System.arraycopy(b.data, 0, q.bias, 0, W.cols);
        return q;
    }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != in) throw new IllegalArgumentException("BF16Dense expects " + in + " columns, got " + x.cols);
        float[] wf = Tensor.scratchB(in * out);       // one dequantized copy of this layer, reused
        for (int i = 0; i < in * out; i++) wf[i] = Float.intBitsToFloat(((int) w[i]) << 16);
        Tensor y = new Tensor(x.rows, out);
        Tensor.matmulRaw(x.data, 0, x.rows, in, wf, 0, out, y.data, 0);
        for (int i = 0; i < x.rows; i++) for (int j = 0; j < out; j++) y.data[i * out + j] += bias[j];
        return y;
    }

    @Override public Tensor backward(Tensor g) { throw new UnsupportedOperationException("BF16Dense is for prediction only."); }
    @Override public long weightBytes() { return 2L * w.length + 4L * bias.length; }
    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != this.in) throw new IllegalArgumentException("BF16Dense expects " + this.in + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return new int[]{out};
    }

    @Override public String spec() { return "BF16Dense " + in + " " + out; }

    @Override public void writeExtra(java.io.DataOutput o) throws java.io.IOException {
        for (short s : w) o.writeShort(s);
        for (float b : bias) o.writeFloat(b);
    }

    @Override public void readExtra(java.io.DataInput i) throws java.io.IOException {
        for (int j = 0; j < w.length; j++) w[j] = i.readShort();
        for (int j = 0; j < out; j++) bias[j] = i.readFloat();
    }
}
