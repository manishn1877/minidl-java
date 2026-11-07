package minidl;

/** Average pooling: replace each k x k square by its average. Smoother than max pooling, and its backward needs no extra memory. */
public class AvgPool2D implements Layer {
    private final int C, H, W, k, outH, outW, inSize, outSize;

    public AvgPool2D(int channels, int height, int width, int k) {
        if (k < 1 || height < k || width < k) throw new IllegalArgumentException("bad pool window");
        this.C = channels; this.H = height; this.W = width; this.k = k;
        this.outH = height / k; this.outW = width / k;
        this.inSize = C * H * W; this.outSize = C * outH * outW;
    }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != inSize) throw new IllegalArgumentException("AvgPool2D expects " + inSize + " numbers per image, got " + x.cols);
        Tensor out = new Tensor(x.rows, outSize);
        float inv = 1f / (k * k);
        for (int i = 0; i < x.rows; i++)
            for (int c = 0; c < C; c++)
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++) {
                        float s = 0;
                        for (int dy = 0; dy < k; dy++) {
                            int row = i * inSize + (c * H + oy * k + dy) * W + ox * k;
                            for (int dx = 0; dx < k; dx++) s += x.data[row + dx];
                        }
                        out.data[i * outSize + (c * outH + oy) * outW + ox] = s * inv;
                    }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        Tensor dx = new Tensor(g.rows, inSize);
        float inv = 1f / (k * k);
        for (int i = 0; i < g.rows; i++)
            for (int c = 0; c < C; c++)
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++) {
                        float v = g.data[i * outSize + (c * outH + oy) * outW + ox] * inv;
                        for (int dy = 0; dy < k; dy++) {
                            int row = i * inSize + (c * H + oy * k + dy) * W + ox * k;
                            for (int dx2 = 0; dx2 < k; dx2++) dx.data[row + dx2] += v;
                        }
                    }
        return dx;
    }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != inSize) throw new IllegalArgumentException("AvgPool2D expects %d numbers per sample but gets %s = %d".formatted(inSize, java.util.Arrays.toString(in), NDTensor.product(in)));
        return new int[]{C, outH, outW};
    }

    @Override public String spec() { return "AvgPool2D " + C + " " + H + " " + W + " " + k; }
}
