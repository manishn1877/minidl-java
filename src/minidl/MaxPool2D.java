package minidl;

import java.util.Arrays;

/**
 * Max pooling: look at each k x k square and keep only the biggest number.
 * Shrinks the image (28x28 -> 14x14 for k=2), keeps the strongest signals, and makes the
 * next layers cheaper.
 *
 * Memory saving: to send blame back we only need to know WHICH of the k*k numbers won.
 * That position fits in ONE BYTE (we do not store the pixel values again).
 */
public class MaxPool2D implements Layer {
    private final int C, H, W, k, outH, outW, inSize, outSize;
    private byte[] winner = new byte[0]; // position of the maximum inside each window (0 .. k*k-1)

    public MaxPool2D(int channels, int height, int width, int k) {
        if (k < 1 || k > 11) throw new IllegalArgumentException("k must be between 1 and 11");
        if (height < k || width < k) throw new IllegalArgumentException("pool window bigger than image");
        this.C = channels; this.H = height; this.W = width; this.k = k;
        this.outH = height / k; this.outW = width / k;
        this.inSize = C * H * W; this.outSize = C * outH * outW;
    }

    public int outputSize() { return outSize; }
    public int outH() { return outH; }
    public int outW() { return outW; }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != inSize) throw new IllegalArgumentException("MaxPool2D expects " + inSize + " numbers per image, got " + x.cols);
        int n = x.rows;
        Tensor out = new Tensor(n, outSize);
        if (winner.length < n * outSize) winner = new byte[n * outSize];
        if (k == 2) { // the common case, without inner loops
            final float[] xd = x.data;
            for (int i = 0; i < n; i++)
                for (int c = 0; c < C; c++)
                    for (int oy = 0; oy < outH; oy++) {
                        int row0 = i * inSize + (c * H + 2 * oy) * W, row1 = row0 + W;
                        int o = i * outSize + (c * outH + oy) * outW;
                        for (int ox = 0; ox < outW; ox++) {
                            int p = row0 + 2 * ox, q = row1 + 2 * ox;
                            float best = xd[p]; int bi = 0;
                            float v = xd[p + 1]; if (v > best) { best = v; bi = 1; }
                            v = xd[q]; if (v > best) { best = v; bi = 2; }
                            v = xd[q + 1]; if (v > best) { best = v; bi = 3; }
                            out.data[o + ox] = best;
                            winner[o + ox] = (byte) bi;
                        }
                    }
            return out;
        }
        for (int i = 0; i < n; i++)
            for (int c = 0; c < C; c++)
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++) {
                        float best = Float.NEGATIVE_INFINITY; int bi = 0;
                        for (int dy = 0; dy < k; dy++) {
                            int row = i * inSize + (c * H + oy * k + dy) * W + ox * k;
                            for (int dx = 0; dx < k; dx++) {
                                float v = x.data[row + dx];
                                if (v > best) { best = v; bi = dy * k + dx; }
                            }
                        }
                        int o = i * outSize + (c * outH + oy) * outW + ox;
                        out.data[o] = best;
                        winner[o] = (byte) bi;
                    }
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        int n = g.rows;
        Tensor dx = new Tensor(n, inSize);
        if (k == 2) {
            for (int i = 0; i < n; i++)
                for (int c = 0; c < C; c++)
                    for (int oy = 0; oy < outH; oy++) {
                        int row0 = i * inSize + (c * H + 2 * oy) * W;
                        int o = i * outSize + (c * outH + oy) * outW;
                        for (int ox = 0; ox < outW; ox++) {
                            int bi = winner[o + ox];
                            dx.data[row0 + 2 * ox + (bi >> 1) * W + (bi & 1)] += g.data[o + ox];
                        }
                    }
            return dx;
        }
        for (int i = 0; i < n; i++)
            for (int c = 0; c < C; c++)
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++) {
                        int o = i * outSize + (c * outH + oy) * outW + ox;
                        int bi = winner[o];
                        int pos = i * inSize + (c * H + oy * k + bi / k) * W + ox * k + bi % k;
                        dx.data[pos] += g.data[o];
                    }
        return dx;
    }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != inSize) throw new IllegalArgumentException("MaxPool2D expects %d numbers per sample but gets %s = %d".formatted(inSize, java.util.Arrays.toString(in), NDTensor.product(in)));
        return new int[]{C, outH, outW};
    }

    @Override public String spec() { return "MaxPool2D " + C + " " + H + " " + W + " " + k; }
}
