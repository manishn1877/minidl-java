package minidl;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * 2D convolution: slides small filters over an image to find patterns (edges, curves, ...).
 *
 * Data layout: every row of the tensor is ONE image, flattened as [channel][y][x].
 * So a 28x28 grey image is a row of 784 numbers, and 8 filters give a row of 8*26*26 numbers.
 *
 * Options besides size / stride / padding:
 *   dilation  the filter looks at pixels that are `dilation` apart (a wider view with the same number of weights)
 *   groups    the channels are split into independent groups, each with its own filters
 *             (groups = number of channels is a "depthwise" convolution, the cheap kind used in mobile networks)
 *
 * How it works: "im2col" copies each little patch the filter looks at into a column of a matrix.
 * Then the convolution becomes ONE matrix multiply, which we already made fast.
 *
 * Memory saving: the expanded patch matrix is big (it repeats every pixel up to k*k times).
 * We keep only ONE small reusable buffer per CPU thread, and in backward we REBUILD the patches
 * from the input instead of storing them for the whole batch. (Trade a little compute for a lot of memory;
 * big frameworks call this "recomputation" or "activation checkpointing".)
 */
public class Conv2D implements Layer {
    private final int inC, inH, inW, outC, k, stride, pad, dil, groups, outH, outW;
    private final int cin, cout;            // channels per group (in / out)
    private final int K, P, inSize, outSize; // K = patch length (per group), P = number of patch positions per image
    private final Param W, b;
    private Tensor input;      // remembered for backward (32-bit) ...
    private short[] input16;   // ... or in 16 bits when activation storage is FP16 / BF16
    private Precision store = Precision.FP32;
    private boolean inputGrad = true;
    private float[][] colBuf, dcolBuf, gradBuf, inBuf; // one set per chunk of images, reused every call

    public Conv2D(int inC, int inH, int inW, int outC, int k, int stride, int pad, Random rnd) {
        this(inC, inH, inW, outC, k, stride, pad, 1, 1, rnd);
    }

    public Conv2D(int inC, int inH, int inW, int outC, int k, int stride, int pad, int dilation, int groups, Random rnd) {
        if (k < 1 || stride < 1 || pad < 0 || dilation < 1 || groups < 1) throw new IllegalArgumentException("bad conv settings");
        if (inC % groups != 0 || outC % groups != 0) throw new IllegalArgumentException("channels (" + inC + " in, " + outC + " out) must be divisible by groups (" + groups + ")");
        int span = dilation * (k - 1) + 1; // how far the filter reaches
        if (inH + 2 * pad < span || inW + 2 * pad < span) throw new IllegalArgumentException("filter bigger than image");
        this.inC = inC; this.inH = inH; this.inW = inW; this.outC = outC;
        this.k = k; this.stride = stride; this.pad = pad; this.dil = dilation; this.groups = groups;
        this.outH = (inH + 2 * pad - span) / stride + 1;
        this.outW = (inW + 2 * pad - span) / stride + 1;
        this.cin = inC / groups; this.cout = outC / groups;
        this.K = cin * k * k;
        this.P = outH * outW;
        this.inSize = inC * inH * inW;
        this.outSize = outC * P;
        W = new Param(Tensor.randn(outC, K, Math.sqrt(2.0 / K), rnd), true); // He init
        b = new Param(new Tensor(1, outC), false);
    }

    public int outputSize() { return outSize; }
    public int outH() { return outH; }
    public int outW() { return outW; }
    public int outChannels() { return outC; }

    @Override public Tensor forward(Tensor x) {
        if (x.cols != inSize) throw new IllegalArgumentException("Conv2D expects " + inSize + " numbers per image, got " + x.cols);
        if (store == Precision.FP32) { input = x; input16 = null; }
        else { input16 = Half.pack(x.data, 0, x.data.length, store, input16); input = null; }
        int n = x.rows;
        Tensor out = new Tensor(n, outSize);
        int chunks = chunkCount(n);
        ensureBuffers(chunks);
        runChunks(chunks, n, (cid, lo, hi) -> {
            float[] col = colBuf[cid];
            for (int i = lo; i < hi; i++) {
                for (int g = 0; g < groups; g++) {
                    im2col(x.data, i * inSize, col, g * cin, cin, inH, inW, k, stride, pad, dil, outH, outW);
                    // out_g (cout x P) = W_g (cout x K) . col (K x P)
                    Tensor.gemmNN(W.value.data, g * cout * K, K, col, 0, P, out.data, i * outSize + g * cout * P, 0, cout);
                }
                for (int c = 0; c < outC; c++) {
                    float bv = b.value.data[c];
                    int base = i * outSize + c * P;
                    for (int p = 0; p < P; p++) out.data[base + p] += bv;
                }
            }
        });
        return out;
    }

    @Override public Tensor backward(Tensor g) {
        int n = g.rows;
        Tensor dx = inputGrad ? new Tensor(n, inSize) : null;
        int chunks = chunkCount(n);
        ensureBuffers(chunks);
        runChunks(chunks, n, (cid, lo, hi) -> {
            float[] col = colBuf[cid], dcol = dcolBuf[cid], grad = gradBuf[cid]; // grad = [dW (outC*K) | db (outC)]
            Arrays.fill(grad, 0f);
            for (int i = lo; i < hi; i++) {
                float[] src; int srcOff;
                if (input != null) { src = input.data; srcOff = i * inSize; }
                else { src = inBuf[cid]; srcOff = 0; Half.unpack(input16, i * inSize, src, 0, inSize, store); } // one image back to 32-bit
                for (int gr = 0; gr < groups; gr++) {
                    im2col(src, srcOff, col, gr * cin, cin, inH, inW, k, stride, pad, dil, outH, outW); // rebuilt, not stored
                    int go = i * outSize + gr * cout * P;
                    // dW_g (cout x K) += g_g (cout x P) . col^T
                    Tensor.gemmNT(g.data, go, P, col, 0, K, grad, gr * cout * K, 0, cout);
                    if (inputGrad) {
                        Arrays.fill(dcol, 0f);
                        // dcol (K x P) = W_g^T . g_g
                        Tensor.gemmTN(W.value.data, gr * cout * K, cout, K, g.data, go, P, dcol, 0, 0, K);
                        col2im(dcol, dx.data, i * inSize, gr * cin, cin, inH, inW, k, stride, pad, dil, outH, outW);
                    }
                }
                for (int c = 0; c < outC; c++) {
                    float s = 0;
                    int base = i * outSize + c * P;
                    for (int p = 0; p < P; p++) s += g.data[base + p];
                    grad[outC * K + c] += s;
                }
            }
        });
        Arrays.fill(W.grad.data, 0f);
        Arrays.fill(b.grad.data, 0f);
        for (int c = 0; c < chunks; c++) {
            float[] grad = gradBuf[c];
            for (int j = 0; j < outC * K; j++) W.grad.data[j] += grad[j];
            for (int j = 0; j < outC; j++) b.grad.data[j] += grad[outC * K + j];
        }
        return dx;
    }

    // ------------------------------------------------------------------ patches <-> image
    // col has K rows (one per filter position inside the patch) and P columns (one per patch position): col[kk * P + p]

    /**
     * Copies every patch of channels [c0, c0+cg) of one image into `col`. Used by Conv2D and QuantizedConv2D.
     * Row kk = (channel * k + dy) * k + dx holds that pixel offset for every patch position.
     */
    static void im2col(float[] in, int off, float[] col, int c0, int cg, int inH, int inW, int k, int stride, int pad, int dil, int outH, int outW) {
        int P = outH * outW;
        for (int ci = 0; ci < cg; ci++)
            for (int ky = 0; ky < k; ky++)
                for (int kx = 0; kx < k; kx++) {
                    int rowBase = ((ci * k + ky) * k + kx) * P;
                    for (int oy = 0; oy < outH; oy++) {
                        int dst = rowBase + oy * outW;
                        int iy = oy * stride - pad + ky * dil;
                        if (iy < 0 || iy >= inH) { for (int ox = 0; ox < outW; ox++) col[dst + ox] = 0f; continue; }
                        int inRow = off + ((c0 + ci) * inH + iy) * inW;
                        int ix = -pad + kx * dil;
                        for (int ox = 0; ox < outW; ox++, ix += stride) col[dst + ox] = (ix < 0 || ix >= inW) ? 0f : in[inRow + ix];
                    }
                }
    }

    /** The reverse of im2col: adds every patch value back onto the pixel it came from. */
    static void col2im(float[] dcol, float[] dx, int off, int c0, int cg, int inH, int inW, int k, int stride, int pad, int dil, int outH, int outW) {
        int P = outH * outW;
        for (int ci = 0; ci < cg; ci++)
            for (int ky = 0; ky < k; ky++)
                for (int kx = 0; kx < k; kx++) {
                    int rowBase = ((ci * k + ky) * k + kx) * P;
                    for (int oy = 0; oy < outH; oy++) {
                        int iy = oy * stride - pad + ky * dil;
                        if (iy < 0 || iy >= inH) continue;
                        int src = rowBase + oy * outW;
                        int inRow = off + ((c0 + ci) * inH + iy) * inW;
                        int ix = -pad + kx * dil;
                        for (int ox = 0; ox < outW; ox++, ix += stride) if (ix >= 0 && ix < inW) dx[inRow + ix] += dcol[src + ox];
                    }
                }
    }

    int[] geometry() { return new int[]{inC, inH, inW, outC, k, stride, pad, dil, groups}; }
    boolean isPlain() { return dil == 1 && groups == 1; }

    // ------------------------------------------------------------------ threads + buffers

    private interface ChunkBody { void run(int chunk, int from, int to); }

    private static int chunkCount(int n) {
        return Tensor.isParallel() ? Math.max(1, Math.min(n, Runtime.getRuntime().availableProcessors())) : 1;
    }

    private static void runChunks(int chunks, int n, ChunkBody body) {
        if (chunks == 1) { body.run(0, 0, n); return; }
        IntStream.range(0, chunks).parallel().forEach(c -> body.run(c, (int) ((long) c * n / chunks), (int) ((long) (c + 1) * n / chunks)));
    }

    private void ensureBuffers(int chunks) {
        if (colBuf != null && colBuf.length >= chunks) return;
        colBuf = new float[chunks][P * K];
        dcolBuf = new float[chunks][P * K];
        gradBuf = new float[chunks][outC * K + outC];
        inBuf = new float[chunks][inSize];
    }

    @Override public void setComputeInputGrad(boolean on) { inputGrad = on; }
    @Override public void setActivationStorage(Precision p) { store = p; }
    @Override public List<Param> params() { return List.of(W, b); }

    @Override public int[] outputShape(int[] in) {
        if (NDTensor.product(in) != inSize) throw new IllegalArgumentException("Conv2D expects " + inC + "x" + inH + "x" + inW + " = " + inSize + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return new int[]{outC, outH, outW};
    }

    @Override public String spec() {
        String base = "Conv2D " + inC + " " + inH + " " + inW + " " + outC + " " + k + " " + stride + " " + pad;
        return isPlain() ? base : base + " " + dil + " " + groups;
    }
}
