package minidl;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/**
 * A 2D grid of numbers (rows x cols), stored as 32-bit floats in one flat array.
 *
 * Why float and not double: half the memory for weights, gradients, optimizer state and data,
 * and the CPU can do twice as many float operations per instruction.
 *
 * Speed tricks used here:
 *  - matmul loops in i-k-j order and handles 4 rows per pass (memory-friendly, JIT can vectorize)
 *  - fused "transposed" multiplies (matmulTA / matmulTB): no temporary transposed copy
 *  - "Into" versions write into an existing Tensor: no new memory per batch
 *  - big multiplies are split across CPU cores (each thread owns whole output rows, no locking)
 */
public class Tensor {
    public final int rows, cols;
    public final float[] data;

    private static volatile boolean parallel = Runtime.getRuntime().availableProcessors() > 1;
    private static final long PAR_THRESHOLD = 1L << 17; // below ~131k multiply-adds, threads cost more than they save

    public static void setParallel(boolean on) { parallel = on; }
    public static boolean isParallel() { return parallel; }

    public Tensor(int rows, int cols) {
        this(rows, cols, new float[rows * cols]);
    }

    public Tensor(int rows, int cols, float[] data) {
        if (data.length != rows * cols)
            throw new IllegalArgumentException("data length != rows*cols");
        this.rows = rows;
        this.cols = cols;
        this.data = data;
    }

    public static Tensor of(double[][] v) {
        Tensor t = new Tensor(v.length, v[0].length);
        for (int i = 0; i < t.rows; i++)
            for (int j = 0; j < t.cols; j++)
                t.data[i * t.cols + j] = (float) v[i][j];
        return t;
    }

    /** Random numbers from a bell curve (mean 0, given std). */
    public static Tensor randn(int rows, int cols, double std, Random rnd) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = (float) (rnd.nextGaussian() * std);
        return t;
    }

    public double get(int r, int c) { return data[r * cols + c]; }
    public void set(int r, int c, double v) { data[r * cols + c] = (float) v; }

    // ------------------------------------------------------------------ choosing the kernel

    /** The Vector API kernel if Java was started with --add-modules jdk.incubator.vector and the class was compiled in; else null. */
    private static final GemmKernel VEC = loadVector();
    private static volatile boolean useVec = VEC != null;

    private static GemmKernel loadVector() {
        if ("false".equalsIgnoreCase(System.getProperty("minidl.vector"))) return null;
        try { return (GemmKernel) Class.forName("minidl.VectorGemm").getDeclaredConstructor().newInstance(); }
        catch (Throwable t) { return null; } // module not enabled, or class not compiled: fall back to the portable kernels
    }

    public static boolean vectorAvailable() { return VEC != null; }
    public static void setVector(boolean on) { useVec = on && VEC != null; }
    public static boolean isVector() { return useVec; }
    public static String kernelName() { return useVec ? VEC.name() : "portable Java loops (JIT auto-vectorized)"; }

    private static final ThreadLocal<float[]> SCRATCH = ThreadLocal.withInitial(() -> new float[0]);

    /** A reusable work array for this thread (contents undefined). */
    static float[] scratch(int n) {
        float[] s = SCRATCH.get();
        if (s.length < n) { s = new float[n]; SCRATCH.set(s); }
        return s;
    }

    private static final ThreadLocal<float[]> SCRATCH2 = ThreadLocal.withInitial(() -> new float[0]);

    /** A second reusable work array for this thread (used to hold one dequantized layer while it is being applied). */
    static float[] scratchB(int n) {
        float[] s = SCRATCH2.get();
        if (s.length < n) { s = new float[n]; SCRATCH2.set(s); }
        return s;
    }

    /** c[rows x N] = a[rows x K] . b[K x N] on raw arrays (offsets ao, bo, co), using the fast kernel and all cores when the job is big. */
    static void matmulRaw(float[] a, int ao, int rows, int K, float[] b, int bo, int N, float[] c, int co) {
        Arrays.fill(c, co, co + rows * N, 0f);
        forRowBlocks(rows, (long) rows * K * N, (f, t) -> gemmNN(a, ao, K, b, bo, N, c, co, f, t));
    }

    // ------------------------------------------------------------------ raw kernels (also used by Conv2D)
    // gemmNN / gemmNT / gemmTN pick the fast kernel when it is on and the sizes are big enough to pay off.

    /** C[from..to) += A . B     A is (.. x K), B is (K x N), C is (.. x N). All arrays are row-major. */
    static void gemmNN(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int from, int to) {
        if (useVec && N >= 16 && (long) (to - from) * K * N >= 2048) VEC.nn(a, ao, K, b, bo, N, c, co, from, to);
        else scalarNN(a, ao, K, b, bo, N, c, co, from, to);
    }

    /** C[from..to) += A . B^T   A is (.. x K), B is (N x K), C is (.. x N). Adds to C. */
    static void gemmNT(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int from, int to) {
        if (useVec && K >= 16 && (long) (to - from) * K * N >= 2048) VEC.nt(a, ao, K, b, bo, N, c, co, from, to);
        else scalarNT(a, ao, K, b, bo, N, c, co, from, to);
    }

    /** C[from..to) += A^T . B    A is (R x Cc), B is (R x N), C is (Cc x N). from/to index rows of C. */
    static void gemmTN(float[] a, int ao, int R, int Cc, float[] b, int bo, int N, float[] c, int co, int from, int to) {
        if (useVec && N >= 16 && (long) (to - from) * R * N >= 2048) {
            int rows = to - from;
            float[] tr = scratch(rows * R);           // A^T for just these rows, so the fast kernel can read it in a straight line
            for (int i = 0; i < R; i++) {
                int src = ao + i * Cc + from;
                for (int k = 0; k < rows; k++) tr[k * R + i] = a[src + k];
            }
            VEC.nn(tr, 0, R, b, bo, N, c, co + from * N, 0, rows);
        } else scalarTN(a, ao, R, Cc, b, bo, N, c, co, from, to);
    }

    // ---- portable kernels (plain Java loops; the JIT turns the inner loops into SIMD code by itself)

    static void scalarNN(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int from, int to) {
        for (int i = from; i < to; i++) {
            int ai = ao + i * K, ci = co + i * N;
            int k = 0;
            // 4 rows of B per pass: the output row is read/written once per 4 multiplies
            for (; k + 3 < K; k += 4) {
                float a0 = a[ai + k], a1 = a[ai + k + 1], a2 = a[ai + k + 2], a3 = a[ai + k + 3];
                int b0 = bo + k * N, b1 = b0 + N, b2 = b1 + N, b3 = b2 + N;
                for (int j = 0; j < N; j++)
                    c[ci + j] += a0 * b[b0 + j] + a1 * b[b1 + j] + a2 * b[b2 + j] + a3 * b[b3 + j];
            }
            for (; k < K; k++) {
                float av = a[ai + k];
                int bk = bo + k * N;
                for (int j = 0; j < N; j++) c[ci + j] += av * b[bk + j];
            }
        }
    }

    static void scalarNT(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int from, int to) {
        for (int i = from; i < to; i++) {
            int ai = ao + i * K, ci = co + i * N;
            int j = 0;
            // 4 dot products at once: 4 independent sums keep the CPU busy and share each load of a[]
            for (; j + 3 < N; j += 4) {
                int b0 = bo + j * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (int k = 0; k < K; k++) {
                    float av = a[ai + k];
                    s0 += av * b[b0 + k]; s1 += av * b[b1 + k];
                    s2 += av * b[b2 + k]; s3 += av * b[b3 + k];
                }
                c[ci + j] += s0; c[ci + j + 1] += s1; c[ci + j + 2] += s2; c[ci + j + 3] += s3;
            }
            for (; j < N; j++) {
                int bj = bo + j * K;
                float sum = 0;
                for (int k = 0; k < K; k++) sum += a[ai + k] * b[bj + k];
                c[ci + j] += sum;
            }
        }
    }

    static void scalarTN(float[] a, int ao, int R, int Cc, float[] b, int bo, int N, float[] c, int co, int from, int to) {
        for (int k = from; k < to; k++) {
            int ck = co + k * N;
            int i = 0;
            for (; i + 3 < R; i += 4) {
                float a0 = a[ao + i * Cc + k], a1 = a[ao + (i + 1) * Cc + k], a2 = a[ao + (i + 2) * Cc + k], a3 = a[ao + (i + 3) * Cc + k];
                int b0 = bo + i * N, b1 = b0 + N, b2 = b1 + N, b3 = b2 + N;
                for (int j = 0; j < N; j++)
                    c[ck + j] += a0 * b[b0 + j] + a1 * b[b1 + j] + a2 * b[b2 + j] + a3 * b[b3 + j];
            }
            for (; i < R; i++) {
                float av = a[ao + i * Cc + k];
                int bi = bo + i * N;
                for (int j = 0; j < N; j++) c[ck + j] += av * b[bi + j];
            }
        }
    }

    static void forRows(int n, long work, IntConsumer body) {
        if (parallel && n > 1 && work >= PAR_THRESHOLD) IntStream.range(0, n).parallel().forEach(body);
        else for (int i = 0; i < n; i++) body.accept(i);
    }

    private interface RowRange { void run(int from, int to); }

    /** Splits rows into blocks of 16 (so the 4-row tiles of the fast kernel stay intact) and runs them on all cores when the job is big. */
    private static void forRowBlocks(int n, long work, RowRange body) {
        final int BLOCK = 16;
        if (parallel && n > BLOCK && work >= PAR_THRESHOLD)
            IntStream.range(0, (n + BLOCK - 1) / BLOCK).parallel().forEach(blk -> body.run(blk * BLOCK, Math.min(n, (blk + 1) * BLOCK)));
        else body.run(0, n);
    }

    // ------------------------------------------------------------------ matrix multiply

    /** this . o  ->  (rows x o.cols) */
    public Tensor matmul(Tensor o) {
        Tensor out = new Tensor(rows, o.cols);
        matmulInto(o, out);
        return out;
    }

    public void matmulInto(Tensor o, Tensor out) {
        if (cols != o.rows) throw new IllegalArgumentException(
            "matmul shape mismatch: " + shape() + " . " + o.shape());
        checkOut(out, rows, o.cols);
        Arrays.fill(out.data, 0f);
        final int K = cols, n = o.cols;
        forRowBlocks(rows, (long) rows * K * n, (f, t) -> gemmNN(data, 0, K, o.data, 0, n, out.data, 0, f, t));
    }

    /** this^T . o  ->  (cols x o.cols). Same as transpose().matmul(o) but with no temporary copy. */
    public Tensor matmulTA(Tensor o) {
        Tensor out = new Tensor(cols, o.cols);
        matmulTAInto(o, out);
        return out;
    }

    public void matmulTAInto(Tensor o, Tensor out) {
        if (rows != o.rows) throw new IllegalArgumentException(
            "matmulTA shape mismatch: " + shape() + "^T . " + o.shape());
        checkOut(out, cols, o.cols);
        Arrays.fill(out.data, 0f);
        final int R = rows, C = cols, n = o.cols;
        forRowBlocks(C, (long) R * C * n, (f, t) -> gemmTN(data, 0, R, C, o.data, 0, n, out.data, 0, f, t));
    }

    /** this . o^T  ->  (rows x o.rows). Same as matmul(o.transpose()) but with no temporary copy. */
    public Tensor matmulTB(Tensor o) {
        Tensor out = new Tensor(rows, o.rows);
        matmulTBInto(o, out);
        return out;
    }

    public void matmulTBInto(Tensor o, Tensor out) {
        if (cols != o.cols) throw new IllegalArgumentException(
            "matmulTB shape mismatch: " + shape() + " . " + o.shape() + "^T");
        checkOut(out, rows, o.rows);
        Arrays.fill(out.data, 0f);
        final int C = cols, m = o.rows;
        long work = (long) rows * C * m;
        if (useVec && rows >= 8 && m >= 16 && work >= 2048) {
            // lay B out transposed once (cheap next to the multiply), then use the fast straight-line kernel
            final float[] bt = scratch(C * m);
            for (int j = 0; j < m; j++) { int src = j * C; for (int k = 0; k < C; k++) bt[k * m + j] = o.data[src + k]; }
            forRowBlocks(rows, work, (f, t) -> VEC.nn(data, 0, C, bt, 0, m, out.data, 0, f, t));
        } else {
            forRowBlocks(rows, work, (f, t) -> gemmNT(data, 0, C, o.data, 0, m, out.data, 0, f, t));
        }
    }

    /** Slow textbook version (double accumulation). Only for tests and benchmarks. */
    public Tensor matmulNaive(Tensor o) {
        if (cols != o.rows) throw new IllegalArgumentException("matmul shape mismatch");
        Tensor out = new Tensor(rows, o.cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < o.cols; j++) {
                double s = 0;
                for (int k = 0; k < cols; k++) s += get(i, k) * o.get(k, j);
                out.set(i, j, s);
            }
        return out;
    }

    // ------------------------------------------------------------------ other maths

    public Tensor transpose() {
        Tensor out = new Tensor(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out.data[j * rows + i] = data[i * cols + j];
        return out;
    }

    public Tensor add(Tensor o) { checkSame(o); return zip(o, 1); }
    public Tensor sub(Tensor o) { checkSame(o); return zip(o, -1); }

    public Tensor mul(Tensor o) { // element by element
        checkSame(o);
        Tensor out = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) out.data[i] = data[i] * o.data[i];
        return out;
    }

    /** Adds a (1 x cols) row to every row. Used for bias. */
    public Tensor addRowVector(Tensor bias) {
        Tensor out = copy();
        out.addRowVectorInPlace(bias);
        return out;
    }

    public void addRowVectorInPlace(Tensor bias) {
        if (bias.rows != 1 || bias.cols != cols)
            throw new IllegalArgumentException("bias must be 1 x " + cols);
        for (int i = 0; i < rows; i++) {
            int base = i * cols;
            for (int j = 0; j < cols; j++) data[base + j] += bias.data[j];
        }
    }

    public void addInPlace(Tensor o) {
        checkSame(o);
        for (int i = 0; i < data.length; i++) data[i] += o.data[i];
    }

    public void scaleInPlace(double s) {
        float f = (float) s;
        for (int i = 0; i < data.length; i++) data[i] *= f;
    }

    public Tensor scale(double s) {
        Tensor out = new Tensor(rows, cols);
        float f = (float) s;
        for (int i = 0; i < data.length; i++) out.data[i] = data[i] * f;
        return out;
    }

    public Tensor map(DoubleUnaryOperator f) {
        Tensor out = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) out.data[i] = (float) f.applyAsDouble(data[i]);
        return out;
    }

    /** Adds up all rows -> (1 x cols). */
    public Tensor sumRows() {
        Tensor out = new Tensor(1, cols);
        sumRowsInto(out);
        return out;
    }

    public void sumRowsInto(Tensor out) {
        checkOut(out, 1, cols);
        Arrays.fill(out.data, 0f);
        for (int i = 0; i < rows; i++) {
            int base = i * cols;
            for (int j = 0; j < cols; j++) out.data[j] += data[base + j];
        }
    }

    /** Adds up each row: (rows x cols) -> (rows x 1). */
    public Tensor sumCols() {
        Tensor out = new Tensor(rows, 1);
        for (int i = 0; i < rows; i++) { double s = 0; for (int j = 0; j < cols; j++) s += data[i * cols + j]; out.data[i] = (float) s; }
        return out;
    }

    public Tensor selectRows(int[] idx, int from, int to) {
        Tensor out = new Tensor(to - from, cols);
        for (int r = from; r < to; r++)
            System.arraycopy(data, idx[r] * cols, out.data, (r - from) * cols, cols);
        return out;
    }

    /** Rows [from, to) as a new Tensor (one fast block copy). */
    public Tensor sliceRows(int from, int to) {
        Tensor out = new Tensor(to - from, cols);
        System.arraycopy(data, from * cols, out.data, 0, (to - from) * cols);
        return out;
    }

    public int argmaxRow(int r) {
        int best = 0;
        for (int j = 1; j < cols; j++)
            if (data[r * cols + j] > data[r * cols + best]) best = j;
        return best;
    }

    /** The same numbers as an N-dimensional view (no copy). */
    public NDTensor nd(int... shape) { return NDTensor.of(this, shape); }

    public Tensor copy() { return new Tensor(rows, cols, data.clone()); }
    public String shape() { return "(" + rows + "x" + cols + ")"; }
    /** Memory used by the numbers in this tensor. */
    public long bytes() { return 4L * data.length; }

    private Tensor zip(Tensor o, int sign) {
        Tensor out = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) out.data[i] = data[i] + sign * o.data[i];
        return out;
    }

    private void checkSame(Tensor o) {
        if (rows != o.rows || cols != o.cols)
            throw new IllegalArgumentException("shape mismatch: " + shape() + " vs " + o.shape());
    }

    private static void checkOut(Tensor out, int r, int c) {
        if (out.rows != r || out.cols != c)
            throw new IllegalArgumentException("output tensor must be (" + r + "x" + c + ") but is " + out.shape());
    }

    @Override public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) sb.append(String.format("%8.4f ", get(i, j)));
            sb.append('\n');
        }
        return sb.toString();
    }
}
