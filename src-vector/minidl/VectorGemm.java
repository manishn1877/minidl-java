package minidl;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Matrix multiply with Java's Vector API (SIMD): the CPU multiplies and adds 8 or 16 floats per instruction.
 *
 * The trick is a "register tile": a small block of the answer (4 rows x 2 vectors of numbers) is kept INSIDE
 * the CPU registers while we walk down the shared dimension, so the answer is read and written once per
 * block instead of once per multiply. Memory traffic, not arithmetic, is what limits ordinary loops.
 * The shared dimension is cut into chunks of 256 so the piece of B being used stays in the fast cache.
 *
 * Compile and run with:  --add-modules jdk.incubator.vector   (see run.sh)
 */
public class VectorGemm implements GemmKernel {
    private static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
    private static final int W = SP.length();
    private static final int KC = 256;

    @Override public String name() { return "Vector API, " + W + " float lanes"; }

    @Override public void nn(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int from, int to) {
        final int N2 = 2 * W;
        int nFull = (N / N2) * N2;
        int iEnd4 = from + ((to - from) / 4) * 4;
        for (int j = 0; j < nFull; j += N2)
            for (int k0 = 0; k0 < K; k0 += KC) {
                int k1 = Math.min(K, k0 + KC);
                for (int i = from; i < iEnd4; i += 4) tile4x2(a, ao, K, b, bo, N, c, co, i, j, k0, k1);
                for (int i = iEnd4; i < to; i++) tile1x2(a, ao, K, b, bo, N, c, co, i, j, k0, k1);
            }
        int jv = nFull;
        for (; jv + W <= N; jv += W) {          // one leftover vector-wide strip
            int i = from;
            for (; i + 3 < to; i += 4) tile4x1(a, ao, K, b, bo, N, c, co, i, jv);
            for (; i < to; i++) tile1x1(a, ao, K, b, bo, N, c, co, i, jv);
        }
        if (jv < N)                              // last few columns, plain loops
            for (int i = from; i < to; i++) {
                int ai = ao + i * K, ci = co + i * N;
                for (int k = 0; k < K; k++) { float av = a[ai + k]; int bk = bo + k * N; for (int j = jv; j < N; j++) c[ci + j] += av * b[bk + j]; }
            }
    }

    private static void tile4x2(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int i, int j, int k0, int k1) {
        int r0 = co + i * N + j, r1 = r0 + N, r2 = r1 + N, r3 = r2 + N;
        FloatVector c00 = FloatVector.fromArray(SP, c, r0), c01 = FloatVector.fromArray(SP, c, r0 + W);
        FloatVector c10 = FloatVector.fromArray(SP, c, r1), c11 = FloatVector.fromArray(SP, c, r1 + W);
        FloatVector c20 = FloatVector.fromArray(SP, c, r2), c21 = FloatVector.fromArray(SP, c, r2 + W);
        FloatVector c30 = FloatVector.fromArray(SP, c, r3), c31 = FloatVector.fromArray(SP, c, r3 + W);
        int a0 = ao + i * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K;
        for (int k = k0; k < k1; k++) {
            int bk = bo + k * N + j;
            FloatVector b0 = FloatVector.fromArray(SP, b, bk), b1 = FloatVector.fromArray(SP, b, bk + W);
            FloatVector x0 = FloatVector.broadcast(SP, a[a0 + k]);
            c00 = b0.fma(x0, c00); c01 = b1.fma(x0, c01);
            FloatVector x1 = FloatVector.broadcast(SP, a[a1 + k]);
            c10 = b0.fma(x1, c10); c11 = b1.fma(x1, c11);
            FloatVector x2 = FloatVector.broadcast(SP, a[a2 + k]);
            c20 = b0.fma(x2, c20); c21 = b1.fma(x2, c21);
            FloatVector x3 = FloatVector.broadcast(SP, a[a3 + k]);
            c30 = b0.fma(x3, c30); c31 = b1.fma(x3, c31);
        }
        c00.intoArray(c, r0); c01.intoArray(c, r0 + W); c10.intoArray(c, r1); c11.intoArray(c, r1 + W);
        c20.intoArray(c, r2); c21.intoArray(c, r2 + W); c30.intoArray(c, r3); c31.intoArray(c, r3 + W);
    }

    private static void tile1x2(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int i, int j, int k0, int k1) {
        int r0 = co + i * N + j, a0 = ao + i * K;
        FloatVector c0 = FloatVector.fromArray(SP, c, r0), c1 = FloatVector.fromArray(SP, c, r0 + W);
        for (int k = k0; k < k1; k++) {
            int bk = bo + k * N + j;
            FloatVector x = FloatVector.broadcast(SP, a[a0 + k]);
            c0 = FloatVector.fromArray(SP, b, bk).fma(x, c0);
            c1 = FloatVector.fromArray(SP, b, bk + W).fma(x, c1);
        }
        c0.intoArray(c, r0); c1.intoArray(c, r0 + W);
    }

    private static void tile4x1(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int i, int j) {
        int r0 = co + i * N + j, r1 = r0 + N, r2 = r1 + N, r3 = r2 + N;
        FloatVector c0 = FloatVector.fromArray(SP, c, r0), c1 = FloatVector.fromArray(SP, c, r1);
        FloatVector c2 = FloatVector.fromArray(SP, c, r2), c3 = FloatVector.fromArray(SP, c, r3);
        int a0 = ao + i * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K;
        for (int k = 0; k < K; k++) {
            FloatVector bv = FloatVector.fromArray(SP, b, bo + k * N + j);
            c0 = bv.fma(FloatVector.broadcast(SP, a[a0 + k]), c0);
            c1 = bv.fma(FloatVector.broadcast(SP, a[a1 + k]), c1);
            c2 = bv.fma(FloatVector.broadcast(SP, a[a2 + k]), c2);
            c3 = bv.fma(FloatVector.broadcast(SP, a[a3 + k]), c3);
        }
        c0.intoArray(c, r0); c1.intoArray(c, r1); c2.intoArray(c, r2); c3.intoArray(c, r3);
    }

    private static void tile1x1(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int i, int j) {
        int r0 = co + i * N + j, a0 = ao + i * K;
        FloatVector acc = FloatVector.fromArray(SP, c, r0);
        for (int k = 0; k < K; k++) acc = FloatVector.fromArray(SP, b, bo + k * N + j).fma(FloatVector.broadcast(SP, a[a0 + k]), acc);
        acc.intoArray(c, r0);
    }

    /** Dot products along K: 4 rows of B at a time share every load of A, then the lanes are added up. */
    @Override public void nt(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int from, int to) {
        int kv = (K / W) * W;
        for (int i = from; i < to; i++) {
            int ai = ao + i * K, ci = co + i * N;
            int j = 0;
            for (; j + 3 < N; j += 4) {
                int b0 = bo + j * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
                FloatVector s0 = FloatVector.zero(SP), s1 = s0, s2 = s0, s3 = s0;
                for (int k = 0; k < kv; k += W) {
                    FloatVector av = FloatVector.fromArray(SP, a, ai + k);
                    s0 = FloatVector.fromArray(SP, b, b0 + k).fma(av, s0);
                    s1 = FloatVector.fromArray(SP, b, b1 + k).fma(av, s1);
                    s2 = FloatVector.fromArray(SP, b, b2 + k).fma(av, s2);
                    s3 = FloatVector.fromArray(SP, b, b3 + k).fma(av, s3);
                }
                float t0 = s0.reduceLanes(VectorOperators.ADD), t1 = s1.reduceLanes(VectorOperators.ADD);
                float t2 = s2.reduceLanes(VectorOperators.ADD), t3 = s3.reduceLanes(VectorOperators.ADD);
                for (int k = kv; k < K; k++) { float av = a[ai + k]; t0 += av * b[b0 + k]; t1 += av * b[b1 + k]; t2 += av * b[b2 + k]; t3 += av * b[b3 + k]; }
                c[ci + j] += t0; c[ci + j + 1] += t1; c[ci + j + 2] += t2; c[ci + j + 3] += t3;
            }
            for (; j < N; j++) {
                int bj = bo + j * K;
                FloatVector s = FloatVector.zero(SP);
                for (int k = 0; k < kv; k += W) s = FloatVector.fromArray(SP, b, bj + k).fma(FloatVector.fromArray(SP, a, ai + k), s);
                float t = s.reduceLanes(VectorOperators.ADD);
                for (int k = kv; k < K; k++) t += a[ai + k] * b[bj + k];
                c[ci + j] += t;
            }
        }
    }
}
