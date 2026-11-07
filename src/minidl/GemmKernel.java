package minidl;

/**
 * A fast matrix-multiply engine that can be plugged into Tensor.
 * The portable Java kernels are always there; an implementation of this interface (see
 * src-vector/minidl/VectorGemm.java) is found at run time when Java's Vector API is switched on.
 * All arrays are row-major and every matrix may start at an offset (ao / bo / co).
 */
interface GemmKernel {
    String name();

    /** C[from..to) += A . B     A is (.. x K), B is (K x N), C is (.. x N). */
    void nn(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int from, int to);

    /** C[from..to) += A . B^T   A is (.. x K), B is (N x K), C is (.. x N). */
    void nt(float[] a, int ao, int K, float[] b, int bo, int N, float[] c, int co, int from, int to);
}
