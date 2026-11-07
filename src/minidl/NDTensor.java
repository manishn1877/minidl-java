package minidl;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * An N-dimensional array of floats (images [N, C, H, W], token batches [N, T, D], ...), stored row-major in one flat array.
 *
 *   - reshape() and toMatrix() are VIEWS: they share the same numbers, nothing is copied.
 *   - add / sub / mul / div follow the usual broadcasting rule: shapes are lined up from the right, and a
 *     dimension of size 1 is stretched to match ([3,1] + [1,4] gives [3,4]).
 *   - matmul is batched: [..., M, K] . [..., K, N].
 *
 * The training layers still work on 2D Tensors (one flattened sample per row), because that is where the fast kernels
 * live. NDTensor is the friendly front door: Sequential.predict(NDTensor) accepts it, and Tensor.nd(...) turns a Tensor
 * into one (also without copying).
 */
public final class NDTensor {
    public final int[] shape;
    public final float[] data;
    private final int[] strides;

    public NDTensor(int... shape) { this(new float[product(shape)], shape); }

    /** Wraps `data` (NOT copied). */
    public NDTensor(float[] data, int... shape) {
        if (shape.length == 0) throw new IllegalArgumentException("need at least one dimension");
        if (data.length != product(shape)) throw new IllegalArgumentException("shape " + Arrays.toString(shape) + " needs " + product(shape) + " numbers but got " + data.length);
        this.data = data;
        this.shape = shape.clone();
        this.strides = stridesOf(this.shape);
    }

    /** A view of a 2D Tensor with a new shape (no copy). */
    public static NDTensor of(Tensor t, int... shape) { return new NDTensor(t.data, shape); }

    public static int product(int... s) {
        long p = 1;
        for (int d : s) {
            if (d < 0) throw new IllegalArgumentException("negative dimension in " + Arrays.toString(s));
            p *= d;
            if (p > Integer.MAX_VALUE) throw new IllegalArgumentException("shape " + Arrays.toString(s) + " is too big");
        }
        return (int) p;
    }

    private static int[] stridesOf(int[] shape) {
        int[] st = new int[shape.length];
        int acc = 1;
        for (int i = shape.length - 1; i >= 0; i--) { st[i] = acc; acc *= shape[i]; }
        return st;
    }

    public int rank() { return shape.length; }
    public int dim(int i) { return shape[i]; }
    public int size() { return data.length; }

    public float get(int... idx) { return data[offset(idx)]; }
    public void set(float v, int... idx) { data[offset(idx)] = v; }

    private int offset(int[] idx) {
        if (idx.length != shape.length) throw new IllegalArgumentException("need " + shape.length + " indices, got " + idx.length);
        int o = 0;
        for (int i = 0; i < idx.length; i++) {
            if (idx[i] < 0 || idx[i] >= shape[i]) throw new IndexOutOfBoundsException("index " + idx[i] + " is outside 0.." + (shape[i] - 1) + " in dimension " + i);
            o += idx[i] * strides[i];
        }
        return o;
    }

    // ------------------------------------------------------------------ views

    /** Same numbers, new shape. One dimension may be -1 (worked out for you). A view: nothing is copied. */
    public NDTensor reshape(int... newShape) {
        int[] s = newShape.clone();
        int unknown = -1, known = 1;
        for (int i = 0; i < s.length; i++) {
            if (s[i] == -1) { if (unknown >= 0) throw new IllegalArgumentException("only one dimension can be -1"); unknown = i; }
            else known *= s[i];
        }
        if (unknown >= 0) {
            if (known == 0 || data.length % known != 0) throw new IllegalArgumentException("cannot reshape " + data.length + " numbers into " + Arrays.toString(newShape));
            s[unknown] = data.length / known;
        }
        return new NDTensor(data, s);
    }

    /** The first `rowDims` dimensions become rows, the rest become columns. A view (no copy). rowDims = 1 means "one sample per row". */
    public Tensor toMatrix(int rowDims) {
        if (rowDims < 0 || rowDims > rank()) throw new IllegalArgumentException("rowDims must be 0.." + rank());
        int rows = 1, cols = 1;
        for (int i = 0; i < rank(); i++) { if (i < rowDims) rows *= shape[i]; else cols *= shape[i]; }
        return new Tensor(rows, cols, data);
    }

    // ------------------------------------------------------------------ rearranging (these copy)

    /** Reorders the dimensions: result.shape[i] = shape[axes[i]]. */
    public NDTensor permute(int... axes) {
        int r = rank();
        if (axes.length != r) throw new IllegalArgumentException("permute needs " + r + " axes");
        boolean[] seen = new boolean[r];
        for (int a : axes) { if (a < 0 || a >= r || seen[a]) throw new IllegalArgumentException("axes must be a rearrangement of 0.." + (r - 1)); seen[a] = true; }
        int[] outShape = new int[r], step = new int[r];
        for (int i = 0; i < r; i++) { outShape[i] = shape[axes[i]]; step[i] = strides[axes[i]]; }
        NDTensor out = new NDTensor(outShape);
        int[] idx = new int[r];
        int off = 0;
        for (int n = 0; n < out.data.length; n++) {
            out.data[n] = data[off];
            for (int d = r - 1; d >= 0; d--) {
                idx[d]++; off += step[d];
                if (idx[d] < outShape[d]) break;
                off -= step[d] * outShape[d]; idx[d] = 0;
            }
        }
        return out;
    }

    public NDTensor transpose(int a, int b) {
        int[] axes = new int[rank()];
        for (int i = 0; i < axes.length; i++) axes[i] = i;
        axes[a] = b; axes[b] = a;
        return permute(axes);
    }

    /** Positions [from, to) along one axis. */
    public NDTensor slice(int axis, int from, int to) {
        if (axis < 0 || axis >= rank() || from < 0 || to > shape[axis] || from > to) throw new IllegalArgumentException("bad slice " + from + ".." + to + " of axis " + axis + " with size " + (axis < rank() ? shape[axis] : -1));
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= shape[i];
        for (int i = axis + 1; i < rank(); i++) inner *= shape[i];
        int[] s = shape.clone();
        s[axis] = to - from;
        NDTensor out = new NDTensor(s);
        int len = to - from;
        for (int o = 0; o < outer; o++)
            System.arraycopy(data, (o * shape[axis] + from) * inner, out.data, o * len * inner, len * inner);
        return out;
    }

    /** Joins tensors along one axis. All other dimensions must match. */
    public static NDTensor concat(int axis, NDTensor... parts) {
        NDTensor f = parts[0];
        int total = 0;
        for (NDTensor p : parts) {
            if (p.rank() != f.rank()) throw new IllegalArgumentException("concat needs the same rank");
            for (int i = 0; i < f.rank(); i++) if (i != axis && p.shape[i] != f.shape[i]) throw new IllegalArgumentException("concat shapes differ in dimension " + i);
            total += p.shape[axis];
        }
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= f.shape[i];
        for (int i = axis + 1; i < f.rank(); i++) inner *= f.shape[i];
        int[] s = f.shape.clone();
        s[axis] = total;
        NDTensor out = new NDTensor(s);
        int pos = 0;
        for (NDTensor p : parts) {
            int len = p.shape[axis];
            for (int o = 0; o < outer; o++) System.arraycopy(p.data, o * len * inner, out.data, (o * total + pos) * inner, len * inner);
            pos += len;
        }
        return out;
    }

    // ------------------------------------------------------------------ element-wise maths with broadcasting

    private interface Op { float apply(float a, float b); }

    private NDTensor broadcast(NDTensor o, Op op) {
        int r = Math.max(rank(), o.rank());
        int[] out = new int[r], sa = new int[r], sb = new int[r];
        for (int i = 0; i < r; i++) {
            int ia = i - (r - rank()), ib = i - (r - o.rank());
            int da = ia >= 0 ? shape[ia] : 1, db = ib >= 0 ? o.shape[ib] : 1;
            if (da != db && da != 1 && db != 1)
                throw new IllegalArgumentException("cannot broadcast " + Arrays.toString(shape) + " with " + Arrays.toString(o.shape));
            out[i] = da == 1 ? db : da;
            sa[i] = (ia >= 0 && da != 1) ? strides[ia] : 0;   // stretched dimensions do not move
            sb[i] = (ib >= 0 && db != 1) ? o.strides[ib] : 0;
        }
        NDTensor res = new NDTensor(out);
        int[] idx = new int[r];
        int offA = 0, offB = 0;
        for (int n = 0; n < res.data.length; n++) {
            res.data[n] = op.apply(data[offA], o.data[offB]);
            for (int d = r - 1; d >= 0; d--) {
                idx[d]++; offA += sa[d]; offB += sb[d];
                if (idx[d] < out[d]) break;
                offA -= sa[d] * out[d]; offB -= sb[d] * out[d]; idx[d] = 0;
            }
        }
        return res;
    }

    public NDTensor add(NDTensor o) { return broadcast(o, (a, b) -> a + b); }
    public NDTensor sub(NDTensor o) { return broadcast(o, (a, b) -> a - b); }
    public NDTensor mul(NDTensor o) { return broadcast(o, (a, b) -> a * b); }
    public NDTensor div(NDTensor o) { return broadcast(o, (a, b) -> a / b); }
    public NDTensor add(double s) { return map(v -> v + s); }
    public NDTensor mul(double s) { return map(v -> v * s); }

    public NDTensor map(DoubleUnaryOperator f) {
        NDTensor out = new NDTensor(shape);
        for (int i = 0; i < data.length; i++) out.data[i] = (float) f.applyAsDouble(data[i]);
        return out;
    }

    // ------------------------------------------------------------------ reductions

    private int[] outerLenInner(int axis) {
        if (axis < 0 || axis >= rank()) throw new IllegalArgumentException("axis " + axis + " does not exist in a rank-" + rank() + " tensor");
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= shape[i];
        for (int i = axis + 1; i < rank(); i++) inner *= shape[i];
        return new int[]{outer, shape[axis], inner};
    }

    private int[] reducedShape(int axis, boolean keepDims) {
        if (keepDims) { int[] s = shape.clone(); s[axis] = 1; return s; }
        if (rank() == 1) return new int[]{1};
        int[] s = new int[rank() - 1];
        for (int i = 0, k = 0; i < rank(); i++) if (i != axis) s[k++] = shape[i];
        return s;
    }

    public NDTensor sum(int axis, boolean keepDims) {
        int[] oli = outerLenInner(axis);
        NDTensor out = new NDTensor(reducedShape(axis, keepDims));
        for (int o = 0; o < oli[0]; o++)
            for (int i = 0; i < oli[2]; i++) {
                double s = 0;
                for (int l = 0; l < oli[1]; l++) s += data[(o * oli[1] + l) * oli[2] + i];
                out.data[o * oli[2] + i] = (float) s;
            }
        return out;
    }

    public NDTensor mean(int axis, boolean keepDims) { return sum(axis, keepDims).mul(1.0 / shape[axis]); }

    public NDTensor max(int axis, boolean keepDims) {
        int[] oli = outerLenInner(axis);
        NDTensor out = new NDTensor(reducedShape(axis, keepDims));
        for (int o = 0; o < oli[0]; o++)
            for (int i = 0; i < oli[2]; i++) {
                float m = Float.NEGATIVE_INFINITY;
                for (int l = 0; l < oli[1]; l++) m = Math.max(m, data[(o * oli[1] + l) * oli[2] + i]);
                out.data[o * oli[2] + i] = m;
            }
        return out;
    }

    /** Softmax along one axis (the numbers along that axis become percentages that add up to 1). */
    public NDTensor softmax(int axis) {
        int[] oli = outerLenInner(axis);
        NDTensor out = new NDTensor(shape);
        for (int o = 0; o < oli[0]; o++)
            for (int i = 0; i < oli[2]; i++) {
                double m = Double.NEGATIVE_INFINITY, sum = 0;
                for (int l = 0; l < oli[1]; l++) m = Math.max(m, data[(o * oli[1] + l) * oli[2] + i]);
                for (int l = 0; l < oli[1]; l++) { double e = Math.exp(data[(o * oli[1] + l) * oli[2] + i] - m); out.data[(o * oli[1] + l) * oli[2] + i] = (float) e; sum += e; }
                for (int l = 0; l < oli[1]; l++) out.data[(o * oli[1] + l) * oli[2] + i] = (float) (out.data[(o * oli[1] + l) * oli[2] + i] / sum);
            }
        return out;
    }

    // ------------------------------------------------------------------ batched matrix multiply

    /**
     * [..., M, K] . [..., K, N] -> [..., M, N]. The leading (batch) dimensions must match, or one side may be a plain
     * 2D matrix which is then used for every batch entry (like a weight matrix). Uses the fast kernels.
     */
    public NDTensor matmul(NDTensor o) {
        if (rank() < 2 || o.rank() < 2) throw new IllegalArgumentException("matmul needs rank 2 or more");
        int M = shape[rank() - 2], K = shape[rank() - 1], K2 = o.shape[o.rank() - 2], N = o.shape[o.rank() - 1];
        if (K != K2) throw new IllegalArgumentException("matmul shape mismatch: " + Arrays.toString(shape) + " . " + Arrays.toString(o.shape));
        int[] batchA = Arrays.copyOf(shape, rank() - 2), batchB = Arrays.copyOf(o.shape, o.rank() - 2);
        int[] batch;
        if (o.rank() == 2) batch = batchA;
        else if (rank() == 2) batch = batchB;
        else if (Arrays.equals(batchA, batchB)) batch = batchA;
        else throw new IllegalArgumentException("batch dimensions differ: " + Arrays.toString(shape) + " vs " + Arrays.toString(o.shape));
        int nb = product(batch);
        int[] outShape = Arrays.copyOf(batch, batch.length + 2);
        outShape[batch.length] = M; outShape[batch.length + 1] = N;
        NDTensor out = new NDTensor(outShape);
        if (o.rank() == 2) {            // one weight matrix for everything: a single big multiply
            Tensor.gemmNN(data, 0, K, o.data, 0, N, out.data, 0, 0, nb * M);
        } else {
            for (int b = 0; b < nb; b++)
                Tensor.gemmNN(data, rank() == 2 ? 0 : b * M * K, K, o.data, b * K * N, N, out.data, b * M * N, 0, M);
        }
        return out;
    }

    @Override public String toString() {
        StringBuilder sb = new StringBuilder("NDTensor" + Arrays.toString(shape));
        int show = Math.min(data.length, 12);
        sb.append(" [");
        for (int i = 0; i < show; i++) sb.append(i > 0 ? ", " : "").append(String.format("%.4g", data[i]));
        if (data.length > show) sb.append(", ...");
        return sb.append("]").toString();
    }
}
