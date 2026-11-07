package minidl;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * Automatic differentiation ("autograd"). Every operation on a Var remembers how to send blame backward,
 * so you only write the FORWARD maths and call loss.backward(). Compare with the Layer classes, where every layer's
 * backward() had to be worked out and written by hand.
 *
 *   Var w = new Var(weights), x = new Var(inputs);
 *   Var loss = x.matmul(w).relu().mseLoss(target);
 *   loss.backward();            // w.grad is now filled in
 *
 * How it works: each result keeps its inputs ("parents") and a small function that adds its share of the
 * gradient to them. backward() visits the whole expression in reverse order (a topological sort).
 */
public final class Var {
    public final Tensor value;
    public Tensor grad; // null until something is added to it
    private final List<Var> parents;
    private Runnable back = () -> {};

    public Var(Tensor value) { this(value, List.of()); }
    private Var(Tensor value, List<Var> parents) { this.value = value; this.parents = parents; }

    public void zeroGrad() { grad = null; }

    private Tensor g() {
        if (grad == null) grad = new Tensor(value.rows, value.cols);
        return grad;
    }

    private static Var make(Tensor v, Var... parents) { return new Var(v, List.of(parents)); }

    // ---------------------------------------------------------------- operations

    public Var matmul(Var o) {
        Var r = make(value.matmul(o.value), this, o);
        r.back = () -> {
            g().addInPlace(r.grad.matmulTB(o.value));   // dA = dC . B^T
            o.g().addInPlace(value.matmulTA(r.grad));   // dB = A^T . dC
        };
        return r;
    }

    public Var add(Var o) {
        Var r = make(value.add(o.value), this, o);
        r.back = () -> { g().addInPlace(r.grad); o.g().addInPlace(r.grad); };
        return r;
    }

    public Var sub(Var o) {
        Var r = make(value.sub(o.value), this, o);
        r.back = () -> { g().addInPlace(r.grad); o.g().addInPlace(r.grad.scale(-1)); };
        return r;
    }

    public Var mul(Var o) { // element by element
        Var r = make(value.mul(o.value), this, o);
        r.back = () -> { g().addInPlace(r.grad.mul(o.value)); o.g().addInPlace(r.grad.mul(value)); };
        return r;
    }

    /** Adds a (1 x cols) bias row to every row. */
    public Var addRow(Var bias) {
        Var r = make(value.addRowVector(bias.value), this, bias);
        r.back = () -> { g().addInPlace(r.grad); bias.g().addInPlace(r.grad.sumRows()); };
        return r;
    }

    public Var scale(double s) {
        Var r = make(value.scale(s), this);
        r.back = () -> g().addInPlace(r.grad.scale(s));
        return r;
    }

    public Var relu() {
        Var r = make(value.map(v -> v > 0 ? v : 0), this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < d.data.length; i++) d.data[i] = value.data[i] > 0 ? r.grad.data[i] : 0f;
            g().addInPlace(d);
        };
        return r;
    }

    public Var tanh() {
        Var r = make(value.map(Math::tanh), this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < d.data.length; i++) d.data[i] = r.grad.data[i] * (1 - r.value.data[i] * r.value.data[i]);
            g().addInPlace(d);
        };
        return r;
    }

    public Var sigmoid() {
        Var r = make(value.map(v -> 1.0 / (1.0 + Math.exp(-v))), this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < d.data.length; i++) d.data[i] = r.grad.data[i] * r.value.data[i] * (1 - r.value.data[i]);
            g().addInPlace(d);
        };
        return r;
    }

    /** Average of all numbers, as a 1 x 1 result. */
    public Var mean() {
        double s = 0;
        for (float v : value.data) s += v;
        Tensor t = new Tensor(1, 1);
        t.data[0] = (float) (s / value.data.length);
        Var r = make(t, this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            java.util.Arrays.fill(d.data, r.grad.data[0] / value.data.length);
            g().addInPlace(d);
        };
        return r;
    }

    /** Mean squared error, built from other operations: no hand-written backward at all. */
    public Var mseLoss(Tensor target) {
        Var d = sub(new Var(target));
        return d.mul(d).mean();
    }

    /** Softmax + cross entropy against a one-hot target, as a 1 x 1 result. */
    public Var softmaxCrossEntropy(Tensor oneHot) {
        Tensor probs = Losses.SoftmaxCrossEntropy.softmax(value);
        double s = 0;
        for (int i = 0; i < probs.data.length; i++) if (oneHot.data[i] > 0) s -= oneHot.data[i] * Math.log(probs.data[i] + 1e-12);
        Tensor t = new Tensor(1, 1);
        t.data[0] = (float) (s / value.rows);
        Var r = make(t, this);
        r.back = () -> g().addInPlace(probs.sub(oneHot).scale(r.grad.data[0] / value.rows));
        return r;
    }


    // ---------------------------------------------------------------- more operations (enough to write LayerNorm, GELU and attention)

    public Var neg() { return scale(-1); }

    /** Adds the same constant to every number. */
    public Var shift(double c) {
        Var r = make(value.map(v -> v + c), this);
        r.back = () -> g().addInPlace(r.grad);
        return r;
    }

    public Var exp() {
        Var r = make(value.map(Math::exp), this);
        r.back = () -> g().addInPlace(r.grad.mul(r.value));
        return r;
    }

    public Var log() {
        Var r = make(value.map(Math::log), this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < d.data.length; i++) d.data[i] = r.grad.data[i] / value.data[i];
            g().addInPlace(d);
        };
        return r;
    }

    public Var sqrt() {
        Var r = make(value.map(Math::sqrt), this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < d.data.length; i++) d.data[i] = r.grad.data[i] / (2f * r.value.data[i]);
            g().addInPlace(d);
        };
        return r;
    }

    /** 1 / x for every number. */
    public Var inv() {
        Var r = make(value.map(v -> 1.0 / v), this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < d.data.length; i++) d.data[i] = -r.grad.data[i] * r.value.data[i] * r.value.data[i];
            g().addInPlace(d);
        };
        return r;
    }

    public Var div(Var o) { return mul(o.inv()); }

    /** Adds up each row: (n x c) -> (n x 1). */
    public Var rowSum() {
        Tensor t = new Tensor(value.rows, 1);
        for (int i = 0; i < value.rows; i++) { double s = 0; for (int j = 0; j < value.cols; j++) s += value.data[i * value.cols + j]; t.data[i] = (float) s; }
        Var r = make(t, this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < value.rows; i++) for (int j = 0; j < value.cols; j++) d.data[i * value.cols + j] = r.grad.data[i];
            g().addInPlace(d);
        };
        return r;
    }

    public Var rowMean() { return rowSum().scale(1.0 / value.cols); }

    /** Adds a column (n x 1) to every column. */
    public Var addCol(Var col) {
        Tensor t = new Tensor(value.rows, value.cols);
        for (int i = 0; i < value.rows; i++) for (int j = 0; j < value.cols; j++) t.data[i * value.cols + j] = value.data[i * value.cols + j] + col.value.data[i];
        Var r = make(t, this, col);
        r.back = () -> { g().addInPlace(r.grad); col.g().addInPlace(r.grad.sumCols()); };
        return r;
    }

    public Var subCol(Var col) { return addCol(col.neg()); }

    /** Multiplies every column by a column (n x 1). */
    public Var mulCol(Var col) {
        Tensor t = new Tensor(value.rows, value.cols);
        for (int i = 0; i < value.rows; i++) for (int j = 0; j < value.cols; j++) t.data[i * value.cols + j] = value.data[i * value.cols + j] * col.value.data[i];
        Var r = make(t, this, col);
        r.back = () -> {
            Tensor dx = new Tensor(value.rows, value.cols);
            for (int i = 0; i < value.rows; i++) for (int j = 0; j < value.cols; j++) dx.data[i * value.cols + j] = r.grad.data[i * value.cols + j] * col.value.data[i];
            g().addInPlace(dx);
            col.g().addInPlace(r.grad.mul(value).sumCols());
        };
        return r;
    }

    public Var divCol(Var col) { return mulCol(col.inv()); }

    /** Multiplies every row by a row (1 x c), like a LayerNorm scale. */
    public Var mulRow(Var row) {
        Tensor t = new Tensor(value.rows, value.cols);
        for (int i = 0; i < value.rows; i++) for (int j = 0; j < value.cols; j++) t.data[i * value.cols + j] = value.data[i * value.cols + j] * row.value.data[j];
        Var r = make(t, this, row);
        r.back = () -> {
            Tensor dx = new Tensor(value.rows, value.cols);
            for (int i = 0; i < value.rows; i++) for (int j = 0; j < value.cols; j++) dx.data[i * value.cols + j] = r.grad.data[i * value.cols + j] * row.value.data[j];
            g().addInPlace(dx);
            row.g().addInPlace(r.grad.mul(value).sumRows());
        };
        return r;
    }

    /** Softmax of every row. */
    public Var rowSoftmax() {
        Tensor p = Losses.SoftmaxCrossEntropy.softmax(value);
        Var r = make(p, this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < value.rows; i++) {
                double dot = 0;
                for (int j = 0; j < value.cols; j++) dot += r.grad.data[i * value.cols + j] * p.data[i * value.cols + j];
                for (int j = 0; j < value.cols; j++) d.data[i * value.cols + j] = (float) (p.data[i * value.cols + j] * (r.grad.data[i * value.cols + j] - dot));
            }
            g().addInPlace(d);
        };
        return r;
    }

    public Var transpose() {
        Var r = make(value.transpose(), this);
        r.back = () -> g().addInPlace(r.grad.transpose());
        return r;
    }

    /** Columns [from, to). */
    public Var sliceCols(int from, int to) {
        int w = to - from;
        Tensor t = new Tensor(value.rows, w);
        for (int i = 0; i < value.rows; i++) System.arraycopy(value.data, i * value.cols + from, t.data, i * w, w);
        Var r = make(t, this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            for (int i = 0; i < value.rows; i++) System.arraycopy(r.grad.data, i * w, d.data, i * value.cols + from, w);
            g().addInPlace(d);
        };
        return r;
    }

    /** Rows [from, to). */
    public Var sliceRows(int from, int to) {
        Var r = make(value.sliceRows(from, to), this);
        r.back = () -> {
            Tensor d = new Tensor(value.rows, value.cols);
            System.arraycopy(r.grad.data, 0, d.data, from * value.cols, r.grad.data.length);
            g().addInPlace(d);
        };
        return r;
    }

    /** Puts the pieces side by side (same number of rows). */
    public static Var concatCols(java.util.List<Var> parts) {
        int rows = parts.get(0).value.rows, total = 0;
        for (Var p : parts) total += p.value.cols;
        Tensor t = new Tensor(rows, total);
        int off = 0;
        for (Var p : parts) {
            for (int i = 0; i < rows; i++) System.arraycopy(p.value.data, i * p.value.cols, t.data, i * total + off, p.value.cols);
            off += p.value.cols;
        }
        Var r = new Var(t, new ArrayList<>(parts));
        final int tot = total;
        r.back = () -> {
            int o = 0;
            for (Var p : parts) {
                Tensor d = new Tensor(rows, p.value.cols);
                for (int i = 0; i < rows; i++) System.arraycopy(r.grad.data, i * tot + o, d.data, i * p.value.cols, p.value.cols);
                p.g().addInPlace(d);
                o += p.value.cols;
            }
        };
        return r;
    }

    /** Stacks the pieces on top of each other (same number of columns). */
    public static Var concatRows(java.util.List<Var> parts) {
        int cols = parts.get(0).value.cols, total = 0;
        for (Var p : parts) total += p.value.rows;
        Tensor t = new Tensor(total, cols);
        int off = 0;
        for (Var p : parts) { System.arraycopy(p.value.data, 0, t.data, off * cols, p.value.data.length); off += p.value.rows; }
        Var r = new Var(t, new ArrayList<>(parts));
        r.back = () -> {
            int o = 0;
            for (Var p : parts) {
                Tensor d = new Tensor(p.value.rows, cols);
                System.arraycopy(r.grad.data, o * cols, d.data, 0, d.data.length);
                p.g().addInPlace(d);
                o += p.value.rows;
            }
        };
        return r;
    }

    /** Same numbers, cut into rows of a different length (rows * cols must stay the same). */
    public Var reshape(int rows, int cols) {
        if ((long) rows * cols != value.data.length) throw new IllegalArgumentException("cannot reshape " + value.shape() + " into " + rows + "x" + cols);
        Var r = make(new Tensor(rows, cols, value.data.clone()), this);
        r.back = () -> g().addInPlace(new Tensor(value.rows, value.cols, r.grad.data));
        return r;
    }

    // ---------------------------------------------------------------- backward

    /** Like backward(), but starts from a gradient you give (for a result that is not a single number). */
    public void backward(Tensor seed) {
        if (seed.rows != value.rows || seed.cols != value.cols) throw new IllegalArgumentException("seed must have shape " + value.shape());
        List<Var> order = new ArrayList<>();
        topo(this, new IdentityHashMap<>(), order);
        grad = seed.copy();
        for (int i = order.size() - 1; i >= 0; i--) {
            Var v = order.get(i);
            if (v.grad != null) v.back.run();
        }
    }

    /** Call on a 1 x 1 loss. Fills in `grad` of every Var that took part in computing it. */
    public void backward() {
        if (value.rows != 1 || value.cols != 1) throw new IllegalStateException("backward() needs a single number (a loss), got " + value.shape());
        List<Var> order = new ArrayList<>();
        topo(this, new IdentityHashMap<>(), order);
        grad = new Tensor(1, 1);
        grad.data[0] = 1f;
        for (int i = order.size() - 1; i >= 0; i--) {
            Var v = order.get(i);
            if (v.grad != null) v.back.run();
        }
    }

    private static void topo(Var v, IdentityHashMap<Var, Boolean> seen, List<Var> order) {
        if (seen.put(v, Boolean.TRUE) != null) return;
        for (Var p : v.parents) topo(p, seen, order);
        order.add(v);
    }
}
