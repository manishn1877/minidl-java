package minidl;

import java.util.List;
import java.util.Random;

/** Fully connected layer: out = x . W + b */
public class Dense implements Layer {
    private final int in, out;
    private final Param W, b;
    private Tensor input;      // remembered for backward (32-bit)
    private short[] input16;   // ... or in 16 bits when activation storage is FP16 / BF16
    private int inputRows;
    private Precision store = Precision.FP32;
    private boolean inputGrad = true;

    public Dense(int in, int out, Random rnd) {
        this.in = in;
        this.out = out;
        // He init: keeps signals from shrinking/exploding
        W = new Param(Tensor.randn(in, out, Math.sqrt(2.0 / in), rnd), true);
        b = new Param(new Tensor(1, out), false);
    }

    @Override public Tensor forward(Tensor x) {
        if (store == Precision.FP32) { input = x; input16 = null; }
        else { input16 = Half.pack(x.data, 0, x.data.length, store, input16); inputRows = x.rows; input = null; } // half the memory until backward
        Tensor y = x.matmul(W.value);
        y.addRowVectorInPlace(b.value);       // bias added in place, no extra copy
        return y;
    }

    @Override public Tensor backward(Tensor g) {
        Tensor in = input != null ? input : Half.unpack(input16, inputRows, W.value.rows, store);
        in.matmulTAInto(g, W.grad);           // dLoss/dW = x^T . g   (written straight into the gradient)
        g.sumRowsInto(b.grad);                // dLoss/db = sum of g over the batch
        if (!inputGrad) return null;          // first layer of a model: nobody needs dLoss/dx
        return g.matmulTB(W.value);           // dLoss/dx = g . W^T   (no transposed copy)
    }

    @Override public void setComputeInputGrad(boolean on) { inputGrad = on; }
    @Override public void setActivationStorage(Precision p) { store = p; }

    @Override public List<Param> params() { return List.of(W, b); }

    @Override public int[] outputShape(int[] in) {
        int want = W.value.rows;
        if (NDTensor.product(in) != want) throw new IllegalArgumentException("Dense expects " + want + " numbers per sample but gets " + java.util.Arrays.toString(in) + " = " + NDTensor.product(in));
        return new int[]{W.value.cols};
    }

    @Override public String spec() { return "Dense " + in + " " + out; }
}
