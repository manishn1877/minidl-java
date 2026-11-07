package minidl;

/** A learnable weight: its value, plus the gradient (how the loss changes if we nudge it). */
public class Param {
    public final Tensor value;
    public final Tensor grad;
    /** Should weight decay (L2) apply? Yes for weights, no for biases and BatchNorm scale/shift. */
    public final boolean decay;

    public Param(Tensor value) { this(value, true); }

    public Param(Tensor value, boolean decay) {
        this.value = value;
        this.grad = new Tensor(value.rows, value.cols);
        this.decay = decay;
    }

    public void zeroGrad() { java.util.Arrays.fill(grad.data, 0f); }
}
