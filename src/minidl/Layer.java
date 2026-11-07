package minidl;

import java.util.Collections;
import java.util.List;

/** One building block. forward = go ahead, backward = send blame back. */
public interface Layer {
    Tensor forward(Tensor x);

    /** gradOut = how loss changes w.r.t. this layer's output. Returns same for its input. */
    Tensor backward(Tensor gradOut);

    default List<Param> params() { return Collections.emptyList(); }

    /** Numbers a layer keeps that are NOT learned by gradients (e.g. BatchNorm running averages). Saved with the model. */
    default List<Tensor> state() { return Collections.emptyList(); }

    /** Some layers (Dropout, BatchNorm) act differently while training vs predicting. */
    default void setTraining(boolean training) {}

    /** The first layer of a model never needs to pass blame further back, so it can skip that work. */
    default void setComputeInputGrad(boolean on) {}

    /** Memory used by this layer's learned numbers, in bytes (float32 = 4 bytes each). */
    default long weightBytes() {
        long b = 0;
        for (Param p : params()) b += p.value.bytes();
        for (Tensor t : state()) b += t.bytes();
        return b;
    }

    /** Extra data that is not a Tensor (used by quantized layers) for the binary model file. */
    default void writeExtra(java.io.DataOutput out) throws java.io.IOException {}
    default void readExtra(java.io.DataInput in) throws java.io.IOException {}

    /** Keep what backward needs in 16 bits (FP16 / BF16) instead of 32. Layers that cannot ignore this call. */
    default void setActivationStorage(Precision p) {}

    /**
     * The shape of ONE sample after this layer, given the shape of one sample going in (the batch size is left out).
     * Throws IllegalArgumentException with a clear message when the incoming shape does not fit.
     */
    default int[] outputShape(int[] in) { return in; }

    /** One-line description used when saving a model, e.g. "Dense 3 16". */
    default String spec() { return getClass().getSimpleName(); }
}
