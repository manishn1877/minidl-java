package minidl;

/** Loss = one number saying "how wrong is the model". Lower is better. */
public interface Loss {
    double forward(Tensor pred, Tensor target);
    /** Gradient of the loss w.r.t. the predictions. Call after forward. */
    Tensor backward();
}
