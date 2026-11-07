package minidl;

import java.util.List;

/** Optimizer = uses gradients to update the weights. */
public interface Optimizer {
    void step(List<Param> params);
    void setLearningRate(double lr);
    double getLearningRate();
}
