package minidl;

import java.util.Random;

/**
 * Dropout: while training, randomly switch off some neurons.
 * Stops the model from just memorizing the training data (overfitting).
 * While predicting, it does nothing.
 */
public class Dropout implements Layer {
    private final double p; // chance of switching a neuron off
    private final Random rnd;
    private boolean training = true;
    private Tensor mask;

    public Dropout(double p, Random rnd) {
        if (p < 0 || p >= 1) throw new IllegalArgumentException("p must be in [0,1)");
        this.p = p;
        this.rnd = rnd;
    }

    @Override public void setTraining(boolean training) { this.training = training; }

    @Override public Tensor forward(Tensor x) {
        if (!training) return x;
        double keep = 1.0 - p;
        mask = new Tensor(x.rows, x.cols);
        for (int i = 0; i < mask.data.length; i++)
            mask.data[i] = rnd.nextDouble() < keep ? (float) (1.0 / keep) : 0f; // scale up so average stays same
        return x.mul(mask);
    }

    @Override public Tensor backward(Tensor g) {
        return training ? g.mul(mask) : g;
    }

    @Override public String spec() { return "Dropout " + p; }
}
