package minidl;

/** Learning rate schedule: big steps early, small careful steps later. epoch starts at 0. */
public interface Schedule {
    double rate(int epoch, int totalEpochs);

    static Schedule constant(double lr) { return (e, t) -> lr; }

    /** Multiply the rate by `factor` every `every` epochs. */
    static Schedule stepDecay(double lr, int every, double factor) {
        return (e, t) -> lr * Math.pow(factor, e / every);
    }

    /** Smooth curve from `max` down to `min` over the whole training. */
    static Schedule cosine(double max, double min) {
        return (e, t) -> t <= 1 ? max : min + 0.5 * (max - min) * (1 + Math.cos(Math.PI * e / (t - 1)));
    }
}
