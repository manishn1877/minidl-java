package minidl;

/**
 * A tiny Kalman filter for one coordinate with a constant-velocity model: it smooths a jittery measurement (the fingertip) and
 * can keep guessing the position for a few frames when the measurement is missing. Two of them (x and y) track a fingertip.
 * State: position and velocity. Measurement: position only.
 */
public final class Kalman1D {
    private double pos, vel;
    private double p00 = 1e3, p01, p10, p11 = 1e3;  // covariance of (pos, vel)
    private boolean started;
    private final double q, r;

    /** q = how much the speed may change by itself (per second squared), r = how noisy the measurement is (pixels squared). */
    public Kalman1D(double q, double r) { this.q = q; this.r = r; }

    public boolean started() { return started; }
    public double position() { return pos; }
    public double velocity() { return vel; }

    /** Moves the estimate forward by dt seconds without a measurement. Returns the predicted position. */
    public double predict(double dt) {
        if (!started) return pos;
        pos += vel * dt;
        double n00 = p00 + dt * (p10 + p01) + dt * dt * p11 + q * dt * dt * dt * dt / 4;
        double n01 = p01 + dt * p11 + q * dt * dt * dt / 2;
        double n10 = p10 + dt * p11 + q * dt * dt * dt / 2;
        double n11 = p11 + q * dt * dt;
        p00 = n00; p01 = n01; p10 = n10; p11 = n11;
        return pos;
    }

    /** Predicts dt seconds ahead, then corrects with the measured position. Returns the filtered position. */
    public double update(double measured, double dt) {
        if (!started) { pos = measured; vel = 0; p00 = r; p01 = p10 = 0; p11 = 1e3; started = true; return pos; }
        predict(dt);
        double s = p00 + r, k0 = p00 / s, k1 = p10 / s, y = measured - pos;
        pos += k0 * y;
        vel += k1 * y;
        double n00 = (1 - k0) * p00, n01 = (1 - k0) * p01, n10 = p10 - k1 * p00, n11 = p11 - k1 * p01;
        p00 = n00; p01 = n01; p10 = n10; p11 = n11;
        return pos;
    }

    public void reset() { started = false; pos = vel = 0; p00 = p11 = 1e3; p01 = p10 = 0; }
}
