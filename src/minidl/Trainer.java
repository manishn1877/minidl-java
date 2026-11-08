package minidl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Full training loop with the useful extras:
 *  - validation data (rows the model never trains on, used to watch for overfitting)
 *  - early stopping (stop when validation stops improving, keep the best weights)
 *  - learning rate schedule
 *  - gradient clipping
 *  - works on a DataSource, so big data can stay as compact bytes and become floats one batch at a time
 *
 * Example:
 *   History h = new Trainer().epochs(300).batchSize(32).validation(vx, vy).patience(20)
 *                   .schedule(Schedule.cosine(0.01, 0.0005)).fit(model, x, y, loss, opt);
 */
public class Trainer {
    private int epochs = 100, batchSize = 32, patience = 0, printEvery = 0, maxBatches = 0;
    private long seed = 0;
    private double clipNorm = 0;
    private boolean restoreBest = true;
    private Schedule schedule = null;
    private DataSource val;

    public Trainer epochs(int e) { this.epochs = e; return this; }
    public Trainer batchSize(int b) { this.batchSize = b; return this; }
    public Trainer seed(long s) { this.seed = s; return this; }
    /** Stop each epoch after this many batches (0 = use all data). Handy when the data is huge (text). */
    public Trainer maxBatchesPerEpoch(int m) { this.maxBatches = m; return this; }
    public Trainer printEvery(int p) { this.printEvery = p; return this; }
    public Trainer schedule(Schedule s) { this.schedule = s; return this; }
    public Trainer clipNorm(double c) { this.clipNorm = c; return this; }
    /** Stop if validation loss has not improved for this many epochs. 0 = never stop early. */
    public Trainer patience(int p) { this.patience = p; return this; }
    public Trainer restoreBest(boolean r) { this.restoreBest = r; return this; }
    public Trainer validation(Tensor x, Tensor y) { this.val = new TensorSource(x, y); return this; }
    public Trainer validation(DataSource v) { this.val = v; return this; }

    public static class History {
        public final List<Double> trainLoss = new ArrayList<>();
        public final List<Double> valLoss = new ArrayList<>();
        public int epochsRun = 0, bestEpoch = 0;
        public double bestValLoss = Double.NaN;
        public boolean stoppedEarly = false;
    }

    public History fit(Sequential model, Tensor x, Tensor y, Loss loss, Optimizer opt) {
        return fit(model, new TensorSource(x, y), loss, opt);
    }

    public History fit(Sequential model, DataSource train, Loss loss, Optimizer opt) {
        Random rnd = new Random(seed);
        int n = train.size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;

        History h = new History();
        double bestVal = Double.POSITIVE_INFINITY;
        List<float[]> best = null;
        int sinceBest = 0;

        for (int e = 1; e <= epochs; e++) {
            if (schedule != null) opt.setLearningRate(schedule.rate(e - 1, epochs));

            for (int i = n - 1; i > 0; i--) { // shuffle
                int j = rnd.nextInt(i + 1);
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
            double total = 0; int batches = 0;
            for (int s = 0; s < n; s += batchSize) {
                int end = Math.min(s + batchSize, n);
                total += model.trainStep(train.features(idx, s, end), train.targets(idx, s, end), loss, opt, clipNorm);
                batches++;
                if (maxBatches > 0 && batches >= maxBatches) break;
            }
            double tl = total / batches;
            h.trainLoss.add(tl);
            h.epochsRun = e;

            double vl = Double.NaN;
            if (val != null) {
                vl = model.evalLoss(val, loss);
                h.valLoss.add(vl);
                if (vl < bestVal) {
                    bestVal = vl; h.bestEpoch = e; h.bestValLoss = vl;
                    best = model.snapshot();
                    sinceBest = 0;
                } else sinceBest++;
            }

            if (printEvery > 0 && (e % printEvery == 0 || e == 1)) {
                if (val != null) System.out.printf("epoch %4d  loss %.5f  val %.5f%n", e, tl, vl);
                else System.out.printf("epoch %4d  loss %.5f%n", e, tl);
            }

            if (val != null && patience > 0 && sinceBest >= patience) {
                h.stoppedEarly = true;
                if (printEvery > 0) System.out.printf("early stop at epoch %d (best was epoch %d)%n", e, h.bestEpoch);
                break;
            }
        }
        if (val != null && restoreBest && best != null) model.restore(best);
        model.setTraining(false);
        return h;
    }
}
