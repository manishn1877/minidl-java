package minidl;

import java.io.IOException;
import java.util.Random;

/**
 * Real-life style example: predict food delivery time (minutes)
 * from distance (km), traffic (0-10) and number of items.
 * Data is made up by a formula + noise, so we KNOW the right answer and can check the model.
 */
public class RealWorldDemo {

    /** Hidden "truth" the model must discover: 10 + 3.5*km + 2*traffic + 1.5*items (+ noise). */
    static Tensor[] makeOrders(int n, Random rnd) {
        Tensor x = new Tensor(n, 3), y = new Tensor(n, 1);
        for (int i = 0; i < n; i++) {
            double km = 1 + rnd.nextDouble() * 14;
            int traffic = rnd.nextInt(11);
            int items = 1 + rnd.nextInt(8);
            x.set(i, 0, km); x.set(i, 1, traffic); x.set(i, 2, items);
            y.set(i, 0, 10 + 3.5 * km + 2.0 * traffic + 1.5 * items + rnd.nextGaussian() * 3);
        }
        return new Tensor[]{x, y};
    }

    static Sequential buildModel(Random rnd) {
        return new Sequential()
            .add(new Dense(3, 16, rnd)).add(new Activations.ReLU())
            .add(new Dense(16, 8, rnd)).add(new Activations.ReLU())
            .add(new Dense(8, 1, rnd));
    }

    public static void deliveryTime() {
        System.out.println("\n=== 4. Real example: delivery time prediction ===");
        Random rnd = new Random(11);
        Tensor[] train = makeOrders(400, rnd);
        Tensor[] val = makeOrders(100, rnd);   // used during training to know when to stop
        Tensor[] test = makeOrders(100, rnd);  // never touched until the very end

        // Step 1: put everything on the same scale (learn the scale from training data only)
        Normalizer xn = new Normalizer().fit(train[0]);
        Normalizer yn = new Normalizer().fit(train[1]);

        // Step 2: build + train with validation, early stopping, cosine learning rate, AdamW
        Sequential model = buildModel(rnd);
        System.out.print(model.summary());
        Trainer.History h = new Trainer()
            .epochs(300).batchSize(32).seed(0).printEvery(50)
            .validation(xn.transform(val[0]), yn.transform(val[1]))
            .patience(30)
            .schedule(Schedule.cosine(0.01, 0.0005))
            .fit(model, xn.transform(train[0]), yn.transform(train[1]),
                 new Losses.MSE(), new Optimizers.Adam(0.01, 1e-4));
        System.out.printf("trained %d epochs, best validation at epoch %d%n", h.epochsRun, h.bestEpoch);

        // Step 3: check on orders the model has NEVER seen
        Tensor pred = yn.inverse(model.predict(xn.transform(test[0])));
        System.out.printf("unseen orders: average error %.2f min, RMSE %.2f min, R2 %.3f%n",
            Metrics.mae(pred, test[1]), Metrics.rmse(pred, test[1]), Metrics.r2(pred, test[1]));

        // Step 4: use it on one new order
        Tensor order = Tensor.of(new double[][]{{6.2, 8, 3}});
        double minutes = yn.inverse(model.predict(xn.transform(order))).get(0, 0);
        System.out.printf("new order (6.2 km, traffic 8, 3 items) -> %.1f min (formula says 52.2 without noise)%n", minutes);

        // Step 5: save everything, then load it like a separate app would (no layers rebuilt by hand)
        try {
            ModelIO.save(model, "delivery-model.txt");
            xn.save("delivery-x.txt");
            yn.save("delivery-y.txt");

            Sequential loaded = ModelIO.loadModel("delivery-model.txt", new Random());
            Normalizer lx = Normalizer.load("delivery-x.txt"), ly = Normalizer.load("delivery-y.txt");
            double again = ly.inverse(loaded.predict(lx.transform(order))).get(0, 0);
            System.out.printf("after save + load           -> %.1f min%n", again);
        } catch (IOException e) {
            System.out.println("save/load failed: " + e.getMessage());
        }
    }
}
