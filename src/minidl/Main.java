package minidl;

import java.util.Random;

public class Main {
    public static void main(String[] args) {
        gradientCheck();
        xor();
        spiral();
        RealWorldDemo.deliveryTime();
    }

    /** 1) Is our backprop mathematically correct? */
    static void gradientCheck() {
        System.out.println("=== 1. Gradient check ===");
        Random rnd = new Random(1);
        Sequential m = new Sequential()
            .add(new Dense(3, 5, rnd)).add(new BatchNorm(5)).add(new Activations.GELU())
            .add(new Dense(5, 2, rnd));
        Tensor x = Tensor.randn(4, 3, 1.0, rnd);
        Tensor y = Tensor.randn(4, 2, 1.0, rnd);
        double err = GradCheck.maxRelativeError(m, new Losses.MSE(), x, y);
        System.out.printf("max relative error = %.2e  %s   (float32; a real bug gives 0.1 or more)%n%n", err, err < 5e-3 ? "PASS" : "FAIL");
    }

    /** 2) XOR: a single straight line can't solve it, a hidden layer can. */
    static void xor() {
        System.out.println("=== 2. XOR with SGD ===");
        Random rnd = new Random(42);
        Tensor x = Tensor.of(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Tensor y = Tensor.of(new double[][]{{0}, {1}, {1}, {0}});
        Sequential m = new Sequential()
            .add(new Dense(2, 8, rnd)).add(new Activations.Tanh())
            .add(new Dense(8, 1, rnd)).add(new Activations.Sigmoid());
        m.fit(x, y, new Losses.MSE(), new Optimizers.SGD(0.5, 0.9), 2000, 4, 0, 500);
        System.out.println("predictions:");
        System.out.print(m.predict(x));
        System.out.println();
    }

    /** 3) 3 spirals, classify with ReLU + softmax + Adam. Now with a proper held-out test set. */
    static void spiral() {
        System.out.println("=== 3. Spiral, 3 classes with Adam ===");
        Random rnd = new Random(7);
        int perClass = 100, classes = 3, n = perClass * classes;
        Tensor x = new Tensor(n, 2), y = new Tensor(n, classes);
        for (int c = 0; c < classes; c++)
            for (int i = 0; i < perClass; i++) {
                int row = c * perClass + i;
                double r = i / (double) perClass;
                double t = c * 4 + 4.0 * i / perClass + rnd.nextGaussian() * 0.2;
                x.set(row, 0, r * Math.sin(t));
                x.set(row, 1, r * Math.cos(t));
                y.set(row, c, 1.0);
            }
        Tensor[] s = Data.trainTestSplit(x, y, 0.2, 1); // 80% train, 20% test
        Sequential m = new Sequential()
            .add(new Dense(2, 64, rnd)).add(new Activations.ReLU())
            .add(new Dense(64, 64, rnd)).add(new Activations.ReLU())
            .add(new Dense(64, classes, rnd));
        m.fit(s[0], s[1], new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.01), 300, 32, 0, 100);
        System.out.printf("train accuracy = %.1f%%   TEST accuracy (unseen points) = %.1f%%%n",
            m.accuracy(s[0], s[1]), m.accuracy(s[2], s[3]));
    }
}
