package minidl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A small ResNet-style picture classifier built from Conv2D, BatchNorm2D and Residual blocks.
 *
 *   stem:    Conv 3x3 -> BatchNorm2D -> ReLU
 *   stage 1: [Residual( Conv -> BN -> ReLU -> Conv -> BN ) -> ReLU] x blocks        at full size
 *   stage 2: Conv 3x3 stride 2 (twice the channels) -> BN -> ReLU, then the same blocks   at half size
 *   stage 3: the same again at a quarter of the size
 *   head:    average the whole picture per channel (AvgPool2D) -> Dense -> classes (logits)
 *
 * The shortcut (x + f(x)) lets the gradient flow straight to early layers, so deeper nets still train.
 * Size must be divisible by 4. CIFAR-10: ResNet.build(3, 32, 32, 16, 2, 10, rnd). MNIST: ResNet.build(1, 28, 28, 8, 1, 10, rnd).
 */
public final class ResNet {
    private ResNet() {}

    public static Sequential build(int inC, int h, int w, int width, int blocksPerStage, int classes, Random rnd) {
        if (h % 4 != 0 || w % 4 != 0 || h != w) throw new IllegalArgumentException("ResNet needs a square picture whose size is divisible by 4");
        Sequential m = new Sequential();
        m.add(new Conv2D(inC, h, w, width, 3, 1, 1, rnd)).add(new BatchNorm2D(width, h, w)).add(new Activations.ReLU());
        int c = width, size = h;
        for (int stage = 0; stage < 3; stage++) {
            if (stage > 0) {
                m.add(new Conv2D(c, size, size, c * 2, 3, 2, 1, rnd));
                c *= 2;
                size = (size + 2 - 3) / 2 + 1;
                m.add(new BatchNorm2D(c, size, size)).add(new Activations.ReLU());
            }
            for (int b = 0; b < blocksPerStage; b++) {
                List<Layer> inner = new ArrayList<>();
                inner.add(new Conv2D(c, size, size, c, 3, 1, 1, rnd)); inner.add(new BatchNorm2D(c, size, size)); inner.add(new Activations.ReLU());
                inner.add(new Conv2D(c, size, size, c, 3, 1, 1, rnd)); inner.add(new BatchNorm2D(c, size, size));
                m.add(new Residual(inner)).add(new Activations.ReLU());
            }
        }
        m.add(new AvgPool2D(c, size, size, size));
        m.add(new Dense(c, classes, rnd));
        return m;
    }
}
