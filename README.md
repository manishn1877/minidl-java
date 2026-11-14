# MiniDL (Man-JAVA)

[![CI](https://github.com/manishn1877/minidl-java/actions/workflows/ci.yml/badge.svg)](https://github.com/manishn1877/minidl-java/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Java 17+](https://img.shields.io/badge/Java-17%2B-orange)
![Tests](https://img.shields.io/badge/tests-283%20passing-brightgreen)

**A deep learning library written from scratch in pure Java.** No libraries, no frameworks, just Java 17+.
Tensors, backpropagation, autograd, CNNs, LSTM, a small transformer, quantization, a model server and a handwriting recognition app, all small enough to read.

- Live report (graphs, all test results, PyTorch comparison): https://manishn1877.github.io/minidl-java/
- Handwriting recognition model page: https://manishn1877.github.io/minidl-java/#handwriting
- Author: [@manishn1877](https://github.com/manishn1877)

## Why this exists

Anyone can call `model.fit()`. This project shows what happens inside it: forward pass, loss, backward pass, update, repeated. Every layer is checked against a numerical gradient test, and results are measured on real data.

It is not the first library of its kind. Deeplearning4j (DL4J) and DJL are production libraries for Java, and PyTorch and TensorFlow dominate elsewhere. MiniDL is meant for learning, for portfolios, and for people who want to read every line of a working deep learning stack.

## Results

All numbers are from one run each, on held-out test pictures, 8 epochs (`./train-all.sh`).

| Model | Classes | Weights | Train time | Test accuracy |
|---|---|---|---|---|
| Digits 0-9 (MNIST, with augmentation) | 10 | 27,562 | 103 s | **99.02%** |
| Letters A-Z (EMNIST) | 26 | 110,682 | 427 s | **93.71%** |
| Digits + letters | 36 | 111,972 | 1176 s | **93.60%** |
| A-Z and a-z | 52 | 168,628 | 448 s | **85.23%** |
| Digits + A-Z + a-z | 62 | 170,558 | 957 s | **86.02%** |

A 16-epoch run of the digits CNN reached 99.19%. Other measured results: CIFAR-10 colour photos 70.64%, a tiny transformer at 2.39 bits per character on Tiny Shakespeare (random guessing is 6.02).

Shrinking the digits model (10,000 unseen digits):

| Format | Weights | Accuracy |
|---|---|---|
| float32 | 107.7 KB | 99.02% |
| bfloat16 | 56.4 KB | 99.02% |
| int8 | 27.6 KB | 98.99% |
| int4 (group 32) | 17.8 KB | 98.83% |

### Comparison with PyTorch

The same CNN layer sizes were trained in PyTorch for 8 epochs (Adam, EMNIST via torchvision, Google Colab). That script is not part of this repo, so the project stays 100% Java.

| Model | MiniDL | PyTorch | Gap (PyTorch minus MiniDL) | MiniDL time | PyTorch time |
|---|---|---|---|---|---|
| Digits 0-9 | 99.02% | 98.59% | -0.43 pp | 103 s | 132 s |
| Letters A-Z | 93.71% | 92.93% | -0.78 pp | 427 s | 304 s |
| Digits + letters | 93.60% | 90.26% | -3.34 pp | 1176 s | 1707 s |
| A-Z and a-z | 85.23% | 85.69% | +0.46 pp | 448 s | 966 s |
| Digits + A-Z + a-z | 86.02% | 86.60% | +0.58 pp | 957 s | 1718 s |

How to read it:
- Accuracy is within about 0.8 points on 4 of 5 models, as expected from the same maths on a small CNN.
- Digits + letters is not a fair comparison: the PyTorch version merges lowercase into uppercase (EMNIST byclass), MiniDL uses MNIST digits plus EMNIST letters.
- One run each, so gaps of a few tenths of a point are noise. Training details also differ (MiniDL restores the best validation epoch).
- Times come from different machines. This is not a speed benchmark. PyTorch is far faster on GPUs and big models.

## Quick start

You need a JDK 17 or newer.

```
./build-jar.sh                                 # builds dist/minidl-1.4.0.jar (+ sources, javadoc)
java -jar dist/minidl-1.4.0.jar tests          # 283 tests
java -jar dist/minidl-1.4.0.jar help           # demo, tests, mnist, emnist, cifar, text, precision, bench, web, serve, version
```

Train the handwriting models and open the web app:

```
./get-emnist.sh        # downloads EMNIST (and MNIST)
./train-all.sh         # trains digits, letters, digits+letters, A-Z a-z, full 62
./run-web.sh           # open http://localhost:8080
```

Or without the jar:

```
mkdir out && javac -d out src/minidl/*.java
java -cp out minidl.Tests
java -cp out minidl.Main            # small demos: gradient check, XOR, spirals, delivery time
```

Optional SIMD speed-up with Java's Vector API (7 to 28x faster matrix multiply, 2 to 4x faster training):

```
javac --add-modules jdk.incubator.vector -cp out -d out src-vector/minidl/*.java
java --add-modules jdk.incubator.vector -cp out minidl.Tests
```

## What is inside

| Area | What you get |
|---|---|
| Core | `Tensor`, `NDTensor` (views, broadcasting, batched matmul), `Var` (autograd, 33 operations), shape inference |
| Layers | Dense, Conv2D (stride, padding, dilation, groups), MaxPool2D, AvgPool2D, BatchNorm, BatchNorm2D, LayerNorm, Dropout, LSTM, causal self-attention, Residual, Embedding, TokenWise, Reshape, weight tying, ResNet builder |
| Activations | ReLU, LeakyReLU, Sigmoid, Tanh, GELU |
| Losses | MSE, MAE, BCEWithLogits, SoftmaxCrossEntropy |
| Optimizers | SGD (momentum, weight decay), Adam/AdamW, RMSProp, learning rate schedules, gradient clipping |
| Training | `Trainer` with validation, early stopping (restores best weights), schedules, clipping |
| Data | CSV, MNIST, EMNIST, CIFAR-10, `ByteSource` (1 byte per value), `MappedSource` (data stays on disk), augmentation (shift, rotate, flip), BPE tokenizer |
| Precision | float32, FP16 and BF16 training with loss scaling (numerics simulated, memory saving real) |
| Shrinking | int8, int4 (group-wise), bfloat16 weights, int8 convolution, compact binary files |
| Interop | SafeTensors export/import with PyTorch names and layouts (Dense, Conv2D, BatchNorm2D) |
| Serving | Built-in HTTP model server with batching, scaling and Prometheus metrics |
| Apps | Handwriting recognition web app, character-level transformer text generator, CIFAR-10 demo |

All code is in `src/minidl/`. The optional SIMD kernel is in `src-vector/`.

## Use it in your own project

```
./build-jar.sh                                    # dist/minidl-1.4.0.jar
javac -cp minidl-1.4.0.jar MyApp.java             # then: import minidl.*;
```

With Maven: `mvn verify` builds, tests and packages (no dependencies); `mvn install` puts it in your local repository, then:

```xml
<dependency><groupId>io.github.manishn1877</groupId><artifactId>minidl</artifactId><version>1.4.0</version></dependency>
```

It is not published to Maven Central.

### Example: predict a number

```java
Normalizer xn = new Normalizer().fit(trainX);
Normalizer yn = new Normalizer().fit(trainY);

Sequential m = new Sequential()
    .add(new Dense(numFeatures, 16, rnd)).add(new Activations.ReLU())
    .add(new Dense(16, 8, rnd)).add(new Activations.ReLU())
    .add(new Dense(8, 1, rnd));

new Trainer().epochs(300).batchSize(32)
    .validation(xn.transform(valX), yn.transform(valY)).patience(30)
    .fit(m, xn.transform(trainX), yn.transform(trainY), new Losses.MSE(), new Optimizers.Adam(0.01));

double answer = yn.inverse(m.predict(xn.transform(newRow))).get(0, 0);
```

### Example: image classifier (CNN)

```java
Sequential cnn = new Sequential()
    .add(new Conv2D(1, 28, 28, 8, 3, 1, 0, rnd)).add(new Activations.ReLU())
    .add(new MaxPool2D(8, 26, 26, 2))
    .add(new Conv2D(8, 13, 13, 16, 3, 1, 0, rnd)).add(new Activations.ReLU())
    .add(new MaxPool2D(16, 11, 11, 2))
    .add(new Dense(400, 64, rnd)).add(new Activations.ReLU())
    .add(new Dense(64, 10, rnd));
System.out.print(cnn.summary(1, 28, 28));   // checks every layer size before training
```

### Example: shrink and ship

```java
Sequential small = Quantizer.quantize(model);          // int8; also quantizeInt4(model, 32), toBF16(model)
ModelIO.saveBinary(small, "model-int8.bin");
Sequential served = ModelIO.loadModelBinary("model-int8.bin", new Random());
```

### Example: your own layer in one line (autograd)

```java
Layer swish = new AutoLayer("Swish", List.of(), (x, p) -> x.mul(x.sigmoid()));   // forward maths only
```

To write a layer with a hand-written backward, implement `Layer` (`forward`, `backward`, and `params()` if it has weights), then run `GradCheck` on a small model that contains it. A layer you invent needs `ModelIO.register("MyLayer", ...)` so saved files can be loaded.

## Handwriting recognition web app

Draw in the browser and the model reads it live. The server is pure Java (the JDK's built-in HTTP server) and the page is one HTML file (`web/index.html`).

```
./run-web.sh         # loads every trained model it finds, open http://localhost:8080
```

- **Writing modes:** one character with auto-clear, a line that locks each character after a short pause (so new strokes never merge with earlier ones), and a scan mode for photos.
- **Models:** digits (10), letters (26), digits + letters (36), A-Z a-z (52), digits + A-Z + a-z (62). The page lists every model it finds.
- **Capital vs small letters:** look-alike letters (c C, o O, s S ...) are decided by height against the tallest letter on the line. Buttons let you force upper or lower case.
- **Teach it:** press the correct character and the model learns from your drawing. "Train on my handwriting" walks you through every character.
- **Matches the training data:** preprocessing crops, scales and centres drawings like MNIST/EMNIST, so small or off-centre writing is still read.

GitHub Pages cannot run Java. On the Pages site the page draws, and the Java program on your own computer recognises. Start `./run-web.sh`, open the Pages address, and it connects to `http://localhost:8080`.

Limits: one user at a time (corrections change the one shared model), localhost only, not tested in a real phone browser. Look-alike pairs (0/O, 1/I, 5/S, 8/B) can only be guessed from shape.

## Model server

Serve any saved model over HTTP, pure Java:

```
java -jar dist/minidl-1.4.0.jar serve --port 8080 --model digits=mnist-cnn-aug.bin,batch=32,delay=5,workers=1,max_workers=4,queue=500
curl localhost:8080/ping
curl -X POST localhost:8080/predictions/digits -d '{"inputs": [[... 784 numbers ...]]}'
```

| Call | What it does |
|---|---|
| `GET /ping` | health |
| `GET /models`, `GET /models/{name}` | models, workers, queue, counts, latency p50/p99 |
| `POST /models?url=f.bin&name=n` | load a model (a file name inside the model folder, never a path) |
| `PUT /models/{name}`, `DELETE /models/{name}` | scale, unload |
| `POST /predictions/{name}` | `{"inputs": [[...]]}` returns outputs and argmax |
| `GET /metrics` | Prometheus text |

Requests are grouped into batches (one forward pass for many callers), a full queue answers 503, and a bad request gets 400 without disturbing others. Safety: localhost only, 8 MB body limit, 10,000 rows per request.

## PyTorch interop

MiniDL reads and writes the safetensors format, so weights can move between MiniDL and PyTorch with no Python in this project:

```java
SafeTensors.exportModel(model, "weights.safetensors");
SafeTensors.importModel(model, "weights.safetensors");   // into a model with the same layers
```

Dense, Conv2D and BatchNorm2D map to PyTorch names and layouts, and every name and shape is checked on import. Tested with a file built by hand from the format description. **Not yet tested with a file written by real PyTorch.**

## Tests

```
java -jar dist/minidl-1.4.0.jar tests      # 283 tests, all passing
```

No JUnit needed. The suite covers:

- **Maths:** matmul against hand-checked values and the textbook version, multi-thread equals single-thread, SIMD kernels against textbook on awkward shapes
- **Gradient checks** for every layer type: Dense with all activations, BatchNorm, LayerNorm, Conv2D (stride, padding, dilation, groups, depthwise), pooling, LSTM, ResNet stack, Residual, the whole transformer
- **Autograd:** 24 operations checked, autograd layers equal their hand-written twins, a transformer built only from autograd layers matches the hand-written one
- **16-bit numbers:** all 65,536 FP16 patterns round-trip, rounding, overflow and underflow, loss scaling, overflow step skipping
- **Transformer:** attention against a double-precision reference, no peeking at future characters
- **Shrinking and files:** int8, int4, bfloat16, save and load (text and binary) with identical predictions, damaged files give clear errors
- **Data and apps:** loaders, augmentation, preprocessing, character segmentation, case rules, BPE, safetensors, model server (batching, errors, limits), web security headers

CI runs the tests on Java 17 and 21 on every push (`.github/workflows/ci.yml`). The code compiles with `-Xlint:all` without warnings.

## Honest limits

- **Not production ready.** For real products use DL4J, PyTorch or TensorFlow.
- **Speed:** CPU only. The SIMD kernel needs the incubator Vector API flag and was tested on Java 21 with AVX-512. Only matrix multiplies use SIMD, so whole-network speed-ups are 2 to 4x. The multi-core speed-up is untested (the test machine had one core). Shrunk int8/int4 models save memory but are slower than float32.
- **16-bit training is simulated:** values are rounded like 16-bit hardware, the maths runs in float32. You get the numerics and the memory saving, not the speed.
- **Layers use 2D tables internally.** `NDTensor` and shape inference give N-dimensional shapes, but layers do not use `NDTensor` inside.
- **Autograd layers** are about 2x slower and use about 3x the memory of hand-written ones.
- **Results** are one run each with one seed and no tuning. CIFAR-10 at 70.64% is modest. The text model learned spelling and layout, not meaning.
- **Datasets:** the CIFAR-10 data used for the reported result came from an npm mirror (sizes, label order and class counts were checked), not the official site.

## Roadmap

Done: training toolkit, speed and memory work (float32, Vector API, bytes and mapped data, int8/int4/bfloat16, 16-bit training), convolution, augmentation, autograd, LayerNorm and residuals, attention and a tiny transformer, LSTM, BatchNorm2D and ResNet, BPE tokenizer, weight tying, handwriting recognition (digits, letters, case), model server, safetensors, CI, report site.

Next ideas:
- [ ] Check SafeTensors with a file written by real PyTorch
- [ ] SIMD int8 / int4 kernels (so shrunk models are fast as well as small)
- [ ] Vector API for im2col, pooling, activations and attention
- [ ] Test the SIMD build on Java 17 and on an AVX2 CPU
- [ ] Train the ResNet on CIFAR-10 and see whether it passes 85%
- [ ] Make layers use `NDTensor` internally
- [ ] 16-bit storage for LayerNorm, BatchNorm and attention
- [ ] Record a short demo video of the handwriting app

## Contributing

Contributions are welcome.

1. Fork the repo and create a branch.
2. Build with `./build-jar.sh`.
3. Run `java -jar dist/minidl-1.4.0.jar tests`. All 283 tests must pass.
4. Add a test for anything new. New layers need a gradient check.
5. Keep it pure Java: no outside ML libraries and no Python files in the project (shell scripts and the HTML page are fine).
6. Open a pull request and say what you changed and why.

Look for issues labelled `good first issue`. More detail: `CONTRIBUTING.md`.

## More documentation

- `TUTORIAL.md`: the full step-by-step guide with hand-checkable examples (the old long README)
- `CHANGELOG.md`: what changed in each version
- `docs/DJL-ANALYSIS.md`: notes from studying DJL Serving

## License

MIT. See [LICENSE](LICENSE).

Built by [Manish Gowda N](https://github.com/manishn1877).
