# MiniDL: a tiny deep learning library in pure Java

Built from scratch. **No libraries. No frameworks.** Just Java 17+.
Tensors, backpropagation, automatic differentiation, optimizers, layers (dense, convolution, attention), a full training loop, data loaders, data augmentation, and work on both speed and memory: the same ideas inside PyTorch, TensorFlow and DL4J, but small enough to read in a weekend.

**Real results (measured):**
- A small convolutional network trained on shifted and rotated digits (data augmentation) reaches **99.19%** on the 10,000 unseen MNIST test digits, in pure Java on one CPU core, inside a 256 MB memory limit. Without augmentation (8 epochs) it was 98.91%.
- That model shrinks **6x with int4 weights for a 0.09 point loss** (99.10%), **3.9x with int8** (99.17%), and 1.9x with bfloat16 (99.19%, no loss).
- A **tiny GPT-style transformer** (110,529 weights, built from the library's own layers) learns to write Shakespeare-like text one character at a time: 2.39 bits per character versus 6.02 for random guessing.
- **Autograd**: a second, automatic way to get gradients. A whole transformer built only from autograd layers gives the same loss and gradients as the hand-written one (gradients agree to 8e-7), at about 2x the time and 3x the memory.
- **Optional SIMD kernels** (Java's Vector API): matrix multiply 7 to 28x faster than the portable loops, whole training epochs 2 to 4x faster. The CNN above now trains at 12.3 s per epoch instead of 25.7.
- **16-bit mixed-precision training** with loss scaling (FP16 and bfloat16): activations stored in 16 bits cut the memory held for backprop by 2.5x (plain network) to 5.2x (CNN).
- **CIFAR-10 colour photos**: a 156,074-weight CNN reaches **70.64%** on the 10,000 unseen test photos (random guessing is 10%).
- Dilated and grouped (depthwise) convolution, N-dimensional tensors with broadcasting, and shape inference that catches wrong layer sizes before training.
- 171 automated tests, run twice (portable kernels and Vector API kernels), including numerical gradient checks for every layer type, attention compared with a plain double-precision version, a proof that the transformer cannot peek at future characters, and exhaustive checks of the 16-bit number formats.

> Why this exists: anyone can call `model.fit()`. This project shows you know **what happens inside** `fit()`.

---

## Table of contents

1. [Does this already exist?](#1-does-this-already-exist)
2. [Quick start](#2-quick-start)
3. [The big idea in plain words](#3-the-big-idea-in-plain-words)
4. [Project map](#4-project-map)
5. [Step by step, with real examples](#5-step-by-step-with-real-examples) (Steps 1 to 32)
6. [Efficiency: less memory, more speed](#6-efficiency-less-memory-more-speed)
7. [Full real-life walkthrough: delivery time predictor](#7-full-real-life-walkthrough-delivery-time-predictor)
8. [The other demos (including real MNIST)](#8-the-other-demos-including-real-mnist)
9. [Use it in your own project](#9-use-it-in-your-own-project)
10. [Add your own layer](#10-add-your-own-layer)
11. [How this connects to real AI](#11-how-this-connects-to-real-ai)
12. [Limits (be honest about these)](#12-limits-be-honest-about-these)
13. [Future: what to build next](#13-future-what-to-build-next)
14. [Troubleshooting](#14-troubleshooting)
15. [Glossary](#15-glossary)
16. [Tests](#16-tests)

---

## 1. Does this already exist?

**Short answer: yes, the idea is not new.** Be honest about this in interviews. A quick search of GitHub shows:

- Lots of small **"neural network from scratch in Java"** projects (multi-layer perceptrons with backprop, XOR examples, visualizers). Some say openly that they are learning projects and not efficient.
- A whole family of **"micrograd"-style** projects: tiny autograd engines copied into Python, C++, Rust, JavaScript and more.
- Big **production libraries**: Deeplearning4j (DL4J), and others for Java. TensorFlow and PyTorch dominate everywhere else.

So nobody should call this "the first". What makes *this* project worth showing:

| Many small repos | MiniDL |
|---|---|
| One network, XOR demo | A reusable library: layers, losses, optimizers, schedules, trainer |
| "Trust me, backprop works" | **Gradient checks** prove it, and 171 tests guard it |
| Toy data only | **99.19% on real MNIST** with a convolutional network (held-out test set), plus a transformer that writes text |
| Slow and memory hungry, and says so | Measured speed **and memory** work with a benchmark you can re-run (section 6): float32, byte-sized data, int8 models |
| Code only | A README that explains every step with numbers you can check by hand |
| Plain loops, no SIMD | Optional Vector API kernels: 7 to 28x faster matrix multiply, with a portable fallback and tests for both paths (section 6) |
| Dense layers only | Convolution, pooling, LayerNorm, skip connections and causal self-attention, with memory-saving designs |
| Gradients by hand only | Both: hand-written backward for every layer **and** a small autograd engine (`Var`) |
| Hard to extend | Add a layer in about 10 lines (section 10) |

The honest pitch: *"I built a small deep learning library to understand how training really works, then tested it against real data and measured and improved its speed."*

---

## 2. Quick start

You need a JDK (Java 17 or newer) with `javac`.

```bash
./run.sh                 # compile + 171 tests (twice) + demos (about 13 seconds)
```

Or by hand:

```bash
mkdir out
javac -d out src/minidl/*.java
java -cp out minidl.Tests       # 171 tests
java -cp out minidl.Main        # demos 1-4
java -cp out minidl.Benchmark   # speed + memory test

# Optional: SIMD kernels with Java's Vector API (7 to 28x faster matrix multiply)
javac --add-modules jdk.incubator.vector -cp out -d out src-vector/minidl/*.java
java --add-modules jdk.incubator.vector -cp out minidl.Tests
java --add-modules jdk.incubator.vector -cp out minidl.Benchmark
```

Real MNIST digits (needs internet once, about 11 MB):

```bash
./get-mnist.sh
java -Xmx256m -cp out minidl.MnistDemo mnist 8 cnn     # convolutional network, only 256 MB of memory
java -Xmx256m -cp out minidl.MnistDemo mnist 10 mlp    # plain Dense network
java -Xmx256m -cp out minidl.MnistDemo mnist 16 cnn 60000 aug   # with data augmentation: 99.19% (about 7 minutes)
java -Xmx64m  -cp out minidl.MnistDemo mnist 2 mlp 60000 mmap   # images stay on disk, only 64 MB of Java memory
```

16-bit (mixed precision) training, and colour photos:

```bash
java --add-modules jdk.incubator.vector -Xmx256m -cp out minidl.MixedPrecisionDemo mnist 2   # gradient fidelity + 4 training runs
./get-cifar.sh                      # downloads and converts CIFAR-10 into ./cifar (see Step 32 for where the data comes from)
java --add-modules jdk.incubator.vector -Xmx512m -cp out minidl.CifarDemo cifar-folder 8 aug   # 70.6%, about 8 minutes
```

A character-level transformer (about 10 minutes for 3,000 steps on one core):

```bash
curl -LO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
java -Xmx512m -cp out minidl.CharTransformerDemo input.txt 3000
```

What `Main` prints (numbers from a real run):

```
=== 1. Gradient check ===
max relative error = 1.85e-03  PASS   (float32; a real bug gives 0.1 or more)

=== 2. XOR with SGD ===
predictions: 0.0032, 0.9936, 0.9934, 0.0074      (want 0, 1, 1, 0)

=== 3. Spiral, 3 classes with Adam ===
train accuracy = 99.6%   TEST accuracy (unseen points) = 98.3%

=== 4. Real example: delivery time prediction ===
unseen orders: average error 2.60 min, RMSE 3.22 min, R2 0.963
new order (6.2 km, traffic 8, 3 items) -> 51.2 min (formula says 52.2 without noise)
after save + load           -> 51.2 min
```

---

## 3. The big idea in plain words

A neural network is a big pile of numbers (called **weights**). At the start the weights are random, so the answers are garbage. Training means **fixing the weights, a tiny bit at a time**, until the answers are good.

Every training step is the same 5 moves:

| # | Move | Real-life meaning |
|---|------|-------------------|
| 1 | **Forward**: push data through the layers, get a prediction | You guess: "this delivery will take 20 min" |
| 2 | **Loss**: measure how wrong the guess is | Real time was 30 min. You were off by 10. |
| 3 | **Backward**: find which weights caused the error, and by how much | "My distance weight was too small, that is most of the mistake." |
| 4 | **Update**: nudge each weight a little in the direction that reduces the error | Make the distance weight a bit bigger |
| 5 | **Repeat** thousands of times | Practice |

That is all of deep learning. ChatGPT, image generators, self-driving cars: same 5 moves, much bigger.

### A real calculation you can check by hand

Goal: predict delivery time from distance. Model is the simplest possible: `time = w * distance + b`.
One real order: **5 km took 30 minutes**. We start with a bad guess: `w = 2`, `b = 0`. Learning rate `0.01`.

**Round 1**
- Forward: `2 * 5 + 0 = 10` minutes (model says 10, real is 30)
- Loss: `(10 - 30)^2 = 400`
- Backward (gradients): `dLoss/dw = 2 * (10 - 30) * 5 = -200`, `dLoss/db = 2 * (10 - 30) = -40`
- Update: `w = 2 - 0.01 * (-200) = 4`, `b = 0 - 0.01 * (-40) = 0.4`

**Round 2**
- Forward: `4 * 5 + 0.4 = 20.4`, loss `(20.4 - 30)^2 = 92.16`
- Gradients: `-96` and `-19.2`
- Update: `w = 4.96`, `b = 0.592`

**Round 3**
- Forward: `4.96 * 5 + 0.592 = 25.39`, loss `21.23`

Loss went **400 → 92.16 → 21.23**. The model is learning. MiniDL does exactly this, just for thousands of weights at once.

---

## 4. Project map

All code is in `src/minidl/`.

| File | What it is | Simple version |
|------|-----------|----------------|
| `Tensor.java` | 2D grid of 32-bit numbers + fast matrix maths | The spreadsheet everything lives in |
| `Param.java` | A weight, its gradient, and whether decay applies | A knob, plus a note saying which way to turn it |
| `Layer.java` | Interface: `forward()`, `backward()`, `params()`, `state()` | The shape every building block must follow |
| `Dense.java` | Fully connected layer: `x·W + b` | The main building block |
| `Activations.java` | `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU` | Lets the network learn curves |
| `Conv2D.java` | Convolution (finds patterns in images) | Small filters sliding over a picture |
| `MaxPool2D.java` | Max pooling | Shrinks the picture, keeps the strongest signals |
| `AvgPool2D.java` | Average pooling | Shrinks the picture by averaging squares |
| `LayerNorm.java` | Normalizes each row on its own | The normalization transformers use |
| `Residual.java` | Skip connection: `x + f(x)` | A shortcut for the gradient |
| `Embedding.java` | Token ids to learned vectors, plus positions | How letters/words become numbers |
| `CausalSelfAttention.java` | Multi-head attention, no peeking ahead | The heart of a transformer |
| `TokenWise.java`, `Reshape.java` | Run ordinary layers on every token; re-cut rows without copying | Glue for sequence models |
| `BatchNorm.java` | Batch normalization | Keeps numbers in a healthy range between layers |
| `Dropout.java` | Randomly switches neurons off while training | Stops memorizing |
| `Loss.java`, `Losses.java` | `MSE`, `MAE`, `BCEWithLogits`, `SoftmaxCrossEntropy` | The "how wrong am I" score |
| `Optimizer.java`, `Optimizers.java` | `SGD` (+momentum, +L2), `Adam`/`AdamW`, `RMSProp` | The rule for nudging weights |
| `Schedule.java` | Constant, step decay, cosine | Big steps early, small steps late |
| `GradClip.java` | Gradient clipping | Stops one bad batch from wrecking the weights |
| `Sequential.java` | Stacks layers, summary, snapshot/restore | The model itself |
| `Trainer.java` | Training loop with validation, early stopping, schedules, clipping | The coach |
| `DataSource.java`, `TensorSource.java`, `ByteSource.java` | Where batches come from; `ByteSource` keeps images as 1 byte per pixel | Big data in small memory |
| `QuantizedDense.java`, `QuantizedConv2D.java`, `QuantizedDense4.java`, `BF16Dense.java`, `Quantizer.java` | int8 / int4 / bfloat16 versions of the layers, for prediction | 2x to 8x smaller models |
| `AugmentedSource.java` | Random shifts and rotations of training images | Many more examples for free |
| `MappedSource.java` | Data that stays on disk (memory-mapped) | Datasets bigger than RAM |
| `CharSource.java` | Text windows for next-character prediction | Text as training data |
| `Var.java` | Automatic differentiation (33 operations) | You write forward, it writes backward |
| `AutoLayer.java`, `AutoLayers.java` | Layers written ONLY as forward maths: dense, LayerNorm, GELU, attention | New layers in one line |
| `NDTensor.java` | N-dimensional arrays: views, permute, broadcasting, batched matmul | Images as [N,C,H,W] |
| `GemmKernel.java`, `src-vector/.../VectorGemm.java` | Plug-in matrix-multiply engine; the optional SIMD version | 7 to 28x faster multiply |
| `Precision.java`, `Half.java`, `LossScaler.java` | FP16 / BF16 numbers and loss scaling | 16-bit training |
| `CifarLoader.java`, `CifarConvert.java`, `CifarDemo.java` | CIFAR-10 colour photos: loader, converter, demo | Colour images |
| `MixedPrecisionDemo.java` | 16-bit experiment on real digits | Does loss scaling matter? |
| `Data.java` | CSV loading, one-hot labels, train/test split | Getting real data in |
| `MnistLoader.java` | Reads MNIST digit files | Handwritten digits |
| `Metrics.java` | MAE, RMSE, R², confusion matrix, precision/recall | Report card |
| `Normalizer.java` | Same-scale columns, with save/load | Fair comparison between big and small numbers |
| `ModelIO.java` | Save/load whole models (text or compact binary) | Don't retrain every time |
| `GradCheck.java` | Checks backprop against brute force | Proof that the maths is right |
| `RealWorldDemo.java` | Delivery time predictor | Real-life example |
| `MnistDemo.java`, `QuantReport.java` | Digit recognizer (plain or convolutional, `aug`, `mmap`) and the shrink table | Real-data example |
| `CharTransformerDemo.java` | Tiny GPT-style text generator | Sequence-model example |
| `Benchmark.java` | Speed, memory and int8 tests | Numbers, not feelings |
| `Main.java` | Runs demos 1-4 | Start here |
| `Tests.java` | 171 tests, no JUnit needed | Safety net |

Also: `run.sh` (build, test in both kernel modes, demo), `get-mnist.sh`, `.github/workflows/ci.yml` (runs the tests on every push once the project is on GitHub), `LICENSE` (MIT).

How data flows:

```
 input ──► Dense ──► BatchNorm ──► ReLU ──► Dense ──► output ──► Loss
  x                                                  prediction    │
                                                                   │ how wrong?
 input ◄── Dense ◄── BatchNorm ◄── ReLU ◄── Dense ◄── gradient ◄──┘
        (backward: each layer fixes its own weights and passes blame on)
```

---

## 5. Step by step, with real examples

### Step 1: Tensor (`Tensor.java`)

A tensor here is a table of numbers with `rows` and `cols`, stored as 32-bit numbers (`float[]`) in one flat array.

Real example: 3 delivery orders. Each row is one order, columns are `[distance km, traffic 0-10, items]`:

```
 2.0   3   1
 6.2   8   3
12.0   9   5      ← a 3 x 3 Tensor
```

Main operation: **matrix multiply** (`matmul`). Check this with a pencil:

```
[1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
[3 4] . [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
```

This exact case is in `Tests.java`. Why matmul matters: it lets us process a whole **batch** of orders in one go instead of one by one.

Rule: `(a x b) . (b x c)` gives `(a x c)`. The middle numbers must match, or you get a clear "shape mismatch" error.

### Step 2: Dense layer (`Dense.java`)

A Dense layer has many **neurons**. Each neuron looks at all inputs, multiplies each by a weight, adds them up, adds a bias. In maths: `out = x . W + b`.

Real example. One order: distance = 2, traffic = 3. Two neurons:
- Neuron A: weights `(1, -1)`, bias `0.5`
- Neuron B: weights `(0.5, 2)`, bias `0`

```
A = 2*1 + 3*(-1) + 0.5 = -0.5
B = 2*0.5 + 3*2  + 0   =  7
```

Output: `[-0.5, 7]`. Neuron B "fired strongly" for this order.

Weights start random, with a smart size (He initialization: `std = sqrt(2 / inputs)`) so signals don't explode or vanish.

### Step 3: Activation functions (`Activations.java`)

Problem: if you stack Dense layers with nothing between them, the whole thing collapses into **one** straight-line formula. Useless for curved real-world patterns.

Fix: put a small non-linear function between layers.

| Activation | Rule | Use |
|-----------|------|-----|
| `ReLU` | negative → 0, positive stays | Default for hidden layers |
| `LeakyReLU` | negative → tiny slope instead of 0 | When neurons "die" with ReLU |
| `Tanh` | squashes to -1 … 1 | Small networks, XOR |
| `Sigmoid` | squashes to 0 … 1 | Output that means "probability" |

Real example, continuing Step 2: ReLU on `[-0.5, 7]` gives `[0, 7]`. Neuron A said "not relevant" and gets switched off. Neuron B passes through.

XOR (Demo 2) is the classic proof: a straight line **cannot** solve XOR, but one hidden layer with Tanh can.

### Step 4: Loss (`Losses.java`)

The loss is one number: **how wrong is the model**. Lower is better. Four are included. The first two are below, the others are in Step 12.

**MSE (Mean Squared Error)**: for predicting numbers (delivery minutes, prices).
Predicted 40 min, real 50 min: error `-10`, squared `100`. Squaring makes big mistakes hurt much more than small ones, and removes the sign.

**SoftmaxCrossEntropy**: for picking one class out of many (cat / dog / bird).
The last layer gives raw scores, e.g. `[2.0, 1.0, 0.1]`. Softmax turns them into probabilities that add up to 1:

```
exp(2.0)=7.389  exp(1.0)=2.718  exp(0.1)=1.105   sum = 11.212
probabilities:  cat 0.659   dog 0.242   bird 0.099
```

- If the photo really is a **cat**: loss = `-ln(0.659) = 0.417` → small, good.
- If the photo really is a **bird**: loss = `-ln(0.099) = 2.32` → big, bad.
- Gradient is beautifully simple: `probabilities - truth`. For cat: `[-0.341, 0.242, 0.099]`.

(Code subtracts the max score before `exp` so huge numbers never overflow. There is a test for it.)

### Step 5: Backpropagation (`backward()` in every layer)

Backprop answers: *"If I nudge this weight a tiny bit, how much does the loss change?"* That answer is the **gradient**. It uses the **chain rule**: blame flows backward, layer by layer.

Real example with real numbers. Tiny model: `h = relu(w1 * x)`, `pred = w2 * h`. Input `x = 2`, `w1 = 3`, `w2 = 0.5`, true answer `4`.

Forward:
```
h = relu(3 * 2) = 6
pred = 0.5 * 6 = 3
loss = (3 - 4)^2 = 1
```

Backward (go from the end to the start):
```
dLoss/dpred = 2 * (3 - 4)            = -2
dLoss/dw2   = dLoss/dpred * h        = -2 * 6   = -12
dLoss/dh    = dLoss/dpred * w2       = -2 * 0.5 = -1
ReLU passes the gradient (6 > 0)     = -1
dLoss/dw1   = -1 * x                 = -2
```

Update with learning rate `0.01`: `w2 = 0.62`, `w1 = 3.02`.
Forward again: `h = 6.04`, `pred = 3.7448`, `loss = 0.0651`.
**Loss fell from 1 to 0.065 in one step.**

In the code, `Dense.backward` is just 3 lines of this idea in matrix form:

```java
W.grad = input^T . g      // how each weight contributed
b.grad = sum of g         // how each bias contributed
return g . W^T            // pass blame to the previous layer
```

**How do we know it's correct?** `GradCheck.java` does the slow brute-force way: nudge a weight up by a small amount, nudge it down, measure the loss difference, and compare with our backprop gradient. Max error in this project: about `1e-3` in 32-bit floats (the earlier 64-bit version of the library measured `1e-8` with the same formulas). A real bug shows up as an error of `0.1` or more, so the check still catches mistakes. That is the proof.

### Step 6: Optimizers (`Optimizers.java`)

Gradients say *which way* to go. The optimizer decides *how far*.

**SGD**: `weight = weight - learning_rate * gradient`. Simple and reliable.

**SGD + momentum**: like a bicycle rolling downhill. It keeps some speed from previous steps, so it doesn't get stuck in small bumps and zig-zag less. (`momentum = 0.9` is common.)

**Adam**: gives **each weight its own step size**. Weights with big, noisy gradients take small careful steps; weights with small, steady gradients take bigger ones. Usually trains faster and needs less tuning. Real example: in the delivery model, "distance" affects time a lot and "items" only a little. Adam handles both sensibly without you choosing separate learning rates.

Rule of thumb: Adam with `0.001`–`0.01` is a safe first try.

### Step 7: The training loop (`Sequential.fit`)

Vocabulary with a real example: **400 delivery orders**, batch size **32**.

- **Batch**: a small group of rows used for one update. 32 orders at a time.
- **Epoch**: one full pass over all data. 400 / 32 = 12.5 → **13 batches** per epoch.
- Before each epoch the rows are **shuffled**, so batches are different every time.
- 200 epochs = 200 × 13 = **2600 weight updates**.

Why batches and not all rows at once? It is faster per update and the small randomness helps the model avoid bad spots.

### Step 8: Normalizer (`Normalizer.java`)

Raw columns have different sizes: distance `1–15`, traffic `0–10`, a salary column might be `50000`. Big numbers dominate and training gets unstable (loss shoots to `NaN`).

The Normalizer rescales each column to **mean 0, spread 1**: `(value - mean) / std`.

Important rules (the demo follows them):
1. `fit` on **training data only**, never on test data (otherwise you leak information).
2. Normalize the target (`y`) too for regression, then call `inverse()` to turn the model's answer back into **minutes**.

### Step 9: Dropout and overfitting (`Dropout.java`)

**Overfitting** = the model memorizes the training data instead of learning the pattern. Like a student who memorizes last year's question paper: perfect on it, lost on a new paper. Sign: training loss keeps falling but test error goes up.

**Dropout** fights this. While training, each neuron is randomly switched off with probability `p`. Like a group project where random members stay home each day: everyone has to learn to do the work, nobody is a single point of failure.

- Training: random neurons off, survivors scaled by `1/(1-p)` so the average stays the same.
- Predicting: Dropout does nothing. `model.predict()` switches this off automatically.

```java
new Dense(64, 64, rnd), new Activations.ReLU(), new Dropout(0.2, rnd)
```

### Step 10: Save and load (`ModelIO.java`, `Normalizer.java`)

Training takes time. Do it once, save, and reload in another program.

```java
ModelIO.save(model, "delivery-model.txt");   // architecture + weights + BatchNorm averages (text)
ModelIO.saveBinary(model, "model.bin");       // same thing, compact: 4 bytes per weight
xn.save("delivery-x.txt");                   // the Normalizers too (a model is useless without them)
yn.save("delivery-y.txt");

// later, in a different program, no layers rebuilt by hand:
Sequential loaded = ModelIO.loadModel("delivery-model.txt", new Random());
Normalizer lx = Normalizer.load("delivery-x.txt"), ly = Normalizer.load("delivery-y.txt");
double minutes = ly.inverse(loaded.predict(lx.transform(order))).get(0, 0);
```

The file is plain text. The first lines look like this, so you can read what the model is:

```
MINIDL 2
layers 5
Dense 3 16
ReLU
Dense 16 8
...
```

If you already built the model yourself, `ModelIO.load(model, path)` fills in the weights and gives a clear error if the architecture is different. (`loadModel` knows the built-in layers. For layers you wrote yourself, build the model and use `load`.)

### Step 11: Batch normalization (`BatchNorm.java`)

Problem: as numbers flow through many layers they can drift to huge or tiny values, and training becomes slow and shaky.

Fix: after a layer, re-center and re-scale each feature **using the current batch**, then let the model adjust it with two learned numbers (`gamma` = scale, `beta` = shift).

Real example, one feature across a batch of 4 orders: `[10, 20, 30, 40]`.

```
mean = 25     spread (std) = sqrt(((-15)^2 + (-5)^2 + 5^2 + 15^2) / 4) = 11.18
normalized   = [-1.342, -0.447, 0.447, 1.342]
with gamma = 2, beta = 1:   [-1.683, 0.106, 1.894, 3.683]
```

It is like grading on a curve, but the curve is recomputed for every batch.

- **Training:** uses the batch's own mean and spread, and keeps running averages on the side.
- **Predicting:** uses those running averages (so one single order can be predicted without a batch). `model.predict()` switches this automatically.
- The running averages are saved with the model.
- Small gotcha: the bias of the `Dense` layer right before a `BatchNorm` has gradient exactly 0 (BatchNorm subtracts the mean anyway). Harmless, and the reason the gradient checker uses an absolute tolerance for tiny gradients.

### Step 12: More activations and losses

**GELU**: a smooth version of ReLU, used inside transformers (the models behind modern chatbots).

**MAE (mean absolute error)** versus MSE, with two misses of 10 min and 2 min:

```
MSE = (10^2 + 2^2) / 2 = 52      the 10-minute miss dominates
MAE = (10 + 2) / 2     =  6      every minute counts the same
```

Use MAE when a few crazy outliers should not control the training.

**BCEWithLogits**: for yes/no questions like spam / not spam. The model outputs one raw score (no Sigmoid layer before it). Score `2.0` means probability `1 / (1 + e^-2) = 0.881`.

- Truly spam: loss `-ln(0.881) = 0.127` (small, good)
- Truly not spam: loss `-ln(1 - 0.881) = 2.13` (big, bad)

It is written in a numerically safe form: a score of 1000 does not overflow.

### Step 13: Better optimizers, weight decay, gradient clipping

**Weight decay (L2)**: gently pulls weights toward 0 every step so none grows huge. Test: weight `1.0`, decay `0.5`, learning rate `0.1`, zero gradient gives `1.0 - 0.1 * 0.5 * 1.0 = 0.95`. Biases and BatchNorm numbers are not decayed.

```java
new Optimizers.SGD(0.05, 0.9, 1e-4)   // lr, momentum, weight decay
new Optimizers.Adam(0.01, 1e-4)       // with decay = AdamW
```

**AdamW** applies the decay directly to the weight (cleaner than mixing it into the gradient). **RMSProp** divides each step by a running average of recent gradient sizes.

**Gradient clipping**: if the total gradient is too big, shrink it but keep its direction. Gradient `[3, 4]` has size 5. Clip to 1 and it becomes `[0.6, 0.8]`. Turn on with `new Trainer().clipNorm(1.0)`.

### Step 14: Learning rate schedules (`Schedule.java`)

Like driving: fast on the highway, slow when parking. Big steps early to make progress, small steps late to settle.

```java
Schedule.stepDecay(0.1, 10, 0.5)    // 0.1, then 0.05 after 10 epochs, 0.025 after 20, ...
Schedule.cosine(0.01, 0.0005)       // smooth curve from 0.01 down to 0.0005
```

The `Trainer` sets the rate at the start of every epoch.

### Step 15: Validation, early stopping, the Trainer (`Trainer.java`)

Three groups of data, like school:

- **Training set**: the practice questions. The model learns from these.
- **Validation set**: a mock exam. Never learned from; used to decide when to stop.
- **Test set**: the real exam. Touched once, at the very end.

```java
Trainer.History h = new Trainer()
    .epochs(300).batchSize(32)
    .validation(valX, valY)
    .patience(30)                              // stop if no improvement for 30 epochs
    .schedule(Schedule.cosine(0.01, 0.0005))
    .fit(model, trainX, trainY, new Losses.MSE(), new Optimizers.Adam(0.01, 1e-4));
```

Real run from the delivery demo: best validation score at epoch 20, no improvement for 30 more epochs, so it **stopped at epoch 50** and **put back the weights from epoch 20**. The model that is kept is the best one, not the last one. `History` holds all train/validation losses so you can plot them.

### Step 16: Real data, metrics (`Data.java`, `MnistLoader.java`, `Metrics.java`)

**CSV file** (plain commas, no quotes):

```
km,traffic,items,minutes
6.2,8,3,51
2.0,3,1,22
```

```java
Tensor[] xy = Data.loadCsvXY("orders.csv", true, 1);   // header row, last 1 column = answer
Tensor[] s  = Data.trainTestSplit(xy[0], xy[1], 0.2, 42);  // {xTrain, yTrain, xTest, yTest}
```

For labels written as words (`cat`, `dog`), `Data.loadCsvClassification(path, true)` turns the last column into one-hot labels and gives you the class names. A bad value gives a clear message such as `line 3, column 2: 'oops' is not a number`.

**MNIST**: `MnistLoader.loadImages(...)` gives `(n x 784)` with pixels 0 to 1, `loadLabels(...)` gives one-hot labels. Works with `.gz` files directly.

**Metrics** (what the numbers mean):

- `mae`: "on average I am off by this many minutes"
- `rmse`: same, but punishes big misses more
- `r2`: 1.0 is perfect, 0.0 is no better than always guessing the average
- `confusionMatrix` + `classificationReport`: for classes. From the real MNIST run, 14 real **4**s were read as **9**, the most common mistake. Digit **8** had the lowest recall (0.963): of all real 8s, 96.3% were found.

### Step 17: Convolution (`Conv2D.java`)

A Dense layer looks at every pixel of an image at once and has no idea which pixels are neighbours. A **convolution** looks at small patches (say 3 x 3) and slides over the whole image, so it can learn "an edge", "a curve", "a corner", and find them **anywhere**.

Real example. A 3 x 3 patch of a picture and one 3 x 3 **filter** (the weights):

```
patch         filter (vertical edge finder)
 0  9  9        -1  0  1
 0  9  9        -1  0  1
 0  9  9        -1  0  1

result = (-1*0 + 0*9 + 1*9) * 3 rows = 27      big number = "there IS a vertical edge here"
```

The same filter is reused at every position, so it needs only 9 weights instead of one weight per pixel. That is why the CNN in this project has **27,562 weights** and beats the plain network with **109,386**.

How it is built (the part that matters for speed and memory):

1. **im2col**: copy every patch into one row of a big table. Now the whole convolution is ONE matrix multiply, and we already have a fast one.
2. **Memory trick**: that table repeats every pixel up to 9 times, so it is big. We do not keep it for the whole batch. We keep ONE small buffer (about 24 KB for the first layer) per CPU thread and **rebuild the patches again during backprop** instead of storing them. Real frameworks call this *recomputation* (or activation checkpointing). We pay a little extra compute to save memory, which matters more as images and batches get bigger.
3. Several images are processed in parallel (one chunk per CPU core).

Rows of a tensor are flattened images laid out as `[channel][y][x]`:

```java
new Conv2D(inChannels, height, width, filters, filterSize, stride, padding, rnd)
new Conv2D(1, 28, 28, 8, 3, 1, 0, rnd)   // 28x28 grey image -> 8 feature maps of 26x26
```

**Is it correct?** Two proofs: its output is compared against a plain 6-loop convolution for 4 different settings (padding, stride, 2x2 and 5x5 filters), and gradient checks pass for stride/padding variants and for two stacked convolutions.

### Step 18: Max pooling (`MaxPool2D.java`)

Look at each 2 x 2 square and keep only the biggest number. It halves the image size (so the next layers do 4x less work) and keeps the strongest signals.

```
1 2 | 5 3
4 9 | 1 1        ->    9  5
----+----              8  7
0 0 | 7 2
8 1 | 3 6
```

Memory trick: to send blame backward we only need to know **which** of the 4 numbers won. That position (0 to 3) fits in **one byte**, so we store one byte per window instead of copying values.

Note: max-pool is not a smooth function. If two numbers in a window are almost tied, a gradient check can be fooled. That is a property of max-pool, not a bug. (One of 12 random seeds shows it in the tests.)

### Step 19: Big data in small memory (`ByteSource.java`)

Images are numbers from 0 to 255, so they fit in **one byte**. Storing them as floats wastes 4x, as doubles 8x.

| 60,000 MNIST images | Memory |
|---|---|
| as double (earlier version) | 376 MB |
| as float32 (measured) | 188 MB |
| as bytes with `ByteSource` (measured) | **47 MB** |

`ByteSource` keeps the bytes and turns **only the current batch** into floats. Splitting data into train and validation uses `view(from, to)`, which shares the same bytes (no copy):

```java
ByteSource all   = MnistLoader.loadSource(imagesFile, labelsFile, 60000);
ByteSource train = all.view(0, 54000);
ByteSource val   = all.view(54000, 60000);
new Trainer().validation(val).fit(model, train, loss, opt);   // Trainer works on any DataSource
```

`DataSource` is a tiny interface (`size`, `features`, `targets`), so you can write your own, for example one that reads batches from a file or database. `model.accuracy(source)` and `model.evalLoss(source, loss)` also work chunk by chunk.

### Step 20: Float32 and int8 quantization (`QuantizedDense.java`, `Quantizer.java`)

**Float32 instead of double.** Every weight, gradient, optimizer number and data value takes 4 bytes instead of 8, and the CPU can process twice as many floats per instruction. Cost: about 7 digits of precision instead of 15, which is plenty for learning (the demos give the same results).

**int8 quantization** shrinks a trained model for prediction. Each weight becomes a whole number from -127 to 127 (one byte) plus one scale number per output column.

Real example, one column of weights `[0.50, -0.30, 0.10]`:

```
biggest |weight| = 0.50     scale = 0.50 / 127 = 0.003937
stored bytes:  0.50 / 0.003937 = 127,   -0.30 / 0.003937 = -76,   0.10 / 0.003937 = 25
read back:     127 * 0.003937 = 0.500,  -76 * 0.003937 = -0.299,  25 * 0.003937 = 0.098
```

Tiny rounding errors, 4x less memory. Usage:

```java
Sequential small = Quantizer.quantize(trainedModel);   // Dense layers become int8
double acc = small.accuracy(testData);
ModelIO.saveBinary(small, "model-int8.bin");
```

It is **weight-only**: the numbers flowing through the network stay float. It is **prediction only**: training an int8 model throws a clear error. (Conv2D can be quantized too: see Step 25.)

### Step 21: Data augmentation (`AugmentedSource.java`)

A model memorizes if it sees the same 54,000 pictures again and again. **Augmentation** shows it a slightly different picture every time: shifted by up to 2 pixels and rotated by up to 10 degrees. A "7" that is a bit crooked is still a "7", so the label stays correct, but the model can no longer memorize pixels. It has to learn the shape.

Real example: a 5 x 5 image with one bright pixel at row 2, column 2. Shift it 1 pixel to the right and the pixel is at row 2, column 3 (there is a test for exactly this). Rotate a vertical bar by 90 degrees and you get a horizontal bar (also tested).

How it works: for every NEW pixel it finds where that pixel came from in the OLD image, and blends the 4 nearest old pixels (bilinear interpolation). Outside the picture counts as black.

```java
DataSource trainSrc = new AugmentedSource(train, 28, 28, 2, 10, 7);   // image size, max shift px, max degrees, seed
new Trainer().validation(val).fit(model, trainSrc, loss, opt);         // validation and test stay UNCHANGED
```

Memory cost: almost nothing. The original bytes are untouched; only the current batch is modified. Result on real MNIST with the same small CNN: **98.91% without it (8 epochs), 99.19% with it (16 epochs)**. The epoch counts differ, so the comparison is not perfectly clean.

### Step 22: LayerNorm, skip connections, average pooling

**LayerNorm** normalizes every ROW on its own, so it does not care about batch size or about training versus predicting (BatchNorm does). Row `[2, 4, 6, 8]`: mean 5, spread 2.236, result `[-1.342, -0.447, 0.447, 1.342]`.

**Residual** (skip connection): `output = input + f(input)`. Real example: input `x = 2`, `f(x) = 0.5 * x = 1`, so the output is `3`. Send a blame of `1` backward: the shortcut passes `1` straight on, and `f` adds its own share `0.5`, so the input gets `1.5`. Because of the shortcut, the blame never has to squeeze through every layer, and that is what lets networks with dozens of layers train.

```java
new Residual(new Dense(64, 64, rnd), new Activations.ReLU(), new Dense(64, 64, rnd))   // must keep the same width
```

**AvgPool2D** replaces each square by its average (max pooling keeps the biggest). The 4 x 4 test image pools to `[4, 2.5, 2.25, 4.5]`.

### Step 23: Attention and a tiny transformer (`Embedding`, `CausalSelfAttention`, `TokenWise`, `Reshape`)

A transformer is the architecture behind modern chatbots. In this project it is built from the same pieces as everything else.

**Tokens and embeddings.** Text becomes a list of token ids (here, one id per character). The `Embedding` layer gives each id its own learned vector and each position its own learned vector, and adds them: `vector = token_vector + position_vector`. The position part is how the model knows the order.

**Attention.** Every token asks a question (a *query*), and every token also has a label (*key*) and some content (*value*). A token compares its query with the keys of the tokens it is allowed to see, turns the scores into percentages, and takes that mix of their values. Real numbers: scores `[2.0, 1.0, 0.1]` become percentages `[65.9%, 24.2%, 9.9%]`, so the token takes 66% of the first token's value, 24% of the second's and 10% of the third's.

**Causal** means a token may only look at itself and EARLIER tokens. Token 3 sees tokens 0, 1, 2, 3 and never token 4. Without that rule, a model asked to predict the next letter could just read it. A test changes ONLY the last character of an input and checks that every earlier prediction stays exactly the same (difference below 1e-6) while the last one changes.

**Heads.** 4 heads do this side by side, each on its own quarter of the numbers, so different heads can look for different things (the previous letter, the start of the word, ...).

**A block** is two residual parts: attention, then a small per-token network. `TokenWise` runs ordinary layers on every token separately by looking at the same numbers as (N*T x D) instead of (N x T*D), with no copying.

```java
Sequential m = new Sequential().add(new Embedding(V, D, T, rnd));
for (int i = 0; i < 2; i++) {
    m.add(new Residual(new TokenWise(T, D, new LayerNorm(D)), new CausalSelfAttention(T, D, 4, rnd)));
    m.add(new Residual(new TokenWise(T, D, new LayerNorm(D), new Dense(D, 4 * D, rnd), new Activations.GELU(), new Dense(4 * D, D, rnd))));
}
m.add(new TokenWise(T, D, new LayerNorm(D), new Dense(D, V, rnd))).add(new Reshape(V));   // one row of scores per character
```

**Memory trick:** the T x T table of attention percentages is never stored. Backward rebuilds it from the stored queries and keys (the idea FlashAttention is built on, in a simple form).

**Training data** is `CharSource`: a window of T characters is the input, and at every position the target is the NEXT character. One window teaches T predictions at once.

Full results are in section 8 (Demo 6).

### Step 24: Autograd (`Var.java`)

Until now every layer had a hand-written `backward()`. Autograd removes that work: every operation remembers its inputs and how to pass blame back, so you write only the FORWARD maths and call `backward()` on the loss.

```java
Var w = new Var(weights), b = new Var(bias);
Var loss = new Var(x).matmul(w).addRow(b).tanh().mseLoss(target);
loss.backward();                    // w.grad and b.grad are filled in
```

How it works: each result keeps its parents plus a small function that adds its share of the gradient to them. `backward()` sorts the whole expression so that every value is processed after everything that depends on it (a topological sort), then runs those functions in reverse. If a value is used twice, both uses add to its gradient. Real example: `y = x * x` at `x = 3` gives `dy/dx = 3 + 3 = 6` (tested).

Operations available: `matmul`, `add`, `sub`, `mul`, `addRow`, `scale`, `relu`, `tanh`, `sigmoid`, `mean`, `mseLoss` (built from other operations, so it needs no backward of its own) and `softmaxCrossEntropy`.

**Proof it is right:** for a Dense layer with MSE, and for a 2-layer tanh network with softmax cross entropy, `Var` gives the same gradients as the hand-written layers (difference below 1e-5). It also trains XOR using only forward maths (final loss 0.0009 in the test).

Autograd is the easy way to build and check new ideas. Step 31 turns it into layers you can train with the normal `Trainer`. The built-in layers (`Conv2D`, attention, ...) keep their hand-written backward, because that is where the speed and memory tricks live.

### Step 25: More ways to shrink a model (`Quantizer`)

Three more options next to int8 (Step 20), all for prediction only:

| Option | Storage per weight | Example |
|---|---|---|
| **bfloat16** | 16 bits (the top half of a float32) | `3.14159265` is stored as `3.140625`, an error of 0.03% |
| **int8** | 8 bits + one scale per column or filter | `Quantizer.quantize(model)`: Dense **and Conv2D** |
| **int4, group-wise** | 4 bits (two weights per byte) + one scale per group of 32 rows | `Quantizer.quantizeInt4(model, 32)` |

Real example for int4, one group of weights `[0.70, -0.33, 0.12]`: biggest is 0.70, so scale = `0.70 / 7 = 0.1`. Stored as whole numbers `[7, -3, 1]` (4 bits each, from -7 to 7), read back as `[0.7, -0.3, 0.1]`. Only 16 different values exist, which is coarse, so every group of 32 weights gets its own scale. Result on the real digits model below.

`bfloat16` keeps the same number range as float32 (only the precision is cut), which is why big models can use it without scaling tricks. `Quantizer` also reaches into `Residual` and `TokenWise` blocks, so the Dense layers of a transformer shrink too (its attention weights stay float32).

### Step 26: Data bigger than memory (`MappedSource.java`)

`ByteSource` keeps all the bytes in the Java heap. `MappedSource` keeps them **on disk**: the file is memory-mapped, so the operating system loads the pieces we touch and drops them when it needs room. The Java heap holds almost nothing.

```java
MappedSource all = MappedSource.open("train-images-idx3-ubyte", "train-labels-idx1-ubyte");   // unzipped files
new Trainer().fit(model, all.view(0, 54000), loss, opt);
```

Measured: plain network trained on all 60,000 digits with `java -Xmx64m` (64 MB for everything: model, batches, JVM): 96.92% test accuracy after 2 epochs (5.8 s per epoch). One mapped file can be at most 2 GB (a Java limit), so a bigger dataset would be split into several files.

### Step 27: SIMD and register tiles (the Vector API kernel)

**Why ordinary loops are slow.** A loop that multiplies and adds one number at a time wastes the CPU. This machine's CPU can do 16 float multiply-adds in ONE instruction (AVX-512, called SIMD: single instruction, multiple data). Java's plain loops reached about 2 to 6 GFLOP/s here. The hardware can do many times that, but only if the numbers it needs are already inside the CPU.

**The register tile.** The Vector API (`jdk.incubator.vector`) lets Java use those wide instructions directly. The kernel keeps a small block of the answer, 4 rows x 2 vectors = 4 x 32 numbers, **inside CPU registers** while it walks down the shared dimension:

```
for k in the shared dimension:
    load 2 vectors of B (32 numbers), broadcast 4 numbers of A
    do 8 fused multiply-adds    -> 8 x 16 = 128 multiply-adds, with only 6 loads
```

The answer is read and written once per block, not once per multiply. Memory traffic, not arithmetic, was the limit.

**Cache blocking.** The shared dimension is cut into chunks of 256, so the piece of B being used (256 rows x 32 columns x 4 bytes = 32 KB) stays in the CPU's fastest cache. Measured honestly in a separate test: blocking a plain scalar loop gave only about 1.1x. Register tiles plus SIMD gave the 7 to 12x. Adding the K-chunks on top of the tile helped the large sizes (in the lab test about 26 to 33 GFLOP/s at 1024 x 1024).

**How the library uses it.** Nothing changes for you:

- `Tensor` looks for the class `minidl.VectorGemm` when it starts. If Java was started with `--add-modules jdk.incubator.vector` and the class is compiled, it is used; otherwise the portable loops run. Both paths are tested.
- The transposed products used by backprop (`A^T . B`, `A . B^T`) lay one operand out transposed first (a small copy next to a big multiply), so they use the same fast kernel.
- Convolution now builds its patch table as "filter position x patch position", so the multiply runs along long straight rows.
- Big multiplies are cut into blocks of 16 rows (so no tile is split) and spread over the cores.
- Switch it off with `-Dminidl.vector=false`, or in code with `Tensor.setVector(false)`. `Tensor.kernelName()` says which one is active.

**Measured** (one core, AVX-512, 16 lanes):

| Test | Portable loops | Vector API | Faster |
|---|---|---|---|
| Multiply 256 x 256 | 2.2 GFLOP/s | 61.2 GFLOP/s | 27.6x |
| Multiply 512 x 512 | 4.5 GFLOP/s | 30.0 GFLOP/s | 6.7x |
| Multiply 1024 x 1024 | 6.3 GFLOP/s | 56.0 GFLOP/s | 8.9x |
| Plain network epoch (6,400 digits) | 516 ms | 124 ms | 4.2x |
| CNN (640 images) | 250 ms | 125 ms | 2.0x |
| Transformer (10 steps of 32 sequences) | 1922 ms | 881 ms | 2.2x |
| MNIST CNN with augmentation, per epoch | 25.7 s | **12.3 s** | 2.1x |

The whole-network gains are smaller than the multiply gains because the rest is still plain Java: building patches, pooling, activations, attention loops. The MNIST CNN result stayed 99.17% (it was 99.19%; the difference is float rounding order).

**Honest caveats.** The Vector API is an *incubator* module in Java: it needs the extra flag and its interface may change. It was built and tested here on Java 21; the code uses calls that also exist in Java 17, but that was not tested. A different CPU has a different number of lanes (8 on AVX2), so speeds differ. Repeated runs vary a lot (the 512 multiply measured between 30 and 75 GFLOP/s in different runs), so read the table as orders of magnitude.

### Step 28: 16-bit training with loss scaling (`Half`, `Precision`, `LossScaler`)

Big AI models train in 16-bit numbers: half the memory, and chips that are much faster at them. Two 16-bit formats exist:

| Format | Bits (sign / exponent / fraction) | Range | Precision |
|---|---|---|---|
| FP32 (normal) | 1 / 8 / 23 | huge | about 7 digits |
| **FP16** | 1 / 5 / 10 | 6e-8 up to 65,504 | about 3 digits |
| **BF16** (bfloat16) | 1 / 8 / 7 | same as FP32 | about 2 digits |

**The problem with FP16.** Real examples (all tested):

- `1e-8` becomes **0** in FP16. Gradients are often this small, so the weights silently stop learning (*underflow*).
- `70000` becomes **infinity** in FP16 (*overflow*).
- BF16 has FP32's range, so neither happens, but it keeps fewer digits.

**Loss scaling.** Multiply the loss by a big number S (say 65,536) before backprop, so every gradient is S times bigger and survives. Divide by S again in 32-bit before updating. Real example: `1e-8 x 65536 = 6.6e-4`, which FP16 stores fine; after dividing by 65536 you get back `1.0e-8`. *Dynamic* scaling picks S by itself: halve it when something overflows, double it after 200 good steps.

**What one 16-bit training step does** (`model.setPrecision(Precision.FP16)`):

1. Keep a full 32-bit **master copy** of the weights. Compute with 16-bit rounded weights.
2. Forward: every layer's output is rounded to 16 bits.
3. Start backward from the **scaled** loss. Every gradient passing backward is rounded to 16 bits.
4. Store the weight gradients in 16 bits, then divide by S in 32-bit.
5. If any gradient is infinity or NaN: **skip the step** and halve S. Otherwise the optimizer updates the 32-bit master weights.

```java
model.setPrecision(Precision.FP16);                       // dynamic loss scaling is on by default
model.setPrecision(Precision.BF16);                       // no scaling needed
model.setPrecision(Precision.FP16, LossScaler.fixed(1024));
```

**What is real and what is simulated.** Java has no 16-bit arithmetic, so the multiply-adds still run in float32. The numbers are rounded to exactly what a 16-bit chip would store, so overflow and underflow happen as they would on real hardware, but there is **no speed gain** (the rounding makes steps 1.7 to 3x slower here). The **memory saving is real**: `Dense` and `Conv2D` keep their input for backward in 16 bits (2 bytes), and `ReLU` / `LeakyReLU` keep **one bit** per number (they only need to know which inputs were positive) instead of the whole input.

**Measured: activation memory held between forward and backward** (heap, measured):

| Model | Previous version | Now, float32 | Now, 16-bit storage |
|---|---|---|---|
| Plain network 784-512-512-512-10, batch 2048 | 25.28 MB | 13.07 MB | **9.99 MB** (2.5x less) |
| CNN, batch 256 | 25.64 MB | 6.97 MB | **4.96 MB** (5.2x less) |

The ReLU bit masks alone halve the plain network and quarter the CNN; the 16-bit storage adds on top. (Attention, LayerNorm and BatchNorm still keep 32-bit values.)

**Measured: does loss scaling matter?** A network trained 3 epochs on MNIST digits, gradients of one batch, compared with the FP32 gradients (`MixedPrecisionDemo`):

| Batch 64 | Non-zero gradients lost to zero | Relative error |
|---|---|---|
| BF16, no scaling | 0.0% | 0.0077 |
| **FP16, no scaling** | **3.0%** | 0.0021 |
| FP16, scale 1024 | 0.1% | 0.0021 |
| FP16, scale 65,536 | 0.0% | 0.0021 |
| FP16, scale 16,777,216 | overflow, step skipped | - |

At batch 1024 the loss without scaling is **4.5%** (0.2% with scale 1024, 0.0% with 65,536). So scaling does rescue gradients, and a scale that is too big is caught by the overflow check.

**Measured: training from scratch, 2 epochs, test accuracy on 10,000 digits:**

| Setting | Test accuracy | s / epoch | Skipped steps | Final scale |
|---|---|---|---|---|
| FP32 (normal) | 96.47% | 1.4 | 0 | - |
| BF16 | 96.60% | 2.4 | 0 | - |
| FP16, no loss scaling | 96.54% | 4.4 | 0 | 1 |
| FP16, dynamic loss scaling | 96.60% | 4.1 | 5 | 262,144 |

Honest reading: on this small network all four reach the same accuracy. Losing 3% of the (mostly tiny) gradient entries did not hurt, and Adam rescales gradients anyway. Loss scaling is a **safety net** that matters more for deeper networks and smaller gradients. The dynamic scaler started at 65,536, hit overflow 5 times early on (steps skipped, scale halved), then grew to 262,144. The 0.1% differences between rows are within the run-to-run noise of one seed.

### Step 29: Dilated and grouped convolution

Two settings of `Conv2D`:

```java
new Conv2D(inC, h, w, outC, k, stride, pad, dilation, groups, rnd)
```

**Dilation.** The filter looks at pixels that are `dilation` apart. A 3 x 3 filter with dilation 2 covers a 5 x 5 area with the same 9 weights, a wider view for free. Output size: `(size + 2*pad - (dilation*(k-1)+1)) / stride + 1`. Example: an 8 x 8 image, 3 x 3 filter, dilation 2, no padding: the filter reaches 5, so the output is 4 x 4.

**Groups.** The channels are split into `groups` independent sets; each set of filters only looks at its own set of channels. Weights shrink by the number of groups. Real counts from the tests, 8 channels in and out, 3 x 3 filters:

| Setting | Weights |
|---|---|
| plain (groups 1) | 576 |
| groups = 2 | 288 |
| **depthwise** (groups = channels = 8) | **72** |

A **depthwise-separable** block is a depthwise 3 x 3 (groups = channels) followed by a 1 x 1 "pointwise" convolution to mix channels. For 32 channels in and 64 out: a normal 3 x 3 convolution has 18,432 weights, the separable pair has 288 + 2,048 = **2,336** (7.9x fewer). It is the building block of mobile networks.

All of it is checked against a plain 7-loop convolution (5 settings: dilated, grouped, depthwise, strided with both, 1x1 grouped) and with gradient checks. Saving and loading works. `Quantizer` leaves these convolutions in float32 (only plain ones are quantized). Not optimized: each group is its own small matrix multiply, so a depthwise convolution has few weights but is not necessarily fast here (speed not measured).

### Step 30: N-dimensional tensors and shape checking (`NDTensor`)

Until now everything was a 2D table (one flattened sample per row). `NDTensor` is an N-dimensional array: images `[N, C, H, W]`, token batches `[N, T, D]`.

- **Views, not copies.** `reshape(...)` and `toMatrix(...)` share the same numbers. `tensor.nd(4, 1, 8, 8)` views a 2D `Tensor` as 4D without copying. `model.predict(ndTensor)` takes a `[N, 1, 28, 28]` batch straight in.
- **Broadcasting.** Shapes are lined up from the right and a dimension of size 1 is stretched: `[3,1] + [1,4]` gives `[3,4]` (element (2,3) is 3 + 40 = 43). Add a `[4]` bias to a `[2,3,4]` tensor, or multiply `[2,1,4] * [1,3,1]`. Incompatible shapes (`[3] + [4]`) give a clear error.
- **Rearranging:** `permute(2,0,1)`, `transpose(a,b)`, `slice(axis, from, to)`, `concat(axis, ...)`.
- **Reductions:** `sum`, `mean`, `max` along any axis, and `softmax(axis)`.
- **Batched matmul:** `[..., M, K] . [..., K, N]`, either with one shared weight matrix or one matrix per batch entry (like attention). It uses the fast kernels.

The training layers still work on 2D tables internally, because that is where the fast kernels and memory tricks live. `NDTensor` is the friendly front door, and **shape inference** checks that a model fits together before any data runs. `model.summary(3, 32, 32)` prints the shape after every layer (real output from the CIFAR model):

```
#    Layer                          Output shape          Weights
     input                          [3, 32, 32]
1    Conv2D 3 32 32 16 3 1 1        [16, 32, 32]              448
3    MaxPool2D 16 32 32 2           [16, 16, 16]                0
4    Conv2D 16 16 16 32 3 1 1       [32, 16, 16]             4640
9    MaxPool2D 64 8 8 2             [64, 4, 4]                  0
10   Dense 1024 128                 [128]                  131200
13   Dense 128 10                   [10]                     1290
     TOTAL                                                 156074
```

A wrong size is caught with the layer number and both sizes, instead of a failure in the middle of training:

```
layer 3 (Dense 1000 10): Dense expects 1000 numbers per sample but gets [8, 13, 13] = 1352
```

Works for CNNs, transformers (`Embedding`, attention, `TokenWise`, `Residual`) and quantized models. A `Residual` block that changes the size is caught too.

### Step 31: Layers written with autograd (`AutoLayer`, `AutoLayers`)

Step 24 made autograd. Now layers use it. An `AutoLayer` is defined only by its forward maths. There is no hand-written `backward()`:

```java
Layer square = new AutoLayer("Square", List.of(), (x, params) -> x.mul(x));     // a new layer in one line
```

It plugs into `Sequential`, the `Trainer`, every optimizer, saving and loading. `AutoLayers` has four ready-made ones, each a few lines of forward maths:

| Layer | Written as | Hand-written twin |
|---|---|---|
| `AutoLayers.dense(in, out, rnd)` | `x.matmul(W).addRow(b)` | `Dense` |
| `AutoLayers.layerNorm(F)` | subtract the row mean, divide by the row spread, scale and shift | `LayerNorm` |
| `AutoLayers.gelu()` | built from `tanh`, no GELU primitive | `Activations.GELU` |
| `AutoLayers.attention(T, D, H, rnd)` | for every sequence and head: `softmax(Q . K^T / sqrt(dh) + mask) . V` | `CausalSelfAttention` |

To make that possible `Var` grew from 12 to 33 operations: `exp`, `log`, `sqrt`, `inv`, `div`, `rowSum`, `rowMean`, `rowSoftmax`, `addCol` / `subCol` / `mulCol` / `divCol`, `mulRow`, `transpose`, `sliceCols` / `sliceRows`, `concatCols` / `concatRows`, `reshape`, and `backward(seed)` for results that are not a single number. Every operation is checked against a nudge-and-measure gradient (worst error 1.1e-3 in the test).

**Proof they are the real thing.** Each autograd layer is run next to its hand-written twin with the same weights:

- `AutoDense`, `AutoLayerNorm`, `AutoGELU`, `AutoAttention` give the same output, the same weight gradients and the same input gradient (attention output differs by 2e-7).
- A **whole transformer built only from autograd layers** (embedding, attention, LayerNorm, MLP, residuals) gives the same loss (4 decimals) and the same gradients for all weights as the hand-written transformer; the worst gradient differs by 8e-7 of the gradient size.
- A model made only of autograd layers trains XOR with the normal `Trainer` and Adam. A model mixing hand-written and autograd layers passes the gradient check.

**What it costs** (measured, one core):

| | Hand-written | Autograd | Cost |
|---|---|---|---|
| Plain network, 3,200 samples | 38 ms | 77 ms | 2.0x slower |
| Transformer step, 8 sequences | 24 ms | 54 ms | 2.25x slower |
| Memory kept between forward and backward (batch 1024, 784-512-512-10) | 4.37 MB | 12.67 MB | 2.9x more |

Autograd keeps every intermediate result of the expression until backward, and does not use the fused kernels, the 1-bit masks or the 16-bit storage. That is why the built-in layers stay hand-written. Autograd layers are for inventing and checking new ideas fast. Saving works too: `AutoLayers` registers itself with `ModelIO.register(...)`, so `ModelIO.loadModel` rebuilds them. A layer type you invent needs its own `ModelIO.register("MyLayer", (words, rnd) -> ...)`; without it you get a clear error that says so.

### Step 32: CIFAR-10 colour photos (`CifarLoader`, `CifarDemo`)

CIFAR-10: 60,000 colour photos of 32 x 32 pixels in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), 50,000 for training and 10,000 for testing. Harder than digits: colour, clutter, and animals that look alike.

**Layout.** One photo is a row of 3072 numbers: 1024 red, 1024 green, 1024 blue pixels, i.e. `[channel][y][x]`. That is exactly what `Conv2D` expects with 3 input channels. `CifarLoader` reads the official binary format (1 label byte + 3072 bytes per photo) into a `ByteSource`, so the data stays at **1 byte per number**: 184 MB for all 60,000 photos, where doubles would need 1,475 MB. The demo trains inside `-Xmx512m`.

**Where the data came from.** The official download host (cs.toronto.edu) is not reachable from the sandbox where this was built. The npm package `tfjs-cifar10` contains the same dataset as PNG sheets plus label files. `CifarConvert` turns it into the official binary files. Checks done on the converted files: each batch is exactly 30,730,000 bytes (10,000 x 3,073), the label order begins like the official files (test labels 3, 8, 8, 0, 6, 6, 1, 6, 3, 1; training labels 6, 9, 9, 4, 1, 1, 2, 7, 8, 3), and every class has exactly 5,000 training and 1,000 test photos. `./get-cifar.sh` does the download and the conversion for you. If you have the official files, `CifarDemo` reads them directly.

**Colour augmentation.** `new AugmentedSource(train, 3, 32, 32, 3, 0, true, seed)`: random shift up to 3 pixels and a random left-right **flip**. All three colour channels always move together (tested). A flipped cat is still a cat. (Do not flip digits or text.)

**The model:** three convolution + ReLU + max-pool stages (16, 32, 64 filters), then Dense 128, Dropout 0.3, Dense 10. **156,074 weights.**

```
train 45000, validation 5000, test 10000
epoch    1  loss 1.69059  val 1.25739
epoch    4  loss 1.11224  val 0.94815
epoch    8  loss 0.92866  val 0.81765
trained 8 epochs in 489 s (61.1 s/epoch)
TEST accuracy on 10000 unseen photos: 70.64%   (random guessing: 10%)
```

| Class | Recall | Class | Recall |
|---|---|---|---|
| ship | 84.2% | dog | 59.6% |
| automobile | 83.5% | deer | 60.3% |
| frog | 80.0% | bird | 62.2% |
| horse | 78.3% | airplane | 77.4% |
| truck | 75.4% | **cat** | **45.5%** |

The most common mistake: **178 real cats read as dogs**. The ordering makes sense: vehicles and frogs are easy, four-legged animals are confused with each other.

Honest context: 70.6% is a modest result. The network is small, has no batch normalization in the convolution stages, ran only 8 epochs (validation loss was still falling) and used one seed. Bigger CNNs with more training reach well above 90% on CIFAR-10 (general knowledge, not measured here). The point is that the full colour pipeline works end to end: loader, `ByteSource`, augmentation, `Conv2D` with 3 channels, shape checking, training, per-class report.

---

## 6. Efficiency: less memory, more speed

Rule used here: **measure first, then change, then measure again.** Memory and speed are related: moving fewer bytes is usually the fastest way to make a program quicker.

Test machine: **1 CPU core** in a cloud sandbox. Re-run everything with `java -cp out minidl.Benchmark`. All "before" numbers were measured on the same machine in the same session.

### What changed (four rounds)

| Round | Change | Memory | Speed |
|---|---|---|---|
| 1 | Fused transposed multiplies (no temporary copies), in-place bias, gradients written into existing arrays | less garbage | faster |
| 1 | Multiply 4 rows of B per pass; skip the unused first-layer input gradient | | faster |
| 1 | Multi-threaded multiply (one whole output row per thread, identical results) | | scales with cores* |
| 2 | **float32 everywhere** (weights, gradients, Adam state, data) | **half** | **~1.7x** (twice as many numbers per CPU instruction, half the memory traffic) |
| 2 | **Images kept as bytes** (`ByteSource`), one batch turned into floats at a time | data: **8x less than double**, 4x less than float | slightly more work per batch |
| 2 | **Conv2D recomputation**: one small patch buffer per thread, rebuilt in backward | no big batch-sized table | a little extra compute |
| 2 | **MaxPool2D stores 1 byte per window** (which number won) | 4x less than storing values | |
| 2 | **int8 weights** for prediction (`Quantizer`) | model **~4x smaller** | memory only; for speed see the int8 note below |
| 3 | **int4** (group-wise) and **bfloat16** weights, **int8 convolution** | **8x / 2x / 4x** smaller | about the same speed |
| 3 | **Memory-mapped data** (`MappedSource`) | the dataset leaves the Java heap | |
| 3 | **Attention never stores its T x T table** (rebuilt in backward) | memory no longer grows with T squared per head | a little extra compute |
| 3 | **TokenWise / Reshape share arrays** instead of copying | no copies | |
| 4 | **Vector API kernel**: register tiles, SIMD, K-chunks of 256 (Step 27) | | **7 to 28x** faster multiply, 2 to 4x faster training |
| 4 | **Convolution patch table in "filter x position" layout** | | long straight rows for the fast kernel |
| 4 | **1-bit ReLU masks** (a number needs only its sign for backward) | **32x** smaller than keeping the input | |
| 4 | **16-bit storage** of what Dense / Conv2D need for backward (Step 28) | **2x** smaller | little |
| 4 | **Mixed-precision training** with 32-bit master weights, loss scaling (simulated numerics) | memory real, speed not | 1.7 to 3x slower (rounding) |
| 4 | **Shape inference** (Step 30): wrong sizes caught before training | | saves failed runs |
| 2 | **Binary model files** | 4 bytes per weight instead of ~12 characters of text | much faster to load |

*not measured here: the machine has one core.

### Measured: speed (same machine, same session)

| Test | Result |
|---|---|
| Multiply two 512 x 512 matrices: textbook loops | 261 ms (1.0 GFLOP/s) |
| same, optimized, **float32** | **39 ms (6.9 GFLOP/s)**, 6.7x faster than textbook |
| same, previous double version | 69 ms (3.9 GFLOP/s) |
| One training epoch, 6,400 samples, 784 → 128 → 10: first version of the layer (double) | 1359 ms |
| after round 1 (double) | 875 ms |
| after round 2 (**float32**) | **468 ms**, 2.9x faster than the first version |
| Real MNIST, plain network (54,000 digits), per epoch: double | 9.8 s |
| same, **float32** | **6.3 s** |
| Real MNIST, small CNN, per epoch | 25.5 s |

Run-to-run variation is real: the same textbook multiply measured between 180 and 262 ms in different runs (and the optimized float32 one between 37 and 41 ms), and a later run after the sandbox restarted gave 462 to 475 ms per training epoch, in the same range as the 468 ms above. Read the numbers as ratios and orders of magnitude.

### Measured: memory

| Test | Before (double) | Now |
|---|---|---|
| 60,000 MNIST training images in memory | 376 MB | **47 MB** (bytes), 188 MB if stored as float32 |
| Java memory needed to train on MNIST | **ran out of memory at the 1 GB default**, needed `-Xmx2g` | works with **`-Xmx256m`** (full CNN training) and even **`-Xmx128m`** (plain network) |
| MNIST plain network model file | about 2.3 MB (text) | **427 KB** (binary), **108 KB** (int8 binary) |
| MNIST CNN model, weights | | 107.7 KB float32, **31.1 KB** with int8 Dense layers (3.5x) |
| Plain network, weights in memory | | 427 KB float32 → **108 KB** int8 (3.95x) |

### Measured: round 4 (Vector API, 16-bit)

The multiply and epoch tables are in Step 27, the 16-bit tables in Step 28. In short, on this machine:

| | Round 3 | Round 4 |
|---|---|---|
| 512 x 512 multiply | 4.5 GFLOP/s (portable loops) | 30 GFLOP/s (up to 75 in other runs) |
| MNIST CNN training, per epoch | 25.7 s | **12.3 s** |
| Plain network epoch, 6,400 digits | 516 ms | **124 ms** |
| Activation memory kept for backward, CNN batch 256 | 25.64 MB | **4.96 MB** (5.2x less) |
| Activation memory kept for backward, plain network batch 2048 | 25.28 MB | **9.99 MB** (2.5x less) |
| Whole-library tests | 107 | **171** (run with both kernel sets) |

### Measured: every shrink option on the same trained digit CNN (round 3)

Trained with data augmentation (99.19%), then converted. 10,000 unseen test digits.

| Model | Weights | Smaller by | Test accuracy |
|---|---|---|---|
| float32 | 107.7 KB | 1.0x | 99.19% |
| bfloat16 (Dense) | 56.4 KB | 1.9x | **99.19%** |
| int8 (Dense + Conv) | 27.6 KB | 3.9x | **99.17%** |
| **int4**, group 32 (Dense; Conv int8) | **17.8 KB** | **6.0x** | **99.10%** |

Prediction time for all 10,000 digits in the final run (with the Vector API kernels): float32 0.88 s, bfloat16 1.01 s, int8 2.02 s, int4 3.28 s. **The smaller formats are now SLOWER than float32**: their loops are plain Java, while float32 uses the SIMD kernel. Round 3 measured them as about equal because float32 was also plain loops then. Accuracy in that final run was 99.17% (float32, bfloat16, int8) and 99.08% (int4), sizes as in the table. So today these formats are about memory, not speed. A SIMD int8 kernel would change that (see "What is left").

The text model: float32 weights 432 KB; with int8 for its Dense layers 230 KB. Its attention weights stay float32 in this version.

### Measured: what int8 costs in accuracy (round 2)

| Model | float32 | int8 weights |
|---|---|---|
| Plain network on 10,000 unseen MNIST digits | 98.15% | **98.11%** |
| CNN (only Dense layers quantized) | 98.91% | **98.91%** |

So you trade almost nothing in accuracy for a model that is about 4x smaller.

**Speed of int8 is not always better, and with the Vector API kernels it is now usually worse.** Before the SIMD kernels, int8 was 1.9x faster than float32 on a large synthetic network (2292 to 1186 ms), because big weight tables are limited by memory traffic. After them, float32 runs on the SIMD kernel and int8 does not: the same network takes 410 ms in float32 and 909 ms in int8 (0.45x speed, 4.0x less memory). The shrink formats pay off when memory is the limit, not time. Always measure on your own model.

### How this relates to what AI companies do

The big labs fight the same two problems (memory and speed) with the same families of ideas, just with GPUs and specialized kernels:

| Idea | In big systems | In MiniDL |
|---|---|---|
| Smaller numbers | 16-bit (fp16 / bf16) for training, 8-bit or 4-bit for serving | float32; **16-bit mixed-precision training** with loss scaling (numerics simulated, memory saving real); int8 / int4 / bfloat16 weights for serving |
| Recompute instead of store | activation checkpointing | Conv2D rebuilds patches in backward |
| Compact data and streaming | sharded datasets read batch by batch | `ByteSource` + `DataSource` |
| Cache-friendly math | tiled GPU kernels | **register tiles + SIMD** (Vector API) and K-chunk blocking on the CPU |
| Parallelism | many GPUs | multi-threaded multiply |

MiniDL shows the ideas working and measures them on one CPU core. It does not match GPU frameworks in absolute speed. 16-bit training is simulated (the numbers are rounded exactly as 16-bit hardware would, the maths runs in float32), so it demonstrates the numerics and the memory saving but not the speed-up.

### A lesson from the work

My first "fused" multiply was actually **slower** (0.84x). One dot product at a time cannot use the CPU's vector units well. The fix was computing 4 dot products together. Without the benchmark I would have shipped a slowdown and called it an optimization. In round 2 the gradient-check precision also had to be re-thought: float32 cannot show tiny nudges, so the check uses a bigger nudge and a looser (but still bug-catching) tolerance.

### What is left (ideas)

- A SIMD **int8 / int4 kernel**, so the shrunk models are fast as well as small
- Vector API for the non-multiply loops (im2col, pooling, activations, attention scores)
- Real 16-bit arithmetic (needs hardware or a language that exposes it)
- 16-bit storage for LayerNorm, BatchNorm and attention (only Dense, Conv2D and ReLU use it now)
- A bigger-than-2 GB dataset split over several mapped files
- Confirm the multi-core gain on a multi-core machine (this sandbox has one core)
- Test the Vector API build on Java 17 and on CPUs with 8 lanes (AVX2)

---

## 7. Full real-life walkthrough: delivery time predictor

File: `RealWorldDemo.java`. Think of a food delivery app estimating "arrives in X minutes".

**The hidden truth** (the model does not know this, it must discover it):

```
time = 10 + 3.5 * km + 2 * traffic + 1.5 * items + random noise (about ±3 min)
```

**Step A: make data.** 400 training orders, 100 validation orders (to decide when to stop), 100 test orders (never touched until the end).

**Step B: normalize** inputs and target, learning the scale from training data only.

**Step C: build the model.**

```java
new Sequential()
    .add(new Dense(3, 16, rnd)).add(new Activations.ReLU())
    .add(new Dense(16, 8, rnd)).add(new Activations.ReLU())
    .add(new Dense(8, 1, rnd));      // 1 number out = minutes, no activation
```

`model.summary()` prints the table, 209 weights in total:

```
Layer                       Weights
Dense 3 16                       64
Dense 16 8                      136
Dense 8 1                         9
TOTAL                           209
```

**Step D: train** with `Trainer`: AdamW (0.01, decay 1e-4), cosine learning rate, batch 32, up to 300 epochs, stop after 30 epochs without improvement.

```
epoch    1  loss 0.44770  val 0.19036     ← random weights, terrible
epoch   50  loss 0.03220  val 0.03843
early stop at epoch 50 (best was epoch 20)
```

It stopped on its own at epoch 50 and restored the epoch 20 weights, before it could start memorizing noise.

**Step E: test on unseen orders.**

```
average error 2.60 min,  RMSE 3.22 min,  R2 0.963
```

The noise we added has std 3 min, so even a *perfect* model would average about 2.4 min error (RMSE about 3.0). The model is close to the best possible. It found the real pattern, not the noise.

**Step F: use it.** New order: 6.2 km, traffic 8, 3 items.
- Formula without noise: `10 + 21.7 + 16 + 4.5 = 52.2`
- Model says: **51.2 minutes**.

**Step G: save, then load like a separate app.** `ModelIO.save` + two `Normalizer.save` calls create 3 text files. A fresh program loads them with `ModelIO.loadModel` and `Normalizer.load` (no layers rebuilt by hand) and predicts the same **51.2**.

---

## 8. The other demos (including real MNIST)

**Demo 1: Gradient check.** Proves backprop is right for a model with Dense, BatchNorm and GELU (error about `1.9e-03` in float32). It nudges each weight up and down, measures the loss change, and compares with our gradient. The error measure is relative for big gradients and absolute for tiny ones. Run this first any time you add a layer.

**Demo 2: XOR** (SGD + momentum, Tanh, Sigmoid). Inputs `(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0`. Needs a hidden layer. Final predictions `0.003, 0.994, 0.993, 0.007`.

**Demo 3: Three spirals** (Adam, ReLU, Softmax). Three intertwined arms in 2D, classify each point by arm. Network 2 → 64 → 64 → 3 (4,547 weights). Now with a proper split (240 train / 60 test points):

```
train accuracy = 99.6%    TEST accuracy (unseen points) = 98.3%
```

**Demo 4: Delivery time.** Section 7.

**Demo 5: Real MNIST handwritten digits** (`MnistDemo`). 54,000 training digits, 6,000 for validation (early stopping), and the official 10,000 test digits that are only used at the end. Images stay as bytes (55 MB for all 70,000 digits, 439 MB if stored as double). Everything below ran with `java -Xmx256m`.

**5a. Convolutional network** (`mnist mnist 8 cnn`): Conv 8 filters → ReLU → MaxPool → Conv 16 filters → ReLU → MaxPool → Dense 64 → Dropout → Dense 10. **27,562 weights (107.7 KB).**

```
epoch    1  loss 0.24479  val 0.06384
epoch    4  loss 0.05226  val 0.04200
epoch    8  loss 0.02625  val 0.03479
trained 8 epochs in 203.6 s (25.5 s/epoch)
TEST accuracy on 10000 unseen digits: 98.91%
```

Worst confusions: 7 real **5**s read as **3**, 6 real **8**s read as **9**, 6 real **3**s read as **5**. Digit **8** has the lowest recall (0.982).

**5b. Plain Dense network** (`mnist mnist 10 mlp`): 784 → 128 → 64 → 10, **109,386 weights (427 KB)**.

```
trained 10 epochs in 62.9 s (6.3 s/epoch)
TEST accuracy on 10000 unseen digits: 98.15%
```

**Comparison (before augmentation):**

| Model | Weights | Test accuracy | Time per epoch |
|---|---|---|---|
| Plain network (earlier double version) | 109,386 | 98.08% | 9.8 s |
| Plain network (float32) | 109,386 | 98.15% | 6.3 s |
| **Convolutional network (float32)** | **27,562** | **98.91%** | 25.5 s |

The CNN uses 4x fewer weights and is more accurate, because convolution builds in the idea that nearby pixels belong together. It costs more compute per epoch.

**5c. Shrink it** (printed at the end of every run): `Quantizer.quantize(model)` gives an int8 model, then both are saved as compact binary files.

```
int8 model: test accuracy 98.91%  (float was 98.91%), weights 31.1 KB (float 107.7 KB)     [CNN]
int8 model: test accuracy 98.11%  (float was 98.15%), weights 108.2 KB (float 427.3 KB)     [plain]
files: mnist-cnn.bin 107 KB, mnist-cnn-int8.bin 31 KB
```

**5d. Convolutional network + data augmentation** (`mnist 16 cnn 60000 aug`): same 27,562 weights, trained on randomly shifted (up to 2 px) and rotated (up to 10 degrees) digits. Validation and test digits are NOT augmented. Real run:

```
epoch    1  loss 0.42729  val 0.07980
epoch    8  loss 0.08204  val 0.04289
epoch   16  loss 0.05971  val 0.03608
trained 16 epochs in 411.1 s (25.7 s/epoch)
TEST accuracy on 10000 unseen digits: 99.19%
```

| Model | Weights | Test accuracy |
|---|---|---|
| Plain network (float32) | 109,386 | 98.15% |
| CNN | 27,562 | 98.91% |
| **CNN + augmentation** | **27,562** | **99.19%** |

Augmentation went with +0.28 points (about a quarter fewer mistakes: 109 wrong digits instead of 81 of 10,000). Fair warning: the run without augmentation used 8 epochs and this one 16, so part of the gain may simply come from training longer. A run with 16 epochs and no augmentation was not done. The training loss is higher than without it (0.060 vs 0.026) because the training digits are harder, but the validation loss kept falling to the last epoch, so more epochs would probably help further. Weakest class: digit **9** (recall 0.982).

**Re-run with the Vector API kernels** (the final code): same network, same settings, **99.17%** test accuracy in **196 s (12.3 s per epoch)**, against 99.19% in 411 s (25.7 s) before. The 0.02 point difference is float rounding order, not a change in the method.

Reading the curves for the runs without augmentation: training loss keeps falling while validation levels off. That gap is the start of overfitting, which is why the validation set and early stopping exist. All numbers are from one run with one setting and one random seed, without tuning.

Data source used for testing: the `fgnt/mnist` mirror on GitHub (`get-mnist.sh` downloads it).

**Demo 6: a transformer that writes text** (`CharTransformerDemo`). Training text: Tiny Shakespeare, 1,115,394 characters with 65 different characters. Model: 2 blocks, 64 numbers per token, 4 heads, a window of 32 characters, **110,529 weights**. Batches of 32 windows, 3000 steps with Adam, a cosine learning rate and gradient clipping. About 0.21 s per step on one core, 617 s in total.

```
random guessing:      loss 4.17   (6.02 bits per character)
after   500 steps:    validation loss 2.09
after 3000 steps:    validation loss 1.658  = 2.39 bits per character
```

What the model writes (temperature 0.8, starting from the prompt `ROMEO:`):

```
ROMEO:
That that buves of or now, but he sight like,
Which poor a sold's bloody his prosercan all
Of the wrath the shad rievery must?

VINCENTIGHARD:
Then earth the saw to the maid birng him
To 'been enself bed
that you gentleman: to will
The heart, hence that trutles
To his will my made opince, nor greeks
```

and at temperature 1.0 (more random), prompt `KING:`:

```
KING:
My comisa'd have sword his spretudge.

ROMEO:
Sirrike alon, I see to you slaughter
joh thyself man: youdger eyefuts reight,
Of and when lagge by dawbeen death youngs furn to with figthings.

ScAPULEY:
Why, but you for courses
But is cannot's lick.--mate looks powe
Do blood widow in tof tlage.

OXFOR
```

Read it honestly: it has learned the **shape** of the plays (speaker names in capitals followed by a colon, line breaks, short words, apostrophes, punctuation) and many real small words, but it does not make sense. That is expected for 110 thousand weights, a 32-character memory, 10 minutes of one CPU core, and no word-level knowledge. The point is that every part (embedding, attention, causal mask, LayerNorm, residuals, Adam, cosine schedule, sampling) is the library's own code, checked by tests. Temperature 0.8 sharpens the percentages before sampling (more careful), 1.0 samples them as they are (more varied).

**Demo 7: autograd on XOR.** The whole training loop with no layers and no hand-written backward:

```java
Var p1 = new Var(Tensor.randn(2, 8, 1, rnd)), q1 = new Var(new Tensor(1, 8));
Var p2 = new Var(Tensor.randn(8, 1, 1, rnd)), q2 = new Var(new Tensor(1, 1));
for (int step = 0; step < 1500; step++) {
    for (Var v : new Var[]{p1, q1, p2, q2}) v.zeroGrad();
    Var loss = new Var(x).matmul(p1).addRow(q1).tanh().matmul(p2).addRow(q2).sigmoid().mseLoss(y);
    loss.backward();
    for (Var v : new Var[]{p1, q1, p2, q2})
        for (int i = 0; i < v.value.data.length; i++) v.value.data[i] -= 0.5f * v.grad.data[i];   // plain gradient descent
}
```

**Demo 8: CIFAR-10 colour photos** (`CifarDemo`): **70.64%** on the 10,000 unseen test photos, 156,074 weights, 8 epochs, 61 s per epoch, inside `-Xmx512m`. Weakest class: cat (45.5%); most common mistake: 178 cats read as dogs. Full details, the data source and the per-class table are in Step 32.

**Demo 9: 16-bit training on real digits** (`MixedPrecisionDemo`): the gradient tables and the four training runs (FP32 96.47%, BF16 96.60%, FP16 without scaling 96.54%, FP16 with dynamic scaling 96.60% after 2 epochs) are in Step 28. FP16 without loss scaling lost 3.0% of the gradient entries (4.5% at batch 1024); scaling rescued them.

---

## 9. Use it in your own project

### Recipe A: predict a number (regression)

Delivery time, house price, crop price tomorrow.

```java
Normalizer xn = new Normalizer().fit(trainX);
Normalizer yn = new Normalizer().fit(trainY);

Sequential m = new Sequential()
    .add(new Dense(numFeatures, 16, rnd)).add(new Activations.ReLU())
    .add(new Dense(16, 8, rnd)).add(new Activations.ReLU())
    .add(new Dense(8, 1, rnd));                       // no activation at the end

new Trainer().epochs(300).batchSize(32).validation(xn.transform(valX), yn.transform(valY)).patience(30)
    .fit(m, xn.transform(trainX), yn.transform(trainY), new Losses.MSE(), new Optimizers.Adam(0.01));

double answer = yn.inverse(m.predict(xn.transform(newRow))).get(0, 0);
```

Idea for a project of yours: predict tomorrow's vegetable price from `[last week's price, day of week, rainfall]`. Same recipe.

### Recipe B: pick a class (classification)

Spam / not spam, digit 0 to 9, which crop disease.

```java
// labels must be ONE-HOT: class 2 of 3  →  [0, 0, 1]   (Data.oneHot does this)
Sequential m = new Sequential()
    .add(new Dense(numFeatures, 32, rnd)).add(new Activations.ReLU())
    .add(new Dense(32, numClasses, rnd));             // raw scores, no softmax here

m.fit(x, y, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.01), 100, 32, 0, 10);
double acc = m.accuracy(testX, testY);
```

### Recipe C: yes / no (binary)

```java
Sequential m = new Sequential()
    .add(new Dense(numFeatures, 16, rnd)).add(new Activations.ReLU())
    .add(new Dense(16, 1, rnd));                      // ONE raw score, no Sigmoid layer
m.fit(x, y01, new Losses.BCEWithLogits(), new Optimizers.Adam(0.01), 100, 32, 0, 10);
double score = m.predict(row).get(0, 0);
double probability = 1.0 / (1.0 + Math.exp(-score));
```

### Recipe D: straight from a CSV

```java
Data.Dataset d = Data.loadCsvClassification("flowers.csv", true);   // last column = text label
Tensor[] s = Data.trainTestSplit(d.x, d.y, 0.2, 42);
// ...then Recipe B with numClasses = d.classNames.length
```

### Recipe E: images with a convolutional network

```java
Sequential cnn = new Sequential()
    .add(new Conv2D(1, 28, 28, 8, 3, 1, 0, rnd)).add(new Activations.ReLU())    // 8 x 26 x 26
    .add(new MaxPool2D(8, 26, 26, 2))                                            // 8 x 13 x 13
    .add(new Conv2D(8, 13, 13, 16, 3, 1, 0, rnd)).add(new Activations.ReLU())   // 16 x 11 x 11
    .add(new MaxPool2D(16, 11, 11, 2))                                           // 16 x 5 x 5 = 400 numbers
    .add(new Dense(400, 64, rnd)).add(new Activations.ReLU())
    .add(new Dense(64, 10, rnd));
```

Check the sizes: after a convolution `outSize = (size + 2*padding - filter) / stride + 1`, after a pool `size / window`. The first Dense layer must take `channels * height * width` of the last layer. A wrong number gives a clear "expects N numbers per image" error.

### Recipe F: use little memory, then deploy small

```java
// 1. data stays as bytes, only the current batch becomes floats
ByteSource data = MnistLoader.loadSource(images, labels, 60000);
new Trainer().epochs(8).batchSize(64).validation(data.view(54000, 60000))
    .fit(model, data.view(0, 54000), new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.002));

// 2. shrink for serving, save compactly
Sequential small = Quantizer.quantize(model);
ModelIO.saveBinary(small, "model-int8.bin");

// 3. in the serving program
Sequential served = ModelIO.loadModelBinary("model-int8.bin", new Random());
```

Run with a small Java heap to prove it: `java -Xmx256m -cp out ...`.

### Recipe G: generate text

```java
CharSource text = CharSource.fromText(Files.readString(Path.of("input.txt")), 32);        // window of 32 characters
Sequential lm = CharTransformerDemo.buildModel(text.vocabSize(), new Random(1));          // the model from Step 23
new Trainer().epochs(60).maxBatchesPerEpoch(50).batchSize(32).clipNorm(1.0)
    .schedule(Schedule.cosine(0.003, 0.0003))
    .fit(lm, text.view(0, text.size() - 2100), new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.003));
String story = CharTransformerDemo.generate(lm, text, "ROMEO:\n", 300, 0.8, new Random(3));
```

### Recipe H: your own gradients with autograd

```java
Var w = new Var(Tensor.randn(3, 1, 1, rnd));
Var loss = new Var(x).matmul(w).mseLoss(y);
loss.backward();            // w.grad is ready; update w.value yourself
```

### Recipe I: shrink a model and ship it

```java
Sequential small = Quantizer.quantizeInt4(model, 32);   // or quantize(model) for int8, toBF16(model)
ModelIO.saveBinary(small, "model-int4.bin");
Sequential served = ModelIO.loadModelBinary("model-int4.bin", new Random());
```

### Recipe J: colour photos (CIFAR-10 style)

```java
ByteSource all = CifarLoader.load("data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin");
Sequential cnn = CifarDemo.cnn(new Random(1));
System.out.print(cnn.summary(3, 32, 32));                           // checks every size before training
DataSource aug = new AugmentedSource(all.view(0, 45000), 3, 32, 32, 3, 0, true, 5);   // shift + flip
new Trainer().epochs(8).batchSize(64).validation(all.view(45000, 50000))
    .fit(cnn, aug, new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.002));
```

### Recipe K: 16-bit training

```java
model.setPrecision(Precision.BF16);          // or FP16 (dynamic loss scaling is on by default)
new Trainer().epochs(10).fit(model, train, loss, new Optimizers.Adam(0.001));     // nothing else changes
// model.lossScaler().skippedSteps()  tells you how many steps overflowed and were skipped
```

### Recipe L: invent a layer with autograd

```java
Layer swish = new AutoLayer("Swish", List.of(), (x, p) -> x.mul(x.sigmoid()));   // forward maths only
Sequential m = new Sequential().add(new Dense(2, 16, rnd)).add(swish).add(new Dense(16, 1, rnd));
// gradients, training, GradCheck all work. Check it against a nudge-and-measure gradient before trusting it.
```

### Recipe M: look at the shapes before you train

```java
System.out.print(model.summary(1, 28, 28));     // [channels, height, width] of ONE sample
model.predict(new NDTensor(8, 1, 28, 28));      // a batch of 8 images
```

### Settings cheat sheet

| Setting | Try | If it goes wrong |
|---------|-----|------------------|
| Learning rate (Adam) | `0.001` – `0.01` | Loss = NaN or jumping → lower it. Barely moving → raise it. |
| Learning rate (SGD) | `0.01` – `0.5` | Same |
| Batch size | `16` – `64` | Smaller = noisier, larger = smoother and faster per sample |
| Hidden layer size | `8` – `128` | Too small: can't learn. Too big: overfits. |
| Epochs | let early stopping decide | Test error rising = stop earlier |
| Dropout `p` | `0.1` – `0.5` | Only needed when overfitting |
| Weight decay | `1e-4` – `1e-2` | Too big: model can't learn |
| BatchNorm | after Dense, before the activation | Not needed for tiny networks |

---

## 10. Add your own layer

Every layer implements two methods. Example: a **Square** activation (`x → x²`).

```java
public class Square implements Layer {
    private Tensor in;

    public Tensor forward(Tensor x) {
        in = x;                          // remember input, backward needs it
        return x.mul(x);
    }

    public Tensor backward(Tensor g) {
        return g.mul(in.scale(2));       // derivative of x² is 2x, times incoming blame
    }
}
```

Recipe for any new layer:
1. Write `forward`. Save anything `backward` will need.
2. Work out the derivative on paper. Write `backward`: `incoming gradient × local derivative`.
3. If it has weights, wrap them in `Param` and return them from `params()`. Fill `param.grad` in `backward`.
4. **Run `GradCheck`** on a small model containing it. If the error is below `1e-6`, your maths is right.

---

## 11. How this connects to real AI

The training loop here is the same one used for big models:

| Real AI | What it does | In MiniDL |
|---------|--------------|-----------|
| Language models (the chat kind) | Predict the next token with a transformer. Loss = cross-entropy over the vocabulary, trained with Adam-style optimizers | **The same recipe, tiny**: `CharTransformerDemo` (next character, 110k weights) |
| Image classifiers | Pick a label for a photo | Recipe B |
| Price / demand / ETA prediction | Predict a number | Recipe A (the delivery demo) |
| Fraud and spam detection | Yes / no from many features | 2-class softmax |
| Recommendation systems | Score items for a user | Dense layers on user + item features |

What is different at scale:

| | MiniDL | PyTorch / TensorFlow / DL4J |
|---|---|---|
| Size | up to about 100 thousand weights (MNIST demos) | millions to billions |
| Hardware | one CPU thread | GPUs / many machines |
| Backprop | written by hand per layer, plus an optional autograd engine (`Var`, `AutoLayer`) | **automatic** (autograd builds a graph) |
| Layer types | Dense, Conv2D, pooling, LayerNorm, attention, residual blocks, 5 activations, BatchNorm, Dropout | convolution, attention, recurrent, normalization, hundreds more |
| Data | tensors up to 2D | tensors with any number of dimensions |

Even so, the foundation is identical: **forward → loss → backward → update**. Once you understand this project, a PyTorch training loop will look familiar line by line:

```java
double loss = lossFn.forward(model.forward(x), y);   // forward + loss   (Sequential.forward, Loss.forward)
model.backward(lossFn.backward());                   // backprop          (Sequential.backward)
optimizer.step(model.params());                      // update            (Optimizer.step)
```

---

## 12. Limits (be honest about these)

Knowing the limits of your own project is a good sign, not a bad one.

**Speed**
- With the Vector API kernels: 30 to 61 GFLOP/s in big multiplies, 12.3 s per epoch for the MNIST CNN, 61 s per epoch for the CIFAR CNN, about 0.09 s per training step for the small transformer. GPU-based libraries are built for this and are generally far faster (not measured here).
- Without the flag (portable loops) everything works but is 2 to 4x slower for whole networks.
- The Vector API is an **incubator module**: it needs `--add-modules jdk.incubator.vector` and its interface may change. It was built and tested on Java 21 only (the code uses calls that also exist in Java 17, untested), on an AVX-512 CPU. Speeds differ on other CPUs.
- Only the matrix multiplies use SIMD. Patch building, pooling, activations and the attention loops are plain Java.
- The multi-core code is tested for correctness but its speedup is **unverified**: the test machine had one core. Repeated runs vary a lot (the 512 multiply measured between 30 and 75 GFLOP/s), so read speeds as orders of magnitude.
- The shrunk models (bfloat16, int8, int4) have no SIMD kernel, so they are **slower** than float32 now (int8 0.45x speed). They save memory, not time.
- Grouped and depthwise convolutions run one small multiply per group: few weights, but not tuned for speed.

**Memory and numbers**
- float32 has about 7 digits. 16-bit training is **simulated**: values are rounded exactly as 16-bit hardware would, but Java does the arithmetic in float32. You get the numerics and the memory saving (16-bit storage in Dense, Conv2D, 1-bit ReLU masks), not the speed. Attention, LayerNorm and BatchNorm still keep 32-bit values for backward.
- On the small MNIST network FP16 without loss scaling trained just as well as with it (it lost 3.0% of gradient entries, which did not matter). The benefit of scaling is shown on gradients, not on that accuracy.
- Quantized layers (bfloat16, int8, int4) are **prediction only**. Attention weights are not quantized yet. Dilated / grouped convolutions are not quantized.
- `MappedSource` needs unzipped idx files and is limited to 2 GB per file.
- `ByteSource` / `MappedSource` help data made of 0 to 255 values (images). Other tables are still floats.
- The float32 gradient check is looser (about 1e-3, up to 1e-2 for deep stacks) than the old 64-bit one (1e-8); it still separates a correct layer from a buggy one by a factor of 10 or more.

**Features**
- The training layers work on 2D tables (one flattened sample per row). `NDTensor` and shape inference give N-dimensional shapes, but the layers do not use `NDTensor` inside.
- **No LSTM or other recurrent layers**, no batch normalization for convolutions, no residual CNN demo, no augmentation beyond shifts, rotations and flips.
- The transformer is small and decoder-only (it can generate text but not translate), with no dropout, no weight tying and no tokenizer beyond single characters.
- `Sequential` is a straight line of layers; branching is done with `Residual` blocks. No general graphs.
- Autograd (`Var`, `AutoLayer`) works on 2D tensors with 33 operations. Autograd layers are about 2x slower and use about 3x the memory of the hand-written ones. The built-in layers keep their hand-written backward.
- CSV loader is simple: plain commas, no quoted fields.
- `loadModel` knows the built-in layers and `AutoLayers`; your own layer types need `ModelIO.register(...)`.

**Honesty about results**
- Demos 2 to 4 use **synthetic** data (made by formulas), so the right answer is known. Real data is messier.
- The MNIST and CIFAR numbers are from one run each, one setting, one random seed, no tuning. Larger CNNs with augmentation reach higher on MNIST, and much higher (above 90%) on CIFAR-10. 70.64% on CIFAR-10 is modest.
- The CIFAR-10 data came from an npm mirror (checked: file sizes, label order, class counts), not from the official site.
- The text model does not write meaningful English. It learned spelling and layout.
- Not production ready. For real products use DL4J, PyTorch or TensorFlow.

**What it is good for:** learning, interviews, portfolio, and understanding what the big libraries do, including how they save memory and time.

---

## 13. Future: what to build next

Done items are ticked. Phases 1 to 3 are complete; Phases 4 to 6 are partly done (see the unticked items).

**Phase 1: training toolkit** (done)
- [x] CSV loader, train/validation split, early stopping
- [x] L2 weight decay, AdamW, RMSProp, schedules, gradient clipping
- [x] MNIST demo with a real held-out accuracy
- [x] Save architecture + Normalizer
- [x] BatchNorm, GELU, MAE, BCEWithLogits, metrics, model summary

**Phase 2: speed and memory** (done)
- [x] Cache-friendlier matmul, fused transposed multiplies, in-place operations
- [x] Multi-threaded matmul and convolution (**run the benchmark on a multi-core machine to confirm the gain**)
- [x] float32 everywhere
- [x] Byte-sized data (`ByteSource`), `DataSource` interface, memory-mapped data (`MappedSource`)
- [x] int8 / int4 / bfloat16 weights, int8 convolution, binary model files
- [x] Benchmark class (speed, memory, int8, kernels, autograd)
- [x] **16-bit training** (FP16 and BF16) with loss scaling, master weights, 16-bit activation storage, overflow skipping
- [x] **Java Vector API kernels** (register tiles, SIMD) with a portable fallback, tested both ways
- [x] **Cache blocking** (K-chunks of 256 inside the SIMD kernel; blocking alone on scalar loops measured about 1.1x)

**Phase 3: real upgrade** (done)
- [x] 2D convolution + max pooling, average pooling, strided convolution
- [x] **Dilated and grouped (depthwise) convolution**
- [x] Data augmentation (shift, rotate, flip, also for colour images)
- [x] **N-dimensional tensors** (`NDTensor`: views, broadcasting, permute, reductions, batched matmul) and **shape inference**
- [x] **CIFAR-10** (colour photos): 70.64%
- [x] Autograd (`Var`) and **layers that use it** (`AutoLayer`, `AutoLayers`)
- [x] LayerNorm, skip connections (`Residual`)

**Phase 4: advanced**
- [x] Attention layer and a tiny transformer; character-level text generator on Tiny Shakespeare
- [x] **Recurrent layer / LSTM** for sequences (`LSTM`, full backpropagation through time, gradient-checked; learns to remember the first of 6 numbers: 99% on new data)
- [x] **Capital and small letters**: `EmnistDemo <folder> 8 200000 aug case` (A-Z a-z, 52 classes) and `... full` (0-9 A-Z a-z, 62 classes) from the EMNIST "byclass" files. `LetterCase` decides capital or small for look-alike letters (c C, o O, s S, v V, w W, x X, z Z, k K, p P, u U) by size against the tallest letter on the line. **Trained and tested only on synthetic font data here** (EMNIST could not be downloaded); real accuracy unknown
- [x] **Sub-word tokenizer** (`BpeTokenizer`) now **wired into the transformer demo**: `CharTransformerDemo shakespeare.txt 400 --bpe 400` (Tiny Shakespeare, 1.1M characters = 533,487 tokens; 400 steps in 109 s took the validation loss from 5.99 (random guessing) to 4.12 per token; the samples are rough, the run was short)
- [x] **Dropout** (`--dropout 0.1`) and a **configurable context window** (`--context 64`) in the transformer demo
- [x] **Weight tying** (`TiedOutput`: the output layer reads the input embedding table; `CharTransformerDemo ... --tie`). Gradient-checked, save/load reconnects the tie, a tied model learns a rule. On Tiny Shakespeare (BPE 400, 400 steps) it saves 25,600 weights but was slightly worse after only 400 steps (validation 4.27 vs 4.12 per token): tying usually pays off in longer runs, which were not done
- [x] **Batch normalization for convolutions** (`BatchNorm2D`) and a **ResNet-style model** (`ResNet.build`; MNIST: 97.60% on the 10,000 test digits after 4 epochs on only 12,000 training images)
- [ ] Train the ResNet on CIFAR-10 and see whether it passes 85% (`CifarDemo <folder> 8 aug 50000 resnet`; **not run: no CIFAR data was available when this was built**)
- [ ] GPU through native bindings (very hard, very impressive)

**Next ideas for speed and memory**
- [ ] SIMD int8 / int4 kernels (so shrunk models are fast as well as small)
- [ ] Vector API for im2col, pooling, activations and attention
- [ ] Test the SIMD build on Java 17 and on an AVX2 CPU
- [ ] 16-bit storage for LayerNorm, BatchNorm and attention
- [ ] Make the layers use `NDTensor` internally

**Phase 5: polish for recruiters**
- [x] CI file (Java 17 and 21: tests, jars, version check). It runs when you push; check the Actions tab
- [x] Plots of training vs validation loss (`docs/mnist-training.svg`, drawn by the Java class `Plot` from the real logs in `docs/logs`: `java -jar dist/minidl-1.5.1.jar plot`)
- [x] GitHub Pages site (`SiteBuilder`, section 21)
- [ ] Record a 2-minute demo of MNIST training in a 256 MB limit, and of the text generator (needs a screen recording: not possible to make here)

**Phase 6: serving and PyTorch (new, from analysing DJL Serving)**
- [x] **Model server** (`ModelServer`, `ModelPool`): worker pool per model, dynamic batching, bounded queue (503 when full), scaling, REST API and Prometheus `/metrics`. See section 19 and `docs/DJL-ANALYSIS.md`
- [x] **PyTorch interop, Java only**: `SafeTensors` reads/writes the safetensors format and maps Dense, Conv2D and BatchNorm2D to PyTorch names and layouts (section 20). No Python anywhere in the project
- [ ] Check it with a file written by real PyTorch (none was available here)

**Phase 7: camera and finger tracking (new, 1.5.0; ideas from two example repositories)**
- [x] **Hand and fingertip tracking in pure Java** (`HandTracker`, `Kalman1D`): skin colour (general rule or your own calibration on 9 squares), clean-up, blobs, palm circle, contour, convex hull, convexity defects, fingertips, finger count 0 to 5, two hands, Kalman-smoothed pointer with speed
- [x] **"Start camera" page** (`web/camera.html`, `java -jar minidl.jar camera`): live camera, outlines and fingertips drawn on top, mirror, calibrate, skin-mask view, FPS/pointer/speed display, draw in the air with one finger, trail
- [x] **Write in the air and recognise it** with the MiniDL models (digits, letters, A-Z a-z)
- [x] **Keep faces and the room out** (1.5.1, after real-camera testing showed that colour alone also finds your face, walls and wooden furniture): Learn background, wave to start, face/wall/sliver rejection with the reason shown, hand zone, narrow-at-the-tip fingertip rule
- [ ] **Test with a real webcam and real hands.** Everything so far was tested with drawn hands (JPEG-compressed, 4 skin tones, 3 sizes, 72 of 72 right) and a simulated browser; no real camera was available when this was built
- [ ] Know WHICH finger is which, thumbs-up, OK sign, rock sign (the example web app gets these from a trained landmark model; this tracker only counts fingers)
- [ ] Hand-held pen-up/pen-down by pinching; more than two hands; keep the face out of the way automatically

---

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Loss becomes `NaN` | Learning rate too high, or inputs not normalized | Lower the learning rate (try 10x smaller), use `Normalizer` |
| Loss barely moves | Learning rate too small, or no activation between Dense layers | Raise learning rate, add `ReLU` |
| 3 classes but accuracy stuck near 33% | Labels not one-hot, or last layer size ≠ number of classes | Check label shape: `rows x numClasses` |
| Great on training, bad on new data | Overfitting | Add `Dropout`, more data, smaller model, fewer epochs |
| `matmul shape mismatch` | Layer sizes don't line up | Output size of one `Dense` must equal input size of the next |
| `layer size mismatch` on load | Different architecture than when saved | Build the exact same layers before `load` |
| Predictions look random after reload | Forgot to apply the same `Normalizer` | Normalize inputs with the training-time numbers |
| `OutOfMemoryError` | Whole dataset held as Tensors, or a batch/chunk too big | Use `ByteSource` for images, smaller batches. MNIST needs only `-Xmx256m` this way |
| Gradient check error about `1e-3` | Normal for float32 | A real bug gives `0.1` or more |
| Gradient check on `MaxPool2D` randomly fails | Two numbers in a window almost tied | Not a bug. Try another seed, or check Conv2D without the pool |
| Text model loss stuck near 4.17 (= ln 65) | Learning rate too high or too low, or ids not whole numbers | Try Adam 0.001 to 0.003 with `clipNorm(1.0)`; check `CharSource` |
| `token id N is outside 0..V-1` | Vocabulary size does not match the data | Build the model with `source.vocabSize()` |
| `TokenWise expects N numbers per sequence` | Sequence length or width differs between layers | Keep `T` and `D` the same in `Embedding`, attention and `TokenWise` |
| `Residual block must keep the shape` | The inner layers change the number of columns | Make the last inner layer output the same width as the input |
| `WARNING: Using incubator modules: jdk.incubator.vector` | Normal: the Vector API is an incubator module | Ignore it, or filter it out as `run.sh` does |
| Kernel name says "portable Java loops" | Java was started without `--add-modules jdk.incubator.vector`, or the vector class was not compiled | Compile `src-vector/minidl/*.java` with the flag and start Java with it. Check `Tensor.kernelName()` |
| FP16 training: loss `NaN` or many skipped steps | Overflow: scale too big, or values above 65,504 | Use `LossScaler.dynamic()` (the default), lower the learning rate, or use `Precision.BF16` |
| FP16 training: weights barely change | Gradients underflow to 0 | Turn loss scaling on (`LossScaler.dynamic()`) |
| `layer N (...): ... expects X numbers per sample but gets [..]` | A layer size does not match the layer before it | Run `model.summary(inputShape)` and fix the layer named in the message |
| `cannot broadcast [..] with [..]` | Two `NDTensor` shapes do not line up from the right | Add a dimension of size 1 with `reshape` |
| `groups must divide the channels` | `inC` or `outC` not divisible by `groups` | Pick a `groups` value that divides both |
| `unknown layer type 'X'` when loading | A layer you wrote is not known to `loadModel` | `ModelIO.register("X", (words, rnd) -> ...)` before loading |
| `Conv2D expects N numbers per image` | Wrong height/width/channels | Recompute sizes: `(size + 2*pad - filter)/stride + 1` |
| `QuantizedDense is for prediction only` | Tried to train an int8 model | Train the float model, then `Quantizer.quantize` |
| Training feels slow | Small batches, large layers, one core | Use batch 64+, check `Benchmark`, try a multi-core machine |
| Validation loss rises while training loss falls | Overfitting | Early stopping (`patience`), Dropout, weight decay, more data |
| Bias gradient before BatchNorm is exactly 0 | Normal: BatchNorm cancels it | Nothing to fix |
| Gradient check fails on your new layer | Wrong derivative in `backward` | Re-derive on paper, check which values `forward` must save |

---

## 15. Glossary

- **Tensor**: a table (or stack of tables) of numbers.
- **Weight**: a learnable number. **Bias**: a learnable offset.
- **Neuron**: weighted sum of inputs, plus bias, then (usually) an activation.
- **Layer**: a group of neurons that work side by side.
- **Forward pass**: data goes input → output to make a prediction.
- **Loss**: one number for "how wrong". Lower is better.
- **Gradient**: how much the loss changes if one weight changes a tiny bit.
- **Backpropagation**: computing all gradients using the chain rule, from the output back to the input.
- **Optimizer**: the rule that uses gradients to update weights (SGD, Adam).
- **Learning rate**: how big each update step is.
- **Batch / epoch**: a small group of rows / one pass over all rows.
- **Overfitting**: memorizing the training data instead of learning the pattern.
- **Regularization**: tricks that reduce overfitting (Dropout, weight decay).
- **Normalization**: rescaling columns to the same range.
- **One-hot**: label written as `[0,0,1]` = "class 3 of 3".
- **Softmax**: turns raw scores into probabilities that sum to 1.
- **Hyperparameter**: a setting you choose (learning rate, layer size, epochs), not learned.

---
- **Validation set**: data the model never learns from, used to decide when to stop.
- **Test set**: data used once at the end for an honest score.
- **Early stopping**: stop training when validation stops improving, keep the best weights.
- **Weight decay (L2)**: a small pull of every weight toward 0 to avoid huge weights.
- **Learning rate schedule**: changing the learning rate during training.
- **Gradient clipping**: shrinking a gradient that is too big.
- **Batch normalization**: re-centering and re-scaling a layer's output using the current batch.
- **Logit**: a raw score before softmax or sigmoid turns it into a probability.
- **Confusion matrix**: table of true class versus predicted class.
- **Precision / recall**: when I say X, how often am I right / of all real X, how many did I find.
- **Float32 / double**: numbers stored in 32 bits (about 7 digits) or 64 bits (about 15 digits). Float32 uses half the memory.
- **Quantization (int8)**: storing weights as whole numbers from -127 to 127 plus a scale, to make the model about 4x smaller.
- **Convolution**: sliding a small set of weights (a filter) over an image to detect patterns.
- **Filter (kernel)**: the small grid of weights a convolution slides around, for example 3 x 3.
- **Feature map**: the output of one filter over the whole image.
- **Pooling**: shrinking an image by keeping one number (the biggest, for max pooling) per small square.
- **im2col**: copying image patches into table rows so a convolution becomes one matrix multiply.
- **Recomputation (activation checkpointing)**: rebuilding a value when it is needed again instead of keeping it in memory.
- **Stride / padding**: how far the filter jumps each step / zeros added around the image.
- **Data augmentation**: showing the model slightly changed copies of the training data (shifted, rotated) so it cannot memorize.
- **Token**: one piece of text the model reads (here one character; often a word or word piece).
- **Embedding**: a learned vector for each token id (and each position).
- **Attention**: each token takes a weighted mix of other tokens' values, with weights from comparing a query with keys.
- **Causal mask**: the rule that a token may only look at itself and earlier tokens.
- **Transformer**: a network made of attention blocks and small per-token networks, with residual connections.
- **Residual (skip) connection**: adding a layer's input back onto its output.
- **LayerNorm**: normalizing each row of numbers on its own.
- **Autograd**: automatic differentiation: gradients computed from the forward maths only.
- **bfloat16**: a 16-bit number format with float32's range but less precision.
- **int4 / group-wise quantization**: 4-bit weights with one scale for every small group of weights.
- **Memory-mapped file**: a file the operating system pages in and out on demand, so it does not live in program memory.
- **Temperature**: before sampling, dividing the scores: below 1 makes the model more careful, above 1 more random.
- **Bits per character**: a measure of how well text is predicted (lower is better; random guessing over 65 characters is 6.02).
- **SIMD**: one CPU instruction working on many numbers at once (16 floats here).
- **Register tile**: a small block of the answer kept inside CPU registers while the multiply runs.
- **Cache blocking**: cutting a big job into pieces that fit in the CPU's fast cache.
- **Mixed precision**: computing with 16-bit numbers but keeping 32-bit master copies of the weights.
- **Master weights**: the exact 32-bit weights the optimizer updates in mixed-precision training.
- **Loss scaling**: multiplying the loss by a big number so small gradients survive in FP16, then dividing it out.
- **Underflow / overflow**: a number too small (becomes 0) or too big (becomes infinity) for the format.
- **Dilation**: a convolution filter that looks at pixels spaced apart, for a wider view with the same weights.
- **Depthwise convolution**: every channel gets its own filter (groups = channels).
- **Broadcasting**: stretching a dimension of size 1 so two different shapes can be added or multiplied.
- **View**: a different shape for the same numbers, without copying.
- **Shape inference**: working out the size of every layer's output before running, to catch mistakes.
- **Batched matmul**: many matrix multiplications at once, one per batch entry.

---

## 16. Tests

`java -cp out minidl.Tests` runs **171 checks, all passing**. `run.sh` runs the suite twice, once with the portable kernels and once with the Vector API kernels, so both paths are tested.

- **Maths and kernels:** matmul matches hand-checked values and the textbook version; `A^T.B` and `A.B^T` match; multi-thread equals single-thread; **the Vector API kernels match the textbook multiply on 10 awkward shapes** (1x1, odd sizes, 64x784x128, ...); shape mismatches give clear errors
- **Gradient checks** (float32 tolerance): Dense with Tanh, ReLU, LeakyReLU, GELU, Sigmoid; BatchNorm, LayerNorm; SoftmaxCrossEntropy, BCEWithLogits, MAE; Conv2D (stride, padding, stacked); **dilated, grouped and depthwise-separable convolution**; max and average pooling; a Residual block; the whole mini transformer
- **Convolution correctness:** plain, dilated, grouped, depthwise and strided convolution against a plain loop; Vector API vs portable for forward, input gradient and weight gradient
- **16-bit numbers:** **all 65,536 FP16 bit patterns round-trip**; known values (1, -2, 65,504, 2^-14, 2^-24); overflow to infinity; rounding to the nearest value on 20,000 random numbers; BF16 ties-to-even, infinity and NaN; underflow of 1e-8 and its rescue by loss scaling
- **16-bit training:** a 2e-8 gradient is lost in FP16 without scaling and kept with it; FP32, FP16 and BF16 all learn a classification problem; master weights stay full precision; overflow skips the step, leaves weights untouched and halves the scale; 16-bit storage gives exactly the same gradients; the 1-bit ReLU mask is exact; a transformer takes a BF16 step
- **Transformer:** attention equals a plain double-precision reference; **the future cannot leak into earlier predictions**; embedding = token + position; a tiny transformer learns a text rule
- **Autograd:** all 24 operations match a nudge-and-measure gradient; `AutoDense`, `AutoLayerNorm`, `AutoGELU`, `AutoAttention` equal their hand-written twins; **a whole transformer built from autograd layers matches the hand-written one** (loss and every weight gradient); a model of autograd layers trains XOR; mixed hand-written and autograd layers pass the gradient check; save and load through the registry
- **N-dimensional tensors:** indexing, reshape as a view, toMatrix, permute, broadcasting (and its errors), sum / mean / max, softmax, slice and concat, batched matmul against plain loops, `predict(NDTensor)`
- **Shape inference:** CNN and transformer shapes, wrong sizes caught with the layer number, quantized models keep their shapes
- **Data:** `ByteSource` 1 byte per value, free views, `MappedSource` equals `ByteSource`, `CharSource`, CSV, split, MNIST format, **CIFAR-10 binary format** (with error for wrong files), colour augmentation (channels move together, flips)
- **Shrinking:** int8 Dense and Conv2D, int4, bfloat16, quantizing blocks inside transformers, clear error when training a quantized layer
- **Save/load:** text and binary files rebuild CNNs, nested transformers, grouped convolutions, autograd layers and int8/int4/bfloat16 models with identical predictions
- **Optimizers, schedules, early stopping, Normalizer, BatchNorm, Dropout, Metrics, augmentation**

---

## License

MIT. See `LICENSE`.

Built by Manish Gowda N.

---

## 18. Use it as a library, jar and Maven

```
./build-jar.sh                       # dist/minidl-1.5.1.jar (+ -sources.jar, -javadoc.jar), JDK 17+ only
java -jar dist/minidl-1.5.1.jar help # commands: demo, tests, mnist, emnist, cifar, text, precision, bench, web, version
```
In your own project, put `minidl-1.5.1.jar` on the classpath (`javac -cp minidl-1.5.1.jar MyApp.java`) and `import minidl.*;`.
With Maven: `mvn verify` compiles, runs the test suite and builds the jars (`pom.xml`, no dependencies); `mvn install` puts it in your local repository, then:
```xml
<dependency><groupId>io.github.manishn1877</groupId><artifactId>minidl</artifactId><version>1.2.0</version></dependency>
```
Quality gates: compiles with `-Xlint:all` without a single warning, javadoc has no errors (`-Xdoclint:all,-missing`), CI runs on Java 17 and 21. Model files from outside are checked: a damaged or hostile file gives an `IOException` with a clear message, never a crash or a giant memory allocation.

## 17. Handwriting recognition web app (digits and letters)

Draw in the browser and the model reads it live. Pure Java on the server (the JDK's built-in `com.sun.net.httpserver`, no libraries), one plain HTML file for the page.

```
./run-web.sh                 # digits: uses mnist-cnn-aug.bin / mnist-cnn.bin from MnistDemo
./get-emnist.sh              # letters: download EMNIST letters (~35 MB) into ./emnist
java -Xmx512m -cp out minidl.EmnistDemo emnist 8 124800 aug     # train the letters model
./run-web.sh letters         # letters A-Z, uses emnist-letters-cnn.bin
```
Then open http://localhost:8080. `./run-web.sh` loads every trained model it finds and the page has buttons to switch between **digits+letters**, **letters** and **digits**. If the digits+letters or letters model is missing, a banner says so (a digits-only model can only answer with digits, so it reads an `H` as `4`). Check the labels with your own eyes once: `java -jar dist/minidl-1.5.1.jar check-data emnist mnist`.

What it does:
- **Three ways to write.** *One character (auto-clear)*: write a character, it is recognised, shown, added to a "typed so far" line, and the canvas clears by itself. *Line (locks each character)*: after a short pause (slider, default 0.6 s, or Enter) the character you just wrote is recognised ONCE and LOCKED: its red box, label and picture are frozen and later strokes can never merge with it or move it, even if you write on top of it. The line reads left to right; clear starts again. *Scan*: the old whole-drawing mode for photos and scans. The next bullet describes how scan finds characters.
- **Scan mode.** It finds each character as its own blob of ink, so close writing works (`2310` with 2 px between digits). Touching characters are cut at a pinch in the ink if the model reads both halves well; characters with a doubtful split show a warning mark, uncertain ones a dashed border. Limits: characters whose strokes truly overlap are often read as one (a joined `10` looks like a `0` to the model). See CHANGELOG.md.
- **Preprocessing that matches the training data** (`DigitPreprocess`). Digits (MNIST style): crop to the ink, longer side 20 px, centre by centre of mass. Letters (EMNIST style): longer side 24 px, centred by bounding box. A 99% model reads a small or off-centre drawing badly without this. The page shows "what the model sees".
- **Steadier answers** (`DigitRecognizer`): optional test-time augmentation averages the answer over the picture nudged 1 px in four directions.
- **Teach it.** If it is wrong, press the right character. The model takes a few small training steps on your drawing, mixed with earlier corrections and real training pictures (so it does not forget the rest). Corrections are appended to `feedback.bin` / `feedback-letters.bin` (784 pixel bytes + 1 label byte each). "save tuned model" writes `<model>-tuned.bin`; "reset model" restores the original weights.
- **Train on my handwriting.** The page asks for one character at a time (all of them, shuffled, each round). Draw it, press Enter: the page first checks whether the model already got it right (the score shows how well it knows YOUR writing), then learns from it. Do a round or two and your handwriting gets read better.
- **Load a photo or scan** of handwriting (background removed and contrast stretched in the page).
- Shows the 6 most likely answers, the milliseconds per answer, and the test accuracy.

EMNIST letters: 124,800 train + 20,800 test pictures, 26 classes. `EmnistLoader` turns every picture back upright (EMNIST stores them sideways) and shifts labels 1..26 to 0..25. Upper and lower case share one class, so answers are shown as capitals. Letters are harder than digits ("I" vs "l", "O" vs "Q"): expect roughly 90 to 93% on the test set, not 99%.

Limits: one user at a time is the intended use (corrections change the one shared model); the server listens on localhost only. The EMNIST-style preprocessing follows the paper's description: if letters are read badly, compare "what the model sees" with real EMNIST pictures and tune the 24 px box in `DigitPreprocess`.

**Digits AND letters in one model (36 classes: 0-9, A-Z).** `./train-all.sh` also trains `alnum-cnn.bin` (`AlnumDemo`), and `./run-web.sh all` opens the page with it, so one drawing like `K7B2` is read in one go. `Alnum` puts MNIST digits and EMNIST letters into one shuffled data set, re-making every digit in the EMNIST style so the model sees one kind of picture. Pairs that look the same (0/O, 1/I, 2/Z, 5/S, 8/B) can only be guessed from the shape and the writer's habit, so expect a few more mix-ups than with the two separate models. If you know it is a number, use the digits page.

Files: `DigitPreprocess.java`, `DigitRecognizer.java`, `HandwritingServer.java`, `EmnistLoader.java`, `EmnistDemo.java`, `Alnum.java`, `AlnumDemo.java`, `train-all.sh`, `web/index.html`, `run-web.sh`, `get-emnist.sh`. the full suite is 331 tests.

## 19. Model server (Phase 6)

Serve any saved MiniDL model over HTTP, with the ideas of the Java part of DJL Serving (analysis: `docs/DJL-ANALYSIS.md`). Pure Java, no libraries.

```
java -jar dist/minidl-1.5.1.jar serve                       # serves every *.bin in this folder
java -jar dist/minidl-1.5.1.jar serve --port 8080 --model digits=mnist-cnn-aug.bin,batch=32,delay=5,workers=1,max_workers=4,queue=500
curl localhost:8080/ping
curl -X POST localhost:8080/predictions/digits -d '{"inputs": [[0.0, 0.1, ...784 numbers...]]}'
```
| Call | What it does |
|---|---|
| `GET /ping` | health |
| `GET /models`, `GET /models/{name}` | the models; workers, queue, batch settings, request/error/rejected counts, latency p50/p99 |
| `POST /models?url=f.bin&name=n&batch_size=16&max_batch_delay=5&min_workers=1&max_workers=4&queue_size=100` | load a model (`url` is a file NAME inside the model store folder, never a path) |
| `PUT /models/{name}?min_workers=2&max_workers=4`, `DELETE /models/{name}` | scale, unload |
| `POST /predictions/{name}` | body `{"inputs": [[...], ...]}` or `{"input": [...]}`; answer `{"outputs": [[...]], "argmax": [...]}` |
| `GET /metrics` | Prometheus text: requests, errors, rejected, batches, batch rows, queue, workers, latency histogram |

How it works: every worker thread owns its own copy of the model (a layer remembers its inputs, so one copy cannot be shared).
A worker takes one request, then keeps collecting more until `batch_size` rows or `max_batch_delay` ms, stacks them into ONE matrix
and runs ONE forward pass; each caller gets back its own rows. A full queue answers 503 at once. A bad request (wrong width, broken
JSON) gets 400 and never disturbs the others. Safety: localhost only, 8 MB body limit, at most 10,000 rows per request.

Measured here (digits model, real MNIST test images): 400 single-image requests sent in parallel were answered in 1.3 s with 99.0%
accuracy, grouped into 38 forward passes (about 10.5 images each), p50 31 ms, p99 134 ms, none rejected. One machine, one run: a rough
demonstration, not a benchmark. Any HTTP client works (`curl`, Java `HttpClient`).

## 20. PyTorch interop in Java only: SafeTensors (Phase 6)

DJL Serving runs PyTorch models by starting a Python process. MiniDL stays 100% Java instead: it reads and writes **safetensors**, the plain
number-only file format the PyTorch world uses to share weights (`SafeTensors.java`, no code can hide in such a file). So weights trained
elsewhere can be loaded here, and MiniDL weights can be loaded there, with no Python in this project.

```java
SafeTensors.exportModel(model, "weights.safetensors");      // MiniDL -> PyTorch names and layouts
SafeTensors.importModel(model, "weights.safetensors");      // the other way, into a model with the SAME layers
```
Mapping (`i` = position of the layer in the Sequential, counting every layer; this is the index of an `nn.Sequential` built in the same order,
without `nn.Flatten`): Dense -> `i.weight` (out, in) and `i.bias`; Conv2D -> `i.weight` (out, in/groups, k, k) and `i.bias`;
BatchNorm2D -> `i.weight`, `i.bias`, `i.running_mean`, `i.running_var`, `i.num_batches_tracked`. Other layers with weights (LSTM, LayerNorm, attention)
say clearly that they have no mapping yet. Every name and shape is checked on import.

Tested: a file built by hand from the format description is read correctly; export then import gives identical predictions; damaged files and
wrong shapes give clear errors (10 tests). **Not tested: a file written by real PyTorch** (PyTorch was not available when this was written); the layouts
follow the PyTorch documentation. Try one file from your PyTorch model and tell me if a name or shape does not match.

## 21. Website and GitHub Pages

The handwriting page is `web/index.html`; the Java server serves it. GitHub Pages can only host files, not run Java, so on Pages the page
**draws and the Java program on your own computer recognises**:
1. Turn on Pages: repository Settings, Pages, branch `main`, folder `/docs`. Build the site files with `java -jar dist/minidl-1.5.1.jar site` and commit `docs/`.
2. On your computer run `./run-web.sh` (or `java -jar dist/minidl-1.5.1.jar web`).
3. Open the Pages address. The page finds the server at `http://localhost:8080` (or the address you type and press connect).

Problems fixed in 1.3.0 (found by reading the page against how Pages works; if you saw others, tell me what you saw):
- Every request used an absolute path (`/api/...`), which on a project page (`user.github.io/minidl-java/`) points to the wrong place. Paths are now relative.
- A browser blocks a page from one website calling another unless the server allows it. The server now allows the project's Pages site (`-Dminidl.cors=...` to change) and answers the browser's permission check, including the one for https pages calling localhost.
- Safety: every POST needs the header `X-MiniDL`, so a random website cannot make your browser change or save your local model; other websites get no permission at all.
- When the server cannot be reached the page now says so, shows how to start it, and has a box to type the server address (remembered).
- Phones: the canvas no longer scrolls the page, selects text or opens the long-press menu while drawing; keyboard shortcuts no longer fire while typing in the address box.
Not tested in a real browser (none available here): the headers and paths are tested with an HTTP client; the page itself was only read and checked for broken references.

## 22. Capital and small letters, and the new writing modes (1.4.0)

**Models** (the page lists every model it finds; the broadest opens first): `emnist-full-cnn.bin` digits + A-Z + a-z (62 classes), `emnist-case-cnn.bin` A-Z + a-z (52), `alnum-cnn.bin` digits + merged letters (36), `emnist-letters-cnn.bin` merged letters (26), `mnist-cnn-aug.bin` digits (10).
Train the new ones with `./get-emnist.sh` (also downloads the byclass files; that download link was not checked) and `./train-all.sh`, or one at a time:
`java -Xmx2g -cp out minidl.EmnistDemo emnist 8 200000 aug case` (or `full`). Check the labels first: `java -jar dist/minidl-1.5.1.jar check-data emnist --byclass`.

**Why capital vs small is hard.** EMNIST scales every picture to the same size, so a `c` and a `C` look identical to the model. The page therefore gives the server the height of the tallest other character on the line (`ref`), and `LetterCase` calls a look-alike letter *small* when it is under 68% of that height, otherwise *capital*. Letters with different shapes (b B, d D, e E ...) are left to the model. A letter with nothing to compare to is marked `?`. The **case** buttons (auto / ABC / abc) override it. Limit: a line made only of small letters has no tall letter to compare to, so use the abc button.

**Locking.** The character being written lives on its own layer; the red boxes of finished characters are drawn from stored coordinates and are never recomputed. The page asks the server about one character at a time (cropped to its own ink), so what you write next can not change an earlier answer. The position of a new stroke is NOT used to decide that a character is finished (the strokes of an H or an M can be far apart): only the pause or Enter does. If two characters merge, lower the pause slider; if one letter splits in two (E, H), raise it.

Tested: the case rule, byclass loader, alphabets, server parameters and safety rules in Java (17 tests); the page's behaviour (3218 written close together, a new character drawn on top of locked ones, clear, auto-clear and history, case buttons) in a simulated browser with a real canvas against the real server. Not tested in a real browser or on a phone.

## 23. Finger tracking with the camera (Phase 7)

```
java -jar dist/minidl-1.5.1.jar camera        # starts the server and opens http://localhost:8080/camera
```
(or `./run-web.sh`, then open `http://localhost:8080/camera`, or press **Start camera (finger tracking)** on the handwriting page.)
Press **Start camera**, allow the camera. The JDK has no webcam API, so the **browser** reads the camera and sends small JPEG pictures (320 wide) to the Java
program; **all the finding is Java** (`HandTracker`, about 24 ms per picture at worst, 5 ms typical). Nothing leaves your computer.

What you see: the hand outline (cyan), the convex hull (yellow), the valleys between fingers (blue), fingertips (red), the palm circle (magenta), the pointer
(green cross, smoothed by a Kalman filter), the gesture name and finger count, FPS, server time, pointer position and speed. Buttons: **Mirror**, **Calibrate (Z)**,
**Forget calibration**, **Show skin mask**, **Draw**, **Clear drawing**, colour, pen size, trail, **Recognise drawing**.

How it works, step by step (the same pipeline as the OpenCV finger-tracking tutorial, written from scratch in Java):
1. **Calibrate (optional).** Cover the 9 green squares with your palm and press Calibrate (Z): a hue/saturation histogram of YOUR skin in YOUR light is built. Without it a general skin-colour rule is used.
2. **Skin mask:** every pixel is scored by the histogram (back-projection), smoothed, thresholded; then eroded and dilated once (the tutorial does it twice at twice the resolution).
3. **Hands:** connected blobs, the two biggest that are large enough.
4. **Palm:** distance transform; the centre of the biggest inscribed circle.
5. **Outline, hull, defects:** boundary tracing, Andrew's convex hull, convexity defects (the valleys).
6. **Fingertips:** hull corners far beyond the palm circle and not on the picture border; close ones merged. The count is the number of fingers shown; the farthest tip is the pointer.
7. **Tracking:** hands keep their slot between pictures; a Kalman filter smooths the pointer, gives its speed, and keeps guessing for a few pictures if it is lost.

**Gestures:** 0 fingers FIST, 1 POINT, 2 PEACE, 3 THREE, 4 FOUR, 5 OPEN PALM. **Writing in the air:** press Draw, point with ONE finger to draw, two fingers to move without drawing, a fist lifts the pen,
an open palm held for 1.5 s clears. When you stop, the drawing is recognised by the model you picked (one character or a line); "recognise by itself" also clears it afterwards.

API (same server, same safety rules: POST needs the `X-MiniDL` header): `POST /api/hand` (JPEG body; `?t=seconds`, `?mask=1`, `?wave=0|1`, `?face=0|1`, `?zone=x0,y0,x1,y1` as fractions; the answer also lists `rejected` blobs with reasons) and `POST /api/hand/background?step=begin|add|done|clear` returns hands with fingers, gesture, palm, pointer, tips, hull, valleys, outline;
`POST /api/hand/calibrate`, `POST /api/hand/reset`, `GET /api/hand/config?w=&h=` (where the calibration squares are). Pictures over 3 MB or 2.5 million pixels are refused before decoding.

**Keeping your face and the room out (1.5.1).** The first version only looked at colour, so a real webcam showed that your face, a wooden wall or furniture counted as hands. Now, in this order of strength:
1. **Learn background (B).** Take your hands out of the picture (your face and the room may stay) and press it: 15 pictures become a reference. From then on only what is NEW compared with it can be a hand, so a resting face, wall and shelf are background. The reference follows slow light changes. Press it again if you move the camera or the room.
2. **Wave to start** (on by default, only used when no background is learned): a skin-coloured blob must move before it becomes a NEW hand, so a still face, wall or shelf never does. A hand coming into the picture counts as moving; once found, a hand may rest.
3. **Not a hand, whatever its colour:** a blob with 2 or more eye/mouth-like holes is a face; a blob over 30% of the picture or touching 3 edges is a wall; a sliver without a palm is an edge (for example the new edge of a head that moved); with a learned background, a new blob whose centre is at the old face is the head moving.
4. **Hand zone:** press the button and drag a rectangle on the picture: only skin inside it counts. Fingertips must also be narrow near the tip, so a forehead or forearm does not give false fingers.
5. **You can see it.** Everything ignored is drawn with a grey dashed box and its reason ("ignored: face-like: 2 holes inside (eyes, mouth)"), and listed next to the picture. Untick **ignore faces** or **wave to start** to turn a rule off.

Not perfect: the face rule is a shape test (it needs the eyes to show as holes in the skin picture, which works in decent light), a hand held right at the old face position is also ignored while the background is learned, and a moving camera or a big change of light breaks the learned background (press B again).

**Limits, honestly:** it is colour based. Your face and arm are skin colour too (keep them out of the picture, wear long sleeves, plain background), strong coloured light or a skin-coloured wall confuses it, and a long forearm
makes the palm circle less steady. It counts fingers; it does not know which finger is which, so thumbs-up, rock and OK signs are not recognised (that needs a trained landmark model).
The camera needs a secure page: `http://localhost` works, so does https. On the GitHub Pages site the camera page works the same way if the Java server runs on your computer (open the Pages address, the page finds `http://localhost:8080`).
**Tested:** drawn hands, JPEG-compressed (4 skin tones, 3 sizes, 0 to 5 fingers: 72 of 72 right, no hands invented in empty pictures); a drawn room with a wooden wall, a shelf, a face with eyes and mouth and a hand: only the hand is found, with and without a learned background, with the face moving, with a zone, the geometry and the Kalman filter on their own, the HTTP endpoints and their limits (27 tests),
and the whole page in a simulated browser against the real server (gestures, pointer within 6 px, calibration, writing a 7 in the air and reading it as 7 at 94%; the room scene above through the page). **Not tested here: a real webcam with a real face and room.** The face and wall rules were designed from what you reported and tested on drawn faces; real faces vary, so tell me what it still gets wrong.

Credits: the finger-tracking ideas follow `amarlearning/Finger-Detection-and-Tracking` (Python and OpenCV: palm calibration squares, histogram back-projection, contours, hull, convexity defects, farthest point as fingertip, trail)
and `sparshsinghal1618/finger-track` (browser app: Start Camera button, mirror, HUD with FPS/position/speed, finger-count gestures, drawing in the air, two hands). No code was copied; this is a Java implementation of the same ideas.
