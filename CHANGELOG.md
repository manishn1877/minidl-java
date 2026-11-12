# Changelog

## 1.3.0 - Java only, website fixes, Phase 4 leftovers

### Java only
- Removed the `python/` folder (exporter, client, parity scripts) and the Python plot script. Nothing in the project is Python any more.
- `SafeTensors` (Java): reads and writes the safetensors format and maps Dense, Conv2D, BatchNorm2D to PyTorch names and layouts, so PyTorch interop needs no Python.
  10 tests (hand-built file, round trip, damaged files, wrong shapes). Not tested with a file written by real PyTorch.
- `Plot` (Java) draws the training curves as SVG (`docs/mnist-training.svg`, 3 charts from the real logs); `SiteBuilder` builds the GitHub Pages site into `docs/`.
- New jar commands: `plot`, `site`.

### Website (GitHub Pages and phones)
- Relative request paths (absolute `/api/...` pointed to the wrong place on a project page); optional server address (`?server=` or the box on the page, remembered).
- The server allows the project's Pages site to call it (CORS, including the private-network permission for https pages calling localhost); `-Dminidl.cors=` changes the list.
- Safety: every POST needs the header `X-MiniDL`; other websites get no permission. 5 tests.
- A clear "cannot reach the Java server" message with the command to start it; touch screens no longer scroll or select while drawing; shortcuts ignored while typing.
- Not tested in a real browser. Only the headers, paths and server behaviour are tested.

### Phase 4 leftovers
- `TokenSource` and `CharTransformerDemo --bpe VOCAB --context T --dropout P`: the transformer over BPE tokens, with dropout and a chosen window.
  Measured: Tiny Shakespeare, 400 steps in 109 s, validation loss 5.99 (random) to 4.12 per token.
- Still open: weight tying (needs shared parameters), GPU, CIFAR-10 ResNet training (no data here).

## 1.2.0 - Phase 4 layers, serving, PyTorch bridge (from analysing DJL Serving)

Source of the ideas: `djl-serving-master.zip` (222 Java files). Only its Java design was used; analysis in `docs/DJL-ANALYSIS.md`.

### Phase 6: model server (new)
- `ModelServer`, `ModelPool`, `MiniJson`: worker pool per model with its own model copies, dynamic batching, bounded queue (503), scaling,
  REST API (`/ping`, `/predictions/{name}`, `/models`) and Prometheus `/metrics`. `serve` command in the jar.
- Safety: localhost only; model files are names inside the store folder (path tricks, missing files, wrong types give 400); body limit 8 MB; broken input gives 400, not 500.
- 21 tests: batching gives every caller its own correct row, bad request isolated, queue full refuses, scaling, 7 kinds of broken JSON,
  paths outside the store, 30 parallel HTTP requests all correct, metrics format, delete.

### Phase 4 (new layers)
- `BatchNorm2D` (per-channel normalization for pictures), `LSTM` (full BPTT), `ResNet.build` (stem, residual stages, stride-2 downsampling, global average pool),
  `BpeTokenizer` (byte-pair encoding). `ModelIO` can save/load the new layers; `MnistDemo ... resnet` and `CifarDemo ... resnet` train the ResNet.
- Tests: gradient checks for BatchNorm2D, LSTM (both output modes) and a ResNet-style stack; per-channel mean 0 / std 1; the LSTM learns a memory task (99%);
  ResNet trains and survives save/load; BPE round trip, unknown character, save/load.
- Measured: ResNet on MNIST, 4 epochs, 12,000 training images: 97.60% on the 10,000 test digits.
- The ResNet gradient check uses tanh: with ReLU the finite-difference check fails through batch normalization because of ReLU kinks (checked: with tanh the same stack
  passes at 0.0004, so the backward code is right).
- NOT done: CIFAR-10 ResNet training (no data here), GPU, weight tying, dropout/bigger context in the transformer, BPE wired into the text demo.

### Phase 5
- `docs/mnist-training.png` plus the real logs in `docs/logs`; CI runs Java 17 and 21, builds the jars and runs the Python parity check.

### PyTorch bridge (new, `python/`; REMOVED again in 1.3.0, see above)
- `minidl_torch.py` (export), `minidl_format.py` (writer and conversions, NumPy only), `minidl_client.py` (client for the server),
  `train_mnist_torch.py`, `verify_parity.py` (real PyTorch), `verify_numpy_parity.py` (NumPy reference).
- Verified with NumPy reference only (worst difference 7e-5). The real-PyTorch scripts were NOT run: PyTorch could not be imported here.

### Fixed
- Pooling in the exporter follows the floor rule of PyTorch (an 11 x 11 picture with kernel 2 gives 5 x 5).

## 1.1.0 - handwriting fixes: close characters and letters

### Problem 1: characters written close together were joined (`2310` was read as `2` or `0`)
Cause: `DigitPreprocess.segment()` started a new character only after a run of EMPTY columns as wide as 1/40 of the canvas
width (18 px on the 720 px canvas). Anything closer than that was read as ONE picture. It also copied the neighbour's ink
into the box of the character next to it.

Changes (`DigitPreprocess.java`):
- Characters are now found as connected blobs of ink (so `1` and `0` 2 px apart, or slanted so their columns overlap, are two characters).
- A blob stacked on a bigger one (the dot of an i, a loose bar) joins it; dust is dropped.
- Every character gets its own mask: a neighbour's ink inside its bounding box is no longer copied into it.
- Safeguards that use the model (`DigitPreprocess.Scorer`): a shape that is too wide for one character is cut at a real pinch in
  its ink profile if the model reads both parts well (touching `1` + `0`); two neighbouring blobs are joined again if the model
  reads them much better together (a `4` drawn with two strokes). Cuts need: model not sure about the whole shape (or far too wide),
  a pinch at most 0.6 x the smaller peak, parts at least 6% of the height wide, both parts read with confidence 0.9 on average.
- Each piece can carry a `note` ("split a joined shape (whole 0.43, parts 1.00 and 1.00)"); the page shows it as a warning mark,
  and characters with confidence below 80% get a dashed border.
- New thresholds were tuned on measurements, not guessed: real MNIST digits are up to 1.67 times wider than tall, so "too wide" is 1.3 for digits.

Measured (real MNIST digits composed into `2310 110 100 2026 12345 0 1 8 4`, the 98.9% digits model, 54 sequences per row):

| writing                         | before | after |
|---------------------------------|-------:|------:|
| gap 25 px                       | 53     | 53    |
| gap 10 px                       | 24     | 51    |
| gap 5 px                        | 22     | 51    |
| gap 1 px                        | 23     | 52    |
| gap 6 px, slanted               | 22     | 52    |
| strokes overlap 4 px            | 23     | 38    |
| overlap 3 px, slanted           | 24     | 49    |

400 single digits (some slanted): 0.2% were wrongly cut in two (1 of 400); before tuning it was 3%.
Remaining failures are mostly single digits the model misreads, and digits whose strokes really overlap and that the model is
sure about as one shape (a joined `10` is read as `0` with 98% confidence). That cannot be fixed by segmentation alone.

### Problem 2: letters were read as numbers
Cause: `./run-web.sh` loaded the digits-only model (`mnist-cnn-aug.bin`, 10 outputs) unless started with `all`, and nothing on
the page said so. A model with 10 outputs can only answer 0-9, so a handwritten `H` became `4` or `1`.

Changes (`HandwritingServer.java`, `web/index.html`, `run-web.sh`):
- The server loads every model it finds (`alnum-cnn.bin` = digits+letters, `emnist-letters-cnn.bin`, `mnist-cnn-aug.bin`) and the page has
  buttons to switch. It opens with the broadest one.
- If the digits+letters or letters model is missing, a yellow banner says so and tells you what to run. Nothing is silently replaced.
- Asking for a model that is not loaded gives an error, not a different model's answer.
- Each answer shows the character, not a class number: the class -> character mapping is tested (class 10 = A, 35 = Z).
- New `DataCheck` command (`java -jar minidl.jar check-data emnist mnist`) prints one picture per class so you can SEE that the labels match.

Label mapping that the code uses (EMNIST "letters" split): labels 1..26 = A..Z (upper and lower case merged), pictures stored sideways
(`EmnistLoader` turns them upright and shifts labels to 0..25). Digits+letters model: classes 0-9 = `0`-`9`, 10-35 = `A`-`Z`.
Not verified here against the real EMNIST files (the download is blocked in the build environment): run `check-data` once on your copy.
Ambiguous pairs (0/O, 1/I, 2/Z, 5/S, 8/B, 6/G) can only be guessed from shape and habit; the page shows the confidence and the top 6 answers.

### Other
- Version 1.1.0 (`MiniDL.VERSION`, `pom.xml`), CI uses the new jar name.
- 215 tests (was 197): segmentation (close, slanted, touching, dots, many characters, order, single `0` and `8` not cut),
  scale/position invariance of the preprocessing, class mapping, model filing.

## 1.0.0
Handwriting web app, EMNIST letters, digits+letters model, jar and Maven build.
