# Changelog

## 1.5.1 - keep faces, walls and furniture out of the finger tracking

Reported after testing with a real camera: the tracker counted the face and background objects as hands (colour alone cannot tell skin from a wooden wall or a face).
- **Learn background** (button, key B, `POST /api/hand/background?step=begin|add|done|clear`): 15 pictures without hands become a reference (per-pixel mean and spread, follows slow light changes); only new things can be hands.
- **Wave to start** (checkbox, default on): without a background a blob must move before it becomes a new hand; tracked hands may rest.
- **Rejection with reasons:** faces (2 or more enclosed holes), blobs over 30% of the picture or touching 3 edges (walls), slivers with no palm, new blobs whose centre is at the face seen in the background (head moved). Rejected blobs are returned (`rejected`), drawn grey and listed.
- **Hand zone** (drag a rectangle; `?zone=`), **ignore faces** checkbox (`?face=`), **wave** (`?wave=`).
- **Fingertips must be narrow near the tip**: found with a drawn face (a forehead gave false fingers), also protects against forearms.
- Camera page: guidance for the two steps, status for skin and background, learning countdown.
- Tests: 331 (was 310): drawn room (wooden wall, shelf, face with eyes and mouth, hand): only the hand is found; wave to start; resting hand kept; background learned over HTTP; head moving; light drift +12; zone; holes; bad zone; too few background pictures. The whole scenario also runs through the page in a simulated browser.
- Not tested: a real webcam with a real face and room. The face rule is a shape test and real faces vary.

## 1.5.0 - camera and finger tracking in Java (Phase 7)

Ideas from `amarlearning/Finger-Detection-and-Tracking` (OpenCV pipeline) and `sparshsinghal1618/finger-track` (browser app with a Start Camera button). Java only, no code copied.

- `HandTracker`: skin colour (general YCrCb rule, or a histogram calibrated on 9 squares), cleanup, blobs, palm circle (distance transform), outline tracing, convex hull, convexity defects,
  fingertips, finger count 0 to 5, up to two hands with stable slots, Kalman-smoothed pointer and speed (`Kalman1D`).
- Server: `POST /api/hand`, `/api/hand/calibrate`, `/api/hand/reset`, `GET /api/hand/config`, and the page `/camera`. Same safety rules as the other writing calls (X-MiniDL header), pictures over 3 MB or 2.5 million pixels refused before decoding.
- `web/camera.html`: **Start camera** (the browser reads the camera, Java analyses the pictures), live overlays, mirror, calibrate, skin-mask view, FPS / server time / pointer / speed, gesture names,
  draw in the air with one finger (two fingers move, fist lifts the pen, open palm held 1.5 s clears), colour and pen size, trail, **recognise the drawing** with any MiniDL model (one character or a line), auto-recognise and auto-clear.
  The handwriting page has a **Start camera (finger tracking)** button; `java -jar minidl.jar camera` starts the server and opens the page; the GitHub Pages site gets the camera page too (`SiteBuilder`).
- Found and fixed while testing: the cleanup settings were tuned for 640x480 (erode/dilate twice) but the tracker works at 320x240, and JPEG compression (what a camera stream is) broke thin fingers in calibrated mode
  (a 3-finger hand read as 0 or 1). A sweep over erode, smoothing and threshold found a stable setting (erode 1, smoothing radius 4, threshold 40); the tests now JPEG-compress their pictures.
- Measured (drawn hands, JPEG quality 0.7, 4 skin tones, 3 sizes, 0 to 5 fingers): 72 of 72 counted right, 0 hands invented in empty pictures, 5 ms typical per picture, 13 fps through the page in a simulated browser.
- Tests: 310 (was 283): 16 for the tracker, geometry and Kalman filter, 11 for the camera endpoints.
- **Not tested: a real webcam with real hands.** Limits: colour based (face and arms are skin), counts fingers only (no thumbs-up / OK sign / which finger).

## 1.4.0 - character locking, auto-clear, capital + small letters, weight tying

### Web page
- **Single mode clears itself** after each recognised character (the answer stays on screen and is added to "typed so far").
- **Line mode locks each character**: after a pause (slider, default 0.6 s) or Enter, the character is recognised once and frozen (box, label, picture). New strokes cannot merge with it, move it, or change its box, even when drawn over it. Order is left to right. Clear starts again.
- Scan mode keeps the old whole-drawing recognition (used for photos).
- Teach (correct a wrong answer) works on a locked character; switching models re-reads the locked characters.
- Found while testing: choosing the end of a character by the position of the next stroke split an H into three pieces, so only the pause decides.

### Capital and small letters
- `EmnistLoader.byClass` (62 classes, or the 52 letters), `EmnistDemo ... letters|case|full`, alphabets for 52 and 62 classes, servers files them as `case` / `full`, `check-data --byclass`.
- `LetterCase`: look-alike letters (c C, o O, s S, v V, w W, x X, z Z, k K, p P, u U) get their case from their size against the tallest character on the line; case buttons auto / ABC / abc; the answer is marked `amb` when the case is only a guess.
- Only tested on synthetic font data (the EMNIST download is blocked in the build environment): the pipeline works, real accuracy is unknown.

### Phase 4
- **Weight tying**: `TiedOutput` shares the embedding table with the output layer; `CharTransformerDemo --tie`. Gradient-checked; save/load reconnects it; learns a rule. Short Tiny Shakespeare run: 25,600 fewer weights, validation 4.27 vs 4.12 per token (slightly worse after 400 steps).
- Still open: CIFAR-10 ResNet training (no data), GPU. Phase 5: demo recording. Phase 6: check SafeTensors against a file written by real PyTorch.

### Tests: 283 (was 266). New: byclass loader, alphabets, case rule, server case parameters, weight tying (4).

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
