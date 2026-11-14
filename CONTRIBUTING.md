# Contributing to MiniDL

Thanks for helping! MiniDL is a deep learning library written from scratch in pure Java, so contributions that keep it small, readable and well tested are the best ones.

## Quick setup

You need a JDK 17 or newer.

```
git clone https://github.com/manishn1877/minidl-java.git
cd minidl-java
./build-jar.sh                              # builds dist/minidl-1.4.0.jar
java -jar dist/minidl-1.4.0.jar tests       # all 283 tests must pass
```

Optional, to test the SIMD kernels too:

```
javac --add-modules jdk.incubator.vector -cp out -d out src-vector/minidl/*.java
java --add-modules jdk.incubator.vector -cp out minidl.Tests
```

## How to contribute

1. Fork the repo and create a branch (`git checkout -b my-change`).
2. Make your change in `src/minidl/`.
3. Add a test for it in `Tests.java`. A new layer needs a gradient check (`GradCheck`).
4. Run the tests again. Everything must pass.
5. Open a pull request. Say what you changed and why, and paste the test result.

## Rules

- **Pure Java only.** No outside machine learning libraries, and no Python files in the project. Shell scripts and the HTML page are fine.
- **Keep it readable.** This project exists so people can read it. Prefer clear code over clever code, and add a short comment when the maths is not obvious.
- **Test everything new.** A change without a test is hard to trust.
- **No warnings.** The code compiles with `-Xlint:all` without warnings.
- **Be honest about results.** If you report an accuracy or speed number, say how you measured it (one run, which machine, which settings).

## Adding a new layer

1. Implement `Layer` (`forward`, `backward`, and `params()` if it has weights).
2. Save in `forward` whatever `backward` needs.
3. Work out the derivative on paper, then write `backward`.
4. Add a gradient check test on a small model that contains your layer.
5. If it should load from saved files, register it with `ModelIO.register("MyLayer", ...)`.

For a quick experiment you can write only the forward maths with `AutoLayer` (autograd) and compare it against a nudge-and-measure gradient.

## Good first issues

Look for issues labelled `good first issue`, for example a new activation or loss, a Dockerfile for the web demo, or README improvements.

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened, your Java version (`java -version`) and whether you used the Vector API flag.

## Code of conduct

Be kind and patient. Everyone here is learning.
