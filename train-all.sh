#!/bin/sh
# Downloads the data and trains BOTH models: digits (MNIST) and letters (EMNIST). Needs JDK 17+ and internet.
# Takes roughly 10 to 30 minutes depending on your laptop. Afterwards:
#   ./run-web.sh            digits
#   ./run-web.sh letters    letters A-Z
set -e
mkdir -p out
jc() { if command -v javac >/dev/null 2>&1; then javac "$@"; else java -m jdk.compiler/com.sun.tools.javac.Main "$@"; fi; }
jc -d out src/minidl/*.java
[ -d mnist ]  || ./get-mnist.sh
[ -d emnist ] || ./get-emnist.sh
echo "=== digits 0-9 ===";   java -Xmx1g -cp out minidl.MnistDemo mnist 8 cnn 60000 aug
echo "=== letters A-Z ==="; java -Xmx1g -cp out minidl.EmnistDemo emnist 8 124800 aug
echo "=== digits + letters in ONE model ==="; java -Xmx1g -cp out minidl.AlnumDemo mnist emnist 8 60000 aug
echo "done: mnist-cnn-aug.bin (digits), emnist-letters-cnn.bin (letters), alnum-cnn.bin (both). ./run-web.sh [letters|all]"
