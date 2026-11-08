#!/bin/sh
# Handwriting recognition web app. Needs JDK 17+ and a trained model.
#   ./run-web.sh                digits, using mnist-cnn-aug.bin or mnist-cnn.bin (made by MnistDemo)
#   ./run-web.sh letters        letters A-Z, using emnist-letters-cnn.bin (made by EmnistDemo)
#   ./run-web.sh all            digits AND letters in one model, using alnum-cnn.bin (made by AlnumDemo)
#   ./run-web.sh digits 9000    different port
# If ./mnist (or ./emnist) exists, the page also shows test accuracy and corrections are mixed with real data.
set -e
mkdir -p out
jc() { if command -v javac >/dev/null 2>&1; then javac "$@"; else java -m jdk.compiler/com.sun.tools.javac.Main "$@"; fi; }
jc -d out src/minidl/*.java
if [ "$1" = "all" ]; then MODEL=alnum-cnn.bin; DATA=.; elif [ "$1" = "letters" ]; then MODEL=emnist-letters-cnn.bin; DATA=emnist; else
  MODEL=mnist-cnn.bin; for f in mnist-cnn-aug.bin mnist-cnn.bin; do [ -f "$f" ] && MODEL="$f" && break; done; DATA=mnist; fi
[ -d "$DATA" ] || DATA=""
[ "$1" = "all" ] && { [ -d mnist ] && [ -d emnist ] || DATA=""; }
exec java -Xmx512m -cp out minidl.HandwritingServer "$MODEL" "${2:-8080}" $DATA
