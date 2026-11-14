#!/bin/sh
# Handwriting recognition web app. Needs JDK 17+ and trained models.
#   ./run-web.sh                loads EVERY trained model it finds (emnist-full-cnn.bin, emnist-case-cnn.bin, alnum-cnn.bin, emnist-letters-cnn.bin, mnist-cnn-aug.bin)
#                               and the page lets you switch between them. It opens with digits+letters if you trained it.
#   ./run-web.sh digits         only the digits model        ./run-web.sh letters   only letters
#   ./run-web.sh all            only the digits+letters model (stops with a message if it is not trained yet)
#   ./run-web.sh auto 9000      different port
# If ./mnist and ./emnist exist, the page also shows the test accuracy and corrections are mixed with real pictures.
set -e
mkdir -p out
jc() { if command -v javac >/dev/null 2>&1; then javac "$@"; else java -m jdk.compiler/com.sun.tools.javac.Main "$@"; fi; }
jc -d out src/minidl/*.java
case "$1" in
  all)     MODEL=alnum-cnn.bin ;;
  letters) MODEL=emnist-letters-cnn.bin ;;
  case)    MODEL=emnist-case-cnn.bin ;;
  full)    MODEL=emnist-full-cnn.bin ;;
  digits)  MODEL=mnist-cnn.bin; for f in mnist-cnn-aug.bin mnist-cnn.bin; do [ -f "$f" ] && MODEL="$f" && break; done ;;
  *)       MODEL=auto ;;
esac
exec java -Xmx1g -cp out minidl.HandwritingServer "$MODEL" "${2:-8080}" .
