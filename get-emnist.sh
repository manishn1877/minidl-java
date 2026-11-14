#!/bin/sh
# Downloads the 4 EMNIST LETTERS files into ./emnist  (about 35 MB).
# The official source is https://www.nist.gov/itl/products-and-services/emnist-dataset (one big zip, "gzip.zip").
# This script uses a Hugging Face copy of the same files; if the download fails, take the 4
# emnist-letters-*-idx*-ubyte.gz files from the official zip and put them in ./emnist
set -e
mkdir -p emnist
for f in emnist-letters-train-images-idx3-ubyte emnist-letters-train-labels-idx1-ubyte emnist-letters-test-images-idx3-ubyte emnist-letters-test-labels-idx1-ubyte; do
  curl -fL -o "emnist/$f.gz" "https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main/$f.gz"
done
# The "byclass" files (62 classes: digits, capital AND small letters) for the A-Z a-z models. Same mirror; this part is optional and was not
# checked when this script was written: if it fails, take emnist-byclass-*-idx*-ubyte.gz from the official zip (about 560 MB for all splits).
for f in emnist-byclass-train-images-idx3-ubyte emnist-byclass-train-labels-idx1-ubyte emnist-byclass-test-images-idx3-ubyte emnist-byclass-test-labels-idx1-ubyte; do
  curl -fL -o "emnist/$f.gz" "https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main/$f.gz" || echo "could not download $f (letters-only models still work)"
done
echo "done. now run:  java -Xmx512m -cp out minidl.EmnistDemo emnist 8 124800 aug"
