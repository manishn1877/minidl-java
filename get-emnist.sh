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
echo "done. now run:  java -Xmx512m -cp out minidl.EmnistDemo emnist 8 124800 aug"
