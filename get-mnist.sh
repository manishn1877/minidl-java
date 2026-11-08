#!/bin/sh
# Downloads the 4 MNIST files into ./mnist  (about 11 MB)
set -e
mkdir -p mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -L -o "mnist/$f.gz" "https://raw.githubusercontent.com/fgnt/mnist/master/$f.gz"
done
echo "done. now run:  java -Xmx256m -cp out minidl.MnistDemo mnist 8 cnn"
