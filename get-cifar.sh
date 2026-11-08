#!/bin/sh
# Gets CIFAR-10 (60,000 colour photos, about 163 MB as binary) into ./cifar.
# The official site may be unreachable, so this uses the copy inside the npm package "tfjs-cifar10"
# (PNG sheets + label files) and converts it to the official binary files. Run ./run.sh first (it builds ./out).
set -e
mkdir -p cifar-src cifar
curl -L -o cifar-src/pkg.tgz https://registry.npmjs.org/tfjs-cifar10/-/tfjs-cifar10-1.1.1.tgz
tar xzf cifar-src/pkg.tgz -C cifar-src
java -Xmx1500m -cp out minidl.CifarConvert cifar-src/package cifar
rm -rf cifar-src
echo "done. now run:  java -Xmx512m -cp out minidl.CifarDemo cifar 8 aug"
