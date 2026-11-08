#!/bin/sh
# Compile, test, and run the demos. Needs JDK 17+.
#   - the normal library needs no special flags
#   - src-vector/ holds the optional SIMD kernels (Java's Vector API, an incubator module);
#     if they compile, the tests run twice: once without them, once with them
set -e
mkdir -p out

jc() { # javac, or the compiler module when only a java launcher is installed
  if command -v javac >/dev/null 2>&1; then javac "$@"; else java -m jdk.compiler/com.sun.tools.javac.Main "$@"; fi
}

jc -d out src/minidl/*.java

VEC=0
if [ -d src-vector ] && jc --add-modules jdk.incubator.vector -cp out -d out src-vector/minidl/*.java >/dev/null 2>&1; then VEC=1; fi

echo "--- tests (portable kernels)"
java -Dminidl.vector=false -cp out minidl.Tests
if [ "$VEC" = "1" ]; then
  echo "--- tests (Vector API kernels)"
  java --add-modules jdk.incubator.vector -cp out minidl.Tests 2>&1 | grep -v "^WARNING"
  echo "--- demos (Vector API kernels)"
  java --add-modules jdk.incubator.vector -cp out minidl.Main 2>&1 | grep -v "^WARNING"
else
  echo "(Vector API kernels not available in this Java: skipped)"
  echo "--- demos"
  java -cp out minidl.Main
fi
