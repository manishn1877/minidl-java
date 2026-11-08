#!/bin/sh
# Builds dist/minidl-<version>.jar (+ -sources.jar and -javadoc.jar). Needs JDK 17+, nothing else.
#   java -jar dist/minidl-1.0.0.jar help
set -e
VERSION=$(sed -n 's/.*VERSION = "\(.*\)";.*/\1/p' src/minidl/MiniDL.java)
rm -rf out dist; mkdir -p out dist
jc() { if command -v javac >/dev/null 2>&1; then javac "$@"; else java -m jdk.compiler/com.sun.tools.javac.Main "$@"; fi; }
jr() { if command -v jar >/dev/null 2>&1; then jar "$@"; else java -m jdk.jartool/sun.tools.jar.Main "$@"; fi; }
jd() { if command -v javadoc >/dev/null 2>&1; then javadoc "$@"; else java -m jdk.javadoc/jdk.javadoc.internal.tool.Main "$@"; fi; }
jc -Xlint:all --release 17 -d out src/minidl/*.java 2>/dev/null || jc -Xlint:all -d out src/minidl/*.java   # --release 17 makes sure it runs on Java 17; older JDK installs without ct.sym fall back
mkdir -p out/web && cp web/index.html out/web/
printf 'Main-Class: minidl.MiniDL\nImplementation-Title: MiniDL\nImplementation-Version: %s\n' "$VERSION" > out/MANIFEST.MF
jr --create --file "dist/minidl-$VERSION.jar" --manifest out/MANIFEST.MF -C out minidl -C out web
jr --create --file "dist/minidl-$VERSION-sources.jar" -C src minidl
mkdir -p out/javadoc && jd -quiet -Xdoclint:all,-missing -d out/javadoc -sourcepath src -subpackages minidl
jr --create --file "dist/minidl-$VERSION-javadoc.jar" -C out/javadoc .
ls -l dist
