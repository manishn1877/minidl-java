# Contributing to MiniDL

Thanks for helping!

1. Fork the repo and create a branch.
2. Build: bash build-jar.sh
3. Run the tests: java -jar dist/minidl-1.4.0.jar tests (all 283 must pass)
4. Add a test for anything new (new layers need a gradient check).
5. Keep it pure Java, no outside ML libraries.
6. Open a pull request and say what you changed and why.

Look for issues labelled "good first issue".
