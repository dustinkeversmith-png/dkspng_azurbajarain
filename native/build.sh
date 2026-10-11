#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m pip install pybind11
suffix="$(python3-config --extension-suffix)"
includes="$(python3 -m pybind11 --includes)"
g++ -O3 -Wall -shared -std=c++17 -fPIC -pthread $includes \
    "$root/native/processor.cpp" \
    -o "$root/tools/native_engine$suffix"
g++ -O3 -Wall -shared -std=c++17 -fPIC -pthread $includes \
    "$root/native/tensor_engine.cpp" \
    -o "$root/tools/tensor_engine$suffix"
