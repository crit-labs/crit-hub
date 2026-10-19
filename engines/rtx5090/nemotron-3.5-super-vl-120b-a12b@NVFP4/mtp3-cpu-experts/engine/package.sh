#!/bin/bash
# Build the engine and assemble its runnable package: package.sh SOURCE_DIR BUILD_DIR PACKAGE_DIR CORE_PREFIX
set -e
SRC=$1; BUILD=$2; PKG=$3; CORE=$4
export PATH=/usr/local/cuda-13.1/bin:$PATH
mkdir -p "$BUILD" "$PKG"
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CORE"
cmake --build "$BUILD"
cp "$BUILD/libnmh.so" "$SRC/python/eval_adapter.py" "$SRC/python/serve_adapter.py" "$SRC/NOTICE.md" "$PKG/"
cp "$SRC/third_party/kernels/cutlass/artifacts/LICENSE.txt" "$PKG/CUTLASS-LICENSE.txt"
cp "$SRC/third_party/kernels/cutlass/dependency.json" "$PKG/cutlass-dependency.json"
