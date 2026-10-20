#!/bin/bash
# Build kestrel from this source tree and assemble a runnable package.
#   build.sh SOURCE_DIR BUILD_DIR PACKAGE_DIR CORE_PREFIX
# CORE_PREFIX: an installed crit Core SDK (CRIT_CORE_CUDA=ON, position-independent code), see CORE.md.
set -euo pipefail
SRC=$(realpath "$1"); BUILD=$2; PKG=$3; CORE=$(realpath "$4")
NVCC=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CORE" \
      -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_CUDA_COMPILER="$NVCC"
cmake --build "$BUILD"
mkdir -p "$PKG"
cp "$BUILD/libkestrel.so" "$PKG/"
cp "$SRC"/python/*.py "$SRC"/python/*.txt "$PKG/"
cp "$SRC/NOTICE.md" "$SRC/CORE.md" "$PKG/"
