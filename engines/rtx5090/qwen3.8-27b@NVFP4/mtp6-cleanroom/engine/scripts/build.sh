#!/bin/bash
# Configure and build the engine (WSL2 Ubuntu 24.04, CUDA 13.1, CMake 3.28+, Ninja, GCC 13).
#   scripts/build.sh ENGINE_DIR BUILD_DIR
set -euo pipefail
SRC=${1:?engine dir}; BUILD=${2:?build dir}
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 >/dev/null
cmake --build "$BUILD" 2>&1 | grep -v "^\[" || true
test -x "$BUILD/q38bench" && test -f "$BUILD/libq38.so"
