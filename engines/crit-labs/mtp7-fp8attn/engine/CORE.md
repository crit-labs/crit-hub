# Core build inputs

- crit revision `bdaae9af7e452a58414f1b40f06b251ad9990ca9`
  (crit 0.1.0 plus the model reconnaissance step).
- Core SDK 0.1.0 built and installed with:

```sh
cmake -S $CRIT/core -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRIT_CORE_CUDA=ON -DBUILD_TESTING=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build/core && cmake --install build/core --prefix core-install
```

- Toolchain: CUDA 13.1.115 (`/usr/local/cuda-13.1`), GCC 13.3.0, CMake 3.28.3, Ninja 1.11.1, Ubuntu 24.04 (WSL2).
- Runtime: driver 596.49 (CUDA 13.2 driver API), shared `libcudart.so.13` and `libcublasLt.so.13` from CUDA 13.1.
