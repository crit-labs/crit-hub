# Core build inputs

Added at export (the sealed CORE.md names the run machine's checkout path); `build.sh` copies this file into the
package.

- crit Core 0.2.1 (`v0.2.1`; the run used a later checkout with the same Core C++ sources plus config-reader
  fixes), built and installed with:

```sh
cmake -S crit/core -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRIT_CORE_CUDA=ON -DBUILD_TESTING=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build/core && cmake --install build/core --prefix core-install
```

- Toolchain: CUDA 13.1, GCC 13.3.0, CMake 3.28.3, Ninja, Ubuntu 24.04 (WSL2).
- Runtime: shared `libcudart.so.13` and `libcublasLt.so.13` from CUDA 13.1.
