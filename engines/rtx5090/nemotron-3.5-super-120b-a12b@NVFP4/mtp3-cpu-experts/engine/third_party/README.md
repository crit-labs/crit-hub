# Third-party kernels

`kernels/cutlass/` holds the NVIDIA CUTLASS v4.7.1 headers (git `cb4247394dd82148787aed73e5dc7cef33cbf862`,
BSD-3-Clause) that `src/fp4_grouped.cu` compiles into `libnmh.so` for the SM120 block-scaled grouped
NVFP4 prompt GEMMs. They were retained with `crit kernels retain`:

| Path | What it is |
| --- | --- |
| `kernels/cutlass/artifacts/` | The retained headers and CUTLASS's `LICENSE.txt`, unchanged |
| `kernels/cutlass/request.json`, `catalog.json`, `assessment.json`, `dependency.json` | crit's record of the request, the catalog entry, its assessment and the pinned dependency |

`package.sh` copies the license and the dependency record into the package. Credits for every
component are in the engine's [THIRD_PARTY.md](../../THIRD_PARTY.md).
