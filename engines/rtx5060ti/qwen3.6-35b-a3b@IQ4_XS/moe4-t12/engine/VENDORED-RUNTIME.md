# Vendored runtime (fetched, not tracked)

This engine needs a prebuilt llama.cpp CUDA backend at run time. The
binary payload is **not** tracked in this repository: the contribution
rules keep binaries outside it, and several of these files exceed
GitHub's 100 MB per-file limit, so a push carrying them is rejected.
Fetch the bytes below into `engine/vendor/llamacpp/b11463/` and verify them
before running the engine.

## Upstream

| Asset | Source | Extract into |
| --- | --- | --- |
| `llama-b11463-bin-win-cuda-13.4-x64.zip` | https://github.com/ggml-org/llama.cpp/releases (build b11463, commit d2a79e604) | `engine/vendor/llamacpp/b11463/bin/` |
| `cudart-llama-bin-win-cuda-13.4-x64.zip` | same release | `engine/vendor/llamacpp/b11463/runtime/` |

Both archives are extracted unchanged. `29` files, 0.69 GB in total.

Files above the 100 MB limit: `bin/ggml-cuda.dll` (141 MB), `runtime/cublasLt64_13.dll` (470 MB).

## Required files and identities

| Path under `engine/vendor/llamacpp/b11463/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `bin/LICENSE-LLVM-OpenMP` | 19741 | `fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4` |
| `bin/ggml-base.dll` | 801792 | `a7d0183165831ff87f073373192e2ae82237d996efcb732d42569d7bc1fb386e` |
| `bin/ggml-cpu-alderlake.dll` | 1435136 | `3cf66ecff8f2a18d64aec0058e537b3f4797655815213a310701cde9752f5fff` |
| `bin/ggml-cpu-cannonlake.dll` | 1614848 | `d94490e4d95c1e494d5c8bd579758c4583f8d8029e2aaa22f472425e39b93ef4` |
| `bin/ggml-cpu-cascadelake.dll` | 1599488 | `2cd37fe8f8a8895b7f8753a9a99e78201b27eeeeb1b436fd0d2d477c54cd4a9e` |
| `bin/ggml-cpu-cooperlake.dll` | 1609216 | `7841c0c83aacb7d99a25a2ff934e597c9917b627a4e368ddfc2deebefcd010af` |
| `bin/ggml-cpu-haswell.dll` | 1439744 | `d4eb5175043e8abba0732bf9bf30c54980a18fa5df6543fb1d1c3a6c076eb2c1` |
| `bin/ggml-cpu-icelake.dll` | 1606656 | `1bcd282e751fcbb401b3a54341e0d6d1f51b119333d4c310ce4a7ba6fa96f35f` |
| `bin/ggml-cpu-ivybridge.dll` | 1315328 | `fccf4da5977812ba63cb7ef24cdc70b7318b1f951882aea0fa642b99512eda42` |
| `bin/ggml-cpu-piledriver.dll` | 1308160 | `4e923cb87755e8bc86307e52bd8092a40cdd984f5fd1aba4388f4793a3875c4c` |
| `bin/ggml-cpu-sandybridge.dll` | 1290240 | `2cc503373dca552935a8fb2c8b639dbf82172649f6cd40899c495d77131d13c5` |
| `bin/ggml-cpu-sapphirerapids.dll` | 1886208 | `646f2732d4b410ae102ab90cb320002c030eff55bc32b12866abf4369492295d` |
| `bin/ggml-cpu-skylakex.dll` | 1608704 | `8f27ed8d33fe78e27155f184832754ac659d625f314f28a36ef220b9eb9c3461` |
| `bin/ggml-cpu-sse42.dll` | 915968 | `d4dcb3cf30be124849f392bcd9c8c7880b08426b3a78d7f047a304f701afbbb6` |
| `bin/ggml-cpu-x64.dll` | 908800 | `3fe9f33c54f324fafde44d7d8bf2792ac32359df7daf93e9c83dda9ca1eeb9a2` |
| `bin/ggml-cpu-zen4.dll` | 1615872 | `63cda17b9d13617ab7461798547184f869b10252744068426288cdac61cccbe5` |
| `bin/ggml-cuda.dll` | 147823616 | `16ecb592d3f00a51d8375612312b84014114e5cc411ae4e07e694de5c590121f` |
| `bin/ggml-rpc.dll` | 204800 | `e546ee27c18082ec99429592ad871c76f4fcb16d8d6e38ff09cfb19f2c7b049d` |
| `bin/ggml.dll` | 79872 | `28ccff149a3503cb8ee02bb6dd21557748c00526e0d6d7683e79b427175a0fdd` |
| `bin/libomp.dll` | 768000 | `a12116ba72d1d6820407cf30be23da04ce79d6bb8a71a5ee71759c5a1faa6f1c` |
| `bin/llama-cli-impl.dll` | 3315712 | `e19a03de99ff68a020e2a3601f1a0982d756b6563e85f3f078752a4216189510` |
| `bin/llama-common.dll` | 7969280 | `a976d22cb75d1eb375e954c4f0ddb9ace5cb4f43ec00b35c5a0bb048b8fb812c` |
| `bin/llama-server-impl.dll` | 9170944 | `1eaa4b2ebc45eac4e276941c968d1bff441d608468462e365f58201bb1baa493` |
| `bin/llama-server.exe` | 9216 | `500974e4928ddab9cf47e4896ed31b7b29c2650392b81552656a94bbad6ab31c` |
| `bin/llama.dll` | 3337728 | `649262f4c462f6a9be9a7ea2303516af2ea75da3a305315f386ad46ad2c1a7ec` |
| `bin/mtmd.dll` | 1796096 | `473e4a17eab07c45219586e4570e3fa7bf30f17611b911850eb875fcd130b320` |
| `runtime/cublas64_13.dll` | 54942320 | `1119dbca0a808e0c8850bb4221e330daf0b5ddb348d62d37b6ac533854723df9` |
| `runtime/cublasLt64_13.dll` | 492752496 | `0f5bc315fef706b5626248ebd74e8a48a9e82fa2f0cad38330f5f70984913107` |
| `runtime/cudart64_13.dll` | 551024 | `05bfafcb97bd53b0089568a96e1e5bd6921637bd0853ff7f5342bd31a6a16890` |

## Verify

Run from the engine folder. Any mismatch means the runtime is not the
build this engine was measured with.

```python
import hashlib, pathlib, sys
ROOT = pathlib.Path("vendor/llamacpp/b11463")
EXPECTED = {
    "bin/LICENSE-LLVM-OpenMP": "fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4",
    "bin/ggml-base.dll": "a7d0183165831ff87f073373192e2ae82237d996efcb732d42569d7bc1fb386e",
    "bin/ggml-cpu-alderlake.dll": "3cf66ecff8f2a18d64aec0058e537b3f4797655815213a310701cde9752f5fff",
    "bin/ggml-cpu-cannonlake.dll": "d94490e4d95c1e494d5c8bd579758c4583f8d8029e2aaa22f472425e39b93ef4",
    "bin/ggml-cpu-cascadelake.dll": "2cd37fe8f8a8895b7f8753a9a99e78201b27eeeeb1b436fd0d2d477c54cd4a9e",
    "bin/ggml-cpu-cooperlake.dll": "7841c0c83aacb7d99a25a2ff934e597c9917b627a4e368ddfc2deebefcd010af",
    "bin/ggml-cpu-haswell.dll": "d4eb5175043e8abba0732bf9bf30c54980a18fa5df6543fb1d1c3a6c076eb2c1",
    "bin/ggml-cpu-icelake.dll": "1bcd282e751fcbb401b3a54341e0d6d1f51b119333d4c310ce4a7ba6fa96f35f",
    "bin/ggml-cpu-ivybridge.dll": "fccf4da5977812ba63cb7ef24cdc70b7318b1f951882aea0fa642b99512eda42",
    "bin/ggml-cpu-piledriver.dll": "4e923cb87755e8bc86307e52bd8092a40cdd984f5fd1aba4388f4793a3875c4c",
    "bin/ggml-cpu-sandybridge.dll": "2cc503373dca552935a8fb2c8b639dbf82172649f6cd40899c495d77131d13c5",
    "bin/ggml-cpu-sapphirerapids.dll": "646f2732d4b410ae102ab90cb320002c030eff55bc32b12866abf4369492295d",
    "bin/ggml-cpu-skylakex.dll": "8f27ed8d33fe78e27155f184832754ac659d625f314f28a36ef220b9eb9c3461",
    "bin/ggml-cpu-sse42.dll": "d4dcb3cf30be124849f392bcd9c8c7880b08426b3a78d7f047a304f701afbbb6",
    "bin/ggml-cpu-x64.dll": "3fe9f33c54f324fafde44d7d8bf2792ac32359df7daf93e9c83dda9ca1eeb9a2",
    "bin/ggml-cpu-zen4.dll": "63cda17b9d13617ab7461798547184f869b10252744068426288cdac61cccbe5",
    "bin/ggml-cuda.dll": "16ecb592d3f00a51d8375612312b84014114e5cc411ae4e07e694de5c590121f",
    "bin/ggml-rpc.dll": "e546ee27c18082ec99429592ad871c76f4fcb16d8d6e38ff09cfb19f2c7b049d",
    "bin/ggml.dll": "28ccff149a3503cb8ee02bb6dd21557748c00526e0d6d7683e79b427175a0fdd",
    "bin/libomp.dll": "a12116ba72d1d6820407cf30be23da04ce79d6bb8a71a5ee71759c5a1faa6f1c",
    "bin/llama-cli-impl.dll": "e19a03de99ff68a020e2a3601f1a0982d756b6563e85f3f078752a4216189510",
    "bin/llama-common.dll": "a976d22cb75d1eb375e954c4f0ddb9ace5cb4f43ec00b35c5a0bb048b8fb812c",
    "bin/llama-server-impl.dll": "1eaa4b2ebc45eac4e276941c968d1bff441d608468462e365f58201bb1baa493",
    "bin/llama-server.exe": "500974e4928ddab9cf47e4896ed31b7b29c2650392b81552656a94bbad6ab31c",
    "bin/llama.dll": "649262f4c462f6a9be9a7ea2303516af2ea75da3a305315f386ad46ad2c1a7ec",
    "bin/mtmd.dll": "473e4a17eab07c45219586e4570e3fa7bf30f17611b911850eb875fcd130b320",
    "runtime/cublas64_13.dll": "1119dbca0a808e0c8850bb4221e330daf0b5ddb348d62d37b6ac533854723df9",
    "runtime/cublasLt64_13.dll": "0f5bc315fef706b5626248ebd74e8a48a9e82fa2f0cad38330f5f70984913107",
    "runtime/cudart64_13.dll": "05bfafcb97bd53b0089568a96e1e5bd6921637bd0853ff7f5342bd31a6a16890",
}
bad = []
for name, want in EXPECTED.items():
    p = ROOT / name
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    if got != want:
        bad.append((name, got))
print("OK: %d files match" % len(EXPECTED) if not bad else "MISMATCH: %r" % (bad,))
sys.exit(1 if bad else 0)
```

## Licences

- llama.cpp is MIT; the upstream `LICENSE` ships inside the release archive.
- `libomp.dll` and `LICENSE-LLVM-OpenMP` are under Apache-2.0 with LLVM exceptions.
- `cublas*_13.dll` and `cudart64_13.dll` are NVIDIA redistributables under the CUDA Toolkit EULA.
- Full notices: [THIRD_PARTY.md](../../THIRD_PARTY.md) and [LICENSES.txt](../../LICENSES.txt).
