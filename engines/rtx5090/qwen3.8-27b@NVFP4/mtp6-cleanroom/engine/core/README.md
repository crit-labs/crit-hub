# crit Core

Core is the engine foundation. [Forge](../forge/README.md) supplies optional
techniques and [Market](../market/README.md) guides kernel selection. Neither a
technique nor an external kernel provider is required to integrate an engine.

The core owns shared lifecycle and resource infrastructure. Model semantics,
kernels, state layouts and tokenizer behavior belong to engine adaptations.

A complete engine integrates through the public Model/Session/Backend boundary:
declare capabilities and resource needs, advance requests, expose ready progress,
commit accepted output and drain owned work. Its internal execution strategy
stays with the engine. Dense or expert-routed computation, CPU or accelerator
execution, caches and graph capture do not change that common boundary.
CUDA and concurrent requests are opt-in; Forge techniques are optional.
The SDK covers token generation and scoring. Each integration must validate its
model/backend combination; the interface alone does not establish compatibility.
Per-request [sampling, stop and extension options](runtime/README.md#request-options)
are capability-checked and retained through request retirement.

| Part | Responsibility | Source |
| --- | --- | --- |
| [Runtime](runtime/README.md) | Requests, bounded scheduling, readiness, commit/publication, cancellation and retirement | [runtime.cpp](runtime/runtime.cpp) |
| [Memory](memory/README.md) | Resource budgets and reservation accounting | [resources.cpp](memory/resources.cpp) |
| [CUDA backend](backends/cuda/README.md) | Explicit allocation, streams/events and ownership services | [cuda.cpp](backends/cuda/cuda.cpp) |
| Public SDK | Stable headers consumed by model adaptations | [core.h](include/crit/core.h), [cuda.h](include/crit/cuda.h) |
| [Contracts](contracts/README.md) | Current native ownership and request responsibilities | [CONTRACTS.json](contracts/CONTRACTS.json) |

## Build and use

For a small engine built directly from this checkout, start with the
[Core + Forge walkthrough](examples/forge-projection/README.md). It connects a
native quantized projection to a Model and Session, with no installed SDK step.

Use CMake 3.28+ and a C++20 compiler: GCC, Clang or MSVC. CUDA is off by default;
enable `CRIT_CORE_CUDA` with Toolkit 13.1+. MSVC uses `/W4 /WX`; GCC/Clang builds
use `-Wall -Wextra -Wpedantic -Werror`. The optional `CRIT_CORE_SANITIZERS` setting
currently accepts only GCC/Clang driver flags.

```sh
cmake -S core -B build/core -DBUILD_TESTING=ON
cmake --build build/core
ctest --test-dir build/core --output-on-failure
cmake --install build/core --prefix build/core-install
```

For multi-configuration generators such as Visual Studio, append `--config
Release` to build/install commands and `-C Release` to CTest. With `CRIT_CORE_CUDA=ON`,
portable CUDA fixtures run on Windows and Linux and require an accessible device.
Staged-graph and foreign stream ordering fixtures also run on both platforms.
Additional injected allocation/drain/graph failures require linker `--wrap`;
configuration reports when those additional cases are omitted. Host tests alone
do not validate CUDA execution.

Verify the installed public headers and library with independent host consumers:

```sh
cmake -S tests/core/concurrency_consumer -B build/consumer -DCMAKE_PREFIX_PATH=/absolute/path/to/build/core-install
cmake --build build/consumer
ctest --test-dir build/consumer --output-on-failure
```

These consumers exercise concurrent generation and independent text decoders
without accessing private Core headers or linking its sources directly.

The SDK is version 0.2.0. Include `<crit/core.h>` and link the installed target:

```cmake
find_package(critCore 0.2.0 EXACT CONFIG REQUIRED)
target_link_libraries(your_engine PRIVATE crit::core)
```

Supply `build/core-install` through `CMAKE_PREFIX_PATH`. Optional CUDA services
use `<crit/cuda.h>` and `crit::cuda`. See the
[installed consumer example](../tests/core/concurrency_consumer/main.cpp).

Adapters can return an independent `TextFrontend` from `Model::text_frontend()`
for tokenization during generation and private incremental text decoders. Existing
adapters keep their idle-only model callbacks. See the
[text service contract](runtime/README.md#text-services) and
[native fixture](../tests/core/frontend_tests.cpp).

For lifecycle diagnostics, enable `Profile::trace` and read `Engine::trace()`.
The [runtime guide](runtime/README.md#lifecycle-tracing) explains bounded storage,
phase timing and loss counters. Tracing is disabled by default.

Host fixtures cover lifecycle, resource accounting and concurrency. They do not
establish model arithmetic, CUDA correctness or workload performance; qualify
those in your [engine project](../docs/BUILDING_ENGINES.md).
