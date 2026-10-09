# Connect Forge arithmetic to a Core engine

This example connects the native [grouped-integer projection](../../../forge/quantized-projections/code/grouped_int.h)
to Core's Model/Session/Backend interfaces. It builds from this checkout with
CMake 3.28+ and a C++20 compiler, and runs on the CPU.

The model has four tokens, `A` through `D`, and a hand-authored weight matrix that
predicts the next letter cyclically. It uses actual signed-int4 G32 encoding and
projection from Forge. Its purpose is to show the integration; it is a tiny
teaching model, with no pretrained weights or LLM benchmark results.

From the repository root:

```sh
cmake -S core/examples/forge-projection -B build/core-forge -DCMAKE_BUILD_TYPE=Release
cmake --build build/core-forge --config Release
ctest --test-dir build/core-forge -C Release --output-on-failure
```

Run `build/core-forge/crit_core_forge_projection` directly to see the output.
With Visual Studio, run `build/core-forge/Release/crit_core_forge_projection.exe`.

```text
ABCDAB -> CDABCD
D -> ABC
```

## Follow the integration

Read [main.cpp](main.cpp) in this order:

1. **`ProjectionModel` owns the representation.** `make_weights()` uses Forge's
   `quantize()` to produce packed codes and scales. The immutable `PackedRows`
   stays alive for every session; the model reserves its weight payload in Core's
   resource ledger.
2. **`ProjectionSession` owns request state.** Prefill consumes the prompt within
   Core's chunk budget. During decode, it constructs the activation vector and
   calls the native Forge implementation:

   ```cpp
   const auto logits = crit::quant::project(weights_.view(), activations, 1);
   pending_token_ = static_cast<Token>(std::max_element(logits.begin(), logits.end()) - logits.begin());
   ```

   The model-specific choice is greedy next-token selection. This example uses
   Forge's C++ implementation directly; a Python-only reference would need a
   native adaptation or a supplied backend for a C++ engine.
3. **Progress remains provisional until `commit()`.** The session returns a
   borrowed view of `pending_token_`. Core reads that view, decides how much to
   accept, and calls `commit()` to advance the session's private state. Core then
   publishes the accepted output. `close()` releases the prompt view and reservation.
4. **`HostBackend` establishes completion.** The projection is synchronous, so
   no readiness object is necessary and `drain()` has no queued work to settle.
   An asynchronous backend must supply its actual readiness and drain behavior.

The test exercises several prefill chunks, multiple decode/commit rounds and a
second request with new private state and the same retained weights. A real
engine supplies its own model layers, encoded weights, tokenizer, sampling and
backend while using the same Core boundary.
