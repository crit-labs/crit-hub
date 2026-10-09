# Qwen3.8-27B @ NVFP4 · RTX 5090 · mtp6-cleanroom

An engine written from scratch on crit Core 0.2.0 and Forge in one session.
It runs text generation on WSL2 Ubuntu 24.04 with CUDA 13.1 on sm_120.

Model: `unsloth/Qwen3.8-27B-NVFP4`, revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`.
Weights stay external and are read word-for-word from the original Safetensors.

| Frozen 512-input / 512-output case | Prose | Coding | Reasoning |
| --- | ---: | ---: | ---: |
| Resident decode, tok/s (sealed candidate-0005) | 174.8 | 192.0 | 228.6 |
| Tokens per speculative round | 3.76 | 4.16 | 4.97 |

Each figure is the median of five measured repetitions after two warmups. Decoding
is greedy with lowest-ID ties, concurrency is one and EOS is disabled. Decode tok/s
counts all 512 outputs over the time from end of prefill to the last token.

Other measurements:

- **Prefill:** about 1,950 tok/s, with the first token at about 285 ms.
- **Run-to-run variation:** dev runs of the same binary measured about 6% higher
  (187 / 202 / 241). Desktop GPU load on this card causes variation between
  processes.
- **Plain decode** (no speculation): 49–52 tok/s for the first baseline, 64 tok/s with
  the final kernels.

## Checks

These are the gates candidate-0005 passed.

- **Weight words:** a device read-back of all 1,267 uploaded tensors (22.50 GB)
  matches the original file words exactly.
- **NLL:** mean NLL over 339 fixed targets is within +0.05 nats/token of an
  independent CPU reference on every case. Deltas are −0.015 / −0.038 / −0.038 on
  the prefill path and −0.011 / −0.023 / −0.023 on the decode path.
- **Functional:**
  - A repeated request gives identical output.
  - Stop trimming works.
  - Cancellation takes effect within 25 ms.
  - Output after a cancellation or a rejected 4096-token prompt is identical to
    before.
- **Speculative vs ordinary:** speculative output matched ordinary decoding token
  for token on all three cases, comparing the recorded candidate-0001 and
  candidate-0002 runs (reported, not gated).
- **Chat:** tested through crit's own resident chat host with consecutive prompts,
  switching candidate-0005 → candidate-0004 → candidate-0005, New chat and unload.

## Design

- **Weights:** all weights stay resident in their stored encodings: NVFP4
  (E2M1 + E4M3 group-16 scales + FP32 divisor), row-scaled FP8 and BF16.
- **GEMV:** decode projections use a tensor-core `mma.sync` GEMV. Dequantization
  happens in registers, two 16-row tiles share each activation fragment, and the
  SwiGLU epilogue is fused. Narrow residual projections split K into partials,
  which the next RMSNorm sums in a fixed order. All kernels use programmatic
  dependent launch.
- **Gated DeltaNet:** decode uses one fused kernel per value head with the gated
  norm in its epilogue. Recurrent and conv state for the accepted prefix are
  replayed at the start of the next round from per-column stashes.
- **Speculation:** the Forge device round runs draft, verify, greedy accept and
  frontier advance in two CUDA graphs with one readback per round. The drafter is
  the checkpoint's own MTP layer, with K = 6 and a derived per-row FP8 copy. Its
  head covers the first 98,304 head rows plus the chat control tokens. Verify
  always uses the full original head.
- **Prefill:** exact BF16 dequantization followed by a cuBLAS BF16 GEMM.

Limits: one sequence, greedy decoding, text only and a 4096-token context.
Each engine instance loads its own weights; they are not shared with crit's model
pool across candidate switches.

## Build and run

`engine/` holds:

- `core/`: crit Core 0.2.0, unmodified.
- `src/`: the engine, including two unmodified Forge `device-speculative-round`
  files under `src/forge/`.
- `python/chat_adapter.py`: crit resident chat.
- `python/chat.py`: a standalone terminal chat. It was added at export and is not part of the sealed candidate source. `scripts/make_package.sh` also gained a fallback that finds `THIRD_PARTY.md` one level up in this layout.
- `scripts/`: build and package scripts.

Builds need CMake 3.28+, Ninja, GCC 13 and the CUDA 13.1 toolkit. Python 3.12
needs the pinned packages in `engine/requirements.txt`, which the package script
vendors.

```sh
pip install --target /path/to/vendor -r engine/requirements.txt
bash engine/scripts/build.sh engine /path/to/build
bash engine/scripts/make_package.sh engine /path/to/build /path/to/vendor /path/to/package
cp engine/configuration.json /path/to/package/
python engine/python/chat.py --model /path/to/model --package /path/to/package
```

`/new` starts a fresh conversation and `/quit` releases the engine. The sealed
candidate also opens in crit's terminal:
`crit --workspace <crit>/runs/qwen38-cleanroom-20261009 --project <crit>`.

Credits and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES.txt](LICENSES.txt).
