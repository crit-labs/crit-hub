# fnl: crit engine for nvidia/Qwen3.8-Flash-Next-FP4-Local

[crit-hub](../../../../README.md) › [engines](../../../README.md) › [mtp3-expert-cache](../README.md) › engine

A native CUDA engine on crit Core for the `qwen4_exp` text model of
`nvidia/Qwen3.8-Flash-Next-FP4-Local` (48 layers: 36 gated-delta linear-attention and 12
QSA-indexed attention layers, 4-stream gated-residual hyper-connections, a PLE n-gram
embedding at layer 2, 512 routed experts top-10 plus a shared expert). Written for one
RTX 5090 (sm_120a) with 82 GiB host RAM under WSL2.

## Weights (used as stored)

| Part | Stored format | Where | Arithmetic |
| --- | --- | --- | --- |
| Attention, linear-attention, shared-expert projections | MXFP8 (E4M3, UE8M0 per 32) | device | decode: FP32 activations; prompt: FP16 tensor cores on exactly converted tiles |
| Routed experts (24,576) | NVFP4 (E2M1, E4M3 per 16, FP32 global); block scales in the checkpoint's lossless CSF encoding | pinned host records (CSF kept compressed), device slot cache | CSF decoded on the device when an expert is copied or read |
| Hyper-connections, router, PLE projections, output head | BF16 | device | decode FP32 activations; prompt FP16 |
| Token embeddings | BF16 | pinned host, read through the device mapping | |
| PLE n-gram table (320M rows) | NVFP4 | memory-mapped checkpoint (no read-ahead) | rows gathered on the host per token |

CSF decoding follows flashinfer's b12x `nvfp4_csf` byte-window codec (Apache-2.0): per 16-row
slab, 16 base bytes and 4-bit offsets; U32 exception words override single scales.

## Execution

- **Decode pass (1 to 4 tokens):** FP32 activations end to end. MXFP8/BF16 GEMVs read each weight
  once for every token of the pass. Routed experts of a pass are the union of its tokens' routes,
  resolved by a device-side directory (CLOCK replacement, slots touched in the pass protected): a
  miss is read straight from mapped host memory by the expert GEMV, which writes it through into
  its new slot; its scales are decoded from the host CSF planes. Gated delta rule as a sequential
  FP32 recurrence that leaves the state after each token in a ring of four, so a pass can keep any
  prefix of its tokens; conv and PLE states are committed for the kept tokens. QSA: FP32 index
  scores over pooled block keys, radix top-512 when more than 512 blocks are complete, split-K
  attention over the selected blocks and the open block. Each pass runs as CUDA graphs (head with
  layer 0, then the rest; a speculative round's drafter steps in a third) while the host gathers
  the PLE n-gram rows with a thread pool.
  Weights are decoded once for all tokens of a pass. Dense GEMVs: a warp takes 1 to 4 rows and
  issues the next row's loads before the current row's arithmetic, against activations staged once
  per block in shared memory with a bank-conflict-free swizzle (wide rows with more than two tokens
  read them through L1). Expert down projection: blocks per (entry, 64 rows) with the entry's
  activations staged, then a sum over the entries in plan order. The kernels retain the per-lane accumulation order of the
  one-row versions; whole-engine token equivalence is measured separately. The pass's
  expert union is built by all threads of the plan block; missed experts' CSF scales are decoded
  by 8 blocks per matrix.
- **Speculative decoding (greedy requests):** the checkpoint's MTP block, used as released, drafts
  3 tokens per round: fused input (fc_embedding of the normed token embedding plus a shared
  fc_hidden over each normed hidden stream), one QSA attention layer with its own KV cache and
  index, 512 routed NVFP4 experts kept on the device, final mixer, and a drafting head re-encoded
  as MXFP8 (the verified pass uses the BF16 head). A round verifies the input token and the 3
  drafts in one pass, keeps the drafts that match the model's argmax plus the model's next token,
  commits the recurrent states for the kept tokens, runs the drafter's extend pass over the kept
  positions and chains the next drafts, all on the device in one graph. Exactness is measured, not assumed: candidate-0006 matches plain greedy
  in 3/6 Eval cases; FP8 KV and execution-path differences remain subject to the quality budget. The device reports the kept count and each draft as it is
  known (mapped memory), so the host gathers the next round's n-gram rows while the drafter runs.
  Near the end of the context the last tokens decode one per pass. Sampled requests decode one
  token per pass.
- **Prompt (chunks of `prefill_chunk` tokens):** FP16 tensor-core GEMMs (mma.sync) with weight
  tiles converted in shared memory; routed experts grouped by expert after a host-planned
  residency pass (DMA copies of missing experts; for long chunks the next layer's experts are
  prefetched while the current layer computes). Every surviving prefetched slot has its scales
  initialized, including experts not routed by the prompt, before the directory becomes usable
  by decoding. Pending scale-job destinations are protected from eviction. Index scores computed near FP32 precision through a
  split-FP16 GEMM; gathered tensor-core attention per query. For speculative requests the MTP
  block's KV cache and index keys are filled for the chunk's positions. The n-gram rows of a chunk
  are gathered by up to 48 threads (page faults on a cold table). A layer's expert copies go out as
  one `cudaMemcpyBatchAsync`. In chunks under 512 tokens, missing experts routed by at most 4 tokens
  are copied into a transient pool in the unused tail of the prompt workspace instead of the expert
  cache, so a short prompt does not evict the experts decoding keeps using (the prompt GEMMs read the
  same bytes either way, so outputs do not depend on it). Optional, off by default
  (`FNL_PROMPT_CPU_THREADS=6`): chunks under 512 tokens hand about 43% of their missing experts
  (routed by at most 4 tokens) to an AVX-512 pool while the GPU copies the rest; it cuts short
  prompts' time to first token by about a third but the split depends on the cache state, so
  outputs are not repeatable.
- **Memory:** FP8 (E4M3) K/V for the 12 attention layers and the MTP layer, with one FP32 scale per
  256-dim key or value vector (amax / 448), converted to FP32 in decode attention and to FP16 tiles
  in prompt attention; FP32 pooled index keys; sized for the required context at load. KV
  positions a request cannot reach and the prompt workspace between prompt passes are lent to the
  expert cache, which takes all device memory but `vram_margin_gib` (2 GiB in candidate-0006) at load.
- **Scoring paths:** prefill, decode (one-token passes) and verify (4-token passes of the request's
  own tokens, the passes that emit speculative tokens).
- **Eval facts:** `draft_bytes_per_step` (MTP block, one token's routed MTP experts, MXFP8 drafting
  head) and, per request, device time in target passes (`verify_ms`) and in drafter steps
  (`draft_ms`) from CUDA events around the pass and drafter graphs.
- **CPU expert computation** (`FNL_CPU_SHARE`, off by default): an AVX-512 pool can compute a share
  of each pass's missed experts from their host records through a mailbox in mapped memory, the
  device filling their slots in the background. On this PC it is slower than device reads of the
  misses (memory contention halves the CPU's rate), so it stays off.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | CUDA/C++ engine: `model.*` (weights, decode path), `prefill_net.cu` (prompt path), kernels, `engine.cu` (crit Core Model/Session/Backend and the C API) |
| `python/` | `fnl_native.py` (ctypes), `fnl_eval_adapter.py` (crit Eval eval-v1), `fnl_serve.py` (crit serve resident adapter), `chat.py` (standalone terminal/prompt-file launcher) |
| `tools/` | Native checks and microbenchmarks; full reference/evaluation records are retained separately from this source export |

## Build

Use the complete SDK, Python-environment and standalone-chat commands in [the engine entry](../README.md#build-and-chat). The native runtime uses Core 0.1.0; Python serving/text handling uses crit 0.2.0.
