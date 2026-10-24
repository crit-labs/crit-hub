<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# dflash2-int8kv

> [!WARNING]
> **Contributed by** giveen · [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| GPU · model | Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rtx5090 · `qwen3.8-27b@NVFP4` | – | **510.7 tok/s** | – | – | **4.28116 (absolute)** | – | **196,032 tok** |

- **Measured with:** self-reported (LocalMaxxing) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Ubuntu 26.10; RTX 5090 32 GB
- **Requires:** Linux · sm_120a · 32 GB GPU memory · CUDA 13.3
- **Checkpoint** (`qwen3.8-27b@NVFP4`): [`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4/tree/4a22293aa6d1026844bc93a8727f1837e48acd89) at `4a22293a`
- **Download** (artifact): [`jabbatheduck/ninfer-ext-models/qwen3.8-27b-quasar-nvfp4`](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/qwen3.8-27b-quasar-nvfp4), about 19.6 GB: the converted artifact, with SHA256SUMS; download it instead of converting the checkpoint
- **Download** (drafter): [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/015e795645c74b1a0eeef3b570031fb62e769bc5): only to convert the checkpoint yourself; the published artifact already holds it
- **Source:** [giveen/ninfer-ext @ `7b586bde`](https://github.com/giveen/ninfer-ext/tree/7b586bde50747eac26d198530aa04d83216947a9)
- **License:** Apache-2.0

NInfer, a from-scratch C++23/CUDA inference engine (an extended fork of upstream NInfer). This entry runs the Qwen3.8-27B QUASAR
QAT checkpoint converted to NInfer's v3 `.ninfer` artifact, with DFlash2 block drafting and an INT8
KV cache, which is what lets a 196,032-token context fit next to the draft companion.

## Design

| Part | How |
| --- | --- |
| Execution | One Program per Engine on one GPU, one resident model, a compact decode batch per round, one CUDA graph per round; startup-fixed concurrency 1–8, FIFO ingress, no preemption |
| Weights | NVFP4 imported as stored (E4M3 scale-of-16 blocks plus a per-tensor FP32 scale); FP8 E4M3 row-max embedding and head; BF16 norms; the GDN a/b projections, which have no NVFP4 route, decoded to BF16 |
| Decode | NVFP4 MMA (`mma.m16n8k32` block-scaled) and FP8 MMA through per-format routes; weights stay in the format they were converted to, nothing is repacked at load |
| Speculation | DFlash2 (block diffusion, block size 8, five target taps, selector rank 256/top-k 16), converted into the artifact and proposed through the target's output head (`--lm-head-draft`), drafts 9 tokens per round; the verify pass runs as columns of the same decode batch, greedy acceptance (the CLI defaults to the request's sampler) |
| KV cache | INT8 group-64 (FP16 group scale) in 64-token pages; `--kv-dtype fp8`, `nvfp4`, `k8v4` and `bf16` are the other selectable representations, and `--kv-capacity auto` resolves the page count from `--max-context` and free device memory |
| Serving | OpenAI-compatible and Anthropic-compatible HTTP APIs, SSE streaming, grammar-constrained output (xgrammar), chat templates (llama-jinja) |
| Arithmetic contract | Each numerical op is qualified against a naive FP32/FP64 oracle at registered shapes; the contract is in `docs/maintainer/op-development.md` upstream |

## Measured

Machine: Ubuntu 26.10, GCC 15.3, CMake 4.3, Ninja. All runs below are
`--max-concurrency 1`, concise reasoning disabled, process pinned with `taskset`.

The headline decode is LocalMaxxing's standardized
[run](https://www.localmaxxing.com/en/runs/cmvfhjc4t0177rf0137adcri2/qwen3-8-27b-quasar-nvfp4-rtx-5090-510-7-tps),
whose page carries the exact test settings and depth. The tables below are this entry's own
measurements with the commands under Build and chat.

### Context capacity and memory

Startup resolves the KV page count and refuses a plan that does not fit next to the weights and
the 1 GiB automatic headroom. Measured with the artifact above, DFlash2 K=9 and `--lm-head-draft`:

| KV dtype | `--max-context` | Pages | Runtime reservation | Free after startup |
| --- | ---: | ---: | --- | --- |
| bf16 | 196,000 | 3,063 | fails: needs 13.95 GB + 1.0 GiB | 12.53 GB available |
| bf16 | 158,720 | 2,480 | fails by 32 MB | 12.55 GB available |
| bf16 | 155,648 | 2,432 | 10.7 GiB | 1.46 GiB |
| int8 | 196,032 | 3,063 | 7.65 GiB | 4.51 GiB |
| fp8 | 196,032 | 3,063 | 8.09 GiB | 4.06 GiB |
| nvfp4 | 196,032 | 3,063 | 4.85 GiB | 7.30 GiB |
| k8v4 | 196,032 | 3,063 | 6.76 GiB | 5.39 GiB |
| int8 | 262,144 | 4,096 | 9.73 GiB | 2.41 GiB |
| nvfp4 | 262,144 | 4,096 | 5.98 GiB | 6.13 GiB |

The BF16 ceiling at K=9 is about 157k tokens; the draft companion holds roughly 2.6 GB of device
memory outside the KV reservation (at 200,704 tokens the same artifact without `--spec` starts with
BF16 KV, while with DFlash2 it does not). Free memory is read at load time, so anything else holding
device memory when a server starts moves the ceiling by up to about 0.7 GB.

### Quality (KV representation)

`ninfer-perplexity` on the repository's `ninfer-ppl-1m-v1` corpus, `--quick`, `--context 16384
--stride 8192`, 261,167 scored tokens, one run per row, same artifact:

| KV dtype | mean_nll | PPL | vs bf16 |
| --- | ---: | ---: | ---: |
| bf16 | 1.456402 | 4.290495 | — |
| int8 | 1.454224 | 4.281160 | −0.22% |
| fp8 | 1.446866 | 4.249773 | −0.95% |
| nvfp4 | 1.459392 | 4.303342 | +0.30% |
| k8v4 | 1.459816 | 4.305169 | +0.34% |

The differences are within run-to-run variation at this window size; only nvfp4 and k8v4 show a
consistent cost. No KLD-budget campaign was run for this checkpoint.

## Checks

- **Capacity and memory:** the table above is the engine's own startup resolution
  (`cudaMemGetInfo` after weights, page count, runtime reservation) printed on every start; values
  were read from those logs.
- **Quality:** perplexity over 261,167 tokens per KV representation, against a fixed corpus and
  fixed window protocol.
- **Speculation:** startup, drafting and verify were exercised by the capacity runs (each started,
  listened, and was stopped); speculative output is not bitwise equal to plain greedy decoding, by
  design.
- **Arithmetic:** the op oracles that qualify every numerical route live upstream
  (`tests/ops/`, `docs/maintainer/op-development.md`) and are not re-run by this entry.

## Limits

- Text only; the artifact carries the vision tower but this entry does not exercise it.
- Needs a 32 GB-class GPU. With DFlash2 K=9 the BF16 KV ceiling is about 157k tokens; 196,032
  tokens requires `--kv-dtype int8` (or nvfp4/k8v4/fp8).
- Single GPU, one resident model, concurrency 1–8 fixed at startup, no request preemption.
- Speculative decoding changes rounding versus plain decode and is not bit-exact.
- Long-context quality was measured with 16,384-token windows; quantization error grows with the
  amount of history the KV cache holds.
- The measured ceiling depends on free device memory at load; the startup check is strict, so a
  desktop holding memory can reject a capacity that started earlier.

## Build and chat

Prerequisites: CMake 3.28+, Ninja, a C++23 host compiler, CUDA 13.x (`sm_120a`), FFmpeg
development libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`), libcurl 7.85+,
and Python 3.11 with `numpy`, `torch`, `transformers` for conversion.

```sh
git clone https://github.com/giveen/ninfer-ext.git
git -C ninfer-ext checkout 7b586bde50747eac26d198530aa04d83216947a9
cd ninfer-ext
cmake --preset release
cmake --build --preset release
# binaries: build/apps/ninfer, build/apps/ninfer-serve, build/apps/ninfer-perplexity
```

Check the checkpoint against [card.lock.json](card.lock.json) from a crit-hub checkout:
`python -B tools/verify_model.py <this engine folder> /path/to/checkpoint`.

Convert the checkpoint from the cloned tree:

```sh
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --recipe qwen3_8_27b_quasar_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --proposal \
  --name qwen3.8-27b-quasar \
  --out models/qwen3_8_27b_quasar_nvfp4.ninfer
```

`--proposal` reads `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` from the clone.

Serve and chat:

```sh
./build/apps/ninfer-serve models/qwen3_8_27b_quasar_nvfp4.ninfer \
  --model-id qwen3.8-27b --max-concurrency 1 --max-context 196000 --kv-capacity auto \
  --kv-dtype int8 --spec dflash2 --draft-tokens 9 --lm-head-draft
```

The server listens on `http://127.0.0.1:8080` with the OpenAI-compatible
`/v1/chat/completions` and Anthropic-compatible `/v1/messages` APIs. One-shot terminal chat:

```sh
./build/apps/ninfer models/qwen3_8_27b_quasar_nvfp4.ninfer \
  --prompt "Explain why data races are hard to debug." --max-context 196000 \
  --kv-capacity auto --kv-dtype int8 --spec dflash2 --draft-tokens 9 --lm-head-draft \
  --max-new 256 --greedy --no-thinking
```

## Credits

NInfer is licensed under the Apache License 2.0; the license text is at the pinned commit,
[LICENSE](https://github.com/giveen/ninfer-ext/blob/7b586bde50747eac26d198530aa04d83216947a9/LICENSE).
NInfer is a fork of [Neroued/ninfer](https://github.com/Neroued/ninfer), and the components it
links (CUDA, FFmpeg, libcurl, xgrammar, llama-jinja) and the checkpoint and drafter weights keep
their own licenses and terms, recorded in that repository.
