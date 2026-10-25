<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# vllm-flash-next-sharp

### rtxpro6000 · qwen3.8-flash-next@NVFP4-BF16MTP

> [!WARNING]
> **Contributed by** WombatSoftware · [WombatSoftware/vllm-qwen3.8-flash-next-rtx-pro-6000-sharp-monitoring](https://github.com/WombatSoftware/vllm-qwen3.8-flash-next-rtx-pro-6000-sharp-monitoring)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **156.06 tok/s** | **20,457.64 tok/s** | – | – | – | **262,144 tok** |

- **Measured with:** self-reported (llama-benchy 0.4.0) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** RTX PRO 6000 Blackwell Max-Q Workstation Edition 96 GB (300 W); driver 610.57.04; 244 GiB host RAM
- **Further figures:** long-context decode 155.08 tok/s
- **Requires:** Linux · sm_120a · 96 GB GPU memory · CUDA 13.3
- **Checkpoint** (`qwen3.8-flash-next@NVFP4-BF16MTP`): [`dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix`](https://huggingface.co/dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix/tree/97a68bdf43a57c40814a945573a35781c498ec2c) at `97a68bdf`
- **Source:** [WombatSoftware/vllm-qwen3.8-flash-next-rtx-pro-6000-sharp-monitoring @ `b34cc936`](https://github.com/WombatSoftware/vllm-qwen3.8-flash-next-rtx-pro-6000-sharp-monitoring/tree/b34cc936f0bb156cd4bf440b3f405fe41acbe056)
- **License:** Apache-2.0

A single-GPU Docker Compose recipe that serves Qwen3.8-Flash-Next with upstream vLLM on one RTX PRO
6000 Blackwell, with the Qwen-Sharp chat template and a bundled Prometheus and Grafana stack. It
makes its choices on purpose: one resharded NVFP4 checkpoint with the official BF16 MTP head, a
nightly vLLM pinned by commit, MTP-3, fp8 KV with a 16,384-token batch budget, FlashInfer autotune on,
and tuning for tool-calling agents. Every non-default flag was benchmarked.

---

## Design

| Part | How |
| --- | --- |
| Engine | Bare upstream `vllm/vllm-openai` nightly `18f8f960`; v0.29.0 cannot load the model and v0.30.0 lacks the fp8 QSA kernels and fixes it uses |
| Weights | NVIDIA's NVFP4 quantization resharded into 141 shards, official BF16 MTP head; the 47.7 GiB FP8 PLE table pinned in host RAM |
| Speculation | MTP with 3 tokens (2.695 mean acceptance length) |
| KV cache | fp8, 341,041-token pool (1.30x one 262,144-token request) |
| Serving | OpenAI API with auto tool choice (Qwen XML tool blocks), split reasoning; Grafana dashboard with 28 panels |

---

## Measured

Max-Q card (300 W), 2026-10-04, llama-benchy 0.4.0, mean of 3 runs:

| Context already loaded | Prefill 2,048 tokens, 1 stream | Decode 128 tokens, 1 stream | Decode, 4 streams aggregate |
| --- | ---: | ---: | ---: |
| none | **20,458 tok/s** | **156.1 tok/s** | 400.0 tok/s |
| 8,192 | 3,887 tok/s | 149.6 tok/s | 363.0 tok/s |
| 32,768 | 4,288 tok/s | 166.1 tok/s | 274.2 tok/s |
| 131,072 | 2,249 tok/s | 155.1 tok/s | 15.9 tok/s |

Single-stream decode stays flat at roughly 133-166 tok/s up to 131K tokens of context; four streams
at deep context fall off the KV pool's edge.

---

## Checks

- **Template:** `scripts/verify-chat-template.sh` checks the vendored Qwen-Sharp v22.5.0 template against upstream.
- **Benchmarks:** `scripts/benchmark.sh` runs under an isolation guard that records a verdict with each result.

---

## Limits

- Not every nightly boots; moving the pin needs the checks in the README.
- The CUDA version of the pinned nightly image is not stated upstream; the measurement driver reported CUDA 13.3.
- MTP-3 costs 10-35% on concurrent context-load throughput; the endpoint and Prometheus have no authentication.

---

## Build and chat

```sh
git clone https://github.com/WombatSoftware/vllm-qwen3.8-flash-next-rtx-pro-6000-sharp-monitoring.git recipe
git -C recipe checkout b34cc936f0bb156cd4bf440b3f405fe41acbe056
docker pull vllm/vllm-openai:nightly-18f8f96025b556071eb627076f94df560fbd3a22
```

Set `MODELS_DIR` and a Grafana password in `.env`, download
`dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix` into it, check it with
`python -B tools/verify_model.py <this engine folder> <model folder>`, then `docker compose up -d`.
The API answers on `http://127.0.0.1:8000/v1` as `qwen-3-8-flash-next`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/WombatSoftware/vllm-qwen3.8-flash-next-rtx-pro-6000-sharp-monitoring/blob/b34cc936f0bb156cd4bf440b3f405fe41acbe056/LICENSE)
at the pinned commit. The recipe is glue around the vLLM project, peculiar-ragdoll's Qwen-Sharp and
froggeric's chat templates, dicksondickson's resharded checkpoint, NVIDIA's NVFP4 quantization and
Qwen's model; see the repository's NOTICE.
