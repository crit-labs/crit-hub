<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# openjet-engine

### rtx3090 · qwen3.8-27b@Q4-K-M

> [!WARNING]
> **Contributed by** L-Forster · [L-Forster/open-jet](https://github.com/L-Forster/open-jet)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **106.7 tok/s** | **1,530–1,600 tok/s** | – | – | – | – |

- **Measured with:** self-reported (OpenJet megakernel benchmarks) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** RTX 3090 24 GB
- **Requires:** Linux · sm_86, sm_89, sm_120 · 24 GB GPU memory · CUDA 13.2
- **Checkpoint** (`qwen3.8-27b@Q4-K-M`): [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4121cb19390a7984a0e6dc0f46bea9177b846f15) at `4121cb19`
- **Source:** [L-Forster/open-jet/megakernel @ `72ba4a7e`](https://github.com/L-Forster/open-jet/tree/72ba4a7e587ed0cda1312e3384e415962eaa0db3/megakernel)
- **License:** AGPL-3.0-only

OpenJet is a local coding agent and runtime; its OpenJet Engine (`megakernel/`) is a CUDA engine
written for one model, Qwen3.8-27B Q4_K_M, on RTX 30, 40 and 50-series cards with 24 GB or more. Each
decode step runs as a single persistent kernel launch synchronized by a software grid barrier, and
verifies 4-5 drafted tokens per pass; prompts go through batched tensor-core kernels. It replaces
`llama-server` in OpenJet when the model and GPU match, and has its own `openjet-engine` server.

---

## Design

| Part | How |
| --- | --- |
| Decode | One persistent kernel per step with a software grid barrier; the host reads back only accepted tokens |
| Speculation | The model's MTP drafts, 4-5 tokens verified per pass |
| Prefill | Batched tensor-core kernels |
| Weights | unsloth's Q4_K_M GGUF (a separate build serves UD-Q4_K_XL); embedding table in system RAM |
| KV cache | fp16, or q8_0 when fp16 does not fit the requested context |

---

## Measured

RTX 3090, single conversation, against `llama-server` with open-jet's flags (MTP drafting, q8_0 KV):

| | llama.cpp | OpenJet Engine |
| --- | ---: | ---: |
| Decode, reasoning mode (thinking on, sampled) | 50.9 tok/s | **78.0 tok/s** |
| Decode, greedy, mean of 4 prompts | 63.1 tok/s | **106.7 tok/s** |
| Decode, greedy, writing code | 72.5 tok/s | 139.7 tok/s |
| Prompt processing, 2-8K tokens | 1,050-1,140 tok/s | **1,530-1,600 tok/s** |

Without speculation both run near the memory-bandwidth limit (40.4 vs 42.9 tok/s); the lead comes
from verifying the drafts at almost no extra cost.

---

## Checks

- **Sampling:** with drafts the token distribution matches sampling without them (2,000-sample test per position).
- **KLD:** `llama_ref` teacher-forces 2,048-token chunks to compare against llama.cpp.

---

## Limits

- One model only (Qwen3.8-27B Q4_K_M / UD-Q4_K_XL); everything else falls back to `llama-server`.
- Tuned on sm_86; no measurement date is given.
- AGPL-3.0, including its network-use source requirements.

---

## Build and chat

```sh
git clone https://github.com/L-Forster/open-jet.git
git -C open-jet checkout 72ba4a7e587ed0cda1312e3384e415962eaa0db3
cd open-jet/megakernel && make
```

Or `pipx install open-jet && openjet setup` and choose "Qwen3.8 27B Q4_K_M MTP". Check the GGUF with
`python -B tools/verify_model.py <this engine folder> <model folder>`; build and server options are in
[megakernel/README.md](https://github.com/L-Forster/open-jet/blob/72ba4a7e587ed0cda1312e3384e415962eaa0db3/megakernel/README.md).

---

## Credits

GNU AGPL v3,
[LICENSE](https://github.com/L-Forster/open-jet/blob/72ba4a7e587ed0cda1312e3384e415962eaa0db3/LICENSE)
at the pinned commit. Built against llama.cpp; the GGUF is unsloth's and the model is Qwen's.
