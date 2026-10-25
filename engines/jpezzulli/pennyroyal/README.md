<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->

# pennyroyal

### rtxpro6000 · qwen3.8-flash-next@NVFP4

> [!WARNING]
> **Contributed by** jpezzulli · [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
>
> External engine: crit-hub has not reviewed its source. Read it before you build or run it.

| Decode (sampled) | Decode (greedy) | Prefill | TTFT | Perplexity | Mean KL divergence | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| – | **207.12 tok/s** | – | – | – | – | **524,288 tok** |

- **Measured with:** self-reported (Pennyroyal v2.5.0 benchmarks) · [self-reported.json](evidence/self-reported.json)
- **Hardware:** Linux; RTX PRO 6000 Blackwell Workstation Edition 96 GB
- **Further figures:** long-context decode 172.64 tok/s
- **Requires:** Linux · sm_120a · 96 GB GPU memory · CUDA 13.3
- **Checkpoint** (`qwen3.8-flash-next@NVFP4`): [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) at `7b719225`
- **Checkpoint** (`qwen3.8-27b@FP8`): [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a) at `017b9c7a`
- **Download** (drafter): [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/015e795645c74b1a0eeef3b570031fb62e769bc5): the DFlash2 draft for the 27B profile
- **Source:** [aiueo52/sglang-rtxpro6000 @ `468e63a9`](https://github.com/aiueo52/sglang-rtxpro6000/tree/468e63a96a63628f8cd1bc217e2287c164088456)
- **License:** Apache-2.0

Pennyroyal is an SGLang-derived runtime for one RTX PRO 6000 Blackwell (96 GB, SM120) with two
launch profiles from the same patched source: Qwen3.8 Flash-Next NVFP4 with native NEXTN MTP and
FR-Spec, and Qwen3.8-27B FP8 with DFlash2. Both run with 524,288-token context, and HiCache/NIXL
keeps conversation prefixes across GPU eviction and server restarts. A prebuilt container and a
native build are both documented.

---

## Design

| Part | How |
| --- | --- |
| Flash-Next | NVFP4 target, FP8 KV, native NEXTN MTP with FR-Spec; optional online FP8 for selected weights and NVMe-backed PLE |
| 27B | FP8 target and KV with the DFlash2 draft |
| Persistence | GPU radix state, page-first HiCache in host RAM, NIXL POSIX file storage (io_uring, O_DIRECT); RecoverSSM for the recurrent state |
| SM120 | Scoped attention, GDN and MoE paths with recovery CUDA graphs |
| Serving | OpenAI-compatible SGLang server with tool calls, reasoning and image input |

---

## Measured

One RTX PRO 6000, 524,288-token context, HiCache and NIXL on:

| Flash-Next NVFP4, v2.5.0 (2026-09-11), 1,024 output tokens | Online FP8 off | Online FP8 on |
| --- | ---: | ---: |
| Short prompt | 161.47 tok/s | **207.12 tok/s** |
| 128K context | 154.70 tok/s | **195.63 tok/s** |
| 490K context | 149.14 tok/s | **172.64 tok/s** |

| 27B FP8 + DFlash2, v2.4.1 (2026-09-10) | Speed | |
| --- | ---: | --- |
| 64K cold prefill | **6,570 tok/s** | 10.468 s to first token |
| 490K cold prefill | 1,646 tok/s | 302.3 s to first token; three needles found |
| Single-request decode | **108.31 tok/s** | after the first token |
| Four requests at once | 374.98 tok/s aggregate | including time to first token |

The 27B figures used `orcarouter/Qwen3.8-27B-Uncensored-FP8`, an abliterated derivative of the
official checkpoint. That repository is gated, so the card records only the Flash-Next result and
lists the official `Qwen/Qwen3.8-27B-FP8`, which runs with the same recipe, as supported.

---

## Checks

- **Validation:** reasoning, tool calling, long-context needles, vision and concurrent decode in
  [jpezzulli/pennyroyal-validation](https://github.com/jpezzulli/pennyroyal-validation).
- **v2.5.3:** fixes the native MTP checkpoint selection behind the phantom-token problem in issue #17.

---

## Limits

- The pinned commit is aiueo52's fork of the canonical repository; the upstream release is
  [pennyroyal-v2.5.3](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v2.5.3).
- Short-context decode figures are three-run medians, the long-context rows single runs.
- NVMe PLE saves RAM but was slower with concurrent requests; one profile per GPU at a time.

---

## Build and chat

```sh
git clone https://github.com/aiueo52/sglang-rtxpro6000.git
git -C sglang-rtxpro6000 checkout 468e63a96a63628f8cd1bc217e2287c164088456
```

Build natively with
[BUILD.md](https://github.com/aiueo52/sglang-rtxpro6000/blob/468e63a96a63628f8cd1bc217e2287c164088456/BUILD.md)
or use the prebuilt image (`docker/pennyroyal/README.md`), then launch a profile from
[RUN.md](https://github.com/aiueo52/sglang-rtxpro6000/blob/468e63a96a63628f8cd1bc217e2287c164088456/RUN.md).
Check a download with `python -B tools/verify_model.py <this engine folder> <model folder>`.

---

## Credits

Apache License 2.0,
[LICENSE](https://github.com/aiueo52/sglang-rtxpro6000/blob/468e63a96a63628f8cd1bc217e2287c164088456/LICENSE)
at the pinned commit, as SGLang. By John Pezzulli; FR-Spec from
[gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120),
online FP8 inspired by mratsim's sglang-qwen38fn-sm120-turbo, NVMe PLE adapted from Garner
McCloud's SSD Stream. The DFlash2 draft is Inco AI's; checkpoints keep their own licenses.
