## Engine folder layout

Each engine lives once, in `engines/<owner>/<name>/`: `<owner>` is the contributor's GitHub
account or organization and `<name>` describes the engine's design. Its `engine.json` says what
it is and which GPUs and models it runs, and [browse/](../browse/README.md) indexes every engine by
GPU, then model and encoding. An engine tailored to one GPU and one model and an engine that runs
many sit side by side there.

Each engine records its origin in `engine.json`, whatever toolkit built it:

- **`crit`**: built with [crit](https://github.com/crit-labs/crit) and exported from a sealed
  candidate with `tools/export_engine.py`. The folder carries the runtime source.
- **`external`**: built with any other toolkit. The folder pins the source at a commit in the
  contributor's repository and carries no source of its own. crit-hub has not reviewed that
  code; read it before you build or run it.

```text
engines/<owner>/<name>/        origin: crit
  README.md          contributor repository, target, measured results, limits, build and chat
  engine.json        what the engine is and runs, its provenance and exact checkpoints
  engine/            source, configuration and required runtime data
  THIRD_PARTY.md     dependency credits, when applicable
  LICENSES.txt       applicable license texts and notices

engines/<owner>/<name>/        origin: external
  README.md          the same template, opening with the external-engine notice
  engine.json        the same, with the source repository and commit
```

Check a model download against the checkpoint an engine was measured on with
`python -B tools/verify_model.py <engine folder> <model folder>`.

<a id="engine-json"></a>

## engine.json

The website and coding agents read every engine from its `engine.json`; the README is the page
people read.

```json
{
  "summary": "Qwen3.8-27B with up to 7 MTP drafts per round and FP8 attention",
  "origin": "crit",
  "contributor": {"name": "crit-labs", "repository": "https://github.com/crit-labs/crit"},
  "license": "Apache-2.0",
  "features": ["openai-api", "tool-calls", "speculative"],
  "results": [
    {"gpu": "rtx5090", "model": "qwen3.8-27b@NVFP4", "measured_with": "crit-eval-v5",
     "decode_tok_s": 215.7, "prefill_tok_s": 8917, "ttft_ms": 120, "context_tokens": 262144}
  ],
  "supports": [{"gpus": ["rtx4090"], "models": ["qwen3.8-27b@NVFP4"]}],
  "provenance": {"run_id": "...", "candidate_id": "...", "toolchain": {"os": "...", "cuda": "..."}},
  "checkpoints": {
    "qwen3.8-27b@NVFP4": {"repository": "unsloth/Qwen3.8-27B-NVFP4", "revision": "<commit>",
                          "files": [{"path": "config.json", "bytes": 4321, "sha256": "<sha256>"}]}
  }
}
```

- **`summary`**: one line, at most 120 characters, on what the engine does.
- **`license`**: the SPDX id of the engine source. **`features`**: any of `openai-api`,
  `tool-calls`, `vision` and `speculative`.
- **`results`** (required) lists every GPU and model the engine was measured on, with the
  headline figures of the README's **Measured** section: `decode_tok_s` (required),
  `prefill_tok_s`, `ttft_ms`, `context_tokens` and, when the engine keeps weights in host memory,
  `host_ram_gib`. A figure is a number, or `[low, high]` across runs, and must appear in the
  README as written there, for example `8,917` or `1.61 s`. **`measured_with`** names the
  method: figures under the same major version of the crit Eval protocol, such as `crit-eval-v5`
  and `crit-eval-v5.1`, are comparable across engines; anything else is the contributor's own
  report.
- **`supports`** (optional) names further pairs the engine runs but nobody has measured yet. Each
  group pairs every GPU in `gpus` with every model in `models`.
- **`provenance`**: for a crit engine the sealed run, candidate, hashes and toolchain the exporter
  records; for an external engine `source` (`repository` and the 40-character `revision`), the
  toolchain and runtime prerequisites.
- **`checkpoints`** pins the exact checkpoint of every measured model: Hugging Face `repository`,
  `revision`, and `path`, `bytes` and `sha256` of each file.

A GPU id is the card in lowercase, such as `rtx5090`; a model id is the model in lowercase and its
encoding, such as `qwen3.8-27b@NVFP4`. Reuse the ids already in [browse/](../browse/README.md) so an
engine lands on the same page as the others for that GPU and model. Every model needs an entry in
[models.json](../models.json), keyed by its id without the encoding, naming the lab that released it
and its family, such as `"qwen3.8-27b": {"lab": "Alibaba", "family": "Qwen"}`, and every encoding
(the quant after `@`) an entry in [quants.json](../quants.json) with its bits per weight and format,
such as `"NVFP4": {"bits": 4, "format": "NVIDIA FP4 (NVFP4)"}`; `tools/check_engines.py` fails
without them.

To add an engine, follow [Contributing](../README.md#contributing).
