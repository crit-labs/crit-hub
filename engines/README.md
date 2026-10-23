## Engine folder layout

Each engine lives once, in `engines/<owner>/<name>/`: `<owner>` is the contributor's GitHub
account or organization and `<name>` describes the engine's design. These folders are the single
source of truth: anything that lists engines by GPU, model or checkpoint, such as the website, is
generated from them. An engine tailored to one GPU and one model and an engine that runs many sit
side by side.

```text
engines/<owner>/<name>/
  card.yaml            what the engine is, what it measured and its README's prose: the one file you write
  card.lock.json       what tools write: the exact checkpoint files and how the engine was built
  evidence/
    crit-eval.json     the crit Eval result behind crit Eval figures, unchanged (crit-eval-2.json and on)
    self-reported.json optional: the evidence behind your own figures (method, link, settings)
  README.md            rendered from card.yaml by tools/card.py; never edited by hand

crit-engines: <name>/  origin: crit, at the pinned commit
  engine/              source, configuration and required runtime data
  THIRD_PARTY.md       dependency credits
  LICENSES.txt         applicable license texts and notices
```

Each engine records its origin in card.yaml, whatever toolkit built it:

- **`crit`**: built with [crit](https://github.com/crit-labs/crit) and exported from a sealed
  candidate with `tools/export_engine.py`. Its source, dependency credits and license texts are in
  `<name>/` of [crit-engines](https://github.com/crit-labs/crit-engines), at the commit the card pins.
- **`external`**: built with any other toolkit. The source is at a commit in the contributor's
  repository. crit-hub has not reviewed that code; read it before you build or run it.

Start from [`_template/`](_template/): its card.yaml documents every key, and
[`schemas/card.schema.json`](../schemas/card.schema.json) describes it for editors and agents. Render
the README with `python -B tools/card.py engines/<owner>/<name>`, then run
`python -B tools/check_engines.py`. Check a model download against the checkpoint an engine was
measured on with `python -B tools/verify_model.py <engine folder> <model folder>`.

<a id="stats"></a>

## The six stats

Every result states the same six headline stats, in this order, and every page that shows an engine
shows them the same way: its README opens with them, the website's engine pages and tables use
them, and crit's case studies use them.

| Stat | card.yaml | Shown as |
| --- | --- | --- |
| Decode | `decode_tok_s`: from crit-eval-v7 at the model's own sampling, through v6 greedy | `252.4 tok/s` |
| Prefill | `prefill_tok_s` | `7,425 tok/s` |
| TTFT | `ttft_ms`: time to first token | `150 ms`, or `1.61 s` from one second on |
| Perplexity vs reference | `ppl: {vs_reference: 1.001}`, or `ppl: {absolute: 4.28}` when no reference was measured | `+0.1%`, or `4.28` |
| Mean KL divergence | `kld_mean`: against the reference | `0.0028` |
| Context | `context_tokens`: the maximum context it was measured at | `262,144 tok` |

A stat nobody measured is `null` and shows as `–`; decode is required. Decode, prefill and TTFT
may be `[low, high]` across runs. The README shows the six as one table, with a caption under it
saying what was measured, how, when and from which evidence. Further figures go under `more`:
`greedy_decode_tok_s`, `long_context_decode_tok_s`, `heldout_decode_tok_s`, `host_ram_gib` and `quality_budget_pct`; the
caption lists them too.

Real requests sample, so from crit-eval-v7 decode is the sampled decode, measured on the speed cases
at the model's `generation_config.json` sampling (`sampled_speed.decode_score` in the evidence), and
the greedy decode goes under `more` as `greedy_decode_tok_s` (`speed.decode_score`). A crit-eval-v6
result's decode is greedy and has no `greedy_decode_tok_s`. The README and the website label the
column Decode (sampled) or Decode (greedy) from `measured_with`; a self-reported result says in its
prose which it is. A v7 decode sampled at other settings than the model's own (a user's
`crit eval prepare --sampling` override, or the protocol default for a model that does not sample)
states them under the result's `sampling: {source, settings}`, as its result file records them, and
the caption names them.

<a id="provenance"></a>

## Evidence

A figure is either **crit Eval with its evidence** or **the contributor's own report**, whoever
contributed the engine.

- A result whose `measured_with` is a crit Eval protocol (`crit-eval-v7`) lists the result file
  `crit eval run` wrote, unchanged, as `evidence/crit-eval.json` with its SHA-256 (`crit-eval-2.json`
  and on for further runs or GPU/model pairs), and gives `measured_on`, the day it ran.
  `tools/check_engines.py` reads it and requires the protocol, the GPU, a complete speed run with no
  failed gate, and every stat and figure the card states: decode, prefill, first token and held-out
  decode from the speed scores, context from the measured context, perplexity and KL divergence from
  the worst served path of the quality section, the quality budget share and long-context decode.
  Figures under the same major version of the protocol, such as `crit-eval-v7` and `crit-eval-v7.1`,
  are comparable across engines.
- Any other `measured_with`, such as a leaderboard or your own benchmark, is shown as your own
  report. Its evidence can go in `evidence/self-reported.json` (optional): for each GPU and model,
  the measurements behind its figures, each with its `method`, a `link` to the run, leaderboard or
  card, the `settings` measured, and the `figures` as measured. Each stat and figure the card states
  must then be one the file records:

```json
{"results": [{"gpu": "rtx5090", "model": "qwen3.8-27b@NVFP4", "measurements": [
  {"method": "LocalMaxxing leaderboard run", "link": "https://www.localmaxxing.com/en/runs/...",
   "settings": "--max-concurrency 1, INT8 KV, 9 draft tokens", "figures": {"decode_tok_s": 510.7}}]}]}
```

<a id="card"></a>

## card.yaml

The website and coding agents read every engine from its card; the rendered README is the page
people read. [`_template/card.yaml`](_template/card.yaml) documents every key; in short:

- **`title`**, **`summary`** (one line, at most 120 characters), **`origin`**, **`contributor`**
  (`name`, which names the folder, and `repository`), **`license`** (SPDX id) and **`features`**
  (any of `openai-api`, `tool-calls`, `vision`, `speculative`).
- **`source`**: the `repository` and 40-character commit `revision` that hold the engine's source,
  and optionally `path`, its folder there.
- **`requirements`**: the `platforms` it was built and run on, the `gpu_arch` it is compiled for,
  the `gpu_memory_gb` it needs, the `cuda` toolkit and optionally the `runtime` it links.
- **`downloads`** (optional): what to fetch besides the checkpoint, such as a `drafter` or a
  converted `artifact`, pinned by `revision` or the `sha256` of its `file`.
- **`checkpoints`**: the Hugging Face `repository` and `revision` of every model the engine measures
  or supports, keyed by model id; card.lock.json lists every file of each measured one with its size
  and SHA-256. The website reads the checkpoint's base model, lab and quantization from Hugging Face,
  so a checkpoint whose model card names its `base_model` lands under the right model by itself.
- **`supports`** (optional): further pairs it runs that nobody has measured yet.
- **`derived_from`** (optional, written by the exporter): the crit-hub engine the run started from
  (`crit run create --from-engine`), as `engine` (`<owner>/<name>`) and the `source` it pinned, so an
  improved, ported or adapted engine says where it came from.
- **`built`** (optional, written by the exporter from the run's `cost.json`): what building the engine
  took, as `hours`, `candidates`, `agent_tokens`, `api_usd` and the `agent`.
- **`results`**: for each GPU and model it was measured on, `measured_with`, `measured_on`, the six
  `stats`, optional `more` and its `evidence`.
- **`readme`**: the README's prose as markdown blocks, rendered in this order: `intro`, `model`,
  `design`, `measured`, `checks`, `limits`, `build_and_chat`, the optional `benchmark` and
  `credits`. The renderer adds the title, the credit (or the external-engine notice), one stats row
  per result and the facts: requirements, checkpoints, downloads, source and license.

A GPU id is the card in lowercase, such as `rtx5090`; a model id is the model in lowercase and its
encoding, such as `qwen3.8-27b@NVFP4`. Reuse the ids other engines here already use so an engine
lands on the same page as the others for that GPU and model.

To add an engine, follow [CONTRIBUTING.md](../CONTRIBUTING.md).
