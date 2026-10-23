# Contributing an engine

Every working engine you add saves the next person from building it again. An engine is one folder,
`engines/<your account>/<name>/`, named after its design, for example `mtp7-device-confidence`.
Several designs can share a GPU and model, whether they were built with crit or not. The
[folder layout](engines/README.md) describes what goes in it; in short, you write one file,
`card.yaml`, and tools write the rest.

Keep model weights, builds, binaries, development scripts and full experiment records outside the
repository; the [.gitignore](.gitignore) uses a whitelist for published files.

## The numbers

Every engine states the same [five stats](engines/README.md#stats): decode, prefill, context,
perplexity against the reference and mean KL divergence. Figures measured with crit Eval are
compared with every other engine's on the same protocol, provided the result file `crit eval run`
wrote sits unchanged in the folder as `evidence/crit-eval.json`. Anything else, such as your own
benchmark or a leaderboard run, is shown as your own report, with its method, link and settings in
`evidence/self-reported.json` if you add it.

crit Eval needs no crit engine:
[`crit eval run --url`](https://github.com/crit-labs/crit/blob/main/docs/eval/README.md#engines-behind-a-server)
measures any engine behind an OpenAI-compatible chat API, sending every prompt to
`/chat/completions` and scoring quality on the engine's own replies. Run it on the commit you pin:

```sh
crit eval run --url http://127.0.0.1:8000/v1 --model <model folder> --data <prepared Eval data> \
  --context <the context the server was started with> --out result.json
```

Prepare the Eval data and score the reference first (`crit eval prepare` and `crit eval reference`,
as crit's [Eval guide](https://github.com/crit-labs/crit/blob/main/docs/eval/README.md) describes),
or the result carries no quality figures. Then copy `result.json` to `evidence/crit-eval.json`.

## Add a crit engine

1. **Export the source.** `python -B tools/export_engine.py source` verifies the candidate's seal
   and copies the runtime source and data you select, the configuration, dependency credits and the
   license texts into `<name>/` of a [crit-engines](https://github.com/crit-labs/crit-engines)
   checkout. Commit it there.
2. **Write the README text** as markdown with the sections of the
   [template](engines/_template/card.yaml): an intro, then `## Model`, `## Design`, `## Measured`,
   `## Checks`, `## Limits`, `## Build and chat`, optionally `## Benchmark your machine`, and
   `## Credits`. Build and chat starts from
   `git clone https://github.com/crit-labs/crit-engines && git -C crit-engines checkout <crit-engines commit>`;
   the exporter fills in the commit.
3. **Export the entry.** `python -B tools/export_engine.py entry` writes the folder: card.yaml with
   your README's sections as its prose, pinned to the crit-engines commit; card.lock.json with the
   checkpoint's files and the sealed run's provenance; and, for each `--eval attempt-N`, the
   candidate's sealed crit Eval result in `evidence/`, from which it fills the five stats and further
   figures.
4. **Fill in the card**: summary, license, features and requirements, and any downloads.
5. **Render and check:** `python -B tools/card.py engines/<owner>/<name>` writes README.md;
   `python -B tools/check_engines.py` must pass.
6. Open one pull request to crit-engines with the source and, once it is merged, one to crit-hub
   with the entry pinned to the merged commit.

`--model-repo` and `--model-revision` name the checkpoint the run used, `--gpu` and `--model` what it
was measured on, and `--contributor` and `--contributor-repo` your account (which names the folder)
and the repository shown in the credit. See `python -B tools/export_engine.py entry --help`.

## Add an external engine

An engine built with any other toolkit is welcome. Its source stays in your repository, and
crit-hub records where it is, what it targets and what it measured.

1. Copy [`engines/_template/`](engines/_template/) to `engines/<your account>/<name>/` and set
   `origin: external`.
2. Pin your source to the full 40-character commit you built and measured, under `source`.
3. Fill in every key of card.yaml: its comments say what each one means. The prose goes under
   `readme`; in **measured**, state the hardware, OS, toolchain, workload and protocol behind each
   number, and in **credits**, the source's license, linked at the pinned commit.
4. In card.lock.json, list every file of each measured checkpoint with its size and SHA-256, and
   your toolchain and runtime prerequisites under `provenance`.
5. Add your evidence: `evidence/crit-eval.json` for crit Eval figures, or
   `evidence/self-reported.json` for your own, and list it with its SHA-256 under the result.
6. **Render and check:** `python -B tools/card.py engines/<owner>/<name>`, then
   `python -B tools/check_engines.py`.

The folder holds no source, binaries, test data, scripts or license texts; those stay with your
source. The rendered README opens with the external-engine notice: crit-hub has not reviewed the
code.

## Update an engine

Change the pinned `source.revision`, the results and anything else the new commit changes in one
change. A result describes the commit it was measured on, so a new commit brings new measurements
and a new `measured_on`, never the old figures carried over.
