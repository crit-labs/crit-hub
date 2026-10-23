<div align="center">

<p align="center">
  <img src="docs/assets/wordmark.png" width="200" alt="wordmark" />
  <br />
</p>

### Huggingface for tailored inference engines. Don't rebuild an engine that already exists.

[![License Apache-2.0](docs/assets/badge-license.svg)](LICENSE)
[![Registry of engines](docs/assets/badge-registry.svg)](engines/)
[![Built with crit](docs/assets/badge-crit.svg)](https://github.com/crit-labs/crit)

[Summary](#summary) • [Quick start](#quick-start) • [What crit-hub solves](#what-crit-hub-solves) •
[Engines](engines/) • [Contributing](CONTRIBUTING.md) • [License](#license)

</div>

---

<a id="summary"></a>

## Summary

**`crit-hub` is a registry of bespoke inference engines, organized by GPU and model,
so you can find one built for your hardware and target LLM.**
> [!IMPORTANT]
> Tailored engines are scattered across personal repositories, forks and forum posts, and
> there is no Hugging Face for inference engines. crit-hub collects them in one place: each
> engine lives once in [`engines/<owner>/<engine>/`](engines/), the single source of truth: a
> `card.yaml` its README is rendered from. Each engine pins its source and the exact checkpoint it
> was built on, opens with the same five stats as every other (decode, prefill, context,
> perplexity and KL divergence against the reference), and states its evidence, known limits and a
> link to the repository of whoever contributed it.
> Engines come from any toolkit, and every one links its source at a pinned commit. Those built
> with [crit](https://github.com/crit-labs/crit) keep their sealed source in
> [crit-engines](https://github.com/crit-labs/crit-engines); external engines link the contributor's repository and are
> marked as third-party code crit-hub has not reviewed.
> It is a registry, not a scoreboard: several designs can sit side by side for the same GPU
> and model, each documenting its own trade-offs.

---

<a id="quick-start"></a>

## Quick start

Look through [engines/](engines/): each engine's README opens with its five stats for every GPU
and model it was measured on, then what it requires, the exact Hugging Face checkpoint it runs and
its pinned source, and explains how to build it and open chat. It says whether it was built with
crit or is an external engine, whose source you should read before you build it; its `card.yaml`
holds the same in a form your agent can read. Or give this prompt to your coding agent and name your
GPU and target LLM:
```bash
Check github.com/crit-labs/crit-hub for an inference engine for my GPU and model, then build and run it.
```

Nothing fits yet? Build one with [crit](https://github.com/crit-labs/crit) and [contribute](CONTRIBUTING.md) it.

---

<a id="what-crit-hub-solves"></a>

## What `crit-hub` solves

Find what already exists. Stop rebuilding it.

**Discovery in one place 🔎**
> Today you can rarely tell whether an engine for your exact hardware and model exists.
> crit-hub makes that a lookup: every engine states the GPUs, models and exact checkpoints it
> runs in one `card.yaml`.

**A registry, not a scoreboard 📇**
> Engines are listed by what they target, not ranked against each other. Each one states
> the same five stats with their evidence, its own workload, checks and limits, so you can
> choose the design that fits your use rather than the one with the biggest number.

**A jump-off point for developers 🛠️**
> Every engine pins its complete source at a commit, in [crit-engines](https://github.com/crit-labs/crit-engines) or the
> contributor's repository, with its checkpoint pinned by SHA-256 and its toolchain recorded. Start from a working engine to improve it, port it to another GPU or adapt it
> to a related model, and follow the contributor's repository for its upstream work.

---

<a id="contributing"></a>

## Contributing

Every working engine you add saves the next person from building it again. An engine is one folder
with one hand-written file, `card.yaml`; [CONTRIBUTING.md](CONTRIBUTING.md) walks through it:

- **A crit engine:** `tools/export_engine.py` exports a sealed candidate's source into
  [crit-engines](https://github.com/crit-labs/crit-engines), then writes its card, pinned to that
  commit, with its five stats filled from its sealed crit Eval result.
- **An external engine:** copy [`engines/_template/`](engines/_template/), fill in its commented
  card, pin your source at a commit and add your evidence. `crit eval run --url` gives any engine
  behind an OpenAI-compatible API crit Eval figures.

Then `python -B tools/card.py` renders the README from the card, and `python -B tools/check_engines.py`
checks the folder.

---

<a id="license"></a>

## License

Original content is licensed under [Apache-2.0](LICENSE). Engine source retains its own terms:
crit engines' dependency credits and license texts sit next to their source in
[crit-engines](https://github.com/crit-labs/crit-engines), and external engines' in their own repositories.
