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
[Engines](engines/README.md) • [Contributing](#contributing) • [License](#license)

</div>

---

<a id="summary"></a>

## Summary

**`crit-hub` is a registry of bespoke inference engines, organized by GPU and model,
so you can find one built for your hardware and target LLM.**
> [!IMPORTANT]
> Tailored engines are scattered across personal repositories, forks and forum posts, and
> there is no Hugging Face for inference engines. crit-hub collects them in one tree,
> `engines/<gpu>/<model>@<encoding>/<engine>/`. Each engine pins its source and the exact
> checkpoint it was built on, and states what was measured, its known limits and a link to
> the repository of whoever contributed it.
> Engines come from any toolkit. Those built with [crit](https://github.com/crit-labs/crit)
> carry their sealed source here; external engines link to their source at a pinned commit
> and are marked as third-party code crit-hub has not reviewed.
> It is a registry, not a scoreboard: several designs can sit side by side for the same GPU
> and model, each documenting its own trade-offs.

---

<a id="quick-start"></a>

## Quick start

Open [engines/](engines/), pick your GPU, then your model and encoding. Each engine folder's
README explains how to build it and open chat, and says whether it was built with crit or is
an external engine, whose source you should read before you build it. Or give this prompt to your coding agent and
name your GPU and target LLM:
```bash
Check github.com/crit-labs/crit-hub for an inference engine for my GPU and model, then build and run it.
```

Nothing fits yet? Build one with [crit](https://github.com/crit-labs/crit) and [contribute](#contributing) it.

---

<a id="what-crit-hub-solves"></a>

## What `crit-hub` solves

Find what already exists. Stop rebuilding it.

**Discovery in one place 🔎**
> Today you can rarely tell whether an engine for your exact hardware and model exists.
> crit-hub makes that a folder lookup: GPU, then model and encoding, then every engine
> built for that pair.

**A registry, not a scoreboard 📇**
> Engines are listed by what they target, not ranked against each other. Each one states
> its own workload, results, checks and limits, so you can choose the design that fits
> your use rather than the one with the biggest number.

**A jump-off point for developers 🛠️**
> Every engine pins its complete source, in its folder or at a commit in the contributor's
> repository, with its checkpoint pinned by SHA-256 and its toolchain recorded. Start from a working engine to improve it, port it to another GPU or adapt it
> to a related model, and follow the contributor's repository for its upstream work.

---

<a id="contributing"></a>

## Contributing

Every working engine you add saves the next person from building it again.

---

<details>
<summary><strong>📋 Add a crit engine</strong></summary>

<br />

1. [Export the sealed candidate](#export) into the [folder layout](engines/README.md#engine-folder-layout)
   with a name that describes its design, for example `mtp7-fp8attn`. Several designs can share
   a GPU and model/encoding, whether they were built with crit or not.
2. Include runnable source and a README that follows the [engine README template](docs/ENGINE_README.md):
   model, design, measured results, checks, limits, build and chat.
3. Document the measured workload, results, check coverage and known limits.
4. [Credit your repository](#credit) directly under the README's title.
5. Retain dependency credits and applicable license texts and notices.

Keep model weights, builds, binaries, development scripts and full experiment records
outside the repository. The [.gitignore](.gitignore) uses a whitelist for published files.

</details>

---

<a id="external"></a>

<details>
<summary><strong>🧩 Add an external engine</strong></summary>

<br />

An engine built with any other toolkit is welcome. Its source stays in your repository, and
crit-hub records where it is, what it targets and what it measured.

1. Place it under the [folder layout](engines/README.md#engine-folder-layout) with a name that
   describes its design. The folder holds exactly three files: `README.md`, `model.json` and
   `engine/provenance.json`. No source, binaries, test data, scripts or license texts; those
   stay with your source.
2. Pin your source to a full 40-character commit SHA, the commit you built and measured.
3. Write the README from the [engine README template](docs/ENGINE_README.md), opening with the
   external-engine notice. Build and chat start from a clone of the pinned commit.
4. In **Measured**, state the hardware, OS, toolchain, workload and protocol behind each number.
   Links to third-party leaderboards can add context, but the README carries its own results.
5. `model.json` pins the checkpoint in the same format as crit engines (`repository`, `revision`,
   and `path`, `bytes` and `sha256` for each file), so `tools/verify_model.py` checks it.
6. `engine/provenance.json` records:

   ```json
   {
     "origin": "external",
     "contributor": {"name": "your-name", "repository": "https://github.com/your-name/your-engine"},
     "source": {"repository": "https://github.com/your-name/your-engine", "revision": "<40-character commit SHA>"},
     "toolchain": {"os": "...", "cuda": "...", "cxx": "...", "cmake": "..."},
     "runtime_prerequisites": ["..."]
   }
   ```

7. Name your source's license in the README's **Credits** section.

To update an external engine, change the pinned commit, the measurements and anything else the new
commit changes in one change.

</details>

---

<a id="credit"></a>

<details>
<summary><strong>🔗 Credit your repository</strong></summary>

<br />

Each engine links the repository of whoever contributed it, where people can follow its
upstream work, report issues and find related engines. The link appears in two places:

- **The engine README**, directly under the title, so it is the first thing anyone opening
  the folder sees:

  ```markdown
  # Qwen3.8-27B @ NVFP4 · RTX 5090 · mtp7-fp8attn

  > [!NOTE]
  > **Contributed by** your-name · [your-name/your-engine](https://github.com/your-name/your-engine)
  ```

- **`engine/provenance.json`**, as `"contributor": {"name": ..., "repository": ...}`.

For a crit engine, the exporter below writes both from `--contributor` and `--contributor-repo`,
and adds the built-with-crit line under the credit. An external engine opens with the
external-engine notice from the [template](docs/ENGINE_README.md) instead.

</details>

---

<a id="export"></a>

<details>
<summary><strong>📦 Export a sealed crit candidate</strong> → <a href="tools/export_engine.py">Export tool</a></summary>

<br />

```sh
python -B tools/export_engine.py --help
```

The exporter verifies the complete candidate seal and copies explicitly selected
runtime source and data into a **new** engine folder, retaining the supplied
README, licenses, dependency credits, configuration and provenance. `--model-repo`
and `--model-revision` name the checkpoint the run used; `model.json` lists its files
with the hashes the run recorded. `--contributor` and `--contributor-repo` credit your
repository under the README's title and in `provenance.json`. Keep the original sealed
run separately.

</details>

---

<a id="license"></a>

## License

Original content is licensed under [Apache-2.0](LICENSE). Third-party components
retain their own terms; see each engine's `THIRD_PARTY.md` and `LICENSES.txt`.
