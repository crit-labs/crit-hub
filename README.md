<div align="center">

# crit-hub

### Bespoke inference engines built with [crit](https://github.com/crit-labs/crit).

[Browse engines](engines/) • [Contributing](#contributing) • [License](#license)

</div>

---

### Engines

Browse [engines/](engines/) by GPU, model/encoding and engine design:

```text
engines/<gpu>/<model>@<encoding>/<name>/
  README.md          target, measured results, limits, build and chat
  engine/            source, configuration and required runtime data
  THIRD_PARTY.md     dependency credits, when applicable
  LICENSES.txt       applicable license texts and notices
```

Choose a folder that matches your hardware and model, then follow its README
to build and open chat. Requirements and measurements belong with each engine.
Several designs can share the same GPU and model/encoding.

---

<a id="contributing"></a>

### Contributing

Add an engine under the layout above, with a name that describes its design.
Include runnable source and a short README covering the target hardware and OS,
model and checkpoint revision, encoding, dependencies, build steps and chat command.
Document the measured workload, results, check coverage and known limits.

Retain dependency credits and applicable license texts and notices. Keep model
weights, builds, binaries, development scripts and full experiment records outside
the repository. The [.gitignore](.gitignore) uses a whitelist for published files.

<details>
<summary><strong>Export a sealed crit candidate</strong> → <a href="tools/export_engine.py">Export tool</a></summary>

<br />

```sh
python -B tools/export_engine.py --help
```

The exporter verifies the complete candidate seal and copies explicitly selected
runtime source and data into a **new** engine folder, retaining the supplied
README, licenses, dependency credits, configuration and provenance. Keep the
original sealed run separately.

</details>

---

<a id="license"></a>

### License

Original content is licensed under [Apache-2.0](LICENSE). Third-party components
retain their own terms; see each engine's `THIRD_PARTY.md` and `LICENSES.txt`.
