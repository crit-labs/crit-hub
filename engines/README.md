## Engine folder layout

Engines are listed by GPU, then model and encoding, whatever toolkit built them. Each one
records its origin in `engine/provenance.json`:

- **`crit`**: built with [crit](https://github.com/crit-labs/crit) and exported from a sealed
  candidate with `tools/export_engine.py`. The folder carries the runtime source.
- **`external`**: built with any other toolkit. The folder pins the source at a commit in the
  contributor's repository and carries no source of its own. crit-hub has not reviewed that
  code; read it before you build or run it.

```text
engines/<gpu>/<model>@<encoding>/<name>/        origin: crit
  README.md          contributor repository, target, measured results, limits, build and chat
  engine/            source, configuration, provenance.json and required runtime data
  model.json         the exact checkpoint: repository, revision, file sizes and SHA-256
  THIRD_PARTY.md     dependency credits, when applicable
  LICENSES.txt       applicable license texts and notices

engines/<gpu>/<model>@<encoding>/<name>/        origin: external
  README.md          the same template, opening with the external-engine notice
  engine/
    provenance.json  origin, contributor, source repository and commit, toolchain
  model.json         the exact checkpoint: repository, revision, file sizes and SHA-256
```

For a crit engine, `engine/provenance.json` records the sealed run and candidate, the toolchain
and the contributor. Check a model download against any engine's checkpoint with
`python -B tools/verify_model.py <engine>/model.json <model folder>`.

To add an engine, follow [Contributing](../README.md#contributing).
