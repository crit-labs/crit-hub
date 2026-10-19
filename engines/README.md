## Engine folder layout

```text
engines/<gpu>/<model>@<encoding>/<name>/
  README.md          contributor repository, target, measured results, limits, build and serve
  engine/            source, configuration, provenance.json and required runtime data
  model.json         the exact checkpoint: repository, revision, file sizes and SHA-256
  THIRD_PARTY.md     dependency credits, when applicable
  LICENSES.txt       applicable license texts and notices
```

`engine/provenance.json` records the sealed run and candidate, the toolchain and the
contributor. Check a model download against an engine's checkpoint with
`python -B tools/verify_model.py <engine>/model.json <model folder>`.

To add an engine, follow [Contributing](../README.md#contributing) and add a row to the table above
and to the README of its GPU and model folders.
