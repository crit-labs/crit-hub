"""Check that a local model folder is the exact checkpoint an engine was measured on.

Reads the checkpoint from the engine's engine.json and compares every listed file's size and SHA-256.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(engine, folder, model_id=None):
    checkpoints = json.loads((Path(engine) / "engine.json").read_text(encoding="utf-8"))["checkpoints"]
    if model_id is None and len(checkpoints) != 1:
        raise SystemExit(f"{engine} pins several checkpoints; name one with --model: {', '.join(sorted(checkpoints))}")
    model = checkpoints[model_id] if model_id else next(iter(checkpoints.values()))
    root = Path(folder)
    bad = []
    for entry in model["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            bad.append(f"{entry['path']}: missing or wrong size")
        elif digest(path) != entry["sha256"]:
            bad.append(f"{entry['path']}: SHA-256 differs")
    name = f"{model['repository']}@{model['revision']}"
    if bad:
        raise SystemExit(f"{folder} is not {name}:\n  " + "\n  ".join(bad))
    print(f"{folder} matches {name} ({len(model['files'])} files)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", help="the engine's folder, e.g. engines/crit-labs/mtp7-fp8attn")
    p.add_argument("folder", help="local model folder, e.g. from huggingface-cli download REPO --revision REV")
    p.add_argument("--model", help="the model id, when the engine pins several checkpoints")
    a = p.parse_args()
    verify(a.engine, a.folder, a.model)
