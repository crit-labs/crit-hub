"""Check that a local model folder is the exact checkpoint an engine was built on.

Reads the engine's model.json and compares every listed file's size and SHA-256.
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


def verify(model_json, folder):
    model = json.loads(Path(model_json).read_text(encoding="utf-8"))
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
    p.add_argument("model_json", help="the engine's model.json")
    p.add_argument("folder", help="local model folder, e.g. from huggingface-cli download REPO --revision REV")
    a = p.parse_args()
    verify(a.model_json, a.folder)
