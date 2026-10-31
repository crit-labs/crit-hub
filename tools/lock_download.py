"""List a drafter download's files in an engine's card.lock.json.

A drafter that card.yaml downloads by revision (kind: drafter) is checked by crit pull file by file, as the
checkpoint is: this writes the size and SHA-256 of every file of a local copy of it under the lock's downloads,
replacing an earlier entry for the same repository, revision and path. Download it first, for example with
huggingface-cli download REPO --revision REV --local-dir FOLDER, and pass the folder the engine loads (the
download's path inside the repository, when it names one).
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import card as cardlib  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files(folder):
    """Every file of the folder, download caches (.cache/, .gitattributes and other dot files) left out."""
    folder = Path(folder)
    rows = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder).as_posix()
        if any(part.startswith(".") for part in relative.split("/")):
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def lock(engine, folder, repository, revision, path=None):
    engine = Path(engine)
    card, value = cardlib.load(engine), cardlib.load_lock(engine)
    if value is None:
        raise SystemExit(f"{engine} has no card.lock.json yet; export the entry first")
    key = (repository, revision, path)
    if key not in {(d.get("repository"), d.get("revision"), d.get("path")) for d in card.get("downloads") or []
                   if d.get("kind") == "drafter"}:
        raise SystemExit(f"card.yaml downloads no drafter {repository} at {revision[:12]}" + (f" ({path})" if path else ""))
    rows = files(folder)
    if not rows:
        raise SystemExit(f"{folder} holds no files")
    kept = [d for d in value.get("downloads") or [] if (d.get("repository"), d.get("revision"), d.get("path")) != key]
    value["downloads"] = kept + [{"repository": repository, "revision": revision, **({"path": path} if path else {}),
                                  "files": rows}]
    cardlib.write(engine, card, value)
    print(f"card.lock.json lists {len(rows)} files of {repository}@{revision[:12]}" + (f"/{path}" if path else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("engine", help="the engine's folder, e.g. engines/crit-labs/mtp7-device-confidence")
    p.add_argument("folder", help="local copy of the drafter: the folder the engine loads")
    p.add_argument("--repository", required=True, help="the download's Hugging Face repository, as card.yaml names it")
    p.add_argument("--revision", required=True, help="the download's full 40-character revision, as card.yaml pins it")
    p.add_argument("--path", help="the download's path in the repository, when card.yaml names one")
    a = p.parse_args()
    lock(a.engine, a.folder, a.repository, a.revision, a.path)
