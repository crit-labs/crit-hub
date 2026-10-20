"""Check every engine folder against the layout for its origin.

A crit engine needs its sealed-run provenance and the built-with-crit line under its credit.
An external engine holds exactly README.md, model.json and engine/provenance.json, pins its
source to a full commit SHA and opens its README with the external-engine notice.
"""
import argparse
import json
import re
from pathlib import Path

REPOSITORY = re.compile(r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9_.~-]+)+/?")
COMMIT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
EXTERNAL_FILES = {"README.md", "model.json", "engine/provenance.json"}


def check_model(model):
    problems = []
    if not all(isinstance(model.get(k), str) and model[k] for k in ("repository", "revision")):
        problems.append("model.json needs the checkpoint repository and revision")
    files = model.get("files")
    if not isinstance(files, list) or not files:
        return problems + ["model.json lists no checkpoint files"]
    for entry in files:
        if not (isinstance(entry.get("path"), str) and isinstance(entry.get("bytes"), int)
                and SHA256.fullmatch(str(entry.get("sha256")))):
            problems.append(f"model.json entry needs path, bytes and sha256: {entry.get('path')}")
    return problems


def check_contributor(provenance):
    contributor = provenance.get("contributor") or {}
    if not str(contributor.get("name", "")).strip() or not REPOSITORY.fullmatch(str(contributor.get("repository"))):
        return ["provenance.json needs contributor.name and the https URL of contributor.repository"]
    return []


def check_crit(folder, provenance, head):
    problems = check_contributor(provenance)
    if not provenance.get("run_id") or not provenance.get("candidate_id"):
        problems.append("provenance.json needs the sealed run_id and candidate_id")
    if not any(line.startswith("> **Contributed by**") for line in head):
        problems.append("README.md needs the credit directly under its title")
    if not any(line.startswith("> Built with [crit]") for line in head):
        problems.append("README.md needs the built-with-crit line under the credit")
    return problems


def check_external(folder, provenance, head, text):
    problems = check_contributor(provenance)
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if files - EXTERNAL_FILES:
        problems.append("an external engine holds only README.md, model.json and engine/provenance.json; "
                        f"also found {len(files - EXTERNAL_FILES)} other file(s), e.g. {sorted(files - EXTERNAL_FILES)[0]}")
    source = provenance.get("source") or {}
    if not REPOSITORY.fullmatch(str(source.get("repository"))):
        problems.append("provenance.json needs the https URL of source.repository")
    revision = str(source.get("revision"))
    if not COMMIT.fullmatch(revision):
        problems.append("provenance.json source.revision must be a full 40-character commit SHA")
    elif revision not in text:
        problems.append("README.md must link the pinned source commit")
    if "> [!WARNING]" not in head or not any(line.startswith("> **External engine.**") for line in head):
        problems.append("README.md must open with the external-engine notice")
    return problems


def check(folder):
    for name in ("README.md", "model.json", "engine/provenance.json"):
        if not (folder / name).is_file():
            return [f"missing {name}"]
    try:
        provenance = json.loads((folder / "engine/provenance.json").read_text(encoding="utf-8"))
        model = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]
    text = (folder / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        return ["README.md must open with a '# ' title"]
    head = lines[1:12]
    problems = check_model(model)
    origin = provenance.get("origin")
    if origin == "crit":
        problems += check_crit(folder, provenance, head)
    elif origin == "external":
        problems += check_external(folder, provenance, head, text)
    else:
        problems.append('provenance.json needs "origin": "crit" or "external"')
    return problems


def main(hub):
    folders = sorted(p for p in (Path(hub) / "engines").glob("*/*/*") if p.is_dir())
    failed = 0
    for folder in folders:
        problems = check(folder)
        name = folder.relative_to(Path(hub) / "engines").as_posix()
        if problems:
            failed += 1
            print(f"FAIL {name}\n  " + "\n  ".join(problems))
        else:
            print(f"ok   {name}")
    if failed:
        raise SystemExit(f"{failed} of {len(folders)} engine folders need changes")
    print(f"{len(folders)} engine folders match their layout")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hub", default=".", help="crit-hub checkout (default: the current folder)")
    main(p.parse_args().hub)
