"""Export a sealed crit candidate as a compact source delivery; never copy research.

The destination, engines/<contributor>/<name>/, must be new. Supply a short README and the
applicable original license files. Full candidate records and executable packages stay in the
run. The exported engine.json carries the provenance and the exact checkpoint, and names the GPU
and model the candidate was measured on; fill in the summary, features, measured_with and the
README's measured figures, and the SPDX license of the engine source, before tools/check_engines.py
passes.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def child(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe retained path: {name}")
    path = root / str(relative)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Escaping retained path: {name}")
    return path


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


LOCAL_PATH = re.compile(r"/mnt/[a-z]/|/home/[^/\s]+/|/root/|/Users/[^/\s]+/|[A-Za-z]:[\\/]+Users[\\/]|wsl\.localhost", re.IGNORECASE)


def refuse_local_paths(files, run):
    """Published text must not name the exporter's machine: home folders, mounts or the run's own location."""
    workspace = run.parent.as_posix()
    for path, name in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if LOCAL_PATH.search(line) or workspace in line:
                raise ValueError(f"Local path in {name}:{number}: {line.strip()[:120]}")


def checkpoint(repository, revision, run_manifest):
    """The exact checkpoint the candidate was built and measured on; download caches are not model files."""
    files = [{k: e[k] for k in ("path", "bytes", "sha256")} for e in run_manifest
             if not PurePosixPath(e["path"]).parts[0].startswith(".")]
    return {"repository": repository, "revision": revision, "files": files,
            "run_model_manifest_sha256": hashlib.sha256(json.dumps(run_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()}


REPOSITORY = re.compile(r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9_.~-]+)+/?")


BUILT_WITH_CRIT = "> Built with [crit](https://github.com/crit-labs/crit): a sealed crit candidate, exported with `tools/export_engine.py`."


def credit(text, name, repository):
    """The contributor's repository sits directly under the engine README's title."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("The engine README must open with a '# ' title")
    if any(line.startswith("> **Contributed by**") and repository in line for line in lines[1:6]):
        return text
    shown = repository.split("://", 1)[1].split("/", 1)[1].rstrip("/")
    rest = "\n".join(lines[1:]).lstrip("\n")
    return f"{lines[0]}\n\n> [!NOTE]\n> **Contributed by** {name} · [{shown}]({repository})\n>\n{BUILT_WITH_CRIT}\n\n{rest}\n"


def export(args):
    for name in (args.gpu, args.model, args.name, args.candidate):
        if not re.fullmatch(r"[A-Za-z0-9@_.+-]+", name) or name in (".", ".."):
            raise ValueError("GPU/model/name/candidate must be a single folder name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.contributor) or args.contributor in (".", ".."):
        raise ValueError("The contributor names the engine's folder: use your GitHub account or organization")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", args.gpu) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*@[A-Za-z0-9.+-]+", args.model):
        raise ValueError("Name the GPU like rtx5090 and the model like qwen3.8-27b@NVFP4")
    run = Path(args.run).resolve()
    candidate = child(run, f"candidates/{args.candidate}")
    seal = read(candidate / "seal.json")
    if seal.get("record_kind") != "candidate_seal" or seal.get("candidate_id") != args.candidate:
        raise ValueError("A sealed candidate is required")
    # Verify the complete seal, including source, package, checks and measurements.
    for entry in seal["files"]:
        path = child(candidate, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Seal mismatch: {entry['path']}")
    source = candidate / "source"
    package = candidate / "package"
    manifest = read(package / "manifest.json")
    target = Path(args.hub).resolve() / "engines" / args.contributor / args.name
    if target.exists():
        raise ValueError("Destination exists; archive it before replacing it")
    # Validate all caller-supplied inputs before creating the destination.
    readme = Path(args.readme).resolve()
    if not readme.is_file():
        raise ValueError("A concise engine README is required")
    licenses = []
    for item in args.license:
        label, filename = item.split("=", 1)
        path = Path(filename).resolve()
        if not label or not path.is_file():
            raise ValueError(f"Missing license input: {item}")
        licenses.append((label, path))
    if not licenses:
        raise ValueError("Retain applicable licenses, including original contributions")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.model_repo) or not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        raise ValueError("Name the model as a Hugging Face repository and its full commit revision")
    if not args.contributor.strip() or "\n" in args.contributor or not REPOSITORY.fullmatch(args.contributor_repo):
        raise ValueError("Credit the contributor by name and the https URL of their repository")
    if not args.runtime_source:
        raise ValueError("Select the engine's runtime source explicitly")
    selected = []
    for name in args.runtime_source:
        path = child(source, name)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        if not path.exists():
            raise ValueError(f"Runtime source missing: {name}")
        for file in files:
            if file.is_file() and "__pycache__" not in file.parts:
                relative = file.relative_to(source).as_posix()
                selected.append((child(source, relative), relative))
    data = [(child(package, name), name) for name in args.runtime_data]
    # Dependency credits come from the sealed source unless the candidate predates them.
    third_party = Path(args.third_party).resolve() if args.third_party else source / "THIRD_PARTY.md"
    required = [third_party, package / "configuration.json"] + [path for path, _ in data]
    if not selected or any(not path.is_file() for path in required):
        raise ValueError("Incomplete runtime source, data or notices")
    refuse_local_paths(selected + data + [(third_party, "THIRD_PARTY.md"), (readme, "README.md"),
                                          (package / "configuration.json", "configuration.json")], run)
    credited = credit(readme.read_text(encoding="utf-8"), args.contributor.strip(), args.contributor_repo)
    engine = target / "engine"
    target.mkdir(parents=True)
    for path, name in selected:
        copy(path, child(engine, name))
    copy(third_party, target / "THIRD_PARTY.md")
    (target / "README.md").write_text(credited, encoding="utf-8", newline="\n")
    copy(package / "configuration.json", engine / "configuration.json")
    # Runtime data is explicit, never inferred from work/ or historical scripts.
    for path, name in data:
        copy(path, child(engine, name))
    text = "\n\n".join(f"{'=' * 72}\n{label}\n{'=' * 72}\n{path.read_text(encoding='utf-8')}" for label, path in licenses)
    (target / "LICENSES.txt").write_text(text + "\n", encoding="utf-8")
    provenance = {
        "run_id": seal["run_id"],
        "candidate_id": args.candidate,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "original_package_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "configuration_sha256": manifest["configuration_sha256"],
        "model_manifest_sha256": hashlib.sha256(json.dumps(read(run / "inputs.json")["model_manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "toolchain": manifest["build"]["toolchain"],
        "runtime_prerequisites": manifest.get("extensions", {}).get("runtime_prerequisites", []),
    }
    if LOCAL_PATH.search(json.dumps(provenance)) or run.parent.as_posix() in json.dumps(provenance):
        raise ValueError("Local path in the candidate's toolchain record; remove the export and describe the toolchain without it")
    # The summary, features, protocol and measured figures come from the README; the contributor fills them in.
    engine_json = {
        "summary": None,
        "origin": "crit",
        "contributor": {"name": args.contributor.strip(), "repository": args.contributor_repo},
        "license": None,
        "features": [],
        "results": [{"gpu": args.gpu, "model": args.model, "measured_with": None, "decode_tok_s": None}],
        "provenance": provenance,
        "checkpoints": {args.model: checkpoint(args.model_repo, args.model_revision, read(run / "inputs.json")["model_manifest"])},
    }
    (target / "engine.json").write_text(json.dumps(engine_json, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"export": str(target), "files": sum(p.is_file() for p in target.rglob("*")), "verified_seal_files": len(seal["files"])}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--hub", required=True, help="crit-hub repository root")
    p.add_argument("--gpu", required=True, help="GPU id the candidate was measured on, e.g. rtx5090")
    p.add_argument("--model", required=True, help="model id, <model>@<encoding>, e.g. qwen3.8-27b@NVFP4")
    p.add_argument("--name", required=True, help="engine name describing its design, e.g. mtp6-cleanroom")
    p.add_argument("--model-repo", required=True, help="Hugging Face repository, e.g. unsloth/Qwen3.8-27B-NVFP4")
    p.add_argument("--model-revision", required=True, help="full commit of that repository the run used")
    p.add_argument("--contributor", required=True, help="your GitHub account or organization: credited in the README and the engine's folder, engines/<contributor>/<name>/")
    p.add_argument("--contributor-repo", required=True, help="https URL of your repository, shown under the README's title")
    p.add_argument("--readme", required=True)
    p.add_argument("--third-party", help="THIRD_PARTY.md to use when the sealed source has none")
    p.add_argument("--runtime-source", action="append", default=[])
    p.add_argument("--runtime-data", action="append", default=[])
    p.add_argument("--license", action="append", default=[])
    export(p.parse_args())
