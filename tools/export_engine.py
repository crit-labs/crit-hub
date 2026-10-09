"""Export a sealed crit candidate as a compact source delivery; never copy research.

The destination must be new. Supply a short README and the applicable original
license files. Full candidate records and executable packages stay in the run.
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


def export(args):
    for name in (args.gpu, args.model, args.name, args.candidate):
        if not re.fullmatch(r"[A-Za-z0-9@_.+-]+", name) or name in (".", ".."):
            raise ValueError("GPU/model/name/candidate must be a single folder name")
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
    target = Path(args.hub).resolve() / "engines" / args.gpu / args.model / args.name
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
    required = [source / "THIRD_PARTY.md", package / "configuration.json"] + [path for path, _ in data]
    if not selected or any(not path.is_file() for path in required):
        raise ValueError("Incomplete runtime source, data or notices")
    engine = target / "engine"
    target.mkdir(parents=True)
    for path, name in selected:
        copy(path, child(engine, name))
    copy(source / "THIRD_PARTY.md", target / "THIRD_PARTY.md")
    copy(readme, target / "README.md")
    copy(package / "configuration.json", engine / "configuration.json")
    # Runtime data is explicit, never inferred from work/ or historical scripts.
    for path, name in data:
        copy(path, child(engine, name))
    text = "\n\n".join(f"{'=' * 72}\n{label}\n{'=' * 72}\n{path.read_text(encoding='utf-8')}" for label, path in licenses)
    (target / "LICENSES.txt").write_text(text + "\n", encoding="utf-8")
    provenance = {
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "original_package_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "configuration_sha256": manifest["configuration_sha256"],
        "model_manifest_sha256": hashlib.sha256(json.dumps(read(run / "inputs.json")["model_manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "toolchain": manifest["build"]["toolchain"],
        "runtime_prerequisites": manifest.get("extensions", {}).get("runtime_prerequisites", []),
    }
    (engine / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"export": str(target), "files": sum(p.is_file() for p in target.rglob("*")), "verified_seal_files": len(seal["files"])}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--hub", required=True, help="crit-hub repository root")
    p.add_argument("--gpu", required=True)
    p.add_argument("--model", required=True, help="<model>@<encoding>, e.g. qwen3.8-27b@NVFP4")
    p.add_argument("--name", required=True, help="engine name describing its design, e.g. mtp6-cleanroom")
    p.add_argument("--readme", required=True)
    p.add_argument("--runtime-source", action="append", default=[])
    p.add_argument("--runtime-data", action="append", default=[])
    p.add_argument("--license", action="append", default=[])
    export(p.parse_args())
