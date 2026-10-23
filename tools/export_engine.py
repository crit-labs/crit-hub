"""Export a sealed crit candidate in two steps: its source into crit-engines, then its crit-hub entry.

source copies the explicitly selected runtime source and data, the configuration, dependency credits
and the applicable license texts into <name>/ of a crit-engines checkout; never research, full
candidate records or executable packages, which stay in the run. Commit that folder in crit-engines.
entry then writes engines/<contributor>/<name>/ in crit-hub, pinned to the crit-engines commit that
holds the source (the checkout's HEAD, which must contain <name>/ with no uncommitted changes):
card.yaml with the supplied README's sections as its prose, card.lock.json with the exact checkpoint
and the sealed run's provenance, and README.md rendered from the card. The supplied README writes the
commit as <crit-engines commit>, which entry fills in. --eval names the candidate's sealed crit Eval
attempts (attempt-0003): entry copies each result into evidence/ as crit-eval.json (then
crit-eval-2.json and on) and fills the result's six stats and further figures from them; without
--eval, fill them in as your own report. A run started from a crit-hub engine (crit run create
--from-engine) gives the card derived_from, and a run with cost.json (crit run cost) gives it built.
Then fill in the summary, license, features and requirements in card.yaml and render the README with
tools/card.py before tools/check_engines.py passes. Both steps verify the complete seal, and both
destinations must be new.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import card as cardlib  # noqa: E402


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


REVISION = "<crit-engines commit>"


def sealed(args):
    """The run, and the candidate's verified seal, source and package."""
    for name in (args.name, args.candidate):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError("The name and candidate must each be a single folder name")
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
    return run, seal, candidate / "source", candidate / "package"


def export_source(args):
    run, seal, source, package = sealed(args)
    target = Path(args.engines).resolve() / args.name
    if target.exists():
        raise ValueError("Destination exists; archive it before replacing it")
    # Validate all caller-supplied inputs before creating the destination.
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
    # Dependency credits come from the sealed source unless the candidate predates them.
    third_party = Path(args.third_party).resolve() if args.third_party else source / "THIRD_PARTY.md"
    required = [third_party, package / "configuration.json"] + [path for path, _ in data]
    if not selected or any(not path.is_file() for path in required):
        raise ValueError("Incomplete runtime source, data or notices")
    refuse_local_paths(selected + data + [(third_party, "THIRD_PARTY.md"),
                                          (package / "configuration.json", "configuration.json")], run)
    engine = target / "engine"
    target.mkdir(parents=True)
    for path, name in selected:
        copy(path, child(engine, name))
    copy(third_party, target / "THIRD_PARTY.md")
    copy(package / "configuration.json", engine / "configuration.json")
    # Runtime data is explicit, never inferred from work/ or historical scripts.
    for path, name in data:
        copy(path, child(engine, name))
    text = "\n\n".join(f"{'=' * 72}\n{label}\n{'=' * 72}\n{path.read_text(encoding='utf-8')}" for label, path in licenses)
    (target / "LICENSES.txt").write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"export": str(target), "files": sum(p.is_file() for p in target.rglob("*")), "verified_seal_files": len(seal["files"]),
                      "next": f"commit {args.name}/ in crit-engines, then run entry"}))


def committed(engines, name):
    """The crit-engines commit that holds <name>/: HEAD, with nothing under <name>/ left uncommitted."""
    def git(*command):
        return subprocess.run(["git", "-C", str(engines), *command], capture_output=True, text=True)
    head = git("rev-parse", "HEAD").stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError(f"{engines} is not a crit-engines checkout with a commit")
    if not git("ls-tree", "-r", "--name-only", "HEAD", "--", f"{name}/engine").stdout.strip():
        raise ValueError(f"HEAD of {engines} has no {name}/engine/; run source and commit {name}/ first")
    if git("status", "--porcelain", "--", name).stdout.strip():
        raise ValueError(f"{name}/ in {engines} has uncommitted changes; commit them first")
    return head


def measured(args, seal, candidate, target):
    """The result for --gpu and --model: its six stats and further figures from the candidate's
    sealed crit Eval attempts, copied into evidence/, or left for the contributor to fill in.
    Creates the destination."""
    sealed_files = {entry["path"] for entry in seal["files"]}
    attempts, results = [], []
    for attempt in args.eval:
        if not re.fullmatch(r"attempt-[0-9]+", attempt) or f"eval/{attempt}.json" not in sealed_files:
            raise ValueError(f"No sealed crit Eval result eval/{attempt}.json in {args.candidate}")
        attempts.append(child(candidate, f"eval/{attempt}.json"))
        results.append(read(attempts[-1]))
        if not cardlib.crit_eval(results[-1].get("protocol")) or not (results[-1].get("speed") or {}).get("complete"):
            raise ValueError(f"eval/{attempt}.json is not a complete crit Eval speed run")
    if len({r["protocol"] for r in results}) > 1:
        raise ValueError("The --eval attempts ran under different crit Eval protocols")
    target.mkdir(parents=True)
    if not args.eval:
        return {"gpu": args.gpu, "model": args.model, "measured_with": None, "measured_on": None,
                "stats": {key: None for key in cardlib.STATS}, "evidence": []}
    evidence = []
    for number, path in enumerate(attempts, 1):
        name = "evidence/crit-eval.json" if number == 1 else f"evidence/crit-eval-{number}.json"
        copy(path, target / name)
        evidence.append({"file": name, "sha256": digest(target / name)})

    def figure(values, places):
        values = [round(v, places) if places else round(v) for v in values]
        return values[0] if len(values) == 1 else [min(values), max(values)]

    def worst(field):
        values = [s[field] for r in results for s in ((r.get("quality") or {}).get("sets") or {}).values()
                  if isinstance(s, dict) and isinstance(s.get(field), (int, float))]
        return max(values) if values else None

    tokens = {(r.get("context") or {}).get("tokens") for r in results}
    ppl, kld, budget = worst("ppl_ratio"), worst("kld_mean"), worst("budget_used")
    # From crit-eval-v7 the headline decode is the sampled suite's, the greedy one a further figure.
    sampled = cardlib.eval_major(results[0]["protocol"]) >= 7
    stats = {
        "decode_tok_s": figure([r["sampled_speed" if sampled else "speed"]["decode_score"] for r in results], 1),
        "prefill_tok_s": figure([r["speed"]["prefill_score"] for r in results], 0),
        "ttft_ms": figure([r["speed"]["ttft_score_ms"] for r in results], 0),
        "ppl": {"vs_reference": round(ppl, 3)} if ppl is not None else None,
        "kld_mean": float(f"{kld:.2g}") if kld is not None else None,
        "context_tokens": next(iter(tokens)) if len(tokens) == 1 and isinstance(next(iter(tokens)), int) else None,
    }
    more = {"greedy_decode_tok_s": figure([r["speed"]["decode_score"] for r in results], 1)} if sampled else {}
    context = (((results[0].get("status") or {}).get("kpis") or {}).get("context") or {}).get("decode", {}).get("score")
    if isinstance(context, (int, float)):
        more["long_context_decode_tok_s"] = round(context, 1)
    heldout = [(r.get("heldout") or {}).get("decode_score") for r in results]
    if all(isinstance(v, (int, float)) for v in heldout):
        more["heldout_decode_tok_s"] = figure(heldout, 1)
    if budget is not None:
        more["quality_budget_pct"] = round(budget * 100, 1)
    entry = {"gpu": args.gpu, "model": args.model, "measured_with": results[0]["protocol"],
             "measured_on": max(str(r.get("started_at", ""))[:10] for r in results)}
    # A sampled decode not at the model's generation_config.json says so on the card and in its caption.
    source = (results[0].get("sampled_speed") or {}).get("sampling_source") or "generation_config"
    if sampled and source != "generation_config":
        entry["sampling"] = {"source": source, "settings": results[0]["sampled_speed"]["sampling"]}
    return {**entry, "stats": stats, "more": more, "evidence": evidence}


def export_entry(args):
    for name in (args.gpu, args.model):
        if not re.fullmatch(r"[A-Za-z0-9@_.+-]+", name) or name in (".", ".."):
            raise ValueError("GPU/model must be a single folder name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.contributor) or args.contributor in (".", ".."):
        raise ValueError("The contributor names the engine's folder: use your GitHub account or organization")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", args.gpu) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*@[A-Za-z0-9.+-]+", args.model):
        raise ValueError("Name the GPU like rtx5090 and the model like qwen3.8-27b@NVFP4")
    run, seal, _, package = sealed(args)
    manifest = read(package / "manifest.json")
    target = Path(args.hub).resolve() / "engines" / args.contributor / args.name
    if target.exists():
        raise ValueError("Destination exists; archive it before replacing it")
    # Validate all caller-supplied inputs before creating the destination.
    readme = Path(args.readme).resolve()
    if not readme.is_file():
        raise ValueError("A concise engine README is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.model_repo) or not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        raise ValueError("Name the model as a Hugging Face repository and its full commit revision")
    if not args.contributor.strip() or "\n" in args.contributor or not REPOSITORY.fullmatch(args.contributor_repo):
        raise ValueError("Credit the contributor by name and the https URL of their repository")
    if not REPOSITORY.fullmatch(args.source_repo):
        raise ValueError("Name the https URL of the crit-engines repository the commit is pushed to")
    revision = committed(Path(args.engines).resolve(), args.name)
    refuse_local_paths([(readme, "README.md")], run)
    text = readme.read_text(encoding="utf-8")
    if REVISION not in text:
        raise ValueError(f"The README's Build and chat checks out {REVISION} of crit-engines; entry fills in the commit")
    title, sections = cardlib.split_readme(text.replace(REVISION, revision))
    provenance = {
        "run_id": seal["run_id"],
        "candidate_id": args.candidate,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "original_package_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "configuration_sha256": manifest["configuration_sha256"],
        "model_manifest_sha256": hashlib.sha256(json.dumps(read(run / "inputs.json")["model_manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "toolchain": manifest["build"]["toolchain"],
        "runtime_prerequisites": (manifest.get("extensions") or {}).get("runtime_prerequisites", []),
    }
    if LOCAL_PATH.search(json.dumps(provenance)) or run.parent.as_posix() in json.dumps(provenance):
        raise ValueError("Local path in the candidate's toolchain record; describe the toolchain without it")
    locked = checkpoint(args.model_repo, args.model_revision, read(run / "inputs.json")["model_manifest"])
    lineage = {}
    # A run started from a crit-hub engine (crit run create --from-engine) names it; the new engine derives from it.
    origin = read(run / "run.json")["spec"].get("origin_engine")
    if origin:
        lineage["derived_from"] = {"engine": origin["engine"],
                                   "source": {k: v for k, v in origin["source"].items() if v is not None}}
    # What the run took (crit run cost): hours, the agent's tokens, their API-equivalent price, the agent.
    if (run / "cost.json").is_file():
        cost = read(run / "cost.json")
        if cost.get("record_kind") != "run_cost" or cost.get("run_id") != seal["run_id"]:
            raise ValueError("cost.json is not this run's run_cost record")
        built = {"hours": cost["run"]["hours"], "agent_tokens": cost["run"]["tokens"]["total"], "api_usd": cost["run"]["usd"],
                 "agent": cost.get("agent"), "candidates": len(cost["candidates"])}
        lineage["built"] = {k: v for k, v in built.items() if v is not None}
    # The summary, license, features and requirements are the contributor's to fill in.
    card = {
        "title": title,
        "summary": None,
        "origin": "crit",
        "contributor": {"name": args.contributor.strip(), "repository": args.contributor_repo},
        "license": None,
        "features": [],
        "source": {"repository": args.source_repo, "revision": revision, "path": args.name},
        "requirements": {"platforms": [], "gpu_arch": [], "gpu_memory_gb": None, "cuda": None},
        "checkpoints": {args.model: {"repository": locked["repository"], "revision": locked["revision"]}},
        **lineage,
        "results": [measured(args, seal, package.parent, target)],
        "readme": sections,
    }
    cardlib.write(target, card, {"checkpoints": {args.model: locked}, "provenance": provenance})
    (target / "README.md").write_text(cardlib.render(card), encoding="utf-8", newline="\n")
    print(json.dumps({"export": str(target), "source": f"{args.source_repo}/tree/{revision}/{args.name}", "verified_seal_files": len(seal["files"]),
                      "next": "fill in summary, license, features and requirements in card.yaml, then python -B tools/card.py on the folder"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = p.add_subparsers(dest="step", required=True)
    for step, about in (("source", "copy the candidate's runtime source into a crit-engines checkout"),
                        ("entry", "write its crit-hub entry, pinned to the crit-engines commit that holds the source")):
        q = steps.add_parser(step, help=about, description=about)
        q.add_argument("--run", required=True)
        q.add_argument("--candidate", required=True)
        q.add_argument("--engines", required=True, help="crit-engines checkout")
        q.add_argument("--name", required=True, help="engine name describing its design, e.g. mtp6-cleanroom: <name>/ in crit-engines and engines/<contributor>/<name>/ in crit-hub")
        if step == "source":
            q.add_argument("--third-party", help="THIRD_PARTY.md to use when the sealed source has none")
            q.add_argument("--runtime-source", action="append", default=[])
            q.add_argument("--runtime-data", action="append", default=[])
            q.add_argument("--license", action="append", default=[])
            continue
        q.add_argument("--hub", required=True, help="crit-hub repository root")
        q.add_argument("--source-repo", default="https://github.com/crit-labs/crit-engines", help="https URL the crit-engines commit is pushed to (default: %(default)s)")
        q.add_argument("--gpu", required=True, help="GPU id the candidate was measured on, e.g. rtx5090")
        q.add_argument("--model", required=True, help="model id, <model>@<encoding>, e.g. qwen3.8-27b@NVFP4")
        q.add_argument("--model-repo", required=True, help="Hugging Face repository, e.g. unsloth/Qwen3.8-27B-NVFP4")
        q.add_argument("--model-revision", required=True, help="full commit of that repository the run used")
        q.add_argument("--contributor", required=True, help="your GitHub account or organization: credited in the README and the engine's folder, engines/<contributor>/<name>/")
        q.add_argument("--contributor-repo", required=True, help="https URL of your repository, shown under the README's title")
        q.add_argument("--readme", required=True, help=f"the engine README; it writes the crit-engines commit as {REVISION}")
        q.add_argument("--eval", action="append", default=[], metavar="ATTEMPT",
                       help="a sealed crit Eval attempt of the candidate, e.g. attempt-0003: its result goes to evidence/ and gives the "
                            "result's figures; repeat it for a range across runs")
    args = p.parse_args()
    (export_source if args.step == "source" else export_entry)(args)
