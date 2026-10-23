"""Check every engine folder against the layout for its origin.

Each engine lives in engines/<owner>/<name>/, where <owner> is its contributor, and describes
itself in engine.json: summary, origin, contributor, license, features, the GPU/model pairs it
measured (results) or supports, what it needs to run (requirements) and fetch besides its checkpoint
(downloads), its provenance and the exact Hugging Face checkpoint of every model it measures or
supports.
Every engine holds only README.md and engine.json and pins its source to a full commit SHA in a
repository, optionally a folder in it, which its README links. A crit engine's source is in
crit-labs/crit-engines; it needs its sealed-run provenance and the built-with-crit line under its
credit. An external engine opens its README with the external-engine notice.
"""
import argparse
import json
import re
from pathlib import Path

REPOSITORY = re.compile(r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9_.~-]+)+/?")
COMMIT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9_.-]+")
GPU = re.compile(r"[a-z0-9][a-z0-9.-]*")
MODEL = re.compile(r"[a-z0-9][a-z0-9.-]*@[A-Za-z0-9.+-]+")
FEATURES = {"openai-api", "tool-calls", "vision", "speculative"}
HF_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
FIELDS = {"summary", "origin", "contributor", "license", "features", "results", "supports", "requirements", "downloads",
          "provenance", "checkpoints"}
RESULT_FIELDS = {"gpu", "model", "measured_with", "decode_tok_s", "prefill_tok_s", "ttft_ms", "context_tokens", "host_ram_gib",
                 "quality"}
QUALITY_METRICS = {"budget_pct", "kld_mean", "perplexity"}
PLATFORMS = {"linux", "wsl2", "windows", "macos"}
GPU_ARCH = re.compile(r"sm_[0-9]{2,3}[a-z]?")
CUDA = re.compile(r"[0-9]+\.[0-9]+")
REQUIREMENT_FIELDS = {"platforms", "gpu_arch", "gpu_memory_gb", "cuda", "runtime"}
DOWNLOAD_KINDS = {"drafter", "artifact"}
DOWNLOAD_FIELDS = {"kind", "repository", "revision", "path", "file", "sha256", "size_gb", "note"}
ENGINE_FILES = {"README.md", "engine.json"}


def check_checkpoint(model, checkpoint):
    name = f"checkpoints[{model}]"
    if not isinstance(checkpoint, dict):
        return [f"engine.json {name} is the checkpoint's repository, revision and files"]
    problems = []
    if not all(isinstance(checkpoint.get(k), str) and checkpoint[k] for k in ("repository", "revision")):
        problems.append(f"engine.json {name} needs the checkpoint repository and revision")
    files = checkpoint.get("files")
    if not isinstance(files, list) or not files:
        return problems + [f"engine.json {name} lists no checkpoint files"]
    for entry in files:
        if not (isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("bytes"), int)
                and SHA256.fullmatch(str(entry.get("sha256")))):
            problems.append(f"engine.json {name} file entry needs path, bytes and sha256: {entry.get('path') if isinstance(entry, dict) else entry}")
    return problems


def numbers(value):
    """A figure is a positive number, or [low, high] across runs."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [value] if value > 0 else None
    if isinstance(value, list) and len(value) == 2 and all(numbers(v) for v in value) and value[0] <= value[1]:
        return value
    return None


def scientific(value):
    """0.0051 as 5.1e-3."""
    mantissa, exponent = f"{value:e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def written(value, text, key):
    """The README writes 8917 as 8,917, a time in ms or s (1610 ms as 1.61 s) and a small quality
    figure as 0.0051 or 5.1e-3."""
    values = [value, round(value / 1000, 3)] if key == "ttft_ms" else [value]
    return any(f"{v:,}" in text for v in values) or (key == "quality" and scientific(value) in text)


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def check_quality(pair, quality, text):
    """A quality figure: the share of crit Eval's quality budget the worst path uses (budget_pct),
    a KLD mean against its budget (kld_mean) or a perplexity, each as the README writes it."""
    if not (isinstance(quality, dict) and quality.get("metric") in QUALITY_METRICS and positive(quality.get("value"))
            and set(quality) <= {"metric", "value", "budget", "method"}):
        return [f"engine.json result {pair}: quality is {{\"metric\": one of {', '.join(sorted(QUALITY_METRICS))}, "
                f"\"value\": a positive number}}, with an optional budget and method"]
    problems = []
    if "budget" in quality and not positive(quality["budget"]):
        problems.append(f"engine.json result {pair}: quality budget is a positive number")
    if "method" in quality and not (isinstance(quality["method"], str) and quality["method"].strip()):
        problems.append(f"engine.json result {pair}: quality method says how it was measured")
    if not written(quality["value"], text, "quality"):
        problems.append(f"engine.json result {pair}: quality {quality['value']} is not in README.md")
    return problems


def check_results(engine, text):
    problems = []
    results = engine.get("results")
    if not isinstance(results, list) or not results:
        return ["engine.json needs results: at least one measured GPU and model"]
    seen = set()
    for result in results:
        if not isinstance(result, dict):
            problems.append("engine.json results are objects")
            continue
        pair = f"{result.get('gpu')} · {result.get('model')}"
        if set(result) - RESULT_FIELDS:
            problems.append(f"engine.json result {pair}: unknown field(s) {', '.join(sorted(set(result) - RESULT_FIELDS))}")
        if not GPU.fullmatch(str(result.get("gpu"))) or not MODEL.fullmatch(str(result.get("model"))):
            problems.append(f"engine.json result {pair}: gpu is an id like rtx5090, model one like qwen3.8-27b@NVFP4")
            continue
        if (result["gpu"], result["model"]) in seen:
            problems.append(f"engine.json measures {pair} twice")
        seen.add((result["gpu"], result["model"]))
        if not isinstance(result.get("measured_with"), str) or not result["measured_with"].strip():
            problems.append(f'engine.json result {pair} needs measured_with: "crit-eval-v5" or how else it was measured')
        if "decode_tok_s" not in result:
            problems.append(f"engine.json result {pair} needs decode_tok_s")
        for key in ("decode_tok_s", "prefill_tok_s", "ttft_ms"):
            if key not in result:
                continue
            values = numbers(result[key])
            if values is None:
                problems.append(f"engine.json result {pair}: {key} is a positive number or [low, high]")
                continue
            for value in values:
                if not written(value, text, key):
                    problems.append(f"engine.json result {pair}: {key} {value} is not in README.md")
        for key in ("context_tokens", "host_ram_gib"):
            if key in result and not (isinstance(result[key], int) and not isinstance(result[key], bool) and result[key] > 0):
                problems.append(f"engine.json result {pair}: {key} is a positive whole number")
        if "quality" in result:
            problems += check_quality(pair, result["quality"], text)
    return problems


def check_checkpoints(engine):
    """Every measured or supported model pins its Hugging Face checkpoint, measured ones with the
    hashes of its files."""
    checkpoints = engine.get("checkpoints")
    if not isinstance(checkpoints, dict):
        return ["engine.json checkpoints pins the exact checkpoint of every measured or supported model, keyed by model"]
    measured = {str(r.get("model")) for r in engine.get("results") or [] if isinstance(r, dict)}
    supported = {str(m) for group in engine.get("supports") or [] if isinstance(group, dict)
                 and isinstance(group.get("models"), list) for m in group["models"]} - measured
    problems = []
    for model in sorted(measured - set(checkpoints)):
        problems.append(f"engine.json checkpoints needs the checkpoint {model} was measured on")
    for model in sorted(supported - set(checkpoints)):
        problems.append(f'engine.json checkpoints needs the Hugging Face checkpoint {model} runs: '
                        f'"{model}": {{"repository": "<owner>/<name>", "revision": "<40-character commit>"}}')
    for model in sorted(set(checkpoints) - measured - supported):
        problems.append(f"engine.json checkpoints lists {model}, which it neither measures nor supports")
    for model in sorted(set(checkpoints) & (measured | supported)):
        checkpoint = checkpoints[model]
        if model in measured:
            problems += check_checkpoint(model, checkpoint)
        elif not (isinstance(checkpoint, dict) and isinstance(checkpoint.get("repository"), str) and checkpoint["repository"]):
            problems.append(f"engine.json checkpoints[{model}] needs the Hugging Face repository and revision")
            continue
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("repository"), str):
            continue
        revision = str(checkpoint.get("revision"))
        if not COMMIT.fullmatch(revision):
            problems.append(f"engine.json checkpoints[{model}] revision must be the full 40-character commit")
    return problems


def check_supports(engine):
    supports = engine.get("supports", [])
    if not isinstance(supports, list):
        return ["engine.json supports is a list of {gpus, models} groups"]
    problems = []
    for number, group in enumerate(supports, 1):
        if not (isinstance(group, dict) and set(group) == {"gpus", "models"}
                and isinstance(group["gpus"], list) and group["gpus"] and all(GPU.fullmatch(str(g)) for g in group["gpus"])
                and isinstance(group["models"], list) and group["models"] and all(MODEL.fullmatch(str(m)) for m in group["models"])):
            problems.append(f"engine.json supports group {number} is {{\"gpus\": [ids], \"models\": [ids]}}")
    return problems


def check_requirements(engine):
    """What the engine needs to run: the platforms it was built and run on, the GPU architectures it
    is compiled for, the GPU memory it needs, the CUDA toolkit and, optionally, the runtime it links."""
    requirements = engine.get("requirements")
    if not isinstance(requirements, dict):
        return ['engine.json needs requirements: {"platforms": ["linux"], "gpu_arch": ["sm_120a"], '
                '"gpu_memory_gb": 32, "cuda": "13.1"}, plus an optional runtime']
    problems = []
    if set(requirements) - REQUIREMENT_FIELDS:
        problems.append(f"engine.json requirements: unknown field(s) {', '.join(sorted(set(requirements) - REQUIREMENT_FIELDS))}")
    platforms = requirements.get("platforms")
    if not (isinstance(platforms, list) and platforms and set(platforms) <= PLATFORMS and len(set(platforms)) == len(platforms)):
        problems.append(f"engine.json requirements.platforms lists any of {', '.join(sorted(PLATFORMS))}")
    arch = requirements.get("gpu_arch")
    if not (isinstance(arch, list) and arch and all(GPU_ARCH.fullmatch(str(a)) for a in arch)):
        problems.append("engine.json requirements.gpu_arch lists the compute capabilities it is built for, like sm_120a")
    if not positive(requirements.get("gpu_memory_gb")):
        problems.append("engine.json requirements.gpu_memory_gb is the GPU memory the engine needs, in GB")
    if not CUDA.fullmatch(str(requirements.get("cuda"))):
        problems.append('engine.json requirements.cuda is the CUDA toolkit version it builds with, like "13.1"')
    if "runtime" in requirements and not (isinstance(requirements["runtime"], str) and requirements["runtime"].strip()):
        problems.append('engine.json requirements.runtime names what the engine links, like "crit Core 0.2.1"')
    return problems


def check_downloads(engine):
    """What to fetch besides the pinned checkpoint, each pinned by a commit or a SHA-256."""
    downloads = engine.get("downloads", [])
    if not isinstance(downloads, list):
        return ["engine.json downloads is a list of what to fetch besides the checkpoint"]
    problems = []
    for number, item in enumerate(downloads, 1):
        name = f"engine.json downloads[{number}]"
        if not isinstance(item, dict):
            problems.append(f"{name} is an object")
            continue
        if set(item) - DOWNLOAD_FIELDS:
            problems.append(f"{name}: unknown field(s) {', '.join(sorted(set(item) - DOWNLOAD_FIELDS))}")
        if item.get("kind") not in DOWNLOAD_KINDS:
            problems.append(f"{name}: kind is one of {', '.join(sorted(DOWNLOAD_KINDS))}")
        if not HF_REPOSITORY.fullmatch(str(item.get("repository"))):
            problems.append(f"{name}: repository is a Hugging Face repository, <owner>/<name>")
        if "revision" in item and not COMMIT.fullmatch(str(item["revision"])):
            problems.append(f"{name}: revision is a full 40-character commit")
        if "sha256" in item and not SHA256.fullmatch(str(item["sha256"])):
            problems.append(f"{name}: sha256 is the file's 64-character SHA-256")
        if "revision" not in item and "sha256" not in item:
            problems.append(f"{name}: pin it with a revision or the sha256 of its file")
        if "path" in item and not all(NAME.fullmatch(part) and part not in (".", "..") for part in str(item["path"]).split("/")):
            problems.append(f"{name}: path is a folder in the repository")
        if "file" in item and not NAME.fullmatch(str(item["file"])):
            problems.append(f"{name}: file is a file name")
        if "size_gb" in item and not positive(item["size_gb"]):
            problems.append(f"{name}: size_gb is a positive number")
        if "note" in item and not (isinstance(item["note"], str) and item["note"].strip() and "\n" not in item["note"]):
            problems.append(f"{name}: note is one line")
    return problems


def check_card(folder, engine):
    problems = []
    summary = engine.get("summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 120 or "\n" in summary:
        problems.append("engine.json needs summary: one line of at most 120 characters on what the engine does")
    if not isinstance(engine.get("license"), str) or not engine["license"].strip():
        problems.append('engine.json needs license: the SPDX id of the engine source, e.g. "Apache-2.0"')
    features = engine.get("features")
    if not isinstance(features, list) or set(features) - FEATURES or len(set(features)) != len(features):
        problems.append(f"engine.json features lists any of {', '.join(sorted(FEATURES))}")
    contributor = engine.get("contributor")
    if not isinstance(contributor, dict) or not str(contributor.get("name", "")).strip() \
            or not REPOSITORY.fullmatch(str(contributor.get("repository"))):
        problems.append("engine.json needs contributor.name and the https URL of contributor.repository")
    elif contributor["name"] != folder.parent.name:
        problems.append(f"the engine's folder is engines/<owner>/<name>/ with <owner> its contributor.name, "
                        f"{contributor['name']}, not {folder.parent.name}")
    if not isinstance(engine.get("provenance"), dict):
        problems.append("engine.json needs provenance")
    return problems


def check_source(folder, provenance, text):
    problems = []
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if files - ENGINE_FILES:
        extra = sorted(files - ENGINE_FILES)
        problems.append(f"an engine folder holds only README.md and engine.json; also found {len(extra)} other file(s), e.g. {extra[0]}")
    source = provenance.get("source") or {}
    if not REPOSITORY.fullmatch(str(source.get("repository"))):
        problems.append("engine.json provenance needs the https URL of source.repository")
    revision = str(source.get("revision"))
    if not COMMIT.fullmatch(revision):
        problems.append("engine.json provenance source.revision must be a full 40-character commit SHA")
    elif revision not in text:
        problems.append("README.md must link the pinned source commit")
    if "path" in source and not (isinstance(source["path"], str) and source["path"]
                                 and all(NAME.fullmatch(part) and part not in (".", "..") for part in source["path"].split("/"))):
        problems.append("engine.json provenance source.path is a folder in the source repository, like mtp7-fp8attn")
    return problems


def check_crit(provenance, head):
    problems = []
    if not provenance.get("run_id") or not provenance.get("candidate_id"):
        problems.append("engine.json provenance needs the sealed run_id and candidate_id")
    if not SHA256.fullmatch(str(provenance.get("source_manifest_sha256"))):
        problems.append("engine.json provenance needs the sealed source_manifest_sha256")
    if not any(line.startswith("> **Contributed by**") for line in head):
        problems.append("README.md needs the credit directly under its title")
    if not any(line.startswith("> Built with [crit]") for line in head):
        problems.append("README.md needs the built-with-crit line under the credit")
    return problems


def check_external(head):
    problems = []
    if "> [!WARNING]" not in head or not any(line.startswith("> **External engine.**") for line in head):
        problems.append("README.md must open with the external-engine notice")
    return problems


def check(folder):
    if not NAME.fullmatch(folder.name) or not NAME.fullmatch(folder.parent.name):
        return ["engine folder names use only letters, digits, '.', '_' and '-'"]
    for name in ("README.md", "engine.json"):
        if not (folder / name).is_file():
            return [f"missing {name}"]
    try:
        engine = json.loads((folder / "engine.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"invalid JSON in engine.json: {e}"]
    if not isinstance(engine, dict) or set(engine) - FIELDS:
        return [f"engine.json holds only {', '.join(sorted(FIELDS))}"]
    text = (folder / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        return ["README.md must open with a '# ' title"]
    head = lines[1:12]
    problems = (check_card(folder, engine) + check_results(engine, text) + check_supports(engine) + check_requirements(engine)
                + check_downloads(engine) + check_checkpoints(engine))
    provenance = engine.get("provenance") if isinstance(engine.get("provenance"), dict) else {}
    problems += check_source(folder, provenance, text)
    origin = engine.get("origin")
    if origin == "crit":
        problems += check_crit(provenance, head)
    elif origin == "external":
        problems += check_external(head)
    else:
        problems.append('engine.json needs "origin": "crit" or "external"')
    return problems


def main(hub):
    folders = sorted(p for p in (Path(hub) / "engines").glob("*/*") if p.is_dir())
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
