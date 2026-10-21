"""Check every engine folder against the layout for its origin.

Each engine lives in engines/<owner>/<name>/, where <owner> is its contributor, and describes
itself in engine.json: summary, origin, contributor, license, features, the GPU/model pairs it
measured (results) or supports, its provenance and the exact Hugging Face checkpoint of every model
it measures or supports, which browse/hf.json must already place in Hugging Face's model tree.
A crit engine needs its sealed-run provenance and the built-with-crit line under its credit.
An external engine holds only README.md and engine.json, pins its source to a full commit SHA and
opens its README with the external-engine notice.
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
FIELDS = {"summary", "origin", "contributor", "license", "features", "results", "supports", "provenance", "checkpoints"}
RESULT_FIELDS = {"gpu", "model", "measured_with", "decode_tok_s", "prefill_tok_s", "ttft_ms", "context_tokens", "host_ram_gib"}
EXTERNAL_FILES = {"README.md", "engine.json"}


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


def written(value, text, key):
    """The README writes 8917 as 8,917, and a time in ms or s: 1610 ms as 1.61 s."""
    return any(f"{v:,}" in text for v in ([value, round(value / 1000, 3)] if key == "ttft_ms" else [value]))


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
    return problems


def check_checkpoints(engine, cache):
    """Every measured or supported model pins its Hugging Face checkpoint, measured ones with the
    hashes of its files, and browse/hf.json already places each one in Hugging Face's tree."""
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
        elif f"{checkpoint['repository']}@{revision}" not in cache:
            problems.append(f"browse/hf.json has no entry for {checkpoint['repository']} at {revision[:12]}; run "
                            "python -B tools/build_index.py --refresh to read it from Hugging Face, and commit browse/")
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


def check_crit(folder, provenance, head):
    problems = []
    if not provenance.get("run_id") or not provenance.get("candidate_id"):
        problems.append("engine.json provenance needs the sealed run_id and candidate_id")
    if not (folder / "engine").is_dir():
        problems.append("a crit engine carries its runtime source in engine/")
    if not any(line.startswith("> **Contributed by**") for line in head):
        problems.append("README.md needs the credit directly under its title")
    if not any(line.startswith("> Built with [crit]") for line in head):
        problems.append("README.md needs the built-with-crit line under the credit")
    return problems


def check_external(folder, provenance, head, text):
    problems = []
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if files - EXTERNAL_FILES:
        extra = sorted(files - EXTERNAL_FILES)
        problems.append(f"an external engine holds only README.md and engine.json; also found {len(extra)} other file(s), e.g. {extra[0]}")
    source = provenance.get("source") or {}
    if not REPOSITORY.fullmatch(str(source.get("repository"))):
        problems.append("engine.json provenance needs the https URL of source.repository")
    revision = str(source.get("revision"))
    if not COMMIT.fullmatch(revision):
        problems.append("engine.json provenance source.revision must be a full 40-character commit SHA")
    elif revision not in text:
        problems.append("README.md must link the pinned source commit")
    if "> [!WARNING]" not in head or not any(line.startswith("> **External engine.**") for line in head):
        problems.append("README.md must open with the external-engine notice")
    return problems


def check(folder, cache):
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
    problems = check_card(folder, engine) + check_results(engine, text) + check_supports(engine) + check_checkpoints(engine, cache)
    provenance = engine.get("provenance") if isinstance(engine.get("provenance"), dict) else {}
    origin = engine.get("origin")
    if origin == "crit":
        problems += check_crit(folder, provenance, head)
    elif origin == "external":
        problems += check_external(folder, provenance, head, text)
    else:
        problems.append('engine.json needs "origin": "crit" or "external"')
    return problems


def main(hub):
    path = Path(hub) / "browse" / "hf.json"
    cache = json.loads(path.read_text(encoding="utf-8")).get("checkpoints", {}) if path.is_file() else {}
    folders = sorted(p for p in (Path(hub) / "engines").glob("*/*") if p.is_dir())
    failed = 0
    for folder in folders:
        problems = check(folder, cache)
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
