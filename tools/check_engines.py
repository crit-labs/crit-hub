"""Check every engine folder against its card.

Each engine lives in engines/<owner>/<name>/, where <owner> is its contributor:

  card.yaml        what the engine is, what it measured and the README's prose (tools/card.py)
  card.lock.json   what tools write: the exact checkpoint files and how the engine was built
  evidence/        crit-eval.json (and -2, -3 ...) behind crit Eval results, self-reported.json
                   behind the contributor's own
  README.md        rendered from card.yaml by tools/card.py, never edited by hand

Every result states the six headline stats (decode, prefill, time to first token, perplexity,
mean KL divergence, context; null when not measured). A result measured with crit Eval ("crit-eval-v7") lists its
result files with their SHA-256, and every stat and further figure must be what those files hold;
any other result is the contributor's own report, and when evidence/self-reported.json is given
every figure it states must be one that file records. A crit Eval result was judged in the quality mode its
result files record (schemas/quality-modes.yaml lists the modes; the default when they record none), which the
card states as quality_mode unless it is the default. A crit engine's source is in
crit-labs/crit-engines, pinned to a full commit, with its sealed run in card.lock.json.
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

try:
    import jsonschema
except ImportError:  # the checks below still run; CI installs it
    jsonschema = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import card as cardlib  # noqa: E402
import codeowners  # noqa: E402

REPOSITORY = re.compile(r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9_.~-]+)+/?")
COMMIT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9_.-]+")
GPU = re.compile(r"[a-z0-9][a-z0-9.-]*")
MODEL = re.compile(r"[a-z0-9][a-z0-9.-]*@[A-Za-z0-9.+-]+")
HF_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
FEATURES = {"openai-api", "tool-calls", "vision", "speculative"}
CARD_FIELDS = {"title", "summary", "origin", "contributor", "license", "features", "source", "requirements", "build", "downloads",
               "checkpoints", "supports", "derived_from", "built", "results", "readme"}
ENGINE_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
BUILT_FIELDS = {"hours", "agent_tokens", "api_usd", "agent", "candidates"}
RESULT_FIELDS = {"gpu", "model", "measured_with", "measured_on", "quality_mode", "sampling", "hardware", "stats", "more", "evidence"}
MODE = re.compile(r"[a-z][a-z0-9_]*")
PLATFORMS = {"linux", "wsl2", "windows", "macos"}
GPU_ARCH = re.compile(r"sm_[0-9]{2,3}[a-z]?")
CUDA = re.compile(r"[0-9]+\.[0-9]+")
REQUIREMENT_FIELDS = {"platforms", "gpu_arch", "gpu_memory_gb", "cuda", "runtime"}
DOWNLOAD_KINDS = {"drafter", "artifact"}
DOWNLOAD_FIELDS = {"kind", "repository", "revision", "path", "file", "sha256", "size_gb", "note"}
CRIT_EVAL_FILE = re.compile(r"evidence/crit-eval(-[2-9]|-[1-9][0-9]+)?\.json")
SELF_REPORTED = "evidence/self-reported.json"
EVIDENCE_BYTES = 16 * 1024 * 1024
FIGURES = cardlib.STATS + cardlib.MORE
# Figures a crit Eval result gives directly, and where (ranges across result files allowed).
SCORES = {"decode_tok_s": ("speed", "decode_score"), "prefill_tok_s": ("speed", "prefill_score"),
          "ttft_ms": ("speed", "ttft_score_ms"), "heldout_decode_tok_s": ("heldout", "decode_score")}
# From crit-eval-v7 the headline decode is the sampled suite's and the greedy decode is a further figure.
SCORES_V7 = {**SCORES, "decode_tok_s": ("sampled_speed", "decode_score"), "greedy_decode_tok_s": ("speed", "decode_score")}


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def numbers(value):
    """A figure is a positive number, or [low, high] across runs."""
    if positive(value):
        return [value]
    if isinstance(value, list) and len(value) == 2 and all(positive(v) for v in value) and value[0] <= value[1]:
        return value
    return None


def matches(value, score):
    """A figure as written (252.4, 7425, 0.0028) is its score rounded to the figure's own precision."""
    if not positive(value) or not isinstance(score, (int, float)) or isinstance(score, bool):
        return False
    text = repr(float(value))
    decimals = 0 if float(value).is_integer() else len(text.split(".")[1]) if "e" not in text else 12
    return abs(value - score) <= 0.5 * 10 ** -decimals + 1e-12


def safe_path(path):
    return all(NAME.fullmatch(part) and part not in (".", "..") for part in str(path).split("/"))


# ---------------------------------------------------------------------------------------------
# What the engine is

def check_identity(folder, card):
    problems = []
    if not (isinstance(card.get("title"), str) and card["title"].strip() and "\n" not in card["title"]):
        problems.append("card.yaml needs title: the engine's name, as its folder is named")
    elif card["title"] != Path(folder).name:
        problems.append(f"card.yaml title is the engine's name, {Path(folder).name}")
    summary = card.get("summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 120 or "\n" in summary:
        problems.append("card.yaml needs summary: one line of at most 120 characters on what the engine does")
    if card.get("origin") not in ("crit", "external"):
        problems.append('card.yaml needs origin: crit or external')
    if not isinstance(card.get("license"), str) or not card["license"].strip():
        problems.append('card.yaml needs license: the SPDX id of the engine source, e.g. Apache-2.0')
    features = card.get("features")
    if not isinstance(features, list) or set(features) - FEATURES or len(set(features)) != len(features):
        problems.append(f"card.yaml features lists any of {', '.join(sorted(FEATURES))}")
    contributor = card.get("contributor")
    if not isinstance(contributor, dict) or not str(contributor.get("name", "")).strip() \
            or not REPOSITORY.fullmatch(str(contributor.get("repository"))):
        problems.append("card.yaml needs contributor.name and the https URL of contributor.repository")
    elif contributor["name"] != folder.parent.name:
        problems.append(f"the engine's folder is engines/<owner>/<name>/ with <owner> its contributor.name, "
                        f"{contributor['name']}, not {folder.parent.name}")
    source = card.get("source")
    if not isinstance(source, dict) or set(source) - {"repository", "revision", "path"}:
        return problems + ["card.yaml needs source: the repository, full commit revision and optional path of the engine's source"]
    if not REPOSITORY.fullmatch(str(source.get("repository"))):
        problems.append("card.yaml source.repository is the https URL of the source repository")
    if not COMMIT.fullmatch(str(source.get("revision"))):
        problems.append("card.yaml source.revision is a full 40-character commit SHA")
    if "path" in source and not (isinstance(source["path"], str) and safe_path(source["path"])):
        problems.append("card.yaml source.path is a folder in the source repository, like mtp7-device-confidence")
    return problems


def check_lineage(card):
    """derived_from (the crit-hub engine the run started from) and built (what the run took), both written by
    tools/export_engine.py from the run's records."""
    problems = []
    derived = card.get("derived_from")
    if derived is not None:
        if not isinstance(derived, dict) or set(derived) != {"engine", "source"} or not ENGINE_ID.fullmatch(str(derived.get("engine"))):
            problems.append("card.yaml derived_from is {engine: <owner>/<name>, source: {repository, revision, path}}: the crit-hub "
                            "engine the run started from")
        else:
            source = derived["source"]
            if (not isinstance(source, dict) or set(source) - {"repository", "revision", "path"}
                    or not REPOSITORY.fullmatch(str(source.get("repository"))) or not COMMIT.fullmatch(str(source.get("revision")))):
                problems.append("card.yaml derived_from.source needs the https repository and full commit revision of the origin's source")
    built = card.get("built")
    if built is not None:
        if not isinstance(built, dict) or not built or set(built) - BUILT_FIELDS:
            problems.append(f"card.yaml built holds any of {', '.join(sorted(BUILT_FIELDS))}, from the run's cost.json (crit run cost)")
        else:
            for key in ("hours", "api_usd"):
                if key in built and (type(built[key]) not in (int, float) or built[key] < 0):
                    problems.append(f"card.yaml built.{key} is a non-negative number")
            for key in ("agent_tokens", "candidates"):
                if key in built and (type(built[key]) is not int or built[key] < 0):
                    problems.append(f"card.yaml built.{key} is a non-negative integer")
            if "agent" in built and not (isinstance(built["agent"], str) and built["agent"].strip()):
                problems.append("card.yaml built.agent names the agent and model that built the engine")
    return problems


def check_requirements(card):
    requirements = card.get("requirements")
    if not isinstance(requirements, dict):
        return ["card.yaml needs requirements: platforms, gpu_arch, gpu_memory_gb, cuda and an optional runtime"]
    problems = []
    if set(requirements) - REQUIREMENT_FIELDS:
        problems.append(f"card.yaml requirements: unknown field(s) {', '.join(sorted(set(requirements) - REQUIREMENT_FIELDS))}")
    platforms = requirements.get("platforms")
    if not (isinstance(platforms, list) and platforms and set(platforms) <= PLATFORMS and len(set(platforms)) == len(platforms)):
        problems.append(f"card.yaml requirements.platforms lists any of {', '.join(sorted(PLATFORMS))}")
    arch = requirements.get("gpu_arch")
    if not (isinstance(arch, list) and arch and all(GPU_ARCH.fullmatch(str(a)) for a in arch)):
        problems.append("card.yaml requirements.gpu_arch lists the compute capabilities it is built for, like sm_120a")
    if not positive(requirements.get("gpu_memory_gb")):
        problems.append("card.yaml requirements.gpu_memory_gb is the GPU memory the engine needs, in GB")
    if not CUDA.fullmatch(str(requirements.get("cuda"))):
        problems.append('card.yaml requirements.cuda is the CUDA toolkit version it builds with, quoted, like "13.1"')
    if "runtime" in requirements and not (isinstance(requirements["runtime"], str) and requirements["runtime"].strip()):
        problems.append("card.yaml requirements.runtime names what the engine links, like crit Core 0.5.0")
    return problems


def check_build(card):
    """build.command: the literal arguments crit pull runs in the engine/ folder of a crit engine's pinned source."""
    if "build" not in card:
        return []
    build = card["build"]
    if not isinstance(build, dict) or set(build) != {"command"}:
        return ["card.yaml build holds command: the literal arguments that build the engine in its engine/ folder"]
    command = build["command"]
    if not (isinstance(command, list) and command and all(isinstance(a, str) and a.strip() for a in command)):
        return ["card.yaml build.command is a list of literal arguments, like [python, csrc/build.py]"]
    if card.get("origin") != "crit":
        return ["card.yaml build is for crit engines (origin: crit), which crit pull installs"]
    return []


def check_downloads(card):
    downloads = card.get("downloads", [])
    if not isinstance(downloads, list):
        return ["card.yaml downloads lists what to fetch besides the checkpoint"]
    problems = []
    for number, item in enumerate(downloads, 1):
        name = f"card.yaml downloads[{number}]"
        if not isinstance(item, dict):
            problems.append(f"{name} is a mapping")
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
        if "path" in item and not safe_path(item["path"]):
            problems.append(f"{name}: path is a folder in the repository")
        if "file" in item and not NAME.fullmatch(str(item["file"])):
            problems.append(f"{name}: file is a file name")
        if "size_gb" in item and not positive(item["size_gb"]):
            problems.append(f"{name}: size_gb is a positive number")
        if "note" in item and not (isinstance(item["note"], str) and item["note"].strip() and "\n" not in item["note"]):
            problems.append(f"{name}: note is one line")
    return problems


def check_supports(card):
    supports = card.get("supports", [])
    if not isinstance(supports, list):
        return ["card.yaml supports is a list of {gpus, models} groups"]
    problems = []
    for number, group in enumerate(supports, 1):
        if not (isinstance(group, dict) and set(group) == {"gpus", "models"}
                and isinstance(group["gpus"], list) and group["gpus"] and all(GPU.fullmatch(str(g)) for g in group["gpus"])
                and isinstance(group["models"], list) and group["models"] and all(MODEL.fullmatch(str(m)) for m in group["models"])):
            problems.append(f"card.yaml supports group {number} is {{gpus: [ids], models: [ids]}}")
    return problems


def check_checkpoints(card, lock):
    """Every measured or supported model pins its Hugging Face checkpoint in card.yaml; every measured
    one lists its files with their hashes in card.lock.json, at the same revision."""
    checkpoints = card.get("checkpoints")
    if not isinstance(checkpoints, dict):
        return ["card.yaml checkpoints pins the Hugging Face checkpoint of every measured or supported model, keyed by model"]
    measured = {str(r.get("model")) for r in card.get("results") or [] if isinstance(r, dict)}
    supported = {str(m) for g in card.get("supports") or [] if isinstance(g, dict) and isinstance(g.get("models"), list)
                 for m in g["models"]} - measured
    problems = []
    for model in sorted((measured | supported) - set(checkpoints)):
        problems.append(f"card.yaml checkpoints needs the checkpoint {model} runs: {{repository: <owner>/<name>, revision: <40-character commit>}}")
    for model in sorted(set(checkpoints) - measured - supported):
        problems.append(f"card.yaml checkpoints lists {model}, which it neither measures nor supports")
    locked = (lock or {}).get("checkpoints") or {}
    for model in sorted(set(checkpoints) & (measured | supported)):
        c = checkpoints[model]
        if not (isinstance(c, dict) and set(c) == {"repository", "revision"} and HF_REPOSITORY.fullmatch(str(c["repository"]))
                and COMMIT.fullmatch(str(c["revision"]))):
            problems.append(f"card.yaml checkpoints[{model}] is the Hugging Face repository and its full 40-character revision")
            continue
        if model not in measured:
            continue
        files = locked.get(model)
        if not isinstance(files, dict) or (files.get("repository"), files.get("revision")) != (c["repository"], c["revision"]):
            problems.append(f"card.lock.json checkpoints[{model}] lists the files of {c['repository']} at {c['revision'][:12]}")
            continue
        entries = files.get("files")
        if not isinstance(entries, list) or not entries:
            problems.append(f"card.lock.json checkpoints[{model}] lists no checkpoint files")
        for entry in entries or []:
            if not (isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("bytes"), int)
                    and SHA256.fullmatch(str(entry.get("sha256")))):
                problems.append(f"card.lock.json checkpoints[{model}] file entry needs path, bytes and sha256")
    return problems


def check_lock(card, lock):
    if lock is None:
        return ["missing card.lock.json: the exact checkpoint files and how the engine was built"]
    if not isinstance(lock, dict) or set(lock) != {"checkpoints", "provenance"} or not isinstance(lock["provenance"], dict):
        return ['card.lock.json is {"checkpoints": {...}, "provenance": {...}}']
    provenance = lock["provenance"]
    problems = []
    if card.get("origin") == "crit":
        if not provenance.get("run_id") or not provenance.get("candidate_id"):
            problems.append("card.lock.json provenance needs the sealed run_id and candidate_id")
        if not SHA256.fullmatch(str(provenance.get("source_manifest_sha256"))):
            problems.append("card.lock.json provenance needs the sealed source_manifest_sha256")
    return problems


# ---------------------------------------------------------------------------------------------
# What it measured

def check_stats(name, stats, more):
    problems = []
    if not isinstance(stats, dict) or list(stats) != list(cardlib.STATS):
        return [f"{name}: stats holds {', '.join(cardlib.STATS)}, in that order (null when not measured)"]
    if numbers(stats["decode_tok_s"]) is None:
        problems.append(f"{name}: stats.decode_tok_s is a positive number or [low, high]")
    if stats["prefill_tok_s"] is not None and numbers(stats["prefill_tok_s"]) is None:
        problems.append(f"{name}: stats.prefill_tok_s is a positive number, [low, high] or null")
    if stats["ttft_ms"] is not None and numbers(stats["ttft_ms"]) is None:
        problems.append(f"{name}: stats.ttft_ms is the time to first token in ms, [low, high] or null")
    tokens = stats["context_tokens"]
    if tokens is not None and not (isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0):
        problems.append(f"{name}: stats.context_tokens is the maximum context in tokens, or null")
    ppl = stats["ppl"]
    if ppl is not None and not (isinstance(ppl, dict) and len(ppl) == 1 and set(ppl) <= {"vs_reference", "absolute"}
                                and positive(next(iter(ppl.values())))):
        problems.append(f"{name}: stats.ppl is {{vs_reference: <perplexity ratio>}}, {{absolute: <perplexity>}} or null")
    if stats["kld_mean"] is not None and not positive(stats["kld_mean"]):
        problems.append(f"{name}: stats.kld_mean is the mean KL divergence against the reference, or null")
    if not isinstance(more, dict) or set(more) - set(cardlib.MORE):
        return problems + [f"{name}: more holds any of {', '.join(cardlib.MORE)}"]
    for key in ("long_context_decode_tok_s", "heldout_decode_tok_s"):
        if key in more and numbers(more[key]) is None:
            problems.append(f"{name}: more.{key} is a positive number or [low, high]")
    if "host_ram_gib" in more and not (isinstance(more["host_ram_gib"], int) and more["host_ram_gib"] > 0):
        problems.append(f"{name}: more.host_ram_gib is a positive whole number")
    if "quality_budget_pct" in more and not positive(more["quality_budget_pct"]):
        problems.append(f"{name}: more.quality_budget_pct is a positive number")
    return problems


def read_evidence(folder, name, entries, allowed):
    """The listed evidence files, verified against their hashes: [(file, data)] and problems."""
    problems, loaded = [], []
    for entry in entries:
        file = str(entry.get("file")) if isinstance(entry, dict) else ""
        path = folder / file
        if not allowed(file) or not path.is_file():
            problems.append(f"{name}: no evidence file {file}")
            continue
        data = path.read_bytes()
        if len(data) > EVIDENCE_BYTES:
            problems.append(f"{name}: {file} is over {EVIDENCE_BYTES // 2**20} MiB")
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            problems.append(f"{name}: {file} does not match its sha256")
            continue
        try:
            loaded.append((file, json.loads(data)))
        except json.JSONDecodeError as e:
            problems.append(f"{name}: {file} is not JSON: {e}")
    return loaded, problems


def check_crit_eval(name, result, loaded):
    """Every stat and further figure of a crit Eval result is what its result files hold."""
    problems = []
    gpu = re.sub(r"[^a-z0-9]", "", result["gpu"])
    sampled = cardlib.sampled_eval(result["measured_with"])
    for file, data in loaded:
        if not isinstance(data, dict):
            problems.append(f"{name}: {file} is not a crit Eval result")
            return problems
        if data.get("protocol") != result["measured_with"]:
            problems.append(f"{name}: {file} ran under {data.get('protocol')}, not {result['measured_with']}")
        if data.get("aborted") or not (data.get("speed") or {}).get("complete"):
            problems.append(f"{name}: {file} is not a complete speed run")
        if sampled and not (data.get("sampled_speed") or {}).get("complete"):
            problems.append(f"{name}: {file} is not a complete sampled speed run")
        if sampled:
            sampled = data.get("sampled_speed") or {}
            source = sampled.get("sampling_source") or "generation_config"
            stated = result.get("sampling")
            found = None if source == "generation_config" else {"source": source, "settings": sampled.get("sampling")}
            if stated != found:
                problems.append(f"{name}: {file} sampled its decode at {sampled.get('sampling')} ({source}); the card's "
                                f"sampling must be {found if found else 'absent (the model defaults)'}")
        # The quality mode it was judged in (crit-eval 0.2.0 records it; before, every run was the default's).
        judged = cardlib.evidence_mode(data)
        if judged not in cardlib.quality_modes():
            problems.append(f"{name}: {file} was judged in quality mode {judged}, which crit-hub does not list yet: "
                            f"add it to schemas/quality-modes.yaml")
        elif judged != cardlib.quality_mode(result):
            problems.append(f"{name}: {file} was judged in quality mode {judged}; the card's quality_mode must be {judged}"
                            + (" (or left out, the default)" if judged == cardlib.default_mode() else ""))
        failed = [g.get("name") for g in (data.get("status") or {}).get("gates", []) if isinstance(g, dict) and g.get("state") == "fail"]
        if failed:
            problems.append(f"{name}: {file} failed gate(s) {', '.join(map(str, failed))}")
        if gpu not in re.sub(r"[^a-z0-9]", "", str((data.get("device") or {}).get("device", "")).lower()):
            problems.append(f"{name}: {file} ran on {(data.get('device') or {}).get('device')}, not {result['gpu']}")
    stats, more = result["stats"], result.get("more") or {}
    figures = {**stats, **more}
    for key, (section, field) in (SCORES_V7 if sampled else SCORES).items():
        value, scores = figures.get(key), [(data.get(section) or {}).get(field) for _, data in loaded]
        if value is None:
            continue
        if isinstance(value, list):
            if len(loaded) < 2 or None in scores or not (matches(value[0], min(scores)) and matches(value[1], max(scores))):
                problems.append(f"{name}: {key} {value} is not the range of its result files' {field}, {scores}")
        elif len(loaded) != 1 or not matches(value, scores[0]):
            problems.append(f"{name}: {key} {value} is not its result file's {field}, {scores}")
    tokens = {(data.get("context") or {}).get("tokens") for _, data in loaded}
    if stats["context_tokens"] is not None and tokens != {stats["context_tokens"]}:
        problems.append(f"{name}: context_tokens {stats['context_tokens']} is not its result files' context, {sorted(tokens, key=str)}")
    derived = {
        "ppl": ("ppl_ratio", 1, (stats["ppl"] or {}).get("vs_reference")),
        "kld_mean": ("kld_mean", 1, stats["kld_mean"]),
        "quality_budget_pct": ("budget_used", 100, more.get("quality_budget_pct")),
    }
    for key, (field, scale, value) in derived.items():
        scores = [cardlib.worst_quality(data, field) for _, data in loaded]
        if value is None:
            if key != "quality_budget_pct" and all(s is not None for s in scores):
                problems.append(f"{name}: {key} is measured in its result files; state it")
            continue
        if not any(s is not None and matches(value, s * scale) for s in scores):
            problems.append(f"{name}: {key} {value} is not its result files' worst served {field}")
    if stats["ppl"] and "absolute" in stats["ppl"]:
        problems.append(f"{name}: a crit Eval result gives perplexity against the reference, ppl: {{vs_reference: ...}}")
    if "greedy_decode_tok_s" in more and not sampled:
        problems.append(f"{name}: greedy_decode_tok_s is for crit-eval-v7 results; through v6 decode itself is greedy")
    if "long_context_decode_tok_s" in more:
        scores = [(((data.get("status") or {}).get("kpis") or {}).get("context") or {}).get("decode", {}).get("score") for _, data in loaded]
        if not any(matches(more["long_context_decode_tok_s"], s) for s in scores if s is not None):
            problems.append(f"{name}: long_context_decode_tok_s {more['long_context_decode_tok_s']} is not its result files' context decode score")
    machines = {cardlib.hardware(data.get("host")) for _, data in loaded}
    if len(machines) > 1:
        problems.append(f"{name}: its result files ran on different hardware")
    elif result.get("hardware") != next(iter(machines), None):
        found = next(iter(machines), None)
        problems.append(f"{name}: hardware is " + (f"what its result file records, {found!r}" if found else
                        "left out: its result file records none"))
    dates = {str(data.get("started_at", ""))[:10] for _, data in loaded}
    if str(result.get("measured_on")) not in dates:
        problems.append(f"{name}: measured_on is the day its result file was measured, {', '.join(sorted(dates))}")
    return problems


def check_self_reported(name, result, data):
    """The evidence behind a self-reported result: its measurements, each with method, link, settings
    and figures. Every figure the result states must be one the file records."""
    problems = []
    entries = data.get("results") if isinstance(data, dict) and set(data) == {"results"} else None
    if not isinstance(entries, list):
        return [f'{SELF_REPORTED} is {{"results": [{{"gpu", "model", "measurements": [{{"method", "link", "settings", "figures"}}]}}]}}']
    entry = next((e for e in entries if isinstance(e, dict) and (e.get("gpu"), e.get("model")) == (result["gpu"], result["model"])), None)
    if entry is None:
        return [f"{name}: {SELF_REPORTED} has no entry for this GPU and model"]
    measurements = entry.get("measurements")
    if set(entry) != {"gpu", "model", "measurements"} or not isinstance(measurements, list) or not measurements:
        return [f"{SELF_REPORTED} {result['gpu']} · {result['model']}: {{gpu, model, measurements}}, with at least one measurement"]
    for number, m in enumerate(measurements, 1):
        where = f"{SELF_REPORTED} {result['gpu']} · {result['model']} measurement {number}"
        if not isinstance(m, dict) or set(m) - {"method", "link", "settings", "figures"}:
            problems.append(f"{where} holds method, link, settings and figures")
            continue
        if not (isinstance(m.get("method"), str) and m["method"].strip()):
            problems.append(f"{where} needs method: how it was measured")
        if "link" in m and not REPOSITORY.fullmatch(str(m["link"])):
            problems.append(f"{where}: link is the https URL of the run, leaderboard or card")
        if "settings" in m and not (isinstance(m["settings"], str) and m["settings"].strip()):
            problems.append(f"{where}: settings describe the configuration measured")
        if not isinstance(m.get("figures"), dict) or not m["figures"] or set(m["figures"]) - set(FIGURES):
            problems.append(f"{where}: figures records any of {', '.join(FIGURES)} as measured")
    recorded = [m["figures"] for m in measurements if isinstance(m, dict) and isinstance(m.get("figures"), dict)]
    figures = {**result["stats"], **(result.get("more") or {})}
    for key, value in figures.items():
        if value is not None and not any(f.get(key) == value for f in recorded):
            problems.append(f"{name}: {key} {value} is not a figure {SELF_REPORTED} records")
    return problems


def check_results(folder, card):
    results = card.get("results")
    if not isinstance(results, list) or not results:
        return ["card.yaml needs results: at least one measured GPU and model"], set()
    problems, seen, listed = [], set(), set()
    for result in results:
        if not isinstance(result, dict):
            problems.append("card.yaml results are mappings")
            continue
        name = f"card.yaml result {result.get('gpu')} · {result.get('model')}"
        if set(result) - RESULT_FIELDS:
            problems.append(f"{name}: unknown field(s) {', '.join(sorted(set(result) - RESULT_FIELDS))}")
        if not GPU.fullmatch(str(result.get("gpu"))) or not MODEL.fullmatch(str(result.get("model"))):
            problems.append(f"{name}: gpu is an id like rtx5090, model one like qwen3.8-27b@NVFP4")
            continue
        if (result["gpu"], result["model"]) in seen:
            problems.append(f"card.yaml measures {result['gpu']} · {result['model']} twice")
        seen.add((result["gpu"], result["model"]))
        if not (isinstance(result.get("measured_with"), str) and result["measured_with"].strip()):
            problems.append(f'{name} needs measured_with: crit-eval-v7, or how else it was measured')
            continue
        if isinstance(result.get("measured_on"), datetime.date):
            result["measured_on"] = result["measured_on"].isoformat()
        on = result.get("measured_on")
        if on is not None and not (isinstance(on, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", on)):
            problems.append(f"{name}: measured_on is the day it was measured, YYYY-MM-DD, or null when unknown")
        if "sampling" in result and not cardlib.sampled_eval(result["measured_with"]):
            problems.append(f"{name}: sampling states a crit-eval-v7 result's non-default sampling")
        if "quality_mode" in result and (not MODE.fullmatch(str(result["quality_mode"]))
                                         or result["quality_mode"] not in cardlib.quality_modes()):
            problems.append(f"{name}: quality_mode is one of {', '.join(cardlib.quality_modes())} (schemas/quality-modes.yaml), "
                            f"not {result['quality_mode']}")
        if "hardware" in result and not (isinstance(result["hardware"], str) and result["hardware"].strip() and "\n" not in result["hardware"]):
            problems.append(f"{name}: hardware is one line naming the machine it ran on")
        stat_problems = check_stats(name, result.get("stats"), result.get("more", {}))
        problems += stat_problems
        evidence = result.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(e, dict) and set(e) == {"file", "sha256"} for e in evidence):
            problems.append(f'{name}: evidence lists {{file: evidence/<file>, sha256: <sha256>}}')
            continue
        listed |= {e["file"] for e in evidence}
        if stat_problems:
            continue
        if cardlib.crit_eval(result["measured_with"]):
            if not evidence:
                problems.append(f"{name} is measured with {result['measured_with']}: list its result file, "
                                "evidence: [{file: evidence/crit-eval.json, sha256: ...}], or name how else it was measured")
                continue
            if on is None:
                problems.append(f"{name}: measured_on is required for a crit Eval result")
            loaded, found = read_evidence(folder, name, evidence, CRIT_EVAL_FILE.fullmatch)
            problems += found
            if len(loaded) == len(evidence):
                problems += check_crit_eval(name, result, loaded)
        elif evidence:
            loaded, found = read_evidence(folder, name, evidence, lambda f: f == SELF_REPORTED)
            problems += found
            for _, data in loaded:
                problems += check_self_reported(name, result, data)
    return problems, listed


# ---------------------------------------------------------------------------------------------

SCHEMA = Path(__file__).resolve().parent.parent / "schemas" / "card.schema.json"


def check_schema(card):
    """card.yaml against schemas/card.schema.json, the schema editors and agents write against."""
    if jsonschema is None or not SCHEMA.is_file():
        return []
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    return [f"card.yaml {'.'.join(map(str, e.absolute_path)) or '(top)'}: {e.message}"
            for e in sorted(validator.iter_errors(card), key=lambda e: list(map(str, e.absolute_path)))]


def check_readme(folder, card, drawn):
    readme = card.get("readme")
    keys = [key for key, _ in cardlib.SECTIONS]
    if not isinstance(readme, dict) or set(readme) - set(keys) or not cardlib.REQUIRED_SECTIONS <= set(readme) \
            or not all(isinstance(v, str) and v.strip() for v in readme.values()):
        return [f"card.yaml readme holds the README's prose as markdown blocks: {', '.join(sorted(cardlib.REQUIRED_SECTIONS))}"
                f", and optionally benchmark"]
    path = folder / "README.md"
    try:
        rendered = cardlib.render(card, list(drawn))
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        return [f"README.md cannot be rendered from card.yaml: {e!r}"]
    if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
        return [f"README.md is not the rendered card; run python -B tools/card.py {folder.as_posix()}"]
    stale = cardlib.stale_charts(folder, drawn)
    if stale:
        return [f"charts out of date ({', '.join(stale)}); run python -B tools/card.py {folder.as_posix()}"]
    return []


def check(folder):
    if not NAME.fullmatch(folder.name) or not NAME.fullmatch(folder.parent.name):
        return ["engine folder names use only letters, digits, '.', '_' and '-'"]
    if not (folder / "card.yaml").is_file():
        return ["missing card.yaml"]
    try:
        card = yaml.safe_load((folder / "card.yaml").read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return [f"card.yaml is not YAML: {e}"]
    if not isinstance(card, dict) or set(card) - CARD_FIELDS:
        return [f"card.yaml holds only {', '.join(sorted(CARD_FIELDS))}"]
    try:
        lock = cardlib.load_lock(folder)
    except json.JSONDecodeError as e:
        return [f"card.lock.json is not JSON: {e}"]
    problems = (check_schema(card) + check_identity(folder, card) + check_requirements(card) + check_build(card) + check_downloads(card)
                + check_supports(card) + check_lineage(card))
    result_problems, listed = check_results(folder, card)
    problems += result_problems + check_checkpoints(card, lock) + check_lock(card, lock)
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    try:
        drawn = cardlib.charts(folder, card) if not result_problems else {}
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        drawn = {}
        problems.append(f"charts cannot be drawn from the evidence: {e!r}")
    extra = sorted(files - {"README.md", "card.yaml", "card.lock.json"} - listed - set(drawn))
    if extra:
        problems.append(f"an engine folder holds README.md, card.yaml, card.lock.json, the evidence/ files its results list and its charts/; "
                        f"also found {len(extra)} other file(s), e.g. {extra[0]}")
    if not problems:
        problems += check_readme(folder, card, drawn)
    return problems


def check_modes():
    """schemas/quality-modes.yaml: every mode an id with a label and a one-line summary, exactly one the default."""
    modes = cardlib.quality_modes()
    if not isinstance(modes, dict) or not modes:
        return ["schemas/quality-modes.yaml lists the quality modes, {<mode>: {label, summary, default}}"]
    problems = []
    for mode, about in modes.items():
        one_line = lambda key: isinstance(about.get(key), str) and about[key].strip() and "\n" not in about[key]
        if not MODE.fullmatch(str(mode)) or not isinstance(about, dict) or set(about) - {"label", "summary", "default"} \
                or not (one_line("label") and one_line("summary")) or about.get("default", False) not in (True, False):
            problems.append(f"schemas/quality-modes.yaml {mode}: a lowercase id (letters, digits, _) with one-line label and summary, "
                            "and optionally default: true")
    if sum(isinstance(about, dict) and about.get("default") is True for about in modes.values()) != 1:
        problems.append("schemas/quality-modes.yaml marks exactly one mode default: true")
    return problems


def main(hub):
    modes = check_modes()
    if modes:
        raise SystemExit("FAIL schemas/quality-modes.yaml\n  " + "\n  ".join(modes))
    folders = sorted(p for p in (Path(hub) / "engines").glob("*/*") if p.is_dir() and not p.parent.name.startswith("_"))
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
    print(f"{len(folders)} engine folders match their cards")
    codeowners.main(hub, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hub", default=".", help="crit-hub checkout (default: the current folder)")
    main(p.parse_args().hub)
