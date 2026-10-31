"""Add crit Eval result files to an engine's card, for a pull request to crit-hub.

    python -B tools/add_result.py engines/<owner>/<name> RESULT.json [MORE.json ...] --by <your GitHub account>

The files are one machine's crit Eval results of the engine (crit bench writes one; crit eval run --url measures any
engine behind a server), under one protocol and quality mode. On a GPU and model the card measured, they become a
reproduction of that result (results[].reproductions); on one it did not, a new result, which leaves the card's
supports. Each file is copied to evidence/ as the next crit-eval-N.json, the six stats and further figures are what
the files hold (as tools/export_engine.py fills them), the README and charts are rendered again and the folder is
checked. The GPU comes from the files' device and the card's GPU ids (--gpu names another), the model from the
checkpoint the files name (--model when they name none); a model the card only supports needs its checkpoint's
files locked first (--checkpoint FOLDER hashes a local copy into card.lock.json).
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import card as cardlib  # noqa: E402
import check_engines  # noqa: E402
import export_engine  # noqa: E402

VENDORS = ("nvidiageforce", "nvidia", "amdradeon", "amd", "intel", "apple")


def plain(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def gpu_id(card, device, wanted=None):
    """The card's id for the GPU the files ran on: --gpu, else the card's own GPU id the device's name holds (the
    longest), else the name without its vendor ("NVIDIA GeForce RTX 5090" is rtx5090)."""
    if wanted:
        if not check_engines.GPU.fullmatch(wanted):
            raise SystemExit(f"--gpu is an id like rtx5090, not {wanted}")
        return wanted
    name = plain(device)
    known = {r["gpu"] for r in card.get("results") or []} | {g for s in card.get("supports") or [] for g in s["gpus"]}
    found = sorted((g for g in known if plain(g) and plain(g) in name), key=len)
    if found:
        return found[-1]
    for vendor in VENDORS:
        if name.startswith(vendor):
            name = name[len(vendor):]
            break
    if not name:
        raise SystemExit(f"Name the GPU {device!r} ran as with --gpu (an id like rtx5090)")
    return name


def model_id(card, described, wanted=None):
    """The card's model id of the checkpoint the files ran: --model, else the checkpoint whose repository and revision
    the files name."""
    checkpoints = card.get("checkpoints") or {}
    if wanted:
        if wanted not in checkpoints:
            raise SystemExit(f"{wanted} is not one of the card's checkpoints: {', '.join(sorted(checkpoints))}")
        return wanted
    described = described or {}
    found = [m for m, c in checkpoints.items()
             if (c.get("repository"), c.get("revision")) == (described.get("repository"), described.get("revision"))]
    if len(found) == 1:
        return found[0]
    raise SystemExit(f"The result files name the checkpoint {described.get('repository')}@{described.get('revision')}, "
                     f"none of the card's ({', '.join(sorted(checkpoints))}); name the model with --model")


def read_results(files):
    """The files as crit Eval results of one machine, protocol and quality mode, each a complete speed run."""
    results = []
    for path in files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not cardlib.crit_eval(data.get("protocol")):
            raise SystemExit(f"{path} is not a crit Eval result")
        if data.get("aborted") or not (data.get("speed") or {}).get("complete"):
            raise SystemExit(f"{path} is not a complete crit Eval speed run")
        results.append(data)
    for label, of in (("protocols", lambda r: r["protocol"]), ("quality modes", cardlib.evidence_mode),
                      ("machines", lambda r: cardlib.hardware(r.get("host"))),
                      ("checkpoints", lambda r: json.dumps((r.get("model") or {}).get("revision")))):
        if len({of(r) for r in results}) > 1:
            raise SystemExit(f"The result files ran under different {label}: add them one machine and protocol at a time")
    return results


def lock_checkpoint(lock, card, model, folder):
    """card.lock.json's files of `model`'s checkpoint from a local copy of it (every file but dot files)."""
    checkpoint = card["checkpoints"][model]
    root = Path(folder)
    rows = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": export_engine.digest(p)}
            for p in sorted(root.rglob("*")) if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(root).parts)]
    if not rows:
        raise SystemExit(f"{folder} holds no checkpoint files")
    lock["checkpoints"][model] = {"repository": checkpoint["repository"], "revision": checkpoint["revision"], "files": rows}


def unsupported(card, gpu, model):
    """The card's supports without the pair (gpu, model), regrouped by GPU where a group held it."""
    groups = []
    for group in card.get("supports") or []:
        if gpu not in group["gpus"] or model not in group["models"]:
            groups.append(group)
            continue
        others = [g for g in group["gpus"] if g != gpu]
        if others:
            groups.append({"gpus": others, "models": list(group["models"])})
        rest = [m for m in group["models"] if m != model]
        if rest:
            groups.append({"gpus": [gpu], "models": rest})
    return groups


def add(folder, files, by, gpu=None, model=None, checkpoint=None):
    """Add the result files to the card in `folder`; returns what was added."""
    folder = Path(folder)
    card, lock = cardlib.load(folder), cardlib.load_lock(folder)
    if not check_engines.ACCOUNT.fullmatch(str(by)):
        raise SystemExit("--by is your GitHub account")
    results = read_results(files)
    gpu = gpu_id(card, (results[0].get("device") or {}).get("device"), gpu)
    model = model_id(card, results[0].get("model"), model)
    taken = {e["file"] for r in card.get("results") or [] for e in (r.get("evidence") or [])
             + [e for rep in r.get("reproductions") or [] for e in rep["evidence"]]}
    taken |= {p.relative_to(folder).as_posix() for p in (folder / "evidence").glob("*")} if (folder / "evidence").is_dir() else set()
    names, number = [], 1
    for _ in results:
        while (name := "evidence/crit-eval.json" if number == 1 else f"evidence/crit-eval-{number}.json") in taken:
            number += 1
        names.append(name)
        taken.add(name)
    entry = export_engine.result_entry(results)
    if not entry.get("more"):
        entry.pop("more", None)
    existing = next((r for r in card["results"] if (r["gpu"], r["model"]) == (gpu, model)), None)
    if existing is not None:
        if by == card["contributor"]["name"] and not existing.get("by"):
            raise SystemExit(f"The card measured {gpu} · {model} already: its contributor's new measurement replaces the "
                             "result (CONTRIBUTING.md, Update an engine)")
        if not cardlib.crit_eval(existing.get("measured_with")):
            raise SystemExit(f"The card's {gpu} · {model} result is self-reported: a reproduction is of a crit Eval result")
        kind, target = "reproduction", existing.setdefault("reproductions", [])
        record = {"by": by, **entry}
    else:
        if not (lock.get("checkpoints") or {}).get(model):
            if checkpoint is None:
                raise SystemExit(f"card.lock.json lists no files of {model}'s checkpoint, which the card only supports: "
                                 "pass --checkpoint FOLDER, a local copy of it, to lock them")
            lock_checkpoint(lock, card, model, checkpoint)
        kind, target = "result", card["results"]
        record = {"gpu": gpu, "model": model, **({"by": by} if by != card["contributor"]["name"] else {}), **entry}
        card["supports"] = unsupported(card, gpu, model)
        if not card["supports"]:
            card.pop("supports")
    (folder / "evidence").mkdir(exist_ok=True)
    record["evidence"] = []
    for path, name in zip(files, names):
        shutil.copyfile(path, folder / name)
        record["evidence"].append({"file": name, "sha256": export_engine.digest(folder / name)})
    target.append(record)
    cardlib.write(folder, card, lock)
    cardlib.main([folder], False)
    problems = check_engines.check(folder)
    return {"added": kind, "gpu": gpu, "model": model, "by": by, "evidence": names, "problems": problems}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("engine", help="the engine's folder, e.g. engines/crit-labs/nvfp4-gemv-int8kv")
    p.add_argument("results", nargs="+", help="crit Eval result files of one machine")
    p.add_argument("--by", required=True, help="your GitHub account: who measured it")
    p.add_argument("--gpu", help="the GPU id, when the result's device does not name it (e.g. rtx5090)")
    p.add_argument("--model", help="the card's model id, when the result names no checkpoint (e.g. qwen3.8-27b@NVFP4)")
    p.add_argument("--checkpoint", help="a local copy of a supported model's checkpoint, to lock its files")
    a = p.parse_args()
    found = add(a.engine, a.results, a.by, a.gpu, a.model, a.checkpoint)
    print(json.dumps(found, indent=1))
    if found["problems"]:
        raise SystemExit(f"{a.engine} does not pass tools/check_engines.py yet")
