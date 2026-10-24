"""An engine's card: card.yaml is the engine's single hand-written record, and README.md is rendered from it.

card.yaml holds what the engine is (summary, contributor, license, source pin, requirements,
downloads, checkpoints), what it measured (results, each with the six headline stats, more
figures and its evidence), and the README's prose as markdown block scalars under `readme`.
card.lock.json holds what tools write: the exact checkpoint files and how the engine was built.

    python -B tools/card.py engines/<owner>/<name> [...]          render README.md from card.yaml
    python -B tools/card.py --check engines/<owner>/<name> [...]  fail if README.md is out of date

Without a folder, every engine under engines/ is rendered or checked.
"""
import argparse
import json
import re
from pathlib import Path

import yaml

# The README's prose, in order: the card key and the heading it renders under (None: no heading).
SECTIONS = [
    ("intro", None),
    ("design", "Design"),
    ("measured", "Measured"),
    ("checks", "Checks"),
    ("limits", "Limits"),
    ("build_and_chat", "Build and chat"),
    ("benchmark", "Benchmark your machine"),
    ("credits", "Credits"),
]
REQUIRED_SECTIONS = {"intro", "design", "measured", "checks", "limits", "build_and_chat", "credits"}

# The six headline stats, in the order every page shows them.
STATS = ("decode_tok_s", "prefill_tok_s", "ttft_ms", "ppl", "kld_mean", "context_tokens")
# Further figures a result may give, shown after the stats.
MORE = ("greedy_decode_tok_s", "long_context_decode_tok_s", "heldout_decode_tok_s", "host_ram_gib", "quality_budget_pct")

HEADER = "<!-- Rendered by tools/card.py from card.yaml: edit card.yaml, then run python -B tools/card.py on this folder. -->"
PLATFORMS = {"linux": "Linux", "wsl2": "WSL2", "windows": "Windows", "macos": "macOS"}


# ---------------------------------------------------------------------------------------------
# Reading and writing

class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper, value):
    # Prose as block scalars: markdown stays readable and needs no escaping.
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


# Editors that read this modeline (the YAML language server) check the card against the schema as you type.
MODELINE = "# yaml-language-server: $schema=../../../schemas/card.schema.json\n"


def dumps(card):
    return MODELINE + yaml.dump(card, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=4096, default_flow_style=False)


def load(folder):
    return yaml.safe_load((Path(folder) / "card.yaml").read_text(encoding="utf-8"))


def load_lock(folder):
    path = Path(folder) / "card.lock.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write(folder, card, lock=None):
    folder = Path(folder)
    text = dumps(card)
    if yaml.safe_load(text) != card:
        raise ValueError("card.yaml would not read back as written (trailing spaces or tabs in the prose?)")
    (folder / "card.yaml").write_text(text, encoding="utf-8", newline="\n")
    if lock is not None:
        (folder / "card.lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")


def split_readme(text):
    """A README's title and its prose sections, keyed as in SECTIONS. The credit block under the
    title (a GitHub alert) is dropped: the card renders it. Unknown headings raise."""
    headings = {heading: key for key, heading in SECTIONS if heading}
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or not lines[0].startswith("# "):
        raise ValueError("The README must open with a '# ' title")
    title = lines[0][2:].strip()
    body = lines[1:]
    while body and (not body[0].strip() or body[0].startswith(">")):
        body = body[1:]
    sections, key, current, fenced = {}, "intro", [], False
    for line in body:
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            sections[key] = "\n".join(current).strip("\n")
            name = line[3:].strip()
            if name not in headings:
                raise ValueError(f"Unknown README section: {name}")
            key, current = headings[name], []
            continue
        current.append(line)
    sections[key] = "\n".join(current).strip("\n")
    return title, {k: v + "\n" for k, _ in SECTIONS if (v := sections.get(k))}


# ---------------------------------------------------------------------------------------------
# Rendering

def number(value):
    return f"{value:,}"


def figure(value, unit=""):
    if value is None:
        return "–"
    if isinstance(value, list):
        return f"{number(value[0])}–{number(value[1])}{unit}"
    return f"{number(value)}{unit}"


def ppl_cell(ppl):
    """The perplexity tile: the change against the reference when measured, else the absolute value."""
    if not ppl:
        return "–", "Perplexity vs reference"
    if "vs_reference" in ppl:
        return f"{(ppl['vs_reference'] - 1) * 100:+.1f}%", "Perplexity vs reference"
    return f"{ppl['absolute']:g}", "Perplexity (absolute)"


def ttft(value):
    """A time to first token in ms, or in s from one second on: 150 ms, 1.61 s."""
    if value is None:
        return "–"
    high = value[1] if isinstance(value, list) else value
    if high < 1000:
        return figure(value, " ms")
    return figure([round(v / 1000, 3) for v in value] if isinstance(value, list) else round(value / 1000, 3), " s")


def decodes(result):
    """A result's (sampled, greedy) decode. crit Eval measures both from v7 and only greedy through v6; a
    self-reported decode counts as greedy, as on the website."""
    stats, more, how = result.get("stats") or {}, result.get("more") or {}, result.get("measured_with")
    if crit_eval(how) and eval_major(how) >= 7:
        return stats.get("decode_tok_s"), more.get("greedy_decode_tok_s")
    return None, stats.get("decode_tok_s")


COLUMNS = ("GPU · model", "Decode (sampled)", "Decode (greedy)", "Prefill", "TTFT", "Perplexity", "Mean KL divergence", "Context")


def stat_cells(result):
    """One result's row of the stats table: its GPU and model, then the headline stats with both decodes."""
    stats = result.get("stats") or {}
    sampled, greedy = decodes(result)
    ppl, ppl_label = ppl_cell(stats.get("ppl"))
    return [
        f"{result['gpu']} · `{result['model']}`",
        figure(sampled, " tok/s"),
        figure(greedy, " tok/s"),
        figure(stats.get("prefill_tok_s"), " tok/s"),
        ttft(stats.get("ttft_ms")),
        ppl if ppl == "–" else f"{ppl} {'vs reference' if ppl_label.endswith('reference') else '(absolute)'}",
        figure(stats.get("kld_mean")),
        figure(stats.get("context_tokens"), " tok"),
    ]


def more_text(more):
    """The further figures, but the greedy decode: the stats table shows it."""
    parts = []
    if more.get("long_context_decode_tok_s") is not None:
        parts.append(f"long-context decode {figure(more['long_context_decode_tok_s'], ' tok/s')}")
    if more.get("heldout_decode_tok_s") is not None:
        parts.append(f"held-out decode {figure(more['heldout_decode_tok_s'], ' tok/s')}")
    if more.get("host_ram_gib") is not None:
        parts.append(f"{more['host_ram_gib']} GiB host RAM")
    if more.get("quality_budget_pct") is not None:
        parts.append(f"{more['quality_budget_pct']}% of the quality budget")
    return " · ".join(parts)


def stats_table(results):
    """The stats as one markdown table, one row per GPU and model, the stats in bold."""
    rows = [stat_cells(r) for r in results]
    return "\n".join([
        "| " + " | ".join(COLUMNS) + " |",
        "| :--- |" + "|".join(" :---: " for _ in COLUMNS[1:]) + "|",
        *("| " + " | ".join([row[0]] + [f"**{v}**" if v != "–" else v for v in row[1:]]) + " |" for row in rows),
    ])


def eval_major(measured_with):
    """The crit Eval protocol's major version: 6 for crit-eval-v6.1. Through v6 Eval measures decode greedily."""
    return int(re.fullmatch(r"crit-eval-v([0-9]+)(\.[0-9]+)?", str(measured_with)).group(1))


def crit_eval(measured_with):
    return re.fullmatch(r"crit-eval-v[0-9]+(\.[0-9]+)?", str(measured_with)) is not None


def sampled_text(sampling):
    """How a crit-eval-v7 decode was sampled: at the model's defaults, or the settings and why they differ."""
    if not sampling:
        return "decode sampled at the model's defaults"
    settings = ", ".join(f"{key} {value}" for key, value in sampling["settings"].items())
    why = "a user override, not the model's defaults" if sampling["source"] == "override" else "the protocol default"
    return f"decode sampled at {settings}: {why}"


def measured(results):
    """The facts each result adds under the stats table: how it was measured and from which evidence, the
    hardware it ran on, and its further figures. With several results, each names its GPU and model."""
    lines = []
    for result in results:
        pair = f" ({result['gpu']} · `{result['model']}`)" if len(results) > 1 else ""
        label = lambda name: f"- **{name}**{pair}: " if pair else f"- **{name}:** "
        how = result["measured_with"]
        how = (f"crit Eval ({how}, " + ("greedy decode" if eval_major(how) <= 6 else sampled_text(result.get("sampling"))) + ")"
               if crit_eval(how) else f"self-reported ({how})")
        files = [f"[{Path(e['file']).name}]({e['file']})" for e in result.get("evidence") or []]
        lines.append(label("Measured with") + " · ".join([how] + files))
        if result.get("hardware"):
            lines.append(label("Hardware") + result["hardware"])
        more = more_text(result.get("more") or {})
        if more:
            lines.append(label("Further figures") + more)
    return lines


def hardware(host):
    """One line naming the machine a crit Eval result ran on, from its `host`, as crit Eval's own report
    says it (crit_eval.host.summary); None when the result records no hardware (before crit Eval did)."""
    if not isinstance(host, dict) or not (host.get("os") or host.get("cpu") or host.get("gpus")):
        return None
    gib = lambda n: f"{n / 1024 ** 3:.0f} GiB"
    system = host.get("os") or {}
    parts = [(system.get("distribution") or " ".join(filter(None, (system.get("system"), system.get("release")))))
             + (" (WSL)" if system.get("wsl") else "") or host.get("platform") or "unknown host"]
    processor = host.get("cpu") or {}
    if processor.get("model"):
        cores = "/".join(str(n) for n in (processor.get("physical_cores"), processor.get("logical_cores")) if n)
        parts.append(processor["model"] + (f" ({cores} cores)" if cores else ""))
    if host.get("memory_total_bytes"):
        parts.append(gib(host["memory_total_bytes"]) + " RAM")
    for gpu in host.get("gpus") or []:
        link = gpu.get("pcie") or {}
        text = gpu.get("name") or "GPU"
        if gpu.get("memory_total_bytes"):
            text += " " + gib(gpu["memory_total_bytes"])
        if link.get("gen_max"):
            text += f", PCIe {link['gen_max']}.0 x{link.get('width_max') or '?'}"
            if link.get("gen_gpu_max") and link["gen_gpu_max"] > link["gen_max"]:
                text += f" (GPU supports {link['gen_gpu_max']}.0)"
        parts.append(text)
    return "; ".join(parts)


def short(revision):
    return revision[:8]


def facts(card):
    lines = measured(card.get("results") or [])
    r = card.get("requirements") or {}
    if r:
        parts = [", ".join(PLATFORMS.get(p, p) for p in r.get("platforms", [])), ", ".join(r.get("gpu_arch", [])),
                 f"{r.get('gpu_memory_gb')} GB GPU memory", f"CUDA {r.get('cuda')}", r.get("runtime")]
        lines.append("- **Requires:** " + " · ".join(p for p in parts if p))
    for model, c in (card.get("checkpoints") or {}).items():
        lines.append(f"- **Checkpoint** (`{model}`): [`{c['repository']}`](https://huggingface.co/{c['repository']}/tree/{c['revision']}) "
                     f"at `{short(c['revision'])}`")
    for d in card.get("downloads") or []:
        where = f"https://huggingface.co/{d['repository']}/tree/{d.get('revision', 'main')}" + (f"/{d['path']}" if d.get("path") else "")
        size = f", about {d['size_gb']} GB" if d.get("size_gb") else ""
        note = f": {d['note']}" if d.get("note") else ""
        lines.append(f"- **Download** ({d['kind']}): [`{d['repository']}" + (f"/{d['path']}" if d.get("path") else "") + f"`]({where}){size}{note}")
    s = card["source"]
    at = f"/{s['path']}" if s.get("path") else ""
    shown = s["repository"].split("://", 1)[1].split("/", 1)[1].rstrip("/")
    lines.append(f"- **Source:** [{shown}{at} @ `{short(s['revision'])}`]({s['repository'].rstrip('/')}/tree/{s['revision']}{at})")
    derived = card.get("derived_from")
    if derived:
        origin = derived["source"]
        origin_at = f"/{origin['path']}" if origin.get("path") else ""
        lines.append(f"- **Derived from:** [{derived['engine']}](../../{derived['engine']}/README.md), source "
                     f"[`{short(origin['revision'])}`]({origin['repository'].rstrip('/')}/tree/{origin['revision']}{origin_at})")
    built = card.get("built")
    if built:
        parts = []
        if built.get("hours") is not None:
            parts.append(f"{built['hours']:g} h")
        if built.get("candidates") is not None:
            parts.append(f"{built['candidates']} candidates")
        if built.get("agent_tokens") is not None:
            parts.append(f"{tokens(built['agent_tokens'])} agent tokens")
        if built.get("api_usd") is not None:
            parts.append(f"${built['api_usd']:,.2f} API-equivalent")
        if built.get("agent"):
            parts.append(f"by {built['agent']}")
        lines.append("- **Built:** " + " · ".join(parts))
    lines.append(f"- **License:** {card['license']}")
    return "\n".join(lines)


def tokens(count):
    """155,700,000 as 155.7M, 42,900 as 42.9K."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def credit(card):
    """The box under the title, the same for every engine but its kind: a note for a crit engine, a
    warning for an external one, whose source crit-hub has not reviewed. The source pin is in the facts."""
    c = card["contributor"]
    shown = c["repository"].split("://", 1)[1].split("/", 1)[1].rstrip("/")
    line = f"> **Contributed by** {c['name']} · [{shown}]({c['repository']})"
    if card["origin"] == "crit":
        return "> [!NOTE]\n" + line
    return "> [!WARNING]\n" + line + "\n>\n> External engine: crit-hub has not reviewed its source. Read it before you build or run it."


def render(card):
    """README.md for a card: the engine's name, the credit, the stats table, the facts, then the prose."""
    parts = [HEADER, "", f"# {card['title']}", "", credit(card), ""]
    if card.get("results"):
        parts += [stats_table(card["results"]), ""]
    parts += [facts(card), ""]
    readme = card.get("readme") or {}
    for key, heading in SECTIONS:
        if not readme.get(key):
            continue
        if heading:
            parts += [f"## {heading}", ""]
        parts += [readme[key].rstrip("\n"), ""]
    return "\n".join(parts).rstrip("\n") + "\n"


def engine_folders(hub):
    return sorted(p for p in (Path(hub) / "engines").glob("*/*") if p.is_dir() and (p / "card.yaml").is_file())


def main(folders, check):
    stale = []
    for folder in folders:
        text = render(load(folder))
        path = Path(folder) / "README.md"
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.as_posix())
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"rendered {path.as_posix()}")
    if stale:
        raise SystemExit("README.md out of date; run python -B tools/card.py on its folder:\n  " + "\n  ".join(stale))
    if check:
        print(f"{len(folders)} READMEs match their cards")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="*", help="engine folders (default: every engine under engines/)")
    p.add_argument("--check", action="store_true", help="fail if a README.md differs from its rendered card")
    args = p.parse_args()
    main([Path(f) for f in args.folders] or engine_folders("."), args.check)
