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


def pair(result):
    return f"{result['gpu']} · {result['model']}"


def stat_cells(result):
    """One result's row of the stats table: its GPU and model, then the headline stats with both decodes."""
    stats = result.get("stats") or {}
    sampled, greedy = decodes(result)
    ppl, ppl_label = ppl_cell(stats.get("ppl"))
    return [
        pair(result),
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
    """The stats as one markdown table, the stats in bold. The subheader names the GPU and model, so only
    an engine measured on several opens each row with its GPU and model."""
    rows = [stat_cells(r) for r in results]
    first = 0 if len(rows) > 1 else 1
    return "\n".join([
        "| " + " | ".join(COLUMNS[first:]) + " |",
        "|" + "|".join(" :--- " if i == 0 else " :---: " for i in range(len(COLUMNS)) if i >= first) + "|",
        *("| " + " | ".join(([row[0]] if first == 0 else []) + [f"**{v}**" if v != "–" else v for v in row[1:]]) + " |"
          for row in rows),
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
        named = f" ({pair(result)})" if len(results) > 1 else ""
        label = lambda name: f"- **{name}**{named}: " if named else f"- **{name}:** "
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


# ---------------------------------------------------------------------------------------------
# Charts: SVGs drawn from a result's crit Eval file, written next to the README under charts/ and
# shown in its Measured section: decode by context above the prose, decode by workload below it. The website shows the same files in its own colors.

CHART_STYLE = """<style>
svg { font-family: -apple-system, 'Segoe UI', Inter, Helvetica, Arial, sans-serif; }
.title { fill: #1f1b1c; font-size: 14px; font-weight: 600; }
.subtitle, .tick { fill: #6b6466; font-size: 12px; }
.cat { fill: #4a4446; font-size: 12.5px; }
.value { fill: #1f1b1c; font-size: 12.5px; font-weight: 600; }
.grid { stroke: #e6e1dc; stroke-width: 1; }
.axis { stroke: #cfc8c2; stroke-width: 1; }
.line { fill: none; stroke: #c4513e; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot, .bar { fill: #c4513e; }
.dot { stroke: #ffffff; stroke-width: 2; }
@media (prefers-color-scheme: dark) {
  .title, .value { fill: #f5f1e8; } .subtitle, .tick { fill: #8a8385; } .cat { fill: #b9b2ae; }
  .grid { stroke: #2a2a2a; } .axis { stroke: #3a3a3a; } .line { stroke: #dc6652; } .dot, .bar { fill: #dc6652; } .dot { stroke: #0d1117; }
}
</style>"""
CHART_W = 640


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg(title, subtitle, height, body):
    """A chart: its title and subtitle on top, then the plot; theme-aware for GitHub, plain classes for the website."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CHART_W} {height}" role="img" aria-label="{_esc(title)}">'
            f"<title>{_esc(title)}</title>{CHART_STYLE}"
            f'<text x="0" y="16" class="title">{_esc(title)}</text><text x="0" y="36" class="subtitle">{_esc(subtitle)}</text>'
            f"{body}</svg>\n")


def _step(top, ticks=5):
    return next((s for s in (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if top / s <= ticks), 5000)


def _tick(value):
    return f"{value:,.0f}" if value == int(value) else f"{value:g}"


def bars_chart(title, subtitle, rows, unit_text):
    """Horizontal bars from zero, one per (label, value), each with its value in text ink."""
    left, right, top, row = 120, 140, 52, 30
    plot = len(rows) * row
    peak = max(v for _, v in rows) * 1.1
    step = _step(peak)
    hi = -(-peak // step) * step
    x = lambda v: left + (CHART_W - left - right) * v / hi
    body = []
    v = 0.0
    while v <= hi + 1e-9:
        body.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 4}" y2="{top + plot}" class="{"axis" if v == 0 else "grid"}"/>'
                    f'<text x="{x(v):.1f}" y="{top + plot + 18}" class="tick" text-anchor="middle">{_tick(v)}</text>')
        v += step
    for i, (label, value) in enumerate(rows):
        y = top + i * row + 6
        body.append(f'<text x="{left - 10}" y="{y + 13}" class="cat" text-anchor="end">{_esc(label)}</text>'
                    f'<rect x="{left}" y="{y}" width="{max(2.0, x(value) - left):.1f}" height="18" rx="4" class="bar"/>'
                    f'<text x="{x(value) + 8:.1f}" y="{y + 13}" class="value">{unit_text(value)}</text>')
    return _svg(title, subtitle, top + plot + 28, "".join(body))


def line_chart(title, subtitle, points):
    """A line over evenly spaced x labels (each context doubles the last), from zero, each point labelled."""
    left, right, top, bottom, inset = 48, 16, 64, 30, 28
    height = 250
    peak = max(v for _, v in points) * 1.15
    step = _step(peak)
    hi = -(-peak // step) * step
    x = lambda i: left + inset + i * (CHART_W - left - right - 2 * inset) / max(1, len(points) - 1)
    y = lambda v: top + (height - top - bottom) * (1 - v / hi)
    body = []
    v = 0.0
    while v <= hi + 1e-9:
        body.append(f'<line x1="{left}" x2="{CHART_W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"axis" if v == 0 else "grid"}"/>'
                    f'<text x="{left - 10}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{_tick(v)}</text>')
        v += step
    body.append('<path d="' + " ".join(f"{'L' if i else 'M'}{x(i):.1f} {y(val):.1f}" for i, (_, val) in enumerate(points)) + '" class="line"/>')
    for i, (label, val) in enumerate(points):
        body.append(f'<circle cx="{x(i):.1f}" cy="{y(val):.1f}" r="4.5" class="dot"/>'
                    f'<text x="{x(i):.1f}" y="{y(val) - 12:.1f}" class="value" text-anchor="middle">{val:.1f}</text>'
                    f'<text x="{x(i):.1f}" y="{height - 8}" class="tick" text-anchor="middle">{_esc(label)}</text>')
    return _svg(title, subtitle, height, "".join(body))


def _context(tokens):
    return f"{tokens // 1024}K" if tokens >= 1024 and tokens % 1024 == 0 else f"{tokens:,}"


def result_charts(data):
    """The charts one crit Eval result file gives, by file name: decode by context and decode by workload."""
    charts = {}
    points = sorted((p["context"], p["decode_tps"]) for p in ((((data.get("status") or {}).get("kpis") or {}).get("context") or {})
                    .get("points") or {}).values() if isinstance(p.get("context"), int) and isinstance(p.get("decode_tps"), (int, float)))
    if len(points) >= 2:
        charts["decode-by-context.svg"] = line_chart("Decode by context", "tok/s, one request at each context length (crit Eval)",
                                                     [(_context(c), v) for c, v in points])
    sampled = data.get("sampled_speed") or {}
    kind, suite = ("sampled", sampled) if sampled.get("status") == "measured" and sampled.get("decode_score") else ("greedy", data.get("speed") or {})
    cases = [(name, c) for name, c in (suite.get("cases") or {}).items() if isinstance(c, dict) and isinstance(c.get("decode_tps"), (int, float))]
    if cases:
        charts["decode-by-workload.svg"] = bars_chart("Decode by workload", f"tok/s per crit Eval speed case, {kind} decode",
                                                      [(n, c["decode_tps"]) for n, c in cases], lambda v: f"{v:,.1f} tok/s")
    return charts


def charts(folder, card):
    """Every chart an engine's README shows, as {path under the folder: SVG}: for each result measured with
    crit Eval, from its first result file; under charts/<gpu>_<model>/ when the engine has several results."""
    found = {}
    results = card.get("results") or []
    for result in results:
        files = [e.get("file") for e in result.get("evidence") or [] if re.fullmatch(r"evidence/crit-eval[^/]*\.json", str(e.get("file")))]
        if not crit_eval(result.get("measured_with")) or not files:
            continue
        path = Path(folder) / files[0]
        if not path.is_file():
            continue
        prefix = "charts/" + (f"{result['gpu']}_{result['model']}/" if len(results) > 1 else "")
        for name, svg in result_charts(json.loads(path.read_text(encoding="utf-8"))).items():
            found[prefix + name] = svg
    return found


CHART_ALT = {"decode-by-context.svg": "Decode by context", "decode-by-workload.svg": "Decode by workload"}
# The charts that close the Measured section, after its prose; the others open it.
CHARTS_AFTER = {"decode-by-workload.svg"}


def render(card, figures=()):
    """README.md for a card: the engine's name, the GPUs and models it was measured on as a subheader, the
    credit, the stats table, the facts, then the prose, with its charts (paths) in the Measured section."""
    parts = [HEADER, "", f"# {card['title']}", ""]
    if card.get("results"):
        parts += ["### " + ", ".join(pair(r) for r in card["results"]), ""]
    parts += [credit(card), ""]
    if card.get("results"):
        parts += [stats_table(card["results"]), ""]
    parts += [facts(card), ""]
    readme = card.get("readme") or {}
    for key, heading in SECTIONS:
        if not readme.get(key):
            continue
        if heading:
            # A rule between sections, as crit-hub's own README has: GitHub's headings alone leave little room.
            parts += ["---", "", f"## {heading}", ""]
        images = [f"![{CHART_ALT.get(Path(f).name, Path(f).stem)}]({f})" for f in figures] if key == "measured" else []
        before = [i for i, f in zip(images, figures) if Path(f).name not in CHARTS_AFTER]
        after = [i for i, f in zip(images, figures) if Path(f).name in CHARTS_AFTER]
        parts += (before + [""] if before else []) + [readme[key].rstrip("\n"), ""] + (after + [""] if after else [])
    return "\n".join(parts).rstrip("\n") + "\n"


def engine_folders(hub):
    return sorted(p for p in (Path(hub) / "engines").glob("*/*") if p.is_dir() and (p / "card.yaml").is_file())


def write_charts(folder, drawn):
    """Writes an engine's charts and removes any other SVG under its charts/."""
    folder = Path(folder)
    for old in (folder / "charts").rglob("*.svg") if (folder / "charts").is_dir() else []:
        if old.relative_to(folder).as_posix() not in drawn:
            old.unlink()
    for name, svg in drawn.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(svg, encoding="utf-8", newline="\n")


def stale_charts(folder, drawn):
    """The chart files that differ from what the card's evidence draws, missing or left over."""
    folder = Path(folder)
    on_disk = {p.relative_to(folder).as_posix() for p in (folder / "charts").rglob("*")} if (folder / "charts").is_dir() else set()
    on_disk = {p for p in on_disk if (folder / p).is_file()}
    wrong = [n for n, svg in drawn.items() if not (folder / n).is_file() or (folder / n).read_text(encoding="utf-8") != svg]
    return sorted(set(wrong) | (on_disk - set(drawn)))


def main(folders, check):
    stale = []
    for folder in folders:
        card = load(folder)
        drawn = charts(folder, card)
        text = render(card, list(drawn))
        path = Path(folder) / "README.md"
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.as_posix())
            stale += [(Path(folder) / n).as_posix() for n in stale_charts(folder, drawn)]
        else:
            write_charts(folder, drawn)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"rendered {path.as_posix()}" + (f" and {len(drawn)} chart(s)" if drawn else ""))
    if stale:
        raise SystemExit("README.md or charts out of date; run python -B tools/card.py on its folder:\n  " + "\n  ".join(stale))
    if check:
        print(f"{len(folders)} READMEs and their charts match their cards")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="*", help="engine folders (default: every engine under engines/)")
    p.add_argument("--check", action="store_true", help="fail if a README.md differs from its rendered card")
    args = p.parse_args()
    main([Path(f) for f in args.folders] or engine_folders("."), args.check)
