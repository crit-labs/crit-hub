"""Benchmark crit-hub's Qwen3.8-Flash-Next engine (mtp3-expert-cache) on your machine.

Runs greedy MTP decoding on three fixed prompts at several CPU expert shares
(FNL_CPU_SHARE), one engine load per share, and prints a Markdown block to paste
into a GitHub issue. Build the engine first (its README, "Build and chat").

    python "$ENGINE/tools/flashnext_bench.py" --model "$MODEL" --package "$WORK/package"

Each share reloads the checkpoint (several minutes). Only one engine runs at a time.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

PROMPTS = {
    "chat": "Explain in a few paragraphs how a mixture-of-experts language model decides which experts to use for a token.",
    "code": "Write a Python function that parses an ISO 8601 duration such as P3DT4H5M6S into seconds, with tests.",
}
OUTPUT_TOKENS = 256
REPEATS = 3
MIN_AVAILABLE_GIB = 68  # the engine pins ~61 GB of experts plus working memory


def long_prompt():
    # Deterministic ~14K-token document: the same text on every machine.
    parts = []
    for i in range(1, 181):
        parts.append(
            f"Section {i}. Station {i} logged a temperature of {(i * 37) % 41 - 5} degrees, "
            f"pressure {980 + (i * 13) % 60} hPa and wind {(i * 7) % 30} km/h from bearing {(i * 53) % 360}. "
            f"The operator noted that relay {i % 9} was replaced and that the backup battery read {(i * 11) % 100} percent. "
        )
    return "\n".join(parts) + "\n\nSummarize the five stations with the highest wind speed and what was replaced there."


def available_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def system_info():
    cpu = run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]).strip()
    flags = run(["sh", "-c", "grep -m1 flags /proc/cpuinfo"])
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "avx512_vnni": "avx512_vnni" in flags,
        "mem_total_gib": round(int(run(["sh", "-c", "grep MemTotal /proc/meminfo | awk '{print $2}'"]) or 0) / 1048576, 1),
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"]),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }


def child(args):
    sys.path.insert(0, str(Path(args.package).resolve()))
    import chat  # the engine package's own launcher

    engine = chat.open_engine(args.model, args.package, args.context)
    timings = []
    original = engine.generate_tokens

    def recorded(ids, max_new_tokens, sampling, on_tokens, cancelled):
        tokens, timing = original(ids, max_new_tokens, sampling, on_tokens, cancelled)
        timings.append(dict(timing, output_tokens=len(tokens or [])))
        return tokens, timing

    engine.generate_tokens = recorded
    prompts = dict(PROMPTS, document=long_prompt())
    results = {}
    try:
        for name, text in prompts.items():
            runs = []
            for rep in range(REPEATS + 1):  # first repetition is a warm-up
                request = {"request_id": uuid.uuid4().hex, "candidate_id": "bench", "package_sha256": "local",
                           # The run tag at the front keeps any resident prefix cache from shortening the prefill.
                           "messages": [{"role": "user", "content": f"Run {rep}.\n{text}"}], "temperature": 0,
                           "limits": {"max_output_tokens": OUTPUT_TOKENS}}
                timings.clear()
                start = time.perf_counter()
                engine.generate_stream(request, chat.Cancellation(), lambda _text: None)
                wall = time.perf_counter() - start
                t = timings[-1] if timings else {}
                if rep and t:
                    decode_s = t["decode_ms"] / 1000
                    runs.append({"prompt_tokens": t.get("prefill_tokens"), "output_tokens": t["output_tokens"],
                                 "first_token_s": round(t["first_token_ms"] / 1000, 3),
                                 "decode_tok_s": round((t["output_tokens"] - 1) / decode_s, 2) if decode_s else None,
                                 "prefill_tok_s": round(t["prefill_tokens"] / (t["prefill_ms"] / 1000), 1) if t.get("prefill_ms") else None,
                                 "wall_s": round(wall, 2)})
            results[name] = runs
    finally:
        engine.close()
    print("BENCH_JSON " + json.dumps(results))


def median(values):
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def parent(args):
    shares = [float(s) for s in args.shares.split(",")]
    info = system_info()
    report = {"system": info, "threads": args.threads, "shares": {}}
    for share in shares:
        free = available_gib()
        if free is not None and free < MIN_AVAILABLE_GIB:
            print(f"Stopping: only {free:.1f} GiB available, the engine needs about {MIN_AVAILABLE_GIB}. "
                  "Close other programs (or raise WSL's memory in .wslconfig) and rerun.", file=sys.stderr)
            break
        print(f"Loading the engine with CPU share {share} ({args.threads} threads)...", file=sys.stderr, flush=True)
        env = dict(os.environ, FNL_CPU_SHARE=str(share), FNL_CPU_THREADS=str(args.threads if share > 0 else 0))
        proc = subprocess.run([sys.executable, __file__, "--child", "--model", args.model, "--package", args.package,
                               "--context", str(args.context)], env=env, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("BENCH_JSON ")), None)
        if proc.returncode or line is None:
            report["shares"][str(share)] = {"error": (proc.stderr or proc.stdout)[-2000:]}
            print(f"Share {share} failed; see the report.", file=sys.stderr)
            continue
        report["shares"][str(share)] = json.loads(line[len("BENCH_JSON "):])

    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n### Flash-Next engine benchmark\n")
    print(f"- CPU: {info['cpu']} ({info['logical_cpus']} logical, AVX-512 VNNI: {info['avx512_vnni']})")
    print(f"- RAM visible to Linux: {info['mem_total_gib']} GiB; GPU: {info['gpu']}; kernel: {info['kernel']}")
    print(f"- Greedy MTP, {OUTPUT_TOKENS} output tokens, median of {REPEATS} after one warm-up, CPU threads {args.threads}\n")
    print("| CPU share | prompt | prompt tokens | decode tok/s | first token s | prefill tok/s |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for share, data in report["shares"].items():
        if "error" in data:
            print(f"| {share} | failed | | | | |")
            continue
        for name, runs in data.items():
            print(f"| {share} | {name} | {median([r['prompt_tokens'] for r in runs])} | "
                  f"{median([r['decode_tok_s'] for r in runs])} | {median([r['first_token_s'] for r in runs])} | "
                  f"{median([r['prefill_tok_s'] for r in runs])} |")
    print(f"\nFull results: {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="downloaded checkpoint directory (verified with tools/verify_model.py)")
    parser.add_argument("--package", required=True, help="directory with libfnl.so, chat.py and the Python adapters")
    parser.add_argument("--shares", default="0,0.25,0.5,0.75", help="CPU expert shares to test, comma-separated")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2 - 1),
                        help="CPU threads for missed experts (default: about one per physical core, minus one)")
    parser.add_argument("--context", type=int, default=32768, help="KV capacity to allocate (tokens)")
    parser.add_argument("--out", default="flashnext-bench.json")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    child(args) if args.child else parent(args)


if __name__ == "__main__":
    main()
