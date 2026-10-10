"""llama.cpp backend: one literal argv, no shell string, explicit placement."""

from __future__ import annotations

import os
import re
import subprocess
import time

from .backends import Backend, BackendError, GenerationResult


PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens?"
)
EVAL_RE = re.compile(
    r"\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens?"
)
LOAD_RE = re.compile(r"load time\s*=\s*([0-9.]+)\s*ms")


def render_prompt(messages, template=None) -> str:
    """Render the ChatML structure the Qwen3.6 GGUF advertises.

    llama-cli in single-turn mode takes already-rendered text, so the adapter
    owns template fidelity. Thinking is disabled by seeding an empty thinking
    block, which is the behaviour this engine was tuned for.
    """
    parts = []
    for message in messages:
        parts.append("<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    parts.append("" if template else "")
    return "".join(parts)


def extract_reply(stdout: str) -> str:
    """Pull assistant text out of llama-cli output."""
    marker = "<|im_start|>assistant"
    if marker not in stdout:
        return ""
    tail = stdout.split(marker, 1)[1]
    for stop in ("<|im_end|>", "[end of text]"):
        if stop in tail:
            tail = tail.split(stop, 1)[0]
    return tail.strip()


def parse_perf(stdout: str):
    """Return (prefill_ms, decode_ms, prompt_tokens, output_tokens)."""
    prompt_tokens = output_tokens = 0
    prefill_ms = decode_ms = 0.0
    if m := PROMPT_EVAL_RE.search(stdout):
        prefill_ms = float(m.group(1))
        prompt_tokens = int(m.group(2))
    if m := EVAL_RE.search(stdout):
        decode_ms = float(m.group(1))
        output_tokens = int(m.group(2))
    return prefill_ms, decode_ms, prompt_tokens, output_tokens


class LlamaCppBackend(Backend):
    """One-shot llama-cli process per reply, with explicit tensor placement."""

    def __init__(self, binary, model_path, placement, settings) -> None:
        self.binary = binary
        self.model_path = model_path
        self.placement = placement
        self.settings = settings

    def command(self, messages, limits) -> list:
        cmd = [
            self.binary,
            "-m", self.model_path,
            "-c", str(int(limits.get("kv_tokens", 8192))),
            "-n", str(int(limits.get("max_output_tokens", 128))),
            "-p", render_prompt(messages),
            "--no-warmup",
            "-st",
            "--temp", "0",
            "--top-k", "1",
        ]
        threads = self.settings.get("threads")
        if threads:
            cmd += ["-t", str(threads)]
        cmd += self.placement.to_args()
        return cmd

    def generate(self, messages, limits, cancellation):
        cmd = self.command(messages, limits)
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(self.settings["runtime_paths"]) + os.pathsep + env.get("PATH", "")
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(limits.get("timeout_seconds", 120)),
                env=env,
                cwd=self.settings.get("workdir"),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError("backend timed out after %ss" % limits.get("timeout_seconds")) from exc
        wall_ms = (time.perf_counter() - started) * 1000.0
        if cancellation is not None and cancellation.is_set():
            raise BackendError("cancelled")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-600:]
            raise BackendError("backend exit %s: %s" % (proc.returncode, tail))
        prefill_ms, decode_ms, prompt_tokens, output_tokens = parse_perf(proc.stdout)
        text = extract_reply(proc.stdout)
        missing = {}
        if not prompt_tokens:
            missing["prefill"] = "llama-cli did not report a prompt eval line"
        if not output_tokens:
            missing["decode"] = "llama-cli did not report a decode eval line"
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            ttft_ms=prefill_ms,
            notes={"wall_ms": wall_ms, "missing": missing, "argv": cmd},
        )

    def close(self) -> None:
        return None
