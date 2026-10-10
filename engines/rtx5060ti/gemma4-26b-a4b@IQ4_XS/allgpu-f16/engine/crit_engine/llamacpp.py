"""One-shot llama-cli backend, retained for non-resident and cold comparisons.

The resident chat path uses the server backend so weights persist between
replies. This backend exists so the same placement and sampling can be measured
in a fresh process when a cold comparison is wanted.
"""

from __future__ import annotations

import os
import subprocess

from .backends import Backend, BackendError, GenerationResult


class LlamaCppBackend(Backend):
    """Runs llama-cli once per request with literal arguments."""

    def __init__(self, binary, model_path, placement, settings) -> None:
        self.binary = binary
        self.model_path = model_path
        self.placement = placement
        self.settings = settings

    def _render(self, messages) -> str:
        pieces = []
        for message in messages:
            pieces.append("%s: %s" % (message.get("role", "user"), message.get("content", "")))
        pieces.append("assistant:")
        return "\n".join(pieces)

    def generate(self, messages, limits, cancellation):
        if not self.model_path or str(self.model_path).startswith("PLACEHOLDER"):
            raise BackendError("no model locator available for this engine")
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(self.settings["runtime_paths"]) + os.pathsep + env.get("PATH", "")
        command = [
            self.binary,
            "-m", self.model_path,
            "-c", str(int(self.settings.get("kv_tokens", 8192))),
            "-t", str(int(self.settings.get("threads", 6))),
            "-n", str(int(limits.get("max_output_tokens", 128))),
            "--temp", "0",
            "--top-k", "1",
            "--no-display-prompt",
            "-p", self._render(messages),
        ]
        command += self.placement.to_args()
        command += list(self.settings.get("extra_args") or [])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=self.settings.get("workdir"),
                timeout=float(limits.get("timeout_seconds", 300)),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError("llama-cli exceeded the request timeout") from exc
        if cancellation is not None and cancellation.is_set():
            raise BackendError("cancelled")
        if completed.returncode != 0:
            raise BackendError("llama-cli exited with %s: %s" % (completed.returncode, completed.stderr[:400]))
        return GenerationResult(
            text=(completed.stdout or "").strip(),
            notes={"missing": {"prefill": "one-shot backend does not report phase timings",
                               "decode": "one-shot backend does not report phase timings"}},
        )

    def close(self) -> None:
        return None
