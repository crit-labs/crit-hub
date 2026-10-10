"""llama-server backend for the Gemma 4 26B-A4B engine.

The server applies the model's own embedded Jinja chat template, so the adapter
never hand-renders conversation markers. One server process is started by
`create_engine` and lives for the whole engine lifetime: weights stay resident
across replies, which is what the resident chat contract requires.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

from .backends import Backend, BackendError, GenerationResult


def _free_port(preferred=18080):
    for candidate in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", candidate)) != 0:
                return candidate
    raise BackendError("no free local port for llama-server")


class LlamaServerBackend(Backend):
    """Owns one llama-server process for the lifetime of the engine."""

    def __init__(self, binary, model_path, placement, settings) -> None:
        self.binary = binary
        self.model_path = model_path
        self.placement = placement
        self.settings = settings
        self.port = _free_port(int(settings.get("base_port", 18080)))
        self.process = None
        self.load_seconds = 0.0

    # -- process lifetime --------------------------------------------------

    def command(self) -> list:
        cmd = [
            self.binary,
            "-m", self.model_path,
            "-c", str(int(self.settings.get("kv_tokens", 8192))),
            "-t", str(int(self.settings.get("threads", 6))),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--jinja",
            "-np", "1",
        ]
        cmd += self.placement.to_args()
        cmd += list(self.settings.get("extra_args") or [])
        return cmd

    def start(self, timeout=1800) -> None:
        if self.process is not None:
            return
        if not self.model_path or str(self.model_path).startswith("PLACEHOLDER"):
            raise BackendError("no model locator available for this engine")
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(self.settings["runtime_paths"]) + os.pathsep + env.get("PATH", "")
        started = time.perf_counter()
        self.process = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=self.settings.get("workdir"),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise BackendError("llama-server exited during load with code %s" % self.process.returncode)
            try:
                with urllib.request.urlopen(self._url("/health"), timeout=5) as response:
                    if response.status == 200:
                        self.load_seconds = time.perf_counter() - started
                        return
            except Exception:  # server not up yet
                time.sleep(1.5)
        raise BackendError("llama-server did not become healthy within %ss" % timeout)

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=30)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=15)
            except Exception:
                pass
        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass

    # -- generation --------------------------------------------------------

    def _url(self, path) -> str:
        return "http://127.0.0.1:%d%s" % (self.port, path)

    def reasoning_enabled(self, limits):
        """Whether to let the model use its extended-thinking channel.

        The default comes from the retained configuration so a packaged engine
        is self-describing; a request may override it per turn."""
        override = limits.get("reasoning")
        if override is None:
            override = self.settings.get("reasoning", False)
        if isinstance(override, str):
            return override.strip().lower() in ("1", "true", "yes", "on")
        return bool(override)

    def generate(self, messages, limits, cancellation):
        if self.process is None:
            self.start()
        payload = {
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": int(limits.get("max_output_tokens", 128)),
            "temperature": 0.0,
            "top_k": 1,
            "stream": False,
            "cache_prompt": True,
        }
        # Gemma 4 ships an extended-thinking channel. The model's own Jinja
        # template accepts enable_thinking; when it is false the template
        # pre-emits a closed thought channel so no reasoning tokens are spent.
        # Measured on this host: with thinking on, a 256-token budget was
        # consumed entirely inside the reasoning channel (0 answer characters,
        # finish_reason=length); with thinking off the same prompt finished with
        # a complete answer in about 31 percent less wall time. Chat therefore
        # defaults to skipping the channel and lets the caller opt back in.
        if self.reasoning_enabled(limits):
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._url("/v1/chat/completions"),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=float(limits.get("timeout_seconds", 300))) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BackendError("server HTTP %s: %s" % (exc.code, exc.read()[:400])) from exc
        except Exception as exc:
            raise BackendError("server request failed: %s" % exc) from exc
        wall_ms = (time.perf_counter() - started) * 1000.0
        if cancellation is not None and cancellation.is_set():
            raise BackendError("cancelled")
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason")
        # Some templates separate a thinking channel from the answer. If the
        # token budget is spent inside reasoning, `content` is empty while
        # `reasoning_content` holds every token the model actually produced;
        # reporting an empty reply there would misstate real work.
        truncated_in_reasoning = False
        if not text and reasoning:
            text = reasoning
            truncated_in_reasoning = finish_reason == "length"
        timings = body.get("timings") or {}
        prompt_tokens = int(timings.get("prompt_n") or 0)
        output_tokens = int(timings.get("predicted_n") or 0)
        prefill_ms = float(timings.get("prompt_ms") or 0.0)
        decode_ms = float(timings.get("predicted_ms") or 0.0)
        missing = {}
        if finish_reason == "length":
            missing["complete"] = "output token budget exhausted before the model finished"
        if truncated_in_reasoning:
            missing["answer"] = "budget exhausted inside the reasoning channel; no final answer was emitted"
        if not prompt_tokens:
            missing["prefill"] = "server reported no prompt token count"
        if not output_tokens:
            missing["decode"] = "server reported no predicted token count"
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            ttft_ms=prefill_ms if prompt_tokens else 0.0,
            notes={"wall_ms": wall_ms, "missing": missing, "load_seconds": self.load_seconds,
                   "reasoning_enabled": self.reasoning_enabled(limits)},
        )
