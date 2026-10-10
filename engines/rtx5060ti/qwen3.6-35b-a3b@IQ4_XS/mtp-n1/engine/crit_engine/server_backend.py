"""llama-server backend.

The server applies the model's embedded Jinja chat template and keeps weights
resident across requests, so the adapter no longer hand-renders markers. That
hand-rendering was a real defect: this model uses <|turn>/<turn|> delimiters, not
ChatML <|im_start|> markers, so hand-built prompts were treated as plain text.

The server is started once by create_engine and reused for every reply; its
lifetime is the engine lifetime. Only documented, literal argv is used.
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
            except Exception:  # noqa: BLE001 - server not up yet
                time.sleep(1.5)
        raise BackendError("llama-server did not become healthy within %ss" % timeout)

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                process.kill()
                process.wait(timeout=15)
            except Exception:  # noqa: BLE001
                pass
        finally:
            for stream in (process.stdout,):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass

    # -- generation --------------------------------------------------------

    def _url(self, path) -> str:
        return "http://127.0.0.1:%d%s" % (self.port, path)

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
        except Exception as exc:  # noqa: BLE001
            raise BackendError("server request failed: %s" % exc) from exc
        wall_ms = (time.perf_counter() - started) * 1000.0
        if cancellation is not None and cancellation.is_set():
            raise BackendError("cancelled")
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason")
        # This template separates the thinking channel from the answer. When the
        # token budget is exhausted during reasoning, `content` is empty while
        # `reasoning_content` holds everything the model produced. Reporting an
        # empty reply there would be wrong: real tokens were generated.
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
            notes={"wall_ms": wall_ms, "missing": missing, "load_seconds": self.load_seconds},
        )
