"""Resident chat adapter for the Gemma 4 26B-A4B crit engine.

Declares ``chat-json-v1``, ``chat-resident-v1`` and ``chat-stream-v1``.
``create_engine(context)`` is the factory crit calls on entering chat.

Lifetime contract (docs/CHAT.md):
* weights load once on entering chat and stay resident across replies;
* ``reset`` clears private turn state but keeps the engine and allocation;
* ``close`` drains owned work before crit releases any model lease.
"""

from __future__ import annotations

import json
import os
import threading

from .backends import BackendError, GpuPlacement, SyntheticBackend
from .llamacpp import LlamaCppBackend
from .server_backend import LlamaServerBackend

CAPABILITIES = ["chat-json-v1", "chat-resident-v1", "chat-stream-v1"]


class Engine:
    """Holds the backend and per-conversation state for one candidate."""

    def __init__(self, context, backend, configuration) -> None:
        self.context = context
        self.backend = backend
        self.configuration = configuration
        self._lock = threading.Lock()
        self._turns = 0
        self._closed = False

    # -- required interface ------------------------------------------------

    def generate(self, request, cancellation):
        return self._run(request, cancellation, emit=None)

    def generate_stream(self, request, cancellation, emit):
        return self._run(request, cancellation, emit=emit)

    def reset(self) -> None:
        """Drop per-turn state. Weights and placement stay resident."""
        with self._lock:
            self._turns += 1

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.backend.close()

    # -- internals ---------------------------------------------------------

    def _run(self, request, cancellation, emit):
        messages = request.get("messages") or []
        limits = request.get("limits") or {}
        # A retained package cannot hold an absolute model path. The frozen
        # request carries the verified locator, so adopt it before the first
        # generation and reject a silent substitution afterwards.
        requested_model = request.get("model_path")
        if requested_model:
            current = self.configuration.get("model_path")
            if not current or str(current).startswith("PLACEHOLDER"):
                self.configuration["model_path"] = requested_model
                self.backend.model_path = requested_model
            elif os.path.normcase(str(current)) != os.path.normcase(str(requested_model)):
                raise BackendError("model locator changed for a resident engine")
        result = self.backend.generate(messages, limits, cancellation)
        if emit is not None and result.text:
            emit(result.text)
        return self._response(request, result)

    def _response(self, request, result):
        missing = dict(result.notes.get("missing") or {})
        prefill_tokens = result.prompt_tokens if result.prefill_ms else None
        prefill_ms = result.prefill_ms if result.prompt_tokens else None
        decode_tokens = result.output_tokens if result.decode_ms else None
        decode_ms = result.decode_ms if result.output_tokens else None
        if result.synthetic:
            missing["prefill"] = "synthetic backend reported no prefill timing"
            missing["decode"] = "synthetic backend reported no decode timing"
        return {
            "schema_version": 1,
            "record_kind": "chat_response",
            "request_id": request.get("request_id"),
            "candidate_id": request.get("candidate_id"),
            "package_sha256": request.get("package_sha256"),
            "assistant": result.text,
            "telemetry": {
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "prefill_tokens": prefill_tokens,
                "prefill_ms": prefill_ms,
                "decode_tokens": decode_tokens,
                "decode_ms": decode_ms,
                "ttft_ms": result.ttft_ms,
                "missing": missing,
            },
        }


def _config_argument(arguments, package_root):
    """Resolve --crit-config against the package, not the caller cwd.

    crit launches an engine from a fresh turn folder, so a relative config path
    in the launch recipe is only meaningful relative to the retained package.
    The frozen configuration is also supplied through the engine context, so a
    missing config file degrades to that instead of failing the load.
    """
    for index, item in enumerate(arguments):
        if item == "--crit-config" and index + 1 < len(arguments):
            value = arguments[index + 1]
            if os.path.isabs(value):
                return value if os.path.isfile(value) else None
            joined = os.path.join(package_root, value)
            return joined if os.path.isfile(joined) else None
    return None


def _package_root():
    """Return the retained package root, i.e. the directory above crit_engine."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def _resolve_relative(configuration, base):
    """Make package-relative paths absolute against the package root.

    A retained package is copied to a fresh location, so it cannot store
    absolute paths. The model path is supplied by the caller through the frozen
    request and is never baked into the package.
    """
    def absolute(value):
        if not value or os.path.isabs(value) or str(value).startswith("PLACEHOLDER"):
            return value
        return os.path.normpath(os.path.join(base, str(value)))

    for key in ("binary", "server_binary", "model_path", "workdir"):
        if configuration.get(key):
            configuration[key] = absolute(configuration[key])
    paths = configuration.get("runtime_paths") or []
    configuration["runtime_paths"] = [absolute(item) for item in paths]
    return configuration


def _load_configuration(context):
    configuration = dict(getattr(context, "configuration", None) or {})
    arguments = list(getattr(context, "arguments", None) or [])
    package_root = _package_root()
    config_path = _config_argument(arguments, package_root)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            configuration.update(json.load(handle))
    return _resolve_relative(configuration, package_root)


def create_engine(context):
    """Factory required by the resident chat contract.

    The server backend is the default because it applies the model's own Jinja
    chat template. Hand-rendering conversation markers is a real defect class:
    a model whose template uses unfamiliar delimiters silently treats the
    rendered prompt as plain text.
    """
    configuration = _load_configuration(context)
    mode = configuration.get("backend", "server")
    if mode == "synthetic":
        return Engine(context, SyntheticBackend(configuration), configuration)

    model_path = configuration.get("model_path") or ""
    for candidate in (getattr(context, "model_path", None),
                      (getattr(context, "limits", None) or {}).get("model_path")):
        if candidate and (not model_path or str(model_path).startswith("PLACEHOLDER")):
            model_path = candidate
    configuration["model_path"] = model_path

    placement_payload = configuration.get("placement") or {}
    placement = GpuPlacement(
        n_gpu_layers=int(placement_payload.get("n_gpu_layers", 99)),
        n_cpu_moe=int(placement_payload.get("n_cpu_moe", 0)),
        tensor_overrides=tuple(placement_payload.get("tensor_overrides") or ()),
        label=placement_payload.get("label", "all-gpu"),
    )
    settings = {
        "runtime_paths": configuration["runtime_paths"],
        "threads": configuration.get("threads", 6),
        "workdir": configuration.get("workdir"),
        "kv_tokens": configuration.get("kv_tokens", 8192),
        "base_port": configuration.get("base_port", 18080),
        "extra_args": tuple(configuration.get("server_args") or ()),
    }
    if mode == "server":
        server_binary = configuration.get("server_binary")
        if not server_binary:
            server_binary = str(configuration["binary"]).replace("llama-cli", "llama-server")
        backend = LlamaServerBackend(server_binary, configuration["model_path"], placement, settings)
        backend.start()
        return Engine(context, backend, configuration)

    backend = LlamaCppBackend(configuration["binary"], configuration["model_path"], placement, settings)
    return Engine(context, backend, configuration)
