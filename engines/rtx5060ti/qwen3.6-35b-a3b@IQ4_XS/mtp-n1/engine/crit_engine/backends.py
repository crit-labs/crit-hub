"""Backend abstraction for the crit adapter.

Two implementations share one interface:

* ``LlamaCppBackend`` drives the prebuilt ``llama-cli`` binary with an explicit
  literal argument list. The adapter never shells out through a shell string.
* ``SyntheticBackend`` produces deterministic scripted text so the adapter
  contract can be exercised without loading 16 GiB of weights. Every response
  it produces is labelled synthetic and must never be reported as a measurement.

The MoE placement decision is expressed as ``GpuPlacement`` because it is the
thing this project actually varies between candidates.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field


class BackendError(RuntimeError):
    """Raised when the backend cannot start or terminates abnormally."""


@dataclass(frozen=True)
class GpuPlacement:
    """Which tensors live in VRAM.

    ``n_gpu_layers`` controls the dense and recurrent part of every block.
    ``n_cpu_moe`` pins routed-expert tensors to system RAM regardless of
    ``n_gpu_layers``. ``tensor_overrides`` maps a tensor-name regex to a buffer
    type (``CPU`` or ``CUDA0``) for finer control.
    """

    n_gpu_layers: int = 99
    n_cpu_moe: int = 0
    tensor_overrides: tuple = ()
    label: str = "all-gpu"

    def to_args(self) -> list:
        args = []
        if self.n_cpu_moe:
            args += ["--n-cpu-moe", str(self.n_cpu_moe)]
        if self.n_gpu_layers is not None:
            args += ["-ngl", str(self.n_gpu_layers)]
        for override in self.tensor_overrides:
            args += ["-ot", override]
        return args


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    synthetic: bool = False
    notes: dict = field(default_factory=dict)


class Backend:
    def generate(self, messages, limits, cancellation) -> GenerationResult:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SyntheticBackend(Backend):
    """Deterministic scripted backend used to exercise the adapter contract."""

    _REPLIES = {
        "sky": (
            "Sunlight scatters off air molecules, and short blue wavelengths scatter far more "
            "strongly than long ones. With the sun overhead that scattered blue light reaches "
            "your eyes from every direction. Near sunset the light travels through much more "
            "atmosphere, so the blue is scattered away and the red end dominates."
        ),
        "17 * 3": "51",
        "python": "def total(values):\n    return sum(values)\n",
    }

    def __init__(self, configuration=None) -> None:
        self.configuration = configuration or {}
        self._calls = 0

    def generate(self, messages, limits, cancellation):
        started = time.perf_counter()
        self._calls += 1
        prompt = messages[-1]["content"] if messages else ""
        lowered = prompt.lower()
        text = "I received your message but could not match a scripted reply."
        for key, reply in self._REPLIES.items():
            if key in lowered:
                text = reply
                break
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        prompt_tokens = len(prompt.split())
        output_tokens = len(text.split())
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefill_ms=0.0,
            decode_ms=elapsed_ms,
            ttft_ms=elapsed_ms,
            synthetic=True,
            notes={"reason": "synthetic backend; no model was executed"},
        )

    def close(self) -> None:
        return None

