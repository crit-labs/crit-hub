"""Backend interface and the placement description for the resident engine.

A backend owns exactly one model allocation. `generate` answers one request;
`close` proves the allocation is released so crit can drop its model lease.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class BackendError(RuntimeError):
    """A backend could not serve the request."""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    ttft_ms: float = 0.0
    synthetic: bool = False
    notes: dict = field(default_factory=dict)


@dataclass
class GpuPlacement:
    """How much of the model lives in VRAM versus host RAM.

    For a sparse MoE model the routed-expert tensors of the first `n_cpu_moe`
    layers stay in system RAM while everything else (dense, attention, shared
    experts) stays resident on the device. That single split is what makes a
    model larger than the card usable at all, and it is also the dominant
    decode-rate lever, so it is the swept variable for this run.
    """

    n_gpu_layers: int = 99
    n_cpu_moe: int = 0
    tensor_overrides: tuple = ()
    label: str = "all-gpu"

    def to_args(self) -> list:
        args = ["-ngl", str(int(self.n_gpu_layers))]
        if self.n_cpu_moe:
            args += ["--n-cpu-moe", str(int(self.n_cpu_moe))]
        for override in self.tensor_overrides:
            args += ["--override-tensor", str(override)]
        return args


class Backend:
    """Minimal contract every backend implements."""

    model_path = ""

    def generate(self, messages, limits, cancellation):  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError


class SyntheticBackend(Backend):
    """Deterministic non-model backend used to exercise the adapter wiring.

    It never claims real timings: every phase is reported as missing so a
    synthetic run can never be mistaken for a measurement.
    """

    def __init__(self, configuration) -> None:
        self.configuration = configuration
        self.model_path = configuration.get("model_path", "")

    def generate(self, messages, limits, cancellation):
        last = messages[-1]["content"] if messages else ""
        return GenerationResult(text="synthetic:" + last.strip()[:64], synthetic=True)

    def close(self) -> None:
        return None
