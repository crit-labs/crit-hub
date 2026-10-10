"""crit engine adapter for Qwen3.6-35B-A3B (GGUF, MoE) on a single RTX 5060 Ti."""

from .engine import create_engine

__all__ = ["create_engine"]
