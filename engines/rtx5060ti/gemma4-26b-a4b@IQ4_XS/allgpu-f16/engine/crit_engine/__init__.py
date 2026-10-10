"""crit engine adapter for Gemma 4 26B-A4B (GGUF, MoE) on a single RTX 5060 Ti."""

from .engine import create_engine

__all__ = ["create_engine"]
