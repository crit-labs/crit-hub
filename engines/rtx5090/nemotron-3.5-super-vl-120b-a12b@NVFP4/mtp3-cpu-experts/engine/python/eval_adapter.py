"""crit Eval adapter (eval-v1) for the nmh engine: crit_core's stock native-engine adapter. Eval opens engines one
after another in one process, so the pinned expert records are kept for the next engine (keep_host_weights)."""
from pathlib import Path

from crit_core.native import eval_engine

DEFAULTS = {"prefill_chunk": 32768, "vram_margin_gib": 1.5, "keep_host_weights": True}


def create_eval_engine(context):
    return eval_engine(context, Path(__file__).resolve().with_name("libnmh.so"), defaults=DEFAULTS)
