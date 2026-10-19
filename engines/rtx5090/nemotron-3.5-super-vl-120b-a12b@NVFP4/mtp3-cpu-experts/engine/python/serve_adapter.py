"""crit serve resident adapter for the nmh engine: crit_core's stock native adapter (ResidentChat: the chat template
with tools and tool history, tool calls in the template's own format, per-request temperature, top_p, top_k and seed,
streaming) with the run's recon preset (temperature 1.0, top_p 0.95, enable_thinking false; stop ids 2 and 11)."""
from pathlib import Path

from crit_core.native import resident_engine

DEFAULTS = {"prefill_chunk": 32768, "vram_margin_gib": 1.5, "temperature": 1.0, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False}}


def create_engine(context):
    return resident_engine(context, Path(__file__).resolve().with_name("libnmh.so"), defaults=DEFAULTS)
