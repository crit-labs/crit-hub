"""crit serve resident adapter for fnl on crit_core.serving.ResidentChat: the model's chat template with tools and
tool history, tool calls in the template's own format, per-request temperature, top_p, top_k and seed, streaming and telemetry.
temperature 0 decodes greedily in MTP-drafted rounds; other temperatures sample one token per pass."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crit_core.serving import ResidentChat  # noqa: E402
from fnl_native import Native  # noqa: E402

DEFAULTS = {"prefill_chunk": 16384, "vram_margin_gib": 1.0, "enable_thinking": False}
NATIVE_KEYS = ("prefill_chunk", "vram_margin_gib", "expert_slots", "temperature", "top_p", "top_k")
STOP_IDS = (248044, 248046)


class Engine(ResidentChat):
    sampling = ("temperature", "top_p", "top_k", "seed")

    def __init__(self, context):
        configuration = dict(DEFAULTS)
        configuration.update(context.configuration or {})
        model_dir = Path(context.model_file("config.json")).parent
        limits = getattr(context, "limits", {}) or {}
        capacity = int(limits.get("kv_tokens") or configuration.get("capacity") or 262144)
        super().__init__(context, model_dir, configuration=configuration, stop_ids=STOP_IDS, capacity=capacity,
                         template_variables={"enable_thinking": bool(configuration.get("enable_thinking"))})
        native = {k: configuration[k] for k in NATIVE_KEYS if k in configuration}
        native.update(model=str(model_dir), capacity=capacity)
        self.native = Native(native)

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        # Absent fields keep the engine's configured sampling (temperature < 0, top_p <= 0, top_k < 0).
        tokens, timing, stopped = self.native.generate(
            ids, max_new_tokens, stops=True, temperature=sampling.get("temperature", -1.0),
            top_p=sampling.get("top_p", -1.0), top_k=sampling.get("top_k", -1), seed=sampling.get("seed"),
            on_tokens=on_tokens, cancelled=cancelled)
        if stopped:
            return None, {}
        return tokens, {"prefill_tokens": len(ids), "prefill_ms": timing["prefill_ms"],
                        "decode_ms": timing["decode_ms"], "first_token_ms": timing["first_token_ms"]}

    def close(self):
        if self.native:
            self.native.close()
            self.native = None


def create_engine(context):
    return Engine(context)
