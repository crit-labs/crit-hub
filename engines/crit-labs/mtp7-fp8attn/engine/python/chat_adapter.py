"""crit serve resident adapter for kestrel on crit Core's shared ResidentChat: the model's chat template with tools and
tool history, tool calls in the template's own format, per-request temperature / top_p / top_k / seed over the frozen
sampling preset, streamed text without the model's stop tokens and Core's phase timing."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kestrel_native import table_path, STOP_TOKENS, Native  # noqa: E402

from crit_core.serving import ResidentChat  # noqa: E402

DEFAULTS = {
    "prefill_chunk": 4096,
    "draft": 7,
    "adaptive_draft": True,
    "draft_cost_ratio": 0.07,
    "gemm_table": "gemm_table.txt",
    "attn_blocks": 40,
    "splits": 8,
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 0},
}
MAX_TOP_K = 64  # the device sampler keeps at most 64 candidates; a larger top_k is refused rather than capped


class Engine(ResidentChat):
    sampling = ("temperature", "top_p", "top_k", "seed")

    def __init__(self, context):
        cfg = {**DEFAULTS, **dict(context.configuration or {})}
        model = Path(context.model_file("config.json")).parent
        super().__init__(context, model, configuration=cfg, stop_ids=STOP_TOKENS)
        self.preset = {**DEFAULTS["sampling"], **(cfg.get("sampling") or {})}
        self.kv_tokens = int(context.limits["kv_tokens"])
        self.native = Native(str(model), self.kv_tokens, int(cfg["prefill_chunk"]), int(cfg["draft"]),
                             int(cfg["attn_blocks"]), int(cfg["splits"]), table_path(cfg))
        self.native.set_adaptive(bool(cfg["adaptive_draft"]), float(cfg["draft_cost_ratio"]))
        self.turn = 0

    def effective_configuration(self):
        applied = dict(self.configuration)
        applied["sampling"] = dict(self.preset)
        return applied

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        values = {**self.preset, **sampling}
        top_k = int(values["top_k"])
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K} for this engine")
        self.turn += 1
        seed = int(sampling["seed"]) if "seed" in sampling else int(self.preset.get("seed", 0)) * 1000003 + self.turn
        stamps = []

        def received(tokens):
            stamps.append(time.perf_counter())
            on_tokens(tokens)

        t0 = time.perf_counter()
        tokens, _rounds, _accepted, prefill_ms = self.native.generate(
            list(ids), int(max_new_tokens), received, temperature=float(values["temperature"]),
            top_p=float(values["top_p"]), top_k=top_k, seed=seed, speculative=True, stops=list(STOP_TOKENS),
            cancel=cancelled)
        if cancelled():
            return None, {}
        first = (stamps[0] - t0) * 1000.0 if stamps else prefill_ms
        decode = (stamps[-1] - stamps[0]) * 1000.0 if len(stamps) > 1 else 0.0
        return tokens, {"prefill_ms": prefill_ms, "first_token_ms": first, "decode_ms": decode}

    def close(self):
        if self.native:
            self.native.close()
            self.native = None


def create_engine(context):
    return Engine(context)
