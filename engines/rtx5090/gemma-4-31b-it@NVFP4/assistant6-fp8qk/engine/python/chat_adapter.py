"""crit serve resident adapter for g4 on crit Core's shared ResidentChat: Gemma 4's chat template with tools and tool
history, per-request temperature / top_p / top_k / seed over the frozen sampling preset, streamed text without the
model's stop tokens, tool calls parsed with the package's own Gemma syntax (gemma_tools) and Core's phase timing."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g4_native import table_path, STOP_TOKENS, Native  # noqa: E402
from gemma_tools import GEMMA_CALL  # noqa: E402

from crit_core.serving import ResidentChat  # noqa: E402

DEFAULTS = {
    "prefill_chunk": 2048,
    "ring": 3200,
    "draft": 6,
    "adaptive_draft": False,
    "draft_cost_ratio": 0.05,
    "gemm_table": "",
    "slide_blocks": 8,
    "global_blocks": 40,
    "splits": 8,
    "draft_model": "/llm/models/crit/google-gemma-4-31B-it-assistant/627c5ec1458b9086b841a91e0512fd31fd2fbbf1",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 0},
}
CHANNEL_OPEN, CHANNEL_CLOSE = 100, 101  # <|channel>, <channel|>
MAX_TOP_K = 64  # the device sampler keeps at most 64 candidates; a larger top_k is refused rather than capped


class Engine(ResidentChat):
    sampling = ("temperature", "top_p", "top_k", "seed")
    tool_format = GEMMA_CALL   # crit registers no Gemma syntax: ResidentChat parses calls with it (chat-tools-v1)

    def __init__(self, context):
        cfg = {**DEFAULTS, **dict(context.configuration or {})}
        model = Path(context.model_file("config.json")).parent
        super().__init__(context, model, configuration=cfg, stop_ids=STOP_TOKENS)
        self.preset = {**DEFAULTS["sampling"], **(cfg.get("sampling") or {})}
        self.kv_tokens = int(context.limits["kv_tokens"])
        draft = int(cfg["draft"])
        self.native = Native(str(model), cfg["draft_model"] if draft else "", self.kv_tokens, int(cfg["prefill_chunk"]),
                             int(cfg["ring"]), draft, int(cfg["slide_blocks"]), int(cfg["global_blocks"]),
                             int(cfg["splits"]), table_path(cfg))
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
        thought = {"open": False}

        def received(tokens):
            stamps.append(time.perf_counter())
            # A thought channel (<|channel>thought ... <channel|>) is reasoning, not reply text: the model opens an
            # empty one after tool results, where the generation prompt does not already hold it.
            kept = []
            for t in tokens:
                if t == CHANNEL_OPEN:
                    thought["open"] = True
                elif t == CHANNEL_CLOSE and thought["open"]:
                    thought["open"] = False
                elif not thought["open"]:
                    kept.append(t)
            on_tokens(kept)

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
