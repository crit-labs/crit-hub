"""crit serve resident adapter (chat-json-v1, chat-resident-v1, chat-stream-v1, chat-config-v1, chat-model-file-v1)
for kestrel. Renders the model's chat template, generates through crit Core with the frozen sampling preset and
streams text without retraction."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kestrel_native import table_path, STOP_TOKENS, Native  # noqa: E402

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


class Engine:
    def __init__(self, context):
        from crit_core.text import load_model_text

        self.context = context
        cfg = dict(context.configuration or {})
        self.config = {**DEFAULTS, **cfg}
        self.sampling = {**DEFAULTS["sampling"], **(cfg.get("sampling") or {})}
        self.model_dir = os.path.dirname(context.model_file("config.json"))
        limits = context.limits
        self.kv_tokens = int(limits["kv_tokens"])
        self.text = load_model_text(self.model_dir, clock=None)
        self.native = Native(self.model_dir, self.kv_tokens, int(self.config["prefill_chunk"]), int(self.config["draft"]),
                             int(self.config["attn_blocks"]), int(self.config["splits"]),
                             table_path(self.config))
        self.native.set_adaptive(bool(self.config["adaptive_draft"]), float(self.config["draft_cost_ratio"]))
        self.turn = 0

    def effective_configuration(self):
        applied = dict(self.config)
        applied["sampling"] = dict(self.sampling)
        return applied

    def _prompt(self, request):
        from crit_core.text.messages import openai_messages

        return self.text.encode_chat(openai_messages(request["messages"]), add_generation_prompt=True)

    def _run(self, request, cancellation, emit):
        ids = list(self._prompt(request))
        limit = int(request["limits"]["max_output_tokens"])
        if len(ids) + limit > self.kv_tokens:
            raise ValueError(f"prompt of {len(ids)} tokens plus {limit} output tokens exceeds the {self.kv_tokens}-token context")
        stream = self.text.text_stream(ids, stop_strings=[], stop_token_ids=list(STOP_TOKENS))
        pieces, stamps = [], []
        state = {"done": False, "consumed": 0}

        def on_tokens(tokens):
            stamps.append(time.perf_counter())
            if state["done"]:
                return
            delta = stream.push(tokens)
            state["consumed"] += delta.consumed
            if delta.text:
                pieces.append(delta.text)
                if emit:
                    emit(delta.text)
            if delta.finish_reason:
                state["done"] = True

        def cancel():
            return state["done"] or cancellation.is_set()

        self.turn += 1
        seed = int(self.sampling.get("seed", 0)) * 1000003 + self.turn
        t0 = time.perf_counter()
        tokens, _rounds, _acc, _pre = self.native.generate(
            ids, limit, on_tokens, temperature=float(self.sampling["temperature"]), top_p=float(self.sampling["top_p"]),
            top_k=int(self.sampling["top_k"]), seed=seed, speculative=True, stops=list(STOP_TOKENS), cancel=cancel)
        cancellation.check()
        if not state["done"]:
            tail = stream.finish()
            if tail.text:
                pieces.append(tail.text)
                if emit:
                    emit(tail.text)
        out = len(tokens)
        telemetry = {"prompt_tokens": len(ids), "output_tokens": out, "missing": {}}
        if stamps:
            ttft = (stamps[0] - t0) * 1000.0
            telemetry.update(prefill_tokens=len(ids), prefill_ms=ttft, ttft_ms=ttft)
            if out > 1 and len(stamps) > 1:
                telemetry.update(decode_tokens=out - 1, decode_ms=(stamps[-1] - stamps[0]) * 1000.0)
            else:
                telemetry.update(decode_tokens=None, decode_ms=None)
                telemetry["missing"]["decode"] = "one output token"
        else:
            telemetry.update(prefill_tokens=None, prefill_ms=None, decode_tokens=None, decode_ms=None, ttft_ms=None)
            telemetry["missing"].update(prefill="no token emitted", decode="no token emitted", ttft="no token emitted")
        return {
            "schema_version": 1,
            "record_kind": "chat_response",
            "request_id": request["request_id"],
            "candidate_id": request["candidate_id"],
            "package_sha256": request["package_sha256"],
            "assistant": "".join(pieces),
            "telemetry": telemetry,
        }

    def generate(self, request, cancellation):
        return self._run(request, cancellation, None)

    def generate_stream(self, request, cancellation, emit):
        return self._run(request, cancellation, emit)

    def reset(self):
        pass  # every request starts from fresh device state; generation is synchronous

    def close(self):
        if self.native:
            self.native.close()
            self.native = None


def create_engine(context):
    return Engine(context)
