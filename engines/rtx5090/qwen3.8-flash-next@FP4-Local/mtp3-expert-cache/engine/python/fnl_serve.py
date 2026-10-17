"""crit serve resident adapter (chat-json-v1, chat-resident-v1, chat-stream-v1, chat-config-v1) for fnl."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnl_native import Native  # noqa: E402

DEFAULTS = {"prefill_chunk": 16384, "vram_margin_gib": 1.0, "enable_thinking": False}
NATIVE_KEYS = ("prefill_chunk", "vram_margin_gib", "expert_slots", "temperature", "top_p", "top_k")


class Engine:
    def __init__(self, context):
        from crit_core.text import load_model_text
        self.context = context
        self.configuration = dict(DEFAULTS)
        self.configuration.update(context.configuration or {})
        model_dir = Path(context.model_file("config.json")).parent
        limits = getattr(context, "limits", {}) or {}
        self.capacity = int(limits.get("kv_tokens") or self.configuration.get("capacity") or 262144)
        native = {k: self.configuration[k] for k in NATIVE_KEYS if k in self.configuration}
        native.update(model=str(model_dir), capacity=self.capacity)
        self.text = load_model_text(str(model_dir), clock=None)
        self.native = Native(native)
        self.stop_ids = [248044, 248046]

    def effective_configuration(self):
        return dict(self.configuration)

    def _prompt(self, request):
        messages = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
        return self.text.encode_chat(messages, add_generation_prompt=True,
                                     enable_thinking=bool(self.configuration.get("enable_thinking")))

    def _run(self, request, cancellation, emit=None):
        ids = self._prompt(request)
        limits = request.get("limits", {})
        max_out = int(limits.get("max_output_tokens") or 1024)
        kv = int(limits.get("kv_tokens") or self.capacity)
        if len(ids) + max_out > min(kv, self.capacity):
            raise ValueError(f"prompt of {len(ids)} tokens plus {max_out} output tokens exceeds the context")
        stream = self.text.text_stream(ids)
        pieces = []
        first = [None]
        started = time.perf_counter()

        def on_tokens(tokens):
            if first[0] is None:
                first[0] = time.perf_counter()
            tokens = [t for t in tokens if t not in self.stop_ids]
            if not tokens:
                return
            delta = stream.push(tokens)
            if delta.text:
                pieces.append(delta.text)
                if emit:
                    emit(delta.text)

        tokens, timing, cancelled = self.native.generate(ids, max_out, stops=True, temperature=-1.0,
                                                         on_tokens=on_tokens,
                                                         cancelled=lambda: cancellation.is_set())
        if cancelled:
            cancellation.check()
            raise RuntimeError("generation cancelled")
        tail = stream.finish()
        if tail.text:
            pieces.append(tail.text)
            if emit:
                emit(tail.text)
        text = "".join(pieces)
        n_out = len(tokens)
        decode_tokens = max(0, timing["decode_steps"])
        telemetry = {
            "prompt_tokens": len(ids), "output_tokens": n_out,
            "prefill_tokens": len(ids), "prefill_ms": timing["prefill_ms"],
            "decode_tokens": min(decode_tokens, max(0, n_out - 1)) if n_out > 1 else None,
            "decode_ms": timing["decode_ms"] if n_out > 1 else None,
            "ttft_ms": timing["first_token_ms"], "missing": {},
        }
        if n_out <= 1:
            telemetry["missing"]["decode"] = "the reply has a single token"
        return text, telemetry

    def _response(self, request, text, telemetry):
        return {"schema_version": 1, "record_kind": "chat_response", "request_id": request["request_id"],
                "candidate_id": request["candidate_id"], "package_sha256": request["package_sha256"],
                "assistant": text, "telemetry": telemetry}

    def generate(self, request, cancellation):
        text, telemetry = self._run(request, cancellation)
        return self._response(request, text, telemetry)

    def generate_stream(self, request, cancellation, emit):
        sent = []

        def forward(piece):
            sent.append(piece)
            emit(piece)

        text, telemetry = self._run(request, cancellation, forward)
        return self._response(request, text, telemetry)

    def reset(self):
        pass

    def close(self):
        if self.native:
            self.native.close()
            self.native = None


def create_engine(context):
    return Engine(context)
