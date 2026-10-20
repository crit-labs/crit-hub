"""crit Eval adapter (eval-v1) for kestrel: greedy generation, replay of the pinned continuation and scoring on
the prefill, decode and verify paths, all through crit Core."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kestrel_native import table_path, KV_BYTES_PER_TOKEN, Native  # noqa: E402

DEFAULTS = {"prefill_chunk": 4096, "draft": 7, "attn_blocks": 40, "splits": 8, "adaptive_draft": True, "draft_cost_ratio": 0.07, "gemm_table": "gemm_table.txt"}


class KestrelEval:
    def __init__(self, context):
        self.config = {**DEFAULTS, **(context.get("config") or {})}
        self.context_tokens = int(context.get("context_tokens") or self.config.get("context", 262144))
        self.speculative = bool(context.get("speculative", True)) and int(self.config["draft"]) > 0
        draft = int(self.config["draft"]) if self.speculative else 0
        self.native = Native(context["model"], self.context_tokens, int(self.config["prefill_chunk"]), draft,
                             int(self.config["attn_blocks"]), int(self.config["splits"]),
                             table_path(self.config))
        self.native.set_adaptive(bool(self.config["adaptive_draft"]), float(self.config["draft_cost_ratio"]))
        self.draft = draft

    def info(self):
        applied = dict(self.config)
        applied["draft"] = self.draft
        applied["context"] = self.context_tokens
        d = self.native.bytes_per_pass()
        info = {
            "name": "kestrel",
            "configuration": applied,
            "max_context": self.context_tokens,
            "speculative": self.speculative,
            "score_paths": ["prefill", "decode", "verify"] if self.speculative else ["prefill", "decode"],
            "bytes_per_pass": d,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "replay": True,
            "native_libraries": [self.native.path],
            "kernel_work": [
                {"match": "gemv_kernel<true", "bytes_per_call": None},
            ],
        }
        info["kernel_work"] = []
        if self.speculative:
            info["draft_width"] = self.draft
            info["draft_bytes_per_step"] = self.native.draft_bytes()
        return info

    def _result(self, tokens, rounds, accepted):
        out = {"tokens": tokens, "rounds": rounds if self.speculative else len(tokens) - 1}
        if self.speculative:
            out["accepted"] = accepted
        return out

    def generate(self, prompt_ids, max_new_tokens, on_tokens):
        tokens, rounds, accepted, _ = self.native.generate(list(prompt_ids), int(max_new_tokens), on_tokens,
                                                           temperature=-1.0, speculative=self.speculative)
        return self._result(tokens, rounds, accepted)

    def replay(self, prompt_ids, continuation, on_tokens):
        cont = list(continuation)
        tokens, rounds, accepted, _ = self.native.generate(list(prompt_ids), len(cont), on_tokens, temperature=-1.0,
                                                           speculative=self.speculative, forced=cont)
        return self._result(tokens, rounds, accepted)

    def score(self, ids, first_target, path, queries):
        target, query, argmax = self.native.score(list(ids), int(first_target), path, [list(q) for q in queries])
        return {"target_logprobs": target, "query_logprobs": query, "argmax": argmax}

    def close(self):
        self.native.close()


def create_eval_engine(context):
    return KestrelEval(context)
