"""crit Eval adapter (eval-v1) for the fnl engine."""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnl_native import Native, library_path  # noqa: E402

DEFAULTS = {"prefill_chunk": 16384, "vram_margin_gib": 1.0}
TOP_K = 10
EXPERTS = 512
GDN_STATE_BYTES = 36 * 48 * 128 * 128 * 4
KV_WINDOW_TOKENS = 512 * 4 + 3          # QSA reads at most the selected blocks plus the open block per layer
KV_BYTES_PER_TOKEN_LAYER = 2 * 2 * (256 + 4)  # K and V, 2 heads x 256 E4M3 values plus a scale each
INDEX_KEY_BYTES = 128 * 4               # one FP32 pooled index key per 4 tokens


def checkpoint_bytes(model):
    """(bytes read by every pass, bytes of the routed experts one token uses, bytes read by one draft step)
    from the safetensors headers. A draft step reads the MTP block with one token's routed experts and the
    drafting head, which is the BF16 lm_head re-encoded as MXFP8 (one byte per weight, one scale per 32)."""
    model = Path(model)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    resident = experts = mtp = mtp_experts = head = 0
    for rel in sorted(set(index["weight_map"].values())):
        with open(model / rel, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        header.pop("__metadata__", None)
        for name, meta in header.items():
            size = meta["data_offsets"][1] - meta["data_offsets"][0]
            if name.startswith("mtp."):
                if ".mlp.experts." in name:
                    mtp_experts += size
                else:
                    mtp += size
                continue
            if name == "lm_head.weight":
                head = size // 2 * 33 // 32
            if "visual" in name or "embed_tokens" in name or "ngram_embedding" in name:
                continue
            if ".mlp.experts." in name:
                experts += size
            else:
                resident += size
    return resident, experts * TOP_K // EXPERTS, mtp + mtp_experts * TOP_K // EXPERTS + head


class FnlEval:
    def __init__(self, context):
        self.context = context
        cfg = dict(DEFAULTS)
        cfg.update(context.get("config") or {})
        cfg["model"] = str(context["model"])
        cfg["capacity"] = int(context.get("context_tokens") or cfg.get("capacity") or 262144)
        # The harness asks for plain one-token decoding (exactness check) with speculative = False.
        cfg["speculative"] = bool(cfg.get("speculative", True)) and bool(context.get("speculative", True))
        self.config = cfg
        self.native = Native(cfg)
        facts = self.native.info()
        self.speculative = bool(facts.get("speculative"))
        self.draft_width = int(facts.get("draft_width") or 0)
        resident, active, draft = checkpoint_bytes(cfg["model"])
        kv_window = 12 * KV_WINDOW_TOKENS * KV_BYTES_PER_TOKEN_LAYER
        self.bytes_per_pass = resident + active + GDN_STATE_BYTES + kv_window
        self.draft_bytes_per_step = draft + KV_WINDOW_TOKENS * KV_BYTES_PER_TOKEN_LAYER

    def info(self):
        return {
            "name": "fnl",
            "configuration": self.config,
            "max_context": self.config["capacity"],
            "speculative": self.speculative,
            "draft_width": self.draft_width if self.speculative else None,
            "score_paths": ["prefill", "decode", "verify"] if self.speculative else ["prefill", "decode"],
            "bytes_per_pass": self.bytes_per_pass,
            "draft_bytes_per_step": self.draft_bytes_per_step if self.speculative else None,
            # Growth per held token: one pooled FP32 index key per 4 tokens in each of the 12 QSA layers
            # (the attended K/V window is bounded and counted in bytes_per_pass).
            "kv_bytes_per_token": 12 * INDEX_KEY_BYTES // 4,
            "native_libraries": [str(self.native.path)],
            "replay": False,
        }

    def generate(self, prompt_ids, max_new_tokens, on_tokens):
        tokens, timing, _ = self.native.generate(list(prompt_ids), int(max_new_tokens), stops=False, temperature=0.0,
                                                 on_tokens=on_tokens)
        if self.speculative:
            accepted = self.native.accepted()
            # Device time of the rounds' target passes (verify, acceptance, state commit; the one-token passes at
            # the context end included) and of their drafter steps (extend pass and chained drafts).
            verify_ms, draft_ms = self.native.pass_timing()
            return {"tokens": tokens, "rounds": len(accepted), "accepted": accepted, "verify_ms": verify_ms,
                    "draft_ms": draft_ms, "timing": timing}
        return {"tokens": tokens, "rounds": max(0, len(tokens) - 1), "timing": timing}

    def score(self, ids, first_target, path, queries):
        target, query, argmax = self.native.score(list(ids), int(first_target), path, queries)
        return {"target_logprobs": target, "query_logprobs": query, "argmax": argmax}

    def close(self):
        self.native.close()


def create_eval_engine(context):
    return FnlEval(context)
