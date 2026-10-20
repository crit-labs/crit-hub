"""ctypes binding of libkestrel's Core-backed C API, shared by the Eval and chat adapters."""
import ctypes
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int)
STOP_TOKENS = (248046, 248044)
KV_BYTES_PER_TOKEN = 16 * 4 * (256 + 4) * 2  # FP8 codes plus one FP32 scale per (position, head), K and V


def table_path(config):
    """Recorded cuBLASLt table named by the configuration (relative to this package), or empty."""
    name = config.get("gemm_table") or ""
    return os.path.join(HERE, name) if name else ""


def library_path():
    return os.environ.get("KESTREL_LIB") or os.path.join(HERE, "libkestrel.so")


class Native:
    def __init__(self, model_dir, context, prefill_chunk=4096, draft=4, attn_blocks=40, splits=8, gemm_table=""):
        self.path = library_path()
        lib = ctypes.CDLL(self.path)
        lib.ks_error.restype = ctypes.c_char_p
        lib.ks_create.restype = ctypes.c_void_p
        lib.ks_create.argtypes = [ctypes.c_char_p] + [ctypes.c_int] * 5
        lib.ks_close.argtypes = [ctypes.c_void_p]
        lib.ks_describe.restype = ctypes.c_char_p
        lib.ks_describe.argtypes = [ctypes.c_void_p]
        lib.ks_bytes_per_pass.restype = ctypes.c_longlong
        lib.ks_bytes_per_pass.argtypes = [ctypes.c_void_p]
        lib.ks_draft_bytes.restype = ctypes.c_longlong
        lib.ks_draft_bytes.argtypes = [ctypes.c_void_p]
        IP = ctypes.POINTER(ctypes.c_int)
        lib.ks_generate.argtypes = [ctypes.c_void_p, IP, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
                                    ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int, IP, ctypes.c_int, IP, ctypes.c_int,
                                    TOKEN_CB, ctypes.c_void_p, IP, ctypes.c_int]
        lib.ks_stats.argtypes = [ctypes.c_void_p, IP, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        lib.ks_score.argtypes = [ctypes.c_void_p, IP, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, IP, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), IP]
        lib.ks_set_adaptive.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        lib.ks_set_gemm_table.argtypes = [ctypes.c_char_p]
        self.lib = lib
        lib.ks_set_gemm_table(gemm_table.encode())
        self.draft = draft
        h = lib.ks_create(model_dir.encode(), int(context), int(prefill_chunk), int(draft), int(attn_blocks), int(splits))
        if not h:
            raise RuntimeError("kestrel: " + lib.ks_error().decode())
        self.h = h

    def set_adaptive(self, on, cost_ratio):
        self.lib.ks_set_adaptive(self.h, 1 if on else 0, float(cost_ratio))

    def describe(self):
        return self.lib.ks_describe(self.h).decode()

    def bytes_per_pass(self):
        return int(self.lib.ks_bytes_per_pass(self.h))

    def draft_bytes(self):
        return int(self.lib.ks_draft_bytes(self.h))

    def generate(self, prompt, max_new, on_tokens=None, temperature=-1.0, top_p=0.95, top_k=20, seed=0,
                 speculative=True, forced=None, stops=None, cancel=None):
        n = len(prompt)
        arr = (ctypes.c_int * n)(*prompt)
        fr = (ctypes.c_int * len(forced))(*forced) if forced else None
        st = (ctypes.c_int * len(stops))(*stops) if stops else None
        out = (ctypes.c_int * max(max_new, 1))()
        err = []

        def cb(_user, toks, k):
            try:
                if on_tokens:
                    on_tokens([toks[i] for i in range(k)])
                if cancel is not None and cancel():
                    return 1
            except Exception as e:  # never unwind through C; cancel the request instead
                err.append(e)
                return 1
            return 0

        ccb = TOKEN_CB(cb)
        got = self.lib.ks_generate(self.h, arr, n, int(max_new), float(temperature), float(top_p), int(top_k),
                                   int(seed), 1 if speculative else 0, fr, len(forced) if forced else 0, st,
                                   len(stops) if stops else 0, ccb, None, out, max(max_new, 1))
        if got < 0:
            raise RuntimeError("kestrel generate: " + self.lib.ks_error().decode())
        if err:
            raise err[0]
        acc = (ctypes.c_int * 65536)()
        pre = ctypes.c_double()
        rounds = self.lib.ks_stats(self.h, acc, 65536, ctypes.byref(pre))
        return list(out[:got]), rounds, list(acc[:min(rounds, 65536)]), pre.value

    def score(self, ids, first, path, queries):
        n = len(ids)
        targets = n - first
        nq = len(queries[0]) if queries else 0
        flat = [q for row in queries for q in row]
        arr = (ctypes.c_int * n)(*ids)
        qa = (ctypes.c_int * max(len(flat), 1))(*flat)
        tl = (ctypes.c_float * targets)()
        ql = (ctypes.c_float * max(targets * nq, 1))()
        am = (ctypes.c_int * targets)()
        if self.lib.ks_score(self.h, arr, n, first, path.encode(), qa, nq, tl, ql, am):
            raise RuntimeError("kestrel score: " + self.lib.ks_error().decode())
        return list(tl), [list(ql[i * nq:(i + 1) * nq]) for i in range(targets)], list(am)

    def close(self):
        if self.h:
            self.lib.ks_close(self.h)
            self.h = None
