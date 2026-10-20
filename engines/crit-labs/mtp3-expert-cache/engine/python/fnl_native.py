"""ctypes binding of libfnl.so (fnl engine on crit Core)."""
import ctypes
import json
import os
from pathlib import Path

_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)


def library_path():
    env = os.environ.get("FNL_LIB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "libfnl.so"


class Native:
    def __init__(self, config, lib=None):
        self.path = Path(lib) if lib else library_path()
        self.lib = ctypes.CDLL(str(self.path))
        L = self.lib
        L.fnl_last_error.restype = ctypes.c_char_p
        L.fnl_open.restype = ctypes.c_void_p
        L.fnl_open.argtypes = [ctypes.c_char_p]
        L.fnl_close.argtypes = [ctypes.c_void_p]
        L.fnl_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        L.fnl_generate.restype = ctypes.c_int
        L.fnl_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_ulonglong,
                                   ctypes.c_int, _CB, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                   ctypes.POINTER(ctypes.c_double)]
        L.fnl_pass_timing.restype = None
        L.fnl_pass_timing.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        L.fnl_accepted.restype = ctypes.c_int
        L.fnl_accepted.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        L.fnl_score.restype = ctypes.c_int
        L.fnl_score.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_int32)]
        self.config = dict(config)
        self.handle = L.fnl_open(json.dumps(self.config).encode())
        if not self.handle:
            raise RuntimeError("fnl_open failed: " + L.fnl_last_error().decode())

    def _error(self, what):
        return RuntimeError(f"{what}: {self.lib.fnl_last_error().decode()}")

    def info(self):
        buf = ctypes.create_string_buffer(4096)
        self.lib.fnl_info(self.handle, buf, 4096)
        return json.loads(buf.value.decode())

    def generate(self, prompt, max_new, stops=False, temperature=0.0, top_p=-1.0, top_k=-1, seed=None,
                 on_tokens=None, cancelled=None):
        """Returns (tokens, timing dict, cancelled flag)."""
        n = len(prompt)
        p = (ctypes.c_int32 * n)(*prompt)
        out = (ctypes.c_int32 * max(1, max_new))()
        timing = (ctypes.c_double * 8)()
        failure = []

        def cb(_user, toks, count):
            try:
                if on_tokens is not None:
                    on_tokens([toks[i] for i in range(count)])
                return 1 if (cancelled is not None and cancelled()) else 0
            except BaseException as error:  # surfaced after the native call returns
                failure.append(error)
                return 1

        callback = _CB(cb)
        r = self.lib.fnl_generate(self.handle, p, n, int(max_new), 1 if stops else 0, float(temperature),
                                  float(top_p), int(top_k), int(seed or 0), 1 if seed is not None else 0, callback,
                                  None, out, timing)
        if failure:
            raise failure[0]
        if r == -1:
            raise self._error("fnl_generate")
        was_cancelled = r == -2
        count = r if r >= 0 else sum(1 for _ in range(0))
        tokens = [out[i] for i in range(count)] if r >= 0 else None
        info = {"prefill_ms": timing[0], "first_token_ms": timing[1], "decode_ms": timing[2], "total_ms": timing[3],
                "decode_steps": int(timing[4]), "finish": int(timing[5])}
        return tokens, info, was_cancelled

    def pass_timing(self):
        """(target-pass ms, drafter ms) of device time in the last generate call."""
        out = (ctypes.c_double * 2)()
        self.lib.fnl_pass_timing(self.handle, out)
        return out[0], out[1]

    def accepted(self):
        """Drafts kept in each speculative round of the last generate call (empty without speculation)."""
        n = self.lib.fnl_accepted(self.handle, None, 0)
        buf = (ctypes.c_int32 * max(1, n))()
        self.lib.fnl_accepted(self.handle, buf, n)
        return [buf[i] for i in range(n)]

    def score(self, ids, first, path, queries=None):
        n = len(ids)
        targets = n - first
        per = len(queries[0]) if queries else 0
        a = (ctypes.c_int32 * n)(*ids)
        q = (ctypes.c_int32 * max(1, targets * per))()
        if per:
            flat = [v for row in queries for v in row]
            for i, v in enumerate(flat):
                q[i] = v
        t = (ctypes.c_float * max(1, targets))()
        ql = (ctypes.c_float * max(1, targets * per))()
        am = (ctypes.c_int32 * max(1, targets))()
        code = {"prefill": 0, "decode": 1, "verify": 2}[path]
        r = self.lib.fnl_score(self.handle, a, n, first, code, q, per, t, ql, am)
        if r != 0:
            raise self._error("fnl_score")
        return ([t[i] for i in range(targets)], [[ql[i * per + j] for j in range(per)] for i in range(targets)],
                [am[i] for i in range(targets)])

    def close(self):
        if self.handle:
            self.lib.fnl_close(self.handle)
            self.handle = None
