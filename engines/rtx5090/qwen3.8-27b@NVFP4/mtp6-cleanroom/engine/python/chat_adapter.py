"""crit resident chat adapter (chat-json-v1, chat-resident-v1, chat-stream-v1,
chat-model-file-v1) for the clean-room Qwen3.8-27B NVFP4 engine.

The native engine (lib/libq38.so) owns weights, KV/recurrent state and CUDA
graphs for the engine lifetime; each generate() is one crit Core request.
"""
import ctypes
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENDOR = os.path.join(_ROOT, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import jinja2  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
STOP_TOKENS = (248046, 248044)  # <|im_end|>, <|endoftext|>
NATIVE_KEYS = ("context", "prefill_chunk", "draft_tokens", "draft_rows", "graphs", "attn_split")


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


class QwenFrontend:
    """Original tokenizer.json and chat template, non-thinking generation prompt."""

    def __init__(self, tokenizer_path, template_path):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = _raise
        with open(template_path, encoding="utf-8") as handle:
            self.template = env.from_string(handle.read())

    def render(self, messages):
        return self.template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)

    def encode(self, messages):
        return self.tokenizer.encode(self.render(messages), add_special_tokens=False).ids

    def decode(self, tokens):
        return self.tokenizer.decode(list(tokens), skip_special_tokens=False)


class NativeEngine:
    def __init__(self, main_file, mtp_file, configuration, lib_path=None):
        self.lib = ctypes.CDLL(lib_path or os.path.join(_ROOT, "lib", "libq38.so"))
        self.lib.q38_open_files.restype = ctypes.c_void_p
        self.lib.q38_open_files.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.q38_generate.restype = ctypes.c_int
        self.lib.q38_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, _CB, ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                          ctypes.c_char_p, ctypes.c_int]
        self.lib.q38_close.argtypes = [ctypes.c_void_p]
        err = ctypes.create_string_buffer(2048)
        self.handle = self.lib.q38_open_files(str(main_file).encode(), str(mtp_file).encode(),
                                              json.dumps(configuration).encode(), err, len(err))
        if not self.handle:
            raise RuntimeError("Engine construction failed: " + err.value.decode(errors="replace"))
        self.context = int(configuration.get("context", 4096))

    def generate(self, prompt, max_out, on_token, stops=STOP_TOKENS):
        n = len(prompt)
        prompt_arr = (ctypes.c_int * n)(*prompt)
        stop_arr = (ctypes.c_int * max(1, len(stops)))(*stops)
        out = (ctypes.c_int * max_out)()
        stats = (ctypes.c_double * 8)()
        err = ctypes.create_string_buffer(2048)
        failure = []

        def trampoline(token, _user):
            try:
                return int(bool(on_token(token)))
            except BaseException as error:  # stop generation; re-raised below
                failure.append(error)
                return 1

        callback = _CB(trampoline)
        count = self.lib.q38_generate(self.handle, prompt_arr, n, max_out, stop_arr, len(stops), 0, callback, None,
                                      out, max_out, stats, err, len(err))
        if failure:
            raise failure[0]
        if count < 0:
            raise RuntimeError("Generation failed: " + err.value.decode(errors="replace"))
        return list(out[:count]), list(stats)

    def close(self):
        if self.handle:
            self.lib.q38_close(self.handle)
            self.handle = None


class Engine:
    def __init__(self, context):
        configuration = dict(context.configuration or {})
        self.frontend = QwenFrontend(context.model_file("tokenizer.json"), context.model_file("chat_template.jinja"))
        native = {k: v for k, v in configuration.items() if k in NATIVE_KEYS}
        self.native = NativeEngine(context.model_file("model.safetensors"), context.model_file("model_mtp.safetensors"), native)

    def generate(self, request, cancellation):
        return self.generate_stream(request, cancellation, None)

    def generate_stream(self, request, cancellation, emit):
        limits = request["limits"]
        prompt = self.frontend.encode(request["messages"])
        capacity = min(int(limits["kv_tokens"]), self.native.context)
        if len(prompt) >= capacity:
            raise ValueError("Conversation of %d tokens exceeds the %d-token context" % (len(prompt), capacity))
        max_out = min(int(limits["max_output_tokens"]), capacity - len(prompt))
        text_tokens, emitted = [], [""]

        def on_token(token):
            if token not in STOP_TOKENS:
                text_tokens.append(token)
                if emit is not None:
                    text = self.frontend.decode(text_tokens)
                    if not text.endswith("�") and len(text) > len(emitted[0]) and text.startswith(emitted[0]):
                        emit(text[len(emitted[0]):])
                        emitted[0] = text
            return cancellation.is_set()

        tokens, stats = self.native.generate(prompt, max_out, on_token)
        cancellation.check()
        assistant = self.frontend.decode(text_tokens)
        if emit is not None:
            if not assistant.startswith(emitted[0]):
                raise RuntimeError("Incremental decoding diverged from the final text")
            if len(assistant) > len(emitted[0]):
                emit(assistant[len(emitted[0]):])
        prefill_tokens, prefill_ms, decode_tokens, decode_ms, ttft_ms = int(stats[1]), stats[2], int(stats[3]), stats[4], stats[5]
        missing = {}
        if prefill_tokens <= 0:
            missing["prefill"] = "single-token prompt: its only token is processed by the first decode round"
        if decode_tokens <= 0:
            missing["decode"] = "no tokens generated"
        if not tokens:
            missing["ttft"] = "no tokens generated"
        telemetry = {
            "prompt_tokens": len(prompt), "output_tokens": len(tokens),
            "prefill_tokens": prefill_tokens if prefill_tokens > 0 else None,
            "prefill_ms": prefill_ms if prefill_tokens > 0 else None,
            "decode_tokens": decode_tokens if decode_tokens > 0 else None,
            "decode_ms": decode_ms if decode_tokens > 0 else None,
            "ttft_ms": ttft_ms if tokens else None, "missing": missing,
        }
        return {
            "schema_version": 1, "record_kind": "chat_response",
            "request_id": request["request_id"], "candidate_id": request["candidate_id"],
            "package_sha256": request["package_sha256"], "assistant": assistant, "telemetry": telemetry,
        }

    def reset(self):
        # Native request state is private and reset when the next request opens;
        # generate() returns only after Core drained and retired the request.
        pass

    def close(self):
        self.native.close()


def create_engine(context):
    return Engine(context)
