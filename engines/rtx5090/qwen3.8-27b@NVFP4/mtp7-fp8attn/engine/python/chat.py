"""Standalone terminal chat for the kestrel engine (outside the crit CLI).

  python engine/python/chat.py --model /path/to/model --package /path/to/package [--context 262144] [--output-tokens 2048]

The package is build.sh's output (libkestrel.so next to the adapters and gemm_table.txt) plus configuration.json.
crit_core (from the crit checkout) must be importable. /new starts a fresh conversation, /quit exits. Added at export;
it is not part of the sealed candidate source.
"""
import argparse
import json
import os
import signal
import sys
import threading
import uuid


class _Context:
    def __init__(self, model, configuration, kv_tokens):
        self.configuration = configuration
        self.limits = {"kv_tokens": kv_tokens}
        self._model = model

    def model_file(self, name):
        return os.path.join(self._model, name)


class _Cancellation:
    def __init__(self):
        self.event = threading.Event()

    def is_set(self):
        return self.event.is_set()

    def check(self):
        if self.event.is_set():
            raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint folder (model.safetensors, model_mtp.safetensors, tokenizer)")
    ap.add_argument("--package", required=True)
    ap.add_argument("--context", type=int, default=262144, help="KV tokens to allocate")
    ap.add_argument("--output-tokens", type=int, default=2048)
    a = ap.parse_args()
    package = os.path.abspath(a.package)
    sys.path.insert(0, package)
    import chat_adapter  # noqa: E402

    configuration = json.load(open(os.path.join(package, "configuration.json"), encoding="utf-8"))
    print("loading ...", flush=True)
    engine = chat_adapter.create_engine(_Context(os.path.abspath(a.model), configuration, a.context))
    messages = []
    try:
        while True:
            try:
                line = input("\nyou> ").strip()
            except EOFError:
                break
            if line == "/quit":
                break
            if line == "/new":
                messages = []
                print("(new conversation)")
                continue
            if not line:
                continue
            messages.append({"role": "user", "content": line})
            request = {"request_id": uuid.uuid4().hex, "candidate_id": "local", "package_sha256": "local",
                       "messages": messages, "limits": {"max_output_tokens": a.output_tokens}}
            cancellation = _Cancellation()
            # Ctrl+C during a reply cancels it (the engine polls the flag every round); at the prompt it exits.
            previous = signal.signal(signal.SIGINT, lambda *_: cancellation.event.set())
            print("\nassistant> ", end="", flush=True)
            try:
                reply = engine.generate_stream(request, cancellation, lambda text: print(text, end="", flush=True))
            except KeyboardInterrupt:
                print("\n(stopped)")
                messages.pop()
                continue
            finally:
                signal.signal(signal.SIGINT, previous)
            t = reply["telemetry"]
            print(f"\n[{t['output_tokens']} tokens, first token {t.get('ttft_ms') or 0:.0f} ms]")
            messages.append({"role": "assistant", "content": reply["assistant"]})
    finally:
        engine.close()


if __name__ == "__main__":
    main()
