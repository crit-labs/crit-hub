"""Standalone terminal chat for the clean-room engine (outside the crit CLI).

  python engine/python/chat.py --model /path/to/model --package /path/to/package [--output-tokens 512]

The package is a built engine package (lib/libq38.so + vendor/), for example the
output of scripts/make_package.sh. /new starts a fresh conversation, /quit exits.
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--config", default=None, help="engine configuration JSON (default: package configuration.json)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.package, "python"))
    import chat_adapter

    config = json.loads(a.config) if a.config else json.load(open(os.path.join(a.package, "configuration.json")))
    native = {k: v for k, v in config.items() if k in chat_adapter.NATIVE_KEYS}
    frontend = chat_adapter.QwenFrontend(os.path.join(a.model, "tokenizer.json"), os.path.join(a.model, "chat_template.jinja"))
    print("loading weights...", flush=True)
    engine = chat_adapter.NativeEngine(os.path.join(a.model, "model.safetensors"), os.path.join(a.model, "model_mtp.safetensors"),
                                       native, os.path.join(a.package, "lib", "libq38.so"))
    messages = []
    try:
        while True:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                break
            if text == "/quit":
                break
            if text == "/new":
                messages = []
                print("(new conversation)")
                continue
            if not text:
                continue
            messages.append({"role": "user", "content": text})
            prompt = frontend.encode(messages)
            if len(prompt) >= engine.context:
                print("(conversation exceeds the %d-token context; use /new)" % engine.context)
                messages.pop()
                continue
            budget = min(a.output_tokens, engine.context - len(prompt))
            reply_tokens, shown = [], [""]
            sys.stdout.write("assistant> ")

            def on_token(token):
                if token not in chat_adapter.STOP_TOKENS:
                    reply_tokens.append(token)
                    out = frontend.decode(reply_tokens)
                    if not out.endswith("�") and out.startswith(shown[0]):
                        sys.stdout.write(out[len(shown[0]):])
                        sys.stdout.flush()
                        shown[0] = out
                return False

            t0 = time.time()
            tokens, stats = engine.generate(prompt, budget, on_token)
            reply = frontend.decode(reply_tokens)
            if reply.startswith(shown[0]):
                sys.stdout.write(reply[len(shown[0]):])
            rate = stats[3] / stats[4] * 1000 if stats[4] else 0
            print("\n[%d prompt tokens, %d output tokens, %.0f tok/s decode, %.0f ms first token]" % (len(prompt), len(tokens), rate, stats[5]))
            messages.append({"role": "assistant", "content": reply})
    finally:
        engine.close()


if __name__ == "__main__":
    main()
