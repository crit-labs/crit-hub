"""Standalone Flash terminal chat using a locally built package.

Added at export; this launcher is separate from the sealed native candidate.
The package must contain libfnl.so, fnl_native.py, fnl_serve.py and configuration.json.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
import uuid


class Context:
    def __init__(self, model, configuration, capacity):
        self.model = Path(model).resolve()
        self.configuration = configuration
        self.limits = {"kv_tokens": capacity}

    def model_file(self, name):
        return str(self.model / name)


class Cancellation:
    def __init__(self):
        self.event = threading.Event()

    def is_set(self):
        return self.event.is_set()

    def check(self):
        if self.is_set():
            raise KeyboardInterrupt


def open_engine(model, package, capacity=262144):
    package = Path(package).resolve()
    for name in ("libfnl.so", "fnl_native.py", "fnl_serve.py", "configuration.json"):
        if not (package / name).is_file():
            raise ValueError(f"Missing package file: {package / name}")
    if not (Path(model) / "config.json").is_file():
        raise ValueError("--model must be the downloaded checkpoint directory")
    config = json.loads((package / "configuration.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location("flash_chat_adapter", package / "fnl_serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native_spec = importlib.util.spec_from_file_location("flash_chat_native", package / "fnl_native.py")
    native = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(native)
    # Honor the selected package even when another adapter was imported in this process.
    module.Native = lambda config: native.Native(config, lib=package / "libfnl.so")
    return module.create_engine(Context(model, config, capacity))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the locally built Flash engine in a terminal")
    parser.add_argument("--model", required=True, help="downloaded checkpoint directory")
    parser.add_argument("--package", required=True, help="directory containing libfnl.so and Python adapters")
    parser.add_argument("--context", type=int, default=262144, help="total prompt plus output capacity, at most 262144")
    parser.add_argument("--output-tokens", type=int, default=1024)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="answer one prompt and exit; omit for interactive chat")
    prompt.add_argument("--prompt-file", type=Path, help="read one UTF-8 prompt file, including long multiline documents")
    args = parser.parse_args(argv)
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    if not 2 <= args.context <= 262144:
        parser.error("context must be between 2 and the supported 262144-token capacity")
    if not 1 <= args.output_tokens < args.context:
        parser.error("output-tokens must be positive and smaller than context")
    print("Loading Flash (several minutes on a cold checkpoint)...", file=sys.stderr, flush=True)
    engine = open_engine(args.model, args.package, args.context)
    messages = []
    try:
        while True:
            if args.prompt is not None:
                line = args.prompt
            else:
                try:
                    line = input("\nyou> ").strip()
                except EOFError:
                    break
                if line == "/quit":
                    break
                if line == "/new":
                    messages.clear()
                    print("(new conversation)")
                    continue
                if not line:
                    continue
            messages.append({"role": "user", "content": line})
            request = {"request_id": uuid.uuid4().hex, "candidate_id": "local", "package_sha256": "local",
                       "messages": messages, "limits": {"max_output_tokens": args.output_tokens}}
            cancellation = Cancellation()
            previous = signal.signal(signal.SIGINT, lambda *_: cancellation.event.set())
            if args.prompt is None:
                print("assistant> ", end="", flush=True)
            try:
                reply = engine.generate_stream(request, cancellation, lambda text: print(text, end="", flush=True))
                messages.append({"role": "assistant", "content": reply["assistant"]})
                print()
            except KeyboardInterrupt:
                messages.pop()
                print("\n(cancelled)", file=sys.stderr)
                if args.prompt is not None:
                    return 130
            except Exception:
                messages.pop()
                raise
            finally:
                signal.signal(signal.SIGINT, previous)
            if args.prompt is not None:
                break
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
