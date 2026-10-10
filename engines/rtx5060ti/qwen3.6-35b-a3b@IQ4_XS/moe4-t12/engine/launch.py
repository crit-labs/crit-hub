"""Retained launch entry for the crit resident chat contract.

crit launches this file literally as:  python -B launch.py ARGS...
It is the package's `launch[0]`. The adapter in `crit_engine` implements
`create_engine(context)`; this module only wires a process to that factory.

Two modes are supported:

* `--crit-chat-request PATH` / `--crit-chat-response PATH`: the documented
  one-shot client interface, used when crit drives the engine directly.
* `--model PATH`: a small local harness for manual smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crit_engine import create_engine  # noqa: E402


class _Context:
    def __init__(self, configuration):
        self.configuration = configuration
        self.arguments = []
        self.limits = {}
        self.model_path = None


class _Cancellation:
    def __init__(self):
        self._flag = False

    def is_set(self):
        return self._flag

    def check(self):
        if self._flag:
            raise RuntimeError("cancelled")

    def set(self):
        self._flag = True


def _configuration(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _one_shot(config_path, request_path, response_path):
    context = _Context(_configuration(config_path))
    context.arguments = ["--crit-config", config_path]
    engine = create_engine(context)
    try:
        with open(request_path, "r", encoding="utf-8") as handle:
            request = json.load(handle)
        response = engine.generate(request, _Cancellation())
    finally:
        engine.reset()
        engine.close()
    temporary = response_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(response, handle, ensure_ascii=False)
    os.replace(temporary, response_path)


def _smoke(config_path, model_path, prompt):
    context = _Context(_configuration(config_path))
    context.arguments = ["--crit-config", config_path]
    context.model_path = model_path
    engine = create_engine(context)
    try:
        response = engine.generate({
            "schema_version": 1, "record_kind": "chat_request", "request_id": "launch-smoke",
            "candidate_id": "local", "package_sha256": "0" * 64,
            "model_path": model_path,
            "messages": [{"role": "user", "content": prompt}],
            "limits": {"max_output_tokens": 512, "kv_tokens": 8192, "timeout_seconds": 600},
        }, _Cancellation())
    finally:
        engine.reset()
        engine.close()
    print(json.dumps({"assistant": response["assistant"], "telemetry": response["telemetry"]},
                     indent=1, ensure_ascii=False))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--crit-config", required=True)
    parser.add_argument("--crit-chat-request")
    parser.add_argument("--crit-chat-response")
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="What is the capital of France? Answer in one short sentence.")
    args = parser.parse_args(argv)
    if args.crit_chat_request and args.crit_chat_response:
        return _one_shot(args.crit_config, args.crit_chat_request, args.crit_chat_response)
    if args.model:
        return _smoke(args.crit_config, args.model, args.prompt)
    raise SystemExit("launch.py needs --crit-chat-request/--crit-chat-response or --model")


if __name__ == "__main__":
    main(sys.argv[1:])

