"""Gemma 4's tool-call syntax for crit_core.serving.ResidentChat.tool_format: crit registers only syntaxes several
model families share, so this package carries its model's own parser.

<|tool_call>call:NAME{key:VALUE,...}<tool_call|>, values in the chat template's format_argument syntax: strings
between <|"|> markers, bare object keys, true / false, null, numbers, [...] and {...}. All three markers are special
tokens in Gemma's tokenizer; ResidentChat keeps them in the reply text because the format lists them.
"""
import json

from crit_core.text import ToolFormat

GEMMA_OPEN, GEMMA_CLOSE, GEMMA_QUOTE = "<|tool_call>", "<tool_call|>", '<|"|>'


def _reject_constant(value):
    raise ValueError("non-finite JSON number: " + value)


def _json(text):
    return json.loads(text, parse_constant=_reject_constant)


class _GemmaValues:
    """Gemma 4's format_argument syntax from position `i` of `text`; ValueError when it does not parse."""

    def __init__(self, text, i):
        self.s, self.i = text, i

    def _space(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def _string(self):
        end = self.s.find(GEMMA_QUOTE, self.i + len(GEMMA_QUOTE))
        if end < 0:
            raise ValueError("unterminated string")
        value, self.i = self.s[self.i + len(GEMMA_QUOTE):end], end + len(GEMMA_QUOTE)
        return value

    def _items(self, close, item):
        self.i += 1
        self._space()
        if self.s.startswith(close, self.i):
            self.i += 1
            return
        while True:
            item()
            self._space()
            if self.s.startswith(",", self.i):
                self.i += 1
            elif self.s.startswith(close, self.i):
                self.i += 1
                return
            else:
                raise ValueError("expected , or " + close)

    def value(self):
        self._space()
        if self.s.startswith(GEMMA_QUOTE, self.i):
            return self._string()
        if self.s.startswith("{", self.i):
            return self.object()
        if self.s.startswith("[", self.i):
            items = []
            self._items("]", lambda: items.append(self.value()))
            return items
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ",}]":
            self.i += 1
        word = self.s[start:self.i].strip()
        if word in ("true", "false"):
            return word == "true"
        if word == "null":
            return None
        value = _json(word)  # a number: the template writes nothing else unquoted
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("not a number")
        return value

    def object(self):
        if not self.s.startswith("{", self.i):
            raise ValueError("expected {")
        out = {}

        def item():
            self._space()
            if self.s.startswith(GEMMA_QUOTE, self.i):
                key = self._string()
            else:
                colon = self.s.find(":", self.i)
                if colon < 0:
                    raise ValueError("expected :")
                key, self.i = self.s[self.i:colon].strip(), colon
            self._space()
            if not key or key in out or not self.s.startswith(":", self.i):
                raise ValueError("bad key")
            self.i += 1
            out[key] = self.value()
        self._items("}", item)
        return out


def _gemma_call(body, tools):
    if not body.startswith("call:"):
        return None
    brace = body.find("{")
    name = body[len("call:"):brace].strip() if brace > 0 else ""
    if name not in tools:
        return None
    parser = _GemmaValues(body, brace)
    try:
        arguments = parser.object()
    except ValueError:
        return None
    return (name, arguments) if not body[parser.i:].strip() else None


GEMMA_CALL = ToolFormat(_gemma_call, GEMMA_OPEN, GEMMA_CLOSE, (GEMMA_QUOTE,))
