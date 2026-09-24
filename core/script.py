"""Script syntax (docs/INTERFACE.md §4).

    先生！{开心}やった！[pause:0.8][普拉娜]……おはようございます。

Tokens, in order:
- ``{情绪}``           -> Emotion("情绪"); ``{main}`` switches back.
- ``[名字]``            -> Speaker("名字")
- ``[pause:1.5]``, ``[pause:500ms]``, ``[停顿:1s]``, ``[wait:2]`` -> Pause(seconds)
- anything else        -> Text

The ``{...}`` rule matches Anomalous ``parseTaggedSpeech`` and ComfyUI-F5-TTS.
Square brackets that do not look like a tag (e.g. ``[1]`` inside a sentence
with spaces) are still read as a speaker name; unknown speakers are reported
by the planner, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Union

MAIN = "main"

_TOKEN = re.compile(r"\{([^{}\[\]]+)\}|\[([^\[\]{}]+)\]")
_PAUSE = re.compile(r"^\s*(?:pause|wait|stop|停顿|暂停)\s*[:：]\s*([0-9]*\.?[0-9]+)\s*(ms|s|秒)?\s*$", re.I)
MAX_PAUSE_SEC = 30.0


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Emotion:
    name: str


@dataclass(frozen=True)
class Speaker:
    name: str


@dataclass(frozen=True)
class Pause:
    seconds: float


Token = Union[Text, Emotion, Speaker, Pause]


def _bracket(content: str) -> Token:
    m = _PAUSE.match(content)
    if m:
        value = float(m.group(1))
        if (m.group(2) or "").lower() == "ms":
            value /= 1000.0
        return Pause(min(value, MAX_PAUSE_SEC))
    return Speaker(content.strip())


def tokenize(script: str) -> List[Token]:
    out: List[Token] = []
    pos = 0
    for m in _TOKEN.finditer(script or ""):
        if m.start() > pos:
            out.append(Text(script[pos : m.start()]))
        if m.group(1) is not None:
            out.append(Emotion(m.group(1).strip() or MAIN))
        else:
            out.append(_bracket(m.group(2)))
        pos = m.end()
    if pos < len(script or ""):
        out.append(Text(script[pos:]))
    return [t for t in out if not (isinstance(t, Text) and not t.text.strip())]
