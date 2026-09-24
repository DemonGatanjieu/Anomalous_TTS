"""Split a script with ``{情绪}`` tags.

Same rule as Anomalous Model Browser ``parseTaggedSpeech`` and ComfyUI-F5-TTS:
split before every ``{...}``; text before the first tag, and ``{main}``, use
the main reference; empty parts are dropped.

    "先生！{开心}太好了！{main}那我们走吧。"
    -> [("main", "先生！"), ("开心", "太好了！"), ("main", "那我们走吧。")]
"""

from __future__ import annotations

import re
from typing import List, Tuple

MAIN = "main"
_SPLIT = re.compile(r"(?=\{[^}]+\})")
_TAG = re.compile(r"^\{([^}]+)\}")


def parse(script: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for part in _SPLIT.split(script or ""):
        m = _TAG.match(part)
        emotion = m.group(1).strip() if m else MAIN
        text = (part[m.end():] if m else part).strip()
        if text:
            out.append((emotion or MAIN, text))
    return out
