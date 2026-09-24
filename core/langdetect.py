"""Guess the language of a piece of script text: ja / zh / en.

Deliberately simple: kana means Japanese; Han characters without kana are
Chinese, unless the speaking model is Japanese (a line like 「通常授業！」 is
Japanese for a Japanese voice); Latin letters only means English.
"""

from __future__ import annotations

import re
from typing import Optional

_KANA = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
_HAN = re.compile(r"[㐀-䶿一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def detect(text: str, model_language: Optional[str] = None) -> Optional[str]:
    """Return "ja", "zh", "en", or None when the text has no letters at all."""
    if _KANA.search(text):
        return "ja"
    if _HAN.search(text):
        return "ja" if model_language == "ja" else "zh"
    if _LATIN.search(text):
        return "en"
    return None
