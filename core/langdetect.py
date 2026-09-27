"""Guess the language of a piece of script text: ja / zh / en.

Deliberately simple: kana means Japanese; Han characters without kana are
Chinese, unless the speaking model is Japanese (a line like 「通常授業！」 is
Japanese for a Japanese voice) and the line has no character Japanese does not
write (simplified forms such as 这 / 说 / 师, or words such as 的 / 很);
Latin letters only means English.
"""

from __future__ import annotations

import re
from typing import Optional

_KANA = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
_HAN = re.compile(r"[㐀-䶿一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")
# Chinese words Japanese does not write although Shift_JIS can encode them.
_ZH_WORDS = re.compile("[呀哦嘛它很的气个从]")


def _written_only_in_chinese(text: str) -> bool:
    """True if a character is one Japanese does not use: outside Shift_JIS (cp932, which
    holds every kanji Japanese writes; simplified forms such as 这 / 说 / 师 are not in it),
    or one of the few Chinese words above."""
    if _ZH_WORDS.search(text):
        return True
    for ch in _HAN.findall(text):
        try:
            ch.encode("cp932")
        except UnicodeEncodeError:
            return True
    return False


def detect(text: str, model_language: Optional[str] = None) -> Optional[str]:
    """Return "ja", "zh", "en", or None when the text has no letters at all."""
    if _KANA.search(text):
        return "ja"
    if _HAN.search(text):
        return "ja" if model_language == "ja" and not _written_only_in_chinese(text) else "zh"
    if _LATIN.search(text):
        return "en"
    return None
