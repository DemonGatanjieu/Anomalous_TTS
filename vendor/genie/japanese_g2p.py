# -*- coding: utf-8 -*-
"""Japanese grapheme-to-phoneme for GPT-SoVITS v2 symbols.

Based on Genie-TTS 2.0.2 ``genie_tts/G2P/Japanese/JapaneseG2P.py``
(MIT, High-Logic), which is itself a trimmed copy of GPT-SoVITS
``text/japanese.py`` (MIT, RVC-Boss).

Anomalous_TTS changes (see UPSTREAM.md):
- Consecutive-punctuation collapsing follows GPT-SoVITS (any run of the
  symbol-table punctuation, not only repeats of one character).
- Phonemes missing from the symbol table become "UNK" like GPT-SoVITS,
  instead of being dropped.
- The symbol table is GPT-SoVITS ``text/symbols2.py`` (identical to Genie's).
- No GPT-SoVITS user dictionary (``ja_userdic``); English words inside Japanese
  text are read by OpenJTalk's own rules.
"""

import re
from typing import List

import pyopenjtalk

from ..gpt_sovits.text.symbols2 import punctuation, symbols

_SYMBOL_TO_ID = {s: i for i, s in enumerate(symbols)}

_PUNCT_CLASS = "".join(re.escape(p) for p in punctuation)
_CONSECUTIVE_PUNCTUATION_RE = re.compile(f"([{_PUNCT_CLASS}])([{_PUNCT_CLASS}])+")

_SYMBOLS_TO_JAPANESE = [
    (re.compile("%"), "パーセント"),
    (re.compile("％"), "パーセント"),
]

_JAPANESE_MARKS_RE = re.compile(
    r"[^A-Za-z\d々぀-ヿ一-鿿１-９Ａ-Ｚａ-ｚｦ-ﾝ]"
)

_POST_REPLACE = {
    "：": ",", "；": ",", "，": ",", "。": ".",
    "！": "!", "？": "?", "\n": ".", "·": ",",
    "、": ",", "...": "…",
}


def text_normalize(text: str) -> str:
    return _CONSECUTIVE_PUNCTUATION_RE.sub(r"\1", text)


def _numeric_feature_by_regex(regex: str, s: str) -> int:
    match = re.search(regex, s)
    return int(match.group(1)) if match else -50


def _g2p_prosody(text: str) -> List[str]:
    """Phonemes plus prosody marks from OpenJTalk full-context labels."""
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    phones: List[str] = []
    for n, lab_curr in enumerate(labels):
        p3 = re.search(r"-(.*?)\+", lab_curr).group(1)
        if p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == "sil":
            if n == 0:
                phones.append("^")
            elif n == len(labels) - 1:
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                phones.append("?" if e3 == 1 else "$")
            continue
        if p3 == "pau":
            phones.append("_")
            continue
        phones.append(p3)

        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        lab_next = labels[n + 1] if n + 1 < len(labels) else ""
        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", lab_next)

        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        elif a2 == 1 and a2_next == 2:
            phones.append("[")
    return phones


def g2p(norm_text: str) -> List[str]:
    """Normalized Japanese text -> phoneme strings (GPT-SoVITS v2 symbols)."""
    text = norm_text
    for regex, replacement in _SYMBOLS_TO_JAPANESE:
        text = regex.sub(replacement, text)
    text = text.lower()

    segments = _JAPANESE_MARKS_RE.split(text)
    marks = _JAPANESE_MARKS_RE.findall(text)
    phones: List[str] = []
    for i, segment in enumerate(segments):
        if segment:
            # Drop the sentence-level "^" / "$" because we work per segment.
            phones.extend(_g2p_prosody(segment)[1:-1])
        if i < len(marks):
            mark = marks[i].replace(" ", "")
            if mark:
                phones.append(mark)
    phones = [_POST_REPLACE.get(p, p) for p in phones]
    return ["UNK" if p not in _SYMBOL_TO_ID else p for p in phones]


def to_ids(phones: List[str]) -> List[int]:
    return [_SYMBOL_TO_ID[p] for p in phones]
