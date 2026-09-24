"""Text -> (phoneme ids, BERT features) for GPT-SoVITS v2.

Mirrors ``clean_text`` (text/cleaner.py) and ``get_phones_and_bert`` /
``get_bert_feature`` (inference_webui.py) in GPT-SoVITS for the languages we
support. Only Chinese uses BERT features; other languages get zeros.

Mixed text:
- Chinese: runs of Latin letters are read as English (GPT-SoVITS does the same
  with its language segmenter in "zh" mode).
- Japanese: Latin letters stay Japanese and are read by OpenJTalk plus the
  GPT-SoVITS user dictionary (like GPT-SoVITS "all_ja" mode).
"""

from __future__ import annotations

import re
from typing import List, Protocol, Tuple

import torch

BERT_DIM = 1024

# Sentence-ending punctuation used by GPT-SoVITS (``splits`` in the webui).
SPLITS = set("，。？！,.?!~:：—…")

# GPT-SoVITS text/cleaner.py: special silence symbols for Chinese.
_ZH_SPECIAL = [("￥", "SP2"), ("^", "SP3")]

# A run of English words inside Chinese text.
_LATIN_RUN = re.compile(r"[A-Za-z]+(?:[\s'’.\-]+[A-Za-z]+)*")


class Context(Protocol):
    def bert(self, norm_text: str, word2ph: List[int]) -> torch.Tensor: ...

    def prepare(self, lang: str) -> None: ...

    def can_use(self, lang: str) -> bool: ...


def _symbol_ids(phones: List[str]) -> List[int]:
    from ..vendor.gpt_sovits.text.symbols2 import symbols

    table = {s: i for i, s in enumerate(symbols)}
    return [table["UNK" if p not in table else p] for p in phones]


def _clean_zh(text: str) -> Tuple[List[str], List[int], str]:
    from ..vendor.gpt_sovits.text import chinese2

    for special, target in _ZH_SPECIAL:
        if special in text:
            text = text.replace(special, ",")
            norm = chinese2.text_normalize(text)
            phones, word2ph = chinese2.g2p(norm)
            return [target if p == "," else p for p in phones], word2ph, norm
    norm = chinese2.text_normalize(text)
    phones, word2ph = chinese2.g2p(norm)
    assert len(phones) == sum(word2ph) and len(norm) == len(word2ph)
    return phones, word2ph, norm


def _clean_en(text: str) -> Tuple[List[str], str]:
    from ..vendor.gpt_sovits.text import english

    norm = english.text_normalize(text)
    phones = english.g2p(norm)
    if len(phones) < 4:
        phones = [","] + phones
    return phones, norm


def _segments(text: str, lang: str) -> List[Tuple[str, str]]:
    """Split Chinese text into (lang, text) runs; other languages stay whole."""
    if lang != "zh" or not _LATIN_RUN.search(text):
        return [(lang, text)]
    out: List[Tuple[str, str]] = []
    pos = 0
    for m in _LATIN_RUN.finditer(text):
        if m.start() > pos:
            out.append(("zh", text[pos : m.start()]))
        out.append(("en", m.group()))
        pos = m.end()
    if pos < len(text):
        out.append(("zh", text[pos:]))
    return [(l, t) for l, t in out if t.strip()]


def _phones_and_bert_one(text: str, lang: str, ctx: Context) -> Tuple[List[int], torch.Tensor]:
    ctx.prepare(lang)
    if lang == "ja":
        from ..vendor.genie import japanese_g2p

        ids = japanese_g2p.to_ids(japanese_g2p.g2p(japanese_g2p.text_normalize(text)))
        return ids, torch.zeros((BERT_DIM, len(ids)), dtype=torch.float32)
    if lang == "en":
        phones, _ = _clean_en(text)
        ids = _symbol_ids(phones)
        return ids, torch.zeros((BERT_DIM, len(ids)), dtype=torch.float32)
    if lang == "zh":
        phones, word2ph, norm = _clean_zh(text)
        return _symbol_ids(phones), ctx.bert(norm, word2ph).float().cpu()
    raise NotImplementedError(f"语言 {lang} 还没有接入。")


def get_phones_and_bert(text: str, lang: str, ctx: Context, final: bool = False) -> Tuple[List[int], torch.Tensor]:
    """Return phoneme ids and a (1024, len) float32 BERT feature tensor."""
    text = re.sub(r" {2,}", " ", text)
    ids: List[int] = []
    berts: List[torch.Tensor] = []
    segments = _segments(text, lang)
    if any(l != lang for l, _ in segments) and not all(ctx.can_use(l) for l, _ in segments if l != lang):
        segments = [(lang, text)]  # e.g. English packages missing: fall back to the main language
    for seg_lang, seg_text in segments:
        seg_ids, seg_bert = _phones_and_bert_one(seg_text, seg_lang, ctx)
        ids.extend(seg_ids)
        berts.append(seg_bert)
    if not final and len(ids) < 6:
        # GPT-SoVITS prepends "." to very short inputs so the model gets enough context.
        return get_phones_and_bert("." + text, lang, ctx, final=True)
    bert = torch.cat(berts, dim=1) if berts else torch.zeros((BERT_DIM, 0))
    return ids, bert


def ensure_sentence_end(text: str, lang: str) -> str:
    text = text.strip("\n")
    if text and text[-1] not in SPLITS:
        text += "." if lang == "en" else "。"
    return text
