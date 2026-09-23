"""Text -> (phoneme ids, BERT features) for GPT-SoVITS v2.

Mirrors ``clean_text`` (text/cleaner.py) and ``get_phones_and_bert`` /
``get_bert_feature`` (inference_webui.py) in GPT-SoVITS for the languages we
support. Only Chinese uses BERT features; other languages get zeros.

Language mixing (e.g. English words inside Chinese) is not segmented yet:
Chinese drops Latin letters, Japanese reads them with OpenJTalk.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

import torch

BERT_DIM = 1024

# Display name -> internal code. Keep in sync with nodes.py.
LANGUAGES = {
    "日语": "ja",
    "中文": "zh",
    "英语": "en",
}

# Annotation files (.list) use these codes.
LIST_LANG_CODES = {"JA": "ja", "JP": "ja", "ZH": "zh", "EN": "en"}

# Sentence-ending punctuation used by GPT-SoVITS (``splits`` in the webui).
SPLITS = set("，。？！,.?!~:：—…")

# (norm_text, word2ph) -> (1024, n_phones) tensor
BertFn = Callable[[str, List[int]], torch.Tensor]

# GPT-SoVITS text/cleaner.py: special silence symbols for Chinese.
_ZH_SPECIAL = [("￥", "SP2"), ("^", "SP3")]


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


def _clean(text: str, lang: str) -> Tuple[List[int], Optional[List[int]], str]:
    if lang == "ja":
        from ..vendor.genie import japanese_g2p

        norm = japanese_g2p.text_normalize(text)
        return japanese_g2p.to_ids(japanese_g2p.g2p(norm)), None, norm
    if lang == "zh":
        phones, word2ph, norm = _clean_zh(text)
        return _symbol_ids(phones), word2ph, norm
    raise NotImplementedError(f"语言 {lang} 还没有接入。")


def get_phones_and_bert(
    text: str, lang: str, bert_fn: Optional[BertFn] = None, final: bool = False
) -> Tuple[List[int], torch.Tensor]:
    """Return phoneme ids and a (1024, len) float32 BERT feature tensor."""
    text = re.sub(r" {2,}", " ", text)
    phones, word2ph, norm = _clean(text, lang)
    if not final and len(phones) < 6:
        # GPT-SoVITS prepends "." to very short inputs so the model gets enough context.
        return get_phones_and_bert("." + text, lang, bert_fn, final=True)
    if lang == "zh":
        if bert_fn is None:
            raise RuntimeError("中文需要 BERT 模型。")
        bert = bert_fn(norm, word2ph).float().cpu()
    else:
        bert = torch.zeros((BERT_DIM, len(phones)), dtype=torch.float32)
    return phones, bert


def ensure_sentence_end(text: str, lang: str) -> str:
    text = text.strip("\n")
    if text and text[-1] not in SPLITS:
        text += "." if lang == "en" else "。"
    return text
