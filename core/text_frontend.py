"""Text -> (phoneme ids, BERT features) for GPT-SoVITS v2.

Mirrors ``get_phones_and_bert`` in GPT-SoVITS ``inference_webui.py`` for the
languages we support. Japanese and English get zero BERT features (GPT-SoVITS
only uses BERT for Chinese).
"""

from __future__ import annotations

import re
from typing import List, Tuple

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


def _g2p(text: str, lang: str) -> List[int]:
    if lang == "ja":
        from ..vendor.genie import japanese_g2p

        return japanese_g2p.to_ids(japanese_g2p.g2p(japanese_g2p.text_normalize(text)))
    raise NotImplementedError(f"语言 {lang} 还没有接入（目前只支持日语）。")


def get_phones_and_bert(text: str, lang: str, final: bool = False) -> Tuple[List[int], torch.Tensor]:
    """Return phoneme ids and a (1024, len) float32 BERT feature tensor."""
    text = re.sub(r" {2,}", " ", text)
    phones = _g2p(text, lang)
    if not final and len(phones) < 6:
        # GPT-SoVITS prepends "." to very short inputs so the model gets enough context.
        return get_phones_and_bert("." + text, lang, final=True)
    bert = torch.zeros((BERT_DIM, len(phones)), dtype=torch.float32)
    return phones, bert


def ensure_sentence_end(text: str, lang: str) -> str:
    text = text.strip("\n")
    if text and text[-1] not in SPLITS:
        text += "." if lang == "en" else "。"
    return text
