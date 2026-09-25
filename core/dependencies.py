"""Optional Python packages per language, reported to the UI and in error messages."""

from __future__ import annotations

import importlib.util
import sys
from typing import Dict, List

# import name -> pip name
LANG_PACKAGES: Dict[str, Dict[str, str]] = {
    "ja": {"pyopenjtalk": "pyopenjtalk-plus"},
    "zh": {"pypinyin": "pypinyin", "jieba": "jieba", "opencc": "opencc"},
    "en": {"g2p_en": "g2p_en", "wordsegment": "wordsegment", "nltk": "nltk"},
}


def missing(lang: str) -> List[str]:
    """Pip names of the packages ``lang`` needs that are not installed."""
    return [pip for module, pip in LANG_PACKAGES[lang].items() if importlib.util.find_spec(module) is None]


def install_command(packages: List[str]) -> str:
    """The pip command for the Python running ComfyUI (the portable build has its own)."""
    return f'"{sys.executable}" -m pip install {" ".join(packages)}'
