"""Optional Python packages per language, reported to the UI and in error messages."""

from __future__ import annotations

import importlib.util
import logging
import sys
from typing import Dict, List

log = logging.getLogger("Anomalous_TTS")

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


def hide_broken_pkg_resources() -> None:
    """Call before jieba is imported. jieba reads its dictionary through ``pkg_resources`` and only
    falls back to plain files on ImportError, but a setuptools older than Python 3.12 support raises
    AttributeError (``pkgutil.ImpImporter``), so Chinese could not load at all. Mark it missing instead."""
    if "pkg_resources" in sys.modules:
        return
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        pass
    except AttributeError:
        sys.modules["pkg_resources"] = None
        log.warning("[Anomalous_TTS] setuptools 太旧，和这个 Python 不兼容；中文照常能用，建议升级：%s", install_command(["-U", "setuptools"]))
