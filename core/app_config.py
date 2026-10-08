"""The node's own settings: ``ComfyUI/user/anomalous_tts.json``, written by the user.

They live here instead of ``extra_model_paths.yaml`` so they survive plugin
updates::

    {"format": 1, "storage": "D:/voices", "libraries": ["E:/old voices"], "pretrained": ["D:/GPT-SoVITS/GPT_SoVITS"]}

``storage`` is where characters are kept and imported to (absent =
``models/gpt_sovits``); ``libraries`` are more folders with characters;
``pretrained`` are GPT-SoVITS packages to take base models from instead of
downloading them. The node only reads this file: no request changes it, so a
request cannot choose where folders are created or which folders are read.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import folder_paths

log = logging.getLogger("Anomalous_TTS")

FILENAME = "anomalous_tts.json"
KEYS = ("libraries", "pretrained")
_warned: set = set()


def path() -> str:
    return os.path.join(folder_paths.get_user_directory(), FILENAME)


def _folder(value: Any) -> Optional[str]:
    """An absolute folder path; anything else is ignored (with a warning once), so that an empty or
    relative entry never stands for ComfyUI's working folder."""
    if isinstance(value, str) and value.strip() and os.path.isabs(value.strip()):
        return value.strip()
    if value is not None and repr(value) not in _warned:
        _warned.add(repr(value))
        log.warning("[Anomalous_TTS] %s 里的 %r 不是绝对路径，已忽略。", path(), value)
    return None


def load() -> Dict[str, Any]:
    try:
        with open(path(), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("不是 JSON 对象")
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log.warning("[Anomalous_TTS] 读取 %s 失败，按空设置处理：%s", path(), e)
        data = {}
    out: Dict[str, Any] = {}
    for k in KEYS:
        items = data.get(k, [])
        items = items if isinstance(items, list) else [items]
        out[k] = [f for f in map(_folder, items) if f]
    out["storage"] = _folder(data.get("storage"))
    return out
