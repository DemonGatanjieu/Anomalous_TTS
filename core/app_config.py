"""The node's own settings: ``ComfyUI/user/anomalous_tts.json``.

Folders chosen in the UI live here instead of ``extra_model_paths.yaml`` so they
work without a restart and survive plugin updates::

    {"format": 1, "storage": "D:/voices", "libraries": ["E:/old voices"], "pretrained": ["D:/GPT-SoVITS/GPT_SoVITS"]}

``storage`` is where characters are kept (absent = ``models/gpt_sovits``);
``libraries`` are earlier storage places that still hold characters.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from typing import Any, Dict, Optional

import folder_paths

log = logging.getLogger("Anomalous_TTS")

FILENAME = "anomalous_tts.json"
KEYS = ("libraries", "pretrained")
_lock = threading.Lock()


def path() -> str:
    return os.path.join(folder_paths.get_user_directory(), FILENAME)


def load() -> Dict[str, Any]:
    try:
        with open(path(), encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log.warning("[Anomalous_TTS] 读取 %s 失败，按空设置处理：%s", path(), e)
        data = {}
    out: Dict[str, Any] = {k: [p for p in data.get(k, []) if isinstance(p, str)] for k in KEYS}
    out["storage"] = data.get("storage") if isinstance(data.get("storage"), str) else None
    return out


def _save(data: Dict[str, Any]) -> None:
    target = path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"format": 1, **{k: v for k, v in data.items() if v is not None}}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def add(key: str, folder: str) -> bool:
    """Remember a folder under ``key``. False if it was already there."""
    with _lock:
        data = load()
        if folder in data[key]:
            return False
        data[key].append(folder)
        _save(data)
        return True


def remove(key: str, folder: str) -> bool:
    with _lock:
        data = load()
        if folder not in data[key]:
            return False
        data[key].remove(folder)
        _save(data)
        return True


def set_storage(folder: Optional[str]) -> None:
    """Where characters are kept; None = the default ``models/gpt_sovits``."""
    with _lock:
        data = load()
        data["storage"] = folder
        _save(data)
