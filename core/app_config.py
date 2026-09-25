"""The node's own settings: ``ComfyUI/user/anomalous_tts.json``.

Folders added from the UI live here instead of ``extra_model_paths.yaml`` so they
work without a restart and survive plugin updates::

    {"format": 1, "libraries": ["D:/voices/模型"], "pretrained": ["D:/GPT-SoVITS/GPT_SoVITS"]}
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from typing import Dict, List

import folder_paths

log = logging.getLogger("Anomalous_TTS")

FILENAME = "anomalous_tts.json"
KEYS = ("libraries", "pretrained")
_lock = threading.Lock()


def path() -> str:
    return os.path.join(folder_paths.get_user_directory(), FILENAME)


def load() -> Dict[str, List[str]]:
    try:
        with open(path(), encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log.warning("[Anomalous_TTS] 读取 %s 失败，按空设置处理：%s", path(), e)
        data = {}
    return {k: [p for p in data.get(k, []) if isinstance(p, str)] for k in KEYS}


def _save(data: Dict[str, List[str]]) -> None:
    target = path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"format": 1, **data}, f, ensure_ascii=False, indent=2)
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
