"""The node's own settings: ``ComfyUI/user/anomalous_tts.json``.

They live here instead of ``extra_model_paths.yaml`` so they survive plugin
updates::

    {"format": 1, "storage": "D:/voices", "libraries": ["E:/old voices"],
     "import_folders": ["D:/GPT-SoVITS"], "pretrained": ["D:/GPT-SoVITS/GPT_SoVITS"]}

``storage`` is where characters are kept and imported to (absent =
``models/gpt_sovits``); ``libraries`` are more folders with characters;
``import_folders`` are the only folders the import routes may list and read
files from. The user writes these three in the file; no request changes them, so
a request cannot choose where folders are created or which files are read.
``pretrained`` (GPT-SoVITS packages to take base models from) is also set from
the UI.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

import folder_paths

log = logging.getLogger("Anomalous_TTS")

FILENAME = "anomalous_tts.json"
KEYS = ("libraries", "import_folders", "pretrained")
_lock = threading.Lock()
_warned: set = set()


def path() -> str:
    return os.path.join(folder_paths.get_user_directory(), FILENAME)


def _read(strict: bool) -> Dict[str, Any]:
    """The file as written. A broken file reads as empty, or with ``strict`` is a ValueError, so
    that saving never replaces what the user wrote by hand."""
    try:
        with open(path(), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("不是 JSON 对象")
        return data
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        if strict:
            raise ValueError(f"设置文件读不了，先修好它：{path()}（{e}）") from e
        log.warning("[Anomalous_TTS] 读取 %s 失败，按空设置处理：%s", path(), e)
        return {}


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
    data = _read(strict=False)
    out: Dict[str, Any] = {}
    for k in KEYS:
        items = data.get(k, [])
        items = items if isinstance(items, list) else [items]
        out[k] = [f for f in map(_folder, items) if f]
    out["storage"] = _folder(data.get("storage"))
    return out


def _save(key: str, folders: List[str]) -> None:
    """Write ``key``; everything else in the file stays as the user wrote it."""
    data = _read(strict=True)
    data["format"] = data.get("format", 1)
    data[key] = folders
    target = path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def add(key: str, folder: str) -> bool:
    """Remember a folder under ``key`` (``pretrained`` only: the others are the user's to write).
    False if it was already there."""
    with _lock:
        data = load()
        if folder in data[key]:
            return False
        data[key].append(folder)
        _save(key, data[key])
        return True


def remove(key: str, folder: str) -> bool:
    with _lock:
        data = load()
        if folder not in data[key]:
            return False
        data[key].remove(folder)
        _save(key, data[key])
        return True
