"""Per-character settings file ``anomalous_tts.json`` (see docs/INTERFACE.md §3).

Anomalous Model Browser writes it; the node reads it. Unknown keys are kept.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Iterable, List

FILENAME = "anomalous_tts.json"
FORMAT = 1
LANGS = {"ja", "zh", "en"}
# `defaults` numbers: the node's own input ranges (nodes.py), so a saved value is always usable.
DEFAULT_RANGES = {
    "speed": (0.5, 2.0),
    "top_k": (1, 100),
    "top_p": (0.05, 1.0),
    "temperature": (0.05, 2.0),
    "repetition_penalty": (1.0, 2.0),
}


class SettingsError(ValueError):
    pass


def load(folder: str) -> Dict[str, Any]:
    path = os.path.join(folder, FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise SettingsError(f"{path} 读取失败：{e}") from e
    if not isinstance(data, dict):
        raise SettingsError(f"{path} 必须是一个 JSON 对象。")
    return data


def _check_ref(where: str, ref: Any, audio: Iterable[str], errors: List[str]) -> None:
    if not isinstance(ref, dict):
        errors.append(f"{where} 必须是对象")
        return
    if "audio" in ref and ref["audio"] not in audio:
        errors.append(f"{where}.audio 不是这个角色文件夹里的音频：{ref['audio']}")
    if "text" in ref and not isinstance(ref["text"], str):
        errors.append(f"{where}.text 必须是字符串")
    if "language" in ref and ref["language"] not in LANGS:
        errors.append(f"{where}.language 只能是 ja / zh / en")


def validate(data: Any, gpt: Iterable[str], sovits: Iterable[str], audio: Iterable[str]) -> List[str]:
    """Return a list of problems (empty = valid). Unknown keys are allowed."""
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["设置必须是一个 JSON 对象"]
    audio = set(audio)
    if data.get("format", FORMAT) != FORMAT:
        errors.append(f"format 必须是 {FORMAT}")
    aliases = data.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
        errors.append("aliases 必须是非空字符串的列表")
    if "language" in data and data["language"] not in LANGS:
        errors.append("language 只能是 ja / zh / en")
    if "gpt" in data and data["gpt"] not in set(gpt):
        errors.append(f"gpt 不是这个角色文件夹里的 .ckpt：{data['gpt']}")
    if "sovits" in data and data["sovits"] not in set(sovits):
        errors.append(f"sovits 不是这个角色文件夹里的 .pth：{data['sovits']}")
    if "reference" in data:
        _check_ref("reference", data["reference"], audio, errors)
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        errors.append("defaults 必须是对象")
    else:
        if "language" in defaults and defaults["language"] not in LANGS | {"auto"}:
            errors.append("defaults.language 只能是 auto / ja / zh / en")
        for key, (low, high) in DEFAULT_RANGES.items():
            if key not in defaults:
                continue
            value = defaults[key]
            number = int if key == "top_k" else (int, float)
            if isinstance(value, bool) or not isinstance(value, number) or not low <= value <= high:
                errors.append(f"defaults.{key} 必须是 {low}~{high} 的{'整数' if key == 'top_k' else '数'}")
    replace = data.get("replace", {})
    if not isinstance(replace, dict) or not all(
        isinstance(k, str) and k.strip() and isinstance(v, str) for k, v in replace.items()
    ):
        errors.append("replace 必须是 {\"原文\": \"读法\"} 的对象，原文不能为空")
    emotions = data.get("emotions", {})
    if not isinstance(emotions, dict):
        errors.append("emotions 必须是对象")
    else:
        for name, ref in emotions.items():
            if not name.strip() or name.strip() == "main":
                errors.append(f"情绪名不能为空或 main：{name!r}")
            _check_ref(f"emotions.{name}", ref, audio, errors)
    return errors


def save(folder: str, data: Dict[str, Any]) -> None:
    """Atomic write, UTF-8, readable indentation."""
    data = dict(data)
    data.setdefault("format", FORMAT)
    fd, tmp = tempfile.mkstemp(prefix=".anomalous_tts.", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, os.path.join(folder, FILENAME))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
