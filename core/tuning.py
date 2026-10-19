"""How each line is sampled and how fast it is spoken (docs/INTERFACE.md §3 `defaults`, §5.4).

From lowest to highest priority:

1. the default preset (``ComfyUI/user/anomalous_tts_presets.json``, else "standard"; besides
   the three built-in ones the file can hold presets made in Anomalous, ``custom_<id>``);
2. the character's own ``defaults`` (a preset and / or numbers, and its speed);
3. a line in another language than its reference: the rule for that pair of languages
   ("ja>zh"), else the one for every other pair ("cross", "stable" unless set);
4. the character's ``defaults.by_language`` entry for the line's language;
5. the node's ``preset`` input: every line's sampling numbers from that preset (speed stays);
6. node inputs changed from the node's own defaults (and not just the character's
   ``defaults`` sent back by Anomalous);
7. the line's own ``[preset:…]`` / ``[speed:…]`` in the script (core/script.py).

The presets file is written by Anomalous (``POST /anomalous_tts/presets``, this computer
only); it holds names and numbers, never paths.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import folder_paths

from . import settings

log = logging.getLogger("Anomalous_TTS")

FILENAME = "anomalous_tts_presets.json"
FORMAT = 1
SAMPLING = ("top_k", "top_p", "temperature", "repetition_penalty")
# "standard" is what the node always used; "stable" is what it used for a line in another
# language than its reference (top_k at most 10, temperature at most 0.8).
PRESETS: Dict[str, Dict[str, float]] = {
    "stable": {"top_k": 10, "top_p": 1.0, "temperature": 0.8, "repetition_penalty": 1.35},
    "standard": {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35},
    "lively": {"top_k": 30, "top_p": 1.0, "temperature": 1.1, "repetition_penalty": 1.35},
}
DEFAULT = "standard"
CROSS = "cross"  # every pair of different languages without a rule of its own
CROSS_DEFAULT = "stable"
MAX_CUSTOM = 20
LABEL_LENGTH = 24


@dataclass(frozen=True)
class Tuning:
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    speed: float = 1.0
    source: str = DEFAULT  # a preset, "character" or "node": what decided it, for the info output

    def sampling_key(self) -> tuple:
        return (int(self.top_k), float(self.top_p), float(self.temperature), float(self.repetition_penalty))


# ---------- the presets file ----------
def path() -> str:
    return os.path.join(folder_paths.get_user_directory(), FILENAME)


def validate(data: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["预设必须是一个 JSON 对象"]
    if data.get("format", FORMAT) != FORMAT:
        errors.append(f"format 必须是 {FORMAT}")
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        errors.append("presets 必须是对象")
        presets = {}
    custom = [name for name in presets if name not in PRESETS]
    if len(custom) > MAX_CUSTOM:
        errors.append(f"自建预设最多 {MAX_CUSTOM} 个")
    for name, block in presets.items():
        if name not in PRESETS and not settings.CUSTOM_PRESET.fullmatch(str(name)):
            errors.append(f"没有这个预设：{name}")
        elif not isinstance(block, dict):
            errors.append(f"presets.{name} 必须是对象")
        elif name in PRESETS:
            if set(block) - set(SAMPLING):
                errors.append(f"presets.{name} 只能写 {' / '.join(SAMPLING)}")
            settings.check_block(f"presets.{name}", block, errors)
        else:
            label = block.get("label")
            if not isinstance(label, str) or not label.strip() or len(label) > LABEL_LENGTH or not label.isprintable():
                errors.append(f"presets.{name}.label 必须是 1~{LABEL_LENGTH} 个字的名字")
            if set(block) != {"label", *SAMPLING}:
                errors.append(f"presets.{name} 要写 label 和 {' / '.join(SAMPLING)}，不能多也不能少")
            settings.check_block(f"presets.{name}", {k: v for k, v in block.items() if k != "label"}, errors)
    known = {*PRESETS, *custom}
    if "default" in data and data["default"] not in known:
        errors.append(f"default 不是已有的预设：{data['default']}")
    languages = data.get("languages", {})
    if not isinstance(languages, dict):
        errors.append("languages 必须是对象")
    else:
        for pair, name in languages.items():
            ref, _, lang = str(pair).partition(">")
            if pair != CROSS and not (ref in settings.LANGS and lang in settings.LANGS and ref != lang):
                errors.append(f"languages 的键是 参考语言>生成语言（如 ja>zh）或 {CROSS}：{pair}")
            if name not in known:
                errors.append(f"languages.{pair} 不是已有的预设：{name}")
    return errors


def load() -> Dict[str, Any]:
    """The presets file, or {} (missing, unreadable or invalid: the built-in presets apply)."""
    try:
        with open(path(), encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("[Anomalous_TTS] 读取 %s 失败，用内置预设：%s", path(), e)
        return {}
    problems = validate(data)
    if problems:
        log.warning("[Anomalous_TTS] %s 不对，用内置预设：%s", path(), "；".join(problems))
        return {}
    return data


def save(data: Dict[str, Any]) -> None:
    """Atomic write; the caller validated ``data``."""
    data = {**data, "format": FORMAT}
    folder = os.path.dirname(path())
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".anomalous_tts_presets.", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path())
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def view(data: Mapping[str, Any]) -> Dict[str, Any]:
    """The presets file as the API shows it: every preset with its numbers in effect, built-in ones first."""
    tuner = Tuner(data)
    return {
        "default": tuner.default,
        "presets": {name: tuner.preset(name) for name in tuner.names},
        "labels": {name: tuner.label(name) for name in tuner.names if name not in PRESETS},
        "builtin": PRESETS,
        "languages": {k: v for k, v in tuner.languages.items() if k != CROSS},
        "cross": tuner.languages.get(CROSS, CROSS_DEFAULT),
    }


# ---------- one line's settings ----------
class Tuner:
    """``config``: the presets file; ``node``: node inputs (SAMPLING and speed) changed from the node's
    defaults; ``cross_lingual`` False leaves the language rules out; ``preset``: the node's choice of one
    preset for every line ("" = the layers decide; a name the file does not know is left out)."""

    def __init__(self, config: Mapping[str, Any], node: Optional[Mapping[str, float]] = None, cross_lingual: bool = True,
                 preset: str = ""):
        self.default = config.get("default", DEFAULT)
        self.overrides = config.get("presets", {})
        self.names = [*PRESETS, *(name for name in self.overrides if name not in PRESETS)]
        self.languages = config.get("languages", {})
        self.node = dict(node or {})
        self.cross_lingual = cross_lingual
        self.chosen = preset if preset in self.names else ""

    def preset(self, name: str) -> Dict[str, float]:
        if name not in PRESETS:
            return {k: self.overrides[name][k] for k in SAMPLING}
        return {**PRESETS[name], **self.overrides.get(name, {})}

    def label(self, name: str) -> str:
        """A preset made in Anomalous by the name it was given; any other source as it is."""
        block = self.overrides.get(name) if name not in PRESETS else None
        return block["label"] if isinstance(block, Mapping) else name

    def _apply(self, values: Dict[str, float], block: Mapping[str, Any]) -> bool:
        """A preset name and / or numbers over ``values``; True if the block set anything.
        A preset deleted since is left out, so the line keeps the general setting."""
        known = block.get("preset") in self.names
        if known:
            values.update(self.preset(block["preset"]))
        numbers = {k: block[k] for k in (*SAMPLING, "speed") if k in block}
        values.update(numbers)
        return known or bool(numbers)

    def for_line(self, defaults: Any, ref_lang: Optional[str], lang: str, line: Optional[Mapping[str, Any]] = None) -> Tuning:
        """``line``: the line's own ``preset`` (a known name) and / or ``speed`` from the script."""
        values: Dict[str, float] = {**self.preset(self.default), "speed": 1.0}
        source = self.default
        defaults = defaults if isinstance(defaults, Mapping) else {}
        if self._apply(values, defaults):
            source = "character"
        if self.cross_lingual and ref_lang and lang != ref_lang:
            source = self.languages.get(f"{ref_lang}>{lang}", self.languages.get(CROSS, CROSS_DEFAULT))
            values.update(self.preset(source))
        own = (defaults.get("by_language") or {}).get(lang)
        if isinstance(own, Mapping) and self._apply(values, own):
            source = "character"
        if self.chosen:
            values.update(self.preset(self.chosen))
            source = self.chosen
        if self.node:
            values.update(self.node)
            source = "node"
        if line:
            if line.get("preset") in self.names:
                values.update(self.preset(line["preset"]))
            if "speed" in line:
                values["speed"] = line["speed"]
            source = "line"
        return Tuning(int(values["top_k"]), float(values["top_p"]), float(values["temperature"]),
                      float(values["repetition_penalty"]), float(values["speed"]), source)
