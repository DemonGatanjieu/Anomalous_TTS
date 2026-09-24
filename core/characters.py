"""Characters: discovery, weights, reference audio and emotions.

See docs/INTERFACE.md §2–3 for the rules; this module is their only
implementation.

- A character is a folder directly under a ``gpt_sovits`` root whose subtree
  holds at least one GPT weight (.ckpt) and one SoVITS weight (.pth). If it has
  two or more subfolders that each hold weights (e.g. 日配 / 中配), each of
  those becomes a character named ``角色/子文件夹``.
- Settings come from ``anomalous_tts.json`` (core/settings.py), then from the
  file-name convention ``原名.情绪.wav``, then from automatic choices.
- Reference text: settings → same-name ``.txt`` → GPT-SoVITS annotation files
  (``.list`` or ``.txt`` with ``path|speaker|LANG|text`` lines), matched by file
  name and then by the name without the emotion part.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import settings as settings_mod

AUTO = "自动"
MAIN = "main"
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}
MAX_DEPTH = 4
SKIP_DIRS = {"__pycache__", ".git"}
REF_MIN_SEC = 3.0
REF_MAX_SEC = 10.0
LIST_LANG_CODES = {"JA": "ja", "JP": "ja", "ZH": "zh", "EN": "en"}
_LIST_LINE = re.compile(r"^[^|]+\|[^|]*\|([A-Za-z_]+)\|(.+)$")
_GPT_EPOCH = re.compile(r"-e(\d+)", re.I)
_SOVITS_EPOCH = re.compile(r"_e(\d+)(?:_s(\d+))?", re.I)


@dataclass
class Reference:
    audio: str  # path relative to the character folder
    text: str  # "" = no reference text (GPT-SoVITS "ref free" mode)
    language: Optional[str]
    source: str  # settings | filename | auto | node

    def to_api(self) -> Dict[str, Any]:
        return {"audio": self.audio, "text": self.text, "language": self.language, "source": self.source}


def emotion_of(audio_rel: str) -> Optional[str]:
    """``X.开心.wav`` -> ``开心``; ``X.wav`` -> None (same rule as Anomalous parse_voice_name)."""
    stem = os.path.splitext(os.path.basename(audio_rel))[0]
    _, sep, emotion = stem.partition(".")
    return emotion.strip() if sep and emotion.strip() else None


# ---------- annotation (.list) files ----------
_list_cache: Dict[Tuple[str, float], Dict[str, Tuple[str, str]]] = {}


def _read_list(path: str) -> Dict[str, Tuple[str, str]]:
    """file name (lower case) -> (text, LANG). Empty if the file is not an annotation file."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return {}
    if key in _list_cache:
        return _list_cache[key]
    table: Dict[str, Tuple[str, str]] = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            lines = [line.strip() for line in f if line.strip()]
    except (OSError, UnicodeDecodeError):
        lines = []
    if lines and _LIST_LINE.match(lines[0]):
        for line in lines:
            m = _LIST_LINE.match(line)
            if m:
                audio = line.split("|", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
                table.setdefault(audio.lower(), (m.group(2).strip(), m.group(1).upper()))
    _list_cache[key] = table
    return table


def _seconds(path: str) -> float:
    import soundfile as sf

    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return 0.0


@dataclass
class Character:
    name: str
    folder: str
    gpt: List[str] = field(default_factory=list)  # relative paths, "/" separated
    sovits: List[str] = field(default_factory=list)
    audio: List[str] = field(default_factory=list)
    text_files: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    settings_error: Optional[str] = None

    def abspath(self, rel: str) -> str:
        return os.path.join(self.folder, *rel.split("/"))

    # ----- identity -----
    @property
    def aliases(self) -> List[str]:
        return [a.strip() for a in self.settings.get("aliases", []) if isinstance(a, str) and a.strip()]

    @cached_property
    def language(self) -> Optional[str]:
        """Settings, else the most common language in the annotation files."""
        lang = self.settings.get("language")
        if lang in settings_mod.LANGS:
            return lang
        counts: Counter = Counter()
        for rel in self.text_files:
            for _, code in _read_list(self.abspath(rel)).values():
                if code in LIST_LANG_CODES:
                    counts[LIST_LANG_CODES[code]] += 1
        return counts.most_common(1)[0][0] if counts else None

    # ----- weights -----
    def weight(self, kind: str, rel: Optional[str] = None) -> str:
        """Absolute path of a GPT (kind="gpt") or SoVITS ("sovits") weight."""
        files = getattr(self, kind)
        rel = rel or self.settings.get(kind)
        if rel:
            if rel not in files:
                raise ValueError(f"角色 {self.name} 里没有 {rel}。")
            return self.abspath(rel)
        if not files:
            raise ValueError(f"角色 {self.name} 没有 {'GPT (.ckpt)' if kind == 'gpt' else 'SoVITS (.pth)'} 权重。")
        pattern = _GPT_EPOCH if kind == "gpt" else _SOVITS_EPOCH

        def key(r: str):
            m = pattern.search(os.path.basename(r))
            epoch = int(m.group(1)) if m else -1
            step = int(m.group(2)) if m and m.lastindex and m.lastindex >= 2 and m.group(2) else -1
            return epoch, step, os.path.getmtime(self.abspath(r))

        return self.abspath(max(files, key=key))

    # ----- reference text -----
    def reference_text(self, audio_rel: str) -> Tuple[str, Optional[str]]:
        """(text, language or None). Empty text if nothing is found."""
        stem = os.path.splitext(audio_rel)[0]
        sidecar = stem + ".txt"
        if sidecar in self.text_files:
            with open(self.abspath(sidecar), encoding="utf-8-sig") as f:
                return f.read().strip(), None
        name, ext = os.path.splitext(os.path.basename(audio_rel))
        keys = [(name + ext).lower()]
        if emotion_of(audio_rel):
            keys.append((name.partition(".")[0] + ext).lower())
        tables = [_read_list(self.abspath(r)) for r in self.text_files if os.path.splitext(r)[0] != stem]
        for key in keys:
            for table in tables:
                if key in table:
                    text, code = table[key]
                    return text, LIST_LANG_CODES.get(code)
        return "", None

    def make_reference(self, audio_rel: str, source: str, spec: Optional[Dict[str, Any]] = None) -> Reference:
        spec = spec or {}
        text, lang = (spec["text"], None) if spec.get("text") else self.reference_text(audio_rel)
        lang = spec.get("language") or lang
        if not lang and text:
            from .langdetect import detect

            lang = detect(text, self.language)
        return Reference(audio_rel, text, lang or self.language, source)

    # ----- main reference and emotions -----
    def default_reference(self) -> Optional[Reference]:
        return self._default_reference

    def emotions(self) -> Dict[str, Reference]:
        return dict(self._emotions)

    @cached_property
    def _default_reference(self) -> Optional[Reference]:
        spec = self.settings.get("reference")
        if isinstance(spec, dict) and spec.get("audio") in self.audio:
            return self.make_reference(spec["audio"], "settings", spec)
        leaf = self.name.split("/")[0]
        ordered = sorted(
            self.audio,
            key=lambda r: (os.path.splitext(os.path.basename(r))[0] != leaf, emotion_of(r) is not None, r),
        )
        fallback = None
        for rel in ordered:
            if not REF_MIN_SEC <= _seconds(self.abspath(rel)) <= REF_MAX_SEC:
                continue
            ref = self.make_reference(rel, "auto")
            if ref.text:
                return ref
            fallback = fallback or ref
        return fallback

    @cached_property
    def _emotions(self) -> Dict[str, Reference]:
        out: Dict[str, Reference] = {}
        for rel in sorted(self.audio):
            emotion = emotion_of(rel)
            if emotion and emotion != MAIN and emotion not in out:
                out[emotion] = self.make_reference(rel, "filename")
        spec = self.settings.get("emotions")
        if isinstance(spec, dict):
            for emotion, ref in spec.items():
                if isinstance(ref, dict) and ref.get("audio") in self.audio and emotion.strip() != MAIN:
                    out[emotion.strip()] = self.make_reference(ref["audio"], "settings", ref)
        return out

    def to_api(self) -> Dict[str, Any]:
        ref = self.default_reference()
        return {
            "name": self.name,
            "aliases": self.aliases,
            "language": self.language,
            "has_settings": os.path.isfile(os.path.join(self.folder, settings_mod.FILENAME)),
            "settings": self.settings,
            "settings_error": self.settings_error,
            "gpt": self.gpt,
            "sovits": self.sovits,
            "audio": self.audio,
            "reference": ref.to_api() if ref else None,
            "emotions": {k: v.to_api() for k, v in self.emotions().items()},
        }


# ---------- discovery ----------
def _is_skipped(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".") or "pretrained" in name.lower()


def _walk(folder: str, depth: int = 0, prefix: str = ""):
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name)
    except OSError:
        return
    for entry in entries:
        rel = f"{prefix}{entry.name}"
        if entry.is_dir(follow_symlinks=True):
            if depth < MAX_DEPTH and not _is_skipped(entry.name):
                yield from _walk(entry.path, depth + 1, rel + "/")
        elif entry.is_file(follow_symlinks=True):
            yield rel


def _has_weights(folder: str) -> bool:
    gpt = sovits = False
    for rel in _walk(folder):
        low = rel.lower()
        gpt = gpt or low.endswith(".ckpt")
        sovits = sovits or low.endswith(".pth")
        if gpt and sovits:
            return True
    return False


def load_character(name: str, folder: str) -> Character:
    c = Character(name=name, folder=folder)
    for rel in _walk(folder):
        ext = os.path.splitext(rel)[1].lower()
        if ext == ".ckpt":
            c.gpt.append(rel)
        elif ext == ".pth":
            c.sovits.append(rel)
        elif ext in AUDIO_EXTS:
            c.audio.append(rel)
        elif ext in (".txt", ".list"):
            c.text_files.append(rel)
    try:
        c.settings = settings_mod.load(folder)
    except settings_mod.SettingsError as e:
        c.settings_error = str(e)
    return c


def _subdirs(path: str):
    try:
        return sorted((e for e in os.scandir(path) if e.is_dir() and not _is_skipped(e.name)), key=lambda e: e.name)
    except OSError:
        return []


def discover(roots: Iterable[str]) -> Dict[str, Character]:
    chars: Dict[str, Character] = {}
    for root in roots:
        for child in _subdirs(root):
            variants = [s for s in _subdirs(child.path) if _has_weights(s.path)]
            if len(variants) >= 2:
                found = [(f"{child.name}/{s.name}", s.path) for s in variants]
            elif _has_weights(child.path):
                found = [(child.name, child.path)]
            else:
                found = []
            for name, folder in found:
                unique, n = name, 2
                while unique in chars:
                    unique, n = f"{name} ({n})", n + 1
                chars[unique] = load_character(unique, folder)
    return chars


_cache: Dict[str, Any] = {"time": 0.0, "chars": {}}
CACHE_SECONDS = 5.0


def scan(force: bool = False) -> Dict[str, Character]:
    from . import paths

    now = time.monotonic()
    if force or now - _cache["time"] >= CACHE_SECONDS:
        _cache["chars"] = discover(paths.roots())
        _cache["time"] = now
    return _cache["chars"]


def invalidate() -> None:
    _cache["time"] = 0.0


# ---------- lookups ----------
def combo_values(chars: Dict[str, Character], kind: str) -> List[str]:
    """Node drop-down values: ``角色名/相对路径``."""
    return [f"{c.name}/{rel}" for c in chars.values() for rel in getattr(c, kind)]


def split_combo(chars: Dict[str, Character], value: str) -> Tuple[Character, str]:
    best = None
    for name in chars:
        if value.startswith(name + "/") and (best is None or len(name) > len(best)):
            best = name
    if best is None:
        raise ValueError(f"找不到文件：{value}")
    return chars[best], value[len(best) + 1 :]


def resolve_name(chars: Dict[str, Character], tag: str, language: Optional[str] = None) -> Character:
    """``[名字]`` in a script -> a character (docs/INTERFACE.md §4)."""
    tag = tag.strip()
    if tag in chars:
        return chars[tag]
    candidates = [c for c in chars.values() if tag in c.aliases]
    if not candidates:
        candidates = [c for c in chars.values() if c.name.split("/")[0] == tag]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and language:
        same = [c for c in candidates if c.language == language]
        if len(same) == 1:
            return same[0]
    if candidates:
        raise ValueError(f"[{tag}] 对应多个角色：{'、'.join(c.name for c in candidates)}。请写全名，或在设置里给其中一个加别名。")
    raise KeyError(tag)
