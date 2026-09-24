"""Find characters, their weights and their reference audio.

A character is a folder directly under a ``gpt_sovits`` root whose subtree holds
at least one GPT weight (.ckpt) and one SoVITS weight (.pth). If that folder has
two or more subfolders that each hold weights (for example ``日配`` and
``中配``), each subfolder becomes its own character, named ``角色/子文件夹``.

Emotion references follow the F5-TTS / Anomalous naming: ``名字.<情绪>.wav``
(the emotion is everything after the first dot of the file name). A script
line ``{开心}...`` uses the first such file whose emotion is ``开心``.

Reference text is looked up, in order, from:
1. a ``.txt`` next to the audio with the same name (F5-TTS convention);
2. a GPT-SoVITS annotation file (``.list``, or ``.txt`` in the same
   ``path|speaker|LANG|text`` format) anywhere in the character folder,
   first by the file name, then by the name without the emotion part (so a
   clip renamed to ``X.开心.wav`` still finds the line for ``X.wav``).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import paths

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}
MAX_DEPTH = 4
SKIP_DIRS = {"__pycache__", ".git"}
_LIST_LINE = re.compile(r"^[^|]+\|[^|]*\|([A-Za-z_]+)\|(.+)$")


@dataclass
class Character:
    name: str
    folder: str
    gpt: List[str] = field(default_factory=list)  # paths relative to folder, "/" separated
    sovits: List[str] = field(default_factory=list)
    audio: List[str] = field(default_factory=list)
    text_files: List[str] = field(default_factory=list)

    def abspath(self, rel: str) -> str:
        return os.path.join(self.folder, *rel.split("/"))


def _is_skipped(name: str) -> bool:
    low = name.lower()
    return name in SKIP_DIRS or name.startswith(".") or "pretrained" in low


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


def _collect(name: str, folder: str) -> Character:
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
    return c


def _has_weights(folder: str) -> bool:
    gpt = sovits = False
    for rel in _walk(folder):
        low = rel.lower()
        gpt = gpt or low.endswith(".ckpt")
        sovits = sovits or low.endswith(".pth")
        if gpt and sovits:
            return True
    return False


_cache: Dict[str, object] = {"time": 0.0, "chars": {}}
CACHE_SECONDS = 5.0


def scan(force: bool = False) -> Dict[str, Character]:
    now = time.monotonic()
    if not force and now - _cache["time"] < CACHE_SECONDS:
        return _cache["chars"]  # type: ignore[return-value]
    chars: Dict[str, Character] = {}
    for root in paths.roots():
        try:
            children = sorted(
                (e for e in os.scandir(root) if e.is_dir() and not _is_skipped(e.name)),
                key=lambda e: e.name,
            )
        except OSError:
            continue
        for child in children:
            try:
                subs = sorted(
                    (e for e in os.scandir(child.path) if e.is_dir() and not _is_skipped(e.name)),
                    key=lambda e: e.name,
                )
            except OSError:
                continue
            variants = [s for s in subs if _has_weights(s.path)]
            if len(variants) >= 2:
                found = [(f"{child.name}/{s.name}", s.path) for s in variants]
            elif _has_weights(child.path):
                found = [(child.name, child.path)]
            else:
                found = []
            for name, folder in found:
                unique = name
                n = 2
                while unique in chars:
                    unique = f"{name} ({n})"
                    n += 1
                chars[unique] = _collect(unique, folder)
    _cache["time"] = now
    _cache["chars"] = chars
    return chars


# ---------- combo values: "角色/相对路径" ----------
def combo_values(kind: str) -> List[str]:
    out = []
    for c in scan().values():
        for rel in getattr(c, kind):
            out.append(f"{c.name}/{rel}")
    return out


def split_combo(value: str) -> Tuple[Character, str]:
    chars = scan()
    best = None
    for name in chars:
        if value.startswith(name + "/") and (best is None or len(name) > len(best)):
            best = name
    if best is None:
        raise ValueError(f"找不到文件：{value}")
    return chars[best], value[len(best) + 1 :]


# ---------- weights ----------
_GPT_EPOCH = re.compile(r"-e(\d+)", re.I)
_SOVITS_EPOCH = re.compile(r"_e(\d+)(?:_s(\d+))?", re.I)


def _epoch_key(c: Character, rel: str, pattern) -> Tuple[int, int, float]:
    m = pattern.search(os.path.basename(rel))
    epoch = int(m.group(1)) if m else -1
    step = int(m.group(2)) if m and m.lastindex and m.lastindex >= 2 and m.group(2) else -1
    return epoch, step, os.path.getmtime(c.abspath(rel))


def pick_weight(c: Character, kind: str, choice: str) -> str:
    """Return an absolute path. ``choice`` is AUTO or a combo value."""
    files = getattr(c, kind)
    if choice and choice != AUTO:
        owner, rel = split_combo(choice)
        if owner.name != c.name:
            raise ValueError(f"{choice} 不属于角色 {c.name}。")
        return owner.abspath(rel)
    if not files:
        raise ValueError(f"角色 {c.name} 没有 {'GPT (.ckpt)' if kind == 'gpt' else 'SoVITS (.pth)'} 权重。")
    pattern = _GPT_EPOCH if kind == "gpt" else _SOVITS_EPOCH
    return c.abspath(max(files, key=lambda rel: _epoch_key(c, rel, pattern)))


AUTO = "自动"


# ---------- reference text ----------
_list_cache: Dict[Tuple[str, float], Dict[str, Tuple[str, str]]] = {}


def _read_list(path: str) -> Dict[str, Tuple[str, str]]:
    key = (path, os.path.getmtime(path))
    if key in _list_cache:
        return _list_cache[key]
    table: Dict[str, Tuple[str, str]] = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            lines = [l.strip() for l in f if l.strip()]
    except (OSError, UnicodeDecodeError):
        lines = []
    if lines and _LIST_LINE.match(lines[0]):
        for line in lines:
            m = _LIST_LINE.match(line)
            if not m:
                continue
            audio = line.split("|", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
            table.setdefault(audio.lower(), (m.group(2).strip(), m.group(1).upper()))
    _list_cache[key] = table
    return table


MAIN = "main"


def emotion_of(audio_rel: str) -> Optional[str]:
    """``X.开心.wav`` -> ``开心``; ``X.wav`` -> None (same rule as Anomalous parse_voice_name)."""
    stem = os.path.splitext(os.path.basename(audio_rel))[0]
    _, sep, emotion = stem.partition(".")
    return emotion.strip() if sep and emotion.strip() else None


def emotions(c: Character) -> Dict[str, str]:
    """Emotion -> audio path (relative). The first file in name order wins."""
    out: Dict[str, str] = {}
    for rel in sorted(c.audio):
        emotion = emotion_of(rel)
        if emotion and emotion != MAIN:
            out.setdefault(emotion, rel)
    return out


def reference_text(c: Character, audio_rel: str) -> Tuple[str, Optional[str]]:
    """Return (text, LIST language code or None). Empty text if unknown."""
    stem = os.path.splitext(audio_rel)[0]
    sidecar = stem + ".txt"
    if sidecar in c.text_files:
        with open(c.abspath(sidecar), encoding="utf-8-sig") as f:
            return f.read().strip(), None
    name, ext = os.path.splitext(os.path.basename(audio_rel))
    keys = [(name + ext).lower()]
    if emotion_of(audio_rel):
        keys.append((name.partition(".")[0] + ext).lower())
    tables = [
        _read_list(c.abspath(rel)) for rel in c.text_files if os.path.splitext(rel)[0] != stem
    ]
    for key in keys:
        for table in tables:
            if key in table:
                return table[key]
    return "", None


def default_reference(c: Character) -> Tuple[str, str, Optional[str]]:
    """Pick a reference: prefer one named like the F5 main voice, then any 3-10 s clip with text."""
    import soundfile as sf

    def seconds(rel: str) -> float:
        try:
            info = sf.info(c.abspath(rel))
            return info.frames / info.samplerate
        except Exception:
            return 0.0

    leaf = c.name.split("/")[0]
    ordered = sorted(
        c.audio,
        key=lambda r: (os.path.splitext(os.path.basename(r))[0] != leaf, emotion_of(r) is not None, r),
    )
    fallback = None
    for rel in ordered:
        if not 3.0 <= seconds(rel) <= 10.0:
            continue
        text, lang = reference_text(c, rel)
        if text:
            return rel, text, lang
        fallback = fallback or rel
    if fallback:
        return fallback, "", None
    raise ValueError(f"角色 {c.name} 里没有 3~10 秒的参考音频。")
