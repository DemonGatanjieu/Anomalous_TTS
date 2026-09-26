"""Server-side folder listing for "choose a folder" dialogs (the browser cannot see local paths)."""

from __future__ import annotations

import os
from typing import Dict, Optional

from .characters import AUDIO_EXTS, TEXT_EXTS
from .paths import norm

MAX_ENTRIES = 5000
MAX_FOLDERS = 5000  # a scan of a whole drive stops here instead of walking it for minutes
SCAN_DEPTH = 6  # a voice folder can nest: 游戏/角色/版本/参考/情绪/clip.wav
REPORT_LIMIT = 100  # skipped / too-deep folders named in a scan result
# Never a character's files, wherever they are: Python environments and base models.
ALWAYS_SKIP = {"runtime", "pretrained_models", "__pycache__", "site-packages", "venv", "node_modules"}
# Program and training folders of a GPT-SoVITS package (training checkpoints, dataset slices,
# tool models). These names are common for a user's own folders too (``output``, ``temp``), so
# they are skipped only inside a package: next to its weight folders, runtime or program files.
PACKAGE_SKIP = {"gpt_sovits", "logs", "output", "temp", "tools"}
# ``GPT_SoVITS`` itself is no sign: users name their own model folders that way too.
_PACKAGE_DIRS = {"runtime"}
_PACKAGE_FILES = {"webui.py", "api.py", "api_v2.py", "inference_webui.py", "s1_train.py", "s2_train.py"}
# The same rules are in Anomalous's tts_import_groups.js (``skipFolder``) for dropped folders.


def is_package(names) -> bool:
    """Whether a folder holding ``names`` (its sub-folders and files) is a GPT-SoVITS package or program."""
    for name in names:
        lower = name.lower()
        if lower in _PACKAGE_DIRS or lower in _PACKAGE_FILES or lower.startswith(("gpt_weights", "sovits_weights")):
            return True
    return False


def skip_folder(name: str, package: bool) -> bool:
    """Whether a scan leaves out sub-folder ``name`` of a folder (``package``: that folder is a package)."""
    lower = name.lower()
    return lower in ALWAYS_SKIP or (package and lower in PACKAGE_SKIP)


def kind_of(name: str) -> Optional[str]:
    ext = os.path.splitext(name)[1].lower()
    if ext == ".ckpt":
        return "gpt"
    if ext == ".pth":
        return "sovits"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in TEXT_EXTS:
        return "text"
    return None


def _hidden(name: str) -> bool:
    return name.startswith((".", "$")) or name == "System Volume Information"


def listing(path: Optional[str]) -> Dict:
    """Sub-folders and the files an import can use. No ``path``: the drives (or ``/``)."""
    if not path:
        drives = os.listdrives() if os.name == "nt" else ["/"]
        return {"path": "", "parent": None, "dirs": [norm(d) for d in drives], "files": [], "truncated": False}
    folder = os.path.abspath(path)
    if not os.path.isdir(folder):
        raise ValueError(f"文件夹不存在：{norm(folder)}")
    dirs, files = [], []
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name.lower())
    except PermissionError as e:
        raise ValueError(f"没有权限打开：{norm(folder)}") from e
    for entry in entries[:MAX_ENTRIES]:
        if _hidden(entry.name):
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                dirs.append(entry.name)
            elif entry.is_file(follow_symlinks=False) and kind_of(entry.name):
                files.append({"name": entry.name, "kind": kind_of(entry.name), "size": entry.stat().st_size})
        except OSError:
            continue
    parent = os.path.dirname(folder)
    return {
        "path": norm(folder),
        "parent": "" if parent == folder else norm(parent),  # "" = the drive list
        "dirs": dirs,
        "files": files,
        "truncated": len(entries) > MAX_ENTRIES,
    }


def scan(path: str) -> Dict:
    """Every usable file under ``path`` (a few levels deep), each with its folder relative to ``path``.
    ``skipped`` and ``too_deep`` name the folders it did not go into (hidden ones aside), so the
    user is told instead of files going missing quietly. ``path`` itself is always scanned."""
    root = os.path.abspath(path or "")
    if not path or not os.path.isdir(root):
        raise ValueError(f"文件夹不存在：{norm(root)}")
    files, skipped, too_deep, truncated, folders = [], [], [], False, 0
    for here, dirs, names in os.walk(root):
        if folders >= MAX_FOLDERS:
            truncated = True
            break
        folders += 1
        rel = os.path.relpath(here, root)
        rel = "" if rel == "." else rel.replace(os.sep, "/")
        depth = rel.count("/") + 1 if rel else 0
        package = is_package(dirs + names)
        kept = []
        for d in sorted(dirs, key=str.lower):
            if _hidden(d):
                continue
            sub = f"{rel}/{d}" if rel else d
            if skip_folder(d, package):
                skipped.append(sub)
            elif depth >= SCAN_DEPTH:
                too_deep.append(sub)
            else:
                kept.append(d)
        dirs[:] = kept
        for name in sorted(names, key=str.lower):
            kind = kind_of(name)
            if not kind or _hidden(name):
                continue
            if len(files) >= MAX_ENTRIES:
                truncated = True
                break
            full = os.path.join(here, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            files.append({"path": norm(full), "name": name, "dir": rel, "kind": kind, "size": size})
        if truncated:
            break
    return {"path": norm(root), "files": files, "truncated": truncated,
            "skipped": skipped[:REPORT_LIMIT], "too_deep": too_deep[:REPORT_LIMIT]}


def audio_file(path: str) -> str:
    """``path`` as an absolute path when it is an existing audio file (import preview), else ValueError."""
    full = os.path.abspath(path or "")
    if not path or not os.path.isfile(full) or kind_of(full) != "audio":
        raise ValueError(f"不是能试听的音频：{norm(full)}")
    return full
