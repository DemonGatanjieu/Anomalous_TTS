"""Server-side folder listing for "choose a folder" dialogs (the browser cannot see local paths)."""

from __future__ import annotations

import os
from typing import Dict, Optional

from .characters import AUDIO_EXTS, TEXT_EXTS
from .paths import norm

MAX_ENTRIES = 5000
SCAN_DEPTH = 4  # a GPT-SoVITS package or a voice folder: weights and clips sit a few levels down
# Folders of the GPT-SoVITS program and its training runs: base models, training checkpoints,
# dataset slices and tool models, not a character's files. ``scan`` does not go into them.
SCAN_SKIP = {"runtime", "gpt_sovits", "pretrained_models", "logs", "output", "temp", "tools",
             "__pycache__", "site-packages", "venv", "node_modules"}


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
    """Every usable file under ``path`` (a few levels deep), each with its folder relative to ``path``."""
    root = os.path.abspath(path or "")
    if not path or not os.path.isdir(root):
        raise ValueError(f"文件夹不存在：{norm(root)}")
    files, truncated = [], False
    for here, dirs, names in os.walk(root):
        rel = os.path.relpath(here, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirs[:] = sorted(d for d in dirs if not _hidden(d) and d.lower() not in SCAN_SKIP) if depth < SCAN_DEPTH else []
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
            files.append({"path": norm(full), "name": name, "dir": "" if rel == "." else rel.replace(os.sep, "/"), "kind": kind, "size": size})
        if truncated:
            break
    return {"path": norm(root), "files": files, "truncated": truncated}
