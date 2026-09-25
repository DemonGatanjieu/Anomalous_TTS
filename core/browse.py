"""Server-side folder listing for "choose a folder" dialogs (the browser cannot see local paths)."""

from __future__ import annotations

import os
from typing import Dict, Optional

from .characters import AUDIO_EXTS, TEXT_EXTS
from .paths import norm

MAX_ENTRIES = 5000


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
