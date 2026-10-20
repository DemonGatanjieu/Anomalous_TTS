"""Removing reference clips from a character: the clips and their own line files go to the
system Recycle Bin (never deleted for good), and the character's settings stop naming them.

Only clips the scan listed for that character are taken; nothing is read or moved by a path
from the request alone. Annotation files (``.list``) are left as they are: a line for a clip
that is gone is not used by anything.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Set

from . import characters, settings
from .characters import SIDECAR_EXTS, Character

MAX_AT_ONCE = 200


class TrashUnavailable(OSError):
    """There is no recycle bin for this location; the files were left in place."""


def _windows_fixed_drive(path: str) -> bool:
    import ctypes

    root = os.path.splitdrive(path)[0]
    if not root or root.startswith("\\\\"):
        return False  # a UNC share: no recycle bin
    return ctypes.windll.kernel32.GetDriveTypeW(root + "\\") == 3  # DRIVE_FIXED


def _windows_recycle(paths: List[str]) -> None:
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    if not all(_windows_fixed_drive(path) for path in paths):
        raise TrashUnavailable("这个位置没有回收站（U 盘或网络盘），为了安全没有删除，请在文件管理器里手动删除。")
    # FO_DELETE with FOF_ALLOWUNDO recycles; no dialogs. FOF_WANTNUKEWARNING makes Windows ask
    # instead of silently deleting if this drive's Recycle Bin is switched off.
    flags = 0x0040 | 0x0010 | 0x0004 | 0x0400 | 0x4000
    op = SHFILEOPSTRUCTW(None, 3, "\0".join(paths) + "\0", None, flags, False, None, None)
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if result != 0 or op.fAnyOperationsAborted:
        raise OSError(result, f"没能移到回收站（代码 {result}），文件可能正在被使用。")


def _posix_recycle(paths: List[str]) -> None:
    import shutil
    import subprocess

    try:
        from send2trash import send2trash
    except ImportError:
        send2trash = None
    if send2trash:
        for path in paths:
            send2trash(path)
        return
    gio = shutil.which("gio")
    if not gio:
        raise TrashUnavailable("这台电脑上没有回收站可用（可以安装 send2trash），没有删除。")
    for path in paths:
        subprocess.run([gio, "trash", path], check=True, capture_output=True)


def move_to_trash(paths: Iterable[str]) -> List[str]:
    """Moves the existing files among ``paths`` to the Recycle Bin, together. Returns those
    moved. Raises TrashUnavailable (nothing moved) or OSError."""
    targets = [os.path.abspath(p) for p in paths if p and os.path.lexists(p)]
    if not targets:
        return []
    if os.name == "nt":
        _windows_recycle(targets)
    else:
        _posix_recycle(targets)
    return targets


def settings_without(data: Dict[str, Any], removed: Set[str]) -> Dict[str, Any]:
    """The settings with the removed clips taken out: a reference naming one loses its audio
    (the main voice is picked automatically again), an emotion naming one goes, auxiliary
    references and fixed lines for them go. Everything else stays."""
    out = dict(data)

    def clean(spec: Any, keep_without_audio: bool):
        if not isinstance(spec, dict):
            return spec
        spec = dict(spec)
        if isinstance(spec.get("aux"), list):
            spec["aux"] = [a for a in spec["aux"] if a not in removed]
            if not spec["aux"]:
                del spec["aux"]
        if spec.get("audio") in removed:
            if not keep_without_audio:
                return None
            del spec["audio"]
            spec.pop("text", None)  # the line was the removed clip's
        return spec

    if "reference" in out:
        ref = clean(out["reference"], True)
        if isinstance(ref, dict) and not ref:
            del out["reference"]
        else:
            out["reference"] = ref
    if isinstance(out.get("emotions"), dict):
        emotions = {}
        for name, spec in out["emotions"].items():
            spec = clean(spec, False)
            if spec is not None:
                emotions[name] = spec
        if emotions:
            out["emotions"] = emotions
        else:
            del out["emotions"]
    if isinstance(out.get("lines"), dict):
        lines = {rel: text for rel, text in out["lines"].items() if rel not in removed}
        if lines:
            out["lines"] = lines
        else:
            del out["lines"]
    return out


def own_text_files(c: Character, rel: str) -> List[str]:
    """The clip's own line files (same name, ``.txt`` / ``.lab``), relative paths."""
    stem = os.path.splitext(rel)[0]
    return [stem + ext for ext in SIDECAR_EXTS if stem + ext in c.text_files]


def remove_clips(name: str, rels: Any) -> Dict[str, Any]:
    """Moves the character's clips ``rels`` and their own line files to the Recycle Bin, then
    takes them out of its settings. ValueError for anything not one of its clips."""
    chars = characters.scan(max_age=0)
    if name not in chars:
        raise ValueError(f"找不到角色：{name}")
    c = chars[name]
    if not isinstance(rels, list) or not rels or not all(isinstance(r, str) for r in rels):
        raise ValueError("paths 必须是音频路径列表")
    if len(rels) > MAX_AT_ONCE:
        raise ValueError(f"一次最多删除 {MAX_AT_ONCE} 条音频")
    known = set(c.audio)
    unknown = [r for r in rels if r not in known]
    if unknown:
        raise ValueError(f"不是这个角色的音频：{unknown[0]}")
    rels = list(dict.fromkeys(rels))
    texts = [t for rel in rels for t in own_text_files(c, rel)]
    move_to_trash([c.abspath(r) for r in rels + texts])
    data = c.settings
    if data:
        cleaned = settings_without(data, set(rels))
        if cleaned != data:
            settings.save(c.folder, cleaned)
    characters.invalidate()
    after = characters.scan(max_age=0).get(name)
    return {"ok": True, "removed": rels, "text_files": texts, "character": after.to_api(detail=True) if after else None}
