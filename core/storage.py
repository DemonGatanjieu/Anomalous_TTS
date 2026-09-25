"""Where characters are kept, and moving them there when that place changes.

There is one storage place (default ``models/gpt_sovits``); imports go there.
Changing it either moves the characters over (in a background thread, one
character folder at a time) or leaves them where they are, in which case the old
place is still read (core/paths.keep_library) until it holds no characters.

Each character moves on its own: a rename on the same disk, otherwise a copy into
the new place's staging folder, a rename into place, and only then deleting the
original. A failure stops the move; the character being moved stays where it was.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from typing import Any, Dict, List, Optional

from . import app_config, characters, paths
from .importer import STAGING  # skipped by discovery

log = logging.getLogger("Anomalous_TTS")

_lock = threading.Lock()
_job: Optional[Dict[str, Any]] = None


def state() -> Optional[Dict[str, Any]]:
    """The running or last move: state (moving | done | error), from, to, total, done, current,
    bytes_total / bytes_done (copies across disks only), moved, error."""
    with _lock:
        return dict(_job) if _job else None


def character_dirs(root: str) -> List[str]:
    """Top-level folders of ``root`` that hold characters (a folder with 日配 / 中配 counts once)."""
    names = set()
    for c in characters.discover([root]).values():
        names.add(os.path.relpath(c.folder, root).split(os.sep)[0])
    return sorted(names)


def _check(new: str, old: str) -> None:
    if not os.path.isdir(new):
        raise ValueError(f"文件夹不存在：{new}")
    if not os.access(new, os.W_OK):
        raise ValueError(f"这个文件夹不能写入：{new}")
    if paths.same_folder(new, old):
        raise ValueError("这里已经是存放位置了")
    if paths.is_inside(new, old) or paths.is_inside(old, new):
        raise ValueError("新位置不能在现在的位置里面，现在的位置也不能在新位置里面")


def change(folder: str, move: bool) -> None:
    """Make ``folder`` the storage place; with ``move``, move the current characters there."""
    global _job
    new, old = paths.norm(folder), paths.storage()
    with _lock:
        if _job and _job["state"] == "moving":
            raise ValueError("正在移动角色，完成后再改")
    _check(new, old)
    names = character_dirs(old)
    if names and not paths.is_default_library(old):
        paths.keep_library(old)  # still read until everything has moved
    if any(paths.same_folder(p, new) for p in app_config.load()["libraries"]):
        paths.forget_library(new)  # an earlier place becomes the storage place again
    app_config.set_storage(None if paths.is_default_library(new) else new)
    paths.register_folder(new)
    characters.invalidate()
    if not (move and names):
        return
    with _lock:
        _job = {"state": "moving", "from": old, "to": new, "total": len(names), "done": 0, "current": None,
                "bytes_total": 0, "bytes_done": 0, "moved": [], "error": None}
    threading.Thread(target=_run, args=(old, new, names), name="Anomalous_TTS move", daemon=True).start()


def _update(**fields: Any) -> None:
    with _lock:
        _job.update(fields)


def _folder_size(folder: str) -> int:
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _copy_counting(src: str, dst: str) -> None:
    shutil.copy2(src, dst)
    with _lock:
        _job["bytes_done"] += os.path.getsize(dst)


def _move_one(src: str, dst: str, same_disk: bool) -> None:
    if os.path.exists(dst):
        raise ValueError(f"新位置已经有同名文件夹：{os.path.basename(dst)}")
    if same_disk:
        os.rename(src, dst)
        return
    work = os.path.join(os.path.dirname(dst), STAGING, f"move-{uuid.uuid4().hex}")
    try:
        shutil.copytree(src, os.path.join(work, os.path.basename(dst)), copy_function=_copy_counting)
        os.rename(os.path.join(work, os.path.basename(dst)), dst)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    try:
        shutil.rmtree(src)
    except OSError as e:  # the copy is complete and in place; only the old copy is left over
        raise RuntimeError(f"已经复制到新位置，但旧文件夹没能删干净，可以手动删除：{paths.norm(src)}（{e}）") from e


def _run(old: str, new: str, names: List[str]) -> None:
    same_disk = os.stat(old).st_dev == os.stat(new).st_dev
    if not same_disk:
        _update(bytes_total=sum(_folder_size(os.path.join(old, n)) for n in names))
    try:
        for name in names:
            _update(current=name)
            _move_one(os.path.join(old, name), os.path.join(new, name), same_disk)
            with _lock:
                _job["done"] += 1
                _job["moved"].append(name)
            characters.invalidate()
    except Exception as e:  # shown on the setup card; the rest stays in the old place
        log.warning("[Anomalous_TTS] 移动角色失败：%s", e)
        _update(state="error", error=str(e), current=None)
        characters.invalidate()
        return
    if not paths.is_default_library(old) and not character_dirs(old):
        paths.forget_library(old)
    _update(state="done", current=None)
    characters.invalidate()
