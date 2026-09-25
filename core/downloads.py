"""Pretrained downloads started from the UI, run one at a time in a background thread.

Downloads also happen on first use (core/paths.py); this only lets the user start
them ahead of time and watch the progress. Nothing is downloaded unless asked.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Dict, Iterable, Optional

from . import paths

log = logging.getLogger("Anomalous_TTS")

_jobs: Dict[str, Dict[str, str]] = {}  # id -> {"state": "queued" | "downloading" | "error", "error": ...}
_queue: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()
_worker: Optional[threading.Thread] = None


def start(ids: Iterable[str]) -> None:
    global _worker
    ids = list(ids)
    for item_id in ids:
        if item_id not in paths.PRETRAINED_IDS:
            raise ValueError(f"未知的底模：{item_id}")
    with _lock:
        for item_id in ids:
            if paths.locate(item_id) or _jobs.get(item_id, {}).get("state") in ("queued", "downloading"):
                continue
            _jobs[item_id] = {"state": "queued"}
            _queue.put(item_id)
        if _worker is None:
            _worker = threading.Thread(target=_run, name="Anomalous_TTS downloads", daemon=True)
            _worker.start()


def _run() -> None:
    while True:
        item_id = _queue.get()
        with _lock:
            _jobs[item_id] = {"state": "downloading"}
        try:
            paths.fetch(item_id)
        except Exception as e:  # network, disk; shown on the setup card
            log.warning("[Anomalous_TTS] 下载 %s 失败：%s", item_id, e)
            with _lock:
                _jobs[item_id] = {"state": "error", "error": str(e)}
        else:
            with _lock:
                _jobs.pop(item_id, None)


def _bytes_on_disk(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for folder, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(folder, f))
            except OSError:
                pass
    return total


def state(item_id: str) -> Optional[Dict]:
    """The running or failed job for ``item_id``, with ``done`` bytes while downloading."""
    with _lock:
        job = dict(_jobs[item_id]) if item_id in _jobs else None
    if job and job["state"] == "downloading":
        job["done"] = sum(_bytes_on_disk(p) for p in paths.download_paths(item_id))
    return job
