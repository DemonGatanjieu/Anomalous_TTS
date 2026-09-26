"""Build a character folder from dropped or chosen files (docs/INTERFACE.md §5.3).

Three steps: upload (or name a local path) → inspect → commit. Files are always
copied; the user's originals are never touched. Uploads are staged inside a
library (``.anomalous_tts_staging``, skipped by discovery) so commit can rename
them into place instead of copying them a second time.

Commit is all or nothing: everything is assembled in a work folder first and
moved into place at the end; on any failure placed files are removed again and
uploads go back to staging.

Adding to a character is forgiving about files it already has: an identical file
is skipped, and an annotation file (``.list`` / list-style ``.txt``) with the same
name gets the new lines appended. Only a different file under the same name is
refused. Inspect with ``target`` says which case each file is, and also looks up
lines in the annotation files the character already has.
"""

from __future__ import annotations

import filecmp
import json
import os
import re
import shutil
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import characters, paths, settings as settings_mod
from .browse import kind_of
from .checkpoints import detect_sovits_version
from .langdetect import detect

STAGING = ".anomalous_tts_staging"
STAGING_MAX_AGE = 24 * 3600
MAX_UPLOAD = 8 * 1024**3
SUBFOLDER = {"gpt": "GPT_weights", "sovits": "SoVITS_weights", "audio": "参考音频", "text": "参考音频"}
SUPPORTED_SOVITS = {"v1", "v2", "v2Pro", "v2ProPlus"}
_RESERVED = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$", re.I)
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class Conflict(ValueError):
    """Something is already there (HTTP 409). ``data`` goes into the response."""

    def __init__(self, message: str, **data: Any):
        super().__init__(message)
        self.data = data


def safe_name(name: Any, what: str = "文件名") -> str:
    """A single file or folder name that is valid on Windows too."""
    if not isinstance(name, str):
        raise ValueError(f"{what}必须是字符串")
    name = name.strip()
    if not name or name in (".", "..") or _BAD_CHARS.search(name) or _RESERVED.match(name) or name.endswith((".", " ")):
        raise ValueError(f"{what}不能用：{name!r}")
    if len(name) > 180:
        raise ValueError(f"{what}太长：{name[:40]}…")
    return name


# ---------- uploads ----------
@dataclass
class Upload:
    id: str
    library: str
    name: str
    size: int

    @property
    def dir(self) -> str:
        return os.path.join(self.library, STAGING, self.id)

    @property
    def path(self) -> str:
        return os.path.join(self.dir, self.name)

    def received(self) -> int:
        return os.path.getsize(self.path) if os.path.isfile(self.path) else 0


_uploads: Dict[str, Upload] = {}
_lock = threading.Lock()


def _writable_library(folder: Optional[str]) -> str:
    """``folder`` if it is a writable library; None = the storage place."""
    libs = [lib for lib in paths.libraries() if lib["writable"]]
    if folder is None:
        home = next((lib for lib in libs if lib["storage"]), None)
        if home is None:
            raise ValueError(f"存放位置不能写入：{paths.storage()}")
        return home["path"]
    for lib in libs:
        if paths.norm(folder).lower() == lib["path"].lower():
            return lib["path"]
    raise ValueError(f"不是可以写入的角色库：{folder}")


def _library_of(folder: str) -> str:
    """The writable library that contains ``folder`` (the innermost one)."""
    libs = [lib["path"] for lib in paths.libraries() if lib["writable"] and paths.is_inside(folder, lib["path"])]
    if not libs:
        raise ValueError("这个角色不在可以写入的角色库里")
    return max(libs, key=len)


def _clean_staging(library: str) -> None:
    root = os.path.join(library, STAGING)
    cutoff = time.time() - STAGING_MAX_AGE
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry.path, ignore_errors=True)
                with _lock:
                    _uploads.pop(entry.name, None)
        except OSError:
            pass


def start_upload(name: Any, size: Any, library: Optional[str] = None) -> str:
    name = safe_name(name)
    if not kind_of(name):
        raise ValueError(f"不支持的文件类型：{name}（只要 .ckpt、.pth、音频、.txt、.lab、.list）")
    if not isinstance(size, int) or not 0 < size <= MAX_UPLOAD:
        raise ValueError("size 必须是正整数（字节），最大 8GB")
    library = _writable_library(library)
    _clean_staging(library)
    up = Upload(uuid.uuid4().hex, library, name, size)
    os.makedirs(up.dir)
    open(up.path, "wb").close()
    with _lock:
        _uploads[up.id] = up
    return up.id


def _upload(upload_id: Any) -> Upload:
    with _lock:
        up = _uploads.get(upload_id) if isinstance(upload_id, str) else None
    if up is None or not os.path.isfile(up.path):
        raise ValueError(f"找不到这个上传（可能已过期）：{upload_id}")
    return up


def write_chunk(upload_id: str, offset: int, data: bytes) -> int:
    """Append ``data`` at ``offset``; returns the bytes received so far."""
    up = _upload(upload_id)
    received = up.received()
    if offset != received:
        raise Conflict("偏移不对", received=received)
    if received + len(data) > up.size:
        raise ValueError("收到的数据比声明的大小多")
    with open(up.path, "ab") as f:
        f.write(data)
    os.utime(up.dir)  # keeps an active upload out of the 24-hour cleanup
    return received + len(data)


def discard(upload_ids: List[Any]) -> None:
    for upload_id in upload_ids:
        with _lock:
            up = _uploads.pop(upload_id, None) if isinstance(upload_id, str) else None
        if up:
            shutil.rmtree(up.dir, ignore_errors=True)


# ---------- sources: an upload or a local path ----------
@dataclass
class Source:
    name: str  # the name inside the character (``spec["name"]`` when given)
    path: str
    kind: str
    upload: Optional[Upload] = None
    original: str = ""  # the file's own name: lines in annotation files and file names refer to it


def _renamed(source: Source, spec: Dict[str, Any]) -> Source:
    """Apply ``spec["name"]`` (two clips both called ``01.wav``, say): same kind of file, a new name."""
    source.original = source.name
    if spec.get("name") is None:
        return source
    name = safe_name(spec["name"])
    if kind_of(name) != source.kind or os.path.splitext(name)[1].lower() != os.path.splitext(source.name)[1].lower():
        raise ValueError(f"改名不能换文件类型：{source.name} → {name}")
    source.name = name
    return source


def _source(spec: Any) -> Source:
    if isinstance(spec, dict) and "upload" in spec:
        up = _upload(spec["upload"])
        if up.received() != up.size:
            raise ValueError(f"{up.name} 还没有上传完")
        return _renamed(Source(up.name, up.path, kind_of(up.name), up), spec)
    if isinstance(spec, dict) and isinstance(spec.get("path"), str):
        path = os.path.abspath(spec["path"])
        if not os.path.isfile(path):
            raise ValueError(f"文件不存在：{paths.norm(path)}")
        kind = kind_of(path)
        if not kind:
            raise ValueError(f"不支持的文件类型：{os.path.basename(path)}")
        return _renamed(Source(os.path.basename(path), path, kind), spec)
    raise ValueError('files 里的每一项必须是 {"upload": id} 或 {"path": 路径}，可以另带 "name"')


def _sources(specs: Any) -> List[Source]:
    if not isinstance(specs, list) or not specs:
        raise ValueError("files 必须是非空列表")
    return [_source(s) for s in specs]


# ---------- inspect ----------
def _is_torch_zip(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"PK"


def _stem_name(sources: List[Source]) -> str:
    for kind, pattern in (("gpt", r"-e\d+.*$"), ("sovits", r"_e\d+.*$"), ("audio", r"\..*$")):
        for s in sources:
            if s.kind == kind:
                stem = re.sub(pattern, "", os.path.splitext(s.name)[0]).strip(" _-")
                if stem:
                    return stem
    return ""


def _target(name: Any) -> "characters.Character":
    c = characters.scan(max_age=0).get(name)
    if c is None:
        raise ValueError(f"找不到角色：{name}")
    return c


def _existing(c: "characters.Character", s: Source, rel: str) -> Optional[str]:
    """How ``s`` meets the character's file at ``rel``: None (not there), same, merge or different."""
    there = c.abspath(rel)
    if not os.path.isfile(there):
        return None
    if filecmp.cmp(s.path, there, shallow=False):
        return "same"
    if s.kind == "text" and characters.read_list(s.path) and characters.read_list(there):
        return "merge"
    return "different"


def inspect(specs: Any, target: Any = None) -> Dict[str, Any]:
    sources = _sources(specs)
    c = _target(target) if target is not None else None
    by_stem = {os.path.splitext(s.name)[0]: s for s in sources
               if s.kind == "text" and os.path.splitext(s.name)[1].lower() in characters.SIDECAR_EXTS}
    annotations = [s.path for s in sources if s.kind == "text"]
    if c is not None:  # lines the character already has (added one clip at a time, say)
        annotations += [c.abspath(r) for r in c.text_files if characters.read_list(c.abspath(r))]
    files, problems = [], []
    languages: Counter = Counter()
    for i, s in enumerate(sources):
        item: Dict[str, Any] = {"ref": i, "name": s.name, "kind": s.kind, "size": os.path.getsize(s.path)}
        if c is not None:
            item["existing"] = _existing(c, s, f"{SUBFOLDER[s.kind]}/{s.name}")
            if item["existing"] == "different":
                problems.append(f"角色里已经有同名但内容不同的文件：{s.name}")
        if s.kind == "gpt" and not _is_torch_zip(s.path):
            problems.append(f"{s.name} 看起来不是 GPT-SoVITS 的 GPT 权重")
        elif s.kind == "sovits":
            try:
                item["version"] = detect_sovits_version(s.path)[1]
                item["supported"] = item["version"] in SUPPORTED_SOVITS
                if not item["supported"]:
                    problems.append(f"{s.name} 是 {item['version']} 模型，暂不支持（支持 v1、v2、v2Pro、v2ProPlus）")
            except ValueError as e:
                item["supported"] = False
                problems.append(str(e))
        elif s.kind == "audio":
            item["seconds"] = round(characters.audio_seconds(s.path), 2)
            stem = os.path.splitext(s.name)[0]
            sidecar = by_stem.get(stem)
            sidecar_path = sidecar.path if sidecar else None
            if sidecar is None and c is not None:  # a same-name .txt / .lab already in the character
                rel = next((f"{SUBFOLDER['audio']}/{stem}{ext}" for ext in characters.SIDECAR_EXTS
                            if f"{SUBFOLDER['audio']}/{stem}{ext}" in c.text_files), None)
                sidecar_path = c.abspath(rel) if rel else None
            text, lang = characters.find_text(
                s.original, sidecar_path, [p for p in annotations if p != sidecar_path])
            source = (os.path.splitext(sidecar_path)[1][1:].lower() if sidecar_path else "list") if text else None
            if not text:
                text = characters.text_from_filename(s.original)
                source = "filename" if text else None
            item["text"] = text
            item["text_source"] = source
            item["language"] = lang or (detect(text) if text else None)
            if item["language"]:
                languages[item["language"]] += 1
        files.append(item)
    usable = [f for f in files if f["kind"] == "audio" and characters.REF_MIN_SEC <= f["seconds"] <= characters.REF_MAX_SEC]
    reference = next((f["ref"] for f in usable if f["text"]), usable[0]["ref"] if usable else None)
    return {
        "files": files,
        "suggested": {
            "name": _stem_name(sources),
            "language": languages.most_common(1)[0][0] if languages else None,
            "reference": reference,
        },
        "problems": problems,
    }


# ---------- commit ----------
def _placements(sources: List[Source]) -> List[str]:
    """Relative path of each source inside the character folder."""
    rels = [f"{SUBFOLDER[s.kind]}/{s.name}" for s in sources]
    dup = [r for r, n in Counter(r.lower() for r in rels).items() if n > 1]
    if dup:
        raise ValueError(f"有重名的文件：{dup[0]}")
    return rels


def _translate(spec: Any, rels: List[str]) -> Dict[str, Any]:
    """Replace ``"file": <index>`` (and ``"gpt"`` / ``"sovits"`` indexes) with relative paths."""
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError("settings 必须是对象")
    out = json.loads(json.dumps(spec))

    def rel_of(index: Any, what: str) -> str:
        if not isinstance(index, int) or not 0 <= index < len(rels):
            raise ValueError(f"{what} 指向的文件序号不对：{index!r}")
        return rels[index]

    for key in ("gpt", "sovits"):
        if isinstance(out.get(key), int):
            out[key] = rel_of(out[key], key)
    refs = [("reference", out.get("reference"))] + [(f"emotions.{k}", v) for k, v in (out.get("emotions") or {}).items()]
    for where, ref in refs:
        if isinstance(ref, dict) and "file" in ref:
            ref["audio"] = rel_of(ref.pop("file"), where)
    return out


def _merge(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """New keys win; emotions are merged by name, everything else in ``old`` stays."""
    merged = dict(old)
    for key, value in new.items():
        if key == "emotions" and isinstance(value, dict) and isinstance(old.get("emotions"), dict):
            merged["emotions"] = {**old["emotions"], **value}
        else:
            merged[key] = value
    return merged


def _append_lines(path: str, source: str) -> int:
    """Append the lines of ``source`` that ``path`` does not have yet. Returns how many."""
    with open(path, encoding="utf-8-sig") as f:
        old = f.read()
    have = {line.strip() for line in old.splitlines() if line.strip()}
    with open(source, encoding="utf-8-sig") as f:
        new = [line.strip() for line in f if line.strip() and line.strip() not in have]
    new = list(dict.fromkeys(new))
    if new:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(("" if not old or old.endswith("\n") else "\n") + "\n".join(new) + "\n")
    return len(new)


def commit(body: Dict[str, Any]) -> Dict[str, Any]:
    """New character (``library`` + ``character``) or files added to one (``target``). Returns the character."""
    return commit_report(body)["character"]


def commit_report(body: Dict[str, Any]) -> Dict[str, Any]:
    """Like ``commit``, plus which files were ``skipped`` (identical) and ``merged`` (annotation lines)."""
    sources = _sources(body.get("files"))
    rels = _placements(sources)
    new_settings = _translate(body.get("settings"), rels)
    keep: Dict[int, str] = {}  # source index -> "same" | "merge": nothing to copy

    if body.get("target") is not None:
        c = _target(body["target"])
        folder, library = c.folder, _library_of(c.folder)
        for i, (s, rel) in enumerate(zip(sources, rels)):
            existing = _existing(c, s, rel)
            if existing == "different":
                raise Conflict(f"角色里已经有同名但内容不同的文件：{rel}")
            if existing:
                keep[i] = existing
        if c.settings_error:
            raise ValueError(f"这个角色的设置文件读不了，先修好它：{c.settings_error}")
        merged = _merge(c.settings, new_settings)
        gpt, sovits, audio = c.gpt, c.sovits, c.audio
    else:
        library = _writable_library(body.get("library"))
        name = safe_name(body.get("character"), "角色名")
        folder = os.path.join(library, name)
        if os.path.exists(folder):
            raise Conflict(f"角色库里已经有同名文件夹：{name}")
        if not any(s.kind == "gpt" for s in sources) or not any(s.kind == "sovits" for s in sources):
            raise ValueError("新角色至少要有一个 GPT 权重（.ckpt）和一个 SoVITS 权重（.pth）")
        merged = new_settings
        gpt, sovits, audio = [], [], []
    gpt, sovits, audio = list(gpt), list(sovits), list(audio)
    for i, (s, rel) in enumerate(zip(sources, rels)):
        files = {"gpt": gpt, "sovits": sovits, "audio": audio}.get(s.kind)
        if files is not None and i not in keep:
            files.append(rel)
    problems = settings_mod.validate(merged, gpt, sovits, audio)
    if problems:
        raise ValueError("；".join(problems))

    work = os.path.join(library, STAGING, f"commit-{uuid.uuid4().hex}")
    built = os.path.join(work, "character")
    uploads_now: Dict[int, str] = {}  # source index -> where that upload currently is
    placed: List[str] = []  # files and folders created inside an existing character, in order
    grown: List[tuple] = []  # (annotation file, size before) for lines appended to it
    try:
        for i, (s, rel) in enumerate(zip(sources, rels)):
            if i in keep:
                continue
            dest = os.path.join(built, *rel.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if s.upload:
                shutil.move(s.path, dest)
                uploads_now[i] = dest
            else:
                shutil.copy2(s.path, dest)
        if body.get("target") is None:
            if merged:
                settings_mod.save(built, merged)
            os.rename(built, folder)  # last step: before it, the library is untouched
        else:
            for i, rel in enumerate(rels):
                if i in keep:
                    continue
                final = os.path.join(folder, *rel.split("/"))
                if not os.path.isdir(os.path.dirname(final)):
                    os.makedirs(os.path.dirname(final))
                    placed.append(os.path.dirname(final))
                os.rename(os.path.join(built, *rel.split("/")), final)
                placed.append(final)
                if i in uploads_now:
                    uploads_now[i] = final
            for i, how in keep.items():
                if how == "merge":
                    there = os.path.join(folder, *rels[i].split("/"))
                    grown.append((there, os.path.getsize(there)))
                    _append_lines(there, sources[i].path)
            if new_settings:
                settings_mod.save(folder, merged)
    except BaseException:
        for i, where in uploads_now.items():  # uploads go back to staging so the user can retry
            if os.path.exists(where):
                os.makedirs(os.path.dirname(sources[i].path), exist_ok=True)
                shutil.move(where, sources[i].path)
        for there, size in grown:
            with open(there, "r+b") as f:
                f.truncate(size)
        for created in reversed(placed):
            if os.path.isfile(created):
                os.remove(created)
            elif os.path.isdir(created) and not os.listdir(created):
                os.rmdir(created)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    discard([s.upload.id for s in sources if s.upload])
    characters.invalidate()
    found = next((c for c in characters.scan().values() if paths.norm(c.folder).lower() == paths.norm(folder).lower()), None)
    if found is None:
        raise RuntimeError("文件已经放好，但扫描不到这个角色")
    return {
        "character": found.to_api(detail=True),
        "skipped": [rels[i] for i, how in sorted(keep.items()) if how == "same"],
        "merged": [rels[i] for i, how in sorted(keep.items()) if how == "merge"],
    }
