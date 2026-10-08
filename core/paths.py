"""Model folders and pretrained files.

Character libraries are the roots of ComfyUI's ``gpt_sovits`` model category:
``ComfyUI/models/gpt_sovits``, folders from ``extra_model_paths.yaml``::

    my_voices:
        base_path: D:/voices
        gpt_sovits: characters

and folders the user wrote in ``ComfyUI/user/anomalous_tts.json``
(core/app_config.py, searched without a restart): the storage place, where
imported characters go (default ``models/gpt_sovits``), and more folders with
characters. No request changes them.

Import folders (also from that file) are the only places the import routes list
and read files from (``import_root``).

Pretrained files are looked up under every library (``<root>/pretrained/<name>``
or ``<root>/<name>``) and in GPT-SoVITS packages added as pretrained sources
(``GPT_SoVITS``, its ``pretrained_models`` and ``text``). Missing ones are
downloaded into ``models/gpt_sovits/pretrained`` on first use or from the UI.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import folder_paths

from . import app_config

log = logging.getLogger("Anomalous_TTS")

CATEGORY = "gpt_sovits"

HF_REPO = "lj1995/GPT-SoVITS"
HUBERT_NAME = "chinese-hubert-base"
HUBERT_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
ROBERTA_NAME = "chinese-roberta-wwm-ext-large"
ROBERTA_FILES = ("config.json", "tokenizer.json", "pytorch_model.bin")
G2PW_NAME = "G2PWModel"
G2PW_FILES = ("g2pW.onnx", "config.py", "POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt")
# The file GPT-SoVITS downloads (text/g2pw/onnx_api.py), pinned to one revision of it and checked
# against its SHA-256 before it is extracted.
G2PW_REVISION = "827f4a519b083e3f37c790c938300241e75692d5"
G2PW_URL = f"https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/{G2PW_REVISION}/G2PWModel_1.1.zip"
G2PW_SHA256 = "b116f6930a7ee55eef6576a8d8e14bf40c1106583439e8ae924b901512379c64"
SV_FILE = "pretrained_eres2netv2w24s4ep4.ckpt"

# Pinned to the GPT-SoVITS commit our vendored code comes from (see UPSTREAM.md).
GSV_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
JA_USERDICT_URL = (
    f"https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/{GSV_COMMIT}/GPT_SoVITS/text/ja_userdic/userdict.csv"
)
EN_DICT_FILES = ("cmudict.rep", "cmudict-fast.rep", "engdict-hot.rep")  # names: vendor text/namedict.rep
EN_DICT_URL = f"https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/{GSV_COMMIT}/GPT_SoVITS/text/{{name}}"
# nltk_data packages g2p_en / GPT-SoVITS english.py need.
NLTK_PACKAGES = (
    "taggers/averaged_perceptron_tagger_eng",
    "taggers/averaged_perceptron_tagger",
    "corpora/cmudict",
)
NLTK_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/{name}.zip"

_lock = threading.Lock()
_sources: List[str] = []  # pretrained sources from app_config, loaded by register()


def norm(path: str) -> str:
    """Absolute path with ``/`` separators, as shown in the UI and stored in app_config."""
    return os.path.abspath(path).replace("\\", "/")


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def is_inside(path: str, root: str) -> bool:
    path, root = _key(path), _key(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


# ---------- character libraries ----------
def _default_library() -> str:
    return os.path.join(folder_paths.models_dir, CATEGORY)


def register() -> None:
    folder_paths.add_model_folder_path(CATEGORY, _default_library(), is_default=True)
    config = app_config.load()
    for folder in ([config["storage"]] if config["storage"] else []) + config["libraries"]:
        if os.path.isdir(folder):
            folder_paths.add_model_folder_path(CATEGORY, folder)
        else:
            log.warning("[Anomalous_TTS] 找不到角色库 %s，已跳过（可以在 %s 里删掉）。", folder, app_config.path())
    _sources[:] = config["pretrained"]


def _register_configured() -> None:
    """Folders added to the settings file, or missing at startup (unplugged disk), are searched
    once they are there."""
    config = app_config.load()
    for folder in ([config["storage"]] if config["storage"] else []) + config["libraries"]:
        if os.path.isdir(folder):
            register_folder(folder)


def roots() -> List[str]:
    _register_configured()
    try:
        paths = folder_paths.get_folder_paths(CATEGORY)
    except KeyError:
        paths = []
    return [p for p in paths if os.path.isdir(p)]


def default_root() -> str:
    path = folder_paths.get_folder_paths(CATEGORY)[0]
    os.makedirs(path, exist_ok=True)
    return path


def storage() -> str:
    """Where characters are kept and imported to: the configured place (even while it is missing,
    so nothing is written elsewhere behind the user's back), else ``models/gpt_sovits``."""
    folder = app_config.load()["storage"]
    return norm(folder) if folder else norm(_default_library())


def libraries() -> List[Dict]:
    """Every folder searched for characters: where it comes from, and which one is the storage place
    (listed even while it is missing, then not writable)."""
    _register_configured()
    config = app_config.load()
    added = {_key(p) for p in config["libraries"]}
    default, current = _key(_default_library()), _key(storage())
    custom = _key(config["storage"]) if config["storage"] else None
    out = []
    for p in folder_paths.get_folder_paths(CATEGORY):
        k = _key(p)
        source = "default" if k == default else "storage" if k == custom else "app" if k in added else "yaml"
        exists = os.path.isdir(p)
        out.append({"path": norm(p), "source": source, "storage": k == current, "exists": exists,
                    "writable": exists and os.access(p, os.W_OK)})
    if not any(lib["storage"] for lib in out):
        out.append({"path": storage(), "source": "storage", "storage": True, "exists": False, "writable": False})
    return out


def same_folder(a: str, b: str) -> bool:
    return _key(a) == _key(b)


def is_default_library(folder: str) -> bool:
    return same_folder(folder, _default_library())


def register_folder(folder: str) -> None:
    """Search ``folder`` for characters from now on (no restart)."""
    if not any(same_folder(p, folder) for p in folder_paths.get_folder_paths(CATEGORY)):
        folder_paths.add_model_folder_path(CATEGORY, norm(folder))


# ---------- import folders ----------
def import_folders() -> List[str]:
    """The folders imports may read from, as the user listed them (existing ones only)."""
    return [norm(p) for p in app_config.load()["import_folders"] if os.path.isdir(p)]


def import_root(path: str) -> Optional[str]:
    """The import folder that holds ``path``, or None. Links are followed first, so a link inside
    an import folder that points elsewhere does not count as inside."""
    real = os.path.realpath(path)
    for root in import_folders():
        if is_inside(real, os.path.realpath(root)):
            return root
    return None


def require_import_path(path: str) -> str:
    """``path`` as an absolute path when it is inside an import folder, else ValueError."""
    if not path or import_root(path) is None:
        raise ValueError(f"不在导入文件夹里：{norm(path or '')}。可以导入的文件夹写在 {app_config.path()} 的 import_folders 里")
    return os.path.abspath(path)


# ---------- pretrained lookup ----------
def _source_dirs(source: str) -> List[str]:
    return [source, os.path.join(source, "pretrained_models"), os.path.join(source, "text")]


def _search_dirs() -> List[str]:
    dirs = []
    for root in folder_paths.get_folder_paths(CATEGORY):
        dirs += [os.path.join(root, "pretrained"), root]  # a root may *be* a pretrained_models folder
    for source in _sources:
        dirs += _source_dirs(source)
    return dirs


def _has(folder: str, files: Tuple[str, ...]) -> bool:
    return all(os.path.isfile(os.path.join(folder, f)) for f in files)


def find_pretrained(name: str, required: Tuple[str, ...], dirs: Optional[List[str]] = None) -> Optional[str]:
    for d in _search_dirs() if dirs is None else dirs:
        path = os.path.join(d, name)
        if _has(path, required):
            return path
    return None


def _find_en_dict(dirs: Optional[List[str]] = None) -> Optional[str]:
    for d in _search_dirs() if dirs is None else dirs:
        for cand in (os.path.join(d, "en_dict"), d):  # a GPT-SoVITS ``text`` folder holds them directly
            if _has(cand, EN_DICT_FILES):
                return cand
    return None


def _download_target() -> str:
    return os.path.join(default_root(), "pretrained")


def _nltk_missing() -> List[str]:
    nltk_dir = os.path.join(_download_target(), "nltk_data")
    return [
        n for n in NLTK_PACKAGES
        if not (os.path.isdir(os.path.join(nltk_dir, *n.split("/"))) or os.path.isfile(os.path.join(nltk_dir, *n.split("/")) + ".zip"))
    ]


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def _hf_pretrained(name: str, files: tuple) -> str:
    """Return a pretrained folder from lj1995/GPT-SoVITS, downloading it on first use."""
    with _lock:
        found = find_pretrained(name, files)
        if found:
            return found
        target = _download_target()
        log.info("[Anomalous_TTS] 下载 %s 到 %s ...", name, target)
        try:
            from huggingface_hub import hf_hub_download

            for f in files:
                hf_hub_download(HF_REPO, f"{name}/{f}", local_dir=target)
        except Exception as e:  # network, proxy, missing package
            raise RuntimeError(
                f"无法下载 {name}（{e}）。请手动从 https://huggingface.co/{HF_REPO}/tree/main/{name} "
                f"下载 {', '.join(files)}，放到 {os.path.join(target, name)}。"
                "也可以在 Anomalous 的音频页里指定已有的 GPT-SoVITS 整合包，"
                "或在 extra_model_paths.yaml 里把整合包的 GPT_SoVITS/pretrained_models 加到 gpt_sovits。"
            ) from e
        return os.path.join(target, name)


def hubert_dir() -> str:
    return _hf_pretrained(HUBERT_NAME, HUBERT_FILES)


def roberta_dir() -> str:
    return _hf_pretrained(ROBERTA_NAME, ROBERTA_FILES)


def sv_path() -> str:
    """Speaker encoder for v2Pro / v2ProPlus (~100MB), downloaded on first use."""
    return os.path.join(_hf_pretrained("sv", (SV_FILE,)), SV_FILE)


def english_dirs():
    """English G2P data: (dictionary dir, nltk data dir). Downloads on first use.

    nltk_data is fetched directly instead of with nltk.download(): recent NLTK refuses
    downloads through an HTTP proxy, which is common on users' machines.
    """
    with _lock:
        pretrained = _download_target()
        dict_dir = _find_en_dict()
        if dict_dir is None:
            dict_dir = os.path.join(pretrained, "en_dict")
            log.info("[Anomalous_TTS] 下载英语词典到 %s ...", dict_dir)
            for name in EN_DICT_FILES:
                if not os.path.isfile(os.path.join(dict_dir, name)):
                    _download(EN_DICT_URL.format(name=name), os.path.join(dict_dir, name))

        nltk_dir = os.path.join(pretrained, "nltk_data")
        for name in _nltk_missing():
            target = os.path.join(nltk_dir, *name.split("/"))
            log.info("[Anomalous_TTS] 下载 nltk 数据 %s ...", name)
            _download(NLTK_URL.format(name=name), target + ".zip")
            with zipfile.ZipFile(target + ".zip") as zf:
                zf.extractall(os.path.dirname(target))
        return dict_dir, nltk_dir


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_g2pw() -> None:
    """Called with ``_lock`` held. Leaves nothing half-written on failure, and extracts nothing
    whose SHA-256 is not the pinned one."""
    target = _download_target()
    os.makedirs(target, exist_ok=True)
    zip_path = os.path.join(target, "G2PWModel_1.1.zip")
    log.info("[Anomalous_TTS] 下载中文多音字模型 G2PWModel 到 %s ...", target)
    try:
        _download(G2PW_URL, zip_path)
        digest = _sha256(zip_path)
        if digest != G2PW_SHA256:
            raise RuntimeError(f"G2PWModel_1.1.zip is not the expected file (SHA-256 {digest}); nothing was extracted")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        extracted = os.path.join(target, "G2PWModel_1.1")
        final = os.path.join(target, G2PW_NAME)
        if os.path.isdir(extracted) and not os.path.exists(final):
            shutil.move(extracted, final)
    finally:
        for leftover in (zip_path, zip_path + ".part"):
            if os.path.exists(leftover):
                os.remove(leftover)


_g2pw_state = {"tried": False}


def g2pw_dir() -> Optional[str]:
    """Chinese polyphone model (~600MB). Returns None if unavailable; callers fall back to pypinyin."""
    with _lock:
        found = find_pretrained(G2PW_NAME, G2PW_FILES)
        if found or _g2pw_state["tried"]:
            return found
        _g2pw_state["tried"] = True
        try:
            _download_g2pw()
        except Exception as e:
            log.warning("[Anomalous_TTS] G2PWModel 下载失败：%s。可手动下载 %s 解压到 %s", e, G2PW_URL, _download_target())
        return find_pretrained(G2PW_NAME, G2PW_FILES)


def _ja_userdic_dir(download: bool) -> Optional[str]:
    """Called with ``_lock`` held."""
    folder = find_pretrained("ja_userdic", ("userdict.csv",))
    if folder or not download:
        return folder
    folder = os.path.join(_download_target(), "ja_userdic")
    _download(JA_USERDICT_URL, os.path.join(folder, "userdict.csv"))
    return folder


class ComfyResources:
    """engine.Resources backed by ComfyUI model folders."""

    hubert_dir = staticmethod(hubert_dir)
    roberta_dir = staticmethod(roberta_dir)
    g2pw_dir = staticmethod(g2pw_dir)
    sv_path = staticmethod(sv_path)
    english_dirs = staticmethod(english_dirs)


_ja_userdict_state = {"done": False}


def ensure_ja_userdict() -> None:
    """Load GPT-SoVITS' Japanese user dictionary (English words -> katakana). Best effort."""
    if _ja_userdict_state["done"]:
        return
    with _lock:
        if _ja_userdict_state["done"]:
            return
        _ja_userdict_state["done"] = True
        try:
            import pyopenjtalk

            folder = _ja_userdic_dir(download=True)
            csv_path = os.path.join(folder, "userdict.csv")
            dic_path = os.path.join(folder, "user.dict")
            if not os.path.isfile(dic_path) or os.path.getmtime(dic_path) < os.path.getmtime(csv_path):
                pyopenjtalk.mecab_dict_index(csv_path, dic_path)
            pyopenjtalk.update_global_jtalk_with_user_dict(dic_path)
        except Exception as e:
            log.warning("[Anomalous_TTS] 日语用户词典不可用，英文单词会按字母读：%s", e)


# ---------- pretrained status and explicit downloads (the UI's setup card) ----------
@dataclass(frozen=True)
class Pretrained:
    id: str
    label: str
    needed_for: str  # "all", "zh", "en", "ja" or "v2pro"
    size: int  # approximate download size in bytes, for the progress bar
    required: bool  # False: the node works without it, with lower quality


PRETRAINED = (
    Pretrained("hubert", HUBERT_NAME, "all", 190_000_000, True),
    Pretrained("roberta", ROBERTA_NAME, "zh", 650_000_000, True),
    Pretrained("g2pw", G2PW_NAME, "zh", 610_000_000, False),
    Pretrained("sv", "sv (ERes2NetV2)", "v2pro", 105_000_000, True),
    Pretrained("english", "英语词典 + nltk 数据", "en", 30_000_000, True),
    Pretrained("ja_userdic", "日语用户词典", "ja", 17_000_000, False),
)
PRETRAINED_IDS = tuple(p.id for p in PRETRAINED)


def locate(item_id: str, dirs: Optional[List[str]] = None) -> Optional[str]:
    """Where a pretrained item is, or None. ``dirs`` limits the search (used to check a new source)."""
    if item_id == "hubert":
        return find_pretrained(HUBERT_NAME, HUBERT_FILES, dirs)
    if item_id == "roberta":
        return find_pretrained(ROBERTA_NAME, ROBERTA_FILES, dirs)
    if item_id == "g2pw":
        return find_pretrained(G2PW_NAME, G2PW_FILES, dirs)
    if item_id == "sv":
        return find_pretrained("sv", (SV_FILE,), dirs)
    if item_id == "english":
        found = _find_en_dict(dirs)
        return found if dirs is not None or not _nltk_missing() else None
    if item_id == "ja_userdic":
        return find_pretrained("ja_userdic", ("userdict.csv",), dirs)
    raise ValueError(f"未知的底模：{item_id}")


def fetch(item_id: str) -> None:
    """Download one item now (blocking). Raises with a readable message on failure."""
    if item_id == "hubert":
        hubert_dir()
    elif item_id == "roberta":
        roberta_dir()
    elif item_id == "sv":
        sv_path()
    elif item_id == "english":
        english_dirs()
    elif item_id == "g2pw":
        with _lock:
            if not find_pretrained(G2PW_NAME, G2PW_FILES):
                _download_g2pw()
    elif item_id == "ja_userdic":
        with _lock:
            _ja_userdic_dir(download=True)
    else:
        raise ValueError(f"未知的底模：{item_id}")


def download_paths(item_id: str) -> List[str]:
    """Files and folders a download of ``item_id`` writes to, for measuring progress."""
    target = _download_target()
    hf_partial = os.path.join(target, ".cache", "huggingface", "download")
    names = {"hubert": HUBERT_NAME, "roberta": ROBERTA_NAME, "sv": "sv"}
    if item_id in names:
        return [os.path.join(target, names[item_id]), os.path.join(hf_partial, names[item_id])]
    if item_id == "g2pw":
        return [os.path.join(target, "G2PWModel_1.1.zip.part"), os.path.join(target, "G2PWModel_1.1"), os.path.join(target, G2PW_NAME)]
    if item_id == "english":
        return [os.path.join(target, "en_dict"), os.path.join(target, "nltk_data")]
    return [os.path.join(target, "ja_userdic")]


def pretrained_status() -> List[Dict]:
    out = []
    for p in PRETRAINED:
        found = locate(p.id)
        item = {"id": p.id, "label": p.label, "needed_for": p.needed_for, "size": p.size,
                "required": p.required, "state": "ok" if found else "missing"}
        if found:
            item["path"] = norm(found)
        out.append(item)
    return out


def pretrained_sources() -> List[str]:
    return list(_sources)


def add_pretrained_source(folder: str) -> None:
    """A GPT-SoVITS package folder (``GPT_SoVITS``, or its ``pretrained_models``)."""
    folder = norm(folder)
    if not os.path.isdir(folder):
        raise ValueError(f"文件夹不存在：{folder}")
    if any(_key(s) == _key(folder) for s in _sources):
        raise ValueError(f"已经添加过了：{folder}")
    dirs = _source_dirs(folder)
    if not any(locate(i, dirs) for i in PRETRAINED_IDS):
        raise ValueError(
            f"{folder} 里没有找到底模。请选择 GPT-SoVITS 整合包里的 GPT_SoVITS 文件夹"
            "（里面有 pretrained_models 和 text）。"
        )
    app_config.add("pretrained", folder)
    _sources.append(folder)


def remove_pretrained_source(folder: str) -> None:
    stored = next((s for s in _sources if _key(s) == _key(folder)), None)
    if stored is None:
        raise ValueError(f"没有添加过这个底模来源：{folder}")
    app_config.remove("pretrained", stored)
    _sources.remove(stored)
