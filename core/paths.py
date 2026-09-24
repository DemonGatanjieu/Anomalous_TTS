"""Model folders and pretrained files.

All paths come from ComfyUI's ``folder_paths``. The ``gpt_sovits`` category
defaults to ``ComfyUI/models/gpt_sovits`` and can be extended in
``extra_model_paths.yaml``::

    my_voices:
        base_path: D:/voices
        gpt_sovits: characters
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from typing import List, Optional

import folder_paths

log = logging.getLogger("Anomalous_TTS")

CATEGORY = "gpt_sovits"
PRETRAINED_DIRNAMES = ("pretrained", "pretrained_models")

HF_REPO = "lj1995/GPT-SoVITS"
HUBERT_NAME = "chinese-hubert-base"
HUBERT_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
ROBERTA_NAME = "chinese-roberta-wwm-ext-large"
ROBERTA_FILES = ("config.json", "tokenizer.json", "pytorch_model.bin")
G2PW_NAME = "G2PWModel"
G2PW_FILES = ("g2pW.onnx", "config.py", "POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt")
# Same source GPT-SoVITS downloads from (text/g2pw/onnx_api.py).
G2PW_URL = "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"

# Pinned to the GPT-SoVITS commit our vendored code comes from (see UPSTREAM.md).
GSV_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
JA_USERDICT_URL = (
    f"https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/{GSV_COMMIT}/GPT_SoVITS/text/ja_userdic/userdict.csv"
)

_lock = threading.Lock()


def register() -> None:
    default = os.path.join(folder_paths.models_dir, CATEGORY)
    folder_paths.add_model_folder_path(CATEGORY, default, is_default=True)


def roots() -> List[str]:
    try:
        paths = folder_paths.get_folder_paths(CATEGORY)
    except KeyError:
        paths = []
    return [p for p in paths if os.path.isdir(p)]


def default_root() -> str:
    path = folder_paths.get_folder_paths(CATEGORY)[0]
    os.makedirs(path, exist_ok=True)
    return path


def _candidates(name: str) -> List[str]:
    out = []
    for root in folder_paths.get_folder_paths(CATEGORY):
        out.append(os.path.join(root, "pretrained", name))
        out.append(os.path.join(root, name))  # a root that *is* a GPT-SoVITS pretrained_models folder
    return out


def find_pretrained(name: str, required: tuple) -> Optional[str]:
    for path in _candidates(name):
        if all(os.path.isfile(os.path.join(path, f)) for f in required):
            return path
    return None


def _hf_pretrained(name: str, files: tuple) -> str:
    """Return a pretrained folder from lj1995/GPT-SoVITS, downloading it on first use."""
    with _lock:
        found = find_pretrained(name, files)
        if found:
            return found
        target = os.path.join(default_root(), "pretrained")
        log.info("[Anomalous_TTS] 下载 %s 到 %s ...", name, target)
        try:
            from huggingface_hub import hf_hub_download

            for f in files:
                hf_hub_download(HF_REPO, f"{name}/{f}", local_dir=target)
        except Exception as e:  # network, proxy, missing package
            raise RuntimeError(
                f"无法下载 {name}（{e}）。请手动从 https://huggingface.co/{HF_REPO}/tree/main/{name} "
                f"下载 {', '.join(files)}，放到 {os.path.join(target, name)}。"
                "也可以在 extra_model_paths.yaml 里把 GPT-SoVITS 整合包的 GPT_SoVITS/pretrained_models 加到 gpt_sovits。"
            ) from e
        return os.path.join(target, name)


def hubert_dir() -> str:
    return _hf_pretrained(HUBERT_NAME, HUBERT_FILES)


def roberta_dir() -> str:
    return _hf_pretrained(ROBERTA_NAME, ROBERTA_FILES)


SV_FILE = "pretrained_eres2netv2w24s4ep4.ckpt"


def sv_path() -> str:
    """Speaker encoder for v2Pro / v2ProPlus (~100MB), downloaded on first use."""
    return os.path.join(_hf_pretrained("sv", (SV_FILE,)), SV_FILE)


EN_DICT_FILES = ("cmudict.rep", "cmudict-fast.rep", "engdict-hot.rep", "namedict_cache.pickle")
EN_DICT_URL = f"https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/{GSV_COMMIT}/GPT_SoVITS/text/{{name}}"
# nltk_data packages g2p_en / GPT-SoVITS english.py need.
NLTK_PACKAGES = (
    "taggers/averaged_perceptron_tagger_eng",
    "taggers/averaged_perceptron_tagger",
    "corpora/cmudict",
)
NLTK_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/{name}.zip"


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def english_dirs():
    """English G2P data: (dictionary dir, writable cache dir, nltk data dir). Downloads on first use.

    nltk_data is fetched directly instead of with nltk.download(): recent NLTK refuses
    downloads through an HTTP proxy, which is common on users' machines.
    """
    with _lock:
        pretrained = os.path.join(default_root(), "pretrained")
        dict_dir = None
        for root in folder_paths.get_folder_paths(CATEGORY):
            for cand in (os.path.join(root, "pretrained", "en_dict"), os.path.join(root, "en_dict"), root):
                if all(os.path.isfile(os.path.join(cand, f)) for f in EN_DICT_FILES):
                    dict_dir = cand
                    break
            if dict_dir:
                break
        if dict_dir is None:
            dict_dir = os.path.join(pretrained, "en_dict")
            log.info("[Anomalous_TTS] 下载英语词典到 %s ...", dict_dir)
            for name in EN_DICT_FILES:
                if not os.path.isfile(os.path.join(dict_dir, name)):
                    _download(EN_DICT_URL.format(name=name), os.path.join(dict_dir, name))
        cache_dir = os.path.join(pretrained, "en_dict")
        os.makedirs(cache_dir, exist_ok=True)

        nltk_dir = os.path.join(pretrained, "nltk_data")
        for name in NLTK_PACKAGES:
            target = os.path.join(nltk_dir, *name.split("/"))
            if os.path.isdir(target) or os.path.isfile(target + ".zip"):
                continue
            import zipfile

            log.info("[Anomalous_TTS] 下载 nltk 数据 %s ...", name)
            _download(NLTK_URL.format(name=name), target + ".zip")
            with zipfile.ZipFile(target + ".zip") as zf:
                zf.extractall(os.path.dirname(target))
        return dict_dir, cache_dir, nltk_dir


_g2pw_state = {"tried": False}


def g2pw_dir() -> Optional[str]:
    """Chinese polyphone model (~600MB). Returns None if unavailable; callers fall back to pypinyin."""
    with _lock:
        found = find_pretrained(G2PW_NAME, G2PW_FILES)
        if found or _g2pw_state["tried"]:
            return found
        _g2pw_state["tried"] = True
        target = os.path.join(default_root(), "pretrained")
        os.makedirs(target, exist_ok=True)
        zip_path = os.path.join(target, "G2PWModel_1.1.zip")
        try:
            import shutil
            import zipfile

            log.info("[Anomalous_TTS] 下载中文多音字模型 G2PWModel 到 %s ...", target)
            urllib.request.urlretrieve(G2PW_URL, zip_path + ".part")
            os.replace(zip_path + ".part", zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
            extracted = os.path.join(target, "G2PWModel_1.1")
            final = os.path.join(target, G2PW_NAME)
            if os.path.isdir(extracted) and not os.path.exists(final):
                shutil.move(extracted, final)
            os.remove(zip_path)
        except Exception as e:
            log.warning("[Anomalous_TTS] G2PWModel 下载失败：%s。可手动下载 %s 解压到 %s", e, G2PW_URL, target)
            for leftover in (zip_path, zip_path + ".part"):
                if os.path.exists(leftover):
                    os.remove(leftover)
        return find_pretrained(G2PW_NAME, G2PW_FILES)


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

            folder = None
            for path in _candidates("ja_userdic"):
                if os.path.isfile(os.path.join(path, "userdict.csv")):
                    folder = path
                    break
            if folder is None:
                folder = os.path.join(default_root(), "pretrained", "ja_userdic")
                os.makedirs(folder, exist_ok=True)
                tmp = os.path.join(folder, "userdict.csv.part")
                urllib.request.urlretrieve(JA_USERDICT_URL, tmp)
                os.replace(tmp, os.path.join(folder, "userdict.csv"))
            csv_path = os.path.join(folder, "userdict.csv")
            dic_path = os.path.join(folder, "user.dict")
            if not os.path.isfile(dic_path) or os.path.getmtime(dic_path) < os.path.getmtime(csv_path):
                pyopenjtalk.mecab_dict_index(csv_path, dic_path)
            pyopenjtalk.update_global_jtalk_with_user_dict(dic_path)
        except Exception as e:
            log.warning("[Anomalous_TTS] 日语用户词典不可用，英文单词会按字母读：%s", e)
