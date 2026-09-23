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


def hubert_dir() -> str:
    """Return the chinese-hubert-base folder, downloading it on first use."""
    with _lock:
        found = find_pretrained(HUBERT_NAME, HUBERT_FILES)
        if found:
            return found
        target = os.path.join(default_root(), "pretrained")
        log.info("[Anomalous_TTS] 下载 %s 到 %s ...", HUBERT_NAME, target)
        try:
            from huggingface_hub import hf_hub_download

            for f in HUBERT_FILES:
                hf_hub_download(HF_REPO, f"{HUBERT_NAME}/{f}", local_dir=target)
        except Exception as e:  # network, proxy, missing package
            raise RuntimeError(
                f"无法下载 {HUBERT_NAME}（{e}）。请手动从 https://huggingface.co/{HF_REPO}/tree/main/{HUBERT_NAME} "
                f"下载 {', '.join(HUBERT_FILES)}，放到 {os.path.join(target, HUBERT_NAME)}。"
                "也可以在 extra_model_paths.yaml 里把 GPT-SoVITS 整合包的 GPT_SoVITS/pretrained_models 加到 gpt_sovits。"
            ) from e
        return os.path.join(target, HUBERT_NAME)


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
