"""Load GPT-SoVITS weights without executing arbitrary pickled code.

GPT-SoVITS saves its config as ``utils.HParams`` inside the checkpoint, which
normally forces ``torch.load(weights_only=False)``. Community models are
downloaded from the internet, so we load with ``weights_only=True`` and map
``utils.HParams`` to a tiny local class instead.

Version detection follows GPT-SoVITS ``process_ckpt.py`` (MIT, RVC-Boss).
"""

from __future__ import annotations

import hashlib
import io
import os
from typing import Any, Dict, Tuple

import torch


class HParams:
    """Stand-in for GPT-SoVITS ``utils.HParams`` (attribute bag)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, HParams):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _safe_load(source) -> Dict[str, Any]:
    with torch.serialization.safe_globals([(HParams, "utils.HParams")]):
        data = torch.load(source, map_location="cpu", weights_only=True)
    if "config" in data:
        data["config"] = _to_dict(data["config"])
    return data


# First two bytes of a SoVITS .pth. Newer GPT-SoVITS versions overwrite the zip
# magic "PK" with a version tag. Values: (symbol version, model version).
_HEAD_TO_VERSION = {
    b"00": ("v1", "v1"),
    b"01": ("v2", "v2"),
    b"02": ("v2", "v3"),
    b"03": ("v2", "v3"),  # v3 LoRA
    b"04": ("v2", "v4"),  # v4 LoRA
    b"05": ("v2", "v2Pro"),
    b"06": ("v2", "v2ProPlus"),
}

# md5 of the first 8 KiB of the official pretrained SoVITS files.
_PRETRAINED_HASHES = {
    "dc3c97e17592963677a4a1681f30c653": ("v2", "v2"),
    "43797be674a37c1c83ee81081941ed0f": ("v2", "v3"),
    "6642b37f3dbb1f76882b69937c95a5f3": ("v2", "v2"),
    "4f26b9476d0c5033e04162c486074374": ("v2", "v4"),
    "c7e9fce2223f3db685cdfa1e6368728a": ("v2", "v2Pro"),
    "66b313e39455b57ab1b0bc0b239c9d0a": ("v2", "v2ProPlus"),
}

SUPPORTED_SOVITS = {"v1", "v2", "v2Pro", "v2ProPlus"}


def detect_sovits_version(path: str) -> Tuple[str, str]:
    """Return (symbol_version, model_version) for a SoVITS .pth file."""
    with open(path, "rb") as f:
        head = f.read(8192)
    digest = hashlib.md5(head).hexdigest()
    if digest in _PRETRAINED_HASHES:
        return _PRETRAINED_HASHES[digest]
    tag = head[:2]
    if tag != b"PK":
        if tag not in _HEAD_TO_VERSION:
            raise ValueError(f"无法识别的 SoVITS 权重格式：{os.path.basename(path)}")
        return _HEAD_TO_VERSION[tag]
    size = os.path.getsize(path)
    if size < 82978 * 1024:
        return "v1", "v1"
    if size < 700 * 1024 * 1024:
        return "v2", "v2"
    return "v2", "v3"


def load_sovits_checkpoint(path: str) -> Tuple[Dict[str, Any], str]:
    """Return (checkpoint dict, model_version)."""
    _, model_version = detect_sovits_version(path)
    if model_version not in SUPPORTED_SOVITS:
        raise ValueError(
            f"{os.path.basename(path)} 是 {model_version} 模型，目前只支持 v1 / v2 / v2Pro / v2ProPlus。"
        )
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] != b"PK":
        raw = b"PK" + raw[2:]
    data = _safe_load(io.BytesIO(raw))
    weights = data["weight"]
    emb = weights.get("enc_p.text_embedding.weight")
    if model_version in ("v1", "v2") and emb is not None:
        # Same rule as GPT-SoVITS: the symbol table size tells v1 from v2.
        model_version = "v1" if emb.shape[0] == 322 else "v2"
    return data, model_version


def load_gpt_checkpoint(path: str) -> Dict[str, Any]:
    data = _safe_load(path)
    if "weight" not in data or "config" not in data:
        raise ValueError(f"{os.path.basename(path)} 不是 GPT-SoVITS 的 GPT 权重（.ckpt）。")
    return data
