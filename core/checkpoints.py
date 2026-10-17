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
from typing import Any, Callable, Dict, NamedTuple, Tuple

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
# magic "PK" with a version tag. Values: (model version, LoRA on the official pretrained model).
_HEAD_TO_VERSION = {
    b"00": ("v1", False),
    b"01": ("v2", False),
    b"02": ("v3", False),
    b"03": ("v3", True),
    b"04": ("v4", True),
    b"05": ("v2Pro", False),
    b"06": ("v2ProPlus", False),
    b"07": ("v5dev", True),
    b"08": ("v5turbo", True),
}

# md5 of the first 8 KiB of the official pretrained SoVITS files.
_PRETRAINED_HASHES = {
    "dc3c97e17592963677a4a1681f30c653": ("v2", False),
    "43797be674a37c1c83ee81081941ed0f": ("v3", False),
    "6642b37f3dbb1f76882b69937c95a5f3": ("v2", False),
    "4f26b9476d0c5033e04162c486074374": ("v4", False),
    "c7e9fce2223f3db685cdfa1e6368728a": ("v2Pro", False),
    "66b313e39455b57ab1b0bc0b239c9d0a": ("v2ProPlus", False),
    "c4ce1f839a0d271e41963f9fb16ade76": ("v5dev", False),
    "6bc5cb2a195cc8ec22b2e723cd4d6f47": ("v5turbo", False),
}

SUPPORTED_SOVITS = {"v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4", "v5dev", "v5turbo"}
# Versions that turn the SoVITS features into a mel spectrogram (diffusion) and then audio (vocoder).
MEL_VERSIONS = {"v3", "v4", "v5dev", "v5turbo"}


class SovitsKind(NamedTuple):
    version: str
    lora: bool  # only the fine-tuned part: needs the official pretrained model of its version


def detect_sovits_version(path: str) -> SovitsKind:
    """Model version of a SoVITS .pth file, from its first 8 KiB (and size, for old files)."""
    with open(path, "rb") as f:
        head = f.read(8192)
    digest = hashlib.md5(head).hexdigest()
    if digest in _PRETRAINED_HASHES:
        return SovitsKind(*_PRETRAINED_HASHES[digest])
    tag = head[:2]
    if tag != b"PK":
        if tag not in _HEAD_TO_VERSION:
            raise ValueError(f"无法识别的 SoVITS 权重格式：{os.path.basename(path)}")
        return SovitsKind(*_HEAD_TO_VERSION[tag])
    size = os.path.getsize(path)
    if size < 82978 * 1024:
        return SovitsKind("v1", False)
    if size < 700 * 1024 * 1024:
        return SovitsKind("v2", False)
    return SovitsKind("v3", False)


def _read_sovits(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] != b"PK":
        raw = b"PK" + raw[2:]
    return _safe_load(io.BytesIO(raw))


_PEFT = "cfm.base_model.model."


def merge_lora(base: Dict[str, torch.Tensor], lora: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The weights GPT-SoVITS loads for a LoRA file, without peft: the pretrained weights, the parts the
    file trained in full, and each LoRA pair merged in as ``W + B @ A`` (its lora_alpha equals the rank,
    so the scale is 1; same as peft's merge_and_unload)."""
    weights = dict(base)
    for key, value in lora.items():
        name = "cfm." + key[len(_PEFT):] if key.startswith(_PEFT) else key
        if ".lora_B." in name:
            continue
        if ".lora_A." in name:
            target = name.split(".lora_A.")[0] + ".weight"
            b = lora[key.replace(".lora_A.", ".lora_B.")]
            weights[target] = (base[target].float() + b.float() @ value.float()).to(base[target].dtype)
        else:
            weights[name] = value
    return weights


def load_sovits_checkpoint(path: str, pretrained: Callable[[str], str]) -> Tuple[Dict[str, Any], str]:
    """Return (checkpoint dict, model_version). ``pretrained(version)`` is the official pretrained
    SoVITS file a LoRA file is merged into."""
    kind = detect_sovits_version(path)
    if kind.version not in SUPPORTED_SOVITS:
        raise ValueError(f"{os.path.basename(path)} 是 {kind.version} 模型，这个版本的节点不认识。")
    data = _read_sovits(path)
    model_version = kind.version
    if kind.lora:
        base = _read_sovits(pretrained(model_version))
        data = {**data, "weight": merge_lora(base["weight"], data["weight"])}
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
