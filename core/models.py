"""Loading GPT-SoVITS models and the small caches that hold them."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Callable, Hashable, Optional, Protocol, Tuple, TypeVar

import torch

from .checkpoints import MEL_VERSIONS, load_gpt_checkpoint, load_sovits_checkpoint

log = logging.getLogger("Anomalous_TTS")
T = TypeVar("T")


class Resources(Protocol):
    """Where pretrained files live. core/paths.ComfyResources in ComfyUI; plain objects in tests."""

    def hubert_dir(self) -> str: ...

    def roberta_dir(self) -> str: ...

    def g2pw_dir(self) -> Optional[str]: ...

    def sv_path(self) -> str: ...

    def english_dirs(self) -> Tuple[str, str]:
        """(dictionary dir, nltk data dir)"""

    def sovits_pretrained(self, version: str) -> str:
        """The official SoVITS model a v3 / v4 / v5 LoRA file is merged into."""

    def vocoder(self, version: str) -> str:
        """The v3 BigVGAN folder, or the v4 / v5 ``vocoder.pth``."""


class LRU(OrderedDict):
    """Keeps at most ``capacity`` items, or ``max_bytes`` if ``sizeof`` is given."""

    def __init__(self, capacity: int = 0, max_bytes: int = 0, sizeof: Optional[Callable[[object], int]] = None):
        super().__init__()
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.sizeof = sizeof
        self.bytes = 0

    def get(self, key: Hashable, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def put(self, key: Hashable, value) -> None:
        if key in self:
            self.bytes -= self.sizeof(self[key]) if self.sizeof else 0
            del self[key]
        self[key] = value
        self.bytes += self.sizeof(value) if self.sizeof else 0
        while self and (
            (self.capacity and len(self) > self.capacity) or (self.max_bytes and self.bytes > self.max_bytes)
        ):
            _, old = self.popitem(last=False)
            self.bytes -= self.sizeof(old) if self.sizeof else 0

    def get_or_create(self, key: Hashable, factory: Callable[[], T]) -> T:
        if key in self:
            return self.get(key)
        value = factory()
        self.put(key, value)
        return value

    def clear(self) -> None:
        super().clear()
        self.bytes = 0


def file_key(path: str) -> Tuple[str, float]:
    """Cache key that changes when the file is replaced."""
    return path, os.path.getmtime(path)


class GPTModel:
    def __init__(self, path: str, device: torch.device, dtype: torch.dtype):
        from ..vendor.gpt_sovits.AR.models.t2s_model import Text2SemanticDecoder

        data = load_gpt_checkpoint(path)
        config = data["config"]
        self.max_sec = config["data"]["max_sec"]
        model = Text2SemanticDecoder(config=config, top_k=3)
        state = {k[len("model.") :]: v for k, v in data["weight"].items() if k.startswith("model.")}
        model.load_state_dict(state)
        self.model = model.to(device=device, dtype=dtype).eval()


class SoVITSModel:
    def __init__(self, path: str, device: torch.device, dtype: torch.dtype, pretrained: Callable[[str], str]):
        from ..vendor.gpt_sovits.module.models import SynthesizerTrn, SynthesizerTrnV3

        data, version = load_sovits_checkpoint(path, pretrained)
        self.version = version
        self.is_v2pro = version in ("v2Pro", "v2ProPlus")
        self.makes_mel = version in MEL_VERSIONS
        hps = data["config"]
        hps["model"]["semantic_frame_rate"] = "25hz"
        hps["model"]["version"] = version
        d = hps["data"]
        self.sampling_rate = d["sampling_rate"]  # of the reference spectrogram
        self.output_rate = (24000 if version == "v3" else 48000) if self.makes_mel else self.sampling_rate
        self.filter_length = d["filter_length"]
        self.hop_length = d["hop_length"]
        self.win_length = d["win_length"]
        model = (SynthesizerTrnV3 if self.makes_mel else SynthesizerTrn)(
            self.filter_length // 2 + 1,
            hps["train"]["segment_size"] // self.hop_length,
            n_speakers=d["n_speakers"],
            **hps["model"],
        )
        if hasattr(model, "enc_q"):
            del model.enc_q  # posterior encoder: training only
        result = model.load_state_dict(data["weight"], strict=False)
        # The rotary table is computed; only LoRA files happen to store it.
        missing = [k for k in result.missing_keys if not k.endswith("rotary_embed.inv_freq")]
        if missing:
            log.warning("[Anomalous_TTS] %s 缺少 %d 个权重：%s", os.path.basename(path), len(missing), missing[:5])
        self.model = model.to(device=device, dtype=dtype).eval()


def load_vocoder(version: str, path: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Mel spectrogram -> audio for v3 (BigVGAN, 24 kHz) and v4 / v5 (GPT-SoVITS' HiFi-GAN, 48 kHz),
    built like ``init_bigvgan`` / ``init_hifigan`` in GPT-SoVITS ``inference_webui.py``."""
    if version == "v3":
        from ..vendor.gpt_sovits.BigVGAN import bigvgan

        model = bigvgan.BigVGAN(bigvgan.load_hparams_from_json(os.path.join(path, "config.json")))
        state = torch.load(os.path.join(path, "bigvgan_generator.pt"), map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state["generator"])
            model.remove_weight_norm()
        except RuntimeError:  # a copy saved without weight norm, as BigVGAN's own loader allows
            model.remove_weight_norm()
            model.load_state_dict(state["generator"])
    else:
        from ..vendor.gpt_sovits.module.models import Generator

        model = Generator(
            initial_channel=100, resblock="1", resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], upsample_rates=[10, 6, 2, 2, 2],
            upsample_initial_channel=512, upsample_kernel_sizes=[20, 12, 4, 4, 4], gin_channels=0, is_bias=True,
        )
        model.remove_weight_norm()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device=device, dtype=dtype).eval()
