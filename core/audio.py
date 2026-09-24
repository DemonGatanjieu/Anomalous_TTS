"""Small audio helpers shared by the engine."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import soundfile as sf


def load_mono(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def resample(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """soxr "HQ" is what librosa.load(sr=...) uses, which GPT-SoVITS calls for 16 kHz input."""
    if sr_from == sr_to:
        return wav
    import soxr

    return soxr.resample(wav, sr_from, sr_to, quality="HQ").astype(np.float32)


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sr))), dtype=np.float32)
