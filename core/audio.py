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


def level(wav: np.ndarray, sr: int, target_dbfs: float, peak_dbfs: float = -1.0, max_gain_db: float = 20.0) -> np.ndarray:
    """Scale a clip so its voiced part has ``target_dbfs`` RMS, without its peak passing ``peak_dbfs``.

    Voiced = 20 ms frames within 35 dB of the loudest one, so pauses and breaths do not pull the
    level down. Gain is capped at ``max_gain_db`` so a near-silent clip is not blown up into noise.
    """
    frame = max(1, int(sr * 0.02))
    n = wav.size // frame
    if n == 0:
        return wav
    rms = np.sqrt(np.mean(wav[: n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1))
    loudest = float(rms.max())
    if loudest < 1e-4:  # -80 dBFS: silence
        return wav
    voiced = rms[rms >= loudest * 10 ** (-35 / 20)]
    gain = 10 ** (target_dbfs / 20) / float(np.sqrt(np.mean(voiced ** 2)))
    gain = min(gain, 10 ** (max_gain_db / 20), 10 ** (peak_dbfs / 20) / float(np.abs(wav).max()))
    return (wav * gain).astype(np.float32)


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sr))), dtype=np.float32)
