"""Volume: each sentence's voiced part is levelled (core/audio.level, used by Engine._assemble)."""

import numpy as np

from Anomalous_TTS.core.audio import level
from Anomalous_TTS.core.engine import Engine, SynthesisParams
from Anomalous_TTS.core.planner import Line, Plan, Voice

SR = 16000


def tone(amp, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))))


def test_level_sets_the_voiced_part_and_ignores_pauses():
    for amp in (0.02, 0.3):
        clip = np.concatenate([tone(amp), np.zeros(SR, dtype=np.float32)])
        out = level(clip, SR, -20.0)
        assert abs(db(out[:SR]) + 20) < 0.1


def test_level_keeps_peaks_under_the_limit_and_leaves_silence_alone():
    spiky = tone(0.01)
    spiky[100] = 0.5
    assert abs(float(np.abs(level(spiky, SR, -20.0)).max()) - 10 ** (-1 / 20)) < 1e-3
    silent = np.zeros(SR, dtype=np.float32)
    assert level(silent, SR, -20.0) is silent


def test_every_sentence_is_levelled_on_its_own():
    voice = Voice("g", "s", "r.wav", "", "ja", "v")
    plan = Plan(items=[Line(voice, "一。", "zh", 1), Line(voice, "二。", "zh", 2)])
    clips = [(tone(0.02), SR), (tone(0.4), SR)]
    params = SynthesisParams(pause_sec=0.5)
    out, _ = Engine._assemble(plan, clips, params)
    first, second = out[:SR], out[SR + SR // 2: 2 * SR + SR // 2]
    assert abs(db(first) + 20) < 0.1 and abs(db(second) + 20) < 0.1

    raw, _ = Engine._assemble(plan, clips, SynthesisParams(pause_sec=0.5, loudness_db=None))
    assert np.array_equal(raw[:SR], clips[0][0])
