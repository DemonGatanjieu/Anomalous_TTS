"""SoVITS versions: telling them apart, merging LoRA files, and the stand-ins for x-transformers and
librosa that v3 / v4 / v5 use (compared with the real packages when they are installed)."""

import numpy as np
import pytest
import torch

from Anomalous_TTS.core import checkpoints, paths


@pytest.mark.parametrize("head, version, lora", [
    (b"00", "v1", False), (b"01", "v2", False), (b"02", "v3", False), (b"03", "v3", True), (b"04", "v4", True),
    (b"05", "v2Pro", False), (b"06", "v2ProPlus", False), (b"07", "v5dev", True), (b"08", "v5turbo", True),
])
def test_version_from_head(tmp_path, head, version, lora):
    f = tmp_path / "x.pth"
    f.write_bytes(head + b"\0" * 100)
    assert checkpoints.detect_sovits_version(str(f)) == (version, lora)
    assert version in checkpoints.SUPPORTED_SOVITS


def test_unknown_head_is_refused(tmp_path):
    f = tmp_path / "x.pth"
    f.write_bytes(b"99" + b"\0" * 100)
    with pytest.raises(ValueError):
        checkpoints.detect_sovits_version(str(f))


def test_pretrained_each_version_needs():
    assert paths.pretrained_for("v2", False) == []
    assert paths.pretrained_for("v2ProPlus", False) == ["sv"]
    assert paths.pretrained_for("v3", True) == ["v3", "bigvgan"]
    assert paths.pretrained_for("v3", False) == ["bigvgan"]  # a full v3 model brings its own weights
    assert paths.pretrained_for("v4", True) == ["v4"]  # base model and vocoder come together
    assert paths.pretrained_for("v5turbo", True) == ["v5turbo"]
    assert {i for v in checkpoints.MEL_VERSIONS for i in paths.pretrained_for(v, True)} <= set(paths.PRETRAINED_IDS)


def test_lora_merge_is_w_plus_b_a():
    torch.manual_seed(0)
    key = "cfm.estimator.transformer_blocks.0.attn.to_q"
    base = {f"{key}.weight": torch.randn(8, 6), "ref_enc.fc.weight": torch.zeros(3), "enc_p.x": torch.ones(2)}
    a, b = torch.randn(2, 6), torch.randn(8, 2)
    lora = {
        "cfm.base_model.model.estimator.transformer_blocks.0.attn.to_q.lora_A.default.weight": a,
        "cfm.base_model.model.estimator.transformer_blocks.0.attn.to_q.lora_B.default.weight": b,
        "ref_enc.fc.weight": torch.ones(3),  # trained in full: replaces the base weight
    }
    merged = checkpoints.merge_lora(base, lora)
    assert torch.allclose(merged[f"{key}.weight"], base[f"{key}.weight"] + b @ a)
    assert torch.equal(merged["ref_enc.fc.weight"], torch.ones(3)) and torch.equal(merged["enc_p.x"], torch.ones(2))
    assert not any("lora_" in k or "base_model" in k for k in merged)


def test_rotary_matches_x_transformers():
    x = pytest.importorskip("x_transformers.x_transformers")
    from Anomalous_TTS.vendor.gpt_sovits.f5_tts.model import rotary

    ours, theirs = rotary.RotaryEmbedding(64), x.RotaryEmbedding(64)
    (fa, sa), (fb, sb) = ours.forward_from_seq_len(300), theirs.forward_from_seq_len(300)
    assert torch.equal(fa, fb) and sa == sb
    t = torch.randn(2, 300, 1024)  # the DiT rotates only the first 64 of 1024 channels, as upstream does
    assert torch.equal(rotary.apply_rotary_pos_emb(t, fa, sa), x.apply_rotary_pos_emb(t, fb, sb))


@pytest.mark.parametrize("sr, n_fft", [(24000, 1024), (32000, 1280), (32000, 2048)])
def test_mel_filters_match_librosa(sr, n_fft):
    librosa = pytest.importorskip("librosa")
    from Anomalous_TTS.vendor.gpt_sovits.module.mel_processing import librosa_mel_fn

    ours = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=100, fmin=0, fmax=None)
    theirs = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=100, fmin=0, fmax=None)
    assert ours.shape == theirs.shape and np.abs(ours - theirs).max() < 1e-6
