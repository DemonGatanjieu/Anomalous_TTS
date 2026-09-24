import json

import numpy as np
import pytest
import soundfile as sf

from Anomalous_TTS.core import characters
from Anomalous_TTS.core.planner import Gap, NodeOptions, build_plan


def wav(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype=np.float32), 16000)


def make_char(root, name, lang_code="JA", settings=None):
    folder = root / name
    (folder / "GPT_weights_v2").mkdir(parents=True)
    (folder / "SoVITS_weights_v2").mkdir(parents=True)
    for e in (5, 15):
        (folder / "GPT_weights_v2" / f"X-e{e}.ckpt").write_bytes(b"")
    for e in (4, 16):
        (folder / "SoVITS_weights_v2" / f"X_e{e}_s{e * 10}.pth").write_bytes(b"")
    wav(folder / "ref" / "a.wav", 5)
    wav(folder / "ref" / "b.wav", 6)
    wav(folder / "ref" / "c.开心.wav", 4)
    wav(folder / "ref" / "short.wav", 1)
    (folder / "all.list").write_text(
        f"/x/a.wav|s|{lang_code}|台词A\n/x/b.wav|s|{lang_code}|台词B\n/x/c.wav|s|{lang_code}|台词C\n", encoding="utf-8"
    )
    if settings is not None:
        (folder / "anomalous_tts.json").write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
    return folder


@pytest.fixture
def chars(tmp_path):
    root = tmp_path / "gpt_sovits"
    make_char(root / "阿罗娜", "日配", "JA", {"aliases": ["阿罗娜"], "language": "ja",
                                           "emotions": {"平静": {"audio": "ref/b.wav"}}})
    make_char(root / "阿罗娜", "中配", "ZH", {"aliases": ["阿罗娜"], "language": "zh"})
    make_char(root, "普拉娜", "JA")
    return characters.discover([str(root)])


def test_discovery_and_variants(chars):
    assert set(chars) == {"阿罗娜/日配", "阿罗娜/中配", "普拉娜"}
    c = chars["阿罗娜/日配"]
    assert c.language == "ja"
    assert c.weight("gpt").endswith("X-e15.ckpt")
    assert c.weight("sovits").endswith("X_e16_s160.pth")


def test_references_and_emotions(chars):
    c = chars["阿罗娜/日配"]
    ref = c.default_reference()
    assert (ref.audio, ref.text, ref.language, ref.source) == ("ref/a.wav", "台词A", "ja", "auto")
    emotions = c.emotions()
    assert emotions["开心"].audio == "ref/c.开心.wav" and emotions["开心"].text == "台词C"  # untagged name in .list
    assert emotions["开心"].source == "filename"
    assert emotions["平静"].audio == "ref/b.wav" and emotions["平静"].source == "settings"


def test_plan_emotions_speakers_pauses(chars):
    plan = build_plan(chars, NodeOptions(character="阿罗娜/日配", seed=1),
                      "先生、おはよう。{开心}やった！[pause:1][普拉娜]こんにちは。[阿罗娜]老师好。{生气}哼。[谁]你好。")
    kinds = [type(i).__name__ for i in plan.items]
    assert kinds == ["Line", "Line", "Gap", "Line", "Line", "Line", "Line"]
    lines = plan.lines
    assert lines[0].voice.label == "阿罗娜/日配" and lines[1].voice.label == "阿罗娜/日配{开心}"
    assert lines[2].voice.label == "普拉娜"
    assert lines[3].voice.label == "阿罗娜/中配" and lines[3].language == "zh"  # alias picked by language
    assert lines[4].voice.label == "阿罗娜/中配"  # unknown emotion -> main voice
    assert lines[5].voice.label == "阿罗娜/中配"  # unknown speaker -> keep current
    assert any("生气" in w for w in plan.warnings) and any("[谁]" in w for w in plan.warnings)
    assert plan.items[2] == Gap(1.0)


def test_seeds_depend_only_on_the_line(chars):
    opts = NodeOptions(character="阿罗娜/日配", seed=7)
    a = build_plan(chars, opts, "一つ目。二つ目。三つ目。").lines
    b = build_plan(chars, opts, "一つ目。変えた。三つ目。").lines
    assert a[0].seed == b[0].seed and a[2].seed == b[2].seed and a[1].seed != b[1].seed
    c = build_plan(chars, opts, "これは同じ文です。これは同じ文です。").lines
    assert c[0].seed != c[1].seed  # repeated sentence gets a different seed


def test_node_overrides(chars):
    opts = NodeOptions(character="阿罗娜/日配", reference_audio="ref/b.wav", reference_text="手动", gpt="GPT_weights_v2/X-e5.ckpt")
    line = build_plan(chars, opts, "テスト。").lines[0]
    assert line.voice.ref_wav.endswith("b.wav") and line.voice.ref_text == "手动"
    assert line.voice.gpt_path.endswith("X-e5.ckpt")


def test_ambiguous_alias_without_language_hint(chars):
    with pytest.raises(ValueError):
        characters.resolve_name(chars, "阿罗娜", None)


def test_extension_like_parts_are_not_emotions():
    assert characters.emotion_of("a/X.ogg.wav") is None
    assert characters.emotion_of("a/X.ogg (1).ogg") is None
    assert characters.emotion_of("a/X.开心.wav") == "开心"
    assert characters.emotion_of("a/X.wav") is None


def test_api_summary_and_detail(chars):
    c = chars["阿罗娜/日配"]
    summary, detail = c.to_api(), c.to_api(detail=True)
    assert "audio" not in summary and summary["counts"]["audio"] == 4
    assert set(summary["emotions"]) == {"开心", "平静"} and summary["reference"]["audio"] == "ref/a.wav"
    assert detail["audio"] == c.audio and detail["gpt"] == c.gpt
