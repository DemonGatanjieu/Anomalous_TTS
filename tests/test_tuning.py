"""Generation presets: what each line is sampled with, the presets file, and the node's part."""

import json

import pytest

from Anomalous_TTS import nodes, server
from Anomalous_TTS.core import characters, planner, settings, tuning

from test_planner import make_char


def sampling(t):
    return (t.top_k, t.top_p, t.temperature, t.repetition_penalty)


def test_without_any_setting_lines_sample_as_before():
    tuner = tuning.Tuner({})
    same = tuner.for_line({}, "ja", "ja")
    assert (sampling(same), same.speed, same.source) == ((15, 1.0, 1.0, 1.35), 1.0, "standard")
    other = tuner.for_line({}, "ja", "zh")  # what the node did for another language: top_k <= 10, temperature <= 0.8
    assert (sampling(other), other.source) == ((10, 1.0, 0.8, 1.35), "stable")
    assert tuning.Tuner({}, cross_lingual=False).for_line({}, "ja", "zh").source == "standard"


def test_order_of_the_layers():
    config = {"default": "lively", "presets": {"stable": {"top_k": 8}}, "languages": {"ja>en": "standard", "cross": "stable"}}
    tuner = tuning.Tuner(config)
    assert tuner.for_line({}, "zh", "zh").source == "lively"
    assert sampling(tuner.for_line({}, "ja", "zh"))[:3] == (8, 1.0, 0.8)  # "cross" with the changed number
    assert tuner.for_line({}, "ja", "en").source == "standard"  # the pair's own rule
    character = {"preset": "standard", "speed": 1.2, "by_language": {"zh": {"preset": "lively", "speed": 0.9}}}
    own_language = tuner.for_line(character, "ja", "ja")
    assert (own_language.source, own_language.top_k, own_language.speed) == ("character", 15, 1.2)
    english = tuner.for_line(character, "ja", "en")  # the language rule over the character's own defaults
    assert (english.source, english.top_k, english.speed) == ("standard", 15, 1.2)
    chinese = tuner.for_line(character, "ja", "zh")  # the character's setting for this language over the rule
    assert (chinese.source, chinese.top_k, chinese.temperature, chinese.speed) == ("character", 30, 1.1, 0.9)
    node = tuning.Tuner(config, {"temperature": 0.5}).for_line(character, "ja", "zh")
    assert (node.source, node.top_k, node.temperature) == ("node", 30, 0.5)


def test_presets_file_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "path", lambda: str(tmp_path / tuning.FILENAME))
    assert tuning.load() == {}
    assert tuning.validate({"default": "loud"}) and tuning.validate({"languages": {"ja>ja": "stable"}})
    assert tuning.validate({"languages": {"ja>zh": "x"}}) and tuning.validate({"presets": {"stable": {"speed": 1.2}}})
    assert tuning.validate({"presets": {"stable": {"top_k": 0}}}) and tuning.validate({"presets": {"mine": {}}})
    good = {"default": "stable", "presets": {"lively": {"temperature": 1.2}}, "languages": {"zh>ja": "lively", "cross": "standard"}}
    assert tuning.validate(good) == []
    tuning.save(good)
    assert json.loads((tmp_path / tuning.FILENAME).read_text(encoding="utf-8"))["format"] == 1
    view = tuning.view(tuning.load())
    assert (view["default"], view["presets"]["lively"]["temperature"], view["languages"], view["cross"]) == (
        "stable", 1.2, {"zh>ja": "lively"}, "standard")
    (tmp_path / tuning.FILENAME).write_text('{"default": "loud"}', encoding="utf-8")
    assert tuning.load() == {}  # a broken file: the built-in presets


def test_presets_route_saves_only_valid_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "path", lambda: str(tmp_path / tuning.FILENAME))
    with pytest.raises(server.web.HTTPBadRequest):
        server._save_presets({"presets": {"default": "loud"}})
    assert not (tmp_path / tuning.FILENAME).exists()
    out = server._save_presets({"presets": {"default": "lively"}})
    assert out["ok"] and out["default"] == "lively" and out["format"] == server.API_FORMAT


def test_character_generation_settings_are_checked():
    check = lambda defaults: settings.validate({"defaults": defaults}, [], [], [])  # noqa: E731
    assert check({"preset": "lively", "speed": 1.1, "by_language": {"zh": {"preset": "stable", "top_k": 5}}}) == []
    assert check({"preset": "loud"}) and check({"by_language": {"fr": {}}}) and check({"by_language": {"zh": {"top_k": 0}}})


def test_each_line_carries_its_tuning(tmp_path):
    root = tmp_path / "gpt_sovits"
    make_char(root, "阿罗娜", "JA", {"language": "ja", "defaults": {"by_language": {"zh": {"preset": "lively"}}}})
    chars = characters.discover([str(root)])
    plan = planner.build_plan(chars, planner.NodeOptions("阿罗娜"), "おはようございます。老师，这是什么？")
    assert [(line.language, line.tuning.source) for line in plan.lines] == [("ja", "standard"), ("zh", "character")]


def test_node_values_count_only_when_changed(tmp_path, monkeypatch):
    root = tmp_path / "gpt_sovits"
    make_char(root, "阿罗娜", "JA", {"language": "ja", "defaults": {"speed": 1.2, "top_k": 20}})
    chars = characters.discover([str(root)])
    monkeypatch.setattr(characters, "scan", lambda max_age=0: chars)
    seen = {}

    class FakeEngine:
        def synthesize(self, plan, params, progress=None):
            seen["lines"] = plan.lines
            import numpy as np
            from Anomalous_TTS.core.engine import Report
            return np.zeros(10, dtype=np.float32), 32000, Report(lines=len(plan.lines))

    monkeypatch.setattr(nodes, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(nodes.paths, "ensure_ja_userdict", lambda: None)
    node = nodes.AnomalousTTS_CharacterSpeech()
    node.generate("阿罗娜", "おはよう。", speed=1.2, top_k=20)  # Anomalous sending the character's defaults back
    first = seen["lines"][0].tuning
    assert (first.source, first.top_k, first.speed) == ("character", 20, 1.2)
    node.generate("阿罗娜", "おはよう。", temperature=0.5)
    assert (seen["lines"][0].tuning.source, seen["lines"][0].tuning.temperature) == ("node", 0.5)
    assert nodes.tuning_summary(seen["lines"]) == "生成设置：节点上改的值 1 句"
