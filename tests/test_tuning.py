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


NARRATION = {"label": "念旁白", "top_k": 12, "top_p": 1.0, "temperature": 0.9, "repetition_penalty": 1.35}


def test_presets_made_in_anomalous():
    config = {"default": "custom_a1", "presets": {"custom_a1": NARRATION}, "languages": {"ja>zh": "custom_a1"}}
    assert tuning.validate(config) == []
    tuner = tuning.Tuner(config)
    line = tuner.for_line({}, "ja", "ja")
    assert (line.source, line.top_k, line.temperature, tuner.label(line.source)) == ("custom_a1", 12, 0.9, "念旁白")
    assert tuner.for_line({}, "ja", "zh").source == "custom_a1"
    view = tuning.view(config)
    assert list(view["presets"]) == ["stable", "standard", "lively", "custom_a1"] and view["labels"] == {"custom_a1": "念旁白"}
    # A character naming a preset deleted since keeps the general setting.
    gone = tuning.Tuner({}).for_line({"by_language": {"zh": {"preset": "custom_a1"}}}, "ja", "zh")
    assert gone.source == "stable"
    assert settings.validate({"defaults": {"preset": "custom_a1"}}, [], [], []) == []
    # Checked: a known name, a label, all four numbers, used names defined in the file.
    assert tuning.validate({"presets": {"custom_a1": {**NARRATION, "label": ""}}})
    assert tuning.validate({"presets": {"custom_a1": {k: v for k, v in NARRATION.items() if k != "top_p"}}})
    assert tuning.validate({"presets": {"custom_a1": {**NARRATION, "speed": 1.1}}})
    assert tuning.validate({"presets": {"Custom_A": NARRATION}}) and tuning.validate({"default": "custom_b2"})
    assert tuning.validate({"languages": {"cross": "custom_b2"}})
    many = {f"custom_{i}": NARRATION for i in range(tuning.MAX_CUSTOM + 1)}
    assert tuning.validate({"presets": many})


def test_one_preset_for_every_line():
    config = {"presets": {"custom_a1": NARRATION}}
    character = {"speed": 1.2, "by_language": {"zh": {"preset": "lively", "speed": 0.9}}}
    tuner = tuning.Tuner(config, preset="custom_a1")
    chinese = tuner.for_line(character, "ja", "zh")  # over the language rule and the character's own preset; speed stays
    assert (chinese.source, chinese.top_k, chinese.speed) == ("custom_a1", 12, 0.9)
    assert tuner.for_line({}, "ja", "ja").source == "custom_a1"
    node = tuning.Tuner(config, {"top_k": 40}, preset="standard").for_line({}, "ja", "zh")  # numbers changed on the node still go first
    assert (node.source, node.top_k, node.temperature) == ("node", 40, 1.0)
    assert tuning.Tuner(config, preset="custom_gone").chosen == ""


def test_a_line_of_its_own_goes_over_everything(tmp_path):
    root = tmp_path / "gpt_sovits"
    make_char(root, "阿罗娜", "JA", {"language": "ja", "defaults": {"speed": 1.2, "preset": "stable"}})
    make_char(root, "普拉娜", "JA")
    chars = characters.discover([str(root)])
    tuner = tuning.Tuner({"presets": {"custom_a1": NARRATION}}, {"top_k": 40}, preset="standard")
    script = ("おはよう。\n[speed:0.8][preset:lively]今日は晴れ。また明日。\n"
              "[普拉娜][lang:zh][speed:9][preset:custom_a1]早上好。\n[preset:loud][speed:x]さようなら。\n终わり。")
    plan = planner.build_plan(chars, planner.NodeOptions("阿罗娜"), script, tuner)
    lines = [(line.text, line.language, line.tuning.source, line.tuning.top_k, line.tuning.speed) for line in plan.lines]
    assert lines[0] == ("おはよう。", "ja", "node", 40, 1.2)
    assert lines[1][2:] == ("line", 30, 0.8) and lines[2][2:] == ("line", 30, 0.8)  # every sentence of that line
    assert lines[3] == ("早上好。", "zh", "line", 12, 2.0)  # language as written, speed kept in range
    assert lines[4][2:] == ("node", 40, 1.0) and lines[5][2] == "node"  # unusable values: left out, the line after is plain
    assert any("loud" in w for w in plan.warnings) and any("[speed:x]" in w for w in plan.warnings)


def test_a_line_can_go_back_to_detecting_and_to_the_presets(tmp_path):
    root = tmp_path / "gpt_sovits"
    make_char(root, "阿罗娜", "JA", {"language": "ja", "defaults": {"preset": "stable"}})
    chars = characters.discover([str(root)])
    tuner = tuning.Tuner({}, preset="lively")
    plan = planner.build_plan(chars, planner.NodeOptions("阿罗娜", language="zh"), "老师好。\n[lang:auto][preset:auto]おはよう。", tuner)
    first, second = plan.lines
    assert (first.language, first.tuning.source) == ("zh", "lively")
    assert (second.language, second.tuning.source, second.tuning.top_k) == ("ja", "character", 10)


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
    assert nodes.tuning_summary(seen["lines"], tuning.Tuner({})) == "语气：节点上改的值 1 句"
    node.generate("阿罗娜", "おはよう。", preset="stable")
    assert (seen["lines"][0].tuning.source, seen["lines"][0].tuning.top_k, seen["lines"][0].tuning.speed) == ("stable", 10, 1.2)
    out = node.generate("阿罗娜", "おはよう。", preset="custom_gone")
    assert "没有叫 custom_gone 的预设" in out["result"][1]
