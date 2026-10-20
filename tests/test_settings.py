from Anomalous_TTS.core import settings


def test_validate_ok_and_unknown_keys_allowed():
    data = {"format": 1, "aliases": ["阿罗娜"], "language": "ja", "gpt": "a.ckpt",
            "emotions": {"开心": {"audio": "x.wav", "text": "t"}}, "anything": 1}
    assert settings.validate(data, ["a.ckpt"], ["a.pth"], ["x.wav"]) == []


def test_validate_rejects_outside_files():
    problems = settings.validate({"emotions": {"开心": {"audio": "../x.wav"}}, "sovits": "b.pth"}, [], ["a.pth"], ["x.wav"])
    assert len(problems) == 2


def test_validate_rejects_main_and_bad_language():
    problems = settings.validate({"language": "ko", "emotions": {"main": {"audio": "x.wav"}}}, [], [], ["x.wav"])
    assert len(problems) == 2


def test_save_roundtrip(tmp_path):
    settings.save(str(tmp_path), {"aliases": ["阿罗娜"], "x": {"keep": True}})
    assert settings.load(str(tmp_path)) == {"aliases": ["阿罗娜"], "x": {"keep": True}, "format": 1}


def test_validate_replace_table():
    assert settings.validate({"replace": {"C站": "西站", "插件": ""}}, [], [], []) == []
    assert settings.validate({"replace": {" ": "x"}}, [], [], [])
    assert settings.validate({"replace": ["C站"]}, [], [], [])


def test_validate_defaults():
    assert settings.validate({"defaults": {"language": "zh", "speed": 1.1}}, [], [], []) == []
    assert settings.validate({"defaults": {"language": "auto"}}, [], [], []) == []
    assert len(settings.validate({"defaults": {"language": "ko", "speed": 3}}, [], [], [])) == 2
    good = {"top_k": 10, "top_p": 0.9, "temperature": 0.8, "repetition_penalty": 1.3}
    assert settings.validate({"defaults": good}, [], [], []) == []
    assert len(settings.validate({"defaults": {"top_k": 1.5, "top_p": 0, "temperature": 3, "repetition_penalty": True}}, [], [], [])) == 4
    assert settings.validate({"defaults": []}, [], [], [])


def test_validate_aux_and_lines():
    audio = ["a.wav", "b.wav", "c.wav"]
    good = {"reference": {"audio": "a.wav", "aux": ["b.wav", "c.wav"]}, "emotions": {"开心": {"audio": "b.wav", "aux": []}},
            "lines": {"c.wav": "改过的台词"}}
    assert settings.validate(good, [], [], audio) == []
    assert settings.validate({"reference": {"aux": ["b.wav"]}}, [], [], audio) == []  # the main reference stays automatic
    assert len(settings.validate({"reference": {"aux": "b.wav"}}, [], [], audio)) == 1
    assert len(settings.validate({"reference": {"aux": ["b.wav", "b.wav"]}}, [], [], audio)) == 1
    assert len(settings.validate({"emotions": {"开心": {"audio": "a.wav", "aux": ["../x.wav"]}}}, [], [], audio)) == 1
    assert len(settings.validate({"lines": {"x.wav": "t", "a.wav": " ", "b.wav": 3}}, [], [], audio)) == 3
    assert len(settings.validate({"lines": {"a.wav": "长" * (settings.LINE_LIMIT + 1)}}, [], [], audio)) == 1
    assert len(settings.validate({"lines": []}, [], [], audio)) == 1
