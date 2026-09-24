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
