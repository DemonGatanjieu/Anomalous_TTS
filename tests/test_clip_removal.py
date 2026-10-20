import json

import pytest

from Anomalous_TTS import server
from Anomalous_TTS.core import characters, clip_removal

from test_planner import make_char


def test_settings_without_removed_clips():
    data = {
        "format": 1, "keep": 1,
        "reference": {"audio": "a.wav", "text": "A", "language": "ja", "aux": ["b.wav", "c.wav"]},
        "emotions": {"开心": {"audio": "b.wav"}, "生气": {"audio": "c.wav", "aux": ["a.wav"]}},
        "lines": {"a.wav": "x", "c.wav": "y"},
    }
    out = clip_removal.settings_without(data, {"a.wav", "b.wav"})
    assert out == {
        "format": 1, "keep": 1,
        "reference": {"language": "ja", "aux": ["c.wav"]},  # the main reference is picked automatically again
        "emotions": {"生气": {"audio": "c.wav"}},
        "lines": {"c.wav": "y"},
    }
    assert data["reference"]["audio"] == "a.wav", "the input is left as it was"
    assert clip_removal.settings_without({"reference": {"audio": "a.wav"}, "lines": {"a.wav": "x"}}, {"a.wav"}) == {}


@pytest.fixture
def lib(tmp_path, monkeypatch):
    root = tmp_path / "gpt_sovits"
    folder = make_char(root, "普拉娜", "JA", {
        "reference": {"audio": "ref/a.wav", "aux": ["ref/b.wav"]},
        "emotions": {"平静": {"audio": "ref/b.wav"}},
    })
    (folder / "ref" / "b.lab").write_text("台词", encoding="utf-8")
    monkeypatch.setattr(characters, "scan", lambda max_age=None: characters.discover([str(root)]))
    moved = []

    def fake_trash(paths):
        paths = list(paths)
        for path in paths:
            moved.append(path)
            import os
            os.remove(path)
        return paths

    monkeypatch.setattr(clip_removal, "move_to_trash", fake_trash)
    return folder, moved


def test_remove_clips_to_the_recycle_bin_and_out_of_the_settings(lib):
    folder, moved = lib
    out = server._remove_audio({"character": "普拉娜", "paths": ["ref/b.wav"]})
    assert out["removed"] == ["ref/b.wav"] and out["text_files"] == ["ref/b.lab"]
    assert sorted(p.replace("\\", "/").split("/ref/")[-1] for p in moved) == ["b.lab", "b.wav"]
    assert not (folder / "ref" / "b.wav").exists() and (folder / "all.list").exists()  # the annotation file stays
    saved = json.loads((folder / "anomalous_tts.json").read_text(encoding="utf-8"))
    assert saved["reference"] == {"audio": "ref/a.wav"} and "emotions" not in saved
    assert "ref/b.wav" not in out["character"]["audio"]


def test_remove_takes_only_the_characters_own_clips(lib):
    folder, moved = lib
    for bad in (["../x.wav"], ["ref/none.wav"], [], "ref/a.wav", [1]):
        with pytest.raises(ValueError):
            server._remove_audio({"character": "普拉娜", "paths": bad})
    with pytest.raises(ValueError):
        server._remove_audio({"character": "没有这个人", "paths": ["ref/a.wav"]})
    assert moved == []


def test_no_recycle_bin_deletes_nothing(lib, monkeypatch):
    folder, _ = lib

    def no_bin(paths):
        raise clip_removal.TrashUnavailable("没有回收站")

    monkeypatch.setattr(clip_removal, "move_to_trash", no_bin)
    with pytest.raises(ValueError, match="回收站"):
        server._remove_audio({"character": "普拉娜", "paths": ["ref/a.wav"]})
    assert (folder / "ref" / "a.wav").exists()
    assert "audio" in json.loads((folder / "anomalous_tts.json").read_text(encoding="utf-8"))["reference"]
