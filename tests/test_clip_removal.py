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


@pytest.fixture
def library(tmp_path, monkeypatch):
    from Anomalous_TTS.core import paths

    root = tmp_path / "gpt_sovits"
    make_char(root, "普拉娜", "JA", {})
    make_char(root / "阿罗娜", "日配", "JA", {})
    make_char(root / "阿罗娜", "中配", "ZH", {})
    monkeypatch.setattr(characters, "scan", lambda max_age=None: characters.discover([str(root)]))
    monkeypatch.setattr(paths, "roots", lambda: [str(root)])
    moved = []

    def fake_trash(targets):
        import shutil
        targets = list(targets)
        for path in targets:
            moved.append(path)
            shutil.rmtree(path)
        return targets

    monkeypatch.setattr(clip_removal, "move_to_trash", fake_trash)
    return root, moved


def test_remove_character_moves_its_folder_to_the_recycle_bin(library):
    root, moved = library
    assert server._remove_character({"character": "普拉娜"}) == {"ok": True, "removed": "普拉娜"}
    assert [p.replace("\\", "/").split("/gpt_sovits/")[-1] for p in moved] == ["普拉娜"]
    assert not (root / "普拉娜").exists()
    # A variant takes only its own subfolder.
    server._remove_character({"character": "阿罗娜/日配"})
    assert not (root / "阿罗娜" / "日配").exists() and (root / "阿罗娜" / "中配").exists()


def test_remove_character_takes_only_a_known_character_inside_a_library(library, monkeypatch):
    from Anomalous_TTS.core import paths

    root, moved = library
    for bad in ({"character": "没有这个人"}, {"character": ""}, {"character": ["普拉娜"]}, {}):
        with pytest.raises(ValueError):
            server._remove_character(bad)
    monkeypatch.setattr(paths, "roots", lambda: [str(root / "普拉娜")])  # the folder is a library itself
    with pytest.raises(ValueError, match="角色库"):
        server._remove_character({"character": "普拉娜"})
    monkeypatch.setattr(paths, "roots", lambda: [])
    with pytest.raises(ValueError, match="角色库"):
        server._remove_character({"character": "普拉娜"})
    assert moved == [] and (root / "普拉娜").exists()


def test_remove_character_without_a_recycle_bin_deletes_nothing(library, monkeypatch):
    root, _ = library

    def no_bin(targets):
        raise clip_removal.TrashUnavailable("没有回收站")

    monkeypatch.setattr(clip_removal, "move_to_trash", no_bin)
    with pytest.raises(ValueError, match="回收站"):
        server._remove_character({"character": "普拉娜"})
    assert (root / "普拉娜").exists()
