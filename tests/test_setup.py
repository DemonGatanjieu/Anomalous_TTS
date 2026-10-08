"""Setup from the UI: pretrained sources, downloads, folder browsing, status (storage: test_storage.py)."""

import json
import os
import time
import types

import folder_paths
import pytest
from aiohttp import web

from Anomalous_TTS import server
from Anomalous_TTS.core import app_config, browse, characters, downloads, importer, paths

from test_planner import make_char


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """Empty user directory and only the default library, restored afterwards."""
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path / "user"))
    registered = folder_paths.folder_names_and_paths[paths.CATEGORY][0]
    saved, saved_sources = list(registered), list(paths._sources)
    registered[:] = [p for p in registered if paths._key(p) == paths._key(paths._default_library())]
    paths._sources.clear()
    characters.invalidate()
    yield tmp_path
    registered[:] = saved
    paths._sources[:] = saved_sources
    characters.invalidate()


def configure(**fields):
    """Write the settings file the way a user does (folders as plain strings)."""
    data = {"format": 1, **{k: [str(p) for p in v] if isinstance(v, list) else str(v) for k, v in fields.items()}}
    os.makedirs(os.path.dirname(app_config.path()), exist_ok=True)
    with open(app_config.path(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _fake_package(root):
    """A GPT-SoVITS package's GPT_SoVITS folder with HuBERT and the English dictionaries."""
    hubert = root / "pretrained_models" / paths.HUBERT_NAME
    hubert.mkdir(parents=True)
    for f in paths.HUBERT_FILES:
        (hubert / f).write_bytes(b"x")
    text = root / "text"
    text.mkdir()
    for f in paths.EN_DICT_FILES:
        (text / f).write_bytes(b"x")
    return root


def test_pretrained_source_is_searched(fresh):
    pkg = _fake_package(fresh / "GPT-SoVITS" / "GPT_SoVITS")
    assert paths.locate("hubert") is None
    paths.add_pretrained_source(str(pkg))
    assert paths.is_inside(paths.locate("hubert"), str(pkg))
    assert paths.is_inside(paths._find_en_dict(), str(pkg / "text"))
    assert paths.pretrained_sources() == [paths.norm(str(pkg))]
    assert app_config.load()["pretrained"] == [paths.norm(str(pkg))]

    paths.remove_pretrained_source(str(pkg))
    assert paths.locate("hubert") is None


def test_pretrained_source_without_models_is_rejected(fresh):
    (fresh / "empty").mkdir()
    with pytest.raises(ValueError, match="没有找到底模"):
        paths.add_pretrained_source(str(fresh / "empty"))
    assert app_config.load()["pretrained"] == []


def _wait_for(check, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_download_needs_explicit_ids():
    for body in ({}, {"ids": "g2pw"}, {"ids": [1]}):
        with pytest.raises(web.HTTPBadRequest):
            server._download_ids(body)
    assert server._download_ids({"ids": ["g2pw"]}) == ["g2pw"]


def test_download_runs_in_background_and_reports_errors(fresh, monkeypatch):
    fetched = []

    def fake_fetch(item_id):
        if item_id == "sv":
            raise RuntimeError("网络不通")
        fetched.append(item_id)

    monkeypatch.setattr(paths, "fetch", fake_fetch)
    monkeypatch.setattr(paths, "locate", lambda item_id, dirs=None: "done" if item_id in fetched else None)
    downloads.start(["hubert", "sv"])
    _wait_for(lambda: downloads.state("hubert") is None and (downloads.state("sv") or {}).get("state") == "error")
    assert fetched == ["hubert"]
    assert downloads.state("sv")["error"] == "网络不通"

    downloads.start(["hubert"])  # already there: no new job
    assert downloads.state("hubert") is None
    with pytest.raises(ValueError):
        downloads.start(["nope"])


def test_browse_lists_folders_and_usable_files(fresh):
    folder = fresh / "pkg"
    (folder / "GPT_weights_v2").mkdir(parents=True)
    (folder / ".git").mkdir()
    for name in ("a-e15.ckpt", "b_e8_s100.pth", "ref.wav", "all.list", "readme.md"):
        (folder / name).write_bytes(b"xx")
    configure(import_folders=[fresh])
    out = browse.listing(str(folder))
    assert out["path"] == paths.norm(str(folder))
    assert out["parent"] == paths.norm(str(fresh))
    assert out["dirs"] == ["GPT_weights_v2"]
    assert {f["name"]: f["kind"] for f in out["files"]} == {
        "a-e15.ckpt": "gpt", "b_e8_s100.pth": "sovits", "ref.wav": "audio", "all.list": "text"}

    with pytest.raises(ValueError):
        browse.listing(str(fresh / "nope"))


def test_browsing_stays_inside_the_import_folders(fresh):
    inside, outside = fresh / "GPT-SoVITS", fresh / "private"
    (inside / "ref").mkdir(parents=True)
    outside.mkdir()
    (outside / "secret.wav").write_bytes(b"x")
    assert browse.listing(None)["dirs"] == []  # nothing configured: nothing to list
    configure(import_folders=[inside, fresh / "unplugged"])
    top = browse.listing(None)
    assert (top["dirs"], top["parent"]) == ([paths.norm(str(inside))], None)  # not the drives
    assert browse.listing(str(inside))["parent"] == ""  # up from an import folder: the list above
    assert browse.listing(str(inside / "ref"))["parent"] == paths.norm(str(inside))
    for bad in (str(fresh), str(outside), str(inside / ".." / "private")):
        with pytest.raises(ValueError, match="不在导入文件夹里"):
            browse.listing(bad)
        with pytest.raises(ValueError, match="不在导入文件夹里"):
            browse.scan(bad)
    with pytest.raises(ValueError, match="不在导入文件夹里"):
        browse.scan("")
    with pytest.raises(ValueError, match="不在导入文件夹里"):
        browse.audio_file(str(outside / "secret.wav"))
    with pytest.raises(ValueError, match="不在导入文件夹里"):
        importer.inspect([{"path": str(outside / "secret.wav")}])


def test_a_link_inside_an_import_folder_does_not_lead_out(fresh):
    inside, outside = fresh / "in", fresh / "out"
    inside.mkdir()
    outside.mkdir()
    (outside / "secret.wav").write_bytes(b"x")
    try:
        os.symlink(outside / "secret.wav", inside / "link.wav")
        os.symlink(outside, inside / "linked", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot make links here")
    configure(import_folders=[inside])
    assert browse.listing(str(inside))["dirs"] == []  # linked folders are not listed
    for bad in (inside / "link.wav", inside / "linked" / "secret.wav"):
        with pytest.raises(ValueError, match="不在导入文件夹里"):
            browse.audio_file(str(bad))
        with pytest.raises(ValueError, match="不在导入文件夹里"):
            importer.inspect([{"path": str(bad)}])
    with pytest.raises(ValueError, match="不在导入文件夹里"):
        browse.scan(str(inside / "linked"))


def test_browse_scan_lists_usable_files_with_their_folders(fresh):
    root = fresh / "pkg"
    (root / "GPT_weights_v2").mkdir(parents=True)
    (root / "GPT_weights_v2" / "A-e10.ckpt").write_bytes(b"PK")
    configure(import_folders=[root])
    (root / "voices" / "A" / ".hidden").mkdir(parents=True)
    (root / "voices" / "A" / "hi.wav").write_bytes(b"x")
    (root / "voices" / "A" / "notes.docx").write_bytes(b"x")
    (root / "voices" / "A" / ".hidden" / "x.wav").write_bytes(b"x")
    deep = root / "1" / "2" / "3" / "4" / "5" / "6" / "7"
    deep.mkdir(parents=True)
    (deep.parent / "deep_enough.wav").write_bytes(b"x")
    (deep / "too_deep.wav").write_bytes(b"x")
    for skipped in ("runtime", "GPT_SoVITS/pretrained_models", "logs/A/logs_s2_v2", "output/slicer_opt"):  # a GPT-SoVITS package
        (root / skipped).mkdir(parents=True)
        (root / skipped / "G_2333.pth").write_bytes(b"PK")
    out = browse.scan(str(root))
    assert [(f["dir"], f["name"], f["kind"]) for f in out["files"]] == [
        ("1/2/3/4/5/6", "deep_enough.wav", "audio"), ("GPT_weights_v2", "A-e10.ckpt", "gpt"), ("voices/A", "hi.wav", "audio")]
    assert out["truncated"] is False
    assert out["skipped"] == ["GPT_SoVITS", "logs", "output", "runtime"]  # said, not dropped quietly
    assert out["too_deep"] == ["1/2/3/4/5/6/7"]
    with pytest.raises(ValueError):
        browse.scan(str(fresh / "nope"))


def test_browse_scan_skips_program_folder_names_only_inside_a_package(fresh):
    """A user's own ``output`` or ``GPT_SoVITS`` folder is scanned; the chosen folder always is."""
    root = fresh / "我的音色"
    configure(import_folders=[root])
    for folder in ("output/派蒙", "GPT_SoVITS/可莉", "temp", "venv"):
        (root / folder).mkdir(parents=True)
    (root / "output" / "派蒙" / "a.wav").write_bytes(b"x")
    (root / "GPT_SoVITS" / "可莉" / "k-e8.ckpt").write_bytes(b"PK")
    (root / "temp" / "t.wav").write_bytes(b"x")
    (root / "venv" / "r.wav").write_bytes(b"x")  # Python environments are never voices
    out = browse.scan(str(root))
    assert [f["dir"] for f in out["files"]] == ["GPT_SoVITS/可莉", "output/派蒙", "temp"]
    assert out["skipped"] == ["venv"]
    assert [f["name"] for f in browse.scan(str(root / "output"))["files"]] == ["a.wav"]
    assert browse.is_package(["webui.py", "tools"]) and browse.is_package(["SoVITS_weights_v2"])
    assert not browse.is_package(["GPT_SoVITS", "output"])
    assert browse.skip_folder("Output", True) and not browse.skip_folder("Output", False)


def test_browse_scan_stops_at_the_folder_limit(fresh, monkeypatch):
    root = fresh / "many"
    for i in range(4):
        (root / f"d{i}").mkdir(parents=True)
        (root / f"d{i}" / "a.wav").write_bytes(b"x")
    configure(import_folders=[root])
    monkeypatch.setattr(browse, "MAX_FOLDERS", 3)
    out = browse.scan(str(root))
    assert out["truncated"] is True and len(out["files"]) == 2


def test_import_preview_serves_audio_files_only(fresh):
    clip = fresh / "clips" / "a.wav"
    clip.parent.mkdir()
    clip.write_bytes(b"x")
    (fresh / "clips" / "w.pth").write_bytes(b"PK")
    configure(import_folders=[fresh / "clips"])
    assert browse.audio_file(str(clip)) == str(clip)
    for bad in (str(fresh / "clips" / "w.pth"), str(fresh / "clips" / "nope.wav"), str(fresh / "clips"), ""):
        with pytest.raises(ValueError):
            browse.audio_file(bad)


def test_status_counts_characters_per_library(fresh):
    lib = fresh / "voices"
    make_char(lib, "阿罗娜")
    make_char(lib, "普拉娜")
    (fresh / "GPT-SoVITS").mkdir()
    configure(storage=lib, import_folders=[fresh / "GPT-SoVITS"])
    status = server._status_payload(local=True)
    assert status["format"] == server.API_FORMAT and status["local"] is True
    assert status["storage"] == paths.norm(str(lib))
    assert status["settings_file"] == app_config.path()
    assert status["import_folders"] == [paths.norm(str(fresh / "GPT-SoVITS"))]
    counts = {l["path"]: l["characters"] for l in status["libraries"]}
    assert counts[paths.norm(str(lib))] == 2
    assert [p["id"] for p in status["pretrained"]] == list(paths.PRETRAINED_IDS)
    assert set(status["dependencies"]) == {"ja", "zh", "en"}


def test_writes_are_refused_from_other_computers():
    server._require_local(types.SimpleNamespace(remote="127.0.0.1"))
    with pytest.raises(web.HTTPForbidden):
        server._require_local(types.SimpleNamespace(remote="192.168.1.20"))
