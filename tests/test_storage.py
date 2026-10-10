"""Where folders come from: only what the user wrote in the settings file, never a request."""

import json

import folder_paths
import pytest
from aiohttp import web

from Anomalous_TTS import server
from Anomalous_TTS.core import app_config, characters, importer, paths

from test_planner import make_char
from test_setup import configure, fresh  # noqa: F401  (fixture)


def _names():
    return set(characters.scan(max_age=0))


def test_storage_comes_from_the_settings_file(fresh):  # noqa: F811
    assert paths.storage() == paths.norm(paths._default_library())
    place = fresh / "voices"
    make_char(place, "阿罗娜")
    configure(storage=place)
    assert paths.storage() == paths.norm(str(place))
    assert "阿罗娜" in _names()  # no restart needed
    assert [lib["path"] for lib in paths.libraries() if lib["storage"]] == [paths.norm(str(place))]

    registered = folder_paths.folder_names_and_paths[paths.CATEGORY][0]
    registered[:] = registered[:1]  # next start
    paths.register()
    assert "阿罗娜" in _names()


def test_more_libraries_come_from_the_settings_file(fresh):  # noqa: F811
    old, new = fresh / "old", fresh / "new"
    make_char(old, "阿罗娜")
    new.mkdir()
    configure(storage=new, libraries=[old])
    assert paths.storage() == paths.norm(str(new))
    assert "阿罗娜" in _names()
    assert {lib["path"]: lib["source"] for lib in paths.libraries()}[paths.norm(str(old))] == "app"


def test_no_route_takes_a_folder_or_a_file_on_this_computer(fresh):  # noqa: F811
    routes = web.RouteTableDef()
    server.register(type("PromptServer", (), {"routes": routes})())
    assert {(r.method, r.path) for r in routes} == {
        ("GET", "/anomalous_tts/characters"), ("GET", "/anomalous_tts/audio"), ("GET", "/anomalous_tts/reference_text"),
        ("GET", "/anomalous_tts/status"),
        ("POST", "/anomalous_tts/settings"), ("POST", "/anomalous_tts/pretrained/download"),
        ("POST", "/anomalous_tts/import/upload"), ("POST", "/anomalous_tts/import/inspect"),
        ("POST", "/anomalous_tts/import/commit"), ("POST", "/anomalous_tts/import/discard"),
    }  # no storage, libraries, pretrained/source, browse or import/preview
    clip = fresh / "clip.wav"
    clip.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="upload"):
        importer.inspect([{"path": str(clip)}])  # a file is imported only by uploading it


def test_a_fresh_install_can_import_at_once(fresh, monkeypatch):  # noqa: F811
    models = fresh / "models"
    monkeypatch.setattr(folder_paths, "models_dir", str(models))
    paths.register()  # ComfyUI starting with no models/gpt_sovits yet
    home = next(lib for lib in paths.libraries() if lib["storage"])
    assert home["path"] == paths.norm(str(models / paths.CATEGORY))
    assert home["exists"] is True and home["writable"] is True


def test_unusable_entries_in_the_settings_file_are_ignored(fresh):  # noqa: F811
    configure()
    with open(app_config.path(), "w", encoding="utf-8") as f:
        json.dump({"format": 1, "storage": "", "libraries": "voices", "pretrained": ["relative", 3, " "]}, f)
    config = app_config.load()
    assert (config["storage"], config["libraries"], config["pretrained"]) == (None, [], [])
    assert paths.storage() == paths.norm(paths._default_library())  # not ComfyUI's working folder


def test_a_missing_storage_place_is_never_swapped_for_another(fresh, tmp_path):  # noqa: F811
    place = tmp_path / "usb"
    place.mkdir()
    configure(storage=place)
    place.rmdir()  # unplugged
    assert paths.storage() == paths.norm(str(place))
    home = next(lib for lib in paths.libraries() if lib["storage"])
    assert home["exists"] is False and home["writable"] is False
    with pytest.raises(ValueError, match="存放位置不能写入"):
        importer.start_upload("a.wav", 10, None)
    place.mkdir()  # plugged back in: found again without a restart
    assert next(lib for lib in paths.libraries() if lib["storage"])["writable"] is True
