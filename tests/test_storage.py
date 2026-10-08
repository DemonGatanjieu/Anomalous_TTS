"""Where characters are kept: only what the user wrote in the settings file, never a request."""

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


def test_no_route_changes_where_folders_are(fresh):  # noqa: F811
    routes = web.RouteTableDef()
    server.register(type("PromptServer", (), {"routes": routes})())
    posts = {r.path for r in routes if r.method == "POST"}
    assert "/anomalous_tts/storage" not in posts and "/anomalous_tts/libraries" not in posts
    assert not any("import_folders" in p for p in posts)


def test_unusable_entries_in_the_settings_file_are_ignored(fresh):  # noqa: F811
    configure()
    with open(app_config.path(), "w", encoding="utf-8") as f:
        json.dump({"format": 1, "storage": "", "libraries": "voices", "import_folders": ["relative", 3, " "]}, f)
    config = app_config.load()
    assert (config["storage"], config["libraries"], config["import_folders"]) == (None, [], [])
    assert paths.storage() == paths.norm(paths._default_library())  # not ComfyUI's working folder


def test_saving_keeps_what_the_user_wrote(fresh):  # noqa: F811
    configure(storage=fresh / "voices", import_folders=[fresh / "GPT-SoVITS"])
    with open(app_config.path(), encoding="utf-8") as f:
        data = json.load(f)
    data["note"] = "mine"
    with open(app_config.path(), "w", encoding="utf-8") as f:
        json.dump(data, f)
    app_config.add("pretrained", "D:/GPT-SoVITS/GPT_SoVITS")
    with open(app_config.path(), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == {**data, "pretrained": ["D:/GPT-SoVITS/GPT_SoVITS"]}

    with open(app_config.path(), "w", encoding="utf-8") as f:
        f.write('{"storage": "D:/voices",')  # a typo while editing by hand
    with pytest.raises(ValueError, match="设置文件读不了"):
        app_config.add("pretrained", "D:/other")
    with open(app_config.path(), encoding="utf-8") as f:
        assert f.read() == '{"storage": "D:/voices",'  # not replaced


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
