"""One storage place: changing it, moving characters there, and reading places left behind."""

import os
import time

import folder_paths
import pytest

from Anomalous_TTS.core import app_config, characters, importer, paths, storage

from test_planner import make_char
from test_setup import fresh  # noqa: F401  (fixture)


def _wait_move(seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        job = storage.state()
        if job and job["state"] != "moving":
            return job
        time.sleep(0.02)
    raise AssertionError("move did not finish")


def _names():
    return set(characters.scan(max_age=0))


def test_storage_defaults_to_models_folder_and_is_remembered(fresh):  # noqa: F811
    assert paths.storage() == paths.norm(paths._default_library())
    place = fresh / "voices"
    make_char(place, "阿罗娜")
    storage.change(str(place), move=False)
    assert paths.storage() == paths.norm(str(place))
    assert app_config.load()["storage"] == paths.norm(str(place))
    assert "阿罗娜" in _names()
    assert [lib["path"] for lib in paths.libraries() if lib["storage"]] == [paths.norm(str(place))]

    registered = folder_paths.folder_names_and_paths[paths.CATEGORY][0]
    registered[:] = registered[:1]  # next start
    paths.register()
    assert "阿罗娜" in _names()


def test_change_without_moving_keeps_reading_the_old_place(fresh):  # noqa: F811
    old, new = fresh / "old", fresh / "new"
    make_char(old, "阿罗娜")
    new.mkdir()
    storage.change(str(old), move=False)
    storage.change(str(new), move=False)
    assert paths.storage() == paths.norm(str(new))
    assert app_config.load()["libraries"] == [paths.norm(str(old))]
    assert "阿罗娜" in _names()
    assert {lib["path"]: lib["source"] for lib in paths.libraries()}[paths.norm(str(old))] == "app"

    paths.forget_library(str(old))  # "remove": stop reading it, files stay
    assert "阿罗娜" not in _names()
    assert (old / "阿罗娜").is_dir()
    with pytest.raises(ValueError):
        paths.forget_library(paths._default_library())


def test_change_with_move_moves_every_character(fresh):  # noqa: F811
    old, new = fresh / "old", fresh / "new"
    make_char(old / "阿罗娜", "日配")
    make_char(old / "阿罗娜", "中配")
    make_char(old, "普拉娜")
    (old / "notes").mkdir()  # not a character: stays
    new.mkdir()
    storage.change(str(old), move=False)
    storage.change(str(new), move=True)
    job = _wait_move()
    assert (job["state"], job["done"], job["total"]) == ("done", 2, 2)
    assert sorted(os.listdir(new)) == ["普拉娜", "阿罗娜"]
    assert sorted(os.listdir(old)) == ["notes"]
    assert _names() == {"阿罗娜/中配", "阿罗娜/日配", "普拉娜"}
    assert all(paths.is_inside(c.folder, str(new)) for c in characters.scan(max_age=0).values())
    assert app_config.load()["libraries"] == []  # nothing left to read in the old place


def test_move_stops_at_a_name_clash_and_keeps_that_character(fresh):  # noqa: F811
    old, new = fresh / "old", fresh / "new"
    make_char(old, "阿罗娜")
    make_char(old, "普拉娜")
    (new / "阿罗娜").mkdir(parents=True)  # moved in name order: 普拉娜 first, then this one clashes
    storage.change(str(old), move=False)
    storage.change(str(new), move=True)
    job = _wait_move()
    assert job["state"] == "error" and "阿罗娜" in job["error"]
    assert job["moved"] == ["普拉娜"]
    assert (old / "阿罗娜" / "GPT_weights_v2").is_dir()  # untouched
    assert app_config.load()["libraries"] == [paths.norm(str(old))]  # still read


def test_copy_across_disks_counts_bytes_and_removes_the_original(fresh):  # noqa: F811
    src, dst_root = fresh / "a", fresh / "b"
    make_char(src, "阿罗娜")
    dst_root.mkdir()
    storage._job = {"bytes_done": 0}
    storage._move_one(str(src / "阿罗娜"), str(dst_root / "阿罗娜"), same_disk=False)
    assert not (src / "阿罗娜").exists()
    assert (dst_root / "阿罗娜" / "all.list").is_file()
    assert storage._job["bytes_done"] > 0
    assert os.listdir(dst_root / storage.STAGING) == []


def test_bad_storage_places_are_refused(fresh):  # noqa: F811
    place = fresh / "voices"
    place.mkdir()
    storage.change(str(place), move=False)
    with pytest.raises(ValueError, match="不存在"):
        storage.change(str(fresh / "nope"), move=False)
    with pytest.raises(ValueError, match="已经"):
        storage.change(str(place), move=False)
    (place / "inner").mkdir()
    with pytest.raises(ValueError, match="里面"):
        storage.change(str(place / "inner"), move=True)


def test_a_missing_storage_place_is_never_swapped_for_another(fresh, tmp_path):
    place = tmp_path / "usb"
    place.mkdir()
    storage.change(str(place), move=False)
    place.rmdir()  # unplugged
    assert paths.storage() == paths.norm(str(place))
    home = next(lib for lib in paths.libraries() if lib["storage"])
    assert home["exists"] is False and home["writable"] is False
    with pytest.raises(ValueError, match="存放位置不能写入"):
        importer.start_upload("a.wav", 10, None)
    place.mkdir()  # plugged back in: found again without a restart
    assert next(lib for lib in paths.libraries() if lib["storage"])["writable"] is True
