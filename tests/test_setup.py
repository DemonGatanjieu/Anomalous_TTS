"""Setup: pretrained sources, downloads, status (where folders come from: test_storage.py)."""

import json
import os
import time
import types

import folder_paths
import pytest
from aiohttp import web

from Anomalous_TTS import server
from Anomalous_TTS.core import app_config, characters, downloads, paths

from test_planner import make_char


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """Empty user directory and only the default library, restored afterwards."""
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path / "user"))
    registered = folder_paths.folder_names_and_paths[paths.CATEGORY][0]
    saved = list(registered)
    registered[:] = [p for p in registered if paths._key(p) == paths._key(paths._default_library())]
    characters.invalidate()
    yield tmp_path
    registered[:] = saved
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
    configure(pretrained=[pkg])  # written by the user; found without a restart
    assert paths.is_inside(paths.locate("hubert"), str(pkg))
    assert paths.is_inside(paths._find_en_dict(), str(pkg / "text"))
    assert paths.pretrained_sources() == [paths.norm(str(pkg))]

    configure()
    assert paths.locate("hubert") is None


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


def test_status_counts_characters_per_library(fresh):
    lib = fresh / "voices"
    make_char(lib, "阿罗娜")
    make_char(lib, "普拉娜")
    configure(storage=lib)
    status = server._status_payload(local=True)
    assert status["format"] == server.API_FORMAT and status["local"] is True
    assert status["storage"] == paths.norm(str(lib))
    assert status["settings_file"] == app_config.path()
    counts = {l["path"]: l["characters"] for l in status["libraries"]}
    assert counts[paths.norm(str(lib))] == 2
    assert [p["id"] for p in status["pretrained"]] == list(paths.PRETRAINED_IDS)
    assert set(status["dependencies"]) == {"ja", "zh", "en"}


def test_writes_are_refused_from_other_computers():
    server._require_local(types.SimpleNamespace(remote="127.0.0.1"))
    with pytest.raises(web.HTTPForbidden):
        server._require_local(types.SimpleNamespace(remote="192.168.1.20"))
