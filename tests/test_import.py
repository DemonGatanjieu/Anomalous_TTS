"""Import: chunked upload, inspect, commit (new character / add files), rollback."""

import os

import pytest

from Anomalous_TTS.core import characters, importer, settings, storage

from test_planner import make_char, wav
from test_setup import fresh  # noqa: F401  (fixture)


@pytest.fixture
def lib(fresh):  # noqa: F811
    folder = fresh / "voices"
    folder.mkdir()
    storage.change(str(folder), move=False)  # imports go to the storage place
    return folder


@pytest.fixture
def pkg(fresh):  # noqa: F811
    """Flat GPT-SoVITS package output, as users have it."""
    root = fresh / "GPT-SoVITS"
    (root / "GPT_weights_v2").mkdir(parents=True)
    (root / "SoVITS_weights_v2").mkdir()
    (root / "GPT_weights_v2" / "ALuoNa-e15.ckpt").write_bytes(b"PK" + b"\0" * 100)
    (root / "SoVITS_weights_v2" / "ALuoNa_e16_s224.pth").write_bytes(b"01" + b"\0" * 100)
    (root / "SoVITS_weights_v2" / "New_e8_s80.pth").write_bytes(b"04" + b"\0" * 100)  # v4: unsupported
    wav(root / "ref" / "Talk_3.wav", 4)
    wav(root / "ref" / "Happy.wav", 5)
    wav(root / "ref" / "short.wav", 1)
    (root / "ref" / "Happy.txt").write_text("やった！", encoding="utf-8")
    (root / "ref" / "all.list").write_text("/x/Talk_3.wav|s|JA|通常授業！\n", encoding="utf-8")
    return root


def upload(path, library=None, chunk=40_000):
    data = path.read_bytes()
    upload_id = importer.start_upload(path.name, len(data), library)
    for offset in range(0, len(data), chunk):
        importer.write_chunk(upload_id, offset, data[offset:offset + chunk])
    return {"upload": upload_id}


def local(path):
    return {"path": str(path)}


def test_chunked_upload_checks_offsets_and_size(lib, pkg):
    data = (pkg / "ref" / "Talk_3.wav").read_bytes()
    upload_id = importer.start_upload("Talk_3.wav", len(data))
    assert importer.write_chunk(upload_id, 0, data[:100]) == 100
    with pytest.raises(importer.Conflict) as e:
        importer.write_chunk(upload_id, 0, data[:100])
    assert e.value.data == {"received": 100}
    with pytest.raises(ValueError, match="还没有上传完"):
        importer.inspect([{"upload": upload_id}])
    importer.write_chunk(upload_id, 100, data[100:])
    with pytest.raises(ValueError, match="多"):
        importer.write_chunk(upload_id, len(data), b"x")
    assert importer.inspect([{"upload": upload_id}])["files"][0]["seconds"] == 4.0

    with pytest.raises(ValueError, match="不支持"):
        importer.start_upload("notes.md", 10)
    importer.discard([upload_id])
    assert not os.listdir(lib / importer.STAGING)


def test_inspect_suggests_name_reference_and_text(lib, pkg):
    files = [local(pkg / "GPT_weights_v2" / "ALuoNa-e15.ckpt"), upload(pkg / "SoVITS_weights_v2" / "ALuoNa_e16_s224.pth"),
             upload(pkg / "ref" / "Talk_3.wav"), local(pkg / "ref" / "Happy.wav"), local(pkg / "ref" / "Happy.txt"),
             local(pkg / "ref" / "all.list"), local(pkg / "ref" / "short.wav"), local(pkg / "SoVITS_weights_v2" / "New_e8_s80.pth")]
    out = importer.inspect(files)
    by_name = {f["name"]: f for f in out["files"]}
    assert out["suggested"] == {"name": "ALuoNa", "language": "ja", "reference": 2}
    assert by_name["ALuoNa_e16_s224.pth"]["version"] == "v2" and by_name["ALuoNa_e16_s224.pth"]["supported"]
    assert by_name["New_e8_s80.pth"]["supported"] is False
    assert (by_name["Talk_3.wav"]["text"], by_name["Talk_3.wav"]["text_source"]) == ("通常授業！", "list")
    assert (by_name["Happy.wav"]["text"], by_name["Happy.wav"]["text_source"]) == ("やった！", "txt")
    assert any("short.wav" in p for p in out["problems"]) and any("v4" in p for p in out["problems"])


def test_commit_new_character_copies_files_and_writes_settings(lib, pkg):
    gpt = pkg / "GPT_weights_v2" / "ALuoNa-e15.ckpt"
    files = [local(gpt), upload(pkg / "SoVITS_weights_v2" / "ALuoNa_e16_s224.pth"), upload(pkg / "ref" / "Talk_3.wav"),
             local(pkg / "ref" / "Happy.wav")]
    c = importer.commit({"library": str(lib), "character": "阿罗娜", "files": files, "settings": {
        "aliases": ["阿罗娜"], "language": "ja",
        "reference": {"file": 2, "text": "通常授業！", "language": "ja"},
        "emotions": {"开心": {"file": 3, "text": "やった！"}}}})

    assert c["name"] == "阿罗娜"
    assert c["gpt"] == ["GPT_weights/ALuoNa-e15.ckpt"] and c["sovits"] == ["SoVITS_weights/ALuoNa_e16_s224.pth"]
    assert c["reference"]["audio"] == "参考音频/Talk_3.wav" and c["reference"]["source"] == "settings"
    assert c["emotions"]["开心"]["audio"] == "参考音频/Happy.wav"
    assert gpt.is_file()  # originals stay
    assert os.listdir(lib / importer.STAGING) == []  # uploads and work folder are gone

    with pytest.raises(importer.Conflict):
        importer.commit({"library": str(lib), "character": "阿罗娜", "files": [local(gpt)]})


def test_commit_new_character_needs_both_weights_and_a_valid_name(lib, pkg):
    with pytest.raises(ValueError, match="至少"):
        importer.commit({"library": str(lib), "character": "X", "files": [local(pkg / "GPT_weights_v2" / "ALuoNa-e15.ckpt")]})
    for bad in ("a/b", "CON", "x.", "..", ""):
        with pytest.raises(ValueError):
            importer.safe_name(bad)
    with pytest.raises(ValueError, match="角色库"):
        importer.commit({"library": str(pkg), "character": "X", "files": [local(pkg / "GPT_weights_v2" / "ALuoNa-e15.ckpt")]})


def test_commit_adds_files_to_an_existing_character(lib, pkg):
    make_char(lib, "普拉娜", settings={"aliases": ["普拉娜"], "emotions": {"平静": {"audio": "ref/b.wav"}}})
    c = importer.commit({"target": "普拉娜", "files": [local(pkg / "ref" / "Happy.wav")],
                         "settings": {"emotions": {"开心": {"file": 0}}}})
    assert set(c["emotions"]) >= {"平静", "开心"}
    saved = settings.load(str(lib / "普拉娜"))
    assert saved["aliases"] == ["普拉娜"] and saved["emotions"]["开心"] == {"audio": "参考音频/Happy.wav"}

    with pytest.raises(importer.Conflict):
        importer.commit({"target": "普拉娜", "files": [local(pkg / "ref" / "Happy.wav")]})


def test_failed_commit_leaves_the_library_as_it_was(lib, pkg, monkeypatch):
    make_char(lib, "普拉娜")
    before = sorted(str(p) for p in lib.rglob("*") if importer.STAGING not in p.parts)
    audio = upload(pkg / "ref" / "Talk_3.wav")

    def broken_save(folder, data):
        raise OSError("磁盘满了")

    monkeypatch.setattr(settings, "save", broken_save)
    with pytest.raises(OSError):
        importer.commit({"library": str(lib), "character": "阿罗娜", "settings": {"reference": {"file": 2}},
                         "files": [local(pkg / "GPT_weights_v2" / "ALuoNa-e15.ckpt"),
                                   local(pkg / "SoVITS_weights_v2" / "ALuoNa_e16_s224.pth"), audio]})
    with pytest.raises(OSError):
        importer.commit({"target": "普拉娜", "files": [audio], "settings": {"emotions": {"开心": {"file": 0}}}})

    assert sorted(str(p) for p in lib.rglob("*") if importer.STAGING not in p.parts) == before
    assert importer.inspect([audio])["files"][0]["seconds"] == 4.0  # the upload is still there for a retry
    characters.invalidate()
    assert "阿罗娜" not in characters.scan(max_age=0)


def test_bad_settings_are_rejected_before_anything_is_written(lib, pkg):
    with pytest.raises(ValueError, match="序号"):
        importer.commit({"library": str(lib), "character": "阿罗娜",
                         "files": [local(pkg / "GPT_weights_v2" / "ALuoNa-e15.ckpt"),
                                   local(pkg / "SoVITS_weights_v2" / "ALuoNa_e16_s224.pth")],
                         "settings": {"reference": {"file": 5}}})
    with pytest.raises(ValueError, match="language"):
        importer.commit({"library": str(lib), "character": "阿罗娜",
                         "files": [local(pkg / "GPT_weights_v2" / "ALuoNa-e15.ckpt"),
                                   local(pkg / "SoVITS_weights_v2" / "ALuoNa_e16_s224.pth")],
                         "settings": {"language": "fr"}})
    assert not (lib / "阿罗娜").exists()
    assert set(os.listdir(lib)) <= {importer.STAGING}


def test_lab_files_and_file_names_give_reference_text(lib, pkg):
    ref = pkg / "ref"
    wav(ref / "Lab_clip.wav", 4)
    (ref / "Lab_clip.lab").write_text("ラボのテキスト", encoding="utf-8")
    wav(ref / "【开心】先生、おはようございます.wav", 4)
    wav(ref / "Arona_Talk_9.wav", 4)
    out = importer.inspect([local(ref / "Lab_clip.wav"), local(ref / "Lab_clip.lab"),
                            local(ref / "【开心】先生、おはようございます.wav"), local(ref / "Arona_Talk_9.wav")])
    by_name = {f["name"]: f for f in out["files"]}
    assert (by_name["Lab_clip.wav"]["text"], by_name["Lab_clip.wav"]["text_source"]) == ("ラボのテキスト", "lab")
    assert by_name["Lab_clip.lab"]["kind"] == "text"
    named = by_name["【开心】先生、おはようございます.wav"]
    assert (named["text"], named["text_source"], named["language"]) == ("先生、おはようございます", "filename", "ja")
    assert (by_name["Arona_Talk_9.wav"]["text"], by_name["Arona_Talk_9.wav"]["text_source"]) == ("", None)


def test_lab_sidecar_is_used_when_generating(lib):
    folder = make_char(lib, "普拉娜")
    (folder / "ref" / "a.lab").write_text("ラボ", encoding="utf-8")
    c = characters.scan(max_age=0)["普拉娜"]
    assert c.reference_text("ref/a.wav") == ("ラボ", None)
    assert characters.text_from_filename("ref/b.wav") == ""  # never guessed outside the import form


def test_text_from_filename_rules():
    assert characters.text_from_filename("x/Hello there, teacher!.wav") == "Hello there, teacher!"
    assert characters.text_from_filename("先生おはよう.开心.wav") == "先生おはよう"
    assert characters.text_from_filename("[1][快乐]你好啊朋友.wav") == "你好啊朋友"
    assert characters.text_from_filename("开心.wav") == ""
    assert characters.text_from_filename("Arona_Academy_Talk_3.wav") == ""
