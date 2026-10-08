"""Nothing read from a data file runs as code, and the G2PWModel download is the pinned file.

ComfyUI-Manager's review asked for this: g2pW's config.py comes inside a downloaded archive, and
folders added from the UI decide which copy is read.
"""

import hashlib
import importlib.util
import os
import re
import zipfile
from pathlib import Path

import pytest

from Anomalous_TTS.core import paths

ROOT = Path(__file__).resolve().parent.parent


def _g2pw_utils():
    spec = importlib.util.spec_from_file_location("g2pw_utils", ROOT / "vendor/gpt_sovits/text/g2pw/utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # our own vendored file, not a downloaded one
    return module


def test_g2pw_config_is_read_as_data(tmp_path):
    marker = tmp_path / "ran.txt"
    config = tmp_path / "config.py"
    config.write_text(
        "model_source = 'bert-base-chinese'\n"
        "window_size = 32\n"
        "param_pos = {'weight': 0.1}\n"
        f"open({str(marker)!r}, 'w').write('x')\n"
        "computed = window_size * 2\n",
        encoding="utf-8",
    )
    settings = _g2pw_utils().load_config(str(config), use_default=True)
    assert not marker.exists()  # the file's code never ran
    assert settings.model_source == "bert-base-chinese" and settings.window_size == 32
    assert settings.param_pos == {"weight": 0.1}
    assert not hasattr(settings, "computed")  # only literals are read
    assert settings.use_mask is True  # defaults fill the rest, as before


def _fake_download(content: bytes):
    def download(url, dest):
        assert paths.G2PW_REVISION in url and "/master/" not in url
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        Path(dest).write_bytes(content)
    return download


def _zip_bytes(tmp_path) -> bytes:
    archive = tmp_path / "src.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ("g2pW.onnx", "config.py", "POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt"):
            zf.writestr(f"G2PWModel_1.1/{name}", "x")
    return archive.read_bytes()


def test_g2pw_download_with_another_checksum_extracts_nothing(tmp_path, monkeypatch):
    target = tmp_path / "pretrained"
    monkeypatch.setattr(paths, "_download_target", lambda: str(target))
    monkeypatch.setattr(paths, "_download", _fake_download(_zip_bytes(tmp_path)))
    with pytest.raises(RuntimeError, match="SHA-256"):
        paths._download_g2pw()
    assert sorted(os.listdir(target)) == []  # no model, no archive left behind


def test_g2pw_download_with_the_pinned_checksum_is_extracted(tmp_path, monkeypatch):
    target = tmp_path / "pretrained"
    content = _zip_bytes(tmp_path)
    monkeypatch.setattr(paths, "_download_target", lambda: str(target))
    monkeypatch.setattr(paths, "_download", _fake_download(content))
    monkeypatch.setattr(paths, "G2PW_SHA256", hashlib.sha256(content).hexdigest())
    paths._download_g2pw()
    assert sorted(os.listdir(target)) == ["G2PWModel"]
    assert (target / "G2PWModel" / "config.py").is_file()


def test_no_code_loading_from_files():
    """No unpickling, module loading from a path, or eval() anywhere in the node's code."""
    pattern = re.compile(r"pickle\.loads?\(|exec_module\(|spec_from_file_location\(|(?<![\w.])eval\(|(?<![\w.])exec\(")
    found = []
    for path in ROOT.rglob("*.py"):
        if "tests" in path.relative_to(ROOT).parts or "tools" in path.relative_to(ROOT).parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line.split("#", 1)[0]):  # comments may name them
                found.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert found == []
