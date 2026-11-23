"""Chinese still loads when the installed setuptools is too old for this Python (a viewer's report:
Python 3.13, an old setuptools, ``pkgutil.ImpImporter`` missing as soon as jieba was imported)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# What importing pkg_resources does with a setuptools from before Python 3.12 support.
OLD_PKG_RESOURCES = "raise AttributeError(\"module 'pkgutil' has no attribute 'ImpImporter'\")\n"

SCRIPT = """
import importlib.util, sys
sys.path.insert(0, {old!r})
spec = importlib.util.spec_from_file_location("deps", {deps!r})
deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deps)
if {guard}:
    deps.hide_broken_pkg_resources()
import jieba
print("words", len(jieba.lcut("今天天气很好，我们去公园散步")))
"""


@pytest.mark.skipif(importlib.util.find_spec("jieba") is None, reason="jieba is not installed")
@pytest.mark.parametrize("guard", [False, True])
def test_jieba_loads_with_an_old_setuptools(tmp_path, guard):
    (tmp_path / "pkg_resources").mkdir()
    (tmp_path / "pkg_resources" / "__init__.py").write_text(OLD_PKG_RESOURCES, encoding="utf-8")
    script = SCRIPT.format(old=str(tmp_path), deps=str(ROOT / "core" / "dependencies.py"), guard=guard)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if guard:
        assert run.returncode == 0, run.stderr
        assert int(run.stdout.split()[-1]) > 1  # the dictionary was read: the sentence is split into words
        assert "setuptools" in run.stderr  # one line telling how to upgrade
    else:
        assert run.returncode != 0 and "ImpImporter" in run.stderr  # the failure the viewer saw
