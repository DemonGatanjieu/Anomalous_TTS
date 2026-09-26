"""ARCHITECTURE.md doubles as the file directory: every source file has a line there."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["*.py", "core/*.py", "tools/*.py", "web/*.js"]


def test_every_source_file_is_in_the_architecture_map():
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    files = [p for pattern in SOURCES for p in ROOT.glob(pattern) if p.name != "__init__.py" or p.parent == ROOT]
    missing = sorted(p.relative_to(ROOT).as_posix() for p in files if p.name not in text)
    assert not missing, "add a line for these to ARCHITECTURE.md: " + ", ".join(missing)
