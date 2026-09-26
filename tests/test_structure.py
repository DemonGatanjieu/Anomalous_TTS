"""Structure rules from AGENTS.md section 0: every source file is in the architecture map,
every package module is imported from __init__.py, and files stay under the size threshold."""

import ast
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["*.py", "core/*.py", "tools/*.py", "web/*.js"]
PACKAGE = ["*.py", "core/*.py"]
SIZE_LIMIT = 600


def _files(patterns):
    return [p for pattern in patterns for p in ROOT.glob(pattern) if p.name != "__init__.py" or p.parent == ROOT]


def _imports(path):
    """Package modules that ``path`` imports with relative imports, anywhere in the file."""
    base = path.parent
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        target = base
        for _ in range(node.level - 1):
            target = target.parent
        for part in (node.module or "").split(".") if node.module else []:
            target = target / part
        if target.with_suffix(".py").is_file():
            yield target.with_suffix(".py")
        for alias in node.names:
            if (target / f"{alias.name}.py").is_file():
                yield target / f"{alias.name}.py"


def test_every_source_file_is_in_the_architecture_map():
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    missing = sorted(p.relative_to(ROOT).as_posix() for p in _files(SOURCES) if p.name not in text)
    assert not missing, "add a line for these to ARCHITECTURE.md: " + ", ".join(missing)


def test_every_package_module_is_reachable_from_init():
    package = set(_files(PACKAGE)) | {ROOT / "__init__.py"}
    reached, todo = set(), [ROOT / "__init__.py"]
    while todo:
        path = todo.pop()
        if path not in reached:
            reached.add(path)
            todo.extend(p for p in _imports(path) if p in package)  # vendor/ is upstream code
    unused = sorted(p.relative_to(ROOT).as_posix() for p in _files(PACKAGE) if p not in reached)
    assert not unused, "not imported from __init__.py; remove or wire in: " + ", ".join(unused)


def test_files_over_the_size_threshold_are_reported():
    large = [(p.relative_to(ROOT).as_posix(), n) for p in _files(SOURCES)
             if (n := len(p.read_text(encoding="utf-8").splitlines())) > SIZE_LIMIT]
    for name, lines in large:
        warnings.warn(f"{name} has {lines} lines (over {SIZE_LIMIT}); see AGENTS.md before adding to it")
