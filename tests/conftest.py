"""Test setup: import the package without ComfyUI.

- Stub modules stand in for ComfyUI's ``folder_paths`` and ``comfy``.
- The package is imported as ``Anomalous_TTS`` whatever the checkout folder is called.
- Tests that need model files are skipped unless ANOMALOUS_TTS_ASSETS points to a folder with:
  chinese-hubert-base/, chinese-roberta-wwm-ext-large/, G2PWModel/, and a character
  folder ``character/`` (GPT .ckpt, SoVITS .pth, reference .wav + text). See tests/README.md.
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(tempfile.mkdtemp(prefix="atts_models_"))


def _stub_comfy():
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.models_dir = str(MODELS)
        fp.folder_names_and_paths = {}

        def add_model_folder_path(name, path, is_default=False):
            paths = fp.folder_names_and_paths.setdefault(name, ([], set()))[0]
            if path not in paths:
                paths.insert(0, path) if is_default else paths.append(path)

        fp.add_model_folder_path = add_model_folder_path
        fp.get_folder_paths = lambda name: list(fp.folder_names_and_paths[name][0])
        sys.modules["folder_paths"] = fp
    if "comfy" not in sys.modules:
        import torch

        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: torch.device("cpu")
        mm.should_use_fp16 = lambda *a, **k: False
        mm.throw_exception_if_processing_interrupted = lambda: None
        mm.unload_all_models = lambda: None
        utils = types.ModuleType("comfy.utils")

        class ProgressBar:
            def __init__(self, total):
                self.total = total

            def update_absolute(self, value, total=None):
                pass

        utils.ProgressBar = ProgressBar
        comfy.model_management, comfy.utils = mm, utils
        sys.modules.update({"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils})


def _import_package():
    if "Anomalous_TTS" in sys.modules:
        return sys.modules["Anomalous_TTS"]
    spec = importlib.util.spec_from_file_location(
        "Anomalous_TTS", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["Anomalous_TTS"] = module
    spec.loader.exec_module(module)
    return module


_stub_comfy()
_import_package()


@pytest.fixture
def assets():
    path = os.environ.get("ANOMALOUS_TTS_ASSETS")
    if not path or not Path(path).is_dir():
        pytest.skip("ANOMALOUS_TTS_ASSETS not set (model files needed)")
    return Path(path)


class AssetResources:
    """engine.Resources backed by the ANOMALOUS_TTS_ASSETS folder."""

    def __init__(self, root: Path):
        self.root = root

    def hubert_dir(self):
        return str(self.root / "chinese-hubert-base")

    def roberta_dir(self):
        return str(self.root / "chinese-roberta-wwm-ext-large")

    def g2pw_dir(self):
        return str(self.root / "G2PWModel")

    def sv_path(self):
        return str(self.root / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt")

    def english_dirs(self):
        return str(self.root / "en_dict"), str(self.root / "en_dict"), str(self.root / "nltk_data")


@pytest.fixture(scope="session")
def engine():
    path = os.environ.get("ANOMALOUS_TTS_ASSETS")
    if not path or not Path(path).is_dir():
        pytest.skip("ANOMALOUS_TTS_ASSETS not set (model files needed)")
    import torch

    from Anomalous_TTS.core.engine import Engine

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return Engine(AssetResources(Path(path)), torch.device("cpu"), torch.float32)
