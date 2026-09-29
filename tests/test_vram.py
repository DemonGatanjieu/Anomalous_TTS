"""Our models give VRAM back when ComfyUI needs more than is free (nodes._hook_free_memory)."""

import torch

import comfy.model_management as mm
from Anomalous_TTS import nodes


class FakeEngine:
    device = torch.device("cuda")

    def __init__(self):
        self.loaded = True
        self.cached = True

    def holds_models(self):
        return self.loaded

    def holds_anything(self):
        return self.loaded or self.cached

    def on(self, device):
        return device is None or torch.device(device).type == self.device.type

    def unload(self, everything=False):
        self.loaded = False
        if everything:
            self.cached = False


def test_models_stay_while_memory_is_enough_and_go_when_it_is_not(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(nodes, "_engine", engine)
    monkeypatch.setattr(mm, "get_free_memory", lambda dev=None, torch_free_too=False: 2 * 1024 ** 3)
    mm.free_memory(1 * 1024 ** 3, torch.device("cuda"))
    assert engine.loaded  # 1 GB asked, 2 GB free: keep them for the next speech
    mm.free_memory(1 * 1024 ** 3, torch.device("cpu"))
    assert engine.loaded  # another device
    mm.free_memory(4 * 1024 ** 3, torch.device("cuda"))
    assert not engine.loaded  # an image model needs more than is free
    assert engine.cached  # sentence caches stay for a redo of one line
    engine.loaded = True
    mm.free_memory(1e30, torch.device("cuda"))  # what "Unload models" asks for
    assert not engine.loaded and not engine.cached


def test_unload_models_frees_caches_even_with_models_already_gone(monkeypatch):
    engine = FakeEngine()
    engine.loaded = False
    monkeypatch.setattr(nodes, "_engine", engine)
    mm.free_memory(1e30, torch.device("cpu"))
    assert not engine.cached
