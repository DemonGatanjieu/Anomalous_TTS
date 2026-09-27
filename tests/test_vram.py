"""Our models give VRAM back when ComfyUI needs more than is free (nodes._hook_free_memory)."""

import torch

import comfy.model_management as mm
from Anomalous_TTS import nodes


class FakeEngine:
    device = torch.device("cuda")

    def __init__(self):
        self.loaded = True

    def holds_models(self):
        return self.loaded

    def on(self, device):
        return device is None or torch.device(device).type == self.device.type

    def unload(self):
        self.loaded = False


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
    engine.loaded = True
    mm.free_memory(1e30, torch.device("cuda"))  # what "Unload models" asks for
    assert not engine.loaded
