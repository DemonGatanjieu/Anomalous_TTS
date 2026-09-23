"""Anomalous TTS (GPT-SoVITS): character voices for ComfyUI.

Sister project of Anomalous Model Browser. Engine code is a trimmed copy of
GPT-SoVITS and Genie-TTS (both MIT); see UPSTREAM.md.
"""

from .core import paths as _paths

_paths.register()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
