"""Anomalous TTS (GPT-SoVITS): character voices for ComfyUI.

Sister project of Anomalous Model Browser (docs/INTERFACE.md). Engine code is
a trimmed copy of GPT-SoVITS and Genie-TTS (both MIT); see UPSTREAM.md.
"""

import logging

from .core import paths as _paths

_paths.register()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

WEB_DIRECTORY = "./web"

try:
    from server import PromptServer

    from .server import register as _register_routes

    _register_routes(PromptServer.instance)
except Exception as e:  # not running inside ComfyUI's server (tests, scripts)
    logging.getLogger("Anomalous_TTS").debug("HTTP routes not registered: %s", e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
