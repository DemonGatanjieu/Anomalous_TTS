"""HTTP API for Anomalous Model Browser and the node's own UI (docs/INTERFACE.md §5)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from aiohttp import web

from .core import characters, settings

log = logging.getLogger("Anomalous_TTS")
API_FORMAT = 1


def _characters_payload() -> Dict[str, Any]:
    chars = characters.scan(force=True)
    out = []
    for c in chars.values():
        try:
            out.append(c.to_api())
        except Exception as e:  # one broken folder must not hide the others
            log.warning("[Anomalous_TTS] 读取角色 %s 失败：%s", c.name, e)
            out.append({"name": c.name, "error": str(e)})
    return {"format": API_FORMAT, "characters": out}


def _save_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    name = body.get("character")
    data = body.get("settings")
    chars = characters.scan(force=True)
    if name not in chars:
        raise web.HTTPBadRequest(text=f"找不到角色：{name}")
    c = chars[name]
    problems = settings.validate(data, c.gpt, c.sovits, c.audio)
    if problems:
        raise web.HTTPBadRequest(text="；".join(problems))
    settings.save(c.folder, data)
    characters.invalidate()
    return {"ok": True, "character": characters.scan(force=True)[name].to_api()}


def register(prompt_server) -> None:
    routes = prompt_server.routes

    @routes.get("/anomalous_tts/characters")
    async def get_characters(request):
        payload = await asyncio.get_running_loop().run_in_executor(None, _characters_payload)
        return web.json_response(payload)

    @routes.get("/anomalous_tts/audio")
    async def get_audio(request):
        name = request.query.get("character", "")
        rel = request.query.get("path", "")
        chars = await asyncio.get_running_loop().run_in_executor(None, characters.scan)
        c = chars.get(name)
        if c is None or rel not in c.audio:
            raise web.HTTPNotFound(text="找不到这个音频")
        return web.FileResponse(c.abspath(rel))

    @routes.post("/anomalous_tts/settings")
    async def post_settings(request):
        try:
            body = await request.json()
        except ValueError:
            raise web.HTTPBadRequest(text="请求体不是 JSON")
        result = await asyncio.get_running_loop().run_in_executor(None, _save_settings, body)
        return web.json_response(result)
