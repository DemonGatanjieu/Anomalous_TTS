"""HTTP API for Anomalous Model Browser and the node's own UI (docs/INTERFACE.md §5)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from aiohttp import web

from .core import characters, settings

log = logging.getLogger("Anomalous_TTS")
API_FORMAT = 2


def _summary(c: characters.Character, detail: bool) -> Dict[str, Any]:
    try:
        return c.to_api(detail=detail)
    except Exception as e:  # one broken folder must not hide the others
        log.warning("[Anomalous_TTS] 读取角色 %s 失败：%s", c.name, e)
        return {"name": c.name, "error": str(e)}


def _characters_payload(name: Optional[str], refresh: bool) -> Dict[str, Any]:
    chars = characters.scan(max_age=0 if refresh else characters.CACHE_SECONDS)
    if name is not None:
        if name not in chars:
            raise web.HTTPNotFound(text=f"找不到角色：{name}")
        return {"format": API_FORMAT, "character": _summary(chars[name], detail=True)}
    return {"format": API_FORMAT, "characters": [_summary(c, detail=False) for c in chars.values()]}


def _save_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    name = body.get("character")
    data = body.get("settings")
    chars = characters.scan(max_age=0)
    if name not in chars:
        raise web.HTTPBadRequest(text=f"找不到角色：{name}")
    c = chars[name]
    problems = settings.validate(data, c.gpt, c.sovits, c.audio)
    if problems:
        raise web.HTTPBadRequest(text="；".join(problems))
    settings.save(c.folder, data)
    characters.invalidate()
    return {"ok": True, "character": _summary(characters.scan()[name], detail=True)}


def register(prompt_server) -> None:
    routes = prompt_server.routes

    @routes.get("/anomalous_tts/characters")
    async def get_characters(request):
        name = request.query.get("name")
        refresh = request.query.get("refresh") in ("1", "true")
        payload = await asyncio.get_running_loop().run_in_executor(None, _characters_payload, name, refresh)
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
