"""HTTP API for Anomalous Model Browser and the node's own UI (docs/INTERFACE.md §5)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from aiohttp import web

from .core import browse, characters, dependencies, downloads, paths, settings

log = logging.getLogger("Anomalous_TTS")

API_FORMAT = 3
LOCAL_ADDRESSES = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


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


def _status_payload(local: bool) -> Dict[str, Any]:
    chars = list(characters.scan().values())
    libraries = paths.libraries()
    for lib in libraries:
        lib["characters"] = sum(1 for c in chars if paths.is_inside(c.folder, lib["path"]))
    pretrained = []
    for item in paths.pretrained_status():
        job = downloads.state(item["id"]) if item["state"] == "missing" else None
        pretrained.append({**item, **job} if job else item)
    deps = {}
    for lang in dependencies.LANG_PACKAGES:
        missing = dependencies.missing(lang)
        deps[lang] = {"ok": not missing, "missing": missing}
        if missing:
            deps[lang]["command"] = dependencies.install_command(missing)
    return {
        "format": API_FORMAT,
        "local": local,
        "libraries": libraries,
        "pretrained": pretrained,
        "pretrained_sources": paths.pretrained_sources(),
        "dependencies": deps,
    }


def _is_local(request: web.Request) -> bool:
    return request.remote in LOCAL_ADDRESSES


def _require_local(request: web.Request) -> None:
    if not _is_local(request):
        raise web.HTTPForbidden(text="只能在运行 ComfyUI 的这台电脑上操作。")


async def _json_body(request: web.Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise web.HTTPBadRequest(text="请求体不是 JSON")
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="请求体必须是 JSON 对象")
    return body


async def _in_thread(fn: Callable, *args):
    """Run disk work off the event loop; ValueError from core becomes 400 with its message."""
    try:
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)
    except ValueError as e:
        raise web.HTTPBadRequest(text=str(e))


def _change_library(folder: str, remove: bool) -> None:
    (paths.remove_library if remove else paths.add_library)(folder)
    characters.invalidate()


def _change_pretrained_source(folder: str, remove: bool) -> None:
    (paths.remove_pretrained_source if remove else paths.add_pretrained_source)(folder)


def _path_field(body: Dict[str, Any]) -> str:
    folder = body.get("path")
    if not isinstance(folder, str) or not folder.strip():
        raise web.HTTPBadRequest(text="缺少 path")
    return folder.strip()


def _download_ids(body: Dict[str, Any]) -> List[str]:
    ids = body.get("ids")
    if ids is None:
        return [p["id"] for p in paths.pretrained_status() if p["state"] == "missing"]
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise web.HTTPBadRequest(text="ids 必须是字符串列表")
    return ids


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
        _require_local(request)
        body = await _json_body(request)
        result = await asyncio.get_running_loop().run_in_executor(None, _save_settings, body)
        return web.json_response(result)

    @routes.get("/anomalous_tts/status")
    async def get_status(request):
        return web.json_response(await _in_thread(_status_payload, _is_local(request)))

    @routes.post("/anomalous_tts/libraries")
    async def post_libraries(request):
        _require_local(request)
        body = await _json_body(request)
        await _in_thread(_change_library, _path_field(body), bool(body.get("remove")))
        return web.json_response(await _in_thread(_status_payload, True))

    @routes.post("/anomalous_tts/pretrained/source")
    async def post_pretrained_source(request):
        _require_local(request)
        body = await _json_body(request)
        await _in_thread(_change_pretrained_source, _path_field(body), bool(body.get("remove")))
        return web.json_response(await _in_thread(_status_payload, True))

    @routes.post("/anomalous_tts/pretrained/download")
    async def post_pretrained_download(request):
        _require_local(request)
        body = await _json_body(request)
        ids = await _in_thread(_download_ids, body)
        await _in_thread(downloads.start, ids)
        return web.json_response({"ok": True})

    @routes.get("/anomalous_tts/browse")
    async def get_browse(request):
        _require_local(request)
        return web.json_response(await _in_thread(browse.listing, request.query.get("path")))
