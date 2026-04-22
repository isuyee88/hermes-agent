from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable


async def dispatch_telegram_update(
    update_payload: dict[str, Any],
    *,
    get_runtime: Callable[[], Awaitable[Any]],
) -> dict[str, Any]:
    runtime = await get_runtime()
    adapter = runtime.adapter

    from telegram import Update
    from gateway.platforms.base import MessageType
    from gateway.session import build_session_key

    async def _process_event_sync(event: Any) -> None:
        session_key = build_session_key(
            event.source,
            group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        )

        if session_key in adapter._active_sessions:
            await adapter.handle_message(event)
            return

        adapter._active_sessions[session_key] = asyncio.Event()
        await adapter._process_message_background(event, session_key)

    update = Update.de_json(update_payload, adapter._bot)
    if update is None:
        return {"status": "ignored", "reason": "invalid_update"}

    if update.callback_query:
        await adapter._handle_callback_query(update, None)
        return {"status": "accepted", "kind": "callback_query"}

    message = (
        update.message
        or update.edited_message
        or update.channel_post
        or update.edited_channel_post
    )
    if not message:
        return {"status": "ignored", "reason": "unsupported_update"}

    if message.text:
        if message.text.lstrip().startswith("/"):
            await adapter._handle_command(update, None)
            return {"status": "accepted", "kind": "command"}

        if not adapter._should_process_message(message):
            return {"status": "ignored", "reason": "message_filtered"}
        event = adapter._build_message_event(message, MessageType.TEXT)
        event.text = adapter._clean_bot_trigger_text(event.text)
        await _process_event_sync(event)
        return {"status": "accepted", "kind": "text"}

    if getattr(message, "location", None) or getattr(message, "venue", None):
        await adapter._handle_location_message(update, None)
        return {"status": "accepted", "kind": "location"}

    if any(
        getattr(message, attr, None)
        for attr in ("sticker", "photo", "video", "audio", "voice", "document")
    ):
        await adapter._handle_media_message(update, None)
        return {"status": "accepted", "kind": "media"}

    return {"status": "ignored", "reason": "no_supported_content"}


async def dispatch_qq_update(
    payload: dict[str, Any],
    *,
    headers: dict[str, Any] | None = None,
    get_runtime: Callable[[], Awaitable[Any]],
) -> dict[str, Any]:
    runtime = await get_runtime()
    adapter = runtime.adapter
    return await adapter.handle_webhook_payload(payload, headers=headers or {})


def to_fastapi_response_from_aiohttp(aiohttp_response: Any, *, response_cls: type) -> Any:
    status_code = int(getattr(aiohttp_response, "status", 200) or 200)
    raw_headers = dict(getattr(aiohttp_response, "headers", {}) or {})
    response_headers = {
        key: value
        for key, value in raw_headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "content-encoding", "connection"}
    }
    body = getattr(aiohttp_response, "body", None)
    if body is None:
        text = getattr(aiohttp_response, "text", "")
        body = text.encode("utf-8") if isinstance(text, str) else (text or b"")
    return response_cls(content=body, status_code=status_code, headers=response_headers)
