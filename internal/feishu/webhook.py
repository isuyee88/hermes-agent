from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from typing import Any, Awaitable, Callable


async def parse_webhook_request(
    request: Any,
    *,
    get_security_settings: Callable[[], tuple[str, str]],
    is_signature_valid: Callable[[dict[str, Any], bytes], bool],
    decrypt_payload: Callable[[str, str], dict[str, Any]],
    logger: Any,
    http_exception_cls: type[Exception],
) -> tuple[None, dict[str, Any]]:
    headers = dict(request.headers)

    content_type = str(headers.get("content-type", "") or "").split(";", 1)[0].strip().lower()
    if content_type and content_type != "application/json":
        raise http_exception_cls(status_code=415, detail="Unsupported Media Type")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 1024 * 1024:
                raise http_exception_cls(status_code=413, detail="Request body too large")
        except ValueError:
            pass

    body = await request.body()
    if len(body) > 1024 * 1024:
        raise http_exception_cls(status_code=413, detail="Request body too large")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise http_exception_cls(status_code=400, detail="invalid json")

    encrypt_key, verification_token = get_security_settings()
    if encrypt_key and not is_signature_valid(headers, body):
        raise http_exception_cls(status_code=401, detail="Invalid signature")

    if payload.get("encrypt"):
        try:
            payload = decrypt_payload(str(payload.get("encrypt") or ""), encrypt_key)
        except Exception:
            logger.exception("Feishu encrypted webhook decrypt failed")
            raise http_exception_cls(status_code=400, detail="failed to decrypt webhook payload")

    if verification_token:
        header = payload.get("header") or {}
        incoming_token = str(header.get("token") or payload.get("token") or "")
        if not incoming_token or not hmac.compare_digest(incoming_token, verification_token):
            raise http_exception_cls(status_code=401, detail="Invalid verification token")

    return None, payload


async def await_pending_batches(adapter: Any, *, logger: Any) -> None:
    timeout_raw = os.getenv("HERMES_FEISHU_WEBHOOK_DRAIN_TIMEOUT_SECONDS", "6").strip()
    try:
        timeout_seconds = max(0.0, float(timeout_raw))
    except ValueError:
        timeout_seconds = 6.0
    if timeout_seconds <= 0:
        return

    pending: list[asyncio.Task[Any]] = []
    for attr in ("_pending_text_batch_tasks", "_pending_media_batch_tasks"):
        task_map = getattr(adapter, attr, None)
        if not isinstance(task_map, dict):
            continue
        for task in task_map.values():
            if isinstance(task, asyncio.Task) and not task.done():
                pending.append(task)
    if not pending:
        return

    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "[Feishu] pending batch drain timed out after %.1fs (pending=%d)",
            timeout_seconds,
            len(pending),
        )


async def await_background_tasks(adapter: Any, *, logger: Any) -> None:
    timeout_raw = os.getenv("HERMES_FEISHU_WEBHOOK_BACKGROUND_TIMEOUT_SECONDS", "180").strip()
    try:
        timeout_seconds = max(0.0, float(timeout_raw))
    except ValueError:
        timeout_seconds = 180.0
    if timeout_seconds <= 0:
        return

    background = getattr(adapter, "_background_tasks", None)
    if not isinstance(background, set):
        return
    pending = [task for task in list(background) if isinstance(task, asyncio.Task) and not task.done()]
    if not pending:
        return

    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "[Feishu] background processing timeout after %.1fs (pending=%d)",
            timeout_seconds,
            len(pending),
        )


async def dispatch_payload(
    payload: dict[str, Any],
    *,
    await_background_tasks_enabled: bool,
    runtime_cached: bool,
    get_runtime: Callable[[], Awaitable[Any]],
    extract_event_metadata: Callable[[dict[str, Any]], tuple[str, str]],
    extract_trace_token: Callable[[dict[str, Any]], str],
    append_trace: Callable[..., None],
    logger: Any,
    await_pending_batches_fn: Callable[..., Awaitable[None]],
    await_background_tasks_fn: Callable[..., Awaitable[None]],
    capture_phase_elapsed: Callable[[dict[str, int], str, float], None],
    normalize_phase_timings: Callable[[dict[str, int]], dict[str, int]],
) -> dict[str, Any]:
    runtime_started_at = time.perf_counter()
    runtime = await get_runtime()
    runtime_acquire_elapsed_ms = int((time.perf_counter() - runtime_started_at) * 1000)
    adapter = runtime.adapter
    event_id, event_type = extract_event_metadata(payload)
    trace_token = extract_trace_token(payload)
    data = adapter._namespace_from_mapping(payload)
    dispatch_phase_timings: dict[str, int] = {}

    append_trace(
        "dispatch.start",
        payload,
        runtime_acquire_elapsed_ms=runtime_acquire_elapsed_ms,
        runtime_cached=runtime_cached,
    )
    logger.warning(
        "[Feishu] dispatch start event_type=%s event_id=%s trace_token=%s runtime_acquire_elapsed_ms=%s runtime_cached=%s",
        event_type or "unknown",
        event_id or "none",
        trace_token or "none",
        runtime_acquire_elapsed_ms,
        runtime_cached,
    )

    try:
        if event_type == "im.message.receive_v1":
            handle_started_at = time.perf_counter()
            await adapter._handle_message_event_data(data)
            capture_phase_elapsed(dispatch_phase_timings, "message_handle_elapsed_ms", handle_started_at)
            if await_background_tasks_enabled:
                pending_started_at = time.perf_counter()
                await await_pending_batches_fn(adapter, logger=logger)
                capture_phase_elapsed(dispatch_phase_timings, "pending_batches_elapsed_ms", pending_started_at)
                background_started_at = time.perf_counter()
                await await_background_tasks_fn(adapter, logger=logger)
                capture_phase_elapsed(dispatch_phase_timings, "background_tasks_elapsed_ms", background_started_at)
        elif event_type == "im.message.message_read_v1":
            handle_started_at = time.perf_counter()
            adapter._on_message_read_event(data)
            capture_phase_elapsed(dispatch_phase_timings, "message_read_handle_elapsed_ms", handle_started_at)
        elif event_type == "im.chat.member.bot.added_v1":
            handle_started_at = time.perf_counter()
            adapter._on_bot_added_to_chat(data)
            capture_phase_elapsed(dispatch_phase_timings, "bot_added_handle_elapsed_ms", handle_started_at)
        elif event_type == "im.chat.member.bot.deleted_v1":
            handle_started_at = time.perf_counter()
            adapter._on_bot_removed_from_chat(data)
            capture_phase_elapsed(dispatch_phase_timings, "bot_deleted_handle_elapsed_ms", handle_started_at)
        elif event_type in ("im.message.reaction.created_v1", "im.message.reaction.deleted_v1"):
            handle_started_at = time.perf_counter()
            await adapter._handle_reaction_event(event_type, data)
            capture_phase_elapsed(dispatch_phase_timings, "reaction_handle_elapsed_ms", handle_started_at)
            if await_background_tasks_enabled:
                background_started_at = time.perf_counter()
                await await_background_tasks_fn(adapter, logger=logger)
                capture_phase_elapsed(dispatch_phase_timings, "background_tasks_elapsed_ms", background_started_at)
        elif event_type == "card.action.trigger":
            if await_background_tasks_enabled:
                handle_started_at = time.perf_counter()
                await adapter._handle_card_action_event(data)
                capture_phase_elapsed(dispatch_phase_timings, "card_action_handle_elapsed_ms", handle_started_at)
                background_started_at = time.perf_counter()
                await await_background_tasks_fn(adapter, logger=logger)
                capture_phase_elapsed(dispatch_phase_timings, "background_tasks_elapsed_ms", background_started_at)
            else:
                background = getattr(adapter, "_background_tasks", None)
                task = asyncio.create_task(adapter._handle_card_action_event(data))
                if isinstance(background, set):
                    background.add(task)

                    def _cleanup_background_card_action(done_task: asyncio.Task[Any]) -> None:
                        background.discard(done_task)
                        if done_task.cancelled():
                            return
                        try:
                            exc = done_task.exception()
                        except Exception:
                            logger.exception("[Feishu] Failed to inspect card action background task")
                            return
                        if exc is not None:
                            logger.exception("[Feishu] Card action background task failed", exc_info=exc)

                    task.add_done_callback(_cleanup_background_card_action)
        elif event_type == "application.bot.menu_v6":
            handle_started_at = time.perf_counter()
            await adapter._handle_bot_menu_event(data)
            capture_phase_elapsed(dispatch_phase_timings, "menu_handle_elapsed_ms", handle_started_at)
            if await_background_tasks_enabled:
                background_started_at = time.perf_counter()
                await await_background_tasks_fn(adapter, logger=logger)
                capture_phase_elapsed(dispatch_phase_timings, "background_tasks_elapsed_ms", background_started_at)
        else:
            logger.warning("[Feishu] Ignoring unsupported event type in dispatcher: %s", event_type or "unknown")
    except Exception as exc:
        normalized_phase_timings = normalize_phase_timings(dispatch_phase_timings)
        append_trace("dispatch.error", payload, error=str(exc), phase_timings=normalized_phase_timings)
        raise

    normalized_phase_timings = normalize_phase_timings(dispatch_phase_timings)
    append_trace("dispatch.done", payload, phase_timings=normalized_phase_timings)
    logger.warning(
        "[Feishu] dispatch done event_type=%s event_id=%s phase_timings=%s",
        event_type or "unknown",
        event_id or "none",
        normalized_phase_timings,
    )
    return {
        "event_id": event_id,
        "event_type": event_type,
        "runtime_acquire_elapsed_ms": runtime_acquire_elapsed_ms,
        "phase_timings": normalized_phase_timings,
        "await_background_tasks": await_background_tasks_enabled,
    }
