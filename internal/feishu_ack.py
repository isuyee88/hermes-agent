from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Mapping


def add_ack_reaction_from_payload(
    payload: dict[str, Any],
    *,
    request_started_at_ms: int | None,
    worker_context: Mapping[str, Any] | None,
    extract_queue_context: Callable[[dict[str, Any]], dict[str, Any]],
    extract_event_metadata: Callable[[dict[str, Any]], tuple[str, str]],
    resolve_request_started_at_ms: Callable[[dict[str, Any], int | None], int | None],
    append_trace: Callable[..., None],
    ack_reaction_emoji: str,
    ack_reaction_timeout_seconds: float,
    logger: Any,
) -> dict[str, Any]:
    from tools.feishu_api import build_feishu_client

    context = extract_queue_context(payload)
    message_id = str(context.get("message_id") or "").strip()
    event_id, event_type = extract_event_metadata(payload)
    if not message_id:
        return {
            "status": "skipped",
            "reason": "missing_message_id",
            "event_id": event_id,
            "event_type": event_type,
        }

    started_at = time.perf_counter()
    client_started_at = time.perf_counter()
    client = build_feishu_client(timeout=ack_reaction_timeout_seconds)
    client_acquire_elapsed_ms = int((time.perf_counter() - client_started_at) * 1000)
    reaction_started_at = time.perf_counter()
    success = False
    error_text = ""
    reaction_id = ""
    try:
        response_payload = client.request_json(
            "POST",
            f"/open-apis/im/v1/messages/{message_id}/reactions",
            json_body={"reaction_type": {"emoji_type": ack_reaction_emoji}},
            retries=1,
        )
        reaction_id = str(
            response_payload.get("reaction_id")
            or (response_payload.get("reaction") or {}).get("reaction_id")
            or ""
        ).strip()
        success = True
    except Exception as exc:
        error_text = str(exc)

    reaction_elapsed_ms = int((time.perf_counter() - reaction_started_at) * 1000)
    total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    resolved_request_started_at_ms = resolve_request_started_at_ms(payload, request_started_at_ms)
    from_request_elapsed_ms = (
        max(0, int(time.time() * 1000) - int(resolved_request_started_at_ms))
        if resolved_request_started_at_ms is not None
        else None
    )
    worker_boot_id = str((worker_context or {}).get("worker_boot_id") or "").strip()
    worker_reused = (worker_context or {}).get("container_reused")
    append_trace(
        "webhook.ack_reaction",
        payload,
        success=success,
        message_id=message_id,
        reaction_id=reaction_id or "",
        client_acquire_elapsed_ms=client_acquire_elapsed_ms,
        reaction_elapsed_ms=reaction_elapsed_ms,
        total_elapsed_ms=total_elapsed_ms,
        from_request_elapsed_ms=from_request_elapsed_ms,
        worker_boot_id=worker_boot_id,
        worker_reused=worker_reused,
        error=error_text,
    )
    if success:
        logger.warning(
            "[Feishu] webhook ack reaction event_type=%s event_id=%s message_id=%s success=%s client_acquire_elapsed_ms=%s reaction_elapsed_ms=%s total_elapsed_ms=%s from_request_elapsed_ms=%s worker_boot_id=%s worker_reused=%s",
            event_type or "unknown",
            event_id or "none",
            message_id,
            success,
            client_acquire_elapsed_ms,
            reaction_elapsed_ms,
            total_elapsed_ms,
            from_request_elapsed_ms if from_request_elapsed_ms is not None else "na",
            worker_boot_id or "none",
            worker_reused,
        )
    else:
        logger.warning(
            "[Feishu] webhook ack reaction failed event_type=%s event_id=%s message_id=%s client_acquire_elapsed_ms=%s reaction_elapsed_ms=%s total_elapsed_ms=%s from_request_elapsed_ms=%s worker_boot_id=%s worker_reused=%s error=%s",
            event_type or "unknown",
            event_id or "none",
            message_id,
            client_acquire_elapsed_ms,
            reaction_elapsed_ms,
            total_elapsed_ms,
            from_request_elapsed_ms if from_request_elapsed_ms is not None else "na",
            worker_boot_id or "none",
            worker_reused,
            error_text or "unknown",
        )
    return {
        "status": "ok" if success else "error",
        "event_id": event_id,
        "event_type": event_type,
        "message_id": message_id,
        "reaction_id": reaction_id or "",
        "client_acquire_elapsed_ms": client_acquire_elapsed_ms,
        "reaction_elapsed_ms": reaction_elapsed_ms,
        "total_elapsed_ms": total_elapsed_ms,
        "from_request_elapsed_ms": from_request_elapsed_ms,
        "worker_boot_id": worker_boot_id,
        "worker_reused": worker_reused,
        "error": error_text,
    }


async def add_ack_reaction_inline_async(
    *,
    payload: dict[str, Any],
    request_started_at_ms: int | None,
    add_ack_reaction_from_payload_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    worker_context = {
        "worker_boot_id": f"feishu-webhook-inline-{uuid.uuid4().hex}",
        "container_reused": True,
    }
    return await asyncio.to_thread(
        add_ack_reaction_from_payload_fn,
        payload,
        request_started_at_ms=request_started_at_ms,
        worker_context=worker_context,
    )


async def spawn_ack_reaction_async(
    *,
    payload: dict[str, Any],
    request_started_at: float | None,
    request_started_at_ms: int | None,
    extract_queue_context: Callable[[dict[str, Any]], dict[str, Any]],
    with_internal_meta: Callable[..., dict[str, Any]],
    worker: Any,
    ingress_handoff_timeout_seconds: float,
) -> dict[str, Any]:
    context = extract_queue_context(payload)
    started_at = time.perf_counter()
    message_id = str(context.get("message_id") or "").strip()
    if not message_id:
        return {
            "status": "skipped",
            "reason": "missing_message_id",
            "schedule_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        }
    if worker is None or not hasattr(worker, "spawn"):
        return {
            "status": "unavailable",
            "reason": "process_feishu_ack_reaction_spawn_unavailable",
            "schedule_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "message_id": message_id,
        }

    payload_for_spawn = with_internal_meta(
        payload,
        ack_reaction_requested_at_ms=int(time.time() * 1000),
        webhook_message_id=message_id,
    )
    spawn_handle = getattr(worker, "spawn")
    timeout_seconds = max(float(ingress_handoff_timeout_seconds or 0.0), 0.1)
    if request_started_at_ms is None and request_started_at is not None:
        request_started_at_ms = None
    try:
        if hasattr(spawn_handle, "aio"):
            await asyncio.wait_for(
                spawn_handle.aio(payload=payload_for_spawn, request_started_at_ms=request_started_at_ms),
                timeout=timeout_seconds,
            )
        else:
            await asyncio.wait_for(
                asyncio.to_thread(
                    spawn_handle,
                    payload=payload_for_spawn,
                    request_started_at_ms=request_started_at_ms,
                ),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "reason": "process_feishu_ack_reaction_spawn_timeout",
            "schedule_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "message_id": message_id,
        }
    return {
        "status": "scheduled",
        "reason": "process_feishu_ack_reaction_spawned",
        "schedule_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        "message_id": message_id,
    }
