from __future__ import annotations

import asyncio
import time
from typing import Any, Callable


def extract_feishu_queue_context(
    payload: dict[str, Any],
    *,
    extract_trace_context: Callable[[dict[str, Any]], dict[str, Any]],
    collect_chat_id: Callable[[dict[str, Any]], str],
    collect_actor_id: Callable[[dict[str, Any]], str],
    classify_chat_lane: Callable[[dict[str, Any]], str],
) -> dict[str, str]:
    trace = extract_trace_context(payload)
    event_payload = payload.get("event") if isinstance(payload.get("event"), dict) else payload
    chat_id = collect_chat_id(event_payload if isinstance(event_payload, dict) else payload)
    if not chat_id and isinstance(payload.get("event"), dict):
        event = payload.get("event") or {}
        context = event.get("context") if isinstance(event.get("context"), dict) else {}
        chat = event.get("chat") if isinstance(event.get("chat"), dict) else {}
        chat_id = str(
            chat.get("chat_id")
            or chat.get("open_chat_id")
            or context.get("open_chat_id")
            or context.get("chat_id")
            or event.get("open_chat_id")
            or event.get("chat_id")
            or ""
        ).strip()
    actor_id = collect_actor_id(event_payload if isinstance(event_payload, dict) else payload)
    if not actor_id and isinstance(payload.get("event"), dict):
        event = payload.get("event") or {}
        operator = event.get("operator") if isinstance(event.get("operator"), dict) else {}
        operator_id = operator.get("operator_id") if isinstance(operator.get("operator_id"), dict) else {}
        actor_id = str(
            operator_id.get("open_id")
            or operator.get("open_id")
            or operator_id.get("user_id")
            or event.get("open_id")
            or event.get("user_id")
            or ""
        ).strip()
    event_type = str(trace.get("event_type") or "").strip()
    lane = "inline_local"
    if event_type == "im.message.receive_v1":
        lane = classify_chat_lane(payload)
    elif event_type:
        lane = "control"
    # Keep chat partition stable across webhook retries and contract fixtures.
    # Feishu event_id is request-scoped and would fragment the chat worker lane.
    partition = f"feishu:{lane}:{chat_id or actor_id or 'unknown'}"
    return {
        "platform": "feishu",
        "partition": partition,
        "lane": lane,
        "chat_id": chat_id,
        "message_id": str(trace.get("message_id") or "").strip(),
        "event_id": str(trace.get("event_id") or "").strip(),
        "event_type": event_type,
        "actor_id": actor_id,
        "trace_token": str(trace.get("trace_token") or "").strip(),
    }


def extract_telegram_queue_context(update: dict[str, Any]) -> dict[str, str]:
    update_id = str(update.get("update_id") or "").strip()
    message = (
        update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
        or {}
    )
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = str(chat.get("id") or "").strip()
    user_id = str(sender.get("id") or "").strip()
    partition = f"telegram:{chat_id or user_id or update_id or 'unknown'}"
    return {
        "platform": "telegram",
        "partition": partition,
        "chat_id": chat_id,
        "message_id": str(message.get("message_id") or "").strip(),
        "event_id": update_id,
        "event_type": "telegram.update",
        "actor_id": user_id,
    }


def enqueue_chat_event(
    *,
    platform: str,
    partition: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    get_chat_queue: Callable[[], Any],
    safe_chat_queue_depth: Callable[[], int | None],
    default_chat_queue_claim_ttl_seconds: int,
) -> dict[str, Any]:
    queue = get_chat_queue()
    enqueued_at = int(time.time() * 1000)
    envelope = {
        "platform": platform,
        "partition": partition,
        "payload": payload,
        "metadata": metadata,
        "enqueued_at_ms": enqueued_at,
    }
    queue.put(envelope, partition=partition, partition_ttl=max(default_chat_queue_claim_ttl_seconds, 3600))
    return {
        "status": "enqueued",
        "platform": platform,
        "partition": partition,
        "enqueued_at_ms": enqueued_at,
        "queue_depth": safe_chat_queue_depth(),
        **metadata,
    }


async def enqueue_chat_event_async(
    *,
    platform: str,
    partition: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    include_queue_depth: bool,
    get_chat_queue: Callable[[], Any],
    safe_chat_queue_depth_async: Callable[[], Any],
    enqueue_chat_event_fn: Callable[..., dict[str, Any]],
    default_chat_queue_claim_ttl_seconds: int,
) -> dict[str, Any]:
    try:
        queue = get_chat_queue()
    except Exception:
        return enqueue_chat_event_fn(
            platform=platform,
            partition=partition,
            payload=payload,
            metadata=metadata,
        )

    enqueued_at = int(time.time() * 1000)
    envelope = {
        "platform": platform,
        "partition": partition,
        "payload": payload,
        "metadata": metadata,
        "enqueued_at_ms": enqueued_at,
    }
    partition_ttl = max(default_chat_queue_claim_ttl_seconds, 3600)
    if hasattr(queue, "put") and hasattr(queue.put, "aio"):
        await queue.put.aio(envelope, partition=partition, partition_ttl=partition_ttl)  # type: ignore[union-attr]
    else:
        queue.put(envelope, partition=partition, partition_ttl=partition_ttl)
    return {
        "status": "enqueued",
        "platform": platform,
        "partition": partition,
        "enqueued_at_ms": enqueued_at,
        "queue_depth": await safe_chat_queue_depth_async() if include_queue_depth else None,
        **metadata,
    }


async def spawn_feishu_event_handoff_async(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    with_internal_meta: Callable[..., dict[str, Any]],
    worker: Any,
    ingress_handoff_timeout_seconds: float,
    extract_event_metadata: Callable[[dict[str, Any]], tuple[str, str]],
    logger: Any,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    handoff_requested_at_ms = int(time.time() * 1000)
    payload_for_spawn = with_internal_meta(
        payload,
        handoff_requested_at_ms=handoff_requested_at_ms,
        webhook_event_id=context.get("event_id") or "",
        webhook_partition=context.get("partition") or "",
    )
    if worker is None or not hasattr(worker, "spawn"):
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "unavailable",
            "reason": "process_feishu_event_spawn_unavailable",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            **context,
        }

    spawn_handle = getattr(worker, "spawn")
    timeout_seconds = max(float(ingress_handoff_timeout_seconds or 0.0), 0.1)
    try:
        if hasattr(spawn_handle, "aio"):
            await asyncio.wait_for(
                spawn_handle.aio(payload=payload_for_spawn, warmup_only=False, warmup_context=None),
                timeout=timeout_seconds,
            )
        else:
            await asyncio.wait_for(
                asyncio.to_thread(spawn_handle, payload=payload_for_spawn, warmup_only=False, warmup_context=None),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        event_id, event_type = extract_event_metadata(payload)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.warning(
            "[Feishu] ingress handoff timed out event_type=%s event_id=%s timeout_ms=%s partition=%s",
            event_type or "unknown",
            event_id or "none",
            int(timeout_seconds * 1000),
            context.get("partition") or "none",
        )
        return {
            "status": "timeout",
            "reason": "process_feishu_event_spawn_timeout",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            **context,
        }
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "status": "scheduled",
        "reason": "process_feishu_event_spawned",
        "handoff_wait_elapsed_ms": elapsed_ms,
        "handoff_schedule_wait_elapsed_ms": elapsed_ms,
        **context,
    }


async def spawn_feishu_message_inline_async(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    worker: Any,
    ingress_handoff_timeout_seconds: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if worker is None or not hasattr(worker, "spawn"):
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "unavailable",
            "reason": "process_feishu_message_inline_spawn_unavailable",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            **context,
        }

    spawn_handle = getattr(worker, "spawn")
    timeout_seconds = max(float(ingress_handoff_timeout_seconds or 0.0), 0.1)
    try:
        if hasattr(spawn_handle, "aio"):
            await asyncio.wait_for(spawn_handle.aio(payload=payload), timeout=timeout_seconds)
        else:
            await asyncio.wait_for(asyncio.to_thread(spawn_handle, payload=payload), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "timeout",
            "reason": "process_feishu_message_inline_spawn_timeout",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            **context,
        }
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "status": "scheduled",
        "reason": "process_feishu_message_inline_spawned",
        "handoff_wait_elapsed_ms": elapsed_ms,
        "handoff_schedule_wait_elapsed_ms": elapsed_ms,
        **context,
    }


async def spawn_feishu_background_exec_async(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    handoff_reason: str,
    with_internal_meta: Callable[..., dict[str, Any]],
    worker: Any,
    ingress_handoff_timeout_seconds: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    payload_for_spawn = with_internal_meta(
        payload,
        background_exec_requested_at_ms=int(time.time() * 1000),
        handoff_reason=str(handoff_reason or "provider_slow_path").strip() or "provider_slow_path",
        execution_mode="inline_to_background",
        webhook_event_id=context.get("event_id") or "",
        webhook_partition=context.get("partition") or "",
    )
    if worker is None or not hasattr(worker, "spawn"):
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "unavailable",
            "reason": "process_feishu_background_exec_spawn_unavailable",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            "execution_mode": "inline_to_background",
            "handoff_reason": handoff_reason,
            **context,
        }

    spawn_handle = getattr(worker, "spawn")
    timeout_seconds = max(float(ingress_handoff_timeout_seconds or 0.0), 0.1)
    try:
        if hasattr(spawn_handle, "aio"):
            await asyncio.wait_for(spawn_handle.aio(payload=payload_for_spawn), timeout=timeout_seconds)
        else:
            await asyncio.wait_for(asyncio.to_thread(spawn_handle, payload=payload_for_spawn), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "timeout",
            "reason": "process_feishu_background_exec_spawn_timeout",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            "execution_mode": "inline_to_background",
            "handoff_reason": handoff_reason,
            **context,
        }
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "status": "scheduled",
        "reason": "process_feishu_background_exec_spawned",
        "handoff_wait_elapsed_ms": elapsed_ms,
        "handoff_schedule_wait_elapsed_ms": elapsed_ms,
        "execution_mode": "inline_to_background",
        "handoff_reason": handoff_reason,
        **context,
    }


async def spawn_feishu_ingress_warmup_async(
    *,
    payload: dict[str, Any],
    warmup_context: dict[str, Any],
    with_internal_meta: Callable[..., dict[str, Any]],
    worker: Any,
    ingress_handoff_timeout_seconds: float,
    extract_event_metadata: Callable[[dict[str, Any]], tuple[str, str]],
    spawn_chat_queue_warmup_async_fn: Callable[..., Any],
    logger: Any,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    warmup_requested_at_ms = int(time.time() * 1000)
    payload_for_spawn = with_internal_meta(
        payload,
        handoff_requested_at_ms=warmup_requested_at_ms,
        webhook_event_id=warmup_context.get("event_id") or "",
        webhook_partition=warmup_context.get("partition") or "",
        warmup_only=True,
    )
    if worker is not None and hasattr(worker, "spawn"):
        spawn_handle = getattr(worker, "spawn")
        timeout_seconds = max(float(ingress_handoff_timeout_seconds or 0.0), 0.1)
        try:
            if hasattr(spawn_handle, "aio"):
                await asyncio.wait_for(
                    spawn_handle.aio(payload=payload_for_spawn, warmup_only=True, warmup_context=warmup_context),
                    timeout=timeout_seconds,
                )
            else:
                await asyncio.wait_for(
                    asyncio.to_thread(spawn_handle, payload=payload_for_spawn, warmup_only=True, warmup_context=warmup_context),
                    timeout=timeout_seconds,
                )
        except asyncio.TimeoutError:
            event_id, event_type = extract_event_metadata(payload)
            logger.warning(
                "[Feishu] ingress warmup timed out event_type=%s event_id=%s timeout_ms=%s partition=%s",
                event_type or "unknown",
                event_id or "none",
                int(timeout_seconds * 1000),
                warmup_context.get("partition") or "none",
            )
            fallback_result = await spawn_chat_queue_warmup_async_fn(
                platform="feishu",
                partition=str(warmup_context.get("partition") or "").strip(),
                reason="bot_p2p_chat_entered",
                metadata=warmup_context,
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            fallback_result["warmup_handoff_wait_elapsed_ms"] = elapsed_ms
            fallback_result["warmup_handoff_schedule_wait_elapsed_ms"] = elapsed_ms
            return fallback_result

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "scheduled",
            "spawned": True,
            "mode": "feishu_ingress",
            "reason": "bot_p2p_chat_entered",
            "warmup_handoff_wait_elapsed_ms": elapsed_ms,
            "warmup_handoff_schedule_wait_elapsed_ms": elapsed_ms,
            "metadata": {
                key: value
                for key, value in dict(warmup_context or {}).items()
                if key in {"event_type", "event_id", "chat_id", "actor_id"}
            },
            "partition": warmup_context.get("partition") or "",
        }

    fallback_result = await spawn_chat_queue_warmup_async_fn(
        platform="feishu",
        partition=str(warmup_context.get("partition") or "").strip(),
        reason="bot_p2p_chat_entered",
        metadata=warmup_context,
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    fallback_result["warmup_handoff_wait_elapsed_ms"] = elapsed_ms
    fallback_result["warmup_handoff_schedule_wait_elapsed_ms"] = elapsed_ms
    return fallback_result
