from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Awaitable, Callable, Mapping


def should_spawn_feishu_message_inline(payload: dict[str, Any], context: dict[str, Any]) -> bool:
    event_type = str((context or {}).get("event_type") or ((payload.get("header") or {}).get("event_type") or "")).strip()
    if event_type != "im.message.receive_v1":
        return False

    lane = str((context or {}).get("lane") or "").strip().lower()
    if lane and lane != "chat_light":
        return False

    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return False
    message = event.get("message") or {}
    if not isinstance(message, dict):
        return False

    message_type = str(message.get("message_type") or "").strip().lower()
    if message_type and message_type != "text":
        return False

    chat_type = str(message.get("chat_type") or event.get("chat_type") or "").strip().lower()
    if chat_type and chat_type != "p2p":
        return False

    raw_content = str(message.get("content") or "")
    if not raw_content or len(raw_content) >= 1600:
        return False

    mentions = event.get("mentions") or message.get("mentions") or []
    if isinstance(mentions, list) and len(mentions) >= 3:
        return False

    return True


def resolve_feishu_message_ingress_strategy(
    payload: dict[str, Any],
    context: dict[str, Any],
    *,
    env_value: str | None,
    default_strategy: str,
    legacy_aliases: Mapping[str, str],
    supported_strategies: set[str],
    should_spawn_inline: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> str:
    configured = env_value if env_value is not None else default_strategy
    effective = str(configured or "").strip().lower()
    configured_is_legacy_alias = effective in legacy_aliases
    if effective in legacy_aliases:
        effective = legacy_aliases[effective]
    if (not str(env_value or "").strip() or configured_is_legacy_alias) and should_spawn_inline(payload, context):
        return "spawn_process_feishu_message_inline"
    if effective in supported_strategies:
        return effective
    return "inline_enqueue_spawn"


def coalesce_feishu_chat_queue_items(
    items: list[Any],
    *,
    now_ms: int,
    max_age_ms: int,
    logger: Any,
) -> list[Any]:
    if not items:
        return items

    newest_message_index_by_partition: dict[str, int] = {}
    filtered: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        if str(item.get("platform") or "").strip().lower() != "feishu":
            filtered.append(item)
            continue

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            filtered.append(item)
            continue
        if str(metadata.get("event_type") or "").strip() != "im.message.receive_v1":
            filtered.append(item)
            continue

        partition = str(item.get("partition") or metadata.get("partition") or "").strip()
        enqueued_at_ms = int(item.get("enqueued_at_ms") or 0)
        age_ms = max(0, now_ms - enqueued_at_ms) if enqueued_at_ms else 0
        if age_ms and age_ms > max_age_ms:
            logger.warning(
                "[ChatQueue] skipping stale feishu message event_id=%s partition=%s age_ms=%s max_age_ms=%s",
                metadata.get("event_id") or "none",
                partition or "none",
                age_ms,
                max_age_ms,
            )
            continue

        previous_index = newest_message_index_by_partition.get(partition)
        if previous_index is None:
            newest_message_index_by_partition[partition] = len(filtered)
            filtered.append(item)
            continue

        previous_item = filtered[previous_index]
        previous_enqueued_at_ms = int(previous_item.get("enqueued_at_ms") or 0) if isinstance(previous_item, dict) else 0
        previous_event_id = (
            ((previous_item.get("metadata") or {}).get("event_id") if isinstance(previous_item, dict) else None) or "none"
        )
        if enqueued_at_ms >= previous_enqueued_at_ms:
            logger.warning(
                "[ChatQueue] superseding older feishu message older_event_id=%s newer_event_id=%s partition=%s",
                previous_event_id,
                metadata.get("event_id") or "none",
                partition or "none",
            )
            filtered[previous_index] = item
        else:
            logger.warning(
                "[ChatQueue] dropping superseded feishu message older_event_id=%s newer_event_id=%s partition=%s",
                metadata.get("event_id") or "none",
                previous_event_id,
                partition or "none",
            )

    return filtered


def sample_light_p2p_feishu_message_payload() -> dict[str, Any]:
    return {
        "header": {
            "event_type": "im.message.receive_v1",
        },
        "event": {
            "message": {
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": "ping"}, ensure_ascii=False),
            }
        },
    }


async def process_chat_queue_item_async(
    payload: Any,
    *,
    worker_context: dict[str, Any] | None,
    runtime_prepared: bool,
    build_inline_chat_worker_context: Callable[[], dict[str, Any]],
    is_truthy: Callable[[str | None, bool], bool],
    sync_modal_volume_async: Callable[..., Awaitable[None]],
    build_chat_worker_observability: Callable[[dict[str, Any]], dict[str, Any]],
    append_feishu_trace: Callable[..., None],
    dispatch_feishu_payload: Callable[[dict[str, Any], bool], Awaitable[dict[str, Any]]],
    dispatch_telegram_update: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    logger: Any,
    feishu_chat_worker_timeout_seconds: float,
) -> dict[str, Any]:
    context = dict(worker_context or {})
    if not runtime_prepared:
        inline_context = build_inline_chat_worker_context()
        inline_context.update(context)
        context = inline_context

    if is_truthy(os.getenv("HERMES_MODAL_CHAT_WORKER_RELOAD"), False):
        await sync_modal_volume_async(reload=True)

    envelope = payload if isinstance(payload, dict) else {}
    platform = str(envelope.get("platform") or "").strip().lower()
    partition = str(envelope.get("partition") or "").strip()
    raw_payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    metadata = envelope.get("metadata") if isinstance(envelope.get("metadata"), dict) else {}
    enqueued_at_ms = int(envelope.get("enqueued_at_ms") or 0)
    queue_latency_ms = max(0, int(time.time() * 1000) - enqueued_at_ms) if enqueued_at_ms else None
    lane = str(metadata.get("lane") or "").strip() or None
    ingress_strategy = str(metadata.get("ingress_strategy") or "").strip() or None
    warmup_ready_at = int(context.get("warmup_ready_at") or 0)
    if warmup_ready_at and enqueued_at_ms:
        context["warmup_message_gap_ms"] = max(0, enqueued_at_ms - warmup_ready_at)

    if platform not in {"feishu", "telegram"}:
        return {
            "status": "skipped",
            "reason": "unsupported_platform",
            "platform": platform,
            "partition": partition,
            **build_chat_worker_observability(context),
        }
    if not raw_payload:
        return {
            "status": "skipped",
            "reason": "missing_payload",
            "platform": platform,
            "partition": partition,
            **build_chat_worker_observability(context),
        }

    if platform == "feishu":
        append_feishu_trace(
            "worker.start",
            raw_payload,
            partition=partition,
            lane=lane,
            ingress_strategy=ingress_strategy or "",
            queue_latency_ms=queue_latency_ms,
            **build_chat_worker_observability(context),
        )
        logger.warning(
            "[Feishu] worker start event_type=%s event_id=%s trace_token=%s partition=%s lane=%s queue_latency_ms=%s worker_boot_id=%s reused=%s warmup_status=%s warmup_age_ms=%s warmup_same_container=%s ingress_strategy=%s",
            metadata.get("event_type") or "unknown",
            metadata.get("event_id") or "none",
            metadata.get("trace_token") or "none",
            partition or "none",
            lane or "none",
            queue_latency_ms if queue_latency_ms is not None else "na",
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
            context.get("warmup_status") or "none",
            context.get("warmup_age_ms") if context.get("warmup_age_ms") is not None else "na",
            context.get("warmup_same_container") if context.get("warmup_same_container") is not None else "na",
            ingress_strategy or "",
        )
    else:
        logger.info(
            "Telegram worker start update_id=%s partition=%s queue_latency_ms=%s worker_boot_id=%s reused=%s",
            metadata.get("event_id") or "none",
            partition or "none",
            queue_latency_ms if queue_latency_ms is not None else "na",
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
        )

    started_at = time.time()
    try:
        if platform == "feishu":
            event_type = str(metadata.get("event_type") or "").strip().lower()
            dispatch_coro = dispatch_feishu_payload(raw_payload, True)
            if event_type == "im.message.receive_v1":
                await asyncio.wait_for(dispatch_coro, timeout=feishu_chat_worker_timeout_seconds)
            else:
                await dispatch_coro
        else:
            await dispatch_telegram_update(raw_payload)
    except asyncio.TimeoutError:
        timeout_error = (
            f"feishu dispatch exceeded {feishu_chat_worker_timeout_seconds:.1f}s "
            f"for {str(metadata.get('event_type') or 'unknown')}"
        )
        if platform == "feishu":
            append_feishu_trace(
                "worker.timeout",
                raw_payload,
                partition=partition,
                lane=lane,
                queue_latency_ms=queue_latency_ms,
                error=timeout_error,
                **build_chat_worker_observability(context),
            )
            logger.error(
                "[Feishu] worker timeout event_type=%s event_id=%s partition=%s timeout_seconds=%.1f",
                metadata.get("event_type") or "unknown",
                metadata.get("event_id") or "none",
                partition or "none",
                feishu_chat_worker_timeout_seconds,
            )
        return {
            "status": "error",
            "platform": platform,
            "partition": partition,
            "event_id": metadata.get("event_id"),
            "message_id": metadata.get("message_id"),
            "queue_latency_ms": queue_latency_ms,
            "error": timeout_error,
            **build_chat_worker_observability(context),
        }
    except Exception as exc:
        if platform == "feishu":
            append_feishu_trace(
                "worker.error",
                raw_payload,
                partition=partition,
                lane=lane,
                queue_latency_ms=queue_latency_ms,
                error=str(exc),
                **build_chat_worker_observability(context),
            )
            logger.exception(
                "[Feishu] worker error event_type=%s event_id=%s partition=%s",
                metadata.get("event_type") or "unknown",
                metadata.get("event_id") or "none",
                partition or "none",
            )
        else:
            logger.exception(
                "Telegram worker error update_id=%s partition=%s",
                metadata.get("event_id") or "none",
                partition or "none",
            )
        return {
            "status": "error",
            "platform": platform,
            "partition": partition,
            "event_id": metadata.get("event_id"),
            "message_id": metadata.get("message_id"),
            "queue_latency_ms": queue_latency_ms,
            "error": str(exc),
            **build_chat_worker_observability(context),
        }

    elapsed_ms = int((time.time() - started_at) * 1000)
    if platform == "feishu":
        append_feishu_trace(
            "worker.done",
            raw_payload,
            partition=partition,
            lane=lane,
            queue_latency_ms=queue_latency_ms,
            worker_elapsed_ms=elapsed_ms,
            **build_chat_worker_observability(context),
        )
        logger.warning(
            "[Feishu] worker done event_type=%s event_id=%s trace_token=%s partition=%s worker_elapsed_ms=%s worker_boot_id=%s reused=%s",
            metadata.get("event_type") or "unknown",
            metadata.get("event_id") or "none",
            metadata.get("trace_token") or "none",
            partition or "none",
            elapsed_ms,
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
        )
    else:
        logger.info(
            "Telegram worker done update_id=%s partition=%s worker_elapsed_ms=%s worker_boot_id=%s reused=%s",
            metadata.get("event_id") or "none",
            partition or "none",
            elapsed_ms,
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
        )
    return {
        "status": "ok",
        "platform": platform,
        "partition": partition,
        "event_id": metadata.get("event_id"),
        "message_id": metadata.get("message_id"),
        "queue_latency_ms": queue_latency_ms,
        "worker_elapsed_ms": elapsed_ms,
        **build_chat_worker_observability(context),
    }


async def process_chat_queue_items_async(
    items: list[Any],
    *,
    worker_context: dict[str, Any] | None,
    runtime_prepared: bool,
    process_chat_queue_item_async_fn: Callable[..., Awaitable[dict[str, Any]]],
    coalesce_items: Callable[[list[Any]], list[Any]],
) -> list[dict[str, Any]]:
    items = coalesce_items(items)
    return [
        await process_chat_queue_item_async_fn(
            item,
            worker_context=worker_context,
            runtime_prepared=runtime_prepared,
        )
        for item in items
    ]
