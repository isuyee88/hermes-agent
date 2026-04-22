from __future__ import annotations

import asyncio
import time
from typing import Any, Callable


def spawn_chat_queue_worker_sync(
    *,
    platform: str,
    partition: str,
    max_items: int,
    warmup_wait_seconds: int,
    warmup_metadata: dict[str, Any] | None,
    default_chat_queue_batch_size: int,
    schedule_chat_partition_worker: Callable[..., dict[str, Any]],
    get_process_chat_queue_handle: Callable[[str | None], Any],
    process_chat_queue_impl: Callable[..., dict[str, Any]],
    release_chat_partition_claim: Callable[..., None],
) -> dict[str, Any]:
    schedule_started_at = time.perf_counter()
    schedule_result = schedule_chat_partition_worker(partition, platform=platform)
    schedule_claim_elapsed_ms = int((time.perf_counter() - schedule_started_at) * 1000)
    if schedule_result.get("status") != "scheduled":
        return {**schedule_result, "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms}

    claim_token = str(schedule_result.get("claim_token") or "").strip()
    spawn_kwargs = {
        "platform": platform,
        "partition": partition,
        "max_items": max_items or default_chat_queue_batch_size,
        "claim_token": claim_token,
    }
    if int(warmup_wait_seconds or 0) > 0:
        spawn_kwargs["warmup_wait_seconds"] = int(warmup_wait_seconds)
        spawn_kwargs["warmup_metadata"] = dict(warmup_metadata or {})
    worker = get_process_chat_queue_handle(partition)
    try:
        if worker is not None and hasattr(worker, "spawn"):
            spawn_started_at = time.perf_counter()
            getattr(worker, "spawn")(**spawn_kwargs)
            spawn_rpc_elapsed_ms = int((time.perf_counter() - spawn_started_at) * 1000)
            return {
                **schedule_result,
                "spawned": True,
                "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms,
                "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
            }

        inline_started_at = time.perf_counter()
        process_chat_queue_impl(**spawn_kwargs)
        spawn_rpc_elapsed_ms = int((time.perf_counter() - inline_started_at) * 1000)
        return {
            **schedule_result,
            "spawned": False,
            "mode": "inline",
            "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms,
            "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
        }
    except Exception:
        release_chat_partition_claim(partition, claim_token=claim_token)
        raise


async def spawn_chat_queue_worker_async(
    *,
    platform: str,
    partition: str,
    max_items: int,
    warmup_wait_seconds: int,
    warmup_metadata: dict[str, Any] | None,
    default_chat_queue_batch_size: int,
    schedule_chat_partition_worker_async: Callable[..., Any],
    get_process_chat_queue_handle: Callable[[str | None], Any],
    process_chat_queue_impl: Callable[..., dict[str, Any]],
    release_chat_partition_claim_async: Callable[..., Any],
) -> dict[str, Any]:
    schedule_started_at = time.perf_counter()
    schedule_result = await schedule_chat_partition_worker_async(partition, platform=platform)
    schedule_claim_elapsed_ms = int((time.perf_counter() - schedule_started_at) * 1000)
    if schedule_result.get("status") != "scheduled":
        return {**schedule_result, "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms}

    claim_token = str(schedule_result.get("claim_token") or "").strip()
    spawn_kwargs = {
        "platform": platform,
        "partition": partition,
        "max_items": max_items or default_chat_queue_batch_size,
        "claim_token": claim_token,
    }
    if int(warmup_wait_seconds or 0) > 0:
        spawn_kwargs["warmup_wait_seconds"] = int(warmup_wait_seconds)
        spawn_kwargs["warmup_metadata"] = dict(warmup_metadata or {})
    worker = get_process_chat_queue_handle(partition)
    try:
        if worker is not None and hasattr(worker, "spawn"):
            spawn_handle = getattr(worker, "spawn")
            spawn_started_at = time.perf_counter()
            if hasattr(spawn_handle, "aio"):
                await spawn_handle.aio(**spawn_kwargs)
            else:
                spawn_handle(**spawn_kwargs)
            spawn_rpc_elapsed_ms = int((time.perf_counter() - spawn_started_at) * 1000)
            return {
                **schedule_result,
                "spawned": True,
                "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms,
                "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
            }

        inline_started_at = time.perf_counter()
        process_chat_queue_impl(**spawn_kwargs)
        spawn_rpc_elapsed_ms = int((time.perf_counter() - inline_started_at) * 1000)
        return {
            **schedule_result,
            "spawned": False,
            "mode": "inline",
            "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms,
            "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
        }
    except Exception:
        await release_chat_partition_claim_async(partition, claim_token=claim_token)
        raise


async def spawn_chat_queue_worker_optimistic_async(
    *,
    platform: str,
    partition: str,
    max_items: int,
    warmup_wait_seconds: int,
    warmup_metadata: dict[str, Any] | None,
    default_chat_queue_batch_size: int,
    has_recent_chat_worker_spawn: Callable[[str], bool],
    peek_chat_partition_claim_async: Callable[..., Any],
    claim_is_recent: Callable[..., bool],
    active_claim_skip_seconds: int,
    get_process_chat_queue_handle: Callable[[str | None], Any],
    mark_recent_chat_worker_spawn: Callable[[str], None],
    process_chat_queue_impl: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    normalized_platform = str(platform or "").strip().lower()
    normalized_partition = str(partition or "").strip()
    if not normalized_partition:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "schedule_claim_elapsed_ms": 0,
            "spawn_rpc_elapsed_ms": 0,
        }
    if has_recent_chat_worker_spawn(normalized_partition):
        return {
            "status": "skipped",
            "reason": "recent_spawn_gate",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "spawned": False,
            "schedule_claim_elapsed_ms": 0,
            "spawn_rpc_elapsed_ms": 0,
        }

    existing_claim = await peek_chat_partition_claim_async(normalized_partition, refresh_on_miss=True)
    if existing_claim and claim_is_recent(existing_claim, max_age_seconds=active_claim_skip_seconds):
        return {
            "status": "skipped",
            "reason": f"active_{str(existing_claim.get('status') or 'claimed').strip().lower() or 'claimed'}",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "spawned": False,
            "claim_token": str(existing_claim.get('claim_token') or ''),
            "schedule_claim_elapsed_ms": 0,
            "spawn_rpc_elapsed_ms": 0,
        }

    spawn_kwargs = {
        "platform": normalized_platform,
        "partition": normalized_partition,
        "max_items": max_items or default_chat_queue_batch_size,
        "claim_token": None,
    }
    if int(warmup_wait_seconds or 0) > 0:
        spawn_kwargs["warmup_wait_seconds"] = int(warmup_wait_seconds)
        spawn_kwargs["warmup_metadata"] = dict(warmup_metadata or {})
    worker = get_process_chat_queue_handle(partition)
    if worker is not None and hasattr(worker, "spawn"):
        spawn_handle = getattr(worker, "spawn")
        spawn_started_at = time.perf_counter()
        if hasattr(spawn_handle, "aio"):
            await spawn_handle.aio(**spawn_kwargs)
        else:
            spawn_handle(**spawn_kwargs)
        spawn_rpc_elapsed_ms = int((time.perf_counter() - spawn_started_at) * 1000)
        mark_recent_chat_worker_spawn(normalized_partition)
        return {
            "status": "scheduled",
            "reason": "optimistic_spawned",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "spawned": True,
            "schedule_claim_elapsed_ms": 0,
            "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
        }

    inline_started_at = time.perf_counter()
    process_chat_queue_impl(**spawn_kwargs)
    spawn_rpc_elapsed_ms = int((time.perf_counter() - inline_started_at) * 1000)
    return {
        "status": "scheduled",
        "reason": "optimistic_inline",
        "platform": normalized_platform,
        "partition": normalized_partition,
        "spawned": False,
        "mode": "inline",
        "schedule_claim_elapsed_ms": 0,
        "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
    }


async def spawn_chat_queue_worker_background_task(
    *,
    payload: dict[str, Any],
    platform: str,
    partition: str,
    max_items: int,
    warmup_wait_seconds: int,
    warmup_metadata: dict[str, Any] | None,
    spawn_chat_queue_worker_async_fn: Callable[..., Any],
    append_trace: Callable[..., None],
    extract_feishu_event_metadata: Callable[[dict[str, Any]], tuple[str, str]],
    logger: Any,
) -> None:
    try:
        spawn_result = await spawn_chat_queue_worker_async_fn(
            platform=platform,
            partition=partition,
            max_items=max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
        )
        stage = "webhook.spawned" if spawn_result.get("status") == "scheduled" else "webhook.spawn_skipped"
        append_trace(stage, payload, partition=partition, reason=spawn_result.get("reason") or "")
        event_id, event_type = extract_feishu_event_metadata(payload)
        if spawn_result.get("status") == "scheduled":
            logger.warning(
                "[Feishu] webhook spawned chat worker event_type=%s event_id=%s partition=%s",
                event_type or "unknown",
                event_id or "none",
                partition,
            )
        else:
            logger.warning(
                "[Feishu] webhook skipped chat worker spawn event_type=%s event_id=%s partition=%s reason=%s",
                event_type or "unknown",
                event_id or "none",
                partition,
                spawn_result.get("reason") or "already_scheduled",
            )
    except Exception as exc:
        append_trace("webhook.spawn_error", payload, partition=partition, error=str(exc))
        event_id, event_type = extract_feishu_event_metadata(payload)
        logger.warning(
            "[Feishu] webhook spawn failed event_type=%s event_id=%s partition=%s error=%s",
            event_type or "unknown",
            event_id or "none",
            partition,
            exc,
            exc_info=True,
        )


def schedule_chat_queue_worker_background(
    *,
    payload: dict[str, Any],
    platform: str,
    partition: str,
    max_items: int,
    warmup_wait_seconds: int,
    warmup_metadata: dict[str, Any] | None,
    spawn_chat_queue_worker_background_task_fn: Callable[..., Any],
) -> bool:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    loop.create_task(
        spawn_chat_queue_worker_background_task_fn(
            payload=payload,
            platform=platform,
            partition=partition,
            max_items=max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
        )
    )
    return True


def extract_feishu_warmup_context(
    payload: dict[str, Any],
    *,
    extract_feishu_queue_context: Callable[[dict[str, Any]], dict[str, str]],
) -> dict[str, str]:
    context = extract_feishu_queue_context(payload)
    context["lane"] = "chat_light"
    chat_or_actor = str(context.get("chat_id") or context.get("actor_id") or context.get("event_id") or "unknown").strip()
    context["partition"] = f"feishu:chat_light:{chat_or_actor or 'unknown'}"
    context["started_at_ms"] = str(int(time.time() * 1000))
    if not str(context.get("partition") or "").strip():
        actor_id = str(context.get("actor_id") or "").strip()
        event_id = str(context.get("event_id") or "").strip()
        context["partition"] = f"feishu:{actor_id or event_id or 'warmup'}"
    return context


def warmup_chat_queue_worker_impl(
    *,
    platform: str,
    partition: str,
    reason: str,
    metadata: dict[str, Any] | None,
    bootstrap_chat_queue_worker_context: Callable[[], dict[str, Any]],
    build_chat_worker_observability: Callable[..., dict[str, Any]],
    default_chat_queue_warmup_wait_seconds: int,
) -> dict[str, Any]:
    worker_context = bootstrap_chat_queue_worker_context()
    payload: dict[str, Any] = {
        "status": "warming_inline",
        "platform": platform,
        "partition": partition,
        "reason": str(reason or "").strip() or "manual",
        "warmup_wait_seconds": default_chat_queue_warmup_wait_seconds,
        **build_chat_worker_observability(worker_context, batch_size=0),
    }
    if isinstance(metadata, dict) and metadata:
        payload["metadata"] = {
            key: value
            for key, value in metadata.items()
            if key in {"event_type", "event_id", "chat_id", "actor_id"}
        }
    return payload


async def spawn_chat_queue_warmup_async(
    *,
    platform: str,
    partition: str,
    reason: str,
    metadata: dict[str, Any] | None,
    spawn_chat_queue_worker_async_fn: Callable[..., Any],
    warmup_chat_queue_worker_impl_fn: Callable[..., dict[str, Any]],
    default_chat_queue_warmup_wait_seconds: int,
) -> dict[str, Any]:
    normalized_metadata = {
        key: value
        for key, value in dict(metadata or {}).items()
        if key in {"event_type", "event_id", "chat_id", "actor_id", "started_at_ms"}
    }
    scheduled = await spawn_chat_queue_worker_async_fn(
        platform=platform,
        partition=partition,
        max_items=1,
        warmup_wait_seconds=default_chat_queue_warmup_wait_seconds,
        warmup_metadata=normalized_metadata,
    )
    if scheduled.get("status") != "error":
        scheduled["reason"] = reason
        scheduled["metadata"] = {
            key: value
            for key, value in normalized_metadata.items()
            if key in {"event_type", "event_id", "chat_id", "actor_id"}
        }
        return scheduled

    inline_result = warmup_chat_queue_worker_impl_fn(
        platform=platform,
        partition=partition,
        reason=reason,
        metadata=metadata,
    )
    inline_result["spawned"] = False
    inline_result["mode"] = "inline"
    return inline_result
