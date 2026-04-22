from __future__ import annotations

import asyncio
import queue as pyqueue
import time
import uuid
from typing import Any, Callable


def bootstrap_chat_queue_worker_context(
    snapshot_context: dict[str, Any] | None = None,
    *,
    worker_partition_key: str | None = None,
    prepare_runtime_environment: Callable[[], None],
    load_routing_state: Callable[[], dict[str, Any]],
    settings_from_env: Callable[[], Any],
    get_feishu_gateway_runtime: Callable[[], Any],
    chat_worker_preinit_feishu_runtime_enabled: bool,
    logger: Any,
) -> dict[str, Any]:
    worker_started_at = int(time.time() * 1000)
    enter_started = time.time()
    runtime_prepare_started = time.time()
    prepare_runtime_environment()
    runtime_prepare_elapsed_ms = int((time.time() - runtime_prepare_started) * 1000)

    context: dict[str, Any] = {
        "worker_boot_id": uuid.uuid4().hex,
        "worker_started_at": worker_started_at,
        "runtime_prepare_elapsed_ms": runtime_prepare_elapsed_ms,
        "container_reused": False,
    }
    normalized_partition_key = str(worker_partition_key or "").strip()
    if normalized_partition_key:
        context["worker_partition_key"] = normalized_partition_key
    if isinstance(snapshot_context, dict):
        context.update({key: value for key, value in snapshot_context.items() if value is not None})
        context["snapshot_restored"] = True

    routing_state = load_routing_state()
    routing_state_refreshed_at = int(routing_state.get("refreshed_at") or 0) if isinstance(routing_state, dict) else 0
    if routing_state_refreshed_at:
        context["routing_state_refreshed_at"] = routing_state_refreshed_at

    try:
        from tools.feishu_api import load_feishu_model_registry

        registry_payload = load_feishu_model_registry(force_refresh=False)
        context["model_registry_entry_count"] = len(registry_payload.get("entries") or [])
        model_registry_source = str(registry_payload.get("source") or "").strip()
        if model_registry_source:
            context["model_registry_source"] = model_registry_source
    except Exception as exc:
        logger.warning("Chat queue worker model registry preload failed: %s", exc)

    if chat_worker_preinit_feishu_runtime_enabled:
        settings = settings_from_env()
        if settings.feishu_app_id and settings.feishu_app_secret:
            preinit_started = time.time()
            try:
                asyncio.run(get_feishu_gateway_runtime())
                context["feishu_runtime_preinit_ok"] = True
            except Exception as exc:
                context["feishu_runtime_preinit_ok"] = False
                context["feishu_runtime_preinit_error"] = str(exc)
                logger.warning("Chat queue worker Feishu runtime preinit failed: %s", exc)
            context["feishu_runtime_preinit_elapsed_ms"] = int((time.time() - preinit_started) * 1000)

    context["enter_elapsed_ms"] = int((time.time() - enter_started) * 1000)
    return context


def get_process_chat_queue_handle(
    *,
    partition: str | None = None,
    default_handle: Any,
    worker_cls: Any,
    logger: Any,
) -> Any:
    normalized_partition = str(partition or "").strip()
    if not normalized_partition:
        return default_handle
    if worker_cls is None:
        return default_handle
    if default_handle is not None and type(default_handle).__name__ != "Function":
        return default_handle
    try:
        return worker_cls(partition_key=normalized_partition).process
    except Exception as exc:
        logger.warning(
            "[ChatQueue] parameterized worker handle fallback partition=%s error=%s",
            normalized_partition,
            exc,
        )
        return default_handle


def process_chat_queue(
    *,
    platform: str,
    partition: str,
    max_items: int,
    claim_token: str | None = None,
    worker_context: dict[str, Any] | None = None,
    runtime_prepared: bool = False,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
    default_chat_queue_linger_seconds: float,
    build_chat_worker_observability: Callable[..., dict[str, Any]],
    build_inline_chat_worker_context: Callable[[], dict[str, Any]],
    pop_chat_queue_warmup_snapshot: Callable[[str], dict[str, Any] | None],
    first_non_empty_str: Callable[[Any], str],
    record_chat_queue_warmup_snapshot: Callable[..., dict[str, Any] | None],
    safe_chat_queue_depth: Callable[[], int | None],
    claim_chat_partition: Callable[..., tuple[bool, str | None]],
    get_chat_queue: Callable[[], Any],
    process_chat_queue_items_async: Callable[..., Any],
    sync_modal_volume: Callable[..., None],
    release_chat_partition_claim: Callable[..., None],
    logger: Any,
) -> dict[str, Any]:
    normalized_platform = str(platform or "").strip().lower()
    normalized_partition = str(partition or "").strip()
    if not normalized_platform or not normalized_partition:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": normalized_platform,
            "partition": normalized_partition,
            **build_chat_worker_observability(worker_context),
        }

    context = dict(worker_context or {})
    if not runtime_prepared:
        inline_context = build_inline_chat_worker_context()
        inline_context.update(context)
        context = inline_context

    normalized_warmup_wait_seconds = max(int(warmup_wait_seconds or 0), 0)
    normalized_linger_seconds = max(float(default_chat_queue_linger_seconds or 0.0), 0.0)
    warmup_payload = dict(warmup_metadata or {})
    is_waiting_warmup = normalized_warmup_wait_seconds > 0
    if is_waiting_warmup:
        started_at_ms = int(warmup_payload.get("started_at_ms") or int(time.time() * 1000))
        context.update(
            {
                "event_id": context.get("event_id") or warmup_payload.get("event_id"),
                "event_type": context.get("event_type") or warmup_payload.get("event_type"),
                "actor_id": context.get("actor_id") or warmup_payload.get("actor_id"),
                "chat_id": context.get("chat_id") or warmup_payload.get("chat_id"),
                "started_at_ms": started_at_ms,
                "warmup_status": "waiting",
                "warmup_event_id": first_non_empty_str(warmup_payload.get("event_id")),
                "warmup_started_at": started_at_ms,
                "warmup_actor_id": first_non_empty_str(warmup_payload.get("actor_id")),
                "warmup_chat_id": first_non_empty_str(warmup_payload.get("chat_id")),
            }
        )

    warmup_snapshot = pop_chat_queue_warmup_snapshot(normalized_partition)
    if warmup_snapshot:
        ready_at_ms = int(warmup_snapshot.get("ready_at_ms") or warmup_snapshot.get("updated_at_ms") or 0)
        now_ms = int(time.time() * 1000)
        context.update(
            {
                "warmup_status": str(warmup_snapshot.get("status") or "ready").strip().lower(),
                "warmup_event_id": first_non_empty_str(warmup_snapshot.get("event_id")),
                "warmup_started_at": int(warmup_snapshot.get("started_at_ms") or 0) or None,
                "warmup_ready_at": ready_at_ms or None,
                "warmup_age_ms": max(0, now_ms - ready_at_ms) if ready_at_ms else None,
                "warmup_same_container": bool(
                    warmup_snapshot.get("worker_boot_id")
                    and warmup_snapshot.get("worker_boot_id") == context.get("worker_boot_id")
                ),
                "warmup_actor_id": first_non_empty_str(warmup_snapshot.get("actor_id")),
                "warmup_chat_id": first_non_empty_str(warmup_snapshot.get("chat_id")),
            }
        )

    if int(max_items or 0) <= 0:
        snapshot = record_chat_queue_warmup_snapshot(
            normalized_partition,
            platform=normalized_platform,
            status="ready",
            metadata={
                "event_id": context.get("event_id"),
                "event_type": context.get("event_type"),
                "actor_id": context.get("actor_id"),
                "chat_id": context.get("chat_id"),
                "started_at_ms": context.get("started_at_ms"),
            },
            worker_context=context,
        )
        if snapshot:
            context.update(
                {
                    "warmup_status": str(snapshot.get("status") or "ready").strip().lower(),
                    "warmup_event_id": first_non_empty_str(snapshot.get("event_id")),
                    "warmup_started_at": int(snapshot.get("started_at_ms") or 0) or None,
                    "warmup_ready_at": int(snapshot.get("ready_at_ms") or 0) or None,
                    "warmup_age_ms": 0,
                    "warmup_same_container": True,
                    "warmup_actor_id": first_non_empty_str(snapshot.get("actor_id")),
                    "warmup_chat_id": first_non_empty_str(snapshot.get("chat_id")),
                }
            )
        logger.warning(
            "[ChatQueue] warmup ready platform=%s partition=%s worker_boot_id=%s reused=%s enter_elapsed_ms=%s actor_id=%s chat_id=%s",
            normalized_platform or "unknown",
            normalized_partition or "none",
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
            context.get("enter_elapsed_ms"),
            context.get("warmup_actor_id") or "unknown",
            context.get("warmup_chat_id") or "unknown",
        )
        return {
            "status": "warmed",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "processed_count": 0,
            "results": [],
            "queue_depth": safe_chat_queue_depth(),
            **build_chat_worker_observability(context, batch_size=0),
        }

    claimed, effective_claim_token = claim_chat_partition(
        normalized_partition,
        platform=normalized_platform,
        claim_token=claim_token,
    )
    if not claimed:
        logger.warning(
            "[ChatQueue] worker skipped platform=%s partition=%s reason=already_claimed",
            normalized_platform or "unknown",
            normalized_partition or "none",
        )
        return {
            "status": "skipped",
            "reason": "already_claimed",
            "platform": normalized_platform,
            "partition": normalized_partition,
            **build_chat_worker_observability(context),
        }

    logger.warning(
        "[ChatQueue] worker claimed platform=%s partition=%s worker_pool=%s worker_boot_id=%s reused=%s enter_elapsed_ms=%s feishu_runtime_preinit_ok=%s feishu_runtime_preinit_elapsed_ms=%s",
        normalized_platform or "unknown",
        normalized_partition or "none",
        context.get("worker_partition_key") or "shared",
        context.get("worker_boot_id") or "inline",
        bool(context.get("container_reused", False)),
        context.get("enter_elapsed_ms"),
        context.get("feishu_runtime_preinit_ok"),
        context.get("feishu_runtime_preinit_elapsed_ms"),
    )

    queue = get_chat_queue()
    processed: list[dict[str, Any]] = []
    poll_timed_out = False
    try:
        first_poll = True
        while True:
            try:
                if first_poll:
                    try:
                        items = queue.get_many(
                            max(max_items, 1),
                            block=True,
                            timeout=float(normalized_warmup_wait_seconds or 2.0),
                            partition=normalized_partition,
                        )
                    except TypeError:
                        items = queue.get_many(max(max_items, 1), block=True, partition=normalized_partition)
                    first_poll = False
                elif processed and normalized_linger_seconds > 0:
                    try:
                        items = queue.get_many(
                            max(max_items, 1),
                            block=True,
                            timeout=normalized_linger_seconds,
                            partition=normalized_partition,
                        )
                    except TypeError:
                        items = queue.get_many(max(max_items, 1), block=True, partition=normalized_partition)
                else:
                    items = queue.get_many(max(max_items, 1), block=False, partition=normalized_partition)
            except pyqueue.Empty:
                poll_timed_out = True
                if is_waiting_warmup and not processed:
                    context["warmup_status"] = "timeout"
                if processed and normalized_linger_seconds > 0:
                    logger.warning(
                        "[ChatQueue] worker linger timed out platform=%s partition=%s linger_seconds=%s processed_count=%s",
                        normalized_platform or "unknown",
                        normalized_partition or "none",
                        normalized_linger_seconds,
                        len(processed),
                    )
                else:
                    logger.warning(
                        "[ChatQueue] worker poll timed out platform=%s partition=%s warmup_wait_seconds=%s",
                        normalized_platform or "unknown",
                        normalized_partition or "none",
                        normalized_warmup_wait_seconds if is_waiting_warmup else 0,
                    )
                break
            if not items:
                if is_waiting_warmup and not processed:
                    context["warmup_status"] = "empty"
                logger.warning(
                    "[ChatQueue] worker found no items platform=%s partition=%s",
                    normalized_platform or "unknown",
                    normalized_partition or "none",
                )
                break
            if is_waiting_warmup and not processed:
                now_ms = int(time.time() * 1000)
                started_at_ms = int(context.get("warmup_started_at") or context.get("started_at_ms") or now_ms)
                context.update(
                    {
                        "warmup_status": "hit",
                        "warmup_ready_at": now_ms,
                        "warmup_age_ms": max(0, now_ms - started_at_ms),
                        "warmup_same_container": True,
                    }
                )
                logger.warning(
                    "[ChatQueue] warmup hit platform=%s partition=%s worker_boot_id=%s waited_ms=%s",
                    normalized_platform or "unknown",
                    normalized_partition or "none",
                    context.get("worker_boot_id") or "inline",
                    context.get("warmup_age_ms") if context.get("warmup_age_ms") is not None else "na",
                )
            processed.extend(
                asyncio.run(
                    process_chat_queue_items_async(
                        items,
                        worker_context=context,
                        runtime_prepared=True,
                    )
                )
            )
    finally:
        sync_modal_volume(commit=True)
        release_chat_partition_claim(normalized_partition, claim_token=effective_claim_token)

    logger.warning(
        "[ChatQueue] worker released platform=%s partition=%s processed_count=%s worker_boot_id=%s reused=%s",
        normalized_platform or "unknown",
        normalized_partition or "none",
        len(processed),
        context.get("worker_boot_id") or "inline",
        bool(context.get("container_reused", False)),
    )
    if is_waiting_warmup and not processed and poll_timed_out:
        return {
            "status": "warmup_timeout",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "processed_count": 0,
            "results": [],
            "queue_depth": safe_chat_queue_depth(),
            **build_chat_worker_observability(context, batch_size=0),
        }
    return {
        "status": "ok",
        "platform": normalized_platform,
        "partition": normalized_partition,
        "processed_count": len(processed),
        "results": processed,
        "queue_depth": safe_chat_queue_depth(),
        **build_chat_worker_observability(context, batch_size=len(processed)),
    }
