from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_feishu_internal_runtime_helpers(namespace: Namespace) -> None:
    def _lookup_feishu_internal_result_file(token: str) -> dict[str, Any] | None:
        return namespace["_feishu_lookup_internal_result_file"](
            token,
            store=namespace["_FEISHU_INTERNAL_RESULT_FILES"],
            lock=namespace["_FEISHU_INTERNAL_RESULT_FILE_LOCK"],
        )

    async def _get_feishu_internal_gateway_runtime() -> Any:
        def _get_current_runtime() -> Any:
            return namespace["_FEISHU_INTERNAL_RUNTIME"]

        def _set_runtime(runtime: Any) -> None:
            namespace["_FEISHU_INTERNAL_RUNTIME"] = runtime

        return await namespace["_feishu_get_cached_runtime_bridge"](
            get_current_runtime=_get_current_runtime,
            get_lock=namespace["_get_feishu_internal_runtime_lock"],
            prepare_runtime_environment=namespace["_prepare_runtime_environment"],
            settings_from_env=namespace["RuntimeSettings"].from_env,
            initialize_runtime=namespace["_initialize_feishu_internal_gateway_runtime"],
            set_runtime=_set_runtime,
        )

    namespace["_lookup_feishu_internal_result_file"] = _lookup_feishu_internal_result_file
    namespace["_get_feishu_internal_gateway_runtime"] = _get_feishu_internal_gateway_runtime


def register_worker_bootstrap_helpers(namespace: Namespace) -> None:
    def _bootstrap_feishu_ingress_worker_context(
        snapshot_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        worker_started_at = int(time.time() * 1000)
        enter_started = time.time()
        runtime_prepare_started = time.time()
        namespace["_prepare_runtime_environment"]()
        runtime_prepare_elapsed_ms = int((time.time() - runtime_prepare_started) * 1000)

        queue_prewarm_started = time.time()
        try:
            asyncio.run(namespace["_prewarm_chat_queue_async"]())
            chat_queue_prewarm_ok = True
        except Exception as exc:
            namespace["logger"].warning("Feishu ingress worker queue prewarm failed: %s", exc)
            chat_queue_prewarm_ok = False
        chat_queue_prewarm_elapsed_ms = int((time.time() - queue_prewarm_started) * 1000)

        context = {
            "worker_boot_id": f"feishu-ingress-{uuid.uuid4().hex}",
            "worker_started_at": worker_started_at,
            "runtime_prepare_elapsed_ms": runtime_prepare_elapsed_ms,
            "enter_elapsed_ms": int((time.time() - enter_started) * 1000),
            "chat_queue_prewarm_ok": chat_queue_prewarm_ok,
            "chat_queue_prewarm_elapsed_ms": chat_queue_prewarm_elapsed_ms,
            "container_reused": False,
        }
        if isinstance(snapshot_context, dict):
            context.update({key: value for key, value in snapshot_context.items() if value is not None})
            context["snapshot_restored"] = True
        return context

    def _bootstrap_feishu_ack_reaction_worker_context(
        snapshot_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        worker_started_at = int(time.time() * 1000)
        enter_started = time.time()
        runtime_prepare_started = time.time()
        namespace["_prepare_runtime_environment"]()
        runtime_prepare_elapsed_ms = int((time.time() - runtime_prepare_started) * 1000)

        context = {
            "worker_boot_id": f"feishu-ack-{uuid.uuid4().hex}",
            "worker_started_at": worker_started_at,
            "runtime_prepare_elapsed_ms": runtime_prepare_elapsed_ms,
            "enter_elapsed_ms": int((time.time() - enter_started) * 1000),
            "container_reused": False,
        }
        if isinstance(snapshot_context, dict):
            context.update({key: value for key, value in snapshot_context.items() if value is not None})
            context["snapshot_restored"] = True
        return context

    namespace["_bootstrap_feishu_ingress_worker_context"] = _bootstrap_feishu_ingress_worker_context
    namespace["_bootstrap_feishu_ack_reaction_worker_context"] = _bootstrap_feishu_ack_reaction_worker_context


def register_chat_queue_processing_helpers(namespace: Namespace) -> None:
    async def _process_chat_queue_item_async(
        payload: Any,
        *,
        worker_context: dict[str, Any] | None = None,
        runtime_prepared: bool = False,
    ) -> dict[str, Any]:
        return await namespace["_chat_queue_process_item_async_bridge"](
            payload,
            worker_context=worker_context,
            runtime_prepared=runtime_prepared,
            build_inline_chat_worker_context=namespace["_build_inline_chat_worker_context"],
            is_truthy=namespace["_is_truthy"],
            sync_modal_volume_async=namespace["_sync_modal_volume_async"],
            build_chat_worker_observability=namespace["_build_chat_worker_observability"],
            append_feishu_trace=namespace["_append_feishu_trace"],
            dispatch_feishu_payload=lambda raw_payload, await_background_tasks: namespace["_dispatch_feishu_payload"](
                raw_payload,
                await_background_tasks=await_background_tasks,
            ),
            dispatch_telegram_update=namespace["_dispatch_telegram_update"],
            logger=namespace["logger"],
            feishu_chat_worker_timeout_seconds=namespace["DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS"],
        )

    async def _process_chat_queue_items_async(
        items: list[Any],
        *,
        worker_context: dict[str, Any] | None = None,
        runtime_prepared: bool = False,
    ) -> list[dict[str, Any]]:
        return await namespace["_chat_queue_process_items_async_bridge"](
            items,
            worker_context=worker_context,
            runtime_prepared=runtime_prepared,
            process_chat_queue_item_async_fn=namespace["_process_chat_queue_item_async"],
            coalesce_items=namespace["_coalesce_feishu_chat_queue_items"],
        )

    def _sample_light_p2p_feishu_message_payload() -> dict[str, Any]:
        return namespace["_chat_queue_sample_light_p2p_payload"]()

    def _warmup_chat_queue_worker_impl(
        *,
        platform: str,
        partition: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return namespace["_chat_queue_warmup_worker_impl"](
            platform=platform,
            partition=partition,
            reason=reason,
            metadata=metadata,
            bootstrap_chat_queue_worker_context=namespace["_bootstrap_chat_queue_worker_context"],
            build_chat_worker_observability=namespace["_build_chat_worker_observability"],
            default_chat_queue_warmup_wait_seconds=namespace["DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS"],
        )

    async def _spawn_chat_queue_warmup_async(
        *,
        platform: str,
        partition: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await namespace["_chat_queue_spawn_warmup_async"](
            platform=platform,
            partition=partition,
            reason=reason,
            metadata=metadata,
            spawn_chat_queue_worker_async_fn=namespace["_spawn_chat_queue_worker_async"],
            warmup_chat_queue_worker_impl_fn=namespace["_warmup_chat_queue_worker_impl"],
            default_chat_queue_warmup_wait_seconds=namespace["DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS"],
        )

    namespace["_process_chat_queue_item_async"] = _process_chat_queue_item_async
    namespace["_process_chat_queue_items_async"] = _process_chat_queue_items_async
    namespace["_sample_light_p2p_feishu_message_payload"] = _sample_light_p2p_feishu_message_payload
    namespace["_warmup_chat_queue_worker_impl"] = _warmup_chat_queue_worker_impl
    namespace["_spawn_chat_queue_warmup_async"] = _spawn_chat_queue_warmup_async


def register_chat_spawn_helpers(namespace: Namespace) -> None:
    def _spawn_chat_queue_worker_sync(
        *,
        platform: str,
        partition: str,
        max_items: int | None = None,
        warmup_wait_seconds: int = 0,
        warmup_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return namespace["_chat_queue_spawn_worker_sync"](
            platform=platform,
            partition=partition,
            max_items=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"] if max_items is None else max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
            default_chat_queue_batch_size=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"],
            schedule_chat_partition_worker=namespace["_schedule_chat_partition_worker"],
            get_process_chat_queue_handle=namespace["_get_process_chat_queue_handle"],
            process_chat_queue_impl=namespace["_process_chat_queue_impl"],
            release_chat_partition_claim=namespace["_release_chat_partition_claim"],
        )

    async def _spawn_chat_queue_worker_async(
        *,
        platform: str,
        partition: str,
        max_items: int | None = None,
        warmup_wait_seconds: int = 0,
        warmup_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await namespace["_chat_queue_spawn_worker_async"](
            platform=platform,
            partition=partition,
            max_items=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"] if max_items is None else max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
            default_chat_queue_batch_size=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"],
            schedule_chat_partition_worker_async=namespace["_schedule_chat_partition_worker_async"],
            get_process_chat_queue_handle=namespace["_get_process_chat_queue_handle"],
            process_chat_queue_impl=namespace["_process_chat_queue_impl"],
            release_chat_partition_claim_async=namespace["_release_chat_partition_claim_async"],
        )

    async def _spawn_chat_queue_worker_optimistic_async(
        *,
        platform: str,
        partition: str,
        max_items: int | None = None,
        warmup_wait_seconds: int = 0,
        warmup_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await namespace["_chat_queue_spawn_worker_optimistic_async"](
            platform=platform,
            partition=partition,
            max_items=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"] if max_items is None else max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
            default_chat_queue_batch_size=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"],
            has_recent_chat_worker_spawn=namespace["_has_recent_chat_worker_spawn"],
            peek_chat_partition_claim_async=namespace["_peek_chat_partition_claim_async"],
            claim_is_recent=namespace["_claim_is_recent"],
            active_claim_skip_seconds=namespace["DEFAULT_CHAT_QUEUE_ACTIVE_CLAIM_SKIP_SECONDS"],
            get_process_chat_queue_handle=namespace["_get_process_chat_queue_handle"],
            mark_recent_chat_worker_spawn=namespace["_mark_recent_chat_worker_spawn"],
            process_chat_queue_impl=namespace["_process_chat_queue_impl"],
        )

    async def _spawn_chat_queue_worker_background_task(
        *,
        payload: dict[str, Any],
        platform: str,
        partition: str,
        max_items: int | None = None,
        warmup_wait_seconds: int = 0,
        warmup_metadata: dict[str, Any] | None = None,
    ) -> None:
        await namespace["_chat_queue_spawn_worker_background_task"](
            payload=payload,
            platform=platform,
            partition=partition,
            max_items=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"] if max_items is None else max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
            spawn_chat_queue_worker_async_fn=namespace["_spawn_chat_queue_worker_async"],
            append_trace=namespace["_append_feishu_trace"],
            extract_feishu_event_metadata=namespace["_extract_feishu_event_metadata"],
            logger=namespace["logger"],
        )

    def _schedule_chat_queue_worker_background(
        *,
        payload: dict[str, Any],
        platform: str,
        partition: str,
        max_items: int | None = None,
        warmup_wait_seconds: int = 0,
        warmup_metadata: dict[str, Any] | None = None,
    ) -> bool:
        return namespace["_chat_queue_schedule_background"](
            payload=payload,
            platform=platform,
            partition=partition,
            max_items=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"] if max_items is None else max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
            spawn_chat_queue_worker_background_task_fn=namespace["_spawn_chat_queue_worker_background_task"],
        )

    def _extract_feishu_warmup_context(payload: dict[str, Any]) -> dict[str, str]:
        return namespace["_chat_queue_extract_feishu_warmup_context"](
            payload,
            extract_feishu_queue_context=namespace["_extract_feishu_queue_context"],
        )

    namespace["_spawn_chat_queue_worker_sync"] = _spawn_chat_queue_worker_sync
    namespace["_spawn_chat_queue_worker_async"] = _spawn_chat_queue_worker_async
    namespace["_spawn_chat_queue_worker_optimistic_async"] = _spawn_chat_queue_worker_optimistic_async
    namespace["_spawn_chat_queue_worker_background_task"] = _spawn_chat_queue_worker_background_task
    namespace["_schedule_chat_queue_worker_background"] = _schedule_chat_queue_worker_background
    namespace["_extract_feishu_warmup_context"] = _extract_feishu_warmup_context
