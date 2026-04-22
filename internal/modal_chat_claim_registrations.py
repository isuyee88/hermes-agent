from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_chat_claim_helpers(namespace: Namespace) -> None:
    def _claim_is_stale_for_takeover(
        claim: dict[str, Any],
        *,
        stale_after_seconds: int | None = None,
    ) -> bool:
        return namespace["_chat_claims_claim_is_stale_for_takeover"](
            claim,
            stale_after_seconds=stale_after_seconds,
            default_stale_after_seconds=namespace["DEFAULT_CHAT_QUEUE_STALE_CLAIM_TAKEOVER_SECONDS"],
            claim_age_seconds=namespace["_claim_age_seconds"],
        )

    def _peek_chat_partition_claim(
        partition_key: str,
        *,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        return namespace["_chat_claims_peek_claim"](
            partition_key,
            ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"] if ttl_seconds is None else ttl_seconds,
            lock=namespace["_CHAT_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            prune_claims=namespace["_prune_chat_claims_for_ttl"],
            load_claims=namespace["_load_chat_queue_claims"],
        )

    async def _peek_chat_partition_claim_async(
        partition_key: str,
        *,
        ttl_seconds: int | None = None,
        refresh_on_miss: bool = False,
    ) -> dict[str, Any]:
        return await namespace["_chat_claims_peek_claim_async"](
            partition_key,
            ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"] if ttl_seconds is None else ttl_seconds,
            refresh_on_miss=refresh_on_miss,
            lock=namespace["_CHAT_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume_async=namespace["_sync_modal_volume_async"],
            prune_claims=namespace["_prune_chat_claims_for_ttl"],
            load_claims=namespace["_load_chat_queue_claims"],
        )

    def _has_recent_chat_worker_spawn(
        partition_key: str,
        *,
        ttl_seconds: float | None = None,
    ) -> bool:
        return namespace["_chat_claims_has_recent_worker_spawn"](
            partition_key,
            ttl_seconds=namespace["DEFAULT_FEISHU_RECENT_SPAWN_SKIP_SECONDS"] if ttl_seconds is None else ttl_seconds,
            recent_worker_spawns=namespace["_RECENT_CHAT_WORKER_SPAWNS"],
            recent_worker_spawns_lock=namespace["_RECENT_CHAT_WORKER_SPAWNS_LOCK"],
        )

    def _mark_recent_chat_worker_spawn(partition_key: str) -> None:
        namespace["_chat_claims_mark_recent_worker_spawn"](
            partition_key,
            recent_worker_spawns=namespace["_RECENT_CHAT_WORKER_SPAWNS"],
            recent_worker_spawns_lock=namespace["_RECENT_CHAT_WORKER_SPAWNS_LOCK"],
        )

    def _claim_chat_partition(
        partition_key: str,
        *,
        platform: str,
        claim_token: str | None = None,
        ttl_seconds: int | None = None,
    ) -> tuple[bool, str]:
        return namespace["_chat_claims_claim_partition"](
            partition_key,
            platform=platform,
            claim_token=claim_token,
            ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"] if ttl_seconds is None else ttl_seconds,
            lock=namespace["_CHAT_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            prune_claims=namespace["_prune_chat_claims_for_ttl"],
            load_claims=namespace["_load_chat_queue_claims"],
            save_claims=namespace["_save_chat_queue_claims"],
            claim_is_stale_for_takeover=namespace["_claim_is_stale_for_takeover"],
            claim_age_seconds=namespace["_claim_age_seconds"],
            logger=namespace["logger"],
        )

    def _schedule_chat_partition_worker(
        partition_key: str,
        *,
        platform: str,
        cooldown_seconds: int | None = None,
    ) -> dict[str, Any]:
        return namespace["_chat_claims_schedule_partition_worker"](
            partition_key,
            platform=platform,
            cooldown_seconds=(
                namespace["DEFAULT_CHAT_QUEUE_SPAWN_COOLDOWN_SECONDS"] if cooldown_seconds is None else cooldown_seconds
            ),
            lock=namespace["_CHAT_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            prune_claims=namespace["_prune_chat_claims_for_ttl"],
            load_claims=namespace["_load_chat_queue_claims"],
            save_claims=namespace["_save_chat_queue_claims"],
            claim_is_stale_for_takeover=namespace["_claim_is_stale_for_takeover"],
            claim_age_seconds=namespace["_claim_age_seconds"],
            logger=namespace["logger"],
            claim_ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"],
        )

    async def _schedule_chat_partition_worker_async(
        partition_key: str,
        *,
        platform: str,
        cooldown_seconds: int | None = None,
    ) -> dict[str, Any]:
        return await namespace["_chat_claims_schedule_partition_worker_async"](
            partition_key,
            platform=platform,
            cooldown_seconds=(
                namespace["DEFAULT_CHAT_QUEUE_SPAWN_COOLDOWN_SECONDS"] if cooldown_seconds is None else cooldown_seconds
            ),
            lock=namespace["_CHAT_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume_async=namespace["_sync_modal_volume_async"],
            prune_claims=namespace["_prune_chat_claims_for_ttl"],
            load_claims=namespace["_load_chat_queue_claims"],
            save_claims=namespace["_save_chat_queue_claims"],
            claim_is_stale_for_takeover=namespace["_claim_is_stale_for_takeover"],
            claim_age_seconds=namespace["_claim_age_seconds"],
            logger=namespace["logger"],
            claim_ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"],
        )

    def _release_chat_partition_claim(
        partition_key: str,
        *,
        claim_token: str | None = None,
    ) -> None:
        namespace["_chat_claims_release_claim"](
            partition_key,
            claim_token=claim_token,
            lock=namespace["_CHAT_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            prune_claims=namespace["_prune_chat_claims_for_ttl"],
            load_claims=namespace["_load_chat_queue_claims"],
            save_claims=namespace["_save_chat_queue_claims"],
            claim_ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"],
        )

    async def _release_chat_partition_claim_async(
        partition_key: str,
        *,
        claim_token: str | None = None,
    ) -> None:
        await namespace["_chat_claims_release_claim_async"](
            partition_key,
            claim_token=claim_token,
            lock=namespace["_CHAT_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume_async=namespace["_sync_modal_volume_async"],
            prune_claims=namespace["_prune_chat_claims_for_ttl"],
            load_claims=namespace["_load_chat_queue_claims"],
            save_claims=namespace["_save_chat_queue_claims"],
            claim_ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"],
        )

    def _extract_feishu_queue_context(payload: dict[str, Any]) -> dict[str, str]:
        return namespace["_chat_queue_extract_feishu_queue_context"](
            payload,
            extract_trace_context=namespace["_extract_feishu_trace_context"],
            collect_chat_id=namespace["_collect_feishu_chat_id"],
            collect_actor_id=namespace["_collect_feishu_actor_id"],
            classify_chat_lane=namespace["_classify_feishu_chat_lane"],
        )

    def _enqueue_chat_event(
        *,
        platform: str,
        partition: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        include_queue_depth: bool = True,
    ) -> dict[str, Any]:
        return namespace["_chat_queue_enqueue_event"](
            platform=platform,
            partition=partition,
            payload=payload,
            metadata=metadata,
            include_queue_depth=include_queue_depth,
            get_chat_queue=namespace["_get_chat_queue"],
            safe_chat_queue_depth=namespace["_safe_chat_queue_depth"],
            default_chat_queue_claim_ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"],
        )

    async def _enqueue_chat_event_async(
        *,
        platform: str,
        partition: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        include_queue_depth: bool = True,
    ) -> dict[str, Any]:
        return await namespace["_chat_queue_enqueue_event_async"](
            platform=platform,
            partition=partition,
            payload=payload,
            metadata=metadata,
            include_queue_depth=include_queue_depth,
            get_chat_queue=namespace["_get_chat_queue"],
            safe_chat_queue_depth_async=namespace["_safe_chat_queue_depth_async"],
            enqueue_chat_event_fn=namespace["_enqueue_chat_event"],
            default_chat_queue_claim_ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"],
        )

    namespace["_claim_is_stale_for_takeover"] = _claim_is_stale_for_takeover
    namespace["_peek_chat_partition_claim"] = _peek_chat_partition_claim
    namespace["_peek_chat_partition_claim_async"] = _peek_chat_partition_claim_async
    namespace["_has_recent_chat_worker_spawn"] = _has_recent_chat_worker_spawn
    namespace["_mark_recent_chat_worker_spawn"] = _mark_recent_chat_worker_spawn
    namespace["_claim_chat_partition"] = _claim_chat_partition
    namespace["_schedule_chat_partition_worker"] = _schedule_chat_partition_worker
    namespace["_schedule_chat_partition_worker_async"] = _schedule_chat_partition_worker_async
    namespace["_release_chat_partition_claim"] = _release_chat_partition_claim
    namespace["_release_chat_partition_claim_async"] = _release_chat_partition_claim_async
    namespace["_extract_feishu_queue_context"] = _extract_feishu_queue_context
    namespace["_extract_telegram_queue_context"] = namespace["_chat_queue_extract_telegram_queue_context"]
    namespace["_enqueue_chat_event"] = _enqueue_chat_event
    namespace["_enqueue_chat_event_async"] = _enqueue_chat_event_async
