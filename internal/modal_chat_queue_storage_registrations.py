from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_modal_chat_queue_storage_helpers(namespace: Namespace) -> None:
    def _load_chat_queue_claims() -> dict[str, Any]:
        return namespace["_chat_claims_load_claims"](
            load_json_file=namespace["_load_json_file"],
            path=namespace["CHAT_QUEUE_CLAIMS_PATH"],
        )

    def _save_chat_queue_claims(payload: dict[str, Any]) -> None:
        namespace["_chat_claims_save_claims"](
            payload,
            atomic_json_write=namespace["_atomic_json_write"],
            path=namespace["CHAT_QUEUE_CLAIMS_PATH"],
        )

    def _load_chat_queue_warmups() -> dict[str, Any]:
        return namespace["_chat_claims_load_warmups"](
            load_json_file=namespace["_load_json_file"],
            path=namespace["CHAT_QUEUE_WARMUPS_PATH"],
        )

    def _save_chat_queue_warmups(payload: dict[str, Any]) -> None:
        namespace["_chat_claims_save_warmups"](
            payload,
            atomic_json_write=namespace["_atomic_json_write"],
            path=namespace["CHAT_QUEUE_WARMUPS_PATH"],
        )

    def _prune_chat_queue_warmups(
        warmups: dict[str, Any],
        *,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        return namespace["_chat_claims_prune_warmups"](
            warmups,
            ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_WARMUP_TTL_SECONDS"] if ttl_seconds is None else ttl_seconds,
        )

    def _record_chat_queue_warmup_snapshot(
        partition_key: str,
        *,
        platform: str,
        status: str,
        metadata: dict[str, Any] | None = None,
        worker_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return namespace["_chat_claims_record_warmup_snapshot"](
            partition_key,
            platform=platform,
            status=status,
            metadata=metadata,
            worker_context=worker_context,
            lock=namespace["_CHAT_QUEUE_LOCK"],
            load_warmups=namespace["_load_chat_queue_warmups"],
            save_warmups=namespace["_save_chat_queue_warmups"],
            prune_warmups=namespace["_prune_chat_queue_warmups"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            first_non_empty_str=namespace["_first_non_empty_str"],
        )

    def _pop_chat_queue_warmup_snapshot(partition_key: str) -> dict[str, Any]:
        return namespace["_chat_claims_pop_warmup_snapshot"](
            partition_key,
            lock=namespace["_CHAT_QUEUE_LOCK"],
            load_warmups=namespace["_load_chat_queue_warmups"],
            save_warmups=namespace["_save_chat_queue_warmups"],
            prune_warmups=namespace["_prune_chat_queue_warmups"],
            sync_modal_volume=namespace["_sync_modal_volume"],
        )

    def _prune_chat_queue_claims(
        claims: dict[str, Any],
        *,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        return namespace["_chat_claims_prune_claims"](
            claims,
            ttl_seconds=namespace["DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS"] if ttl_seconds is None else ttl_seconds,
        )

    def _claim_is_recent(claim: Mapping[str, Any], *, max_age_seconds: int) -> bool:
        return namespace["_chat_claims_claim_is_recent"](
            claim,
            max_age_seconds=max_age_seconds,
        )

    def _claim_age_seconds(claim: Mapping[str, Any]) -> int:
        return namespace["_chat_claims_claim_age_seconds"](claim)

    def _prune_chat_claims_for_ttl(claims: dict[str, Any], current_ttl: int) -> dict[str, Any]:
        return namespace["_prune_chat_queue_claims"](claims, ttl_seconds=current_ttl)

    namespace["_load_chat_queue_claims"] = _load_chat_queue_claims
    namespace["_save_chat_queue_claims"] = _save_chat_queue_claims
    namespace["_load_chat_queue_warmups"] = _load_chat_queue_warmups
    namespace["_save_chat_queue_warmups"] = _save_chat_queue_warmups
    namespace["_prune_chat_queue_warmups"] = _prune_chat_queue_warmups
    namespace["_record_chat_queue_warmup_snapshot"] = _record_chat_queue_warmup_snapshot
    namespace["_pop_chat_queue_warmup_snapshot"] = _pop_chat_queue_warmup_snapshot
    namespace["_prune_chat_queue_claims"] = _prune_chat_queue_claims
    namespace["_claim_is_recent"] = _claim_is_recent
    namespace["_claim_age_seconds"] = _claim_age_seconds
    namespace["_prune_chat_claims_for_ttl"] = _prune_chat_claims_for_ttl
