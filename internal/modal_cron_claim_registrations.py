from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_modal_cron_claim_helpers(namespace: Namespace) -> None:
    def _load_cron_queue_claims() -> dict[str, Any]:
        payload = namespace["_load_json_file"](namespace["CRON_QUEUE_CLAIMS_PATH"], {})
        return payload if isinstance(payload, dict) else {}

    def _save_cron_queue_claims(payload: dict[str, Any]) -> None:
        namespace["_atomic_json_write"](namespace["CRON_QUEUE_CLAIMS_PATH"], payload)

    def _prune_cron_queue_claims(
        claims: dict[str, Any],
        *,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        return namespace["_cron_prune_claims"](
            claims,
            ttl_seconds=namespace["DEFAULT_CRON_QUEUE_CLAIM_TTL_SECONDS"] if ttl_seconds is None else ttl_seconds,
        )

    def _cleanup_orphan_cron_queue_claims(live_job_ids: set[str]) -> dict[str, Any]:
        return namespace["_cron_cleanup_orphan_claims"](
            live_job_ids,
            lock=namespace["_CRON_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            load_cron_queue_claims=namespace["_load_cron_queue_claims"],
            save_cron_queue_claims=namespace["_save_cron_queue_claims"],
            prune_cron_queue_claims_fn=namespace["_prune_cron_queue_claims"],
        )

    def _make_cron_claim_token(job: dict[str, Any]) -> str:
        return f"{job.get('id', '')}:{job.get('next_run_at', '')}"

    def _claim_due_cron_job(
        job: dict[str, Any],
        *,
        ttl_seconds: int | None = None,
    ) -> tuple[bool, str]:
        effective_ttl = namespace["DEFAULT_CRON_QUEUE_CLAIM_TTL_SECONDS"] if ttl_seconds is None else ttl_seconds
        return namespace["_cron_claim_due_job"](
            job,
            ttl_seconds=effective_ttl,
            lock=namespace["_CRON_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            load_cron_queue_claims=namespace["_load_cron_queue_claims"],
            save_cron_queue_claims=namespace["_save_cron_queue_claims"],
            prune_cron_queue_claims_fn=lambda claims: namespace["_prune_cron_queue_claims"](
                claims,
                ttl_seconds=effective_ttl,
            ),
        )

    def _release_cron_job_claim(job_id: str, *, claim_token: str | None = None) -> None:
        namespace["_cron_release_job_claim"](
            job_id,
            claim_token=claim_token,
            lock=namespace["_CRON_QUEUE_LOCK"],
            should_reload_modal_volume_for_claims=namespace["_should_reload_modal_volume_for_claims"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            load_cron_queue_claims=namespace["_load_cron_queue_claims"],
            save_cron_queue_claims=namespace["_save_cron_queue_claims"],
            prune_cron_queue_claims_fn=namespace["_prune_cron_queue_claims"],
        )

    namespace["_load_cron_queue_claims"] = _load_cron_queue_claims
    namespace["_save_cron_queue_claims"] = _save_cron_queue_claims
    namespace["_prune_cron_queue_claims"] = _prune_cron_queue_claims
    namespace["_cleanup_orphan_cron_queue_claims"] = _cleanup_orphan_cron_queue_claims
    namespace["_make_cron_claim_token"] = _make_cron_claim_token
    namespace["_claim_due_cron_job"] = _claim_due_cron_job
    namespace["_release_cron_job_claim"] = _release_cron_job_claim
