from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Callable


def prune_cron_queue_claims(
    claims: dict[str, Any],
    *,
    ttl_seconds: int,
) -> dict[str, Any]:
    now = int(time.time())
    pruned: dict[str, Any] = {}
    for job_id, claim in claims.items():
        if not isinstance(claim, dict):
            continue
        claimed_at = int(claim.get("claimed_at") or 0)
        if claimed_at and now - claimed_at < ttl_seconds:
            pruned[job_id] = claim
    return pruned


def cleanup_orphan_cron_queue_claims(
    live_job_ids: set[str],
    *,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    load_cron_queue_claims: Callable[[], dict[str, Any]],
    save_cron_queue_claims: Callable[[dict[str, Any]], None],
    prune_cron_queue_claims_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    with lock:
        if should_reload_modal_volume_for_claims("cron"):
            sync_modal_volume(reload=True)
        claims = prune_cron_queue_claims_fn(load_cron_queue_claims())
        filtered = {job_id: claim for job_id, claim in claims.items() if job_id in live_job_ids}
        if filtered != claims:
            save_cron_queue_claims(filtered)
            sync_modal_volume(commit=True)
        return filtered


def make_cron_claim_token(job: dict[str, Any]) -> str:
    return f"{job.get('id', '')}:{job.get('next_run_at', '')}"


def claim_due_cron_job(
    job: dict[str, Any],
    *,
    ttl_seconds: int,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    load_cron_queue_claims: Callable[[], dict[str, Any]],
    save_cron_queue_claims: Callable[[dict[str, Any]], None],
    prune_cron_queue_claims_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[bool, str]:
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        return False, ""

    claim_token = make_cron_claim_token(job)
    now = int(time.time())

    with lock:
        if should_reload_modal_volume_for_claims("cron"):
            sync_modal_volume(reload=True)
        claims = prune_cron_queue_claims_fn(load_cron_queue_claims())
        existing = claims.get(job_id) or {}
        if existing.get("claim_token") == claim_token:
            return False, claim_token

        claims[job_id] = {
            "claim_token": claim_token,
            "claimed_at": now,
            "next_run_at": job.get("next_run_at"),
            "job_name": job.get("name"),
        }
        save_cron_queue_claims(claims)
        sync_modal_volume(commit=True)
    return True, claim_token


def release_cron_job_claim(
    job_id: str,
    *,
    claim_token: str | None = None,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    load_cron_queue_claims: Callable[[], dict[str, Any]],
    save_cron_queue_claims: Callable[[dict[str, Any]], None],
    prune_cron_queue_claims_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    normalized = str(job_id or "").strip()
    if not normalized:
        return

    with lock:
        if should_reload_modal_volume_for_claims("cron"):
            sync_modal_volume(reload=True)
        claims = prune_cron_queue_claims_fn(load_cron_queue_claims())
        existing = claims.get(normalized)
        if not existing:
            return
        if claim_token and existing.get("claim_token") != claim_token:
            return
        claims.pop(normalized, None)
        save_cron_queue_claims(claims)
        sync_modal_volume(commit=True)


def cron_status_impl(
    *,
    limit: int,
    default_cron_queue_name: str,
    prepare_runtime_environment: Callable[[], None],
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    safe_cron_queue_depth: Callable[[], int | None],
    cleanup_orphan_cron_queue_claims_fn: Callable[[set[str]], dict[str, Any]],
) -> dict[str, Any]:
    prepare_runtime_environment()
    if should_reload_modal_volume_for_claims("cron"):
        sync_modal_volume(reload=True)

    from cron.jobs import get_due_jobs, list_jobs

    jobs = list_jobs(include_disabled=True)
    live_job_ids = {str(job.get("id") or "").strip() for job in jobs if job.get("id")}
    due_jobs = get_due_jobs()
    queue_depth = safe_cron_queue_depth()
    claims = cleanup_orphan_cron_queue_claims_fn(live_job_ids)
    summarized_jobs = []
    for job in jobs[: max(limit, 0)]:
        summarized_jobs.append(
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "state": job.get("state"),
                "enabled": job.get("enabled", True),
                "deliver": job.get("deliver"),
                "schedule": job.get("schedule_display"),
                "next_run_at": job.get("next_run_at"),
                "last_run_at": job.get("last_run_at"),
                "last_status": job.get("last_status"),
                "last_delivery_error": job.get("last_delivery_error"),
            }
        )

    return {
        "status": "ok",
        "queue_name": default_cron_queue_name,
        "queue_depth": queue_depth,
        "claim_count": len(claims),
        "due_count": len(due_jobs),
        "due_jobs": [
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "next_run_at": job.get("next_run_at"),
                "schedule": job.get("schedule_display"),
            }
            for job in due_jobs[: max(limit, 0)]
        ],
        "jobs": summarized_jobs,
    }


def enqueue_due_cron_jobs_impl(
    *,
    limit: int,
    default_cron_queue_claim_ttl_seconds: int,
    prepare_runtime_environment: Callable[[], None],
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    get_cron_queue: Callable[[], Any],
    claim_due_cron_job_fn: Callable[[dict[str, Any]], tuple[bool, str]],
    safe_cron_queue_depth: Callable[[], int | None],
) -> dict[str, Any]:
    prepare_runtime_environment()
    if should_reload_modal_volume_for_claims("cron"):
        sync_modal_volume(reload=True)

    from cron.jobs import get_due_jobs

    due_jobs = get_due_jobs()
    queue = get_cron_queue()
    enqueued: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for job in due_jobs[: max(limit, 0)]:
        claimed, claim_token = claim_due_cron_job_fn(job)
        if not claimed:
            skipped.append({"id": job.get("id"), "name": job.get("name"), "reason": "already_claimed"})
            continue

        payload = {
            "job_id": job.get("id"),
            "job_name": job.get("name"),
            "scheduled_for": job.get("next_run_at"),
            "claim_token": claim_token,
            "enqueued_at": int(time.time()),
        }
        queue.put(payload, partition_ttl=max(default_cron_queue_claim_ttl_seconds, 3600))
        enqueued.append(payload)

    return {
        "status": "ok",
        "due_count": len(due_jobs),
        "enqueued_count": len(enqueued),
        "skipped_count": len(skipped),
        "enqueued": enqueued,
        "skipped": skipped,
        "queue_depth": safe_cron_queue_depth(),
    }


def should_process_queued_cron_job(job: dict[str, Any], payload: dict[str, Any]) -> tuple[bool, str | None]:
    if not job:
        return False, "job_not_found"
    if not job.get("enabled", True):
        return False, "job_disabled"

    scheduled_for = str(payload.get("scheduled_for") or "").strip()
    current_next = str(job.get("next_run_at") or "").strip()
    if scheduled_for and current_next and scheduled_for != current_next:
        return False, "schedule_changed"

    if current_next:
        try:
            next_run_dt = datetime.fromisoformat(current_next)
            now = datetime.now(next_run_dt.tzinfo)
            if next_run_dt > now:
                return False, "not_due"
        except Exception:
            pass

    return True, None


def process_cron_queue_item(
    payload: Any,
    *,
    prepare_runtime_environment: Callable[[], None],
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    release_cron_job_claim_fn: Callable[[str, str | None], None],
    logger: Any,
) -> dict[str, Any]:
    prepare_runtime_environment()
    if should_reload_modal_volume_for_claims("cron"):
        sync_modal_volume(reload=True)

    from cron.jobs import get_job, mark_job_run, save_job_output
    from cron.scheduler import SILENT_MARKER, _deliver_result, run_job

    if not isinstance(payload, dict):
        payload = {"job_id": str(payload or "")}

    job_id = str(payload.get("job_id") or "").strip()
    claim_token = str(payload.get("claim_token") or "").strip() or None
    if not job_id:
        return {"status": "skipped", "reason": "missing_job_id"}

    job = get_job(job_id)
    should_run, skip_reason = should_process_queued_cron_job(job, payload)
    if not should_run:
        release_cron_job_claim_fn(job_id, claim_token)
        return {"status": "skipped", "job_id": job_id, "reason": skip_reason}

    success, output, final_response, error = run_job(job)
    output_file = str(save_job_output(job_id, output))

    delivery_error = None
    deliver_content = final_response if success else f"Cron job '{job.get('name', job_id)}' failed:\n{error}"
    should_deliver = bool(deliver_content)
    if should_deliver and success and SILENT_MARKER in deliver_content.strip().upper():
        should_deliver = False

    if should_deliver:
        try:
            delivery_error = _deliver_result(job, deliver_content)
        except Exception as exc:
            delivery_error = str(exc)
            logger.error("Cron delivery failed for job %s: %s", job_id, exc)

    mark_job_run(job_id, success, error, delivery_error=delivery_error)
    sync_modal_volume(commit=True)
    release_cron_job_claim_fn(job_id, claim_token)

    return {
        "status": "ok" if success else "error",
        "job_id": job_id,
        "job_name": job.get("name"),
        "output_file": output_file,
        "delivery_error": delivery_error,
        "error": error,
    }


def process_cron_queue_impl(
    *,
    max_jobs: int,
    prepare_runtime_environment: Callable[[], None],
    get_cron_queue: Callable[[], Any],
    process_cron_queue_item_fn: Callable[[Any], dict[str, Any]],
    safe_cron_queue_depth: Callable[[], int | None],
) -> dict[str, Any]:
    prepare_runtime_environment()
    queue = get_cron_queue()
    items = queue.get_many(max(max_jobs, 0), block=False) if max_jobs > 0 else []
    results = [process_cron_queue_item_fn(item) for item in items]
    return {
        "status": "ok",
        "processed_count": len(results),
        "results": results,
        "queue_depth": safe_cron_queue_depth(),
    }


def cron_scheduler_tick_impl(
    *,
    enqueue_limit: int,
    worker_count: int,
    enqueue_due_cron_jobs_impl_fn: Callable[[int], dict[str, Any]],
    safe_cron_queue_depth: Callable[[], int | None],
    modal_module: Any,
    process_cron_queue_spawn: Callable[..., Any] | None,
    default_cron_queue_max_jobs_per_worker: int,
) -> dict[str, Any]:
    enqueue_result = enqueue_due_cron_jobs_impl_fn(enqueue_limit)
    queue_depth = enqueue_result.get("queue_depth")
    if queue_depth is None:
        queue_depth = safe_cron_queue_depth() or 0

    spawned_workers = 0
    jobs_per_worker = 0
    if modal_module is not None and worker_count > 0 and queue_depth and process_cron_queue_spawn is not None:
        spawned_workers = min(int(queue_depth), max(worker_count, 0))
        jobs_per_worker = min(
            default_cron_queue_max_jobs_per_worker,
            max(1, math.ceil(int(queue_depth) / max(spawned_workers, 1))),
        )
        for _ in range(spawned_workers):
            process_cron_queue_spawn(max_jobs=jobs_per_worker)

    return {
        "status": "ok",
        "enqueue": enqueue_result,
        "spawned_workers": spawned_workers,
        "jobs_per_worker": jobs_per_worker,
        "queue_depth": safe_cron_queue_depth(),
    }


def maintenance_heartbeat_impl(
    *,
    enqueue_limit: int,
    worker_count: int,
    cron_scheduler_tick_impl_fn: Callable[[int, int], dict[str, Any]],
    feishu_model_registry_heartbeat_impl_fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "ok",
        "cron": cron_scheduler_tick_impl_fn(enqueue_limit, worker_count),
        "feishu_registry": feishu_model_registry_heartbeat_impl_fn(),
    }
