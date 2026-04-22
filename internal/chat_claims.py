from __future__ import annotations

import time
import uuid
from typing import Any, Awaitable, Callable, Mapping


def load_claims(*, load_json_file: Callable[[Any, Any], Any], path: Any) -> dict[str, Any]:
    payload = load_json_file(path, {})
    return payload if isinstance(payload, dict) else {}

def save_claims(payload: dict[str, Any], *, atomic_json_write: Callable[[Any, Any], None], path: Any) -> None:
    atomic_json_write(path, payload)

def load_warmups(*, load_json_file: Callable[[Any, Any], Any], path: Any) -> dict[str, Any]:
    payload = load_json_file(path, {})
    return payload if isinstance(payload, dict) else {}

def save_warmups(payload: dict[str, Any], *, atomic_json_write: Callable[[Any, Any], None], path: Any) -> None:
    atomic_json_write(path, payload)

def prune_warmups(
    warmups: dict[str, Any],
    *,
    ttl_seconds: int,
) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    max_age_ms = max(int(ttl_seconds or 0), 1) * 1000
    pruned: dict[str, Any] = {}
    for partition_key, snapshot in warmups.items():
        if not isinstance(snapshot, dict):
            continue
        updated_at_ms = int(
            snapshot.get("ready_at_ms")
            or snapshot.get("updated_at_ms")
            or snapshot.get("started_at_ms")
            or 0
        )
        if updated_at_ms and now_ms - updated_at_ms < max_age_ms:
            pruned[partition_key] = snapshot
    return pruned


def record_warmup_snapshot(
    partition_key: str,
    *,
    platform: str,
    status: str,
    metadata: dict[str, Any] | None,
    worker_context: dict[str, Any] | None,
    lock: Any,
    load_warmups: Callable[[], dict[str, Any]],
    save_warmups: Callable[[dict[str, Any]], None],
    prune_warmups: Callable[[dict[str, Any]], dict[str, Any]],
    sync_modal_volume: Callable[..., None],
    first_non_empty_str: Callable[..., str],
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {}
    now_ms = int(time.time() * 1000)
    metadata_payload = dict(metadata or {})
    worker_payload = dict(worker_context or {})
    snapshot = {
        "platform": str(platform or "").strip().lower(),
        "partition": normalized,
        "status": str(status or "").strip().lower() or "ready",
        "event_id": first_non_empty_str(metadata_payload.get("event_id")),
        "event_type": first_non_empty_str(metadata_payload.get("event_type")),
        "actor_id": first_non_empty_str(metadata_payload.get("actor_id")),
        "chat_id": first_non_empty_str(metadata_payload.get("chat_id")),
        "started_at_ms": int(metadata_payload.get("started_at_ms") or now_ms),
        "ready_at_ms": now_ms,
        "updated_at_ms": now_ms,
        "worker_boot_id": first_non_empty_str(worker_payload.get("worker_boot_id")),
        "worker_started_at": worker_payload.get("worker_started_at"),
        "enter_elapsed_ms": worker_payload.get("enter_elapsed_ms"),
        "runtime_prepare_elapsed_ms": worker_payload.get("runtime_prepare_elapsed_ms"),
    }
    with lock:
        warmups = prune_warmups(load_warmups())
        warmups[normalized] = snapshot
        save_warmups(warmups)
        sync_modal_volume(commit=True)
    return snapshot


def pop_warmup_snapshot(
    partition_key: str,
    *,
    lock: Any,
    load_warmups: Callable[[], dict[str, Any]],
    save_warmups: Callable[[dict[str, Any]], None],
    prune_warmups: Callable[[dict[str, Any]], dict[str, Any]],
    sync_modal_volume: Callable[..., None],
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {}
    with lock:
        warmups = prune_warmups(load_warmups())
        snapshot = warmups.pop(normalized, None)
        if not isinstance(snapshot, dict):
            return {}
        save_warmups(warmups)
        sync_modal_volume(commit=True)
    return dict(snapshot)


def prune_claims(
    claims: dict[str, Any],
    *,
    ttl_seconds: int,
) -> dict[str, Any]:
    now = int(time.time())
    pruned: dict[str, Any] = {}
    for partition_key, claim in claims.items():
        if not isinstance(claim, dict):
            continue
        claimed_at = int(claim.get("claimed_at") or 0)
        status = str(claim.get("status") or "claimed").strip().lower()
        claim_ttl_seconds = max(int(claim.get("cooldown_seconds") or 0), 1) if status == "scheduled" else ttl_seconds
        if claimed_at and now - claimed_at < claim_ttl_seconds:
            pruned[partition_key] = claim
    return pruned


def claim_is_recent(claim: Mapping[str, Any], *, max_age_seconds: int) -> bool:
    if max_age_seconds <= 0:
        return False
    claimed_at = int((claim or {}).get("claimed_at") or 0)
    if claimed_at <= 0:
        return False
    return int(time.time()) - claimed_at <= max_age_seconds


def claim_age_seconds(claim: Mapping[str, Any]) -> int:
    claimed_at = int((claim or {}).get("claimed_at") or 0)
    if claimed_at <= 0:
        return 0
    return max(0, int(time.time()) - claimed_at)

def claim_is_stale_for_takeover(
    claim: Mapping[str, Any],
    *,
    stale_after_seconds: int | None,
    default_stale_after_seconds: int,
    claim_age_seconds: Callable[[Mapping[str, Any]], int],
) -> bool:
    threshold = default_stale_after_seconds if stale_after_seconds is None else int(stale_after_seconds)
    if threshold <= 0:
        return False
    return claim_age_seconds(claim) >= threshold


def peek_claim(
    partition_key: str,
    *,
    ttl_seconds: int,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    prune_claims: Callable[[dict[str, Any], int], dict[str, Any]],
    load_claims: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {}
    with lock:
        if should_reload_modal_volume_for_claims("chat"):
            sync_modal_volume(reload=True)
        claims = prune_claims(load_claims(), ttl_seconds)
        existing = claims.get(normalized) or {}
        return dict(existing) if isinstance(existing, dict) else {}


async def peek_claim_async(
    partition_key: str,
    *,
    ttl_seconds: int,
    refresh_on_miss: bool,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume_async: Callable[..., Awaitable[None]],
    prune_claims: Callable[[dict[str, Any], int], dict[str, Any]],
    load_claims: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {}
    should_reload = should_reload_modal_volume_for_claims("chat")
    if should_reload:
        await sync_modal_volume_async(reload=True)

    with lock:
        claims = prune_claims(load_claims(), ttl_seconds)
        existing = claims.get(normalized) or {}
        claim = dict(existing) if isinstance(existing, dict) else {}

    if claim or not refresh_on_miss or should_reload:
        return claim

    await sync_modal_volume_async(reload=True)
    with lock:
        claims = prune_claims(load_claims(), ttl_seconds)
        existing = claims.get(normalized) or {}
        return dict(existing) if isinstance(existing, dict) else {}


def has_recent_worker_spawn(
    partition_key: str,
    *,
    ttl_seconds: float,
    recent_worker_spawns: dict[str, float],
    recent_worker_spawns_lock: Any,
) -> bool:
    normalized = str(partition_key or "").strip()
    if not normalized or ttl_seconds <= 0:
        return False
    now = time.monotonic()
    with recent_worker_spawns_lock:
        expired = [key for key, ts in recent_worker_spawns.items() if now - ts > ttl_seconds]
        for key in expired:
            recent_worker_spawns.pop(key, None)
        last_spawned_at = recent_worker_spawns.get(normalized)
        return bool(last_spawned_at is not None and now - last_spawned_at <= ttl_seconds)


def mark_recent_worker_spawn(
    partition_key: str,
    *,
    recent_worker_spawns: dict[str, float],
    recent_worker_spawns_lock: Any,
) -> None:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return
    with recent_worker_spawns_lock:
        recent_worker_spawns[normalized] = time.monotonic()


def claim_partition(
    partition_key: str,
    *,
    platform: str,
    claim_token: str | None,
    ttl_seconds: int,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    prune_claims: Callable[[dict[str, Any], int], dict[str, Any]],
    load_claims: Callable[[], dict[str, Any]],
    save_claims: Callable[[dict[str, Any]], None],
    claim_is_stale_for_takeover: Callable[[Mapping[str, Any]], bool],
    claim_age_seconds: Callable[[Mapping[str, Any]], int],
    logger: Any,
) -> tuple[bool, str]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return False, ""
    now = int(time.time())
    token = str(claim_token or f"{platform}:{normalized}:{now}:{uuid.uuid4().hex[:8]}").strip()
    with lock:
        if should_reload_modal_volume_for_claims("chat"):
            sync_modal_volume(reload=True)
        claims = prune_claims(load_claims(), ttl_seconds)
        existing = claims.get(normalized) or {}
        if existing:
            existing_token = str(existing.get("claim_token") or "").strip()
            if existing_token and claim_token and existing_token == token:
                claims[normalized] = {
                    "claim_token": token,
                    "claimed_at": now,
                    "platform": platform,
                    "status": "claimed",
                }
                save_claims(claims)
                sync_modal_volume(commit=True)
                return True, token
            if not claim_is_stale_for_takeover(existing):
                return False, existing_token
            logger.warning(
                "[ChatQueue] taking over stale claim partition=%s old_status=%s old_age_seconds=%s old_token=%s",
                normalized,
                str(existing.get("status") or "claimed").strip().lower() or "claimed",
                claim_age_seconds(existing),
                existing_token or "none",
            )
        claims[normalized] = {
            "claim_token": token,
            "claimed_at": now,
            "platform": platform,
            "status": "claimed",
        }
        save_claims(claims)
        sync_modal_volume(commit=True)
    return True, token


def schedule_partition_worker(
    partition_key: str,
    *,
    platform: str,
    cooldown_seconds: int,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    prune_claims: Callable[[dict[str, Any], int], dict[str, Any]],
    load_claims: Callable[[], dict[str, Any]],
    save_claims: Callable[[dict[str, Any]], None],
    claim_is_stale_for_takeover: Callable[[Mapping[str, Any]], bool],
    claim_age_seconds: Callable[[Mapping[str, Any]], int],
    logger: Any,
    claim_ttl_seconds: int,
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": str(platform or "").strip().lower(),
            "partition": normalized,
        }

    normalized_platform = str(platform or "").strip().lower()
    now = int(time.time())
    token = f"{normalized_platform}:{normalized}:{now}:{uuid.uuid4().hex[:8]}"
    with lock:
        if should_reload_modal_volume_for_claims("chat"):
            sync_modal_volume(reload=True)
        claims = prune_claims(load_claims(), claim_ttl_seconds)
        existing = claims.get(normalized) or {}
        if existing:
            if not claim_is_stale_for_takeover(existing):
                return {
                    "status": "skipped",
                    "reason": str(existing.get("status") or "already_scheduled"),
                    "platform": normalized_platform,
                    "partition": normalized,
                    "claim_token": str(existing.get("claim_token") or ""),
                }
            logger.warning(
                "[ChatQueue] replacing stale scheduled claim partition=%s old_status=%s old_age_seconds=%s old_token=%s",
                normalized,
                str(existing.get("status") or "claimed").strip().lower() or "claimed",
                claim_age_seconds(existing),
                str(existing.get("claim_token") or "") or "none",
            )
        claims[normalized] = {
            "claim_token": token,
            "claimed_at": now,
            "platform": normalized_platform,
            "status": "scheduled",
            "cooldown_seconds": max(int(cooldown_seconds or 0), 1),
        }
        save_claims(claims)
        sync_modal_volume(commit=True)
    return {
        "status": "scheduled",
        "platform": normalized_platform,
        "partition": normalized,
        "claim_token": token,
    }


async def schedule_partition_worker_async(
    partition_key: str,
    *,
    platform: str,
    cooldown_seconds: int,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume_async: Callable[..., Awaitable[None]],
    prune_claims: Callable[[dict[str, Any], int], dict[str, Any]],
    load_claims: Callable[[], dict[str, Any]],
    save_claims: Callable[[dict[str, Any]], None],
    claim_is_stale_for_takeover: Callable[[Mapping[str, Any]], bool],
    claim_age_seconds: Callable[[Mapping[str, Any]], int],
    logger: Any,
    claim_ttl_seconds: int,
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": str(platform or "").strip().lower(),
            "partition": normalized,
        }

    normalized_platform = str(platform or "").strip().lower()
    now = int(time.time())
    token = f"{normalized_platform}:{normalized}:{now}:{uuid.uuid4().hex[:8]}"
    if should_reload_modal_volume_for_claims("chat"):
        await sync_modal_volume_async(reload=True)

    should_commit = False
    with lock:
        claims = prune_claims(load_claims(), claim_ttl_seconds)
        existing = claims.get(normalized) or {}
        if existing:
            if not claim_is_stale_for_takeover(existing):
                return {
                    "status": "skipped",
                    "reason": str(existing.get("status") or "already_scheduled"),
                    "platform": normalized_platform,
                    "partition": normalized,
                    "claim_token": str(existing.get("claim_token") or ""),
                }
            logger.warning(
                "[ChatQueue] replacing stale scheduled claim partition=%s old_status=%s old_age_seconds=%s old_token=%s",
                normalized,
                str(existing.get("status") or "claimed").strip().lower() or "claimed",
                claim_age_seconds(existing),
                str(existing.get("claim_token") or "") or "none",
            )
        claims[normalized] = {
            "claim_token": token,
            "claimed_at": now,
            "platform": normalized_platform,
            "status": "scheduled",
            "cooldown_seconds": max(int(cooldown_seconds or 0), 1),
        }
        save_claims(claims)
        should_commit = True

    if should_commit:
        await sync_modal_volume_async(commit=True)
    return {
        "status": "scheduled",
        "platform": normalized_platform,
        "partition": normalized,
        "claim_token": token,
    }


def release_claim(
    partition_key: str,
    *,
    claim_token: str | None,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    prune_claims: Callable[[dict[str, Any], int], dict[str, Any]],
    load_claims: Callable[[], dict[str, Any]],
    save_claims: Callable[[dict[str, Any]], None],
    claim_ttl_seconds: int,
) -> None:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return

    with lock:
        if should_reload_modal_volume_for_claims("chat"):
            sync_modal_volume(reload=True)
        claims = prune_claims(load_claims(), claim_ttl_seconds)
        existing = claims.get(normalized)
        if not existing:
            return
        if claim_token and existing.get("claim_token") != claim_token:
            return
        claims.pop(normalized, None)
        save_claims(claims)
        sync_modal_volume(commit=True)


async def release_claim_async(
    partition_key: str,
    *,
    claim_token: str | None,
    lock: Any,
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume_async: Callable[..., Awaitable[None]],
    prune_claims: Callable[[dict[str, Any], int], dict[str, Any]],
    load_claims: Callable[[], dict[str, Any]],
    save_claims: Callable[[dict[str, Any]], None],
    claim_ttl_seconds: int,
) -> None:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return

    if should_reload_modal_volume_for_claims("chat"):
        await sync_modal_volume_async(reload=True)

    should_commit = False
    with lock:
        claims = prune_claims(load_claims(), claim_ttl_seconds)
        existing = claims.get(normalized)
        if not existing:
            return
        if claim_token and existing.get("claim_token") != claim_token:
            return
        claims.pop(normalized, None)
        save_claims(claims)
        should_commit = True

    if should_commit:
        await sync_modal_volume_async(commit=True)
