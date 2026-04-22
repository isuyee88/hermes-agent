from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable


def iter_session_route_payloads(
    *,
    limit: int,
    sessions_dir: Path,
    load_json_file_fn: Callable[[Path, Any], Any],
) -> list[dict[str, Any]]:
    if limit <= 0 or not sessions_dir.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for path in sessions_dir.glob("*.json"):
        payload = load_json_file_fn(path, {})
        if isinstance(payload, dict):
            payloads.append(payload)
    payloads.sort(key=lambda item: int(item.get("updated_at") or 0), reverse=True)
    return payloads[:limit]


def build_recent_session_route_summaries(
    *,
    limit: int,
    iter_session_route_payloads_fn: Callable[[int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = int(time.time())
    for payload in iter_session_route_payloads_fn(limit):
        lease = payload.get("route_lease") or {}
        route_debug = payload.get("route_debug") or {}
        metrics = payload.get("route_metrics") or {}
        rows.append(
            {
                "session_key": payload.get("session_key"),
                "session_id": payload.get("session_id"),
                "updated_at": payload.get("updated_at"),
                "route_lease": lease if isinstance(lease, dict) else None,
                "route_selection": route_debug.get("last_route_selection"),
                "last_failure_reason": route_debug.get("last_failure_reason"),
                "last_error": route_debug.get("last_error"),
                "lease_ttl_remaining_seconds": max(int((lease or {}).get("lease_expires_at") or 0) - now, 0),
                "metrics": metrics if isinstance(metrics, dict) else {},
            }
        )
    return rows


def aggregate_session_route_metrics(
    *,
    limit: int,
    iter_session_route_payloads_fn: Callable[[int], list[dict[str, Any]]],
) -> dict[str, Any]:
    totals = {
        "sticky_hit": 0,
        "fresh_select": 0,
        "explicit_override": 0,
        "refreshed_after_failure": 0,
    }
    sessions = iter_session_route_payloads_fn(limit)
    sessions_with_lease = 0
    for payload in sessions:
        lease = payload.get("route_lease")
        if isinstance(lease, dict) and lease:
            sessions_with_lease += 1
        metrics = payload.get("route_metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for key in totals:
            totals[key] += int(metrics.get(key) or 0)
    totals["sessions_with_route_lease"] = sessions_with_lease
    totals["sampled_sessions"] = len(sessions)
    return totals


def build_model_routing_debug_state(
    *,
    force_refresh: bool = False,
    allow_network: bool = False,
    prepare_runtime_environment: Callable[[], None],
    load_routing_state: Callable[[], dict[str, Any]],
    refresh_free_model_routes: Callable[..., dict[str, Any]],
    candidate_routes_from_state: Callable[..., list[dict[str, Any]]],
    build_recent_session_route_summaries_fn: Callable[[int], list[dict[str, Any]]],
    aggregate_session_route_metrics_fn: Callable[[int], dict[str, Any]],
) -> dict[str, Any]:
    prepare_runtime_environment()
    state = load_routing_state()
    if force_refresh or (allow_network and not state):
        state = refresh_free_model_routes(force=force_refresh)

    preferred_provider = str(
        os.getenv("HERMES_FREE_MODEL_PRIMARY_PROVIDER")
        or ("openrouter" if os.getenv("OPENROUTER_API_KEY", "").strip() else "nvidia")
    ).strip().lower() or None
    candidate_routes = candidate_routes_from_state(state, preferred_provider=preferred_provider)
    primary_route = candidate_routes[0] if candidate_routes else None
    fallback_routes = []
    seen_fallbacks: set[tuple[str, str, str]] = set()
    for route in candidate_routes[1:]:
        key = (
            str(route.get("provider") or "").strip().lower(),
            str(route.get("model") or "").strip(),
            str(route.get("base_url") or "").strip(),
        )
        if key in seen_fallbacks:
            continue
        seen_fallbacks.add(key)
        fallback_routes.append(route)
        if len(fallback_routes) >= 4:
            break

    return {
        "configured_default_model": os.getenv("DEFAULT_MODEL", "openrouter/free"),
        "free_model_primary_provider": str(os.getenv("HERMES_FREE_MODEL_PRIMARY_PROVIDER") or "").strip().lower() or None,
        "cheap_routing_enabled": False,
        "active_primary_route": {
            "provider": primary_route.get("provider"),
            "model": primary_route.get("model"),
            "base_url": primary_route.get("base_url"),
        } if primary_route else None,
        "fallback_candidates": [
            {
                "provider": route.get("provider"),
                "model": route.get("model"),
                "base_url": route.get("base_url"),
            }
            for route in fallback_routes
        ],
        "session_route_metrics": aggregate_session_route_metrics_fn(200),
        "recent_session_routes": build_recent_session_route_summaries_fn(20),
        "routing_state": state,
    }


def debug_session_route_state(
    session_key: str,
    *,
    prepare_runtime_environment: Callable[[], None],
    load_session_state: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    prepare_runtime_environment()
    normalized = str(session_key or "").strip()
    if not normalized:
        return {
            "status": "error",
            "message": "session_key is required",
        }

    payload = load_session_state(normalized)
    lease = payload.get("route_lease")
    now = int(time.time())
    return {
        "status": "ok",
        "session_key": normalized,
        "session_id": payload.get("session_id"),
        "updated_at": payload.get("updated_at"),
        "route_lease": lease,
        "route_debug": payload.get("route_debug") or {},
        "route_metrics": payload.get("route_metrics") or {},
        "lease_active": bool(isinstance(lease, dict) and int((lease or {}).get("lease_expires_at") or 0) > now),
        "lease_ttl_remaining_seconds": max(int((lease or {}).get("lease_expires_at") or 0) - now, 0),
    }


def debug_gateway_session_state(
    session_key: str,
    *,
    prepare_runtime_environment: Callable[[], None],
    logger: Any,
) -> dict[str, Any]:
    prepare_runtime_environment()

    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home

    normalized = str(session_key or "").strip()
    runner = GatewayRunner()
    try:
        runner.session_store._ensure_loaded()
    except Exception:
        logger.debug("Failed to eagerly load gateway session store for debug", exc_info=True)
    entry = runner.session_store._entries.get(normalized)
    sessions_dir = Path(runner.session_store.sessions_dir)
    sessions_file = sessions_dir / "sessions.json"
    return {
        "status": "ok",
        "session_key": normalized,
        "env_hermes_home": os.getenv("HERMES_HOME", ""),
        "resolved_hermes_home": str(get_hermes_home()),
        "gateway_sessions_dir": str(sessions_dir),
        "sessions_file_exists": sessions_file.exists(),
        "session_count": len(runner.session_store._entries),
        "entry": entry.to_dict() if entry else None,
        "known_session_keys_sample": sorted(list(runner.session_store._entries.keys()))[:20],
    }
