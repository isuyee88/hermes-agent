from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable


DIRECT_RUNTIME_PROVIDER_BYPASS = frozenset({"custom", "local", "copilot-acp"})


def provider_requires_runtime_binding(provider_name: str | None) -> bool:
    normalized = str(provider_name or "").strip().lower()
    return bool(
        normalized
        and normalized not in DIRECT_RUNTIME_PROVIDER_BYPASS
        and not normalized.startswith("custom:")
    )


def is_cloudflare_gateway_base_url(base_url: str | None) -> bool:
    return "gateway.ai.cloudflare.com" in str(base_url or "").strip().lower()


def is_dynamic_free_route_alias(model_name: str | None, *, free_model_aliases: set[str]) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in free_model_aliases


def is_freeish_model_name(model_name: str | None, *, free_model_aliases: set[str]) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in free_model_aliases or normalized.endswith(":free")


def route_to_runtime_model_config(route: dict[str, Any]) -> dict[str, Any]:
    provider_name = str(route.get("provider") or "").strip().lower()
    payload: dict[str, Any] = {
        "default": str(route.get("model") or "").strip(),
        "provider": "openrouter" if provider_name == "openrouter" else "custom",
        "base_url": str(route.get("base_url") or "").strip(),
    }
    api_key = str(route.get("api_key") or "").strip()
    if provider_name != "openrouter" and api_key:
        payload["api_key"] = api_key
    return {key: value for key, value in payload.items() if value}


def route_to_fallback_provider(route: dict[str, Any]) -> dict[str, Any]:
    provider_name = str(route.get("provider") or "").strip().lower()
    payload: dict[str, Any] = {
        "provider": "openrouter" if provider_name == "openrouter" else "custom",
        "model": str(route.get("model") or "").strip(),
        "base_url": str(route.get("base_url") or "").strip(),
    }
    api_key = str(route.get("api_key") or "").strip()
    if provider_name != "openrouter" and api_key:
        payload["api_key"] = api_key
    return {key: value for key, value in payload.items() if value}


def route_from_settings(
    settings: Any,
    model_name: str | None,
    *,
    default_openrouter_base_url: str,
    resolve_provider_runtime_binding_fn: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    provider_name = str(settings.provider or ("openrouter" if os.getenv("OPENROUTER_API_KEY") else "")).strip().lower()
    if provider_requires_runtime_binding(provider_name) and resolve_provider_runtime_binding_fn is not None:
        runtime_binding = resolve_provider_runtime_binding_fn(provider_name) or {}
        binding_base_url = str(runtime_binding.get("base_url") or "").strip()
        binding_api_key = str(runtime_binding.get("api_key") or "").strip()
        if binding_base_url and binding_api_key:
            return {
                "provider": str(runtime_binding.get("provider") or provider_name).strip().lower() or provider_name,
                "base_url": binding_base_url,
                "api_key": binding_api_key,
                "model": model_name or settings.model,
            }
        return {
            "provider": provider_name,
            "base_url": None,
            "api_key": None,
            "model": model_name or settings.model,
        }
    return {
        "provider": provider_name,
        "base_url": settings.base_url or ((os.getenv("OPENROUTER_BASE_URL") or default_openrouter_base_url) if os.getenv("OPENROUTER_API_KEY") else None),
        "api_key": settings.api_key,
        "model": model_name or settings.model,
    }


def candidate_routes_from_state(
    state: dict[str, Any],
    *,
    preferred_provider: str | None,
    failed_route: dict[str, Any] | None,
    resolve_provider_runtime_binding_fn: Callable[[str], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    providers = state.get("providers") or {}
    provider_order = list(providers.keys())
    if preferred_provider and preferred_provider in provider_order:
        provider_order = [preferred_provider] + [name for name in provider_order if name != preferred_provider]

    routes: list[dict[str, Any]] = []
    for provider_name in provider_order:
        entry = providers.get(provider_name) or {}
        candidates = entry.get("candidates") or []
        runtime_binding = resolve_provider_runtime_binding_fn(provider_name)
        binding_required = provider_requires_runtime_binding(provider_name)
        resolved_provider = str((runtime_binding or {}).get("provider") or provider_name).strip().lower() or provider_name
        base_url = str((runtime_binding or {}).get("base_url") or "").strip()
        api_key = str((runtime_binding or {}).get("api_key") or "").strip()
        if not base_url and not binding_required:
            base_url = str(entry.get("base_url") or "").strip()
        if not api_key and not binding_required:
            if provider_name == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            elif provider_name == "nvidia":
                api_key = (os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY") or "").strip()
        if not api_key or not base_url:
            continue
        for candidate in candidates:
            route = {
                "provider": resolved_provider,
                "base_url": base_url,
                "api_key": api_key,
                "model": str(candidate).strip(),
            }
            if not route["model"]:
                continue
            if failed_route and route["provider"] == failed_route.get("provider") and route["model"] == failed_route.get("model"):
                continue
            routes.append(route)
    return routes


def select_dynamic_primary_route(
    *,
    force_refresh: bool,
    refresh_free_model_routes_fn: Callable[..., dict[str, Any]],
    candidate_routes_from_state_fn: Callable[..., list[dict[str, Any]]],
) -> dict[str, Any] | None:
    preferred_provider = str(os.getenv("HERMES_FREE_MODEL_PRIMARY_PROVIDER") or ("openrouter" if os.getenv("OPENROUTER_API_KEY", "").strip() else "nvidia")).strip().lower() or None
    state = refresh_free_model_routes_fn(force=force_refresh)
    routes = candidate_routes_from_state_fn(state, preferred_provider=preferred_provider, failed_route=None)
    return routes[0] if routes else None


def select_dynamic_fallback_routes(
    primary_route: dict[str, Any] | None,
    *,
    refresh_free_model_routes_fn: Callable[..., dict[str, Any]],
    candidate_routes_from_state_fn: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    preferred_provider = None
    if primary_route:
        primary_provider = str(primary_route.get("provider") or "").strip().lower()
        if primary_provider == "openrouter" and os.getenv("NVIDIA_API_KEY", "").strip():
            preferred_provider = "nvidia"
        elif primary_provider == "nvidia" and os.getenv("OPENROUTER_API_KEY", "").strip():
            preferred_provider = "openrouter"
    state = refresh_free_model_routes_fn(force=False)
    routes = candidate_routes_from_state_fn(state, preferred_provider=preferred_provider, failed_route=primary_route)
    fallbacks: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for route in routes:
        key = (str(route.get("provider") or "").strip().lower(), str(route.get("model") or "").strip(), str(route.get("base_url") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        fallbacks.append(route)
        if len(fallbacks) >= 4:
            break
    return fallbacks


def materialize_dynamic_free_model_config(
    config_payload: dict[str, Any],
    *,
    is_dynamic_free_route_alias_fn: Callable[[str | None], bool],
    select_dynamic_primary_route_fn: Callable[..., dict[str, Any] | None],
    route_to_runtime_model_config_fn: Callable[[dict[str, Any]], dict[str, Any]],
    select_dynamic_fallback_routes_fn: Callable[[dict[str, Any] | None], list[dict[str, Any]]],
    route_to_fallback_provider_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    model_config = config_payload.get("model")
    if not isinstance(model_config, dict):
        return
    default_model = str(model_config.get("default") or "").strip()
    if not is_dynamic_free_route_alias_fn(default_model):
        return
    primary_route = select_dynamic_primary_route_fn(force_refresh=False)
    if primary_route is None:
        return
    model_config.clear()
    model_config.update(route_to_runtime_model_config_fn(primary_route))
    fallback_routes = select_dynamic_fallback_routes_fn(primary_route)
    if fallback_routes:
        config_payload["fallback_providers"] = [route_to_fallback_provider_fn(route) for route in fallback_routes]
    elif "fallback_providers" in config_payload:
        config_payload.pop("fallback_providers", None)


def is_invalid_model_error_text(error_text: str | None, *, invalid_model_error_markers: tuple[str, ...]) -> bool:
    normalized = str(error_text or "").strip().lower()
    return any(marker in normalized for marker in invalid_model_error_markers)


def is_transient_route_error_text(error_text: str | None, *, transient_route_error_markers: tuple[str, ...]) -> bool:
    normalized = str(error_text or "").strip().lower()
    return any(marker in normalized for marker in transient_route_error_markers)


def resolve_primary_route(
    settings: Any,
    model_name: str | None,
    *,
    route_from_settings_fn: Callable[[Any, str | None], dict[str, Any]],
    is_dynamic_free_route_alias_fn: Callable[[str | None], bool],
    select_dynamic_primary_route_fn: Callable[..., dict[str, Any] | None],
) -> dict[str, Any]:
    primary_route = route_from_settings_fn(settings, model_name)
    if model_name and not is_dynamic_free_route_alias_fn(model_name):
        return primary_route
    if not is_dynamic_free_route_alias_fn(primary_route.get("model")):
        return primary_route
    selected = select_dynamic_primary_route_fn(force_refresh=False)
    return selected or primary_route


def probe_model_route(route: dict[str, Any]) -> bool:
    import httpx

    headers = {"Authorization": f"Bearer {route['api_key']}"}
    if (
        is_cloudflare_gateway_base_url(route.get("base_url"))
        and str(route.get("provider") or "").strip().lower() in GATEWAY_ENFORCED_PROVIDERS
    ):
        headers = {"cf-aig-authorization": f"Bearer {route['api_key']}"}
    with httpx.Client(timeout=20) as client:
        response = client.post(
            f"{str(route['base_url']).rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": route["model"],
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 4,
                "temperature": 0,
            },
        )
    return response.status_code == 200


def refresh_and_select_valid_route(
    *,
    preferred_provider: str | None,
    failed_route: dict[str, Any] | None,
    refresh_free_model_routes_fn: Callable[..., dict[str, Any]],
    candidate_routes_from_state_fn: Callable[..., list[dict[str, Any]]],
    probe_model_route_fn: Callable[[dict[str, Any]], bool],
    logger: Any,
) -> dict[str, Any] | None:
    state = refresh_free_model_routes_fn(force=True)
    for route in candidate_routes_from_state_fn(state, preferred_provider=preferred_provider, failed_route=failed_route):
        try:
            if probe_model_route_fn(route):
                return route
        except Exception as exc:
            logger.warning("Probe failed for %s/%s: %s", route["provider"], route["model"], exc)
    return None


def select_retry_route_for_result(
    primary_route: dict[str, Any],
    result: dict[str, Any],
    *,
    determine_route_refresh_reason_fn: Callable[[dict[str, Any]], str | None],
    refresh_and_select_valid_route_fn: Callable[..., dict[str, Any] | None],
) -> dict[str, Any] | None:
    refresh_reason = determine_route_refresh_reason_fn(result)
    if not refresh_reason:
        return None
    preferred_provider = None
    primary_provider = str(primary_route.get("provider") or "").strip().lower()
    if primary_provider == "openrouter" and os.getenv("NVIDIA_API_KEY", "").strip():
        preferred_provider = "nvidia"
    elif primary_provider == "nvidia" and os.getenv("OPENROUTER_API_KEY", "").strip():
        preferred_provider = "openrouter"
    return refresh_and_select_valid_route_fn(preferred_provider=preferred_provider, failed_route=primary_route)


def session_file(session_key: str, *, sessions_dir: Path) -> Path:
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return sessions_dir / f"{digest}.json"


def load_session_state(
    session_key: str,
    *,
    load_json_file: Callable[[Path, Any], Any],
    session_file_fn: Callable[[str], Path],
) -> dict[str, Any]:
    state = load_json_file(
        session_file_fn(session_key),
        {
            "session_key": session_key,
            "session_id": str(uuid.uuid4()),
            "messages": [],
            "route_lease": None,
            "route_debug": {},
            "route_metrics": {},
            "updated_at": 0,
        },
    )
    state["session_key"] = session_key
    state.setdefault("session_id", str(uuid.uuid4()))
    state.setdefault("messages", [])
    state.setdefault("route_lease", None)
    state.setdefault("route_debug", {})
    state.setdefault("route_metrics", {})
    state.setdefault("updated_at", 0)
    return state


def save_session_state(
    session_key: str,
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    route_lease: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    route_metrics: dict[str, Any] | None,
    atomic_json_write: Callable[[Path, Any], None],
    session_file_fn: Callable[[str], Path],
) -> None:
    atomic_json_write(
        session_file_fn(session_key),
        {
            "session_key": session_key,
            "session_id": session_id,
            "messages": messages,
            "route_lease": route_lease,
            "route_debug": route_debug or {},
            "route_metrics": route_metrics or {},
            "updated_at": int(time.time()),
        },
    )


def route_api_key_source(provider_name: str, base_url: str = "") -> str:
    normalized = str(provider_name or "").strip().lower()
    normalized_base_url = str(base_url or "").strip().lower()
    if "gateway.ai.cloudflare.com" in normalized_base_url:
        return "CLOUDFLARE_API_TOKEN"
    if normalized == "openrouter":
        return "OPENROUTER_API_KEY"
    if normalized == "nvidia":
        return "NVIDIA_API_KEY"
    return normalized or "unknown"


def build_route_lease(
    route: dict[str, Any],
    *,
    selection_reason: str,
    selected_at: int | None,
    last_success_at: int | None,
    fail_count: int,
    lease_ttl_seconds: int | None,
    route_api_key_source_fn: Callable[[str, str], str],
    default_session_route_ttl_seconds: int,
) -> dict[str, Any]:
    now = int(time.time())
    selected_ts = int(selected_at or now)
    success_ts = int(last_success_at or selected_ts)
    ttl = max(int(lease_ttl_seconds or default_session_route_ttl_seconds), 1)
    provider_name = str(route.get("provider") or "").strip().lower()
    return {
        "provider": provider_name,
        "model": str(route.get("model") or "").strip(),
        "base_url": str(route.get("base_url") or "").strip(),
        "api_key_source": route_api_key_source_fn(provider_name, route.get("base_url") or ""),
        "selected_at": selected_ts,
        "last_success_at": success_ts,
        "fail_count": max(int(fail_count), 0),
        "lease_expires_at": success_ts + ttl,
        "selection_reason": str(selection_reason or "fresh_select").strip() or "fresh_select",
    }


def refresh_route_lease(
    existing_lease: dict[str, Any] | None,
    route: dict[str, Any],
    *,
    selection_reason: str | None,
    build_route_lease_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    selected_at = int((existing_lease or {}).get("selected_at") or time.time())
    chosen_reason = str(selection_reason or (existing_lease or {}).get("selection_reason") or "fresh_select").strip() or "fresh_select"
    return build_route_lease_fn(
        route,
        selection_reason=chosen_reason,
        selected_at=selected_at,
        last_success_at=int(time.time()),
        fail_count=0,
        lease_ttl_seconds=None,
    )


def expire_route_lease(
    existing_lease: dict[str, Any] | None,
    *,
    error_text: str,
    failure_reason: str | None,
) -> dict[str, Any] | None:
    if not isinstance(existing_lease, dict):
        return None
    lease = dict(existing_lease)
    lease["fail_count"] = int(lease.get("fail_count") or 0) + 1
    lease["lease_expires_at"] = int(time.time()) - 1
    if error_text:
        lease["last_error"] = str(error_text)
    if failure_reason:
        lease["last_failure_reason"] = str(failure_reason)
    return lease


def hydrate_route_from_lease(
    settings: Any,
    lease: dict[str, Any] | None,
    *,
    resolve_provider_runtime_binding_fn: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any] | None:
    if not isinstance(lease, dict):
        return None
    provider_name = str(lease.get("provider") or "").strip().lower()
    model_name = str(lease.get("model") or "").strip()
    base_url = str(lease.get("base_url") or "").strip()
    if not provider_name or not model_name or not base_url:
        return None

    runtime_binding = None
    if provider_requires_runtime_binding(provider_name):
        runtime_binding = resolve_provider_runtime_binding_fn(provider_name)

    api_key = str((runtime_binding or {}).get("api_key") or "").strip()
    base_url = str((runtime_binding or {}).get("base_url") or base_url).strip()
    provider_name = str((runtime_binding or {}).get("provider") or provider_name).strip().lower() or provider_name
    if not api_key and is_cloudflare_gateway_base_url(base_url):
        api_key = str(
            os.getenv("CLOUDFLARE_API_TOKEN")
            or os.getenv("CLOUDFLARE_AI_GATEWAY_API_KEY")
            or os.getenv("AI_GATEWAY_API_KEY")
            or ""
        ).strip()
    if not api_key and provider_name == "openrouter":
        api_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
    elif not api_key and provider_name == "nvidia":
        api_key = str(os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY") or "").strip()
    elif not api_key and str(getattr(settings, "provider", "") or "").strip().lower() == provider_name and getattr(settings, "api_key", None):
        api_key = str(settings.api_key)
    if not api_key:
        return None
    return {"provider": provider_name, "model": model_name, "base_url": base_url, "api_key": api_key}


def is_route_lease_active(settings: Any, lease: dict[str, Any] | None, *, hydrate_route_from_lease_fn: Callable[[Any, dict[str, Any] | None], dict[str, Any] | None]) -> bool:
    route = hydrate_route_from_lease_fn(settings, lease)
    if route is None:
        return False
    expires_at = int((lease or {}).get("lease_expires_at") or 0)
    return expires_at > int(time.time())


def increment_route_metric(route_metrics: dict[str, Any], key: str) -> dict[str, Any]:
    metrics = dict(route_metrics or {})
    metrics[key] = int(metrics.get(key) or 0) + 1
    metrics["updated_at"] = int(time.time())
    return metrics


def determine_route_refresh_reason(
    result: dict[str, Any],
    *,
    is_invalid_model_error_text_fn: Callable[[str | None], bool],
    is_transient_route_error_text_fn: Callable[[str | None], bool],
) -> str | None:
    error_text = str(result.get("error") or "").strip()
    final_response = str(result.get("final_response") or "").strip()
    if is_invalid_model_error_text_fn(error_text):
        return "invalid_model"
    if is_transient_route_error_text_fn(error_text):
        return "transient_error"
    if result.get("interrupted"):
        return "interrupted"
    if not final_response and not result.get("completed", True):
        return "incomplete_result"
    return None
