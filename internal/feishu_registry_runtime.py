from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable


def build_feishu_capabilities_debug_state(
    *,
    probe: bool = False,
    prepare_runtime_environment: Callable[[], None],
) -> dict[str, Any]:
    prepare_runtime_environment()
    try:
        from tools.feishu_api import get_feishu_capability_snapshot

        return get_feishu_capability_snapshot(probe=probe)
    except Exception as exc:
        return {
            "configured": False,
            "error": str(exc),
        }


def build_feishu_model_registry_debug_state(
    *,
    force_refresh: bool = False,
    prepare_runtime_environment: Callable[[], None],
) -> dict[str, Any]:
    prepare_runtime_environment()
    try:
        from tools.feishu_api import load_feishu_model_registry

        payload = load_feishu_model_registry(force_refresh=force_refresh)
        return {
            "status": "ok",
            **payload,
            "entry_count": len(payload.get("entries") or []),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def prepare_feishu_model_registry_bitable_impl(
    *,
    app_token: str | None,
    table_id: str | None,
    table_name: str,
    create_missing_table: bool,
    create_missing_fields: bool,
    create_missing_views: bool,
    prepare_runtime_environment: Callable[[], None],
    load_feishu_sync_state: Callable[[], dict[str, Any]],
    save_feishu_sync_state: Callable[[dict[str, Any]], None],
    settings_from_env: Callable[[], Any],
    mask_runtime_identifier: Callable[[str | None], str],
    compact_feishu_schema_status: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    prepare_runtime_environment()
    state = load_feishu_sync_state()
    now = int(time.time())
    try:
        from tools.feishu_api import build_feishu_client, ensure_model_registry_bitable_schema, resolve_bitable_target

        client = build_feishu_client()
        resolved_app_token, resolved_table_id = resolve_bitable_target(
            {
                "app_token": app_token,
                "table_id": table_id,
            },
            client,
            require_table_id=False,
        )
        result = ensure_model_registry_bitable_schema(
            client,
            app_token=resolved_app_token,
            table_id=resolved_table_id or None,
            table_name=str(table_name or "Hermes Model Registry").strip() or "Hermes Model Registry",
            create_missing_table=create_missing_table,
            create_missing_fields=create_missing_fields,
            create_missing_views=create_missing_views,
        )
        state["target"] = {
            "app_token": resolved_app_token,
            "app_token_masked": mask_runtime_identifier(resolved_app_token),
            "table_id": str(result.get("table_id") or resolved_table_id or ""),
            "resolution_mode": "app_token" if settings_from_env().feishu_bitable_app_token else "wiki_token",
        }
        compact_schema = compact_feishu_schema_status(result)
        compact_schema["checked_at"] = now
        state["schema"] = compact_schema
        state["last_status"] = "ok"
        state["last_error"] = ""
        save_feishu_sync_state(state)
        return result
    except Exception as exc:
        state["last_status"] = "error"
        state["last_error"] = str(exc)
        state["last_error_at"] = now
        save_feishu_sync_state(state)
        return {
            "status": "error",
            "error": str(exc),
        }


def build_feishu_sync_state_debug_state(
    *,
    prepare_runtime_environment: Callable[[], None],
    load_feishu_sync_state: Callable[[], dict[str, Any]],
    settings_from_env: Callable[[], Any],
    state_file: Path,
    mask_runtime_identifier: Callable[[str | None], str],
) -> dict[str, Any]:
    prepare_runtime_environment()
    state = load_feishu_sync_state()
    settings = settings_from_env()
    payload: dict[str, Any] = {
        "configured": bool(
            (settings.feishu_bitable_app_token or settings.feishu_bitable_wiki_token)
            and settings.feishu_bitable_table_id
        ),
        "mirror_enabled": settings.feishu_model_registry_mirror_enabled,
        "sync_interval_seconds": settings.feishu_model_registry_sync_interval_seconds,
        "state_file": str(state_file),
        "last_attempt_at": state.get("last_attempt_at"),
        "last_success_at": state.get("last_success_at"),
        "last_status": state.get("last_status") or "",
        "last_error": state.get("last_error") or "",
        "next_due_at": state.get("next_due_at"),
        "last_registry_entry_count": state.get("last_registry_entry_count"),
        "target": state.get("target") or {},
        "schema": state.get("schema") or {},
        "last_sync": state.get("last_sync") or {},
    }
    try:
        from tools.feishu_api import build_feishu_client, resolve_bitable_target

        if payload["configured"]:
            client = build_feishu_client()
            app_token, resolved_table_id = resolve_bitable_target({}, client)
            payload["resolved_target"] = {
                "app_token": app_token,
                "app_token_masked": mask_runtime_identifier(app_token),
                "table_id": resolved_table_id,
            }
    except Exception as exc:
        payload["resolved_target_error"] = str(exc)
    return payload


def should_prepare_feishu_registry_schema_on_startup(
    state: dict[str, Any] | None = None,
    *,
    default_recheck_seconds: int,
) -> bool:
    payload = state if isinstance(state, dict) else {}
    schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
    checked_at = int(schema.get("checked_at") or 0)
    if not checked_at:
        return True
    if str(schema.get("status") or "").strip().lower() != "ok":
        return True
    if schema.get("missing_required_fields") or schema.get("missing_views"):
        return True
    return (int(time.time()) - checked_at) >= default_recheck_seconds


def feishu_model_registry_heartbeat_impl(
    *,
    settings_from_env: Callable[[], Any],
    load_feishu_sync_state: Callable[[], dict[str, Any]],
    run_feishu_registry_sync_cycle: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    settings = settings_from_env()
    if not settings.feishu_model_registry_mirror_enabled:
        return {"status": "skipped", "reason": "mirror_disabled"}
    if not settings.feishu_bitable_table_id or not (
        settings.feishu_bitable_app_token or settings.feishu_bitable_wiki_token
    ):
        return {"status": "skipped", "reason": "bitable_not_configured"}

    sync_state = load_feishu_sync_state()
    now = int(time.time())
    next_due_at = int(sync_state.get("next_due_at") or 0)
    if next_due_at and now < next_due_at:
        return {
            "status": "skipped",
            "reason": "not_due",
            "next_due_at": next_due_at,
            "seconds_until_due": next_due_at - now,
        }
    return run_feishu_registry_sync_cycle(force_refresh=True, mirror_to_bitable=True, ensure_schema=True)
