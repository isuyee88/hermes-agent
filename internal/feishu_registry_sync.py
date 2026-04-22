from __future__ import annotations

import time
from typing import Any, Callable


def run_feishu_registry_sync_cycle(
    *,
    force_refresh: bool,
    mirror_to_bitable: bool | None,
    ensure_schema: bool,
    prepare_runtime_environment: Callable[[], None],
    load_feishu_sync_state: Callable[[], dict[str, Any]],
    save_feishu_sync_state: Callable[[dict[str, Any]], None],
    settings_from_env: Callable[[], Any],
    refresh_free_model_routes: Callable[..., dict[str, Any]],
    mask_runtime_identifier: Callable[[str | None], str],
    compact_feishu_schema_status: Callable[[dict[str, Any]], dict[str, Any]],
    logger: Any,
) -> dict[str, Any]:
    prepare_runtime_environment()
    state = load_feishu_sync_state()
    now = int(time.time())
    settings = settings_from_env()
    should_mirror = mirror_to_bitable if mirror_to_bitable is not None else settings.feishu_model_registry_mirror_enabled
    state.update(
        {
            "last_attempt_at": now,
            "sync_interval_seconds": settings.feishu_model_registry_sync_interval_seconds,
            "mirror_enabled": bool(should_mirror),
        }
    )
    try:
        from tools.feishu_api import (
            build_feishu_client,
            build_model_registry,
            ensure_model_registry_bitable_schema,
            mirror_model_registry_to_bitable,
            resolve_bitable_target,
        )

        if force_refresh:
            try:
                refresh_free_model_routes(force=True)
            except Exception as exc:
                logger.warning("Failed refreshing provider routes before model registry sync: %s", exc)

        registry_payload = build_model_registry(force_refresh=force_refresh)
        result: dict[str, Any] = {
            "status": "ok",
            "entry_count": len(registry_payload.get("entries") or []),
            "registry": registry_payload,
            "mirrored": False,
        }
        state["last_registry_entry_count"] = result["entry_count"]

        if should_mirror:
            client = build_feishu_client()
            app_token, table_id = resolve_bitable_target({}, client)
            state["target"] = {
                "app_token": app_token,
                "app_token_masked": mask_runtime_identifier(app_token),
                "table_id": table_id,
                "resolution_mode": "app_token" if settings.feishu_bitable_app_token else "wiki_token",
            }
            if ensure_schema:
                schema_result = ensure_model_registry_bitable_schema(
                    client,
                    app_token=app_token,
                    table_id=table_id or None,
                    table_name="Hermes Model Registry",
                    create_missing_table=True,
                    create_missing_fields=True,
                    create_missing_views=True,
                )
                result["bitable_schema"] = schema_result
                compact_schema = compact_feishu_schema_status(schema_result)
                compact_schema["checked_at"] = now
                state["schema"] = compact_schema

            mirror_result = mirror_model_registry_to_bitable(
                client,
                registry_payload,
                app_token=app_token,
                table_id=table_id,
            )
            result["bitable_mirror"] = mirror_result
            result["mirrored"] = True
            state["last_success_at"] = now
            state["last_status"] = "ok"
            state["last_error"] = ""
            state["last_sync"] = {
                "status": "ok",
                "mirrored": True,
                "created": int(mirror_result.get("created") or 0),
                "updated": int(mirror_result.get("updated") or 0),
                "upserted": int(mirror_result.get("created") or 0) + int(mirror_result.get("updated") or 0),
                "hidden": int(mirror_result.get("hidden") or 0),
                "entry_count": result["entry_count"],
                "completed_at": now,
            }
        else:
            state["last_success_at"] = now
            state["last_status"] = "ok"
            state["last_error"] = ""
            state["last_sync"] = {
                "status": "ok",
                "mirrored": False,
                "entry_count": result["entry_count"],
                "completed_at": now,
            }

        state["next_due_at"] = now + settings.feishu_model_registry_sync_interval_seconds
        save_feishu_sync_state(state)
        return result
    except Exception as exc:
        state["last_status"] = "error"
        state["last_error"] = str(exc)
        state["last_error_at"] = now
        state["next_due_at"] = now + min(settings.feishu_model_registry_sync_interval_seconds, 300)
        save_feishu_sync_state(state)
        return {
            "status": "error",
            "error": str(exc),
        }
