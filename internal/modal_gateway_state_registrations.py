from __future__ import annotations

import asyncio
from collections.abc import MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_modal_gateway_state_helpers(namespace: Namespace) -> None:
    def _load_telegram_webhook_sync_state() -> dict[str, Any]:
        payload = namespace["_load_json_file"](namespace["TELEGRAM_WEBHOOK_SYNC_STATE_PATH"], {})
        return payload if isinstance(payload, dict) else {}

    def _save_telegram_webhook_sync_state(payload: dict[str, Any]) -> None:
        namespace["_atomic_json_write"](namespace["TELEGRAM_WEBHOOK_SYNC_STATE_PATH"], payload)

    def _load_feishu_sync_state() -> dict[str, Any]:
        payload = namespace["_load_json_file"](namespace["FEISHU_SYNC_STATE_PATH"], {})
        return payload if isinstance(payload, dict) else {}

    def _save_feishu_sync_state(payload: dict[str, Any]) -> None:
        namespace["_atomic_json_write"](namespace["FEISHU_SYNC_STATE_PATH"], payload)

    def _get_feishu_sync_interval_seconds() -> int:
        raw = str(namespace["os"].getenv("FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS", "") or "").strip()
        if not raw:
            return namespace["DEFAULT_FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS"]
        try:
            return max(60, int(raw))
        except Exception:
            return namespace["DEFAULT_FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS"]

    def _mask_runtime_identifier(value: str | None) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if len(raw) <= 10:
            return namespace["_mask_secret"](raw)
        return f"{raw[:4]}...{raw[-4:]}"

    def _compact_feishu_schema_status(result: dict[str, Any]) -> dict[str, Any]:
        existing_field_names = result.get("existing_field_names") if isinstance(result.get("existing_field_names"), list) else []
        existing_view_names = result.get("existing_view_names") if isinstance(result.get("existing_view_names"), list) else []
        return {
            "status": result.get("status"),
            "app_token": str(result.get("app_token") or ""),
            "app_token_masked": namespace["_mask_runtime_identifier"](result.get("app_token")),
            "table_id": str(result.get("table_id") or ""),
            "table_name": result.get("table_name"),
            "created_table": bool(result.get("created_table")),
            "created_field_count": len(result.get("created_fields") or []),
            "created_view_count": len(result.get("created_views") or []),
            "field_count": len(existing_field_names),
            "view_count": len(existing_view_names),
            "missing_required_fields": list(result.get("missing_required_fields") or []),
            "missing_optional_fields": list(result.get("missing_optional_fields") or []),
            "missing_views": list(result.get("missing_views") or []),
        }

    def _telegram_webhook_retry_after_seconds(exc: Exception) -> int:
        return namespace["_telegram_gateway_retry_after_seconds"](
            exc,
            default_retry_seconds=namespace["DEFAULT_TELEGRAM_WEBHOOK_SYNC_BACKOFF_SECONDS"],
        )

    def _get_telegram_runtime_lock() -> asyncio.Lock:
        if namespace["_TELEGRAM_RUNTIME_LOCK"] is None:
            namespace["_TELEGRAM_RUNTIME_LOCK"] = asyncio.Lock()
        return namespace["_TELEGRAM_RUNTIME_LOCK"]

    def _get_feishu_runtime_lock() -> asyncio.Lock:
        if namespace["_FEISHU_RUNTIME_LOCK"] is None:
            namespace["_FEISHU_RUNTIME_LOCK"] = asyncio.Lock()
        return namespace["_FEISHU_RUNTIME_LOCK"]

    def _get_feishu_internal_runtime_lock() -> asyncio.Lock:
        if namespace["_FEISHU_INTERNAL_RUNTIME_LOCK"] is None:
            namespace["_FEISHU_INTERNAL_RUNTIME_LOCK"] = asyncio.Lock()
        return namespace["_FEISHU_INTERNAL_RUNTIME_LOCK"]

    def _get_qq_runtime_lock() -> asyncio.Lock:
        if namespace["_QQ_RUNTIME_LOCK"] is None:
            namespace["_QQ_RUNTIME_LOCK"] = asyncio.Lock()
        return namespace["_QQ_RUNTIME_LOCK"]

    namespace["_load_telegram_webhook_sync_state"] = _load_telegram_webhook_sync_state
    namespace["_save_telegram_webhook_sync_state"] = _save_telegram_webhook_sync_state
    namespace["_load_feishu_sync_state"] = _load_feishu_sync_state
    namespace["_save_feishu_sync_state"] = _save_feishu_sync_state
    namespace["_get_feishu_sync_interval_seconds"] = _get_feishu_sync_interval_seconds
    namespace["_mask_runtime_identifier"] = _mask_runtime_identifier
    namespace["_compact_feishu_schema_status"] = _compact_feishu_schema_status
    namespace["_telegram_webhook_retry_after_seconds"] = _telegram_webhook_retry_after_seconds
    namespace["_get_telegram_runtime_lock"] = _get_telegram_runtime_lock
    namespace["_get_feishu_runtime_lock"] = _get_feishu_runtime_lock
    namespace["_get_feishu_internal_runtime_lock"] = _get_feishu_internal_runtime_lock
    namespace["_get_qq_runtime_lock"] = _get_qq_runtime_lock
