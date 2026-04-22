from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_feishu_bot_identity_cache_helpers(namespace: Namespace) -> None:
    def _load_feishu_bot_identity_cache() -> dict[str, Any]:
        payload = namespace["_load_json_file"](namespace["FEISHU_BOT_IDENTITY_CACHE_PATH"], {})
        return payload if isinstance(payload, dict) else {}

    def _save_feishu_bot_identity_cache(payload: dict[str, Any]) -> None:
        namespace["_atomic_json_write"](namespace["FEISHU_BOT_IDENTITY_CACHE_PATH"], payload)

    def _get_cached_feishu_bot_identity(app_id: str | None) -> dict[str, Any]:
        normalized_app_id = str(app_id or "").strip()
        if not normalized_app_id:
            return {}
        with namespace["_FEISHU_BOT_IDENTITY_CACHE_LOCK"]:
            payload = namespace["_load_feishu_bot_identity_cache"]()
            cached = payload.get(normalized_app_id) or {}
            return dict(cached) if isinstance(cached, dict) else {}

    async def _cache_feishu_bot_identity_async(app_id: str | None, *, bot_name: str | None = None) -> None:
        normalized_app_id = str(app_id or "").strip()
        normalized_bot_name = str(bot_name or "").strip()
        if not normalized_app_id or not normalized_bot_name:
            return

        should_commit = False
        with namespace["_FEISHU_BOT_IDENTITY_CACHE_LOCK"]:
            payload = namespace["_load_feishu_bot_identity_cache"]()
            existing = payload.get(normalized_app_id) or {}
            if not isinstance(existing, dict) or str(existing.get("bot_name") or "").strip() != normalized_bot_name:
                payload[normalized_app_id] = {
                    "bot_name": normalized_bot_name,
                    "updated_at": int(time.time()),
                }
                namespace["_save_feishu_bot_identity_cache"](payload)
                should_commit = True

        if should_commit:
            await namespace["_sync_modal_volume_async"](commit=True)

    namespace["_load_feishu_bot_identity_cache"] = _load_feishu_bot_identity_cache
    namespace["_save_feishu_bot_identity_cache"] = _save_feishu_bot_identity_cache
    namespace["_get_cached_feishu_bot_identity"] = _get_cached_feishu_bot_identity
    namespace["_cache_feishu_bot_identity_async"] = _cache_feishu_bot_identity_async
