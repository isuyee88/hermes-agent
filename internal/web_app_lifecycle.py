from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable


async def run_startup_tasks(
    *,
    prepare_runtime_environment: Callable[[], None],
    prewarm_chat_queue_async: Callable[[], Awaitable[None]],
    settings_from_env: Callable[[], Any],
    maybe_sync_telegram_webhook: Callable[[Any], Awaitable[dict[str, Any]]],
    load_feishu_sync_state: Callable[[], dict[str, Any]],
    should_prepare_feishu_registry_schema_on_startup: Callable[[dict[str, Any] | None], bool],
    prepare_feishu_model_registry_bitable_impl: Callable[[], dict[str, Any]],
    default_feishu_startup_schema_recheck_seconds: int,
    logger: Any,
) -> None:
    prepare_runtime_environment()
    await prewarm_chat_queue_async()
    settings = settings_from_env()
    if settings.telegram_bot_token and settings.telegram_webhook_url:
        try:
            status = await maybe_sync_telegram_webhook(settings)
            logger.info(
                "Telegram webhook startup sync: expected=%s registered=%s matched=%s pending=%s",
                status.get("expected_url"),
                status.get("registered_url"),
                status.get("matches_expected"),
                status.get("pending_update_count"),
            )
        except Exception as exc:
            logger.warning("Telegram webhook startup sync failed: %s", exc)

    if not settings.feishu_model_registry_mirror_enabled:
        return
    if not settings.feishu_bitable_table_id or not (
        settings.feishu_bitable_app_token or settings.feishu_bitable_wiki_token
    ):
        return

    sync_state = load_feishu_sync_state()
    if not should_prepare_feishu_registry_schema_on_startup(sync_state):
        logger.info(
            "Feishu registry schema startup prepare skipped: checked_at=%s ttl=%ss",
            (sync_state.get("schema") or {}).get("checked_at"),
            default_feishu_startup_schema_recheck_seconds,
        )
        return

    try:
        result = await asyncio.to_thread(prepare_feishu_model_registry_bitable_impl)
        logger.info(
            "Feishu registry schema startup prepare: status=%s table=%s created_table=%s missing_required=%s",
            result.get("status"),
            result.get("table_id"),
            result.get("created_table"),
            len(result.get("missing_required_fields") or []),
        )
    except Exception as exc:
        logger.warning("Feishu registry schema startup prepare failed: %s", exc)


def build_lifespan(*, startup_tasks: Callable[[], Awaitable[None]]) -> Callable[[Any], Any]:
    @asynccontextmanager
    async def _lifespan(_app: Any):
        await startup_tasks()
        yield

    return _lifespan
