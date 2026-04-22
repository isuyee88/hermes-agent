from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable


async def initialize_telegram_gateway_runtime(
    settings: Any,
    *,
    runtime_cls: type,
    logger: Any,
) -> Any:
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.telegram import (
        Application,
        HTTPXRequest,
        TelegramAdapter,
        TelegramFallbackTransport,
        discover_fallback_ips,
    )
    from gateway.run import GatewayRunner

    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    runner = GatewayRunner()
    telegram_config = runner.config.platforms.get(Platform.TELEGRAM) or PlatformConfig()
    telegram_config.enabled = True
    telegram_config.token = settings.telegram_bot_token

    adapter = TelegramAdapter(telegram_config)
    builder = Application.builder().token(telegram_config.token)
    fallback_ips = adapter._fallback_ips()
    if not fallback_ips:
        try:
            fallback_ips = await discover_fallback_ips()
        except Exception as exc:
            logger.warning("Telegram fallback IP discovery failed: %s", exc)
            fallback_ips = []
    if fallback_ips:
        transport = TelegramFallbackTransport(fallback_ips)
        request = HTTPXRequest(httpx_kwargs={"transport": transport})
        get_updates_request = HTTPXRequest(httpx_kwargs={"transport": transport})
        builder = builder.request(request).get_updates_request(get_updates_request)

    adapter._app = builder.build()
    adapter._bot = adapter._app.bot
    await adapter._app.initialize()
    adapter._mark_connected()
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)

    runner.adapters[Platform.TELEGRAM] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)
    return runtime_cls(runner=runner, adapter=adapter)


async def initialize_feishu_gateway_runtime(
    settings: Any,
    *,
    runtime_cls: type,
    split_csv: Callable[[str | None], list[str]],
    dedupe_keep_order: Callable[[list[str]], list[str]],
    is_truthy: Callable[[str | None, bool], bool],
    get_cached_feishu_bot_identity: Callable[[str | None], dict[str, Any]],
    cache_feishu_bot_identity_async: Callable[..., Any],
    logger: Any,
) -> Any:
    total_started_at = time.perf_counter()
    timings: dict[str, int] = {}

    import_started_at = time.perf_counter()
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.feishu import (
        FEISHU_DOMAIN,
        LARK_DOMAIN,
        FeishuAdapter,
        check_feishu_requirements,
    )
    from gateway.run import GatewayRunner
    timings["imports_ms"] = int((time.perf_counter() - import_started_at) * 1000)

    if not settings.feishu_app_id or not settings.feishu_app_secret:
        raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are not configured")
    if not check_feishu_requirements():
        raise RuntimeError("Feishu dependencies are not installed")
    if not os.getenv("HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS", "").strip():
        os.environ["HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS"] = "0"
    if not os.getenv("HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS", "").strip():
        os.environ["HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS"] = "0"

    allowlist_started_at = time.perf_counter()
    approved_feishu_users: list[str] = []
    try:
        from gateway.pairing import PairingStore

        store = PairingStore()
        approved_feishu_users = [
            str(item.get("user_id") or "").strip()
            for item in store.list_approved("feishu")
            if str(item.get("user_id") or "").strip()
        ]
    except Exception:
        logger.warning("[Feishu] Failed to read approved pairing users for allowlist merge", exc_info=True)

    env_allowed_users = split_csv(os.getenv("FEISHU_ALLOWED_USERS"))
    merged_allowed_users = dedupe_keep_order(env_allowed_users + approved_feishu_users)
    group_policy = str(os.getenv("FEISHU_GROUP_POLICY", "allowlist") or "allowlist").strip().lower() or "allowlist"
    group_require_mention = is_truthy(os.getenv("FEISHU_GROUP_REQUIRE_MENTION"), default=False)
    timings["allowlist_ms"] = int((time.perf_counter() - allowlist_started_at) * 1000)
    logger.info(
        "[Feishu] Runtime policy=%s require_mention=%s env_allowlist=%d paired_allowlist=%d merged_allowlist=%d",
        group_policy,
        group_require_mention,
        len(env_allowed_users),
        len(approved_feishu_users),
        len(merged_allowed_users),
    )

    runner_started_at = time.perf_counter()
    runner = GatewayRunner()
    feishu_config = runner.config.platforms.get(Platform.FEISHU) or PlatformConfig()
    feishu_config.enabled = True
    feishu_config.extra.update(
        {
            "app_id": settings.feishu_app_id,
            "app_secret": settings.feishu_app_secret,
            "domain": settings.feishu_domain or "feishu",
            "connection_mode": "webhook",
            "webhook_path": "/feishu/webhook",
            "group_policy": group_policy,
            "allowed_group_users": merged_allowed_users,
            "group_require_mention": group_require_mention,
        }
    )
    timings["runner_config_ms"] = int((time.perf_counter() - runner_started_at) * 1000)

    adapter_started_at = time.perf_counter()
    adapter = FeishuAdapter(feishu_config)
    adapter._loop = asyncio.get_running_loop()
    domain = FEISHU_DOMAIN if adapter._domain_name != "lark" else LARK_DOMAIN
    adapter._client = adapter._build_lark_client(domain)
    adapter._event_handler = adapter._build_event_handler()
    if adapter._event_handler is None:
        raise RuntimeError("failed to build Feishu event handler")
    timings["adapter_setup_ms"] = int((time.perf_counter() - adapter_started_at) * 1000)

    cached_identity = get_cached_feishu_bot_identity(settings.feishu_app_id)
    cached_bot_name = str(cached_identity.get("bot_name") or "").strip()
    if cached_bot_name:
        adapter._bot_name = cached_bot_name
        timings["hydrate_bot_identity_ms"] = 0
        timings["bot_identity_cache_hit"] = 1
    else:
        hydrate_started_at = time.perf_counter()
        await adapter._hydrate_bot_identity()
        timings["hydrate_bot_identity_ms"] = int((time.perf_counter() - hydrate_started_at) * 1000)
        timings["bot_identity_cache_hit"] = 0
        if str(getattr(adapter, "_bot_name", "") or "").strip():
            await cache_feishu_bot_identity_async(
                settings.feishu_app_id,
                bot_name=str(getattr(adapter, "_bot_name", "") or "").strip(),
            )

    finalize_started_at = time.perf_counter()
    adapter._mark_connected()
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)
    adapter.set_menu_action_handler(runner._handle_feishu_menu_action)
    runner.adapters[Platform.FEISHU] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)
    timings["finalize_ms"] = int((time.perf_counter() - finalize_started_at) * 1000)
    timings["total_ms"] = int((time.perf_counter() - total_started_at) * 1000)
    logger.warning(
        "[Feishu] runtime init timings imports_ms=%s allowlist_ms=%s runner_config_ms=%s adapter_setup_ms=%s hydrate_bot_identity_ms=%s finalize_ms=%s total_ms=%s bot_identity_cache_hit=%s",
        timings.get("imports_ms", 0),
        timings.get("allowlist_ms", 0),
        timings.get("runner_config_ms", 0),
        timings.get("adapter_setup_ms", 0),
        timings.get("hydrate_bot_identity_ms", 0),
        timings.get("finalize_ms", 0),
        timings.get("total_ms", 0),
        bool(timings.get("bot_identity_cache_hit")),
    )
    return runtime_cls(runner=runner, adapter=adapter, init_timings=timings)
