from __future__ import annotations

from typing import Any, Awaitable, Callable

from .feishu.runtime import get_cached_runtime


def initialize_qq_gateway_runtime(settings: Any, *, runtime_cls: type) -> Any:
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.qq import QQAdapter
    from gateway.run import GatewayRunner

    if not settings.qq_app_id or not settings.qq_app_secret:
        raise RuntimeError("QQ_APP_ID and QQ_APP_SECRET are not configured")

    runner = GatewayRunner()
    qq_config = runner.config.platforms.get(Platform.QQ) or PlatformConfig()
    qq_config.enabled = True
    qq_config.extra.update(
        {
            "app_id": settings.qq_app_id,
            "app_secret": settings.qq_app_secret,
            "connection_mode": "webhook",
            "verify_appid_header": True,
            "webhook_path": "/qq/webhook",
        }
    )

    adapter = QQAdapter(qq_config)
    adapter._mark_connected()
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)
    runner.adapters[Platform.QQ] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)
    return runtime_cls(runner=runner, adapter=adapter)


async def get_cached_qq_runtime(
    *,
    get_current_runtime: Callable[[], Any],
    get_lock: Callable[[], Any],
    prepare_runtime_environment: Callable[[], None],
    settings_from_env: Callable[[], Any],
    initialize_runtime: Callable[[Any], Awaitable[Any]],
    set_runtime: Callable[[Any], None],
) -> Any:
    return await get_cached_runtime(
        get_current_runtime=get_current_runtime,
        get_lock=get_lock,
        prepare_runtime_environment=prepare_runtime_environment,
        settings_from_env=settings_from_env,
        initialize_runtime=initialize_runtime,
        set_runtime=set_runtime,
    )
