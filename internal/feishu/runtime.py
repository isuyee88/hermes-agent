from __future__ import annotations

from typing import Any, Awaitable, Callable


def initialize_internal_gateway_runtime(
    settings: Any,
    *,
    live_runtime: Any,
    make_capture_adapter: Callable[[Any], Any],
    runtime_cls: type,
) -> Any:
    from gateway.config import Platform, PlatformConfig
    from gateway.run import GatewayRunner

    runner = GatewayRunner()
    if live_runtime is not None and getattr(live_runtime, "runner", None) is not None:
        live_runner = live_runtime.runner
        runner.session_store = live_runner.session_store
        runner._voice_mode = dict(getattr(live_runner, "_voice_mode", {}) or {})
        runner._session_model_overrides = dict(getattr(live_runner, "_session_model_overrides", {}) or {})
        runner._pending_model_notes = dict(getattr(live_runner, "_pending_model_notes", {}) or {})
        runner._ephemeral_system_prompt = str(getattr(live_runner, "_ephemeral_system_prompt", "") or "")

    feishu_config = runner.config.platforms.get(Platform.FEISHU) or PlatformConfig()
    feishu_config.enabled = True
    feishu_config.extra.update(
        {
            "app_id": settings.feishu_app_id,
            "app_secret": settings.feishu_app_secret,
            "domain": settings.feishu_domain or "feishu",
            "connection_mode": "internal_capture",
            "webhook_path": "/internal/feishu",
        }
    )
    adapter = make_capture_adapter(feishu_config)
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)
    runner.adapters[Platform.FEISHU] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)
    return runtime_cls(
        runner=runner,
        adapter=adapter,
        init_timings={"mode": "internal_capture"},
    )


async def get_cached_runtime(
    *,
    get_current_runtime: Callable[[], Any],
    get_lock: Callable[[], Any],
    prepare_runtime_environment: Callable[[], None],
    settings_from_env: Callable[[], Any],
    initialize_runtime: Callable[[Any], Awaitable[Any]],
    set_runtime: Callable[[Any], None],
) -> Any:
    current_runtime = get_current_runtime()
    if current_runtime is not None:
        return current_runtime

    async with get_lock():
        current_runtime = get_current_runtime()
        if current_runtime is not None:
            return current_runtime
        prepare_runtime_environment()
        settings = settings_from_env()
        initialized_runtime = await initialize_runtime(settings)
        set_runtime(initialized_runtime)
        return initialized_runtime
