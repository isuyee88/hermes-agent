from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from typing import Any

from internal.feishu.delivery_diagnostics import analyze_feishu_delivery_config, sanitize_callback_info

# Global variables to store dependencies
_deps = {}


def _initialize_globals(deps: dict[str, Any]) -> None:
    """Initialize global dependencies."""
    global _deps
    _deps = deps


def debug_feishu_runtime() -> dict[str, Any]:
    deps = _deps
    deps["prepare_runtime_environment"]()
    settings = deps["settings_from_env"]()
    return {
        "status": "ok",
        "configured": bool(settings.feishu_app_id and settings.feishu_app_secret),
    }


def debug_runtime_bootstrap() -> dict[str, Any]:
    deps = _deps
    return deps["build_runtime_bootstrap_debug_state"]()


def debug_feishu_menu_config() -> dict[str, Any]:
    deps = _deps
    deps["prepare_runtime_environment"]()
    settings = deps["settings_from_env"]()
    payload = deps["build_feishu_menu_manifest"]()
    payload["configured"] = bool(settings.feishu_app_id and settings.feishu_app_secret)
    return payload


def web_app():
    deps = _deps
    return deps["create_web_app"]()


def cron_scheduler_heartbeat() -> dict[str, Any]:
    deps = _deps
    return deps["maintenance_heartbeat_impl"](
        enqueue_limit=deps["default_cron_queue_batch_size"],
        worker_count=deps["default_cron_queue_workers"],
    )


def register_modal_debug_exports(
    *,
    app: Any,
    modal_module: Any,
    image: Any,
    volume: Any,
    secrets: list[Any],
    maintenance_heartbeat_schedule: Any,
    deps: dict[str, Any],
) -> dict[str, Any]:
    """Register Modal debug export functions with the app."""
    # Initialize global dependencies
    _initialize_globals(deps)

    # Apply Modal decorators with serialized=True
    debug_feishu_runtime_fn = app.function(
        image=image, volumes={"/data": volume}, secrets=secrets, timeout=60,
        serialized=True,
    )(debug_feishu_runtime)

    debug_runtime_bootstrap_fn = app.function(
        image=image, volumes={"/data": volume}, secrets=secrets, timeout=60,
        serialized=True,
    )(debug_runtime_bootstrap)

    debug_feishu_menu_config_fn = app.function(
        image=image, volumes={"/data": volume}, secrets=secrets, timeout=30,
        serialized=True,
    )(debug_feishu_menu_config)

    web_app_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=300,
        memory=2048,
        cpu=1,
        scaledown_window=60,
        enable_memory_snapshot=False,
        serialized=True,
    )(web_app)

    cron_scheduler_heartbeat_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
        schedule=maintenance_heartbeat_schedule,
        serialized=True,
    )(cron_scheduler_heartbeat)

    return {
        "debug_feishu_runtime": debug_feishu_runtime_fn,
        "debug_runtime_bootstrap": debug_runtime_bootstrap_fn,
        "debug_feishu_menu_config": debug_feishu_menu_config_fn,
        "web_app": web_app_fn,
        "cron_scheduler_heartbeat": cron_scheduler_heartbeat_fn,
    }
