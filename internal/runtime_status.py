from __future__ import annotations

from typing import Any


def load_gateway_import_status() -> tuple[bool, str | None]:
    try:
        import gateway.run  # noqa: F401
        import run_agent  # noqa: F401
        return True, None
    except Exception as exc:
        return False, str(exc)


def build_gateway_health_payload(
    *,
    status: str,
    service: str,
    settings: Any,
    telegram_webhook: dict[str, Any] | None,
    chat_queue_name: str,
    chat_queue_depth: int | None,
    cron_status: dict[str, Any],
    memory_provider: dict[str, Any],
    model_routing: dict[str, Any],
    feishu_sync: dict[str, Any],
    runtime_config: str,
    settings_payload: dict[str, Any],
    gateway_import_ok: bool,
    gateway_import_error: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "service": service,
        "telegram_configured": bool(settings.telegram_bot_token),
        "feishu_configured": bool(settings.feishu_app_id and settings.feishu_app_secret),
        "qq_configured": bool(settings.qq_app_id and settings.qq_app_secret),
        "telegram_webhook": telegram_webhook,
        "chat_queue": {
            "queue_name": chat_queue_name,
            "queue_depth": chat_queue_depth,
        },
        "cron": cron_status,
        "memory_provider": memory_provider,
        "model_routing": model_routing,
        "feishu_sync": feishu_sync,
        "runtime_config": runtime_config,
        "gateway_import_ok": gateway_import_ok,
        "gateway_import_error": gateway_import_error,
        "settings": settings_payload,
    }
    if extra:
        payload.update(extra)
    return payload
