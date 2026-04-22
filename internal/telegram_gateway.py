from __future__ import annotations

import os
import time
from urllib.parse import urlparse
from typing import Any, Callable


def normalize_public_https_url(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    normalized = raw.rstrip("/")
    return normalized or None


def resolve_telegram_webhook_url(
    *,
    explicit_url: str | None = None,
    public_base_url: str | None = None,
    normalize_public_https_url_fn: Callable[[str | None], str | None],
) -> str | None:
    direct = normalize_public_https_url_fn(explicit_url)
    if direct:
        return direct
    base_url = normalize_public_https_url_fn(public_base_url)
    if not base_url:
        return None
    return f"{base_url}/telegram/webhook"


def desired_telegram_webhook_url(
    *,
    resolve_telegram_webhook_url_fn: Callable[..., str | None],
) -> str | None:
    return resolve_telegram_webhook_url_fn(
        explicit_url=os.getenv("TELEGRAM_WEBHOOK_URL"),
        public_base_url=os.getenv("HERMES_PUBLIC_BASE_URL") or os.getenv("PUBLIC_BASE_URL"),
    )


async def fetch_telegram_webhook_info(bot_token: str) -> dict[str, Any]:
    import httpx

    url = f"https://api.telegram.org/bot{bot_token}/getWebhookInfo"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(f"Telegram getWebhookInfo failed: {payload}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"Telegram getWebhookInfo returned invalid result: {payload}")
    return result


async def set_telegram_webhook(
    bot_token: str,
    webhook_url: str,
    *,
    webhook_secret: str | None = None,
    drop_pending_updates: bool = False,
) -> dict[str, Any]:
    import httpx

    payload: dict[str, Any] = {
        "url": webhook_url,
        "drop_pending_updates": bool(drop_pending_updates),
        "allowed_updates": [
            "message",
            "edited_message",
            "callback_query",
            "channel_post",
            "edited_channel_post",
        ],
    }
    if webhook_secret:
        payload["secret_token"] = webhook_secret

    url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError(f"Telegram setWebhook failed: {result}")
    return result


async def get_telegram_webhook_status(
    settings: Any,
    *,
    ensure_registered: bool,
    drop_pending_updates: bool,
    is_valid_telegram_bot_token_format: Callable[[str | None], bool],
    is_truthy: Callable[[str | None, bool], bool],
    set_telegram_webhook_fn: Callable[..., Any],
    fetch_telegram_webhook_info_fn: Callable[[str], Any],
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "configured": bool(settings.telegram_bot_token),
        "token_format_valid": is_valid_telegram_bot_token_format(settings.telegram_bot_token),
        "expected_url": settings.telegram_webhook_url,
        "registered_url": "",
        "matches_expected": False,
        "pending_update_count": None,
        "last_error_message": "",
        "ip_address": "",
        "has_custom_certificate": False,
        "auto_sync_enabled": is_truthy(os.getenv("TELEGRAM_WEBHOOK_AUTO_SYNC"), default=True),
    }
    if not settings.telegram_bot_token:
        status["reason"] = "telegram_bot_token_missing"
        return status
    if not status["token_format_valid"]:
        status["reason"] = "telegram_bot_token_invalid_format"
        return status
    if not settings.telegram_webhook_url:
        status["reason"] = "telegram_webhook_url_missing"
        return status

    if ensure_registered:
        await set_telegram_webhook_fn(
            settings.telegram_bot_token,
            settings.telegram_webhook_url,
            webhook_secret=settings.telegram_webhook_secret,
            drop_pending_updates=drop_pending_updates,
        )

    info = await fetch_telegram_webhook_info_fn(settings.telegram_bot_token)
    registered_url = str(info.get("url") or "").strip()
    status.update(
        {
            "registered_url": registered_url,
            "matches_expected": registered_url == settings.telegram_webhook_url,
            "pending_update_count": info.get("pending_update_count"),
            "last_error_message": str(info.get("last_error_message") or "").strip(),
            "ip_address": str(info.get("ip_address") or "").strip(),
            "has_custom_certificate": bool(info.get("has_custom_certificate")),
        }
    )
    return status


def telegram_webhook_retry_after_seconds(exc: Exception, *, default_retry_seconds: int) -> int:
    retry_after = default_retry_seconds
    response = getattr(exc, "response", None)
    if response is None:
        return retry_after

    header_retry = response.headers.get("retry-after") if getattr(response, "headers", None) else None
    if header_retry:
        try:
            return max(int(header_retry), 1)
        except (TypeError, ValueError):
            pass

    try:
        payload = response.json()
    except Exception:
        payload = {}

    parameters = payload.get("parameters") if isinstance(payload, dict) else {}
    try:
        parsed_retry = int((parameters or {}).get("retry_after") or 0)
        if parsed_retry > 0:
            return parsed_retry
    except (TypeError, ValueError):
        pass
    return retry_after


async def maybe_sync_telegram_webhook(
    settings: Any,
    *,
    drop_pending_updates: bool,
    is_truthy: Callable[[str | None, bool], bool],
    get_telegram_webhook_status_fn: Callable[..., Any],
    save_telegram_webhook_sync_state: Callable[[dict[str, Any]], None],
    load_telegram_webhook_sync_state: Callable[[], dict[str, Any]],
    set_telegram_webhook_fn: Callable[..., Any],
    telegram_webhook_retry_after_seconds_fn: Callable[[Exception], int],
) -> dict[str, Any]:
    if not is_truthy(os.getenv("TELEGRAM_WEBHOOK_AUTO_SYNC"), default=True):
        return await get_telegram_webhook_status_fn(settings, ensure_registered=False, drop_pending_updates=False)

    status = await get_telegram_webhook_status_fn(settings, ensure_registered=False, drop_pending_updates=False)
    if status.get("matches_expected"):
        save_telegram_webhook_sync_state(
            {
                "last_attempt_at": int(time.time()),
                "next_retry_at": 0,
                "registered_url": status.get("registered_url") or "",
                "last_error": "",
            }
        )
        return status

    sync_state = load_telegram_webhook_sync_state()
    now = int(time.time())
    next_retry_at = int(sync_state.get("next_retry_at") or 0)
    if next_retry_at and next_retry_at > now:
        status["sync_deferred"] = True
        status["retry_after_seconds"] = next_retry_at - now
        status["last_sync_error"] = str(sync_state.get("last_error") or "").strip()
        return status

    try:
        await set_telegram_webhook_fn(
            settings.telegram_bot_token,
            settings.telegram_webhook_url,
            webhook_secret=settings.telegram_webhook_secret,
            drop_pending_updates=drop_pending_updates,
        )
    except Exception as exc:
        retry_after_seconds = telegram_webhook_retry_after_seconds_fn(exc)
        save_telegram_webhook_sync_state(
            {
                "last_attempt_at": now,
                "next_retry_at": now + retry_after_seconds,
                "registered_url": status.get("registered_url") or "",
                "last_error": str(exc),
            }
        )
        status["last_sync_error"] = str(exc)
        if "429" in str(exc):
            status["sync_rate_limited"] = True
            status["retry_after_seconds"] = retry_after_seconds
            return status
        raise

    refreshed = await get_telegram_webhook_status_fn(settings, ensure_registered=False, drop_pending_updates=False)
    save_telegram_webhook_sync_state(
        {
            "last_attempt_at": now,
            "next_retry_at": 0,
            "registered_url": refreshed.get("registered_url") or "",
            "last_error": "",
        }
    )
    return refreshed
