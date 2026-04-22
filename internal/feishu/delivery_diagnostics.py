from __future__ import annotations

from typing import Any, Iterable, Mapping

DEFAULT_REQUIRED_WEBHOOK_CALLBACKS = (
    "im.message.receive_v1",
    "im.message.message_read_v1",
    "card.action.trigger",
    "application.bot.menu_v6",
)


def _trim(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = _trim(value)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _mask_secret(value: str) -> str:
    normalized = _trim(value)
    if not normalized:
        return ""
    return f"<masked len={len(normalized)}>"


def _normalize_callback_url(value: str) -> str:
    return _trim(value).rstrip("/")


def _is_accepted_webhook_url(callback_url: str, expected_webhook_url: str) -> bool:
    normalized_callback_url = _normalize_callback_url(callback_url)
    normalized_expected_url = _normalize_callback_url(expected_webhook_url)
    if not normalized_expected_url:
        return True
    if normalized_callback_url == normalized_expected_url:
        return True
    legacy_suffix = "/feishu/webhook"
    if normalized_expected_url.endswith(legacy_suffix):
        legacy_root = normalized_expected_url[: -len(legacy_suffix)]
        if normalized_callback_url == legacy_root:
            return True
    return False


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return {
            str(key): item
            for key, item in data.items()
            if not str(key).startswith("_")
        }
    result: dict[str, Any] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if callable(item):
            continue
        result[str(key)] = item
    return result


def sanitize_callback_info(callback_info: Any) -> dict[str, Any]:
    raw = _coerce_mapping(callback_info)
    sanitized: dict[str, Any] = {}
    for key, value in raw.items():
        lowered = key.lower()
        if isinstance(value, (list, tuple, set)):
            sanitized[key] = _dedupe_strings(value)
            continue
        if isinstance(value, Mapping) or hasattr(value, "__dict__"):
            sanitized[key] = sanitize_callback_info(value)
            continue
        normalized = _trim(value)
        if any(token in lowered for token in ("token", "encrypt", "secret")):
            sanitized[key] = _mask_secret(normalized)
        else:
            sanitized[key] = normalized
    return sanitized


def analyze_feishu_delivery_config(
    *,
    runtime_mode: str,
    callback_info: Any,
    expected_webhook_url: str = "",
    required_callbacks: Iterable[str] | None = None,
) -> dict[str, Any]:
    callback_map = _coerce_mapping(callback_info)
    normalized_runtime_mode = _trim(runtime_mode).lower() or "webhook"
    callback_type = _trim(callback_map.get("callback_type")).lower()
    callback_url = _trim(
        callback_map.get("callback_url")
        or callback_map.get("url")
        or callback_map.get("request_url")
        or callback_map.get("callback")
    )
    subscribed_callbacks = _dedupe_strings(callback_map.get("subscribed_callbacks") or [])
    normalized_expected_url = _normalize_callback_url(expected_webhook_url)
    expected_callbacks = _dedupe_strings(required_callbacks or DEFAULT_REQUIRED_WEBHOOK_CALLBACKS)

    issues: list[dict[str, str]] = []
    recommendation_actions: list[str] = []

    if normalized_runtime_mode == "webhook":
        if callback_type != "webhook":
            issues.append(
                {
                    "code": "callback_type_mismatch",
                    "severity": "critical",
                    "message": (
                        f"Feishu app callback_type is `{callback_type or 'unset'}` but Hermes runtime expects `webhook`."
                    ),
                }
            )
            recommendation_actions.append("切换飞书应用事件订阅模式为 webhook。")
        if normalized_expected_url:
            if not callback_url:
                issues.append(
                    {
                        "code": "callback_url_missing",
                        "severity": "critical",
                        "message": "Feishu app callback_url is empty while Hermes runtime expects a public webhook URL.",
                    }
                )
                recommendation_actions.append(f"将飞书应用事件订阅请求地址设置为 `{normalized_expected_url}`。")
            elif not _is_accepted_webhook_url(callback_url, normalized_expected_url):
                issues.append(
                    {
                        "code": "callback_url_mismatch",
                        "severity": "critical",
                        "message": (
                            f"Feishu app callback_url points to `{callback_url}` instead of `{normalized_expected_url}`."
                        ),
                    }
                )
                recommendation_actions.append(f"将飞书应用事件订阅请求地址改为 `{normalized_expected_url}`。")
        missing_callbacks = [name for name in expected_callbacks if name not in subscribed_callbacks]
        if missing_callbacks:
            issues.append(
                {
                    "code": "missing_callbacks",
                    "severity": "critical",
                    "message": (
                        "Feishu app is missing required callbacks: " + ", ".join(missing_callbacks)
                    ),
                }
            )
            recommendation_actions.append(
                "在飞书应用事件订阅中至少启用这些事件: " + ", ".join(missing_callbacks)
            )
    elif normalized_runtime_mode == "websocket" and callback_type != "websocket":
        issues.append(
            {
                "code": "callback_type_mismatch",
                "severity": "critical",
                "message": (
                    f"Feishu app callback_type is `{callback_type or 'unset'}` but Hermes runtime expects `websocket`."
                ),
            }
        )
        recommendation_actions.append("切换飞书应用事件订阅模式为 websocket，或把 Hermes 改回 webhook。")
        missing_callbacks = []
    else:
        missing_callbacks = []

    status = "ok" if not issues else "error"
    if status == "ok":
        summary = "Feishu app callback configuration matches the Hermes runtime delivery mode."
    else:
        summary = issues[0]["message"]

    return {
        "status": status,
        "summary": summary,
        "runtime_mode": normalized_runtime_mode,
        "callback_type": callback_type or "unset",
        "callback_url": callback_url,
        "expected_webhook_url": normalized_expected_url,
        "mode_matches": callback_type == normalized_runtime_mode,
        "webhook_url_matches": _is_accepted_webhook_url(callback_url, normalized_expected_url),
        "subscribed_callbacks": subscribed_callbacks,
        "required_callbacks": expected_callbacks,
        "missing_callbacks": missing_callbacks,
        "issues": issues,
        "recommendation_actions": recommendation_actions,
    }
