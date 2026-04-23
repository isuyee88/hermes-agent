from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any
import urllib.parse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internal.feishu.delivery_diagnostics import analyze_feishu_delivery_config, sanitize_callback_info

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


DEFAULT_CHAT_PAGE_SIZE = 50
MESSAGE_EVENT_NAME = "im.message.receive_v1"
MESSAGE_READ_EVENT_NAME = "im.message.message_read_v1"
DEFAULT_RECENT_MESSAGE_WINDOW_MINUTES = 0
FEISHU_TOKEN_EXPIRED_CODE = 99991677
FEISHU_SCOPE_MISSING_CODE = 99991672
CHAT_MEMBER_READ_SCOPES = (
    "im:chat.members:read",
    "im:chat.group_info:readonly",
    "im:chat:readonly",
    "im:chat",
)


@dataclass(frozen=True)
class CredentialPair:
    app_id_env: str
    app_secret_env: str
    app_id: str
    app_secret: str


@dataclass(frozen=True)
class BotIdentity:
    app_id_env: str
    app_secret_env: str
    app_id: str
    app_name: str
    bot_open_id: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one or more Feishu app callback configurations, visible chats, "
            "and callback URL reachability against Hermes delivery expectations."
        )
    )
    parser.add_argument("--app-id", default=os.getenv("FEISHU_APP_ID", ""), help="Feishu app id. Defaults to FEISHU_APP_ID.")
    parser.add_argument(
        "--app-secret",
        default=os.getenv("FEISHU_APP_SECRET", ""),
        help="Feishu app secret. Defaults to FEISHU_APP_SECRET.",
    )
    parser.add_argument(
        "--expected-mode",
        default=os.getenv("FEISHU_CONNECTION_MODE", "webhook"),
        help="Expected Hermes Feishu delivery mode. Default: webhook.",
    )
    parser.add_argument(
        "--expected-webhook-url",
        default=os.getenv("HERMES_PUBLIC_BASE_URL", ""),
        help="Expected public webhook base URL or full callback URL.",
    )
    parser.add_argument(
        "--required-callback",
        action="append",
        default=[],
        help="Additional required callback event. Repeat for multiple values.",
    )
    parser.add_argument(
        "--all-env-apps",
        action="store_true",
        help="Audit every FEISHU_APP_ID* / FEISHU_APP_SECRET* pair exported in the current environment.",
    )
    parser.add_argument(
        "--probe-timeout",
        type=int,
        default=12,
        help="Timeout in seconds for callback URL challenge probes. Default: 12.",
    )
    parser.add_argument(
        "--chat-page-size",
        type=int,
        default=DEFAULT_CHAT_PAGE_SIZE,
        help="Maximum number of visible chats to request per audited app. Default: 50.",
    )
    parser.add_argument(
        "--recent-message-window-minutes",
        type=int,
        default=DEFAULT_RECENT_MESSAGE_WINDOW_MINUTES,
        help=(
            "If > 0, inspect recent user messages in visible chats and flag whether they target "
            "the audited bot open_id. Default: 0 (skip recent message alignment checks)."
        ),
    )
    parser.add_argument(
        "--target-chat-id",
        default="",
        help="Optional Feishu chat_id to focus the audit on a single target group or conversation.",
    )
    parser.add_argument(
        "--user-access-token",
        default=os.getenv("FEISHU_USER_ACCESS_TOKEN", ""),
        help="Optional Feishu user_access_token used to get a more complete user-view of target chat members.",
    )
    return parser.parse_args()


def _trim(value: Any) -> str:
    return str(value or "").strip()


def _coerce_int(value: Any) -> int:
    try:
        return int(_trim(value) or 0)
    except (TypeError, ValueError):
        return 0


def _extract_required_scopes(message: Any) -> list[str]:
    text = _trim(message)
    if not text:
        return []
    scopes: list[str] = []
    for scope in CHAT_MEMBER_READ_SCOPES + ("im:message.send_as_user", "im:message:send_as_user"):
        if scope in text and scope not in scopes:
            scopes.append(scope)
    return scopes


def _build_expected_webhook_url(value: str) -> str:
    normalized = _trim(value)
    if not normalized:
        return ""
    parsed = normalized.lower()
    if parsed.endswith("/feishu/webhook"):
        return normalized
    if "://" in normalized:
        path = urllib.parse.urlparse(normalized).path or ""
        if path and path != "/":
            return normalized
    return normalized.rstrip("/") + "/feishu/webhook"


def _sort_suffix(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("FEISHU_APP_ID")
    if not suffix:
        return (0, "")
    if suffix.isdigit():
        return (1, f"{int(suffix):09d}")
    return (2, suffix)


def _discover_credential_pairs() -> list[CredentialPair]:
    pairs: list[CredentialPair] = []
    for app_id_env in sorted((name for name in os.environ if name.startswith("FEISHU_APP_ID")), key=_sort_suffix):
        app_secret_env = app_id_env.replace("APP_ID", "APP_SECRET", 1)
        app_id = _trim(os.getenv(app_id_env))
        app_secret = _trim(os.getenv(app_secret_env))
        if not app_id or not app_secret:
            continue
        pairs.append(
            CredentialPair(
                app_id_env=app_id_env,
                app_secret_env=app_secret_env,
                app_id=app_id,
                app_secret=app_secret,
            )
        )
    return pairs


def _get_tenant_access_token(app_id: str, app_secret: str) -> str:
    response = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    token = _trim(payload.get("tenant_access_token"))
    if not token:
        raise RuntimeError(f"tenant_access_token fetch failed: code={payload.get('code')} msg={payload.get('msg')}")
    return token


def _build_known_bot_catalog(pairs: list[CredentialPair]) -> list[BotIdentity]:
    known: list[BotIdentity] = []
    seen_app_ids: set[str] = set()
    for pair in pairs:
        if pair.app_id in seen_app_ids:
            continue
        seen_app_ids.add(pair.app_id)
        token = _get_tenant_access_token(pair.app_id, pair.app_secret)
        bot_info = _get_bot_info(token)
        known.append(
            BotIdentity(
                app_id_env=pair.app_id_env,
                app_secret_env=pair.app_secret_env,
                app_id=pair.app_id,
                app_name=_trim(bot_info.get("app_name")),
                bot_open_id=_trim(bot_info.get("open_id")),
            )
        )
    return known


def _get_application_info(app_id: str, token: str) -> dict[str, Any]:
    response = requests.get(
        f"https://open.feishu.cn/open-apis/application/v6/applications/{app_id}?lang=en_us",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    app = ((payload.get("data") or {}).get("app") or {}) if isinstance(payload, dict) else {}
    if not isinstance(app, dict):
        raise RuntimeError("application info payload did not contain an app object")
    return app


def _list_app_versions(app_id: str, token: str, *, page_size: int = 20) -> list[dict[str, Any]]:
    response = requests.get(
        f"https://open.feishu.cn/open-apis/application/v6/applications/{app_id}/app_versions",
        headers={"Authorization": f"Bearer {token}"},
        params={"page_size": max(1, min(int(page_size), 100)), "lang": "en_us"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    items = ((payload.get("data") or {}).get("items") or []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _coerce_event_types(event_infos: Any) -> list[str]:
    event_types: list[str] = []
    if not isinstance(event_infos, list):
        return event_types
    for item in event_infos:
        if isinstance(item, dict):
            event_type = _trim(item.get("event_type") or item.get("type") or item.get("event_name"))
        else:
            event_type = _trim(item)
        if event_type and event_type not in event_types:
            event_types.append(event_type)
    return event_types


def _coerce_visibility_open_ids(version: dict[str, Any]) -> list[str]:
    remark = version.get("remark") or {}
    if not isinstance(remark, dict):
        return []
    visibility = remark.get("visibility") or {}
    if not isinstance(visibility, dict):
        return []
    visible_list = visibility.get("visible_list") or {}
    if not isinstance(visible_list, dict):
        return []
    open_ids = visible_list.get("open_ids") or []
    if not isinstance(open_ids, list):
        return []
    return [_trim(item) for item in open_ids if _trim(item)]


def _summarize_published_app_versions(app_versions: list[dict[str, Any]], required_callbacks: list[str]) -> dict[str, Any]:
    versions = [item for item in app_versions if isinstance(item, dict)]
    if not versions:
        return {
            "available": False,
            "latest_published_event_types": [],
            "missing_required_callbacks": list(required_callbacks),
        }
    latest = max(
        versions,
        key=lambda item: (
            _coerce_int(item.get("publish_time")),
            _coerce_int(item.get("create_time")),
        ),
    )
    latest_event_types = _coerce_event_types(latest.get("event_infos"))
    missing_required_callbacks = [name for name in required_callbacks if name not in latest_event_types]
    return {
        "available": True,
        "version_count": len(versions),
        "latest_app_id": _trim(latest.get("app_id")),
        "latest_app_name": _trim(latest.get("app_name")),
        "latest_create_time": _coerce_int(latest.get("create_time")),
        "latest_publish_time": _coerce_int(latest.get("publish_time")),
        "latest_published_event_types": latest_event_types,
        "missing_required_callbacks": missing_required_callbacks,
        "message_receive_enabled": MESSAGE_EVENT_NAME in latest_event_types,
        "message_read_enabled": MESSAGE_READ_EVENT_NAME in latest_event_types,
        "latest_visibility_open_ids": _coerce_visibility_open_ids(latest),
    }


def _list_visible_chats(token: str, page_size: int) -> list[dict[str, Any]]:
    response = requests.get(
        "https://open.feishu.cn/open-apis/im/v1/chats",
        headers={"Authorization": f"Bearer {token}"},
        params={"page_size": max(1, min(int(page_size), 100))},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    items = ((payload.get("data") or {}).get("items") or []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "chat_id": _trim(item.get("chat_id")),
                "name": _trim(item.get("name")),
                "description": _trim(item.get("description")),
                "tenant_key": _trim(item.get("tenant_key")),
            }
        )
    return out


def _filter_visible_chats(visible_chats: list[dict[str, Any]], target_chat_id: str) -> list[dict[str, Any]]:
    normalized_target = _trim(target_chat_id)
    if not normalized_target:
        return visible_chats
    return [chat for chat in visible_chats if _trim(chat.get("chat_id")) == normalized_target]


def _get_chat_detail(token: str, chat_id: str) -> dict[str, Any]:
    response = requests.get(
        f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}",
        headers={"Authorization": f"Bearer {token}"},
        params={"user_id_type": "open_id"},
        timeout=20,
    )
    payload = response.json()
    if not response.ok or _coerce_int(payload.get("code")) != 0:
        return {
            "ok": False,
            "status_code": response.status_code,
            "code": payload.get("code"),
            "msg": _trim(payload.get("msg")),
        }
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    return {
        "ok": True,
        "chat_id": chat_id,
        "name": _trim(data.get("name")),
        "description": _trim(data.get("description")),
        "owner_id": _trim(data.get("owner_id")),
        "owner_id_type": _trim(data.get("owner_id_type")),
        "chat_type": _trim(data.get("chat_type")),
        "chat_mode": _trim(data.get("chat_mode")),
        "bot_count": _coerce_int(data.get("bot_count")),
        "user_count": _coerce_int(data.get("user_count")),
        "tenant_key": _trim(data.get("tenant_key")),
    }


def _get_chat_members_snapshot(token: str, chat_id: str) -> dict[str, Any]:
    response = requests.get(
        f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}/members",
        headers={"Authorization": f"Bearer {token}"},
        params={"member_id_type": "open_id", "page_size": 100},
        timeout=20,
    )
    payload = response.json()
    if not response.ok or _coerce_int(payload.get("code")) != 0:
        required_scopes = _extract_required_scopes(payload.get("msg"))
        return {
            "ok": False,
            "status_code": response.status_code,
            "code": payload.get("code"),
            "msg": _trim(payload.get("msg")),
            "required_scopes": required_scopes,
        }
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    items = data.get("items") or []
    normalized_items: list[dict[str, Any]] = []
    if isinstance(items, list):
        for item in items[:100]:
            if not isinstance(item, dict):
                continue
            normalized_items.append(
                {
                    "member_id": _trim(item.get("member_id")),
                    "member_id_type": _trim(item.get("member_id_type")),
                    "name": _trim(item.get("name")),
                    "tenant_key": _trim(item.get("tenant_key")),
                }
            )
    return {
        "ok": True,
        "source": "tenant",
        "member_total": _coerce_int(data.get("member_total")),
        "returned_count": len(normalized_items),
        "has_more": bool(data.get("has_more")),
        "page_token": _trim(data.get("page_token")),
        "items": normalized_items,
    }


def _get_user_chat_members_snapshot(user_access_token: str, chat_id: str) -> dict[str, Any]:
    normalized_token = _trim(user_access_token)
    if not normalized_token:
        return {"ok": False, "source": "user", "msg": "missing_user_access_token"}
    response = requests.get(
        f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}/members",
        headers={"Authorization": f"Bearer {normalized_token}"},
        params={"member_id_type": "open_id", "page_size": 100},
        timeout=20,
    )
    payload = response.json()
    if not response.ok or _coerce_int(payload.get("code")) != 0:
        required_scopes = _extract_required_scopes(payload.get("msg"))
        return {
            "ok": False,
            "source": "user",
            "status_code": response.status_code,
            "code": payload.get("code"),
            "msg": _trim(payload.get("msg")),
            "required_scopes": required_scopes,
        }
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    items = data.get("items") or []
    normalized_items: list[dict[str, Any]] = []
    if isinstance(items, list):
        for item in items[:100]:
            if not isinstance(item, dict):
                continue
            normalized_items.append(
                {
                    "member_id": _trim(item.get("member_id")),
                    "member_id_type": _trim(item.get("member_id_type")),
                    "name": _trim(item.get("name")),
                    "tenant_key": _trim(item.get("tenant_key")),
                }
            )
    return {
        "ok": True,
        "source": "user",
        "member_total": _coerce_int(data.get("member_total")),
        "returned_count": len(normalized_items),
        "has_more": bool(data.get("has_more")),
        "page_token": _trim(data.get("page_token")),
        "items": normalized_items,
    }


def _classify_user_member_blockers(user_members: dict[str, Any]) -> list[str]:
    if not isinstance(user_members, dict) or user_members.get("ok"):
        return []
    message = _trim(user_members.get("msg"))
    error_code = _coerce_int(user_members.get("code"))
    required_scopes = [str(item) for item in (user_members.get("required_scopes") or []) if _trim(item)]
    blockers: list[str] = []
    if message == "missing_user_access_token":
        blockers.append("user_access_token_missing")
        return blockers
    if error_code == FEISHU_TOKEN_EXPIRED_CODE or message == "Authentication token expired. Please request a new one.":
        blockers.append("user_token_expired")
        return blockers
    if error_code == FEISHU_SCOPE_MISSING_CODE and any(scope in CHAT_MEMBER_READ_SCOPES for scope in required_scopes):
        blockers.append("user_scope_missing_chat_members_read")
        return blockers
    if error_code == FEISHU_SCOPE_MISSING_CODE:
        blockers.append("user_scope_missing")
    return blockers


def _build_chat_topology_snapshot(token: str, chat: dict[str, Any], *, user_access_token: str = "") -> dict[str, Any]:
    chat_id = _trim(chat.get("chat_id"))
    if not chat_id:
        return {
            "chat_id": "",
            "detail": {"ok": False, "msg": "missing_chat_id"},
            "members": {"ok": False, "msg": "missing_chat_id"},
            "user_members": {"ok": False, "source": "user", "msg": "missing_chat_id"},
        }
    detail = _get_chat_detail(token, chat_id)
    members = _get_chat_members_snapshot(token, chat_id)
    user_members = _get_user_chat_members_snapshot(user_access_token, chat_id)
    blockers: list[str] = []
    if detail.get("ok") and _coerce_int(detail.get("bot_count")) > 1:
        blockers.append("multiple_bots_in_target_chat")
    if members.get("ok") and detail.get("ok"):
        expected_total = _coerce_int(detail.get("bot_count")) + _coerce_int(detail.get("user_count"))
        returned_total = _coerce_int(members.get("member_total")) or _coerce_int(members.get("returned_count"))
        if expected_total and returned_total and returned_total < expected_total:
            blockers.append("chat_member_view_incomplete")
    if not members.get("ok"):
        blockers.append("chat_members_unavailable")
    for blocker in _classify_user_member_blockers(user_members):
        if blocker not in blockers:
            blockers.append(blocker)
    return {
        "chat_id": chat_id,
        "detail": detail,
        "members": members,
        "user_members": user_members,
        "blockers": blockers,
    }


def _get_bot_info(token: str) -> dict[str, Any]:
    response = requests.get(
        "https://open.feishu.cn/open-apis/bot/v3/info",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    bot = payload.get("bot") or {}
    if not isinstance(bot, dict):
        return {}
    return {
        "app_name": _trim(bot.get("app_name")),
        "open_id": _trim(bot.get("open_id")),
        "activate_status": bot.get("activate_status"),
    }


def _iter_chat_messages(token: str, chat_id: str, *, page_size: int = 50, max_pages: int = 20) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_token = ""
    for _ in range(max(1, max_pages)):
        params = {
            "container_id_type": "chat",
            "container_id": chat_id,
            "page_size": max(1, min(int(page_size), 100)),
            "sort_type": "ByCreateTimeDesc",
        }
        if page_token:
            params["page_token"] = page_token
        response = requests.get(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        page_items = data.get("items") or []
        if not isinstance(page_items, list) or not page_items:
            break
        for item in page_items:
            if isinstance(item, dict):
                items.append(item)
        if not data.get("has_more"):
            break
        page_token = _trim(data.get("page_token"))
        if not page_token:
            break
    return items


def _message_preview(message: dict[str, Any], *, limit: int = 160) -> str:
    body = message.get("body") or {}
    if not isinstance(body, dict):
        return ""
    return _trim(body.get("content"))[:limit]


def _decorate_known_bot_open_id(
    open_id: str,
    known_bots_by_open_id: dict[str, BotIdentity],
    *,
    display_name: str = "",
) -> dict[str, Any]:
    decorated = {"open_id": open_id}
    if _trim(display_name):
        decorated["display_name"] = _trim(display_name)
    identity = known_bots_by_open_id.get(open_id)
    if identity is None:
        decorated["known_in_env"] = False
        return decorated
    decorated.update(
        {
            "known_in_env": True,
            "app_id": identity.app_id,
            "app_name": identity.app_name,
            "app_id_env": identity.app_id_env,
        }
    )
    return decorated


def _decorate_known_app_id(app_id: str, known_apps_by_id: dict[str, BotIdentity]) -> dict[str, Any]:
    decorated = {"app_id": app_id}
    identity = known_apps_by_id.get(app_id)
    if identity is None:
        decorated["known_in_env"] = False
        return decorated
    decorated.update(
        {
            "known_in_env": True,
            "app_name": identity.app_name,
            "app_id_env": identity.app_id_env,
            "bot_open_id": identity.bot_open_id,
        }
    )
    return decorated


def _format_bot_ref(payload: dict[str, Any]) -> str:
    open_id = _trim(payload.get("open_id"))
    if not open_id:
        return "unknown bot"
    display_name = _trim(payload.get("display_name"))
    if payload.get("known_in_env"):
        app_name = _trim(payload.get("app_name"))
        app_id = _trim(payload.get("app_id"))
        if app_name and app_id:
            return f"{app_name} ({app_id}, {open_id})"
        if app_id:
            return f"{app_id} ({open_id})"
    if display_name:
        return f"{display_name} ({open_id})"
    return open_id


def _format_app_ref(payload: dict[str, Any]) -> str:
    app_id = _trim(payload.get("app_id"))
    if not app_id:
        return "unknown app"
    if payload.get("known_in_env"):
        app_name = _trim(payload.get("app_name"))
        if app_name:
            return f"{app_name} ({app_id})"
    return app_id


def _append_repair_action(actions: list[dict[str, Any]], *, code: str, title: str, detail: str, priority: int) -> None:
    normalized_code = _trim(code)
    if not normalized_code:
        return
    for item in actions:
        if _trim(item.get("code")) == normalized_code:
            return
    actions.append(
        {
            "code": normalized_code,
            "title": _trim(title),
            "detail": _trim(detail),
            "priority": int(priority),
        }
    )


def _repair_action_priority(action: dict[str, Any]) -> int:
    value = action.get("priority")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 99


def _build_app_repair_actions(app_report: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    app_ref = _format_app_ref(app_report)
    published = app_report.get("published_version_summary") or {}
    missing_callbacks = [str(item) for item in (published.get("missing_required_callbacks") or [])]
    for callback in missing_callbacks:
        if callback == MESSAGE_READ_EVENT_NAME:
            _append_repair_action(
                actions,
                code="publish_message_read_callback",
                title="Publish message_read event",
                detail=f"Publish `{app_ref}` with `{MESSAGE_READ_EVENT_NAME}` enabled so read-receipt correlation can be measured.",
                priority=0,
            )
        elif callback == MESSAGE_EVENT_NAME:
            _append_repair_action(
                actions,
                code="publish_message_receive_callback",
                title="Publish message receive event",
                detail=f"Publish `{app_ref}` with `{MESSAGE_EVENT_NAME}` enabled so real text ingress can reach Hermes.",
                priority=0,
            )
        elif callback == "card.action.trigger":
            _append_repair_action(
                actions,
                code="publish_card_action_callback",
                title="Publish card action event",
                detail=f"Publish `{app_ref}` with `card.action.trigger` enabled so card button clicks can reach Hermes.",
                priority=0,
            )
        elif callback == "application.bot.menu_v6":
            _append_repair_action(
                actions,
                code="publish_bot_menu_callback",
                title="Publish bot menu event",
                detail=f"Publish `{app_ref}` with `application.bot.menu_v6` enabled so menu/control actions can be verified on the retained app.",
                priority=1,
            )

    delivery_health = app_report.get("delivery_health") or {}
    for message in delivery_health.get("recommendation_actions") or []:
        _append_repair_action(
            actions,
            code=f"delivery_health:{len(actions)}",
            title="Fix delivery config",
            detail=str(message),
            priority=1,
        )

    recent = app_report.get("recent_chat_activity") or {}
    primary_blocker = _trim(recent.get("primary_blocker"))
    if primary_blocker == "recent_messages_target_different_bot":
        other_bot = (recent.get("total_recent_user_messages_targeting_other_bot_ids") or [{}])[0]
        other_app = (recent.get("total_recent_app_messages_from_other_app_ids") or [{}])[0]
        _append_repair_action(
            actions,
            code="converge_target_bot_mentions",
            title="Converge target chat to one bot identity",
            detail=(
                f"Recent user intent is still targeting `{_format_bot_ref(other_bot)}`"
                + (
                    f" and replies are coming from `{_format_app_ref(other_app)}`"
                    if _trim(other_app.get("app_id"))
                    else ""
                )
                + f". Remove or silence competing bots so `{app_ref}` becomes the only active Hermes bot in the target chat."
            ),
            priority=0,
        )
    elif primary_blocker == "recent_messages_answered_by_other_app":
        other_app = (recent.get("total_recent_app_messages_from_other_app_ids") or [{}])[0]
        _append_repair_action(
            actions,
            code="stop_other_replying_app",
            title="Stop other app from replying in target chat",
            detail=f"Recent replies are still coming from `{_format_app_ref(other_app)}`. Keep `{app_ref}` as the only retained replying app before re-running end-to-end verification.",
            priority=0,
        )
    elif primary_blocker == "recent_messages_do_not_target_current_bot":
        _append_repair_action(
            actions,
            code="retarget_real_user_messages",
            title="Retarget real-user messages to current bot",
            detail=f"Recent user messages are not explicitly targeting `{app_ref}`. Update operator instructions to DM the canonical bot or @mention it explicitly in group chats.",
            priority=1,
        )

    topology = app_report.get("chat_topology") or {}
    if int(topology.get("chats_with_multiple_bots") or 0) > 0:
        _append_repair_action(
            actions,
            code="reduce_multi_bot_contention",
            title="Reduce multi-bot contention",
            detail=f"At least one target chat still contains multiple bots. Remove or operationally silence non-canonical bots before validating ingress and KPI windows for `{app_ref}`.",
            priority=0,
        )
    if int(topology.get("chats_with_expired_user_token") or 0) > 0:
        _append_repair_action(
            actions,
            code="refresh_user_access_token",
            title="Refresh user access token",
            detail="Refresh `FEISHU_USER_ACCESS_TOKEN` and re-run the focused member audit so the full user-view member list can be inspected.",
            priority=1,
        )
    if int(topology.get("chats_with_missing_user_access_token") or 0) > 0:
        _append_repair_action(
            actions,
            code="supply_user_access_token",
            title="Supply user access token",
            detail="Provide a fresh `FEISHU_USER_ACCESS_TOKEN` from the latest OAuth authorization before re-running the focused member audit.",
            priority=1,
        )
    if int(topology.get("chats_with_user_scope_missing_chat_members_read") or 0) > 0:
        _append_repair_action(
            actions,
            code="grant_user_chat_member_scope",
            title="Grant user chat-member scopes",
            detail=(
                "Grant one of `im:chat.members:read`, `im:chat.group_info:readonly`, "
                "`im:chat:readonly`, or `im:chat` for user-token access on the canonical app, "
                "then re-authorize before re-running the member audit."
            ),
            priority=1,
        )
    if int(topology.get("chats_with_incomplete_members") or 0) > 0:
        _append_repair_action(
            actions,
            code="rerun_member_audit",
            title="Re-run full member audit",
            detail="Tenant-view member enumeration is incomplete relative to chat topology. Re-run the audit after refreshing the user token to verify full bot membership.",
            priority=1,
        )

    actions.sort(key=lambda item: (_repair_action_priority(item), _trim(item.get("code"))))
    return actions


def _build_matrix_repair_actions(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for result in results:
        for action in result.get("repair_actions") or []:
            _append_repair_action(
                merged,
                code=_trim(action.get("code")),
                title=_trim(action.get("title")),
                detail=_trim(action.get("detail")),
                priority=_repair_action_priority(action),
            )
    merged.sort(key=lambda item: (_repair_action_priority(item), _trim(item.get("code"))))
    return merged


def _analyze_recent_user_messages(
    messages: list[dict[str, Any]],
    *,
    bot_open_id: str,
    since_ms: int,
    known_bots_by_open_id: dict[str, BotIdentity] | None = None,
) -> dict[str, Any]:
    recent_user_messages: list[dict[str, Any]] = []
    known_bots_by_open_id = known_bots_by_open_id or {}
    targeted_bot_count = 0
    without_mentions_count = 0
    mention_targets_other_bots: dict[str, int] = {}
    mention_names_by_id: dict[str, str] = {}
    for message in messages:
        try:
            created_ms = int(message.get("create_time") or 0)
        except (TypeError, ValueError):
            created_ms = 0
        if created_ms < since_ms:
            continue
        sender = message.get("sender") or {}
        if not isinstance(sender, dict) or _trim(sender.get("sender_type")) != "user":
            continue
        mentions = message.get("mentions") or []
        mention_ids: list[str] = []
        mention_name_pairs: list[tuple[str, str]] = []
        for item in mentions:
            if not isinstance(item, dict):
                continue
            mention_id = _trim(item.get("id"))
            if not mention_id:
                continue
            mention_name = _trim(item.get("name"))
            mention_ids.append(mention_id)
            mention_name_pairs.append((mention_id, mention_name))
            if mention_name and mention_id not in mention_names_by_id:
                mention_names_by_id[mention_id] = mention_name
        targeted_current_bot = bool(bot_open_id) and bot_open_id in mention_ids
        if targeted_current_bot:
            targeted_bot_count += 1
        elif mention_ids:
            for mention_id in mention_ids:
                if mention_id == bot_open_id:
                    continue
                mention_targets_other_bots[mention_id] = mention_targets_other_bots.get(mention_id, 0) + 1
        else:
            without_mentions_count += 1
        recent_user_messages.append(
            {
                "message_id": _trim(message.get("message_id")),
                "create_time_local": datetime.fromtimestamp(created_ms / 1000).astimezone().isoformat(timespec="seconds"),
                "msg_type": _trim(message.get("msg_type")),
                "mention_ids": mention_ids,
                "mention_targets": [
                    _decorate_known_bot_open_id(mention_id, known_bots_by_open_id, display_name=mention_name)
                    for mention_id, mention_name in mention_name_pairs
                ],
                "targets_current_bot": targeted_current_bot,
                "content_preview": _message_preview(message),
            }
        )
    dominant_other_mentions = [
        {
            **_decorate_known_bot_open_id(
                open_id,
                known_bots_by_open_id,
                display_name=mention_names_by_id.get(open_id, ""),
            ),
            "count": count,
        }
        for open_id, count in sorted(mention_targets_other_bots.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "recent_user_message_count": len(recent_user_messages),
        "recent_user_messages_targeting_current_bot": targeted_bot_count,
        "recent_user_messages_without_mentions": without_mentions_count,
        "recent_user_messages_targeting_other_bot_ids": dominant_other_mentions,
        "recent_user_message_samples": recent_user_messages[-10:],
    }


def _analyze_recent_app_messages(
    messages: list[dict[str, Any]],
    *,
    current_app_id: str,
    since_ms: int,
    known_apps_by_id: dict[str, BotIdentity] | None = None,
) -> dict[str, Any]:
    recent_app_messages: list[dict[str, Any]] = []
    known_apps_by_id = known_apps_by_id or {}
    current_app_message_count = 0
    other_app_counts: dict[str, int] = {}
    for message in messages:
        created_ms = _coerce_int(message.get("create_time"))
        if created_ms < since_ms:
            continue
        sender = message.get("sender") or {}
        if not isinstance(sender, dict) or _trim(sender.get("sender_type")) != "app":
            continue
        sender_app_id = _trim(sender.get("id") if _trim(sender.get("id_type")) == "app_id" else sender.get("app_id"))
        from_current_app = bool(current_app_id) and sender_app_id == current_app_id
        if from_current_app:
            current_app_message_count += 1
        elif sender_app_id:
            other_app_counts[sender_app_id] = other_app_counts.get(sender_app_id, 0) + 1
        recent_app_messages.append(
            {
                "message_id": _trim(message.get("message_id")),
                "create_time_local": datetime.fromtimestamp(created_ms / 1000).astimezone().isoformat(timespec="seconds"),
                "msg_type": _trim(message.get("msg_type")),
                "sender_app_id": sender_app_id,
                "sender_app": _decorate_known_app_id(sender_app_id, known_apps_by_id) if sender_app_id else {},
                "from_current_app": from_current_app,
                "content_preview": _message_preview(message),
            }
        )
    dominant_other_app_ids = [
        {
            **_decorate_known_app_id(app_id, known_apps_by_id),
            "count": count,
        }
        for app_id, count in sorted(other_app_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    latest_app_message = recent_app_messages[-1] if recent_app_messages else {}
    return {
        "recent_app_message_count": len(recent_app_messages),
        "recent_app_messages_from_current_app": current_app_message_count,
        "recent_app_messages_from_other_app_ids": dominant_other_app_ids,
        "latest_recent_app_message_sender_app_id": _trim(latest_app_message.get("sender_app_id")),
        "latest_recent_app_message_sender_app": latest_app_message.get("sender_app") or {},
        "recent_app_message_samples": recent_app_messages[-10:],
    }


def _inspect_recent_chat_activity(
    token: str,
    visible_chats: list[dict[str, Any]],
    *,
    current_app_id: str,
    bot_open_id: str,
    window_minutes: int,
    known_bots_by_open_id: dict[str, BotIdentity] | None = None,
    known_apps_by_id: dict[str, BotIdentity] | None = None,
) -> dict[str, Any]:
    normalized_window = max(0, int(window_minutes))
    if normalized_window <= 0:
        return {"enabled": False}
    known_bots_by_open_id = known_bots_by_open_id or {}
    known_apps_by_id = known_apps_by_id or {}
    since_ms = int(time.time() * 1000) - normalized_window * 60 * 1000
    chat_reports: list[dict[str, Any]] = []
    total_recent_user_messages = 0
    total_recent_targeting_current_bot = 0
    total_recent_without_mentions = 0
    total_recent_other_bot_mentions: dict[str, int] = {}
    total_recent_app_messages = 0
    total_recent_app_messages_from_current_app = 0
    total_recent_other_app_messages: dict[str, int] = {}
    for chat in visible_chats:
        chat_id = _trim(chat.get("chat_id"))
        if not chat_id:
            continue
        messages = _iter_chat_messages(token, chat_id)
        analysis = _analyze_recent_user_messages(
            messages,
            bot_open_id=bot_open_id,
            since_ms=since_ms,
            known_bots_by_open_id=known_bots_by_open_id,
        )
        app_analysis = _analyze_recent_app_messages(
            messages,
            current_app_id=current_app_id,
            since_ms=since_ms,
            known_apps_by_id=known_apps_by_id,
        )
        total_recent_user_messages += int(analysis["recent_user_message_count"])
        total_recent_targeting_current_bot += int(analysis["recent_user_messages_targeting_current_bot"])
        total_recent_without_mentions += int(analysis["recent_user_messages_without_mentions"])
        for item in analysis["recent_user_messages_targeting_other_bot_ids"]:
            open_id = _trim(item.get("open_id"))
            if not open_id:
                continue
            total_recent_other_bot_mentions[open_id] = total_recent_other_bot_mentions.get(open_id, 0) + int(item.get("count") or 0)
        total_recent_app_messages += int(app_analysis["recent_app_message_count"])
        total_recent_app_messages_from_current_app += int(app_analysis["recent_app_messages_from_current_app"])
        for item in app_analysis["recent_app_messages_from_other_app_ids"]:
            app_id = _trim(item.get("app_id"))
            if not app_id:
                continue
            total_recent_other_app_messages[app_id] = total_recent_other_app_messages.get(app_id, 0) + int(item.get("count") or 0)
        chat_reports.append(
            {
                "chat_id": chat_id,
                "chat_name": _trim(chat.get("name")),
                **analysis,
                **app_analysis,
            }
        )
    dominant_other_mentions = [
        {
            **_decorate_known_bot_open_id(
                open_id,
                known_bots_by_open_id,
                display_name=next(
                    (
                        _trim(item.get("display_name"))
                        for report in chat_reports
                        for item in (report.get("recent_user_messages_targeting_other_bot_ids") or [])
                        if _trim(item.get("open_id")) == open_id and _trim(item.get("display_name"))
                    ),
                    "",
                ),
            ),
            "count": count,
        }
        for open_id, count in sorted(total_recent_other_bot_mentions.items(), key=lambda item: (-item[1], item[0]))
    ]
    dominant_other_app_messages = [
        {
            **_decorate_known_app_id(app_id, known_apps_by_id),
            "count": count,
        }
        for app_id, count in sorted(total_recent_other_app_messages.items(), key=lambda item: (-item[1], item[0]))
    ]
    if (
        total_recent_user_messages > 0
        and total_recent_app_messages > 0
        and total_recent_app_messages_from_current_app == 0
        and dominant_other_app_messages
    ):
        if total_recent_targeting_current_bot == 0 and dominant_other_mentions:
            primary_blocker = "recent_messages_target_different_bot"
        else:
            primary_blocker = "recent_messages_answered_by_other_app"
    elif total_recent_user_messages > 0 and total_recent_targeting_current_bot == 0:
        if dominant_other_mentions:
            primary_blocker = "recent_messages_target_different_bot"
        else:
            primary_blocker = "recent_messages_do_not_target_current_bot"
    else:
        primary_blocker = ""
    return {
        "enabled": True,
        "window_minutes": normalized_window,
        "current_app_id": current_app_id,
        "current_app": _decorate_known_app_id(current_app_id, known_apps_by_id) if current_app_id else {},
        "bot_open_id": bot_open_id,
        "current_bot": _decorate_known_bot_open_id(bot_open_id, known_bots_by_open_id) if bot_open_id else {},
        "total_recent_user_messages": total_recent_user_messages,
        "total_recent_user_messages_targeting_current_bot": total_recent_targeting_current_bot,
        "total_recent_user_messages_without_mentions": total_recent_without_mentions,
        "total_recent_user_messages_targeting_other_bot_ids": dominant_other_mentions,
        "total_recent_app_messages": total_recent_app_messages,
        "total_recent_app_messages_from_current_app": total_recent_app_messages_from_current_app,
        "total_recent_app_messages_from_other_app_ids": dominant_other_app_messages,
        "chat_reports": chat_reports,
        "primary_blocker": primary_blocker,
    }


def _summarize_chat_topology_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    total_known_bots = 0
    total_known_users = 0
    chats_with_multiple_bots = 0
    chats_with_incomplete_members = 0
    chats_with_member_api_errors = 0
    chats_with_expired_user_token = 0
    chats_with_missing_user_access_token = 0
    chats_with_user_scope_missing_chat_members_read = 0
    for snapshot in snapshots:
        detail = snapshot.get("detail") or {}
        members = snapshot.get("members") or {}
        blockers = snapshot.get("blockers") or []
        if detail.get("ok"):
            total_known_bots += _coerce_int(detail.get("bot_count"))
            total_known_users += _coerce_int(detail.get("user_count"))
        if "multiple_bots_in_target_chat" in blockers:
            chats_with_multiple_bots += 1
        if "chat_member_view_incomplete" in blockers:
            chats_with_incomplete_members += 1
        if "chat_members_unavailable" in blockers:
            chats_with_member_api_errors += 1
        if "user_token_expired" in blockers:
            chats_with_expired_user_token += 1
        if "user_access_token_missing" in blockers:
            chats_with_missing_user_access_token += 1
        if "user_scope_missing_chat_members_read" in blockers:
            chats_with_user_scope_missing_chat_members_read += 1
    return {
        "chat_count": len(snapshots),
        "total_known_bots": total_known_bots,
        "total_known_users": total_known_users,
        "chats_with_multiple_bots": chats_with_multiple_bots,
        "chats_with_incomplete_members": chats_with_incomplete_members,
        "chats_with_member_api_errors": chats_with_member_api_errors,
        "chats_with_expired_user_token": chats_with_expired_user_token,
        "chats_with_missing_user_access_token": chats_with_missing_user_access_token,
        "chats_with_user_scope_missing_chat_members_read": chats_with_user_scope_missing_chat_members_read,
        "snapshots": snapshots,
    }


def _probe_url_verification(url: str, timeout_seconds: int) -> dict[str, Any]:
    target = _trim(url)
    if not target:
        return {"status": "skipped", "reason": "missing_url"}
    challenge = f"probe-{int(time.time() * 1000)}"
    started = time.perf_counter()
    try:
        response = requests.post(
            target,
            json={"type": "url_verification", "challenge": challenge},
            timeout=max(1, timeout_seconds),
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        parsed: Any
        try:
            parsed = response.json()
        except ValueError:
            parsed = {"raw": response.text[:300]}
        returned_challenge = ""
        if isinstance(parsed, dict):
            returned_challenge = _trim(parsed.get("challenge"))
        return {
            "status": "ok" if response.ok and returned_challenge == challenge else "error",
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "challenge_matched": returned_challenge == challenge,
            "response": parsed,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


def _coerce_callback_url(callback_info: dict[str, Any]) -> str:
    return _trim(
        callback_info.get("callback_url")
        or callback_info.get("request_url")
        or callback_info.get("url")
        or callback_info.get("callback")
    )


def _audit_app(
    pair: CredentialPair,
    *,
    expected_mode: str,
    expected_webhook_url: str,
    required_callbacks: list[str],
    probe_timeout: int,
    chat_page_size: int,
    recent_message_window_minutes: int,
    target_chat_id: str,
    user_access_token: str,
    known_bots_by_open_id: dict[str, BotIdentity],
    known_apps_by_id: dict[str, BotIdentity],
) -> dict[str, Any]:
    token = _get_tenant_access_token(pair.app_id, pair.app_secret)
    app = _get_application_info(pair.app_id, token)
    app_versions = _list_app_versions(pair.app_id, token)
    bot_info = _get_bot_info(token)
    callback_info = app.get("callback_info") or {}
    if not isinstance(callback_info, dict):
        callback_info = {}
    visible_chats = _filter_visible_chats(_list_visible_chats(token, chat_page_size), target_chat_id)
    chat_topology = _summarize_chat_topology_snapshots(
        [_build_chat_topology_snapshot(token, chat, user_access_token=user_access_token) for chat in visible_chats]
    )
    recent_chat_activity = _inspect_recent_chat_activity(
        token,
        visible_chats,
        current_app_id=pair.app_id,
        bot_open_id=_trim(bot_info.get("open_id")),
        window_minutes=recent_message_window_minutes,
        known_bots_by_open_id=known_bots_by_open_id,
        known_apps_by_id=known_apps_by_id,
    )
    report = analyze_feishu_delivery_config(
        runtime_mode=expected_mode,
        callback_info=callback_info,
        expected_webhook_url=expected_webhook_url,
        required_callbacks=required_callbacks or None,
    )
    published_version_summary = _summarize_published_app_versions(app_versions, report["required_callbacks"])
    published_missing_callbacks = published_version_summary.get("missing_required_callbacks") or []
    report_missing_callbacks = report.get("missing_callbacks") or []
    effective_missing_callbacks = [
        item for item in report_missing_callbacks if item in published_missing_callbacks
    ]
    if report_missing_callbacks != effective_missing_callbacks:
        retained_issues = [
            issue for issue in report.get("issues", []) if issue.get("code") != "missing_callbacks"
        ]
        report = {
            **report,
            "issues": retained_issues,
            "missing_callbacks": effective_missing_callbacks,
            "status": "ok" if not retained_issues else report["status"],
        }
        if effective_missing_callbacks:
            report["issues"] = retained_issues + [
                {
                    "code": "missing_callbacks",
                    "severity": "critical",
                    "message": "Feishu app is missing required callbacks: " + ", ".join(effective_missing_callbacks),
                }
            ]
            report["status"] = "error"
            report["summary"] = report["issues"][0]["message"]
        else:
            report["summary"] = "Feishu app callback configuration matches the Hermes runtime delivery mode."
    callback_url = _coerce_callback_url(callback_info)
    callback_probe = _probe_url_verification(callback_url, probe_timeout) if callback_url else {"status": "skipped"}
    expected_probe = (
        _probe_url_verification(expected_webhook_url, probe_timeout)
        if expected_webhook_url and expected_webhook_url != callback_url
        else {"status": "skipped"}
    )
    missing_callbacks = report.get("missing_callbacks") or []
    primary_blocker = _trim(recent_chat_activity.get("primary_blocker"))
    if not primary_blocker:
        if MESSAGE_EVENT_NAME in published_missing_callbacks:
            primary_blocker = "published_version_missing_message_receive_callback"
        elif "card.action.trigger" in published_missing_callbacks:
            primary_blocker = "published_version_missing_card_action_callback"
        elif "application.bot.menu_v6" in published_missing_callbacks:
            primary_blocker = "published_version_missing_bot_menu_callback"
        elif MESSAGE_EVENT_NAME in missing_callbacks and not published_version_summary.get("message_receive_enabled"):
            primary_blocker = "api_reported_missing_message_receive_callback"
        elif report.get("issues"):
            primary_blocker = report["issues"][0]["code"]
    repair_actions = _build_app_repair_actions(
        {
            "app_id": app.get("app_id") or pair.app_id,
            "app_name": _trim(app.get("app_name")),
            "published_version_summary": published_version_summary,
            "delivery_health": report,
            "recent_chat_activity": recent_chat_activity,
            "chat_topology": chat_topology,
        }
    )
    return {
        "status": report["status"],
        "app_id_env": pair.app_id_env,
        "app_secret_env": pair.app_secret_env,
        "app_id": app.get("app_id") or pair.app_id,
        "app_name": _trim(app.get("app_name")),
        "bot_info": bot_info,
        "known_bot_identity": _decorate_known_bot_open_id(_trim(bot_info.get("open_id")), known_bots_by_open_id),
        "expected_mode": expected_mode,
        "target_chat_id": _trim(target_chat_id),
        "visible_chat_count": len(visible_chats),
        "visible_chats": visible_chats,
        "chat_topology": chat_topology,
        "recent_chat_activity": recent_chat_activity,
        "callback_info": sanitize_callback_info(callback_info),
        "callback_info_source": "application/v6.app.callback_info",
        "callback_info_observation_note": (
            "The Feishu application info API may report only a subset of event subscriptions. "
            "Prefer the published app version event list when callback_info disagrees with live behavior or the developer console."
        ),
        "published_version_summary": published_version_summary,
        "published_version_source": "application/v6.app_versions",
        "callback_url_probe": callback_probe,
        "expected_webhook_probe": expected_probe,
        "delivery_health": report,
        "delivery_ready": report["status"] == "ok",
        "primary_blocker": primary_blocker,
        "repair_actions": repair_actions,
    }


def _single_pair_from_args(args: argparse.Namespace) -> CredentialPair | None:
    app_id = _trim(args.app_id)
    app_secret = _trim(args.app_secret)
    if not app_id or not app_secret:
        return None
    return CredentialPair(
        app_id_env="FEISHU_APP_ID",
        app_secret_env="FEISHU_APP_SECRET",
        app_id=app_id,
        app_secret=app_secret,
    )


def _summarize_matrix(results: list[dict[str, Any]]) -> tuple[str, str]:
    active_apps = [item for item in results if int(item.get("visible_chat_count") or 0) > 0]
    ready_active_apps = [item for item in active_apps if bool(item.get("delivery_ready"))]
    preferred_active_apps = ready_active_apps + [item for item in active_apps if item not in ready_active_apps]
    for item in preferred_active_apps:
        recent = item.get("recent_chat_activity") or {}
        topology = item.get("chat_topology") or {}
        if _trim(recent.get("primary_blocker")) == "recent_messages_target_different_bot":
            dominant = (recent.get("total_recent_user_messages_targeting_other_bot_ids") or [{}])[0]
            hinted_target = _format_bot_ref(dominant)
            dominant_app = (recent.get("total_recent_app_messages_from_other_app_ids") or [{}])[0]
            hinted_app = _format_app_ref(dominant_app)
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see recent user messages in its visible chats, "
                    f"but none target this bot open_id `{_trim((item.get('bot_info') or {}).get('open_id'))}`. "
                    f"Recent mentions are targeting a different bot like `{hinted_target}` instead."
                    + (
                        f" Recent replies in the same chats are coming from another app like `{hinted_app}`."
                        if hinted_app and hinted_app != "unknown app"
                        else ""
                    )
                    + (
                        f" The target chat currently reports `{int(topology.get('total_known_bots') or 0)}` bot(s), "
                        "so bot competition is a live routing risk."
                        if int(topology.get("total_known_bots") or 0) > 1
                        else ""
                    )
                ),
            )
        if _trim(recent.get("primary_blocker")) == "recent_messages_answered_by_other_app":
            dominant_app = (recent.get("total_recent_app_messages_from_other_app_ids") or [{}])[0]
            hinted_app = _format_app_ref(dominant_app)
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see recent user messages, but recent app replies "
                    f"in the same chats are coming from a different Feishu app like `{hinted_app}` instead."
                ),
            )
        if _trim(recent.get("primary_blocker")) == "recent_messages_do_not_target_current_bot":
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see recent user messages, but none of them target "
                    f"its bot open_id `{_trim((item.get('bot_info') or {}).get('open_id'))}`."
                ),
            )
    for item in preferred_active_apps:
        topology = item.get("chat_topology") or {}
        if int(topology.get("chats_with_multiple_bots") or 0) > 0:
            user_member_gap = ""
            if int(topology.get("chats_with_user_scope_missing_chat_members_read") or 0) > 0:
                user_member_gap = (
                    " The available user token still lacks one of "
                    "`im:chat.members:read`, `im:chat.group_info:readonly`, `im:chat:readonly`, or `im:chat`, "
                    "so the full human-view member list cannot yet be audited."
                )
            elif int(topology.get("chats_with_expired_user_token") or 0) > 0:
                user_member_gap = (
                    " The available user_access_token is expired, so a fresh user token is still needed to verify "
                    "the full human-view member list."
                )
            elif int(topology.get("chats_with_missing_user_access_token") or 0) > 0:
                user_member_gap = (
                    " No user_access_token is available yet, so the full human-view member list cannot be verified."
                )
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see target chats, but at least one target chat still contains "
                    f"`{int(topology.get('total_known_bots') or 0)}` bot(s), so single-bot Hermes routing is not yet enforced."
                    + user_member_gap
                ),
            )
    for item in preferred_active_apps:
        topology = item.get("chat_topology") or {}
        if int(topology.get("chats_with_user_scope_missing_chat_members_read") or 0) > 0:
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see target chats, but the user-token member audit still "
                    "lacks one of `im:chat.members:read`, `im:chat.group_info:readonly`, `im:chat:readonly`, or "
                    "`im:chat`, so the human-view member topology cannot yet be verified."
                ),
            )
    if ready_active_apps:
        first = ready_active_apps[0]
        return (
            "ok",
            (
                f"Active app `{first.get('app_id')}` can see chats and its callback configuration "
                "matches Hermes webhook delivery expectations."
            ),
        )
    for item in active_apps:
        published = item.get("published_version_summary") or {}
        if MESSAGE_EVENT_NAME in (published.get("missing_required_callbacks") or []):
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see {item.get('visible_chat_count')} chat(s), "
                    f"but its latest published app version still does not include `{MESSAGE_EVENT_NAME}`."
                ),
            )
        if "card.action.trigger" in (published.get("missing_required_callbacks") or []):
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see {item.get('visible_chat_count')} chat(s), "
                    "but its latest published app version still does not include `card.action.trigger`, "
                    "so card button clicks will not reach Hermes."
                ),
            )
        if "application.bot.menu_v6" in (published.get("missing_required_callbacks") or []):
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see {item.get('visible_chat_count')} chat(s), "
                    "but its latest published app version still does not include `application.bot.menu_v6`, "
                    "so menu control actions will not reach Hermes."
                ),
            )
    for item in active_apps:
        blocker = _trim(item.get("primary_blocker"))
        if blocker == "api_reported_missing_message_receive_callback":
            return (
                "error",
                (
                    f"Active app `{item.get('app_id')}` can see {item.get('visible_chat_count')} chat(s), "
                    f"but the Feishu application info API does not report `{MESSAGE_EVENT_NAME}`. "
                    "Cross-check the published developer-console event subscription list before treating this as the root cause."
                ),
            )
    if active_apps:
        first = active_apps[0]
        issue = ((first.get("delivery_health") or {}).get("issues") or [{}])[0]
        return (
            "error",
            f"Active app `{first.get('app_id')}` is visible in chats but delivery is blocked by `{_trim(issue.get('code'))}`.",
        )
    if results:
        return ("error", "No audited Feishu app can currently see any target chats, so live delivery cannot succeed.")
    return ("error", "No Feishu app credentials were discovered for audit.")


def main() -> int:
    args = _parse_args()
    expected_mode = _trim(args.expected_mode).lower() or "webhook"
    expected_webhook_url = _build_expected_webhook_url(args.expected_webhook_url)
    target_chat_id = _trim(args.target_chat_id)
    user_access_token = _trim(args.user_access_token)
    credential_pairs = _discover_credential_pairs() if args.all_env_apps else []
    if not credential_pairs:
        pair = _single_pair_from_args(args)
        if pair is not None:
            credential_pairs = [pair]
    if not credential_pairs:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": (
                        "Missing Feishu app credentials. Pass --app-id/--app-secret or export "
                        "FEISHU_APP_ID*/FEISHU_APP_SECRET* and use --all-env-apps."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    required_callbacks = list(args.required_callback or [])
    try:
        known_pairs = _discover_credential_pairs()
        for pair in credential_pairs:
            if not any(item.app_id == pair.app_id for item in known_pairs):
                known_pairs.append(pair)
        known_bots = _build_known_bot_catalog(known_pairs)
        known_bots_by_open_id = {item.bot_open_id: item for item in known_bots if item.bot_open_id}
        known_apps_by_id = {item.app_id: item for item in known_bots if item.app_id}
        results = [
            _audit_app(
                pair,
                expected_mode=expected_mode,
                expected_webhook_url=expected_webhook_url,
                required_callbacks=required_callbacks,
                probe_timeout=args.probe_timeout,
                chat_page_size=args.chat_page_size,
                recent_message_window_minutes=args.recent_message_window_minutes,
                target_chat_id=target_chat_id,
                user_access_token=user_access_token,
                known_bots_by_open_id=known_bots_by_open_id,
                known_apps_by_id=known_apps_by_id,
            )
            for pair in credential_pairs
        ]
        overall_status, summary = _summarize_matrix(results)
        recommended_actions = _build_matrix_repair_actions(results)
        if len(results) == 1 and not args.all_env_apps:
            payload = results[0]
            payload["status"] = overall_status if payload.get("visible_chat_count") else payload["status"]
            payload["summary"] = summary
            payload["recommended_actions"] = recommended_actions
        else:
            payload = {
                "status": overall_status,
                "summary": summary,
                "expected_mode": expected_mode,
                "expected_webhook_url": expected_webhook_url,
                "target_chat_id": target_chat_id,
                "known_bots": [
                    {
                        "app_id_env": item.app_id_env,
                        "app_id": item.app_id,
                        "app_name": item.app_name,
                        "bot_open_id": item.bot_open_id,
                    }
                    for item in known_bots
                ],
                "apps": results,
                "recommended_actions": recommended_actions,
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if overall_status == "ok" else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
