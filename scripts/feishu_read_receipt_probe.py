from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp


DEFAULT_CHAT_ID = os.getenv("FEISHU_TEST_CHAT_ID") or os.getenv("FEISHU_HOME_CHANNEL") or ""

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


def _trim(value: Any) -> str:
    return str(value or "").strip()


def _parse_epoch_ms(value: Any) -> int:
    raw = _trim(value)
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _format_local_time(epoch_ms: int) -> str:
    if epoch_ms <= 0:
        return ""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _extract_read_users_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        return []
    for key in ("items", "read_users"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _message_content_preview(message: dict[str, Any], limit: int = 120) -> str:
    body = message.get("body") or {}
    if not isinstance(body, dict):
        return ""
    content = _trim(body.get("content"))
    if len(content) <= limit:
        return content
    return content[:limit] + "..."


def _is_current_app_message(message: dict[str, Any], current_app_id: str) -> bool:
    sender = message.get("sender") or {}
    if not isinstance(sender, dict):
        return False
    sender_app_id = _trim(sender.get("sender_id") or sender.get("sender_app_id") or sender.get("id"))
    sender_type = _trim(sender.get("sender_type"))
    return bool(current_app_id) and sender_type == "app" and sender_app_id == current_app_id


def _filter_recent_app_messages(
    messages: list[dict[str, Any]],
    *,
    current_app_id: str,
    window_minutes: int,
) -> list[dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    min_created_ms = now_ms - max(0, window_minutes) * 60 * 1000
    filtered: list[dict[str, Any]] = []
    for message in messages:
        created_ms = _parse_epoch_ms(message.get("create_time"))
        if created_ms and created_ms < min_created_ms:
            continue
        if not _is_current_app_message(message, current_app_id):
            continue
        filtered.append(message)
    filtered.sort(key=lambda item: _parse_epoch_ms(item.get("create_time")), reverse=True)
    return filtered


async def _get_tenant_token(session: aiohttp.ClientSession, app_id: str, app_secret: str) -> str:
    async with session.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
    ) as resp:
        payload = await resp.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"tenant_access_token_failed:{payload}")
        token = _trim(payload.get("tenant_access_token"))
        if not token:
            raise RuntimeError(f"tenant_access_token_missing:{payload}")
        return token


async def _list_recent_messages(
    session: aiohttp.ClientSession,
    *,
    tenant_token: str,
    chat_id: str,
    page_size: int,
) -> list[dict[str, Any]]:
    async with session.get(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        headers={"Authorization": f"Bearer {tenant_token}"},
        params={
            "container_id_type": "chat",
            "container_id": chat_id,
            "page_size": max(1, min(page_size, 100)),
            "sort_type": "ByCreateTimeDesc",
        },
    ) as resp:
        payload = await resp.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"list_messages_failed:{payload}")
        data = payload.get("data") or {}
        items = data.get("items") or []
        return [item for item in items if isinstance(item, dict)]


async def _fetch_read_users(
    session: aiohttp.ClientSession,
    *,
    tenant_token: str,
    message_id: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_token = ""
    while True:
        params = {"user_id_type": "open_id"}
        if page_token:
            params["page_token"] = page_token
        async with session.get(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/read_users",
            headers={"Authorization": f"Bearer {tenant_token}"},
            params=params,
        ) as resp:
            payload = await resp.json()
            if payload.get("code") != 0:
                raise RuntimeError(f"read_users_failed:{message_id}:{payload}")
            page_items = _extract_read_users_items(payload)
            items.extend(page_items)
            data = payload.get("data") or {}
            if not bool(data.get("has_more")):
                break
            page_token = _trim(data.get("page_token"))
            if not page_token:
                break
    return items


def _summarize_message(message: dict[str, Any], read_users: list[dict[str, Any]]) -> dict[str, Any]:
    message_id = _trim(message.get("message_id"))
    created_ms = _parse_epoch_ms(message.get("create_time"))
    read_times = [_parse_epoch_ms(item.get("read_time")) for item in read_users]
    read_times = [item for item in read_times if item > 0]
    first_read_ms = min(read_times) if read_times else 0
    return {
        "message_id": message_id,
        "create_time_ms": created_ms,
        "create_time_local": _format_local_time(created_ms),
        "msg_type": _trim(message.get("msg_type")),
        "content_preview": _message_content_preview(message),
        "read_user_count": len(read_users),
        "first_read_time_ms": first_read_ms or None,
        "first_read_time_local": _format_local_time(first_read_ms) if first_read_ms else "",
        "first_read_latency_ms": (first_read_ms - created_ms) if first_read_ms and created_ms else None,
        "reader_open_ids": [_trim(item.get("reader_id") or item.get("open_id") or item.get("user_id")) for item in read_users],
    }


async def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    app_id = _trim(args.app_id)
    app_secret = _trim(args.app_secret)
    chat_id = _trim(args.chat_id)
    if not app_id or not app_secret or not chat_id:
        raise RuntimeError("missing_app_credentials_or_chat_id")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        tenant_token = await _get_tenant_token(session, app_id, app_secret)
        messages = await _list_recent_messages(
            session,
            tenant_token=tenant_token,
            chat_id=chat_id,
            page_size=args.page_size,
        )
        recent_app_messages = _filter_recent_app_messages(
            messages,
            current_app_id=app_id,
            window_minutes=args.window_minutes,
        )[: args.limit]

        samples: list[dict[str, Any]] = []
        for message in recent_app_messages:
            message_id = _trim(message.get("message_id"))
            if not message_id:
                continue
            read_users = await _fetch_read_users(session, tenant_token=tenant_token, message_id=message_id)
            samples.append(_summarize_message(message, read_users))

    read_latencies = [item["first_read_latency_ms"] for item in samples if isinstance(item.get("first_read_latency_ms"), int)]
    return {
        "status": "ok",
        "app_id": app_id,
        "chat_id": chat_id,
        "window_minutes": args.window_minutes,
        "total_recent_messages_scanned": len(messages),
        "matched_recent_app_messages": len(recent_app_messages),
        "sample_count": len(samples),
        "read_receipt_sample_count": len(read_latencies),
        "read_receipt_min_latency_ms": min(read_latencies) if read_latencies else None,
        "read_receipt_max_latency_ms": max(read_latencies) if read_latencies else None,
        "read_receipt_all_zero": bool(samples) and not read_latencies,
        "samples": samples,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe recent Feishu app messages and collect read-users samples.")
    parser.add_argument("--app-id", default=os.getenv("FEISHU_APP_ID3", ""))
    parser.add_argument("--app-secret", default=os.getenv("FEISHU_APP_SECRET3", ""))
    parser.add_argument("--chat-id", default=DEFAULT_CHAT_ID)
    parser.add_argument("--window-minutes", type=int, default=180)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=10)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        payload = asyncio.run(_run_probe(args))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
