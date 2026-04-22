from __future__ import annotations

import json
from re import Pattern
from typing import Any, Mapping


def extract_event_metadata(payload: dict[str, Any]) -> tuple[str, str]:
    header = payload.get("header") or {}
    event_id = str(header.get("event_id") or payload.get("event_id") or "").strip()
    event_type = str(header.get("event_type") or "").strip()
    return event_id, event_type


def extract_leading_command_text(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    tokens = normalized.split()
    while tokens:
        first = tokens[0]
        if first.startswith("@") or first.startswith("[@"):
            tokens.pop(0)
            continue
        if first.startswith("<at"):
            while tokens:
                token = tokens.pop(0)
                if "</at>" in token:
                    break
            continue
        break
    if not tokens or not tokens[0].startswith("/"):
        return ""
    return " ".join(tokens)


def resolve_inline_fast_command(command_text: str, canonicals: set[str]) -> str | None:
    candidate = extract_leading_command_text(command_text)
    if not candidate:
        return None
    command_token = candidate.split(None, 1)[0].lstrip("/")
    if not command_token:
        return None
    command_token = command_token.split("@", 1)[0].strip().lower()
    if not command_token:
        return None
    return command_token if command_token in canonicals else None


def extract_text_content_from_raw_content(raw_content: str) -> str:
    raw = str(raw_content or "")
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw
    if isinstance(parsed, dict):
        text_value = parsed.get("text", "")
        return str(text_value or "")
    return raw


def extract_trace_token(payload: dict[str, Any], token_pattern: Pattern[str]) -> str:
    event = payload.get("event") or {}
    message = event.get("message") or {}
    raw_content = str(message.get("content") or "")
    if not raw_content:
        return ""
    try:
        text_content = extract_text_content_from_raw_content(raw_content)
    except Exception:
        text_content = raw_content
    match = token_pattern.search(str(text_content or ""))
    return str(match.group(1) or "").strip() if match else ""


def extract_inline_fast_command(payload: dict[str, Any], canonicals: set[str]) -> str | None:
    event = payload.get("event") or {}
    message = event.get("message") or {}
    message_type = str(message.get("message_type") or "").strip().lower()
    if message_type != "text":
        return None
    raw_content = str(message.get("content") or "")
    if not raw_content:
        return None
    try:
        text_content = extract_text_content_from_raw_content(raw_content)
    except Exception:
        text_content = raw_content
    return resolve_inline_fast_command(text_content, canonicals)


def classify_chat_lane(payload: dict[str, Any]) -> str:
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return "chat_light"
    message = event.get("message") or {}
    if not isinstance(message, dict):
        return "chat_light"
    message_type = str(message.get("message_type") or "").strip().lower()
    if message_type and message_type != "text":
        return "chat_heavy"
    raw_content = str(message.get("content") or "")
    text_length = len(raw_content)
    if text_length >= 4000:
        return "chat_heavy"
    if any(key in raw_content for key in ("image_key", "file_key", "audio_key", "media_key")):
        return "chat_heavy"
    mentions = event.get("mentions") or message.get("mentions") or []
    if isinstance(mentions, list) and len(mentions) >= 5:
        return "chat_heavy"
    return "chat_light"


def extract_telegram_inline_fast_command(update: dict[str, Any], canonicals: set[str]) -> str | None:
    message = (
        update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
        or {}
    )
    text = str(message.get("text") or "").strip()
    if not text:
        return None
    return resolve_inline_fast_command(text, canonicals)


def should_inline_control_event(event_type: str) -> bool:
    normalized = str(event_type or "").strip().lower()
    return normalized in {"application.bot.menu_v6", "card.action.trigger"}


def is_session_warmup_event(event_type: str) -> bool:
    normalized = str(event_type or "").strip().lower()
    return "bot_p2p_chat_entered" in normalized
