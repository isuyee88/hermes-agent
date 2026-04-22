from __future__ import annotations

from typing import Any, Callable, Mapping


def build_internal_source(
    payload: Mapping[str, Any],
    *,
    collect_chat_id: Callable[[dict[str, Any]], str],
    collect_actor_id: Callable[[dict[str, Any]], str],
    first_non_empty_str: Callable[..., str],
) -> Any:
    from gateway.config import Platform
    from gateway.session import SessionSource

    raw_message = payload.get("raw_message") if isinstance(payload.get("raw_message"), Mapping) else {}
    raw_event = (
        raw_message.get("event")
        if isinstance(raw_message, Mapping) and isinstance(raw_message.get("event"), Mapping)
        else {}
    )

    chat_id = str(payload.get("chat_id") or "").strip()
    if not chat_id and isinstance(raw_event, dict):
        chat_id = collect_chat_id(raw_event)

    user_id = str(payload.get("user_id") or "").strip()
    if not user_id and isinstance(raw_event, dict):
        user_id = collect_actor_id(raw_event)

    chat_type = str(payload.get("chat_type") or "dm").strip().lower() or "dm"
    raw_user_name = ""
    if isinstance(raw_event, dict):
        sender = raw_event.get("sender") if isinstance(raw_event.get("sender"), dict) else {}
        operator = raw_event.get("operator") if isinstance(raw_event.get("operator"), dict) else {}
        user = raw_event.get("user") if isinstance(raw_event.get("user"), dict) else {}
        raw_user_name = first_non_empty_str(
            payload.get("user_name"),
            sender.get("name"),
            operator.get("name"),
            user.get("name"),
            user_id,
            "feishu-user",
        )

    user_name = str(raw_user_name or payload.get("user_name") or user_id or "feishu-user").strip()
    chat_name = str(payload.get("chat_name") or chat_id or "Feishu Chat").strip()
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_name=chat_name,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
        thread_id=str(payload.get("thread_id") or "").strip() or None,
        user_id_alt=str(payload.get("user_id_alt") or "").strip() or None,
        chat_id_alt=str(payload.get("chat_id_alt") or "").strip() or None,
    )


def normalize_internal_message_type(raw_value: Any) -> Any:
    from gateway.platforms.base import MessageType

    normalized = str(raw_value or "text").strip().lower()
    for candidate in MessageType:
        if candidate.value == normalized:
            return candidate
    return MessageType.TEXT


def build_internal_event(
    payload: Mapping[str, Any],
    *,
    build_source: Callable[[Mapping[str, Any]], Any],
    normalize_message_type: Callable[[Any], Any],
) -> Any:
    from gateway.platforms.base import MessageEvent

    return MessageEvent(
        text=str(payload.get("text") or "").strip(),
        message_type=normalize_message_type(payload.get("message_type")),
        source=build_source(payload),
        raw_message=dict(payload.get("raw_message") or {}),
        message_id=str(payload.get("message_id") or "").strip() or None,
        media_urls=[str(item) for item in (payload.get("media_urls") or []) if str(item or "").strip()],
        media_types=[str(item) for item in (payload.get("media_types") or []) if str(item or "").strip()],
        reply_to_message_id=str(payload.get("reply_to_message_id") or "").strip() or None,
        reply_to_text=str(payload.get("reply_to_text") or "").strip() or None,
        internal=bool(payload.get("internal", False)),
    )
