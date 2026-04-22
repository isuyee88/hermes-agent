from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping


def normalize_pending_reconciles(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_items = payload.get("pending_reconcile")
    if not isinstance(raw_items, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue

        correlation_id = str(item.get("correlation_id") or "").strip()
        if not correlation_id or correlation_id in seen_ids:
            continue

        operation_kinds = [
            str(kind).strip()
            for kind in (item.get("operation_kinds") or [])
            if str(kind or "").strip()
        ]
        normalized.append(
            {
                "correlation_id": correlation_id,
                "assistant_text": str(item.get("assistant_text") or "").strip(),
                "final_response": str(item.get("final_response") or "").strip(),
                "delivered_at": str(item.get("delivered_at") or "").strip(),
                "operation_kinds": operation_kinds,
            }
        )
        seen_ids.add(correlation_id)

    return normalized


def build_pending_reconcile_text(item: Mapping[str, Any]) -> str:
    assistant_text = str(item.get("assistant_text") or "").strip()
    if assistant_text:
        return assistant_text

    final_response = str(item.get("final_response") or "").strip()
    if final_response:
        return final_response

    placeholders = {
        "image": "[Sent image]",
        "image_url": "[Sent image]",
        "image_file": "[Sent image]",
        "file": "[Sent file]",
        "document_file": "[Sent file]",
        "audio": "[Sent audio]",
        "audio_file": "[Sent audio]",
        "media": "[Sent media]",
        "video_file": "[Sent media]",
        "interactive": "[Sent card]",
    }
    parts: list[str] = []
    seen_parts: set[str] = set()
    for kind in item.get("operation_kinds") or []:
        placeholder = placeholders.get(str(kind).strip().lower())
        if placeholder and placeholder not in seen_parts:
            seen_parts.add(placeholder)
            parts.append(placeholder)
    return "\n".join(parts).strip()


def apply_pending_reconciles(
    *,
    runner: Any,
    source: Any,
    payload: Mapping[str, Any],
    normalize_pending_items: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    build_pending_text: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    session_store = getattr(runner, "session_store", None)
    pending_items = normalize_pending_items(payload)
    if session_store is None or not pending_items:
        return {
            "received": len(pending_items),
            "applied": 0,
            "skipped": 0,
            "session_key": str(payload.get("session_key") or "").strip(),
        }

    session_entry = session_store.get_or_create_session(source)
    history = session_store.load_transcript(session_entry.session_id)
    known_ids: set[str] = set()
    for message in history:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "session_meta":
            continue
        if str(message.get("kind") or "").strip().lower() != "pending_reconcile":
            continue

        correlation_id = str(message.get("correlation_id") or message.get("reconcile_id") or "").strip()
        if correlation_id:
            known_ids.add(correlation_id)

    timestamp = datetime.now().isoformat()
    applied = 0
    skipped = 0
    for item in pending_items:
        correlation_id = str(item.get("correlation_id") or "").strip()
        if not correlation_id or correlation_id in known_ids:
            skipped += 1
            continue

        session_store.append_to_transcript(
            session_entry.session_id,
            {
                "role": "session_meta",
                "kind": "pending_reconcile",
                "correlation_id": correlation_id,
                "delivered_at": str(item.get("delivered_at") or "").strip(),
                "operation_kinds": list(item.get("operation_kinds") or []),
                "timestamp": timestamp,
            },
        )
        content = build_pending_text(item)
        if content:
            session_store.append_to_transcript(
                session_entry.session_id,
                {
                    "role": "assistant",
                    "content": content,
                    "correlation_id": correlation_id,
                    "timestamp": timestamp,
                },
            )

        known_ids.add(correlation_id)
        applied += 1

    session_store.update_session(session_entry.session_key)
    return {
        "received": len(pending_items),
        "applied": applied,
        "skipped": skipped,
        "session_key": session_entry.session_key,
    }
