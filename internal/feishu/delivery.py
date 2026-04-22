from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping


def download_internal_attachment(ref: Mapping[str, Any]) -> tuple[str, str] | None:
    from tools.feishu_api import (
        build_download_target_path,
        build_feishu_client,
        extract_filename_from_headers,
        validate_message_resource_type,
    )

    message_id = str(ref.get("message_id") or "").strip()
    file_key = str(ref.get("file_key") or ref.get("image_key") or "").strip()
    if not message_id or not file_key:
        return None

    resource_type = validate_message_resource_type(str(ref.get("resource_type") or "file"))
    default_name = str(ref.get("file_name") or "").strip() or f"{resource_type}_{file_key}"
    content, headers = build_feishu_client().request_bytes(
        "GET",
        f"/open-apis/im/v1/messages/{message_id}/resources/{file_key}",
        params={"type": resource_type},
    )
    target_path = build_download_target_path(
        file_name=extract_filename_from_headers(headers, default_name)
    )
    target_path.write_bytes(content)
    media_type = str(headers.get("content-type", "") or "").strip()
    return str(target_path), media_type


async def hydrate_internal_event_media(payload: Mapping[str, Any], event: Any) -> Any:
    attachment_refs = payload.get("attachment_refs") or []
    if not isinstance(attachment_refs, list) or not attachment_refs:
        return event

    downloaded_media_urls: list[str] = list(getattr(event, "media_urls", []) or [])
    downloaded_media_types: list[str] = list(getattr(event, "media_types", []) or [])
    for raw_ref in attachment_refs:
        if not isinstance(raw_ref, Mapping):
            continue
        downloaded = await asyncio.to_thread(download_internal_attachment, raw_ref)
        if downloaded is None:
            continue
        local_path, media_type = downloaded
        downloaded_media_urls.append(local_path)
        downloaded_media_types.append(media_type)

    if downloaded_media_urls:
        event.media_urls = downloaded_media_urls
        event.media_types = downloaded_media_types
    return event


def serialize_internal_operation(
    operation: Mapping[str, Any],
    *,
    adapter: Any,
    register_result_file: Callable[..., dict[str, Any] | None],
) -> dict[str, Any]:
    kind = str(operation.get("kind") or "").strip()
    payload = {
        "kind": kind,
        "chat_id": str(operation.get("chat_id") or "").strip(),
        "message_id": str(operation.get("message_id") or "").strip(),
        "reply_to": str(operation.get("reply_to") or "").strip(),
        "content": str(operation.get("content") or "").strip(),
        "caption": str(operation.get("caption") or "").strip(),
        "metadata": dict(operation.get("metadata") or {}),
    }
    image_url = str(operation.get("image_url") or "").strip()
    if image_url:
        payload["image_url"] = image_url

    file_path = str(operation.get("file_path") or "").strip()
    if file_path:
        kind_hint = kind
        is_voice = kind_hint == "audio_file"
        if kind_hint not in {"audio_file", "video_file", "image_file", "document_file"}:
            attachment_kind = adapter.classify_local_attachment(file_path)
            kind_hint = {
                "audio": "audio_file",
                "video": "video_file",
                "image": "image_file",
            }.get(attachment_kind, "document_file")
        ticket = register_result_file(
            file_path,
            kind=kind_hint,
            is_voice=is_voice,
        )
        if ticket:
            payload["file"] = ticket
            payload["file_name"] = str(operation.get("file_name") or ticket.get("filename") or "").strip()

    return payload


def build_internal_send_plan(
    *,
    adapter: Any,
    response_text: str | None,
    register_result_file: Callable[..., dict[str, Any] | None],
    serialize_operation: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    operations = [
        serialize_operation(item, adapter=adapter, register_result_file=register_result_file)
        for item in list(getattr(adapter, "captured_operations", []) or [])
        if isinstance(item, dict)
    ]

    response = str(response_text or "").strip()
    if not response:
        return operations

    media_items, cleaned = adapter.extract_media(response)
    image_items, cleaned = adapter.extract_images(cleaned)
    local_files, cleaned = adapter.extract_local_files(cleaned)

    if cleaned.strip():
        operations.append({"kind": "text", "content": cleaned.strip()})

    for image_url, caption in image_items:
        operations.append(
            {
                "kind": "image_url",
                "image_url": image_url,
                "caption": str(caption or "").strip(),
            }
        )

    seen_local_paths: set[str] = set()
    for file_path, is_voice in media_items:
        normalized_path = str(file_path or "").strip()
        if not normalized_path:
            continue
        seen_local_paths.add(normalized_path)
        attachment_kind = adapter.classify_local_attachment(normalized_path)
        kind = {
            "audio": "audio_file",
            "video": "video_file",
            "image": "image_file",
        }.get(attachment_kind, "document_file")
        ticket = register_result_file(
            normalized_path,
            kind=kind,
            is_voice=is_voice,
        )
        if ticket:
            operations.append({"kind": kind, "file": ticket})

    for file_path in local_files:
        normalized_path = str(file_path or "").strip()
        if not normalized_path or normalized_path in seen_local_paths:
            continue
        attachment_kind = adapter.classify_local_attachment(normalized_path)
        kind = {
            "audio": "audio_file",
            "video": "video_file",
            "image": "image_file",
        }.get(attachment_kind, "document_file")
        ticket = register_result_file(normalized_path, kind=kind)
        if ticket:
            operations.append({"kind": kind, "file": ticket})

    return operations
