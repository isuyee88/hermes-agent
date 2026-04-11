"""Feishu tool handlers and registry wiring."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable

from tools.feishu_api import (
    FeishuOpenApiClient,
    build_download_target_path,
    build_feishu_client,
    build_model_registry,
    build_plain_post_payload,
    coerce_local_file_path,
    ensure_model_registry_bitable_schema,
    extract_document_id,
    extract_filename_from_headers,
    extract_spreadsheet_token,
    get_feishu_base_url,
    get_model_registry_path,
    make_capability_check,
    markdown_to_doc_blocks,
    mirror_model_registry_to_bitable,
    normalize_document_summary,
    normalize_user_profile,
    quote_range,
    resolve_bitable_target,
    resolve_user_identifier,
    validate_message_resource_type,
)
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


def _schema(name: str, description: str, properties: Dict[str, Any], required: Iterable[str] | None = None) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(required or []),
        },
    }


FEISHU_DOC_CREATE_SCHEMA = _schema(
    "feishu_doc_create",
    "Create a Feishu/Lark Docx document and return the document token + URL.",
    {
        "title": {"type": "string", "description": "Document title."},
        "folder_token": {"type": "string", "description": "Optional destination folder token."},
    },
    required=["title"],
)

FEISHU_DOC_GET_SCHEMA = _schema(
    "feishu_doc_get",
    "Read Feishu/Lark Docx metadata and optional raw text content using a token or full URL.",
    {
        "document_id_or_url": {"type": "string", "description": "Doc token like doxcn... or a full doc URL."},
        "include_raw_content": {"type": "boolean", "description": "Fetch raw text content too. Defaults to true.", "default": True},
    },
    required=["document_id_or_url"],
)

FEISHU_DOC_APPEND_MARKDOWN_SCHEMA = _schema(
    "feishu_doc_append_markdown",
    "Append markdown/plain text content to an existing Feishu Docx document.",
    {
        "document_id_or_url": {"type": "string", "description": "Doc token like doxcn... or a full doc URL."},
        "markdown": {"type": "string", "description": "Markdown or plain text to append."},
    },
    required=["document_id_or_url", "markdown"],
)

FEISHU_LOOKUP_USER_SCHEMA = _schema(
    "feishu_lookup_user",
    "Resolve a Feishu/Lark user by user_id, open_id, union_id, email, or mobile.",
    {
        "user_id": {"type": "string", "description": "Feishu user_id."},
        "open_id": {"type": "string", "description": "Feishu open_id."},
        "union_id": {"type": "string", "description": "Feishu union_id."},
        "email": {"type": "string", "description": "User email address."},
        "mobile": {"type": "string", "description": "User mobile number."},
    },
)

FEISHU_SHEET_CREATE_SCHEMA = _schema(
    "feishu_sheet_create",
    "Create a Feishu/Lark spreadsheet.",
    {
        "title": {"type": "string", "description": "Spreadsheet title."},
        "folder_token": {"type": "string", "description": "Optional destination folder token."},
    },
    required=["title"],
)

FEISHU_SHEET_READ_RANGE_SCHEMA = _schema(
    "feishu_sheet_read_range",
    "Read a value range from a Feishu spreadsheet.",
    {
        "spreadsheet_token_or_url": {"type": "string", "description": "Spreadsheet token or full sheets URL."},
        "range": {"type": "string", "description": "A1-style range, e.g. Sheet1!A1:C10."},
    },
    required=["spreadsheet_token_or_url", "range"],
)

FEISHU_SHEET_WRITE_RANGE_SCHEMA = _schema(
    "feishu_sheet_write_range",
    "Write a 2D matrix into a Feishu spreadsheet range.",
    {
        "spreadsheet_token_or_url": {"type": "string", "description": "Spreadsheet token or full sheets URL."},
        "range": {"type": "string", "description": "A1-style range, e.g. Sheet1!A1:C10."},
        "values": {"type": "array", "description": "2D array of rows/cells.", "items": {"type": "array", "items": {}}},
    },
    required=["spreadsheet_token_or_url", "range", "values"],
)

FEISHU_BITABLE_GET_SCHEMA_SCHEMA = _schema(
    "feishu_bitable_get_schema",
    "Inspect Feishu Bitable table metadata and field definitions.",
    {
        "app_token": {"type": "string", "description": "Bitable app token, or a wiki/base URL that resolves into one. Falls back to FEISHU_BITABLE_APP_TOKEN."},
        "table_id": {"type": "string", "description": "Bitable table id. Falls back to FEISHU_BITABLE_TABLE_ID."},
    },
)

FEISHU_BITABLE_LIST_RECORDS_SCHEMA = _schema(
    "feishu_bitable_list_records",
    "List Bitable records with optional view/filter/pagination.",
    {
        "app_token": {"type": "string", "description": "Bitable app token, or a wiki/base URL that resolves into one. Falls back to FEISHU_BITABLE_APP_TOKEN."},
        "table_id": {"type": "string", "description": "Bitable table id. Falls back to FEISHU_BITABLE_TABLE_ID."},
        "view_id": {"type": "string", "description": "Optional view id."},
        "field_names": {"type": "array", "items": {"type": "string"}, "description": "Optional list of field names to keep."},
        "page_size": {"type": "integer", "description": "Page size, default 50.", "default": 50},
        "page_token": {"type": "string", "description": "Pagination token from a previous call."},
    },
)

FEISHU_BITABLE_UPSERT_RECORDS_SCHEMA = _schema(
    "feishu_bitable_upsert_records",
    "Create or update Bitable records. If record_id is present it updates, otherwise it creates.",
    {
        "app_token": {"type": "string", "description": "Bitable app token, or a wiki/base URL that resolves into one. Falls back to FEISHU_BITABLE_APP_TOKEN."},
        "table_id": {"type": "string", "description": "Bitable table id. Falls back to FEISHU_BITABLE_TABLE_ID."},
        "records": {"type": "array", "description": "List of record payloads: {'record_id'?: str, 'fields': {...}}", "items": {"type": "object"}},
    },
    required=["records"],
)

FEISHU_MESSAGE_SEND_SCHEMA = _schema(
    "feishu_message_send",
    "Send a Feishu message to a chat. Supports text, post, and interactive card payloads.",
    {
        "chat_id": {"type": "string", "description": "Target chat open_chat_id / chat_id."},
        "message": {"type": "string", "description": "Message text or pre-built JSON content for interactive cards."},
        "msg_type": {"type": "string", "description": "text, post, or interactive. Defaults to text.", "default": "text"},
        "title": {"type": "string", "description": "Optional title when msg_type=post."},
    },
    required=["chat_id", "message"],
)

FEISHU_CHAT_LOOKUP_SCHEMA = _schema(
    "feishu_chat_lookup",
    "Read Feishu chat metadata for a chat_id/open_chat_id.",
    {"chat_id": {"type": "string", "description": "Target chat open_chat_id / chat_id."}},
    required=["chat_id"],
)

FEISHU_FILE_UPLOAD_SCHEMA = _schema(
    "feishu_file_upload",
    "Upload a local file into Feishu IM media storage and return the resulting file_key.",
    {
        "file_path": {"type": "string", "description": "Absolute or workspace-relative local file path."},
        "file_name": {"type": "string", "description": "Optional file name override."},
    },
    required=["file_path"],
)

FEISHU_FILE_SEND_SCHEMA = _schema(
    "feishu_file_send",
    "Upload a local file and send it into a Feishu chat as a native attachment.",
    {
        "chat_id": {"type": "string", "description": "Target chat open_chat_id / chat_id."},
        "file_path": {"type": "string", "description": "Absolute or workspace-relative local file path."},
        "file_name": {"type": "string", "description": "Optional file name override."},
        "caption": {"type": "string", "description": "Optional caption sent alongside the file."},
    },
    required=["chat_id", "file_path"],
)

FEISHU_FILE_DOWNLOAD_SCHEMA = _schema(
    "feishu_file_download",
    "Download a Feishu message resource into the local Modal volume for later AI processing.",
    {
        "message_id": {"type": "string", "description": "Feishu message id."},
        "file_key": {"type": "string", "description": "Feishu file_key or image_key."},
        "resource_type": {"type": "string", "description": "One of file, image, audio, media. Defaults to file.", "default": "file"},
        "file_name": {"type": "string", "description": "Optional desired local file name."},
    },
    required=["message_id", "file_key"],
)

FEISHU_MODEL_REGISTRY_SYNC_SCHEMA = _schema(
    "feishu_model_registry_sync",
    "Build or refresh Hermes' Feishu model registry and optionally mirror it into Bitable.",
    {
        "force_refresh": {"type": "boolean", "description": "Force rebuilding local registry from current routing state.", "default": False},
        "mirror_to_bitable": {"type": "boolean", "description": "Mirror the registry into Bitable after rebuilding.", "default": False},
        "app_token": {"type": "string", "description": "Optional Bitable app token override, or a wiki/base URL that resolves into one."},
        "table_id": {"type": "string", "description": "Optional Bitable table id override."},
        "wiki_token": {"type": "string", "description": "Optional wiki token when the Bitable is mounted under wiki."},
        "bitable_url": {"type": "string", "description": "Optional wiki/base URL. Hermes extracts app_token and table_id when possible."},
    },
)

FEISHU_MODEL_REGISTRY_PREPARE_BITABLE_SCHEMA = _schema(
    "feishu_model_registry_prepare_bitable",
    "Create or validate the recommended Feishu Bitable table, fields, and views for Hermes model registry mirroring.",
    {
        "app_token": {"type": "string", "description": "Bitable app token, or a wiki/base URL that resolves into one. Falls back to FEISHU_BITABLE_APP_TOKEN."},
        "table_id": {"type": "string", "description": "Optional target table id. If omitted, Hermes finds or creates the table by name."},
        "wiki_token": {"type": "string", "description": "Optional wiki token when the Bitable is mounted under wiki."},
        "bitable_url": {"type": "string", "description": "Optional wiki/base URL. Hermes extracts app_token and table_id when possible."},
        "table_name": {"type": "string", "description": "Table name to find or create.", "default": "Hermes Model Registry"},
        "create_missing_table": {"type": "boolean", "description": "Create the table if it does not exist.", "default": True},
        "create_missing_fields": {"type": "boolean", "description": "Create missing recommended fields.", "default": True},
        "create_missing_views": {"type": "boolean", "description": "Create missing recommended views.", "default": True},
    },
)

FEISHU_MODEL_REGISTRY_PUBLISH_CARD_SCHEMA = _schema(
    "feishu_model_registry_publish_card",
    "Publish the current model registry summary into a Feishu chat as an interactive card.",
    {
        "chat_id": {"type": "string", "description": "Target chat open_chat_id / chat_id."},
        "top_n": {"type": "integer", "description": "How many models per provider to show.", "default": 6},
        "include_unavailable": {"type": "boolean", "description": "Include models marked unavailable.", "default": False},
    },
    required=["chat_id"],
)


def _client() -> FeishuOpenApiClient:
    return build_feishu_client()


def feishu_doc_create_tool(args: Dict[str, Any], **_kw: Any) -> str:
    title = str(args.get("title", "") or "").strip()
    if not title:
        return tool_error("title is required")
    payload: Dict[str, Any] = {"title": title}
    folder_token = str(args.get("folder_token", "") or "").strip()
    if folder_token:
        payload["folder_token"] = folder_token
    try:
        data = _client().request_json("POST", "/open-apis/docx/v1/documents", json_body=payload)
        document = data.get("document") if isinstance(data.get("document"), dict) else data
        document_id = (
            document.get("document_id")
            or document.get("token")
            or document.get("obj_token")
            or data.get("document_id")
            or data.get("token")
        )
        if not document_id:
            raise RuntimeError("Feishu create document response did not include a document ID")
        return tool_result(
            success=True,
            document_id=document_id,
            title=document.get("title") or title,
            url=document.get("url") or f"{get_feishu_base_url()}/docx/{document_id}",
        )
    except Exception as exc:
        logger.warning("feishu_doc_create failed: %s", exc)
        return tool_error(f"Failed to create Feishu document: {exc}")


def feishu_doc_get_tool(args: Dict[str, Any], **_kw: Any) -> str:
    try:
        document_id = extract_document_id(args.get("document_id_or_url", ""))
    except Exception as exc:
        return tool_error(str(exc))

    include_raw_content = args.get("include_raw_content", True)
    try:
        client = _client()
        info = client.request_json("GET", f"/open-apis/docx/v1/documents/{document_id}")
        raw_content = None
        if include_raw_content:
            raw = client.request_json("GET", f"/open-apis/docx/v1/documents/{document_id}/raw_content")
            raw_content = str(raw.get("content") or raw.get("raw_content") or "").strip()
        return tool_result(normalize_document_summary(document_id, info, raw_content))
    except Exception as exc:
        logger.warning("feishu_doc_get failed: %s", exc)
        return tool_error(f"Failed to fetch Feishu document: {exc}")


def feishu_doc_append_markdown_tool(args: Dict[str, Any], **_kw: Any) -> str:
    markdown = str(args.get("markdown", "") or "")
    if not markdown.strip():
        return tool_error("markdown is required")
    try:
        document_id = extract_document_id(args.get("document_id_or_url", ""))
        data = _client().request_json(
            "POST",
            f"/open-apis/docx/v1/documents/{document_id}/blocks/{document_id}/children",
            json_body={"children": markdown_to_doc_blocks(markdown), "index": -1},
        )
        return tool_result(
            success=True,
            document_id=document_id,
            inserted_blocks=len(markdown_to_doc_blocks(markdown)),
            document_revision_id=data.get("document_revision_id"),
        )
    except Exception as exc:
        logger.warning("feishu_doc_append_markdown failed: %s", exc)
        return tool_error(f"Failed to append markdown to Feishu document: {exc}")


def feishu_lookup_user_tool(args: Dict[str, Any], **_kw: Any) -> str:
    try:
        client = _client()
        identifier, id_type = resolve_user_identifier(client, args)
        data = client.request_json(
            "GET",
            f"/open-apis/contact/v3/users/{identifier}",
            params={"user_id_type": id_type},
        )
        user = data.get("user") if isinstance(data.get("user"), dict) else data
        if not isinstance(user, dict) or not user:
            raise RuntimeError("Feishu user lookup returned no user payload")
        return tool_result(success=True, user=normalize_user_profile(user, resolved_via=id_type))
    except Exception as exc:
        logger.warning("feishu_lookup_user failed: %s", exc)
        return tool_error(f"Failed to lookup Feishu user: {exc}")


def feishu_sheet_create_tool(args: Dict[str, Any], **_kw: Any) -> str:
    title = str(args.get("title", "") or "").strip()
    if not title:
        return tool_error("title is required")
    payload: Dict[str, Any] = {"title": title}
    folder_token = str(args.get("folder_token", "") or "").strip()
    if folder_token:
        payload["folder_token"] = folder_token
    try:
        data = _client().request_json("POST", "/open-apis/sheets/v3/spreadsheets", json_body=payload)
        spreadsheet = data.get("spreadsheet") if isinstance(data.get("spreadsheet"), dict) else data
        token = (
            spreadsheet.get("spreadsheet_token")
            or spreadsheet.get("spreadsheetToken")
            or spreadsheet.get("token")
            or data.get("spreadsheet_token")
        )
        url = spreadsheet.get("url") or (f"{get_feishu_base_url()}/sheets/{token}" if token else "")
        return tool_result(
            success=True,
            spreadsheet_token=token,
            title=spreadsheet.get("title") or title,
            url=url,
        )
    except Exception as exc:
        logger.warning("feishu_sheet_create failed: %s", exc)
        return tool_error(f"Failed to create Feishu sheet: {exc}")


def feishu_sheet_read_range_tool(args: Dict[str, Any], **_kw: Any) -> str:
    range_name = str(args.get("range", "") or "").strip()
    if not range_name:
        return tool_error("range is required")
    try:
        token = extract_spreadsheet_token(args.get("spreadsheet_token_or_url", ""))
        data = _client().request_json(
            "GET",
            f"/open-apis/sheets/v2/spreadsheets/{token}/values/{quote_range(range_name)}",
        )
        values = data.get("valueRange", {}).get("values")
        if values is None:
            values = data.get("data", {}).get("valueRange", {}).get("values")
        return tool_result(
            success=True,
            spreadsheet_token=token,
            range=range_name,
            values=values or [],
        )
    except Exception as exc:
        logger.warning("feishu_sheet_read_range failed: %s", exc)
        return tool_error(f"Failed to read Feishu sheet range: {exc}")


def feishu_sheet_write_range_tool(args: Dict[str, Any], **_kw: Any) -> str:
    range_name = str(args.get("range", "") or "").strip()
    values = args.get("values")
    if not range_name:
        return tool_error("range is required")
    if not isinstance(values, list):
        return tool_error("values must be a 2D array")
    try:
        token = extract_spreadsheet_token(args.get("spreadsheet_token_or_url", ""))
        data = _client().request_json(
            "PUT",
            f"/open-apis/sheets/v2/spreadsheets/{token}/values",
            json_body={"valueRange": {"range": range_name, "values": values}},
        )
        return tool_result(
            success=True,
            spreadsheet_token=token,
            updated_range=data.get("updatedRange") or range_name,
            updated_rows=data.get("updatedRows"),
        )
    except Exception as exc:
        logger.warning("feishu_sheet_write_range failed: %s", exc)
        return tool_error(f"Failed to write Feishu sheet range: {exc}")


def feishu_bitable_get_schema_tool(args: Dict[str, Any], **_kw: Any) -> str:
    try:
        app_token, table_id = resolve_bitable_target(args)
        client = _client()
        table_payload = client.request_json("GET", f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}")
        fields_payload = client.request_json(
            "GET",
            f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields",
            params={"page_size": 200},
        )
        return tool_result(
            success=True,
            app_token=app_token,
            table_id=table_id,
            table=table_payload.get("table") or table_payload,
            fields=fields_payload.get("items") or [],
        )
    except Exception as exc:
        logger.warning("feishu_bitable_get_schema failed: %s", exc)
        return tool_error(f"Failed to inspect Feishu Bitable schema: {exc}")


def feishu_bitable_list_records_tool(args: Dict[str, Any], **_kw: Any) -> str:
    try:
        app_token, table_id = resolve_bitable_target(args)
        params: Dict[str, Any] = {"page_size": int(args.get("page_size", 50) or 50)}
        view_id = str(args.get("view_id", "") or "").strip()
        if view_id:
            params["view_id"] = view_id
        page_token = str(args.get("page_token", "") or "").strip()
        if page_token:
            params["page_token"] = page_token
        data = _client().request_json(
            "GET",
            f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records",
            params=params,
        )
        items = data.get("items") or []
        requested_fields = [str(item).strip() for item in (args.get("field_names") or []) if str(item).strip()]
        if requested_fields:
            trimmed_items = []
            for item in items:
                fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
                trimmed_items.append({**item, "fields": {name: fields.get(name) for name in requested_fields if name in fields}})
            items = trimmed_items
        return tool_result(
            success=True,
            app_token=app_token,
            table_id=table_id,
            items=items,
            page_token=data.get("page_token"),
            has_more=bool(data.get("has_more")),
            total=data.get("total"),
        )
    except Exception as exc:
        logger.warning("feishu_bitable_list_records failed: %s", exc)
        return tool_error(f"Failed to list Feishu Bitable records: {exc}")


def feishu_bitable_upsert_records_tool(args: Dict[str, Any], **_kw: Any) -> str:
    records = args.get("records")
    if not isinstance(records, list) or not records:
        return tool_error("records must be a non-empty list")
    try:
        app_token, table_id = resolve_bitable_target(args)
        client = _client()
        created = 0
        updated = 0
        result_records: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each record must be an object")
            fields = record.get("fields") if isinstance(record.get("fields"), dict) else None
            if not fields:
                raise ValueError("Each record must include a non-empty fields object")
            record_id = str(record.get("record_id", "") or "").strip()
            if record_id:
                payload = client.request_json(
                    "PUT",
                    f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}",
                    json_body={"fields": fields},
                )
                updated += 1
            else:
                payload = client.request_json(
                    "POST",
                    f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records",
                    json_body={"fields": fields},
                )
                created += 1
            result_records.append(payload.get("record") or payload)
        return tool_result(success=True, app_token=app_token, table_id=table_id, created=created, updated=updated, records=result_records)
    except Exception as exc:
        logger.warning("feishu_bitable_upsert_records failed: %s", exc)
        return tool_error(f"Failed to upsert Feishu Bitable records: {exc}")


def feishu_message_send_tool(args: Dict[str, Any], **_kw: Any) -> str:
    chat_id = str(args.get("chat_id", "") or "").strip()
    message = str(args.get("message", "") or "")
    msg_type = str(args.get("msg_type", "text") or "text").strip().lower() or "text"
    if not chat_id:
        return tool_error("chat_id is required")
    if not message:
        return tool_error("message is required")
    if msg_type not in {"text", "post", "interactive"}:
        return tool_error("msg_type must be one of text, post, interactive")
    try:
        if msg_type == "text":
            content = json.dumps({"text": message}, ensure_ascii=False)
        elif msg_type == "post":
            content = build_plain_post_payload(message, title=str(args.get("title", "") or "").strip() or None)
        else:
            json.loads(message)
            content = message
        data = _client().send_message(chat_id=chat_id, msg_type=msg_type, content=content)
        return tool_result(success=True, chat_id=chat_id, msg_type=msg_type, message_id=data.get("message_id"))
    except Exception as exc:
        logger.warning("feishu_message_send failed: %s", exc)
        return tool_error(f"Failed to send Feishu message: {exc}")


def feishu_chat_lookup_tool(args: Dict[str, Any], **_kw: Any) -> str:
    chat_id = str(args.get("chat_id", "") or "").strip()
    if not chat_id:
        return tool_error("chat_id is required")
    try:
        data = _client().request_json("GET", f"/open-apis/im/v1/chats/{chat_id}")
        return tool_result(success=True, chat=data.get("chat") if isinstance(data.get("chat"), dict) else data)
    except Exception as exc:
        logger.warning("feishu_chat_lookup failed: %s", exc)
        return tool_error(f"Failed to lookup Feishu chat: {exc}")


def feishu_file_upload_tool(args: Dict[str, Any], **_kw: Any) -> str:
    try:
        path = coerce_local_file_path(args.get("file_path", ""))
        file_name = str(args.get("file_name", "") or "").strip() or path.name
        data = _client().upload_im_file(file_path=path, file_name=file_name)
        return tool_result(success=True, file_key=data.get("file_key"), file_name=file_name)
    except Exception as exc:
        logger.warning("feishu_file_upload failed: %s", exc)
        return tool_error(f"Failed to upload Feishu file: {exc}")


def feishu_file_send_tool(args: Dict[str, Any], **_kw: Any) -> str:
    chat_id = str(args.get("chat_id", "") or "").strip()
    if not chat_id:
        return tool_error("chat_id is required")
    try:
        path = coerce_local_file_path(args.get("file_path", ""))
        file_name = str(args.get("file_name", "") or "").strip() or path.name
        caption = str(args.get("caption", "") or "").strip() or None
        client = _client()
        upload = client.upload_im_file(file_path=path, file_name=file_name)
        file_key = str(upload.get("file_key") or "").strip()
        if not file_key:
            raise RuntimeError("Feishu file upload did not return a file_key")
        send_result = client.send_uploaded_file_message(chat_id=chat_id, file_key=file_key, caption=caption, file_name=file_name)
        return tool_result(success=True, chat_id=chat_id, file_key=file_key, file_name=file_name, message_id=send_result.get("message_id"))
    except Exception as exc:
        logger.warning("feishu_file_send failed: %s", exc)
        return tool_error(f"Failed to send Feishu file: {exc}")


def feishu_file_download_tool(args: Dict[str, Any], **_kw: Any) -> str:
    message_id = str(args.get("message_id", "") or "").strip()
    file_key = str(args.get("file_key", "") or "").strip()
    if not message_id or not file_key:
        return tool_error("message_id and file_key are required")
    try:
        resource_type = validate_message_resource_type(args.get("resource_type", "file"))
        default_name = str(args.get("file_name", "") or "").strip() or f"{resource_type}_{file_key}"
        content, headers = _client().request_bytes(
            "GET",
            f"/open-apis/im/v1/messages/{message_id}/resources/{file_key}",
            params={"type": resource_type},
        )
        target_path = build_download_target_path(file_name=extract_filename_from_headers(headers, default_name))
        target_path.write_bytes(content)
        return tool_result(
            success=True,
            message_id=message_id,
            file_key=file_key,
            resource_type=resource_type,
            local_path=str(target_path),
            content_type=str(headers.get("content-type", "") or ""),
        )
    except Exception as exc:
        logger.warning("feishu_file_download failed: %s", exc)
        return tool_error(f"Failed to download Feishu file: {exc}")


def feishu_model_registry_sync_tool(args: Dict[str, Any], **_kw: Any) -> str:
    force_refresh = bool(args.get("force_refresh", False))
    mirror_to_bitable = bool(
        args.get("mirror_to_bitable", False)
        or str(os.getenv("FEISHU_MODEL_REGISTRY_MIRROR_ENABLED", "") or "").strip().lower() in {"1", "true", "yes"}
    )
    try:
        registry_payload = build_model_registry(force_refresh=force_refresh)
        client = _client()
        result: Dict[str, Any] = {
            "success": True,
            "registry_path": str(get_model_registry_path()),
            "entry_count": len(registry_payload.get("entries") or []),
            "refreshed_at": registry_payload.get("refreshed_at"),
            "source": registry_payload.get("source"),
        }
        if mirror_to_bitable:
            app_token, table_id = resolve_bitable_target(args, client)
            result["bitable_mirror"] = mirror_model_registry_to_bitable(
                client,
                registry_payload,
                app_token=app_token,
                table_id=table_id,
            )
        return tool_result(result)
    except Exception as exc:
        logger.warning("feishu_model_registry_sync failed: %s", exc)
        return tool_error(f"Failed to sync Feishu model registry: {exc}")


def feishu_model_registry_prepare_bitable_tool(args: Dict[str, Any], **_kw: Any) -> str:
    try:
        client = _client()
        app_token, table_id = resolve_bitable_target(args, client, require_table_id=False)
        result = ensure_model_registry_bitable_schema(
            client,
            app_token=app_token,
            table_id=table_id or None,
            table_name=str(args.get("table_name") or "Hermes Model Registry").strip() or "Hermes Model Registry",
            create_missing_table=bool(args.get("create_missing_table", True)),
            create_missing_fields=bool(args.get("create_missing_fields", True)),
            create_missing_views=bool(args.get("create_missing_views", True)),
        )
        return tool_result(result)
    except Exception as exc:
        logger.warning("feishu_model_registry_prepare_bitable failed: %s", exc)
        return tool_error(f"Failed to prepare Feishu Bitable model registry schema: {exc}")


def feishu_model_registry_publish_card_tool(args: Dict[str, Any], **_kw: Any) -> str:
    chat_id = str(args.get("chat_id", "") or "").strip()
    if not chat_id:
        return tool_error("chat_id is required")
    top_n = max(1, int(args.get("top_n", 6) or 6))
    include_unavailable = bool(args.get("include_unavailable", False))
    try:
        registry_payload = build_model_registry(force_refresh=False)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in registry_payload.get("entries") or []:
            if entry.get("hidden"):
                continue
            if not include_unavailable and not entry.get("is_available", True):
                continue
            grouped.setdefault(str(entry.get("provider") or "unknown"), []).append(entry)

        elements: list[dict[str, Any]] = [
            {
                "tag": "markdown",
                "content": f"Current registry snapshot: `{len(registry_payload.get('entries') or [])}` models\nGenerated at `{registry_payload.get('generated_at')}`",
            }
        ]
        for provider, entries in grouped.items():
            top_entries = entries[:top_n]
            lines = []
            for item in top_entries:
                free_marker = "free" if item.get("is_free") else "paid"
                hint = str(item.get("selection_hint") or "").strip()
                suffix = f" ({hint})" if hint else ""
                lines.append(f"- `{item.get('model')}` [{free_marker}]{suffix}")
            elements.append({"tag": "markdown", "content": f"**{provider}**\n" + "\n".join(lines)})

        card = {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": "Hermes Model Registry"}, "template": "blue"},
            "elements": elements,
        }
        send_result = _client().send_message(chat_id=chat_id, msg_type="interactive", content=json.dumps(card, ensure_ascii=False))
        return tool_result(success=True, chat_id=chat_id, message_id=send_result.get("message_id"), card_providers=sorted(grouped.keys()))
    except Exception as exc:
        logger.warning("feishu_model_registry_publish_card failed: %s", exc)
        return tool_error(f"Failed to publish Feishu model registry card: {exc}")


registry.register(name="feishu_doc_create", toolset="feishu", schema=FEISHU_DOC_CREATE_SCHEMA, handler=feishu_doc_create_tool, check_fn=make_capability_check("docs"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_doc_get", toolset="feishu", schema=FEISHU_DOC_GET_SCHEMA, handler=feishu_doc_get_tool, check_fn=make_capability_check("docs"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F", max_result_size_chars=20_000)
registry.register(name="feishu_doc_append_markdown", toolset="feishu", schema=FEISHU_DOC_APPEND_MARKDOWN_SCHEMA, handler=feishu_doc_append_markdown_tool, check_fn=make_capability_check("docs"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_lookup_user", toolset="feishu", schema=FEISHU_LOOKUP_USER_SCHEMA, handler=feishu_lookup_user_tool, check_fn=make_capability_check("contacts"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_sheet_create", toolset="feishu", schema=FEISHU_SHEET_CREATE_SCHEMA, handler=feishu_sheet_create_tool, check_fn=make_capability_check("sheets"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_sheet_read_range", toolset="feishu", schema=FEISHU_SHEET_READ_RANGE_SCHEMA, handler=feishu_sheet_read_range_tool, check_fn=make_capability_check("sheets"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_sheet_write_range", toolset="feishu", schema=FEISHU_SHEET_WRITE_RANGE_SCHEMA, handler=feishu_sheet_write_range_tool, check_fn=make_capability_check("sheets"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_bitable_get_schema", toolset="feishu", schema=FEISHU_BITABLE_GET_SCHEMA_SCHEMA, handler=feishu_bitable_get_schema_tool, check_fn=make_capability_check("bitable"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_bitable_list_records", toolset="feishu", schema=FEISHU_BITABLE_LIST_RECORDS_SCHEMA, handler=feishu_bitable_list_records_tool, check_fn=make_capability_check("bitable"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_bitable_upsert_records", toolset="feishu", schema=FEISHU_BITABLE_UPSERT_RECORDS_SCHEMA, handler=feishu_bitable_upsert_records_tool, check_fn=make_capability_check("bitable"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_message_send", toolset="feishu", schema=FEISHU_MESSAGE_SEND_SCHEMA, handler=feishu_message_send_tool, check_fn=make_capability_check("messages"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_chat_lookup", toolset="feishu", schema=FEISHU_CHAT_LOOKUP_SCHEMA, handler=feishu_chat_lookup_tool, check_fn=make_capability_check("messages"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_file_upload", toolset="feishu", schema=FEISHU_FILE_UPLOAD_SCHEMA, handler=feishu_file_upload_tool, check_fn=make_capability_check("files"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_file_send", toolset="feishu", schema=FEISHU_FILE_SEND_SCHEMA, handler=feishu_file_send_tool, check_fn=make_capability_check("files"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_file_download", toolset="feishu", schema=FEISHU_FILE_DOWNLOAD_SCHEMA, handler=feishu_file_download_tool, check_fn=make_capability_check("files"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_model_registry_sync", toolset="feishu", schema=FEISHU_MODEL_REGISTRY_SYNC_SCHEMA, handler=feishu_model_registry_sync_tool, check_fn=make_capability_check("model_registry"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_model_registry_prepare_bitable", toolset="feishu", schema=FEISHU_MODEL_REGISTRY_PREPARE_BITABLE_SCHEMA, handler=feishu_model_registry_prepare_bitable_tool, check_fn=make_capability_check("model_registry"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
registry.register(name="feishu_model_registry_publish_card", toolset="feishu", schema=FEISHU_MODEL_REGISTRY_PUBLISH_CARD_SCHEMA, handler=feishu_model_registry_publish_card_tool, check_fn=make_capability_check("model_registry"), requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="F")
