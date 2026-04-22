from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Callable


def log_feishu_internal_auth_failure(
    *,
    endpoint: str,
    authorization: str | None,
    expected_token: str | None,
    headers: Mapping[str, Any] | None,
    extract_feishu_internal_request_meta: Callable[[Mapping[str, Any] | None], dict[str, str]],
    secret_fingerprint: Callable[[str | None], str],
    logger: Any,
) -> None:
    request_meta = extract_feishu_internal_request_meta(headers)
    provided = ""
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        provided = authorization[len("Bearer ") :].strip()
    logger.warning(
        "feishu.internal.auth.failed endpoint=%s gateway_hop=%s gateway_script=%s correlation_id=%s event_id=%s session_key=%s provided_fp=%s expected_fp=%s has_expected=%s has_authorization=%s",
        endpoint,
        request_meta.get("gateway_hop") or "",
        request_meta.get("gateway_script") or "",
        request_meta.get("gateway_correlation_id") or "",
        request_meta.get("gateway_event_id") or "",
        request_meta.get("gateway_session_key") or "",
        secret_fingerprint(provided),
        secret_fingerprint(expected_token),
        bool(str(expected_token or "").strip()),
        bool(str(authorization or "").strip()),
    )


def build_feishu_webhook_ack_response(
    payload: dict[str, Any],
    body: dict[str, Any],
    *,
    request_started_at: float,
    ack_kind: str,
    partition: str | None = None,
    lane: str | None = None,
    queue_depth: int | None = None,
    reason: str | None = None,
    ingress_strategy: str | None = None,
    phase_timings: dict[str, Any] | None = None,
    normalize_phase_timings: Callable[[dict[str, Any] | None], dict[str, int]],
    extract_feishu_trace_token: Callable[[dict[str, Any]], str | None],
    append_feishu_trace: Callable[..., None],
    extract_feishu_event_metadata: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    response_cls: Any,
    tracked_response_cls: Any,
    logger: Any,
) -> Any:
    ack_elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
    normalized_phase_timings = normalize_phase_timings(phase_timings)
    trace_token = extract_feishu_trace_token(payload)
    append_feishu_trace(
        "webhook.ack",
        payload,
        ack_kind=ack_kind,
        ack_elapsed_ms=ack_elapsed_ms,
        partition=partition or "",
        lane=lane or "",
        queue_depth=queue_depth,
        reason=reason or "",
        ingress_strategy=ingress_strategy or "",
        phase_timings=normalized_phase_timings,
    )
    event_id, event_type = extract_feishu_event_metadata(payload)
    logger.warning(
        "[Feishu] webhook ack event_type=%s event_id=%s trace_token=%s ack_kind=%s ack_elapsed_ms=%s partition=%s lane=%s queue_depth=%s reason=%s ingress_strategy=%s phase_timings=%s",
        event_type or "unknown",
        event_id or "none",
        trace_token or "none",
        ack_kind,
        ack_elapsed_ms,
        partition or "none",
        lane or "none",
        queue_depth,
        reason or "",
        ingress_strategy or "",
        json.dumps(normalized_phase_timings, ensure_ascii=False, sort_keys=True),
    )
    active_response_cls = tracked_response_cls or response_cls
    return active_response_cls(
        body,
        payload=payload,
        request_started_at=request_started_at,
        ack_kind=ack_kind,
        partition=partition,
        lane=lane,
        queue_depth=queue_depth,
        reason=reason,
        ingress_strategy=ingress_strategy,
        phase_timings=normalized_phase_timings,
    )


def extract_feishu_message_read_event_info(payload: Mapping[str, Any]) -> dict[str, Any]:
    header = payload.get("header") if isinstance(payload, Mapping) else {}
    event = payload.get("event") if isinstance(payload, Mapping) else {}
    header = header if isinstance(header, Mapping) else {}
    event = event if isinstance(event, Mapping) else {}
    reader = event.get("reader") if isinstance(event.get("reader"), Mapping) else {}
    reader = reader if isinstance(reader, Mapping) else {}
    reader_id = event.get("reader_id") if isinstance(event.get("reader_id"), Mapping) else {}
    reader_id = reader_id if isinstance(reader_id, Mapping) else {}

    message_ids: list[str] = []
    raw_message_ids = event.get("message_id_list")
    if isinstance(raw_message_ids, list):
        for item in raw_message_ids:
            value = str(item or "").strip()
            if value:
                message_ids.append(value)

    def _first_non_empty(*values: Any) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    return {
        "event_type": str(header.get("event_type") or payload.get("event_type") or "").strip(),
        "event_id": str(header.get("event_id") or payload.get("event_id") or "").strip(),
        "reader_open_id": _first_non_empty(
            reader_id.get("open_id"),
            reader.get("open_id"),
            event.get("open_id"),
        ),
        "reader_user_id": _first_non_empty(
            reader_id.get("user_id"),
            reader.get("user_id"),
            event.get("user_id") if isinstance(event.get("user_id"), str) else "",
        ),
        "reader_union_id": _first_non_empty(
            reader_id.get("union_id"),
            reader.get("union_id"),
        ),
        "tenant_key": _first_non_empty(
            reader.get("tenant_key"),
            event.get("tenant_key"),
        ),
        "read_time": int(reader.get("read_time") or event.get("read_time") or 0),
        "message_id_list": message_ids,
        "message_count": len(message_ids),
    }
