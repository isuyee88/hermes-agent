from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping


FirstNonEmptyStr = Callable[..., str]
CollectChatId = Callable[[dict[str, Any]], str]
ExtractTraceToken = Callable[[dict[str, Any]], str]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return None


def _coerce_optional_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        if "." in normalized:
            return float(normalized)
        return int(normalized)
    except (TypeError, ValueError):
        return None


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def extract_trace_context(
    payload: dict[str, Any],
    *,
    first_non_empty_str: FirstNonEmptyStr,
    collect_chat_id: CollectChatId,
    extract_trace_token: ExtractTraceToken,
) -> dict[str, Any]:
    header = payload.get("header") or {}
    event = payload.get("event") or {}
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    chat = event.get("chat") if isinstance(event.get("chat"), dict) else {}
    sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
    sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
    operator = event.get("operator") if isinstance(event.get("operator"), dict) else {}
    operator_id = operator.get("operator_id") if isinstance(operator.get("operator_id"), dict) else {}
    user = event.get("user") if isinstance(event.get("user"), dict) else {}
    user_id = event.get("user_id") if isinstance(event.get("user_id"), dict) else {}
    ingress_meta = _mapping(payload.get("_hermes_ingress"))
    gateway_meta = _mapping(payload.get("_hermes_gateway_meta"))
    gateway_request = _mapping(payload.get("_hermes_gateway_request"))
    provider_plan = _mapping(payload.get("provider_plan"))
    provider_usage = _mapping(payload.get("provider_usage"))
    raw_message = _mapping(payload.get("raw_message"))
    raw_message_event = _mapping(raw_message.get("event"))
    raw_message_body = _mapping(raw_message_event.get("message"))
    message_id_list = event.get("message_id_list") if isinstance(event.get("message_id_list"), list) else []
    primary_message_id = _first_non_empty(
        payload.get("message_id"),
        message.get("message_id"),
        raw_message_body.get("message_id"),
        message_id_list[0] if message_id_list else "",
    )
    raw_site_prefetch = payload.get("site_prefetch")
    if not isinstance(raw_site_prefetch, Mapping) and isinstance(ingress_meta, Mapping):
        raw_site_prefetch = ingress_meta.get("site_prefetch")
    site_prefetch = dict(raw_site_prefetch) if isinstance(raw_site_prefetch, Mapping) else {}
    try:
        site_prefetch_confidence = float(site_prefetch.get("confidence") or 0.0)
    except (TypeError, ValueError):
        site_prefetch_confidence = 0.0
    cache_eligible = _coerce_optional_bool(
        _first_non_empty(
            payload.get("cache_eligible"),
            ingress_meta.get("cache_eligible"),
            gateway_meta.get("cache_eligible"),
        )
    )
    if cache_eligible is None:
        cache_mode = str(provider_plan.get("cache_mode") or "").strip().lower()
        if cache_mode:
            cache_eligible = cache_mode != "skip"
    capability_match = _coerce_optional_bool(
        _first_non_empty(
            payload.get("capability_match"),
            ingress_meta.get("capability_match"),
            gateway_meta.get("capability_match"),
        )
    )
    preferred_model_selected = _coerce_optional_bool(
        _first_non_empty(
            payload.get("preferred_model_selected"),
            ingress_meta.get("preferred_model_selected"),
            gateway_meta.get("preferred_model_selected"),
        )
    )
    misroute_detected = _coerce_optional_bool(
        _first_non_empty(
            payload.get("misroute_detected"),
            ingress_meta.get("misroute_detected"),
            gateway_meta.get("misroute_detected"),
        )
    )
    return {
        "type": str(payload.get("type") or "").strip(),
        "event_type": str(header.get("event_type") or payload.get("event_type") or "").strip(),
        "event_id": str(header.get("event_id") or payload.get("event_id") or "").strip(),
        "message_id": primary_message_id,
        "chat_type": str(message.get("chat_type") or chat.get("chat_type") or event.get("chat_type") or "").strip(),
        "chat_id": first_non_empty_str(payload.get("chat_id"), collect_chat_id(event)),
        "correlation_id": str(payload.get("correlation_id") or "").strip(),
        "session_key": str(payload.get("session_key") or "").strip(),
        "route_hint": str(payload.get("route_hint") or "").strip(),
        "route_version": _first_non_empty(
            payload.get("route_version"),
            ingress_meta.get("route_version"),
            gateway_meta.get("route_version"),
            gateway_request.get("route_version"),
        ),
        "task_kind": str(payload.get("task_kind") or "").strip(),
        "request_class": str(payload.get("request_class") or "").strip(),
        "route_family": str(payload.get("route_family") or "").strip(),
        "gateway_route_name": str(payload.get("gateway_route_name") or "").strip(),
        "gateway_eligible": payload.get("gateway_eligible"),
        "requires_tools": payload.get("requires_tools"),
        "requires_browser": payload.get("requires_browser"),
        "requires_media_hydration": payload.get("requires_media_hydration"),
        "requires_modal_runtime": payload.get("requires_modal_runtime"),
        "content_modalities": payload.get("content_modalities"),
        "toolset": payload.get("toolset"),
        "modality_profile": str(payload.get("modality_profile") or "").strip(),
        "reason_code": str(payload.get("reason_code") or "").strip(),
        "site_category": str(payload.get("site_category") or "").strip(),
        "site_intent": str(payload.get("site_intent") or "").strip(),
        "target_domain": str(payload.get("target_domain") or "").strip(),
        "target_url": str(payload.get("target_url") or "").strip(),
        "site_skill_name": str(payload.get("site_skill_name") or ingress_meta.get("site_skill_name") or "").strip(),
        "site_prefetch_mode": str(site_prefetch.get("mode") or "").strip(),
        "site_prefetch_status": str(site_prefetch.get("status") or "").strip(),
        "site_prefetch_confidence": site_prefetch_confidence,
        "site_prefetch_direct_navigation": payload.get("site_prefetch_direct_navigation"),
        "site_prefetch_direct_mode": str(payload.get("site_prefetch_direct_mode") or "").strip(),
        "modal_avoided": payload.get("modal_avoided"),
        "browser_backend_selected": str(payload.get("browser_backend_selected") or "").strip(),
        "browser_escalation_reason": str(payload.get("browser_escalation_reason") or "").strip(),
        "browser_target_domain": str(payload.get("browser_target_domain") or "").strip(),
        "estimated_cost_usd": payload.get("estimated_cost_usd"),
        "actual_cost_usd": payload.get("actual_cost_usd"),
        "provider_alias": _first_non_empty(
            payload.get("provider_alias"),
            provider_usage.get("provider"),
            provider_plan.get("provider"),
            gateway_meta.get("provider_alias"),
        ),
        "fallback_reason": _first_non_empty(
            payload.get("fallback_reason"),
            ingress_meta.get("fallback_reason"),
            gateway_meta.get("fallback_reason"),
        ),
        "gateway_error_class": _first_non_empty(
            payload.get("gateway_error_class"),
            ingress_meta.get("gateway_error_class"),
            gateway_meta.get("gateway_error_class"),
        ),
        "misroute_detected": misroute_detected,
        "cache_eligible": cache_eligible,
        "cache_status": _first_non_empty(
            payload.get("cache_status"),
            provider_usage.get("cache_status"),
            ingress_meta.get("cache_status"),
            gateway_meta.get("cache_status"),
        ),
        "ai_call_count": _coerce_optional_number(
            payload.get("ai_call_count")
            if payload.get("ai_call_count") is not None
            else gateway_meta.get("ai_call_count")
        ),
        "capability_match": capability_match,
        "preferred_model_selected": preferred_model_selected,
        "model_catalog_version": _first_non_empty(
            payload.get("model_catalog_version"),
            ingress_meta.get("model_catalog_version"),
            gateway_meta.get("model_catalog_version"),
        ),
        "feedback_score_before": _coerce_optional_number(
            payload.get("feedback_score_before")
            if payload.get("feedback_score_before") is not None
            else gateway_meta.get("feedback_score_before")
        ),
        "feedback_score_after": _coerce_optional_number(
            payload.get("feedback_score_after")
            if payload.get("feedback_score_after") is not None
            else gateway_meta.get("feedback_score_after")
        ),
        "sender_open_id": first_non_empty_str(
            sender_id.get("open_id"),
            operator_id.get("open_id"),
            operator.get("open_id"),
            user_id.get("open_id"),
            user.get("open_id"),
            event.get("open_id"),
        ),
        "sender_user_id": first_non_empty_str(
            sender_id.get("user_id"),
            operator_id.get("user_id"),
            operator.get("user_id"),
            user_id.get("user_id"),
            user.get("user_id"),
            event.get("user_id") if isinstance(event.get("user_id"), str) else "",
        ),
        "trace_token": extract_trace_token(payload),
    }


def get_trace_path() -> Path:
    data_root = Path(str(os.getenv("HERMES_MODAL_DATA_DIR") or "/data/hermes").strip() or "/data/hermes")
    return data_root / "feishu_trace.jsonl"


def read_recent_trace_rows(
    *,
    trace_path: str | Path | None = None,
    limit: int = 4000,
    lookback_seconds: int = 24 * 60 * 60,
) -> list[dict[str, Any]]:
    normalized_limit = max(int(limit or 0), 0)
    if normalized_limit <= 0:
        return []

    path = Path(trace_path) if trace_path is not None else get_trace_path()
    if not path.exists():
        return []

    recent_lines: deque[str] = deque(maxlen=normalized_limit)
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                raw = line.strip()
                if raw:
                    recent_lines.append(raw)
    except OSError:
        return []

    now_ts = int(time.time())
    min_ts = now_ts - max(int(lookback_seconds or 0), 0)
    rows: list[dict[str, Any]] = []
    for raw in recent_lines:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        try:
            ts = int(parsed.get("ts") or 0)
        except (TypeError, ValueError):
            ts = 0
        if min_ts > 0 and ts and ts < min_ts:
            continue
        rows.append(parsed)
    return rows


def correlate_message_read_event(
    read_event: Mapping[str, Any],
    *,
    trace_rows: list[dict[str, Any]] | None = None,
    trace_path: str | Path | None = None,
    limit: int = 4000,
    lookback_seconds: int = 24 * 60 * 60,
) -> dict[str, Any]:
    rows = list(trace_rows or []) or read_recent_trace_rows(
        trace_path=trace_path,
        limit=limit,
        lookback_seconds=lookback_seconds,
    )
    message_ids = [
        str(item or "").strip()
        for item in (read_event.get("message_id_list") if isinstance(read_event.get("message_id_list"), list) else [])
        if str(item or "").strip()
    ]

    inbound_matches: dict[str, dict[str, Any]] = {}
    reply_matches: dict[str, dict[str, Any]] = {}
    for row in rows:
        message_id = str(row.get("message_id") or "").strip()
        send_message_id = str(row.get("send_message_id") or "").strip()
        if message_id:
            existing = inbound_matches.get(message_id)
            if existing is None or _trace_match_rank(row) >= _trace_match_rank(existing):
                inbound_matches[message_id] = row
        if send_message_id and _row_send_succeeded(row):
            existing_reply = reply_matches.get(send_message_id)
            if existing_reply is None or _trace_match_rank(row) >= _trace_match_rank(existing_reply):
                reply_matches[send_message_id] = row

    matches: list[dict[str, Any]] = []
    unmatched_message_ids: list[str] = []
    for message_id in message_ids:
        reply_row = reply_matches.get(message_id)
        if reply_row is not None:
            matches.append(
                {
                    "matched_kind": "reply",
                    "read_message_id": message_id,
                    "read_match_source": "reply_send_message_id",
                    "matched_stage": str(reply_row.get("stage") or "").strip(),
                    "matched_event_id": str(reply_row.get("event_id") or "").strip(),
                    "matched_chat_id": str(reply_row.get("chat_id") or "").strip(),
                    "matched_actor_id": str(reply_row.get("actor_id") or "").strip(),
                    "session_key": str(reply_row.get("session_key") or "").strip(),
                    "correlation_id": str(reply_row.get("correlation_id") or "").strip(),
                    "inbound_message_id": str(reply_row.get("message_id") or "").strip(),
                    "reply_send_message_id": message_id,
                    "reply_send_ts_ms": _trace_row_ts_ms(reply_row),
                }
            )
            continue

        inbound_row = inbound_matches.get(message_id)
        if inbound_row is not None:
            matches.append(
                {
                    "matched_kind": "inbound",
                    "read_message_id": message_id,
                    "read_match_source": "inbound_message_id",
                    "matched_stage": str(inbound_row.get("stage") or "").strip(),
                    "matched_event_id": str(inbound_row.get("event_id") or "").strip(),
                    "matched_chat_id": str(inbound_row.get("chat_id") or "").strip(),
                    "matched_actor_id": str(inbound_row.get("actor_id") or "").strip(),
                    "session_key": str(inbound_row.get("session_key") or "").strip(),
                    "correlation_id": str(inbound_row.get("correlation_id") or "").strip(),
                    "inbound_message_id": message_id,
                    "reply_send_message_id": "",
                    "matched_ts_ms": _trace_row_ts_ms(inbound_row),
                }
            )
            continue

        unmatched_message_ids.append(message_id)

    return {
        "message_count": len(message_ids),
        "matched_count": len(matches),
        "unmatched_count": len(unmatched_message_ids),
        "matches": matches,
        "unmatched_message_ids": unmatched_message_ids,
    }


def build_message_read_correlation_trace_extras(
    read_event: Mapping[str, Any],
    match: Mapping[str, Any],
) -> dict[str, Any]:
    read_time = _coerce_epoch_ms(read_event.get("read_time"))
    extras = {
        "reader_open_id": str(read_event.get("reader_open_id") or "").strip(),
        "reader_user_id": str(read_event.get("reader_user_id") or "").strip(),
        "reader_union_id": str(read_event.get("reader_union_id") or "").strip(),
        "tenant_key": str(read_event.get("tenant_key") or "").strip(),
        "read_time": read_time or int(read_event.get("read_time") or 0),
        "read_message_id": str(match.get("read_message_id") or "").strip(),
        "matched_kind": str(match.get("matched_kind") or "").strip(),
        "read_match_source": str(match.get("read_match_source") or "").strip(),
        "matched_stage": str(match.get("matched_stage") or "").strip(),
        "matched_event_id": str(match.get("matched_event_id") or "").strip(),
        "matched_chat_id": str(match.get("matched_chat_id") or "").strip(),
        "matched_actor_id": str(match.get("matched_actor_id") or "").strip(),
        "session_key": str(match.get("session_key") or "").strip(),
        "correlation_id": str(match.get("correlation_id") or "").strip(),
        "inbound_message_id": str(match.get("inbound_message_id") or "").strip(),
        "reply_send_message_id": str(match.get("reply_send_message_id") or "").strip(),
    }
    if read_time is not None:
        base_ts_ms = _coerce_epoch_ms(match.get("matched_ts_ms"))
        if base_ts_ms is None:
            base_ts_ms = _coerce_epoch_ms(match.get("reply_send_ts_ms"))
        if base_ts_ms is not None and read_time >= base_ts_ms:
            extras["read_latency_ms"] = read_time - base_ts_ms
    return extras


def _trace_match_rank(row: Mapping[str, Any]) -> tuple[int, int, int]:
    stage = str(row.get("stage") or "").strip()
    return (
        1 if str(row.get("session_key") or "").strip() else 0,
        _trace_stage_priority(stage),
        _trace_row_ts_ms(row),
    )


def _trace_stage_priority(stage: str) -> int:
    priorities = {
        "gateway.send.done": 5,
        "internal.agent_exec.done": 4,
        "worker.done": 3,
        "queue.enqueue": 2,
        "webhook.accepted": 1,
    }
    return priorities.get(stage, 0)


def _trace_row_ts_ms(row: Mapping[str, Any]) -> int:
    try:
        ts = int(row.get("ts") or 0)
    except (TypeError, ValueError):
        ts = 0
    if ts <= 0:
        return 0
    return ts * 1000


def _row_send_succeeded(row: Mapping[str, Any]) -> bool:
    value = row.get("send_success")
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    return normalized in {"true", "1", "yes"}


def _coerce_epoch_ms(value: Any) -> int | None:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate <= 0:
        return None
    if candidate < 10_000_000_000:
        return candidate * 1000
    return candidate
