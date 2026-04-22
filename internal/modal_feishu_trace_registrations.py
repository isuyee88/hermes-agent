from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_feishu_trace_helpers(namespace: Namespace) -> None:
    def _build_feishu_webhook_ack_response(
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
    ) -> Any:
        return namespace["_feishu_webhook_support_build_ack_response"](
            payload,
            body,
            request_started_at=request_started_at,
            ack_kind=ack_kind,
            partition=partition,
            lane=lane,
            queue_depth=queue_depth,
            reason=reason,
            ingress_strategy=ingress_strategy,
            phase_timings=phase_timings,
            normalize_phase_timings=namespace["_normalize_phase_timings"],
            extract_feishu_trace_token=namespace["_extract_feishu_trace_token"],
            append_feishu_trace=namespace["_append_feishu_trace"],
            extract_feishu_event_metadata=namespace["_extract_feishu_event_metadata"],
            response_cls=namespace["JSONResponse"],
            tracked_response_cls=namespace["_FeishuTrackedJSONResponse"],
            logger=namespace["logger"],
        )

    def _extract_feishu_message_read_event_info(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = namespace["_feishu_webhook_support_extract_message_read_info"](payload)
        if result.get("reader_open_id"):
            return result

        header = payload.get("header") if isinstance(payload.get("header"), Mapping) else {}
        event = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
        reader = event.get("reader") if isinstance(event.get("reader"), Mapping) else {}
        reader_id = reader.get("reader_id") if isinstance(reader.get("reader_id"), Mapping) else {}
        raw_message_ids = event.get("message_id_list")
        message_ids = [str(item or "").strip() for item in raw_message_ids] if isinstance(raw_message_ids, list) else []
        message_ids = [item for item in message_ids if item]

        def _first_non_empty(*values: Any) -> str:
            for value in values:
                text = str(value or "").strip()
                if text:
                    return text
            return ""

        return {
            "event_type": str(header.get("event_type") or payload.get("event_type") or "").strip(),
            "event_id": str(header.get("event_id") or payload.get("event_id") or "").strip(),
            "reader_open_id": _first_non_empty(reader_id.get("open_id"), reader.get("open_id"), event.get("open_id")),
            "reader_user_id": _first_non_empty(
                reader_id.get("user_id"),
                reader.get("user_id"),
                event.get("user_id") if isinstance(event.get("user_id"), str) else "",
            ),
            "reader_union_id": _first_non_empty(reader_id.get("union_id"), reader.get("union_id")),
            "tenant_key": _first_non_empty(reader.get("tenant_key"), event.get("tenant_key")),
            "read_time": int(reader.get("read_time") or event.get("read_time") or 0),
            "message_id_list": message_ids,
            "message_count": len(message_ids),
        }

    def _read_feishu_trace(limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        namespace["_sync_modal_volume"](reload=True)
        if not namespace["FEISHU_TRACE_PATH"].exists():
            return []
        try:
            lines = namespace["FEISHU_TRACE_PATH"].read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows: list[dict[str, Any]] = []
        for raw in lines[-limit:]:
            if not raw.strip():
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows

    def _build_feishu_perf_summary_from_rows(
        rows: list[dict[str, Any]],
        *,
        since_seconds: int = 86400,
        event_type: str = "im.message.receive_v1",
        experiment_label: str = "",
        app_name_filter: str = "",
        snapshot_profile: str = "",
        include_duplicates: bool = False,
    ) -> dict[str, Any]:
        return namespace["_feishu_build_perf_summary_from_rows"](
            rows,
            since_seconds=since_seconds,
            event_type=event_type,
            experiment_label=experiment_label,
            app_name_filter=app_name_filter,
            snapshot_profile=snapshot_profile,
            include_duplicates=include_duplicates,
            normalize_phase_timings_fn=namespace["_normalize_phase_timings"],
        )

    def _build_feishu_ingress_strategy_debug_state(limit: int = 500) -> dict[str, Any]:
        payload = namespace["_feishu_build_ingress_strategy_debug_state"](
            limit=limit,
            read_feishu_trace_fn=lambda normalized_limit: namespace["_read_feishu_trace"](limit=normalized_limit),
            resolve_message_ingress_strategy_fn=namespace["_resolve_feishu_message_ingress_strategy"],
            supported_strategies=namespace["SUPPORTED_FEISHU_MESSAGE_INGRESS_STRATEGIES"],
        )
        payload["configured_strategy"] = str(namespace["DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY"] or "").strip().lower()
        payload["legacy_aliases"] = dict(namespace["LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES"])
        payload["experiment_candidates"] = namespace["_split_csv"](os.getenv("HERMES_FEISHU_MESSAGE_INGRESS_EXPERIMENT"))
        payload["trace_count"] = payload.pop("trace_rows", 0)
        return payload

    namespace["_build_feishu_webhook_ack_response"] = _build_feishu_webhook_ack_response
    namespace["_extract_feishu_message_read_event_info"] = _extract_feishu_message_read_event_info
    namespace["_read_feishu_trace"] = _read_feishu_trace
    namespace["_build_feishu_perf_summary_from_rows"] = _build_feishu_perf_summary_from_rows
    namespace["_build_feishu_ingress_strategy_debug_state"] = _build_feishu_ingress_strategy_debug_state
