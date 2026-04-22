from __future__ import annotations

import time
from typing import Any, Callable


def _summarize_numeric_series(values: list[int | float]) -> dict[str, Any]:
    series = [float(value) for value in values if value is not None]
    if not series:
        return {"count": 0}
    series.sort()

    def _percentile(p: float) -> float:
        if len(series) == 1:
            return series[0]
        idx = (len(series) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(series) - 1)
        frac = idx - lo
        return series[lo] * (1 - frac) + series[hi] * frac

    return {
        "count": len(series),
        "avg": round(sum(series) / len(series), 1),
        "p50": round(_percentile(0.5), 1),
        "p90": round(_percentile(0.9), 1),
        "max": round(max(series), 1),
    }


def _pick_feishu_stage_row(rows: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    stage_rows = [row for row in rows if str(row.get("stage") or "") == stage]
    if not stage_rows:
        return None
    if stage in {"webhook.ack", "webhook.response_sent"}:
        for row in stage_rows:
            if str(row.get("ack_kind") or "").strip().lower() != "duplicate":
                return row
    return stage_rows[-1]


def _count_by_normalized_value(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        normalized = str(value or "").strip() or "unspecified"
        counts[normalized] = counts.get(normalized, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def build_feishu_perf_summary_from_rows(
    rows: list[dict[str, Any]],
    *,
    since_seconds: int = 86400,
    event_type: str = "im.message.receive_v1",
    experiment_label: str = "",
    app_name_filter: str = "",
    snapshot_profile: str = "",
    include_duplicates: bool = False,
    normalize_phase_timings_fn: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    now_ts = int(time.time())
    min_ts = now_ts - max(int(since_seconds or 0), 0) if since_seconds else None
    normalized_event_type = str(event_type or "").strip()
    normalized_label = str(experiment_label or "").strip().lower()
    normalized_app_name = str(app_name_filter or "").strip().lower()
    normalized_snapshot_profile = str(snapshot_profile or "").strip().lower()

    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_ts = int(row.get("ts") or 0)
        if min_ts is not None and row_ts and row_ts < min_ts:
            continue
        if normalized_event_type and str(row.get("event_type") or "").strip() != normalized_event_type:
            continue
        if normalized_label and str(row.get("experiment_label") or "").strip().lower() != normalized_label:
            continue
        if normalized_app_name and str(row.get("app_name") or "").strip().lower() != normalized_app_name:
            continue
        if normalized_snapshot_profile and str(row.get("snapshot_profile") or "").strip().lower() != normalized_snapshot_profile:
            continue
        filtered_rows.append(row)

    events: dict[str, list[dict[str, Any]]] = {}
    for row in filtered_rows:
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        events.setdefault(event_id, []).append(row)

    metric_series: dict[str, list[float]] = {}
    event_summaries: list[dict[str, Any]] = []
    duplicate_only_events = 0

    for event_id, event_rows in events.items():
        ack_row = _pick_feishu_stage_row(event_rows, "webhook.ack")
        if ack_row is None:
            continue

        def _pick_event_field(name: str, default: str = "") -> str:
            for candidate_row in event_rows:
                value = str((candidate_row or {}).get(name) or "").strip()
                if value:
                    return value
            return default

        def _pick_event_bool(name: str, default: bool = False) -> bool:
            for candidate_row in event_rows:
                value = (candidate_row or {}).get(name)
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)) and value in (0, 1):
                    return bool(value)
                normalized = str(value or "").strip().lower()
                if normalized in {"true", "1", "yes", "on"}:
                    return True
                if normalized in {"false", "0", "no", "off"}:
                    return False
            return default

        def _pick_event_number(name: str) -> float | None:
            for candidate_row in event_rows:
                value = (candidate_row or {}).get(name)
                if isinstance(value, (int, float)):
                    return float(value)
                try:
                    normalized = str(value or "").strip()
                    if normalized:
                        return float(normalized)
                except (TypeError, ValueError):
                    continue
            return None

        ack_kind = str(ack_row.get("ack_kind") or "").strip().lower()
        if ack_kind == "duplicate" and not include_duplicates:
            duplicate_only_events += 1
            continue

        response_row = _pick_feishu_stage_row(event_rows, "webhook.response_sent")
        dispatch_done_row = _pick_feishu_stage_row(event_rows, "dispatch.done")
        inline_done_row = _pick_feishu_stage_row(event_rows, "inline_message.done")
        background_exec_done_row = _pick_feishu_stage_row(event_rows, "background_exec.done")
        gateway_handler_done_row = _pick_feishu_stage_row(event_rows, "gateway.handler.done")
        gateway_send_done_row = _pick_feishu_stage_row(event_rows, "gateway.send.done")
        worker_start_row = _pick_feishu_stage_row(event_rows, "worker.start")
        worker_done_row = _pick_feishu_stage_row(event_rows, "worker.done")
        ack_reaction_row = _pick_feishu_stage_row(event_rows, "webhook.ack_reaction")

        ack_phase_timings = normalize_phase_timings_fn(ack_row.get("phase_timings"))
        dispatch_phase_timings = normalize_phase_timings_fn(
            dispatch_done_row.get("phase_timings") if isinstance(dispatch_done_row, dict) else {}
        )

        event_summary = {
            "event_id": event_id,
            "event_type": str(ack_row.get("event_type") or "").strip(),
            "app_name": str(ack_row.get("app_name") or "").strip(),
            "experiment_label": str(ack_row.get("experiment_label") or "").strip(),
            "snapshot_profile": str(ack_row.get("snapshot_profile") or "").strip(),
            "route_hint": _pick_event_field("route_hint", "unspecified"),
            "route_version": _pick_event_field("route_version", "unspecified"),
            "ack_kind": ack_kind or "unspecified",
            "request_class": _pick_event_field("request_class", "unclassified"),
            "site_category": _pick_event_field("site_category", "none"),
            "site_intent": _pick_event_field("site_intent", "general"),
            "target_domain": _pick_event_field("target_domain", "none"),
            "site_skill_name": _pick_event_field("site_skill_name", "none"),
            "site_prefetch_mode": _pick_event_field("site_prefetch_mode", "none"),
            "site_prefetch_status": _pick_event_field("site_prefetch_status", "none"),
            "site_prefetch_direct_navigation": _pick_event_bool("site_prefetch_direct_navigation"),
            "modal_avoided": any(bool((candidate_row or {}).get("modal_avoided")) for candidate_row in event_rows),
            "browser_backend_selected": _pick_event_field("browser_backend_selected", "none"),
            "browser_escalation_reason": _pick_event_field("browser_escalation_reason", "none"),
            "provider_alias": _pick_event_field("provider_alias", "none"),
            "fallback_reason": _pick_event_field("fallback_reason", "none"),
            "gateway_error_class": _pick_event_field("gateway_error_class", "none"),
            "cache_status": _pick_event_field("cache_status", "none"),
            "model_catalog_version": _pick_event_field("model_catalog_version", "none"),
            "cache_eligible": _pick_event_bool("cache_eligible"),
            "capability_match": _pick_event_bool("capability_match"),
            "preferred_model_selected": _pick_event_bool("preferred_model_selected"),
            "execution_mode": str(
                (inline_done_row or {}).get("execution_mode")
                or (background_exec_done_row or {}).get("execution_mode")
                or ""
            ).strip()
            or "unspecified",
            "handoff_reason": str(
                (inline_done_row or {}).get("handoff_reason")
                or (background_exec_done_row or {}).get("handoff_reason")
                or ""
            ).strip()
            or "unspecified",
            "ingress_strategy": str(
                ack_row.get("ingress_strategy")
                or (worker_start_row or {}).get("ingress_strategy")
                or "unspecified"
            ).strip()
            or "unspecified",
            "ack_elapsed_ms": ack_row.get("ack_elapsed_ms"),
            "response_elapsed_ms": (response_row or {}).get("response_elapsed_ms"),
            "queue_latency_ms": (worker_start_row or {}).get("queue_latency_ms"),
            "worker_elapsed_ms": (worker_done_row or {}).get("worker_elapsed_ms"),
            "inline_elapsed_ms": (inline_done_row or {}).get("inline_elapsed_ms"),
            "background_exec_elapsed_ms": (background_exec_done_row or {}).get("worker_elapsed_ms"),
            "gateway_handler_elapsed_ms": (gateway_handler_done_row or {}).get("handler_elapsed_ms"),
            "gateway_send_elapsed_ms": (gateway_send_done_row or {}).get("send_elapsed_ms"),
            "gateway_send_success": (gateway_send_done_row or {}).get("send_success"),
            "estimated_cost_usd": _pick_event_number("estimated_cost_usd") or _pick_event_number("actual_cost_usd") or 0.0,
            "feedback_score_before": _pick_event_number("feedback_score_before"),
            "feedback_score_after": _pick_event_number("feedback_score_after"),
            "ai_call_count": _pick_event_number("ai_call_count"),
            "ack_reaction_total_elapsed_ms": (ack_reaction_row or {}).get("total_elapsed_ms"),
            "ack_reaction_from_request_elapsed_ms": (ack_reaction_row or {}).get("from_request_elapsed_ms"),
            "message_handle_elapsed_ms": dispatch_phase_timings.get("message_handle_elapsed_ms"),
            "background_tasks_elapsed_ms": dispatch_phase_timings.get("background_tasks_elapsed_ms"),
            "pending_batches_elapsed_ms": dispatch_phase_timings.get("pending_batches_elapsed_ms"),
            "background_send_elapsed_ms": (background_exec_done_row or {}).get("background_send_elapsed_ms"),
            "ack_reaction_inline_elapsed_ms": ack_phase_timings.get("ack_reaction_inline_elapsed_ms"),
            "ack_reaction_schedule_elapsed_ms": ack_phase_timings.get("ack_reaction_schedule_elapsed_ms"),
            "dedupe_elapsed_ms": ack_phase_timings.get("dedupe_elapsed_ms"),
            "context_extract_elapsed_ms": ack_phase_timings.get("context_extract_elapsed_ms"),
            "handoff_wait_elapsed_ms": ack_phase_timings.get("handoff_wait_elapsed_ms"),
            "handoff_schedule_wait_elapsed_ms": ack_phase_timings.get("handoff_schedule_wait_elapsed_ms"),
            "inline_fast_extract_elapsed_ms": ack_phase_timings.get("inline_fast_extract_elapsed_ms"),
            "parse_verify_elapsed_ms": ack_phase_timings.get("parse_verify_elapsed_ms"),
            "read_event_extract_elapsed_ms": ack_phase_timings.get("read_event_extract_elapsed_ms"),
            "worker_spawn_elapsed_ms": ack_phase_timings.get("worker_spawn_elapsed_ms"),
            "schedule_claim_elapsed_ms": ack_phase_timings.get("schedule_claim_elapsed_ms"),
            "spawn_rpc_elapsed_ms": ack_phase_timings.get("spawn_rpc_elapsed_ms"),
        }
        event_summaries.append(event_summary)
        for key, value in event_summary.items():
            if isinstance(value, (int, float)):
                metric_series.setdefault(key, []).append(float(value))

    metrics_summary = {
        metric_name: _summarize_numeric_series(values)
        for metric_name, values in sorted(metric_series.items())
        if values
    }
    event_summaries.sort(key=lambda item: str(item.get("event_id") or ""))

    site_prefetch_completed_events = [
        item for item in event_summaries if str(item.get("site_prefetch_status") or "").strip() == "completed"
    ]
    direct_navigation_events = [item for item in event_summaries if bool(item.get("site_prefetch_direct_navigation"))]
    browser_backend_events = [
        item
        for item in event_summaries
        if str(item.get("browser_backend_selected") or "").strip() in {"local", "cloud"}
    ]
    local_browser_events = [item for item in browser_backend_events if str(item.get("browser_backend_selected")) == "local"]
    cloud_browser_events = [item for item in browser_backend_events if str(item.get("browser_backend_selected")) == "cloud"]

    per_domain_summary: dict[str, dict[str, Any]] = {}
    for item in event_summaries:
        domain = str(item.get("target_domain") or "").strip() or "none"
        bucket = per_domain_summary.setdefault(
            domain,
            {"events": 0, "successful_events": 0, "modal_avoided": 0, "estimated_cost_usd": 0.0},
        )
        bucket["events"] += 1
        if item.get("gateway_send_success") is not False:
            bucket["successful_events"] += 1
        if bool(item.get("modal_avoided")):
            bucket["modal_avoided"] += 1
        bucket["estimated_cost_usd"] += float(item.get("estimated_cost_usd") or 0.0)

    per_domain_success_rate = {
        domain: {
            **values,
            "success_rate": _safe_rate(int(values["successful_events"]), int(values["events"])),
        }
        for domain, values in sorted(per_domain_summary.items(), key=lambda item: (-int(item[1]["events"]), item[0]))
    }
    per_domain_cost = {
        domain: round(float(values["estimated_cost_usd"]), 6)
        for domain, values in sorted(per_domain_summary.items(), key=lambda item: (-float(item[1]["estimated_cost_usd"]), item[0]))
    }
    site_prefetch_cache_hit_events = sum(
        1 for event_rows in events.values() if any(str((row or {}).get("stage") or "").strip() == "feishu.site_prefetch.cache_hit" for row in event_rows)
    )

    return {
        "status": "ok",
        "window": {
            "since_seconds": int(since_seconds or 0),
            "event_type": normalized_event_type,
            "experiment_label": normalized_label,
            "app_name": normalized_app_name,
            "snapshot_profile": normalized_snapshot_profile,
            "include_duplicates": bool(include_duplicates),
        },
        "trace_row_count": len(filtered_rows),
        "event_count": len(event_summaries),
        "duplicate_only_event_count": duplicate_only_events,
        "by_ingress_strategy": _count_by_normalized_value([str(item.get("ingress_strategy") or "") for item in event_summaries]),
        "by_execution_mode": _count_by_normalized_value([str(item.get("execution_mode") or "") for item in event_summaries]),
        "by_handoff_reason": _count_by_normalized_value([str(item.get("handoff_reason") or "") for item in event_summaries]),
        "by_experiment_label": _count_by_normalized_value([str(item.get("experiment_label") or "") for item in event_summaries]),
        "by_snapshot_profile": _count_by_normalized_value([str(item.get("snapshot_profile") or "") for item in event_summaries]),
        "by_app_name": _count_by_normalized_value([str(item.get("app_name") or "") for item in event_summaries]),
        "by_route_hint": _count_by_normalized_value([str(item.get("route_hint") or "") for item in event_summaries]),
        "by_route_version": _count_by_normalized_value([str(item.get("route_version") or "") for item in event_summaries]),
        "by_request_class": _count_by_normalized_value([str(item.get("request_class") or "") for item in event_summaries]),
        "by_provider_alias": _count_by_normalized_value([str(item.get("provider_alias") or "") for item in event_summaries]),
        "by_fallback_reason": _count_by_normalized_value([str(item.get("fallback_reason") or "") for item in event_summaries]),
        "by_gateway_error_class": _count_by_normalized_value(
            [str(item.get("gateway_error_class") or "") for item in event_summaries]
        ),
        "by_cache_status": _count_by_normalized_value([str(item.get("cache_status") or "") for item in event_summaries]),
        "by_model_catalog_version": _count_by_normalized_value(
            [str(item.get("model_catalog_version") or "") for item in event_summaries]
        ),
        "by_site_category": _count_by_normalized_value([str(item.get("site_category") or "") for item in event_summaries]),
        "by_site_intent": _count_by_normalized_value([str(item.get("site_intent") or "") for item in event_summaries]),
        "by_target_domain": _count_by_normalized_value([str(item.get("target_domain") or "") for item in event_summaries]),
        "by_site_skill_name": _count_by_normalized_value([str(item.get("site_skill_name") or "") for item in event_summaries]),
        "by_site_prefetch_mode": _count_by_normalized_value([str(item.get("site_prefetch_mode") or "") for item in event_summaries]),
        "by_site_prefetch_status": _count_by_normalized_value([str(item.get("site_prefetch_status") or "") for item in event_summaries]),
        "modal_avoided_by_site_prefetch": sum(
            1
            for item in event_summaries
            if bool(item.get("modal_avoided")) and str(item.get("site_prefetch_status") or "").strip() == "completed"
        ),
        "by_browser_backend_selected": _count_by_normalized_value(
            [str(item.get("browser_backend_selected") or "") for item in event_summaries]
        ),
        "by_browser_escalation_reason": _count_by_normalized_value(
            [str(item.get("browser_escalation_reason") or "") for item in event_summaries]
        ),
        "site_prefetch_cache_hit_rate": _safe_rate(site_prefetch_cache_hit_events, len(site_prefetch_completed_events)),
        "site_prefetch_direct_navigation_success_rate": _safe_rate(
            sum(1 for item in direct_navigation_events if item.get("gateway_send_success") is not False),
            len(direct_navigation_events),
        ),
        "local_browser_success_rate": _safe_rate(
            sum(1 for item in local_browser_events if item.get("gateway_send_success") is not False),
            len(local_browser_events),
        ),
        "cloud_escalation_rate": _safe_rate(len(cloud_browser_events), len(browser_backend_events)),
        "per_domain_success_rate": per_domain_success_rate,
        "per_domain_cost": per_domain_cost,
        "ack_kind_counts": _count_by_normalized_value([str(item.get("ack_kind") or "") for item in event_summaries]),
        "metrics": metrics_summary,
        "events": event_summaries,
    }


def build_feishu_ingress_strategy_debug_state(
    *,
    limit: int = 500,
    read_feishu_trace_fn: Callable[[int], list[dict[str, Any]]],
    resolve_message_ingress_strategy_fn: Callable[[dict[str, Any], dict[str, Any]], str],
    supported_strategies: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    rows = read_feishu_trace_fn(max(limit, 1))
    ack_rows = [row for row in rows if str(row.get("stage") or "") == "webhook.ack"]
    by_strategy: dict[str, dict[str, list[float]]] = {}
    for row in ack_rows:
        strategy = str(row.get("ingress_strategy") or "").strip() or "unspecified"
        bucket = by_strategy.setdefault(strategy, {"ack_elapsed_ms": []})
        ack_elapsed = row.get("ack_elapsed_ms")
        if isinstance(ack_elapsed, (int, float)):
            bucket["ack_elapsed_ms"].append(float(ack_elapsed))

    handoff_rows = [row for row in rows if str(row.get("stage") or "") == "webhook.handoff_spawned"]
    for row in handoff_rows:
        strategy = str(row.get("ingress_strategy") or "").strip() or "unspecified"
        by_strategy.setdefault(strategy, {"ack_elapsed_ms": []})

    queue_rows = [row for row in rows if str(row.get("stage") or "") == "worker.start"]
    for row in queue_rows:
        strategy = str(row.get("ingress_strategy") or "").strip() or "unspecified"
        bucket = by_strategy.setdefault(strategy, {"ack_elapsed_ms": [], "queue_latency_ms": []})
        queue_latency = row.get("queue_latency_ms")
        if isinstance(queue_latency, (int, float)):
            bucket.setdefault("queue_latency_ms", []).append(float(queue_latency))

    summary = {}
    for strategy, metrics in by_strategy.items():
        summary[strategy] = {
            key: _summarize_numeric_series([int(value) for value in values])
            for key, values in metrics.items()
            if values
        }

    return {
        "status": "ok",
        "supported_strategies": list(supported_strategies),
        "effective_strategy": resolve_message_ingress_strategy_fn({}, {}),
        "trace_rows": len(rows),
        "summary": summary,
    }
