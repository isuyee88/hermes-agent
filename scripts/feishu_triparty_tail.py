import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_report_helpers():
    module_path = REPO_ROOT / "scripts" / "feishu_triparty_pk_report.py"
    spec = importlib.util.spec_from_file_location("feishu_triparty_tail_report_helpers", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report_helpers = _load_report_helpers()


DEFAULT_CF_EVENTS = {
    "feishu.webhook.accepted",
    "feishu.message_read.accepted",
    "feishu.ack_reaction.done",
    "feishu.direct_planned.start",
    "feishu.direct_planned.plan.done",
    "feishu.direct_planned.cf_ai_exec.fallback",
    "feishu.direct_planned.modal_exec.fallback_done",
    "feishu.direct_planned.modal_exec.fallback_error",
    "feishu.direct_planned.workflow_enqueued",
    "feishu.direct_planned.done",
    "feishu.workflow.start",
    "feishu.workflow.plan.done",
    "feishu.workflow.exec_delegation_candidate",
    "feishu.workflow.browser_route",
    "feishu.cf_ai_exec.attempt_failed",
    "feishu.cf_ai_exec.retrying_with_fallback_model",
    "feishu.cf_ai_exec.fallback",
    "feishu.cf_ai_exec.done",
    "feishu.workflow.send.done",
    "feishu.send.operation.start",
    "feishu.send.operation.done",
    "feishu.send.operation.error",
    "feishu.workflow.done",
    "feishu.workflow.enqueue.error",
    "feishu.workflow.enqueue.duplicate",
}

DEFAULT_MODAL_STAGES = {
    "webhook.accepted",
    "webhook.ack",
    "webhook.response_sent",
    "webhook.ack_reaction",
    "webhook.ack_reaction_detached",
    "webhook.message_read",
    "inline_message.start",
    "inline_message.done",
    "background_exec.start",
    "background_exec.done",
    "internal.agent_plan.done",
    "internal.agent_exec.start",
    "internal.agent_exec.done",
    "internal.agent_exec.error",
    "gateway.handler.done",
    "gateway.send.done",
    "worker.start",
    "worker.done",
    "worker.error",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll Cloudflare Worker observability and Modal trace like a live tail for Feishu triparty debugging."
    )
    parser.add_argument("--app", default="hermes-agent")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--lookback-seconds", type=int, default=300)
    parser.add_argument("--trace-limit", type=int, default=1200)
    parser.add_argument("--cloudflare-limit", type=int, default=300)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--jsonl-out", default="")
    parser.add_argument("--state-file", default=str(REPO_ROOT / ".tmp-feishu-triparty-tail-state.json"))
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--correlation-id", action="append", default=[])
    parser.add_argument("--chat-id", action="append", default=[])
    return parser.parse_args()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalized_epoch_ms(value: Any) -> int | None:
    return report_helpers._normalized_epoch_ms(value)


def _stage_ts_ms(row: dict[str, Any]) -> int:
    return report_helpers._stage_ts_ms(row)


def _read_modal_rows(app: str, limit: int) -> list[dict[str, Any]]:
    rows = report_helpers._read_trace_rows(app, limit)
    return rows if isinstance(rows, list) else []


def _fetch_cloudflare_worker_events(
    *,
    repo_root: Path,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> list[dict[str, Any]]:
    token = str(os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
    account_id = str(os.getenv("CLOUDFLARE_ACCOUNT_ID") or "d1215a30b84b673ef0367010b0e78c10").strip()
    script_name = report_helpers._infer_cloudflare_worker_script_name(repo_root)
    if not token:
        return []

    request_body = {
        "queryId": "feishu-triparty-tail",
        "view": "events",
        "limit": max(1, min(int(limit or 0), 2000)),
        "timeframe": {"from": start_ms, "to": end_ms},
        "parameters": {
            "filterCombination": "and",
            "filters": [
                {
                    "kind": "filter",
                    "key": "$workers.scriptName",
                    "operation": "eq",
                    "type": "string",
                    "value": script_name,
                },
                {
                    "kind": "filter",
                    "key": "event",
                    "operation": "exists",
                    "type": "string",
                },
            ],
            "orderBy": {"value": "$metadata.id", "order": "desc"},
        },
    }
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/workers/observability/telemetry/query"
    request = Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    raw_events = ((((payload if isinstance(payload, dict) else {}).get("result") or {}).get("events") or {}).get("events") or [])
    events: list[dict[str, Any]] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        source = dict(source) if isinstance(source, dict) else {}
        event_name = str(source.get("event") or "").strip()
        ts_ms = _normalized_epoch_ms(item.get("timestamp"))
        if not event_name or ts_ms is None or not (start_ms <= ts_ms < end_ms):
            continue
        metadata = item.get("$metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        workers = item.get("$workers")
        workers = dict(workers) if isinstance(workers, dict) else {}
        record = {
            "uid": f"cf:{metadata.get('id') or workers.get('requestId') or ts_ms}:{event_name}",
            "source": "cloudflare_worker",
            "ts_ms": ts_ms,
            "event": event_name,
            "event_id": str(source.get("event_id") or report_helpers._event_id_from_correlation_id(source.get("correlation_id")) or "").strip(),
            "correlation_id": str(source.get("correlation_id") or "").strip(),
            "chat_id": str(source.get("chat_id") or "").strip(),
            "message_id": str(source.get("message_id") or "").strip(),
            "route_hint": str(source.get("route_hint") or "").strip(),
            "execution_mode": str(source.get("execution_mode") or "").strip(),
            "details": {
                "request_id": workers.get("requestId"),
                "workflow_instance_id": workers.get("instanceId"),
                "origin": metadata.get("origin"),
                "service": metadata.get("service"),
                "request_class": source.get("request_class"),
                "content_modalities": source.get("content_modalities"),
                "route_family": source.get("route_family"),
                "gateway_route_name": source.get("gateway_route_name"),
                "gateway_eligible": source.get("gateway_eligible"),
                "requires_tools": source.get("requires_tools"),
                "requires_browser": source.get("requires_browser"),
                "requires_media_hydration": source.get("requires_media_hydration"),
                "toolset": source.get("toolset"),
                "modality_profile": source.get("modality_profile"),
                "gateway_error_class": source.get("gateway_error_class"),
                "misroute_detected": source.get("misroute_detected"),
                "dfmea_failure_mode": source.get("dfmea_failure_mode"),
                "dfmea_control_phase": source.get("dfmea_control_phase"),
                "dfmea_detection_signal": source.get("dfmea_detection_signal"),
                "dfmea_control_action": source.get("dfmea_control_action"),
                "dfmea_severity": source.get("dfmea_severity"),
                "cf_ai_exec_elapsed_ms": source.get("cf_ai_exec_elapsed_ms"),
                "cf_send_elapsed_ms": source.get("cf_send_elapsed_ms"),
                "operation_count": source.get("operation_count"),
                "route_decision_reason": source.get("route_decision_reason"),
                "fallback_reason": source.get("fallback_reason"),
                "fallback_path": source.get("fallback_path"),
                "provider_async_eligible": source.get("provider_async_eligible"),
                "provider_async_observed": source.get("provider_async_observed"),
                "error": source.get("error"),
                "provider": source.get("provider"),
                "model": source.get("model"),
                "status": source.get("status"),
            },
        }
        events.append(record)
    return events


def _normalize_modal_row(row: dict[str, Any]) -> dict[str, Any] | None:
    stage = str(row.get("stage") or "").strip()
    ts_ms = _stage_ts_ms(row)
    if not stage or ts_ms <= 0:
        return None
    event_id = str(row.get("event_id") or "").strip()
    message_id = str(row.get("message_id") or "").strip()
    return {
        "uid": f"modal:{event_id}:{stage}:{ts_ms}:{message_id}",
        "source": "modal_trace",
        "ts_ms": ts_ms,
        "event": stage,
        "event_id": event_id,
        "correlation_id": str(row.get("correlation_id") or "").strip(),
        "chat_id": str(row.get("chat_id") or "").strip(),
        "message_id": message_id,
        "route_hint": str(row.get("route_hint") or "").strip(),
        "execution_mode": str(row.get("execution_mode") or "").strip(),
        "details": {
            "request_class": row.get("request_class"),
            "content_modalities": row.get("content_modalities"),
            "route_family": row.get("route_family"),
            "gateway_route_name": row.get("gateway_route_name"),
            "gateway_eligible": row.get("gateway_eligible"),
            "requires_tools": row.get("requires_tools"),
            "requires_browser": row.get("requires_browser"),
            "requires_media_hydration": row.get("requires_media_hydration"),
            "toolset": row.get("toolset"),
            "modality_profile": row.get("modality_profile"),
            "gateway_error_class": row.get("gateway_error_class"),
            "misroute_detected": row.get("misroute_detected"),
            "dfmea_failure_mode": row.get("dfmea_failure_mode"),
            "dfmea_control_phase": row.get("dfmea_control_phase"),
            "dfmea_detection_signal": row.get("dfmea_detection_signal"),
            "dfmea_control_action": row.get("dfmea_control_action"),
            "dfmea_severity": row.get("dfmea_severity"),
            "send_success": row.get("send_success"),
            "send_elapsed_ms": row.get("send_elapsed_ms"),
            "ack_elapsed_ms": row.get("ack_elapsed_ms"),
            "handler_elapsed_ms": row.get("handler_elapsed_ms"),
            "agent_exec_elapsed_ms": row.get("agent_exec_elapsed_ms"),
            "agent_model_elapsed_ms": row.get("agent_model_elapsed_ms"),
            "agent_non_model_elapsed_ms": row.get("agent_non_model_elapsed_ms"),
            "gateway_run_agent_total_elapsed_ms": row.get("gateway_run_agent_total_elapsed_ms"),
            "provider_wait_elapsed_ms": row.get("provider_wait_elapsed_ms"),
            "tool_exec_elapsed_ms": row.get("tool_exec_elapsed_ms"),
            "hermes_overhead_elapsed_ms": row.get("hermes_overhead_elapsed_ms"),
            "provider_wait_measurement_mode": row.get("provider_wait_measurement_mode"),
            "route_decision_reason": row.get("route_decision_reason"),
            "fallback_reason": row.get("fallback_reason"),
            "worker_elapsed_ms": row.get("worker_elapsed_ms"),
            "worker_boot_id": row.get("worker_boot_id"),
            "gateway_hop": row.get("gateway_hop"),
            "gateway_script": row.get("gateway_script"),
            "error": row.get("error"),
            "read_time": row.get("read_time"),
        },
    }


def _event_matches_filters(
    event: dict[str, Any],
    *,
    event_ids: set[str],
    correlation_ids: set[str],
    chat_ids: set[str],
) -> bool:
    if event_ids and str(event.get("event_id") or "").strip() not in event_ids:
        return False
    if correlation_ids and str(event.get("correlation_id") or "").strip() not in correlation_ids:
        return False
    if chat_ids and str(event.get("chat_id") or "").strip() not in chat_ids:
        return False
    return True


def _is_interesting_event(event: dict[str, Any]) -> bool:
    source = str(event.get("source") or "")
    name = str(event.get("event") or "")
    if source == "cloudflare_worker":
        return name in DEFAULT_CF_EVENTS
    if source == "modal_trace":
        return name in DEFAULT_MODAL_STAGES
    return False


def _format_event_line(event: dict[str, Any], timezone_name: str) -> str:
    tz = report_helpers._get_timezone(timezone_name)
    ts = datetime.fromtimestamp(int(event["ts_ms"]) / 1000, tz=dt_timezone.utc).astimezone(tz)
    details = dict(event.get("details") or {})
    detail_parts = []
    for key in (
        "cf_ai_exec_elapsed_ms",
        "cf_send_elapsed_ms",
        "send_elapsed_ms",
        "ack_elapsed_ms",
        "agent_exec_elapsed_ms",
        "agent_model_elapsed_ms",
        "agent_non_model_elapsed_ms",
        "worker_elapsed_ms",
        "send_success",
        "gateway_hop",
        "gateway_script",
        "fallback_path",
        "provider",
        "model",
        "request_class",
        "route_family",
        "gateway_route_name",
        "gateway_error_class",
        "dfmea_failure_mode",
        "dfmea_control_phase",
        "dfmea_control_action",
        "error",
    ):
        value = details.get(key)
        if value is None or value == "":
            continue
        detail_parts.append(f"{key}={value}")
    summary = " ".join(detail_parts)
    return (
        f"{ts.isoformat()} "
        f"[{event.get('source')}] "
        f"{event.get('event')} "
        f"event_id={event.get('event_id') or '-'} "
        f"corr={event.get('correlation_id') or '-'} "
        f"chat={event.get('chat_id') or '-'} "
        f"msg={event.get('message_id') or '-'} "
        f"route={event.get('route_hint') or '-'} "
        f"exec={event.get('execution_mode') or '-'}"
        + (f" {summary}" if summary else "")
    )


def _load_seen_state(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    seen = payload.get("seen") if isinstance(payload, dict) else []
    return {str(item).strip() for item in seen if str(item).strip()}


def _save_seen_state(path: Path, seen: set[str]) -> None:
    path.write_text(json.dumps({"seen": sorted(seen)[-5000:]}, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()
    state_path = Path(args.state_file).resolve()
    jsonl_path = Path(args.jsonl_out).resolve() if args.jsonl_out else None
    seen = _load_seen_state(state_path)
    event_ids = {str(item).strip() for item in args.event_id if str(item).strip()}
    correlation_ids = {str(item).strip() for item in args.correlation_id if str(item).strip()}
    chat_ids = {str(item).strip() for item in args.chat_id if str(item).strip()}
    iterations = 1 if args.once else max(int(args.max_iterations or 0), 0)
    iteration = 0

    while True:
        end_ms = _now_ms()
        start_ms = end_ms - max(int(args.lookback_seconds or 0), 30) * 1000

        cloudflare_events: list[dict[str, Any]] = []
        try:
            cloudflare_events = _fetch_cloudflare_worker_events(
                repo_root=REPO_ROOT,
                start_ms=start_ms,
                end_ms=end_ms,
                limit=args.cloudflare_limit,
            )
        except Exception as exc:
            print(f"[tail] cloudflare fetch failed: {exc}", file=sys.stderr)

        modal_events: list[dict[str, Any]] = []
        try:
            for row in _read_modal_rows(args.app, args.trace_limit):
                normalized = _normalize_modal_row(row)
                if normalized is None:
                    continue
                if not (start_ms <= int(normalized["ts_ms"]) < end_ms):
                    continue
                modal_events.append(normalized)
        except Exception as exc:
            print(f"[tail] modal trace fetch failed: {exc}", file=sys.stderr)

        merged = cloudflare_events + modal_events
        merged.sort(key=lambda item: (int(item["ts_ms"]), str(item["source"]), str(item["event"])))

        new_seen: set[str] = set()
        for event in merged:
            uid = str(event.get("uid") or "").strip()
            if not uid or uid in seen:
                continue
            if not _is_interesting_event(event):
                continue
            if not _event_matches_filters(
                event,
                event_ids=event_ids,
                correlation_ids=correlation_ids,
                chat_ids=chat_ids,
            ):
                continue
            print(_format_event_line(event, args.timezone))
            if jsonl_path is not None:
                _append_jsonl(jsonl_path, event)
            new_seen.add(uid)

        seen.update(new_seen)
        _save_seen_state(state_path, seen)

        iteration += 1
        if args.once:
            break
        if iterations and iteration >= iterations:
            break
        time.sleep(max(float(args.poll_seconds or 0), 0.5))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
