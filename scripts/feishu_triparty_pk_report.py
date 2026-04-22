import argparse
import httpx
import json
import os
import re
import socket
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from decimal import Decimal, InvalidOperation
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = REPO_ROOT / ".tmp-feishu-triparty-pk-report.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / ".tmp-feishu-triparty-pk-summary.md"
DEFAULT_LOCAL_PROXY_URL = "http://127.0.0.1:12334"
HISTORICAL_T3_PROXY_LABEL = "send_success_proxy"
FIXED_WINDOW_HOURS = (1, 24, 72)
FIXED_TZ_FALLBACKS = {
    "Asia/Shanghai": dt_timezone(timedelta(hours=8)),
    "UTC": dt_timezone.utc,
}
DEFAULT_FUNCTION_RESOURCES = {
    "web_app": {"cpu": 1.0, "memory_mb": 2048, "stage": "webhook.response_sent", "duration_field": "response_elapsed_ms"},
    "internal_agent_exec": {
        "cpu": 1.0,
        "memory_mb": 2048,
        "stage": "internal.agent_exec.done",
        "duration_field": "agent_exec_elapsed_ms",
    },
    "feishu_inline_worker": {
        "cpu": 0.5,
        "memory_mb": 1024,
        "stage": "inline_message.done",
        "duration_field": "inline_elapsed_ms",
    },
    "feishu_background_exec_worker": {
        "cpu": None,
        "memory_mb": None,
        "stage": "background_exec.done",
        "duration_field": "worker_elapsed_ms",
    },
    "feishu_ack_reaction_worker": {
        "cpu": None,
        "memory_mb": None,
        "stage": "webhook.ack_reaction",
        "duration_field": "total_elapsed_ms",
    },
    "chat_queue_worker": {
        "cpu": None,
        "memory_mb": None,
        "stage": "worker.done",
        "duration_field": "worker_elapsed_ms",
    },
}
DEFAULT_GOALS = {
    "read_receipt_ms": 5000,
    "reply_minus_ai_ms": 20000,
    "idle_hourly_cost_usd": Decimal("0.005"),
    "cost_per_session_usd": Decimal("0.0045"),
    "cache_hit_rate": Decimal("0.30"),
    "browser_single_ai_call_completion_rate": Decimal("0.50"),
    "capability_match_rate": Decimal("1.00"),
    "preferred_model_selection_accuracy": Decimal("0.95"),
}


@dataclass(frozen=True)
class AnalysisWindow:
    label: str
    compare_day_shift: int
    start: datetime
    end: datetime

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        return int(self.end.timestamp() * 1000)

    @property
    def hours(self) -> float:
        return max((self.end - self.start).total_seconds() / 3600.0, 0.0)


def _load_module(module_name: str, path: Path):
    spec = spec_from_file_location(module_name, path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_cost_report_module():
    return _load_module("feishu_perf_cost_report_module", REPO_ROOT / "scripts" / "feishu_perf_cost_report.py")


def _load_modal_module():
    return _load_module("feishu_triparty_modal_module", REPO_ROOT / "modal_.py")


def _load_feishu_perf_module():
    return _load_module("feishu_triparty_feishu_perf_module", REPO_ROOT / "internal" / "feishu_perf.py")


def _load_modal_client():
    try:
        import modal
    except ImportError:
        return None
    return modal


def _decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _decimal_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    try:
        return format(value.quantize(Decimal("0.00000001")), "f")
    except InvalidOperation:
        return format(value, "f")


def _safe_pct_delta(current: Decimal, baseline: Decimal | None) -> float | None:
    if baseline is None or baseline == 0:
        return None
    return round(float(((current - baseline) / baseline) * Decimal("100")), 2)


def _safe_ratio(numerator: Decimal | int | float, denominator: Decimal | int | float) -> float | None:
    denominator_decimal = _decimal(denominator)
    if denominator_decimal == 0:
        return None
    return round(float(_decimal(numerator) / denominator_decimal), 6)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _parse_now(now_raw: str | None, timezone_name: str) -> datetime:
    tz = _get_timezone(timezone_name)
    if now_raw:
        parsed = datetime.fromisoformat(now_raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return parsed.astimezone(tz)
    return datetime.now(tz)


def _get_timezone(timezone_name: str):
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        fallback = FIXED_TZ_FALLBACKS.get(timezone_name)
        if fallback is not None:
            return fallback
        raise


def _apply_default_proxy_env() -> None:
    if os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("ALL_PROXY"):
        return
    proxy_url = str(os.getenv("HERMES_LOCAL_PROXY_URL") or DEFAULT_LOCAL_PROXY_URL).strip()
    host = "127.0.0.1"
    port = 12334
    try:
        parsed = proxy_url.removeprefix("http://").removeprefix("https://")
        host_part, port_part = parsed.split(":", 1)
        host = host_part.strip(" /") or host
        port = int(port_part.split("/", 1)[0])
    except Exception:
        return
    try:
        with socket.create_connection((host, port), timeout=0.25):
            os.environ.setdefault("HTTP_PROXY", proxy_url)
            os.environ.setdefault("HTTPS_PROXY", proxy_url)
            os.environ.setdefault("ALL_PROXY", proxy_url)
    except OSError:
        return


def _to_iso(dt: datetime) -> str:
    return dt.astimezone().replace(microsecond=0).isoformat()


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _iter_hour_slots(start: datetime, end: datetime) -> list[datetime]:
    slots: list[datetime] = []
    cursor = _floor_to_hour(start)
    while cursor < end:
        slots.append(cursor)
        cursor += timedelta(hours=1)
    return slots


def _build_analysis_windows(
    *,
    hours: int,
    compare_days: list[int],
    timezone_name: str,
    now: datetime | None = None,
) -> dict[str, AnalysisWindow]:
    tz = _get_timezone(timezone_name)
    end = (now or datetime.now(tz)).astimezone(tz)
    start = end - timedelta(hours=max(int(hours or 0), 0))
    windows: dict[str, AnalysisWindow] = {
        "current": AnalysisWindow(label="current", compare_day_shift=0, start=start, end=end)
    }
    for day in sorted({max(int(day), 0) for day in compare_days if int(day) > 0}):
        delta = timedelta(days=day)
        windows[f"d-{day}"] = AnalysisWindow(
            label=f"d-{day}",
            compare_day_shift=day,
            start=start - delta,
            end=end - delta,
        )
    return windows


def _row_ts_ms(row: dict[str, Any]) -> int:
    ts_ms = _normalized_epoch_ms(row.get("__timestamp_ms"))
    if ts_ms is not None:
        return ts_ms
    ts = row.get("ts")
    try:
        return int(float(ts) * 1000)
    except (TypeError, ValueError):
        return 0


def _normalized_epoch_ms(value: Any) -> int | None:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate < 1_600_000_000_000:
        return None
    return candidate


def _stage_ts_ms(row: dict[str, Any]) -> int:
    if str(row.get("stage") or "").strip() == "webhook.message_read":
        return _normalized_epoch_ms(row.get("read_time")) or _row_ts_ms(row)
    return _row_ts_ms(row)


def _row_in_window(row: dict[str, Any], window: AnalysisWindow) -> bool:
    ts_ms = _stage_ts_ms(row)
    return window.start_ms <= ts_ms < window.end_ms


def _pick_stage(rows: Iterable[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    matching = [row for row in rows if str(row.get("stage") or "").strip() == stage]
    if not matching:
        return None
    matching.sort(key=lambda row: (_stage_ts_ms(row), _row_ts_ms(row)))
    return matching[-1]


def _pick_first_stage(rows: Iterable[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    matching = [row for row in rows if str(row.get("stage") or "").strip() == stage]
    if not matching:
        return None
    matching.sort(key=lambda row: (_stage_ts_ms(row), _row_ts_ms(row)))
    return matching[0]


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    rank = (len(ordered) - 1) * max(0.0, min(percentile, 1.0))
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 1)


def _quantile_decimal(values: list[Decimal], percentile: float) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = Decimal(str(len(ordered) - 1)) * Decimal(str(max(0.0, min(percentile, 1.0))))
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - Decimal(lower)
    value = ordered[lower] * (Decimal("1") - weight) + ordered[upper] * weight
    return value


def _summarize_numeric_series(values: Iterable[int | float | Decimal]) -> dict[str, Any]:
    normalized = [float(value) for value in values if isinstance(value, (int, float, Decimal))]
    if not normalized:
        return {"count": 0}
    return {
        "count": len(normalized),
        "avg": round(sum(normalized) / len(normalized), 1),
        "p50": _quantile(normalized, 0.50),
        "p90": _quantile(normalized, 0.90),
        "p99": _quantile(normalized, 0.99),
        "max": round(max(normalized), 1),
    }


def _summarize_decimal_series(values: Iterable[int | float | Decimal | str]) -> dict[str, Any]:
    normalized = [_decimal(value) for value in values]
    normalized = [value for value in normalized if value is not None]
    if not normalized:
        return {"count": 0}
    average = sum(normalized, Decimal("0")) / len(normalized)
    return {
        "count": len(normalized),
        "avg": _decimal_str(average),
        "p50": _decimal_str(_quantile_decimal(normalized, 0.50)),
        "p90": _decimal_str(_quantile_decimal(normalized, 0.90)),
        "p99": _decimal_str(_quantile_decimal(normalized, 0.99)),
        "max": _decimal_str(max(normalized)),
    }


def _collect_nonnegative_decimals(values: Iterable[int | float | Decimal | str | None]) -> list[Decimal]:
    collected: list[Decimal] = []
    for value in values:
        if value is None:
            continue
        decimal_value = _decimal(value)
        if decimal_value < 0:
            continue
        collected.append(decimal_value)
    return collected


def _goal_status(
    *,
    actual: Decimal | None,
    target: Decimal,
    smaller_is_better: bool = True,
    inclusive: bool = False,
) -> str:
    if actual is None:
        return "not_met"
    if smaller_is_better:
        return "met" if (actual <= target if inclusive else actual < target) else "not_met"
    return "met" if (actual >= target if inclusive else actual > target) else "not_met"


def _resource_shapes(modal_module) -> dict[str, dict[str, Any]]:
    resources = json.loads(json.dumps(DEFAULT_FUNCTION_RESOURCES))
    resources["feishu_background_exec_worker"]["cpu"] = float(
        getattr(modal_module, "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU", 1.0)
    )
    resources["feishu_background_exec_worker"]["memory_mb"] = int(
        getattr(modal_module, "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB", 1024)
    )
    resources["feishu_ack_reaction_worker"]["cpu"] = float(
        getattr(modal_module, "DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU", 0.5)
    )
    resources["feishu_ack_reaction_worker"]["memory_mb"] = int(
        getattr(modal_module, "DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB", 512)
    )
    resources["chat_queue_worker"]["cpu"] = float(getattr(modal_module, "DEFAULT_CHAT_QUEUE_WORKER_CPU", 1.0))
    resources["chat_queue_worker"]["memory_mb"] = int(
        getattr(modal_module, "DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB", 1024)
    )
    return resources


def _function_name_for_row(row: dict[str, Any]) -> str | None:
    stage = str(row.get("stage") or "").strip()
    if stage == "webhook.response_sent":
        return "web_app"
    if stage == "internal.agent_exec.done":
        return "internal_agent_exec"
    if stage == "inline_message.done":
        return "feishu_inline_worker"
    if stage == "background_exec.done":
        return "feishu_background_exec_worker"
    if stage == "worker.done":
        return "chat_queue_worker"
    if stage == "webhook.ack_reaction":
        worker_boot_id = str(row.get("worker_boot_id") or "").strip()
        if worker_boot_id.startswith("feishu-webhook-inline-"):
            return None
        return "feishu_ack_reaction_worker"
    return None


def _row_duration_field(function_name: str) -> str:
    return str(DEFAULT_FUNCTION_RESOURCES.get(function_name, {}).get("duration_field") or "")


def _parse_billing_interval(interval_raw: str, timezone_name: str) -> datetime | None:
    normalized = str(interval_raw or "").strip()
    if not normalized:
        return None
    try:
        dt = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    tz = _get_timezone(timezone_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _hour_key(dt: datetime) -> str:
    return _floor_to_hour(dt).isoformat()


def _hour_key_from_ms(timestamp_ms: int, timezone_name: str) -> str:
    tz = _get_timezone(timezone_name)
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=tz)
    return _hour_key(dt)


def _empty_hourly_map(window: AnalysisWindow) -> dict[str, Decimal]:
    return {_hour_key(slot): Decimal("0") for slot in _iter_hour_slots(window.start, window.end)}


def _match_billing_descriptions(app: str, billing_description: list[str]) -> set[str]:
    descriptions = {item.strip() for item in billing_description if str(item).strip()}
    if not descriptions:
        descriptions.add(app)
    return descriptions


def _read_trace_rows(app: str, limit: int) -> list[dict[str, Any]]:
    modal = _load_modal_client()
    if modal is None:
        return []
    payload = modal.Function.from_name(app, "debug_feishu_trace").remote(limit=max(int(limit or 0), 1))
    rows = payload.get("rows") if isinstance(payload, dict) else []
    return rows if isinstance(rows, list) else []


def _read_trace_rows_from_file(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    if limit > 0:
        lines = lines[-limit:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _filter_trace_rows(rows: list[dict[str, Any]], window: AnalysisWindow) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and _row_in_window(row, window)]


def _build_window_perf_summary(
    modal_module,
    rows: list[dict[str, Any]],
    *,
    window: AnalysisWindow,
    event_type: str,
    experiment_label: str,
    app_name_filter: str,
    snapshot_profile: str,
    include_duplicates: bool,
) -> dict[str, Any]:
    feishu_perf_module = _load_feishu_perf_module()
    return feishu_perf_module.build_feishu_perf_summary_from_rows(
        rows,
        since_seconds=0,
        event_type=event_type,
        experiment_label=experiment_label,
        app_name_filter=app_name_filter,
        snapshot_profile=snapshot_profile,
        include_duplicates=include_duplicates,
        normalize_phase_timings_fn=lambda value: {
            str(key): int(raw)
            for key, raw in (value.items() if isinstance(value, dict) else [])
            if isinstance(raw, (int, float)) or str(raw).strip().lstrip("-").isdigit()
        },
    ) | {
        "window_start": _to_iso(window.start),
        "window_end": _to_iso(window.end),
    }


def _build_cost_items(
    rows: list[dict[str, Any]],
    *,
    window: AnalysisWindow,
    timezone_name: str,
    cost_report_module,
    modal_module,
) -> list[dict[str, Any]]:
    resources = _resource_shapes(modal_module)
    items: list[dict[str, Any]] = []
    for row in rows:
        function_name = _function_name_for_row(row)
        if not function_name:
            continue
        duration_field = _row_duration_field(function_name)
        duration_ms = row.get(duration_field)
        if not isinstance(duration_ms, (int, float)):
            continue
        timestamp_ms = _stage_ts_ms(row)
        if not (window.start_ms <= timestamp_ms < window.end_ms):
            continue
        resource = resources[function_name]
        raw_cost = cost_report_module._estimate_function_cost(
            duration_ms,
            cpu=float(resource["cpu"]),
            memory_mb=float(resource["memory_mb"]),
        )
        if raw_cost <= 0:
            continue
        items.append(
            {
                "event_id": str(row.get("event_id") or "").strip(),
                "message_id": str(row.get("message_id") or "").strip(),
                "session_key": str(row.get("session_key") or "").strip(),
                "function_name": function_name,
                "stage": str(row.get("stage") or "").strip(),
                "timestamp_ms": timestamp_ms,
                "hour_key": _hour_key_from_ms(timestamp_ms, timezone_name),
                "duration_ms": float(duration_ms),
                "worker_boot_id": str(row.get("worker_boot_id") or "").strip(),
                "container_reused": bool(row.get("container_reused")) if row.get("container_reused") is not None else None,
                "raw_estimated_cost_usd": raw_cost,
                "calibrated_cost_usd": None,
                "resource_shape": {
                    "cpu": float(resource["cpu"]),
                    "memory_mb": int(resource["memory_mb"]),
                },
            }
        )
    return items


def _build_official_hourly_costs(
    billing_summary: dict[str, Any],
    *,
    window: AnalysisWindow,
    timezone_name: str,
) -> dict[str, Decimal]:
    hourly = _empty_hourly_map(window)
    for entry in billing_summary.get("hourly_costs") or []:
        if not isinstance(entry, dict):
            continue
        interval_start = _parse_billing_interval(entry.get("interval_start") or "", timezone_name)
        if interval_start is None:
            continue
        key = _hour_key(interval_start)
        hourly[key] = hourly.get(key, Decimal("0")) + _decimal(entry.get("cost_usd"))
    return hourly


def _calibrate_cost_items_by_hour(
    items: list[dict[str, Any]],
    *,
    official_hourly_costs: dict[str, Decimal],
) -> dict[str, Any]:
    raw_hourly: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for item in items:
        raw_hourly[item["hour_key"]] += item["raw_estimated_cost_usd"]

    allocation_gaps: list[dict[str, Any]] = []
    calibration_ratio_by_hour: dict[str, Decimal | None] = {}
    for hour_key, official_cost in official_hourly_costs.items():
        raw_cost = raw_hourly.get(hour_key, Decimal("0"))
        if raw_cost > 0:
            if official_cost > 0:
                calibration_ratio_by_hour[hour_key] = official_cost / raw_cost
            else:
                calibration_ratio_by_hour[hour_key] = Decimal("1")
                allocation_gaps.append(
                    {
                        "hour_key": hour_key,
                        "raw_estimated_cost_usd": _decimal_str(raw_cost),
                        "reason": "official_cost_missing_using_raw_estimate",
                    }
                )
        else:
            calibration_ratio_by_hour[hour_key] = None
            if official_cost > 0:
                allocation_gaps.append(
                    {
                        "hour_key": hour_key,
                        "official_cost_usd": _decimal_str(official_cost),
                        "reason": "official_cost_without_matching_function_trace",
                    }
                )

    for item in items:
        ratio = calibration_ratio_by_hour.get(item["hour_key"])
        item["calibrated_cost_usd"] = item["raw_estimated_cost_usd"] * ratio if isinstance(ratio, Decimal) else None

    return {
        "items": items,
        "raw_hourly_costs": {key: _decimal_str(value) for key, value in sorted(raw_hourly.items())},
        "official_hourly_costs": {key: _decimal_str(value) for key, value in sorted(official_hourly_costs.items())},
        "calibration_ratio_by_hour": {
            key: _decimal_str(value) if isinstance(value, Decimal) else None
            for key, value in sorted(calibration_ratio_by_hour.items())
        },
        "allocation_gaps": allocation_gaps,
    }


def _session_id_for_event(event_id: str, message_id: str) -> str:
    normalized_event_id = str(event_id or "").strip()
    normalized_message_id = str(message_id or "").strip()
    if normalized_event_id:
        return normalized_event_id
    if normalized_message_id:
        return f"message:{normalized_message_id}"
    return "unknown"


def _unique_nonempty_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique_nonempty_strings(value)
    if isinstance(value, tuple):
        return _unique_nonempty_strings(list(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except Exception:
                return [stripped]
            if isinstance(parsed, list):
                return _unique_nonempty_strings(parsed)
        return _unique_nonempty_strings(part.strip() for part in stripped.split(","))
    return []


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if int(value) in {0, 1}:
            return bool(int(value))
        return None
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_hermes_request_summary(value: Any) -> dict[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return {}
    parsed: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        key, item_value = part.split("=", 1)
        normalized_key = str(key or "").strip()
        normalized_value = str(item_value or "").strip()
        if normalized_key and normalized_value:
            parsed[normalized_key] = normalized_value
    return parsed


def _read_receipt_metric_mode(metrics: dict[str, Any]) -> str:
    inbound_metric = metrics.get("t0_to_t2_ms") or {}
    if int(inbound_metric.get("count") or 0) > 0:
        return "inbound_message_read"
    return "insufficient_data"


def _read_receipt_metric_value(metrics: dict[str, Any], metric_name: str) -> Decimal:
    return _decimal(((metrics.get("t0_to_t2_ms") or {}).get(metric_name)))


def _apply_read_receipt_rows_to_sessions(
    sessions: list[dict[str, Any]],
    read_rows: list[dict[str, Any]],
    *,
    source_label: str,
) -> int:
    if not sessions or not read_rows:
        return 0

    sessions_by_reply_message_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sessions_by_inbound_message_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        for send_message_id in _unique_nonempty_strings(
            [session.get("reply_send_message_id")] + list(session.get("reply_send_message_ids") or [])
        ):
            sessions_by_reply_message_id[send_message_id].append(session)
        inbound_message_id = str(session.get("message_id") or "").strip()
        if inbound_message_id:
            sessions_by_inbound_message_id[inbound_message_id].append(session)

    for candidates in sessions_by_reply_message_id.values():
        candidates.sort(key=lambda item: int(item.get("t3_ms") or item.get("t0_ms") or 0))
    for candidates in sessions_by_inbound_message_id.values():
        candidates.sort(key=lambda item: int(item.get("t0_ms") or 0))

    applied = 0
    for row in sorted(read_rows, key=_stage_ts_ms):
        read_time_ms = _normalized_epoch_ms(row.get("read_time")) or _stage_ts_ms(row)
        if read_time_ms is None:
            continue
        for matched_message_id in _unique_nonempty_strings(row.get("message_id_list") or []):
            inbound_candidates = sessions_by_inbound_message_id.get(matched_message_id) or []
            reply_candidates = sessions_by_reply_message_id.get(matched_message_id) or []
            candidate_sessions = inbound_candidates or reply_candidates
            if not candidate_sessions:
                continue
            matched_via_inbound = bool(inbound_candidates)
            matched_session = None
            for session in candidate_sessions:
                session.setdefault("reply_to_read_ms", None)
                session.setdefault("reply_read_ms", None)
                session.setdefault("reply_read_match_message_id", "")
                session.setdefault("reply_read_match_source", "")
                t0_ms = session.get("t0_ms")
                t3_ms = session.get("t3_ms")
                if matched_via_inbound:
                    if isinstance(t0_ms, int) and t0_ms <= read_time_ms:
                        matched_session = session
                        break
                    continue
                if isinstance(t3_ms, int):
                    if t3_ms <= read_time_ms:
                        matched_session = session
                        break
                    continue
                if isinstance(t0_ms, int) and t0_ms <= read_time_ms:
                    matched_session = session
                    break
            if matched_session is None:
                continue
            if matched_via_inbound:
                current_t2_ms = matched_session.get("t2_ms")
                if isinstance(current_t2_ms, int) and current_t2_ms <= read_time_ms:
                    continue
                t3_ms = matched_session.get("t3_ms")
                matched_session["t2_ms"] = read_time_ms
                matched_session["read_matched"] = True
                matched_session["read_match_message_id"] = matched_message_id
                matched_session["read_match_source"] = source_label
                if source_label == "feishu_read_users_api":
                    matched_session["read_receipt_measurement_mode"] = "feishu_read_users_api"
                elif source_label == "cloudflare_worker_message_read":
                    matched_session["read_receipt_measurement_mode"] = "cloudflare_worker_webhook"
                else:
                    matched_session["read_receipt_measurement_mode"] = "webhook_message_read"
                if isinstance(matched_session.get("t0_ms"), int):
                    matched_session["t0_to_t2_ms"] = read_time_ms - int(matched_session["t0_ms"])
                if isinstance(t3_ms, int):
                    matched_session["t2_to_t3_ms"] = t3_ms - read_time_ms if read_time_ms <= t3_ms else None
            else:
                current_reply_read_ms = matched_session.get("reply_read_ms")
                if isinstance(current_reply_read_ms, int) and current_reply_read_ms <= read_time_ms:
                    continue
                t3_ms = matched_session.get("t3_ms")
                matched_session["reply_read_ms"] = read_time_ms
                matched_session["reply_read_match_message_id"] = matched_message_id
                matched_session["reply_read_match_source"] = source_label
                if isinstance(t3_ms, int) and read_time_ms >= t3_ms:
                    matched_session["reply_to_read_ms"] = read_time_ms - t3_ms
                elif isinstance(t3_ms, int):
                    matched_session["reply_to_read_ms"] = None
            applied += 1
            break
    return applied


def _build_session_timelines(
    rows: list[dict[str, Any]],
    *,
    window: AnalysisWindow,
    event_type: str,
    include_duplicates: bool,
    history_visible_mode: str,
) -> dict[str, Any]:
    rows_before_window_end = [row for row in rows if _stage_ts_ms(row) < window.end_ms]
    receive_rows_by_event_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    read_rows: list[dict[str, Any]] = []
    for row in rows_before_window_end:
        if not isinstance(row, dict):
            continue
        stage = str(row.get("stage") or "").strip()
        row_event_type = str(row.get("event_type") or "").strip()
        event_id = str(row.get("event_id") or "").strip()
        if stage == "webhook.message_read":
            read_rows.append(row)
            continue
        if row_event_type == event_type and event_id:
            receive_rows_by_event_id[event_id].append(row)

    read_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows:
        for message_id in row.get("message_id_list") or []:
            normalized = str(message_id or "").strip()
            if normalized:
                read_index[normalized].append(row)
    for entries in read_index.values():
        entries.sort(key=_stage_ts_ms)

    sessions: list[dict[str, Any]] = []
    session_key_map: dict[str, str] = {}
    for event_id, event_rows in receive_rows_by_event_id.items():
        event_rows.sort(key=_stage_ts_ms)
        ack_row = _pick_stage(event_rows, "webhook.ack")
        accepted_row = _pick_first_stage(event_rows, "webhook.accepted")
        internal_plan_row = _pick_stage(event_rows, "internal.agent_plan.done")
        internal_exec_start_row = _pick_first_stage(event_rows, "internal.agent_exec.start")
        internal_exec_done_row = _pick_stage(event_rows, "internal.agent_exec.done")
        if ack_row is not None and str(ack_row.get("ack_kind") or "").strip().lower() == "duplicate" and not include_duplicates:
            continue
        message_id = str(
            _coalesce(
                (accepted_row or {}).get("message_id"),
                (ack_row or {}).get("message_id"),
                next((row.get("message_id") for row in event_rows if str(row.get("message_id") or "").strip()), ""),
            )
            or ""
        ).strip()
        gateway_rows = [
            row
            for row in event_rows
            if str(row.get("stage") or "").strip() == "gateway.send.done" and bool(row.get("send_success"))
        ]
        gateway_rows.sort(key=_stage_ts_ms)
        gateway_row = gateway_rows[0] if gateway_rows else None
        reply_send_message_ids = _unique_nonempty_strings(row.get("send_message_id") for row in gateway_rows)
        reply_send_message_id = reply_send_message_ids[0] if reply_send_message_ids else ""
        ack_ts_ms = _stage_ts_ms(ack_row) if ack_row else None
        accepted_ts_ms = _stage_ts_ms(accepted_row) if accepted_row else None
        ack_elapsed_ms = ack_row.get("ack_elapsed_ms") if ack_row else None
        inferred_t0_ms = None
        if ack_ts_ms is not None and isinstance(ack_elapsed_ms, (int, float)):
            inferred_t0_ms = max(0, int(ack_ts_ms - float(ack_elapsed_ms)))
        t0_ms = inferred_t0_ms or accepted_ts_ms or ack_ts_ms
        measurement_mode = "feishu_full_chain"
        if t0_ms is None and internal_exec_start_row is not None:
            t0_ms = _stage_ts_ms(internal_exec_start_row)
            measurement_mode = "modal_internal_exec_only"
        if t0_ms is None or not (window.start_ms <= t0_ms < window.end_ms):
            continue
        read_candidates = []
        matched_read_message_id = ""
        matched_read_source = ""
        if message_id:
            candidates = [
                entry for entry in read_index.get(message_id, []) if _stage_ts_ms(entry) >= t0_ms and _stage_ts_ms(entry) < window.end_ms
            ]
            if candidates:
                read_candidates = candidates
                matched_read_message_id = message_id
                matched_read_source = "inbound_message_id"
        read_row = read_candidates[0] if read_candidates else None
        t2_ms = _normalized_epoch_ms((read_row or {}).get("read_time")) or (_stage_ts_ms(read_row) if read_row else None)
        t3_ms = _stage_ts_ms(gateway_row) if gateway_row else None
        if t3_ms is None and internal_exec_done_row is not None:
            t3_ms = _stage_ts_ms(internal_exec_done_row)
            measurement_mode = "modal_internal_exec_only"
        model_elapsed_ms_raw = (
            (internal_exec_done_row or {}).get("agent_model_elapsed_ms")
            if internal_exec_done_row is not None
            else None
        )
        model_elapsed_ms = (
            int(model_elapsed_ms_raw)
            if isinstance(model_elapsed_ms_raw, (int, float))
            else None
        )
        provider_wait_elapsed_ms_raw = (
            (internal_exec_done_row or {}).get("provider_wait_elapsed_ms")
            if internal_exec_done_row is not None
            else None
        )
        provider_wait_elapsed_ms = (
            int(provider_wait_elapsed_ms_raw)
            if isinstance(provider_wait_elapsed_ms_raw, (int, float))
            else None
        )
        effective_ai_elapsed_ms = provider_wait_elapsed_ms if isinstance(provider_wait_elapsed_ms, int) else model_elapsed_ms
        non_model_elapsed_ms_raw = (
            (internal_exec_done_row or {}).get("agent_non_model_elapsed_ms")
            if internal_exec_done_row is not None
            else None
        )
        non_model_elapsed_ms = (
            int(non_model_elapsed_ms_raw)
            if isinstance(non_model_elapsed_ms_raw, (int, float))
            else None
        )
        tool_exec_elapsed_ms_raw = (
            (internal_exec_done_row or {}).get("tool_exec_elapsed_ms")
            if internal_exec_done_row is not None
            else None
        )
        tool_exec_elapsed_ms = (
            int(tool_exec_elapsed_ms_raw)
            if isinstance(tool_exec_elapsed_ms_raw, (int, float))
            else None
        )
        gateway_run_agent_total_elapsed_ms_raw = (
            (internal_exec_done_row or {}).get("gateway_run_agent_total_elapsed_ms")
            if internal_exec_done_row is not None
            else None
        )
        gateway_run_agent_total_elapsed_ms = (
            int(gateway_run_agent_total_elapsed_ms_raw)
            if isinstance(gateway_run_agent_total_elapsed_ms_raw, (int, float))
            else None
        )
        hermes_overhead_elapsed_ms_raw = (
            (internal_exec_done_row or {}).get("hermes_overhead_elapsed_ms")
            if internal_exec_done_row is not None
            else None
        )
        hermes_overhead_elapsed_ms = (
            int(hermes_overhead_elapsed_ms_raw)
            if isinstance(hermes_overhead_elapsed_ms_raw, (int, float))
            else None
        )
        provider_wait_measurement_mode = str(
            ((internal_exec_done_row or {}).get("provider_wait_measurement_mode") or "")
        ).strip()
        if not provider_wait_measurement_mode:
            provider_wait_measurement_mode = (
                "provider_wait_observed"
                if isinstance(provider_wait_elapsed_ms, int)
                else ("legacy_proxy" if isinstance(model_elapsed_ms, int) else "insufficient_data")
            )
        session_key = str(
            _coalesce(
                (gateway_row or {}).get("session_key"),
                next((row.get("session_key") for row in event_rows if row.get("session_key")), ""),
                (internal_exec_done_row or {}).get("correlation_id"),
                (internal_exec_start_row or {}).get("correlation_id"),
            )
            or ""
        ).strip()
        session_id = _session_id_for_event(event_id, message_id)
        if session_key:
            session_key_map[session_key] = session_id
        sessions.append(
            {
                "session_id": session_id,
                "event_id": event_id,
                "message_id": message_id,
                "session_key": session_key,
                "reply_send_message_id": reply_send_message_id,
                "reply_send_message_ids": reply_send_message_ids,
                "history_visible_mode": history_visible_mode,
                "measurement_mode": measurement_mode,
                "t0_ms": t0_ms,
                "t1_ms": ack_ts_ms,
                "t2_ms": t2_ms,
                "reply_read_ms": None,
                "t3_ms": t3_ms,
                "ack_kind": str((ack_row or {}).get("ack_kind") or "").strip() or "unspecified",
                "ingress_strategy": str((ack_row or {}).get("ingress_strategy") or "").strip() or "unspecified",
                "execution_mode": str(
                    _coalesce(next((row.get("execution_mode") for row in event_rows if row.get("execution_mode")), ""))
                    or ""
                ).strip()
                or "unspecified",
                "reply_visible_proxy": (
                    "gateway.send.done:send_success=true"
                    if gateway_row
                    else ("internal.agent_exec.done" if internal_exec_done_row is not None else None)
                ),
                "t0_to_t1_ms": (ack_ts_ms - t0_ms) if ack_ts_ms is not None else None,
                "t0_to_t2_ms": (t2_ms - t0_ms) if t2_ms is not None else None,
                "t0_to_t3_ms": (t3_ms - t0_ms) if t3_ms is not None else None,
                "reply_to_read_ms": (t2_ms - t3_ms) if t2_ms is not None and t3_ms is not None and t2_ms >= t3_ms else None,
                "t0_to_t3_minus_model_ms": (
                    max((t3_ms - t0_ms) - effective_ai_elapsed_ms, 0)
                    if t3_ms is not None and isinstance(effective_ai_elapsed_ms, int)
                    else None
                ),
                "t1_to_t3_ms": (t3_ms - ack_ts_ms) if t3_ms is not None and ack_ts_ms is not None else None,
                "t2_to_t3_ms": (t3_ms - t2_ms) if t3_ms is not None and t2_ms is not None and t2_ms <= t3_ms else None,
                "ai_model_elapsed_ms": effective_ai_elapsed_ms,
                "provider_wait_elapsed_ms": provider_wait_elapsed_ms,
                "tool_exec_elapsed_ms": tool_exec_elapsed_ms,
                "gateway_run_agent_total_elapsed_ms": gateway_run_agent_total_elapsed_ms,
                "hermes_overhead_elapsed_ms": hermes_overhead_elapsed_ms,
                "reply_minus_ai_measurement_mode": provider_wait_measurement_mode,
                "non_model_elapsed_ms": non_model_elapsed_ms,
                "provider_billed_cost_usd": _coalesce(
                    ((internal_exec_done_row or {}).get("provider_usage_totals") or {}).get("billed_cost_usd"),
                    ((internal_exec_done_row or {}).get("provider_usage") or {}).get("cost"),
                ),
                "external_exec_candidate": bool((internal_plan_row or {}).get("external_exec_candidate"))
                if internal_plan_row is not None
                else None,
                "route_hint": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("route_hint"),
                        (internal_plan_row or {}).get("route_hint"),
                        next((row.get("route_hint") for row in event_rows if row.get("route_hint")), ""),
                    )
                    or ""
                ).strip(),
                "route_version": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("route_version"),
                        (internal_plan_row or {}).get("route_version"),
                        next((row.get("route_version") for row in event_rows if row.get("route_version")), ""),
                    )
                    or ""
                ).strip(),
                "provider_alias": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("provider_alias"),
                        (internal_plan_row or {}).get("provider_alias"),
                        next((row.get("provider_alias") for row in event_rows if row.get("provider_alias")), ""),
                    )
                    or ""
                ).strip(),
                "model_catalog_version": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("model_catalog_version"),
                        (internal_plan_row or {}).get("model_catalog_version"),
                        next((row.get("model_catalog_version") for row in event_rows if row.get("model_catalog_version")), ""),
                    )
                    or ""
                ).strip(),
                "request_class": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("request_class"),
                        (internal_plan_row or {}).get("request_class"),
                        next((row.get("request_class") for row in event_rows if row.get("request_class")), ""),
                    )
                    or ""
                ).strip(),
                "content_modalities": _coerce_string_list(
                    _coalesce(
                        (internal_exec_done_row or {}).get("content_modalities"),
                        (internal_plan_row or {}).get("content_modalities"),
                        next((row.get("content_modalities") for row in event_rows if row.get("content_modalities")), []),
                    )
                ),
                "route_family": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("route_family"),
                        (internal_plan_row or {}).get("route_family"),
                        next((row.get("route_family") for row in event_rows if row.get("route_family")), ""),
                    )
                    or ""
                ).strip(),
                "gateway_route_name": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("gateway_route_name"),
                        (internal_plan_row or {}).get("gateway_route_name"),
                        next((row.get("gateway_route_name") for row in event_rows if row.get("gateway_route_name")), ""),
                    )
                    or ""
                ).strip(),
                "gateway_eligible": (
                    bool(_coalesce(
                        (internal_exec_done_row or {}).get("gateway_eligible"),
                        (internal_plan_row or {}).get("gateway_eligible"),
                    ))
                    if _coalesce(
                        (internal_exec_done_row or {}).get("gateway_eligible"),
                        (internal_plan_row or {}).get("gateway_eligible"),
                    ) is not None
                    else None
                ),
                "requires_tools": (
                    bool(_coalesce(
                        (internal_exec_done_row or {}).get("requires_tools"),
                        (internal_plan_row or {}).get("requires_tools"),
                    ))
                    if _coalesce(
                        (internal_exec_done_row or {}).get("requires_tools"),
                        (internal_plan_row or {}).get("requires_tools"),
                    ) is not None
                    else None
                ),
                "requires_browser": (
                    bool(_coalesce(
                        (internal_exec_done_row or {}).get("requires_browser"),
                        (internal_plan_row or {}).get("requires_browser"),
                    ))
                    if _coalesce(
                        (internal_exec_done_row or {}).get("requires_browser"),
                        (internal_plan_row or {}).get("requires_browser"),
                    ) is not None
                    else None
                ),
                "requires_media_hydration": (
                    bool(_coalesce(
                        (internal_exec_done_row or {}).get("requires_media_hydration"),
                        (internal_plan_row or {}).get("requires_media_hydration"),
                    ))
                    if _coalesce(
                        (internal_exec_done_row or {}).get("requires_media_hydration"),
                        (internal_plan_row or {}).get("requires_media_hydration"),
                    ) is not None
                    else None
                ),
                "toolset": _coerce_string_list(
                    _coalesce(
                        (internal_exec_done_row or {}).get("toolset"),
                        (internal_plan_row or {}).get("toolset"),
                        next((row.get("toolset") for row in event_rows if row.get("toolset")), []),
                    )
                ),
                "modality_profile": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("modality_profile"),
                        (internal_plan_row or {}).get("modality_profile"),
                        next((row.get("modality_profile") for row in event_rows if row.get("modality_profile")), ""),
                    )
                    or ""
                ).strip(),
                "gateway_error_class": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("gateway_error_class"),
                        next((row.get("gateway_error_class") for row in event_rows if row.get("gateway_error_class")), ""),
                    )
                    or ""
                ).strip(),
                "misroute_detected": any(
                    bool(row.get("misroute_detected"))
                    or str(row.get("gateway_error_class") or "").strip() == "misrouted_request_class"
                    or str(row.get("fallback_reason") or "").strip() == "misrouted_request_class"
                    for row in event_rows
                ),
                "route_decision_reason": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("route_decision_reason"),
                        (internal_plan_row or {}).get("route_decision_reason"),
                        next((row.get("route_decision_reason") for row in event_rows if row.get("route_decision_reason")), ""),
                    )
                    or ""
                ).strip(),
                "fallback_reason": str(
                    _coalesce(
                        (internal_exec_done_row or {}).get("fallback_reason"),
                        next((row.get("fallback_reason") for row in event_rows if row.get("fallback_reason")), ""),
                    )
                    or ""
                ).strip(),
                "cache_eligible": _coerce_optional_bool(
                    _coalesce(
                        (internal_exec_done_row or {}).get("cache_eligible"),
                        (internal_plan_row or {}).get("cache_eligible"),
                        next((row.get("cache_eligible") for row in event_rows if row.get("cache_eligible") is not None), None),
                    )
                ),
                "feedback_score_before": _coalesce(
                    (internal_exec_done_row or {}).get("feedback_score_before"),
                    (internal_plan_row or {}).get("feedback_score_before"),
                    next((row.get("feedback_score_before") for row in event_rows if row.get("feedback_score_before") not in (None, "")), None),
                ),
                "feedback_score_after": _coalesce(
                    (internal_exec_done_row or {}).get("feedback_score_after"),
                    (internal_plan_row or {}).get("feedback_score_after"),
                    next((row.get("feedback_score_after") for row in event_rows if row.get("feedback_score_after") not in (None, "")), None),
                ),
                "ai_call_count": _coerce_optional_int(
                    _coalesce(
                        (internal_exec_done_row or {}).get("ai_call_count"),
                        (internal_plan_row or {}).get("ai_call_count"),
                        next((row.get("ai_call_count") for row in event_rows if row.get("ai_call_count") is not None), None),
                    )
                ),
                "capability_match": _coerce_optional_bool(
                    _coalesce(
                        (internal_exec_done_row or {}).get("capability_match"),
                        (internal_plan_row or {}).get("capability_match"),
                        next((row.get("capability_match") for row in event_rows if row.get("capability_match") is not None), None),
                    )
                ),
                "preferred_model_selected": _coerce_optional_bool(
                    _coalesce(
                        (internal_exec_done_row or {}).get("preferred_model_selected"),
                        (internal_plan_row or {}).get("preferred_model_selected"),
                        next((row.get("preferred_model_selected") for row in event_rows if row.get("preferred_model_selected") is not None), None),
                    )
                ),
                "read_matched": read_row is not None,
                "read_match_message_id": matched_read_message_id,
                "read_match_source": matched_read_source,
                "read_receipt_measurement_mode": "webhook_message_read" if read_row is not None else "",
                "reply_read_match_message_id": "",
                "reply_read_match_source": "",
                "reply_sent": gateway_row is not None or internal_exec_done_row is not None,
            }
        )

    if sessions and read_rows:
        _apply_read_receipt_rows_to_sessions(sessions, read_rows, source_label="webhook_message_read")

    metrics: dict[str, dict[str, Any]] = {}
    for metric_name in (
        "t0_to_t1_ms",
        "t0_to_t2_ms",
        "t0_to_t3_ms",
        "reply_to_read_ms",
        "t0_to_t3_minus_model_ms",
        "t1_to_t3_ms",
        "t2_to_t3_ms",
    ):
        metrics[metric_name] = _summarize_numeric_series(
            session.get(metric_name) for session in sessions if isinstance(session.get(metric_name), (int, float))
        )

    completion = {
        "session_count": len(sessions),
        "read_receipt_count": sum(1 for session in sessions if session.get("read_matched")),
        "reply_sent_count": sum(1 for session in sessions if session.get("reply_sent")),
        "no_read_count": sum(1 for session in sessions if not session.get("read_matched")),
        "no_reply_count": sum(1 for session in sessions if not session.get("reply_sent")),
    }
    completion["read_receipt_rate"] = _safe_ratio(completion["read_receipt_count"], completion["session_count"])
    completion["reply_sent_rate"] = _safe_ratio(completion["reply_sent_count"], completion["session_count"])
    slowest = sorted(
        [session for session in sessions if isinstance(session.get("t0_to_t3_ms"), (int, float))],
        key=lambda session: float(session.get("t0_to_t3_ms") or 0),
        reverse=True,
    )[:10]
    return {
        "sessions": sessions,
        "metrics": metrics,
        "completion": completion,
        "slowest_sessions": slowest,
        "session_key_map": session_key_map,
    }


def _attach_session_costs(cost_items: list[dict[str, Any]], sessions: list[dict[str, Any]]) -> None:
    session_costs: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    message_to_session: dict[str, str] = {}
    for session in sessions:
        session_id = str(session.get("session_id") or "").strip()
        message_id = str(session.get("message_id") or "").strip()
        if session_id and message_id:
            message_to_session[message_id] = session_id
    for item in cost_items:
        session_id = _session_id_for_event(str(item.get("event_id") or ""), str(item.get("message_id") or ""))
        message_session_id = message_to_session.get(str(item.get("message_id") or "").strip())
        if message_session_id and session_id.startswith("message:"):
            session_id = message_session_id
        item["session_id"] = session_id
        item_cost = item.get("calibrated_cost_usd")
        if isinstance(item_cost, Decimal):
            session_costs[session_id] += item_cost
    for session in sessions:
        session["allocated_cost_usd"] = _decimal_str(session_costs.get(str(session.get("session_id") or ""), Decimal("0")))


def _rebuild_session_total_costs(sessions: list[dict[str, Any]]) -> None:
    for session in sessions:
        total = Decimal("0")
        has_cost = False
        for field in ("allocated_cost_usd", "provider_billed_cost_usd", "cloudflare_ai_cost_usd"):
            value = session.get(field)
            if value in (None, ""):
                continue
            total += _decimal(value)
            has_cost = True
        session["total_session_cost_usd"] = _decimal_str(total) if has_cost else None


def _summarize_functions(cost_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_function: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in cost_items:
        grouped[str(item.get("function_name") or "").strip()].append(item)
    for function_name, items in grouped.items():
        calibrated_total = sum(
            (item["calibrated_cost_usd"] for item in items if isinstance(item.get("calibrated_cost_usd"), Decimal)),
            Decimal("0"),
        )
        raw_total = sum((item["raw_estimated_cost_usd"] for item in items), Decimal("0"))
        session_ids = {str(item.get("session_id") or "").strip() for item in items if str(item.get("session_id") or "").strip()}
        reused_samples = [item.get("container_reused") for item in items if item.get("container_reused") is not None]
        cold_starts = sum(1 for value in reused_samples if value is False)
        duration_summary = _summarize_numeric_series(item.get("duration_ms") for item in items)
        invocation_count = len(items)
        by_function[function_name] = {
            "function_name": function_name,
            "invocation_count": invocation_count,
            "session_count": len(session_ids),
            "raw_estimated_cost_usd": _decimal_str(raw_total),
            "calibrated_cost_usd": _decimal_str(calibrated_total),
            "raw_cost_per_invocation_usd": _decimal_str(raw_total / invocation_count) if invocation_count else None,
            "cost_per_session_usd": _decimal_str(calibrated_total / len(session_ids)) if session_ids else None,
            "avg_duration_ms": duration_summary.get("avg"),
            "p90_duration_ms": duration_summary.get("p90"),
            "cold_start_rate": _safe_ratio(cold_starts, len(reused_samples)) if reused_samples else None,
            "worker_boot_id_count": len(
                {str(item.get("worker_boot_id") or "").strip() for item in items if str(item.get("worker_boot_id") or "").strip()}
            ),
        }
    return by_function


def _summarize_containers(cost_items: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in cost_items:
        worker_boot_id = str(item.get("worker_boot_id") or "").strip()
        if worker_boot_id:
            grouped[worker_boot_id].append(item)

    containers: list[dict[str, Any]] = []
    by_function: dict[str, dict[str, Any]] = {}
    for worker_boot_id, items in grouped.items():
        function_name = str(_coalesce(*(item.get("function_name") for item in items)) or "unspecified")
        calibrated_total = sum(
            (item["calibrated_cost_usd"] for item in items if isinstance(item.get("calibrated_cost_usd"), Decimal)),
            Decimal("0"),
        )
        raw_total = sum((item["raw_estimated_cost_usd"] for item in items), Decimal("0"))
        session_ids = {str(item.get("session_id") or "").strip() for item in items if str(item.get("session_id") or "").strip()}
        cold_start_count = sum(1 for item in items if item.get("container_reused") is False)
        duration_summary = _summarize_numeric_series(item.get("duration_ms") for item in items)
        containers.append(
            {
                "worker_boot_id": worker_boot_id,
                "function_name": function_name,
                "invocation_count": len(items),
                "session_count": len(session_ids),
                "cold_start_count": cold_start_count,
                "cold_start_rate": _safe_ratio(cold_start_count, len(items)),
                "total_cost_usd": _decimal_str(calibrated_total),
                "raw_estimated_cost_usd": _decimal_str(raw_total),
                "avg_duration_ms": duration_summary.get("avg"),
                "p90_duration_ms": duration_summary.get("p90"),
                "first_seen_ts_ms": min(int(item.get("timestamp_ms") or 0) for item in items),
                "last_seen_ts_ms": max(int(item.get("timestamp_ms") or 0) for item in items),
                "single_use_container": len(items) == 1,
            }
        )

    by_function_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for container in containers:
        by_function_grouped[str(container.get("function_name") or "").strip()].append(container)

    for function_name, entries in by_function_grouped.items():
        total_cost = sum((_decimal(entry.get("total_cost_usd")) for entry in entries), Decimal("0"))
        invocation_total = sum(int(entry.get("invocation_count") or 0) for entry in entries)
        cold_total = sum(int(entry.get("cold_start_count") or 0) for entry in entries)
        single_use_total = sum(1 for entry in entries if entry.get("single_use_container"))
        by_function[function_name] = {
            "function_name": function_name,
            "container_count": len(entries),
            "invocation_count": invocation_total,
            "cold_start_rate": _safe_ratio(cold_total, invocation_total),
            "single_use_container_rate": _safe_ratio(single_use_total, len(entries)),
            "total_cost_usd": _decimal_str(total_cost),
        }

    containers.sort(key=lambda item: (_decimal(item.get("total_cost_usd")), int(item.get("invocation_count") or 0)), reverse=True)
    return {
        "containers": containers,
        "hot_containers": containers[:10],
        "fleet_summary": {
            "container_count": len(containers),
            "single_use_container_rate": _safe_ratio(sum(1 for entry in containers if entry.get("single_use_container")), len(containers))
            if containers
            else None,
            "cold_start_rate": _safe_ratio(
                sum(int(entry.get("cold_start_count") or 0) for entry in containers),
                sum(int(entry.get("invocation_count") or 0) for entry in containers),
            )
            if containers
            else None,
            "total_cost_usd": _decimal_str(sum((_decimal(entry.get("total_cost_usd")) for entry in containers), Decimal("0"))),
        },
        "by_function": by_function,
    }


def _hourly_session_count(sessions: list[dict[str, Any]], timezone_name: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for session in sessions:
        t0_ms = session.get("t0_ms")
        if isinstance(t0_ms, (int, float)):
            counts[_hour_key_from_ms(int(t0_ms), timezone_name)] += 1
    return counts


def _build_execution_path_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plan_rows = [row for row in rows if str(row.get("stage") or "").strip() == "internal.agent_plan.done"]
    exec_rows = [row for row in rows if str(row.get("stage") or "").strip() == "internal.agent_exec.done"]
    send_rows = [
        row
        for row in rows
        if str(row.get("stage") or "").strip() == "gateway.send.done" and bool(row.get("send_success"))
    ]
    exec_event_ids = {str(row.get("event_id") or "").strip() for row in exec_rows if str(row.get("event_id") or "").strip()}
    send_event_ids = {str(row.get("event_id") or "").strip() for row in send_rows if str(row.get("event_id") or "").strip()}
    candidate_plan_rows = [row for row in plan_rows if bool(row.get("external_exec_candidate"))]
    candidate_event_ids = {
        str(row.get("event_id") or "").strip() for row in candidate_plan_rows if str(row.get("event_id") or "").strip()
    }
    browser_first_count = sum(1 for row in plan_rows if str(row.get("route_hint") or "").strip() == "cf_browser_first")
    candidate_modal_fallback_count = sum(1 for event_id in candidate_event_ids if event_id in exec_event_ids)
    candidate_send_visible_count = sum(1 for event_id in candidate_event_ids if event_id in send_event_ids)
    return {
        "plan_count": len(plan_rows),
        "external_exec_candidate_count": len(candidate_event_ids),
        "external_exec_candidate_modal_fallback_count": candidate_modal_fallback_count,
        "external_exec_candidate_visible_send_count": candidate_send_visible_count,
        "external_exec_candidate_modal_fallback_rate": _safe_ratio(candidate_modal_fallback_count, len(candidate_event_ids))
        if candidate_event_ids
        else None,
        "browser_first_route_count": browser_first_count,
    }


def _parse_cloudflare_log_line(raw_line: str) -> dict[str, Any] | None:
    line = str(raw_line or "").strip()
    if not line:
        return None
    candidates = [line]
    brace_index = line.find("{")
    if brace_index >= 0:
        candidates.append(line[brace_index:])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _iter_cloudflare_log_records(raw_line: str, timezone_name: str) -> list[dict[str, Any]]:
    parsed = _parse_cloudflare_log_line(raw_line)
    if not isinstance(parsed, dict):
        return []
    if isinstance(parsed.get("event"), str) and parsed.get("event"):
        return [parsed]

    logs = parsed.get("logs")
    if not isinstance(logs, list):
        return []

    request = ((parsed.get("event") or {}).get("request") or {}) if isinstance(parsed.get("event"), dict) else {}
    response = ((parsed.get("event") or {}).get("response") or {}) if isinstance(parsed.get("event"), dict) else {}
    script_name = str(parsed.get("scriptName") or "").strip()
    script_version = ((parsed.get("scriptVersion") or {}).get("id") or "") if isinstance(parsed.get("scriptVersion"), dict) else ""
    execution_model = str(parsed.get("executionModel") or "").strip()
    event_timestamp_ms = _extract_cloudflare_timestamp_ms(parsed, timezone_name)

    records: list[dict[str, Any]] = []
    for item in logs:
        if not isinstance(item, dict):
            continue
        item_timestamp_ms = _normalized_epoch_ms(item.get("timestamp")) or event_timestamp_ms
        messages = item.get("message")
        if not isinstance(messages, list):
            continue
        for raw_message in messages:
            nested = _parse_cloudflare_log_line(str(raw_message or ""))
            if not isinstance(nested, dict) or not nested.get("event"):
                continue
            if item_timestamp_ms is not None and nested.get("timestamp_ms") is None:
                nested["timestamp_ms"] = item_timestamp_ms
            nested.setdefault("script_name", script_name)
            nested.setdefault("script_version_id", str(script_version or "").strip())
            nested.setdefault("execution_model", execution_model)
            if isinstance(request, dict):
                nested.setdefault("request_url", request.get("url"))
                nested.setdefault("request_method", request.get("method"))
            if isinstance(response, dict):
                nested.setdefault("response_status", response.get("status"))
            nested.setdefault("wrangler_log_level", item.get("level"))
            records.append(nested)
    return records


def _extract_cloudflare_timestamp_ms(record: dict[str, Any], timezone_name: str) -> int | None:
    for field in ("timestamp_ms", "ts_ms", "ts", "timestamp", "time"):
        epoch_ms = _normalized_epoch_ms(record.get(field))
        if epoch_ms is not None:
            return epoch_ms
    for field in ("timestamp", "time", "datetime"):
        value = record.get(field)
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        tz = _get_timezone(timezone_name)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return int(parsed.astimezone(tz).timestamp() * 1000)
    return None


def _discover_cloudflare_log_paths(repo_root: Path, explicit_paths: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in explicit_paths:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            discovered.append(path.resolve())
    patterns = [
        ".tmp-cf*.log",
        ".tmp-cf*.txt",
        ".tmp-cf*.jsonl",
        ".tmp-cloudflare*.log",
        ".tmp-cloudflare*.txt",
        ".tmp-cloudflare*.jsonl",
        ".tmp-wrangler*.log",
        ".tmp-wrangler*.txt",
        ".tmp-wrangler*.jsonl",
    ]
    for pattern in patterns:
        for candidate in repo_root.glob(pattern):
            if candidate.is_file():
                discovered.append(candidate.resolve())
    return sorted({path for path in discovered})


def _iter_json_stream_documents(raw_text: str) -> Iterable[Any]:
    text = str(raw_text or "")
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            payload, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            next_newline = text.find("\n", index)
            if next_newline < 0:
                break
            index = next_newline + 1
            continue
        yield payload
        index = next_index


def _read_cloudflare_records(paths: list[Path], timezone_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    gaps: list[str] = []
    for path in paths:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            gaps.append(f"failed_to_read_cloudflare_log:{path}")
            continue
        json_documents = list(_iter_json_stream_documents(raw_text))
        if json_documents:
            lines = [json.dumps(payload, ensure_ascii=False, separators=(",", ":")) for payload in json_documents]
        else:
            lines = raw_text.splitlines()
        for line in lines:
            for parsed in _iter_cloudflare_log_records(line, timezone_name):
                if not isinstance(parsed, dict):
                    continue
                if not parsed.get("event"):
                    continue
                parsed["__source_path"] = str(path)
                parsed["__timestamp_ms"] = _extract_cloudflare_timestamp_ms(parsed, timezone_name)
                records.append(parsed)
    return records, gaps


def _infer_cloudflare_gateway_id() -> str:
    base_url = str(os.getenv("CLOUDFLARE_AI_GATEWAY_BASE_URL") or "").strip()
    if base_url:
        path_parts = [part for part in urlparse(base_url).path.split("/") if part]
        if path_parts:
            return path_parts[-1]
    return "affiliate-manager"


def _infer_cloudflare_worker_script_name(repo_root: Path) -> str:
    configured = str(os.getenv("CLOUDFLARE_WORKER_SCRIPT_NAME") or "").strip()
    if configured:
        return configured
    wrangler_path = repo_root / "cloudflare" / "feishu-gateway" / "wrangler.jsonc"
    try:
        raw = wrangler_path.read_text(encoding="utf-8")
    except OSError:
        return "hermes-feishu-gateway"
    match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    if match:
        return str(match.group(1) or "").strip() or "hermes-feishu-gateway"
    return "hermes-feishu-gateway"


def _event_id_from_correlation_id(correlation_id: Any) -> str:
    raw = str(correlation_id or "").strip()
    if not raw:
        return ""
    parts = [part for part in raw.split(":") if part]
    return str(parts[-1] or "").strip() if parts else ""


def _fetch_cloudflare_worker_observability_records(
    *,
    repo_root: Path,
    timezone_name: str,
    windows: dict[str, AnalysisWindow],
) -> tuple[list[dict[str, Any]], list[str]]:
    token = str(os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
    account_id = str(os.getenv("CLOUDFLARE_ACCOUNT_ID") or "d1215a30b84b673ef0367010b0e78c10").strip()
    script_name = _infer_cloudflare_worker_script_name(repo_root)
    if not token:
        return [], ["cloudflare_api_token_missing"]

    earliest_start = min(window.start_ms for window in windows.values())
    latest_end = max(window.end_ms for window in windows.values())
    request_body = {
        "queryId": "feishu-triparty-pk-report",
        "view": "events",
        "limit": 2000,
        "timeframe": {"from": earliest_start, "to": latest_end},
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
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return [], [f"cloudflare_worker_api_fetch_failed:{exc}"]

    events = (((payload if isinstance(payload, dict) else {}).get("result") or {}).get("events") or {}).get("events") or []
    records: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        source = dict(source) if isinstance(source, dict) else {}
        if not source.get("event"):
            continue
        timestamp_ms = _normalized_epoch_ms(item.get("timestamp"))
        if timestamp_ms is None or not (earliest_start <= timestamp_ms < latest_end):
            continue
        record = dict(source)
        record["event"] = str(source.get("event") or "").strip()
        record["timestamp_ms"] = timestamp_ms
        record["__timestamp_ms"] = timestamp_ms
        record["__source_path"] = "cloudflare_api:worker_observability"
        derived_event_id = str(record.get("event_id") or "").strip() or _event_id_from_correlation_id(record.get("correlation_id"))
        if derived_event_id:
            record["derived_event_id"] = derived_event_id
        records.append(record)
    return records, []


def _fetch_cloudflare_ai_gateway_records(
    *,
    timezone_name: str,
    windows: dict[str, AnalysisWindow],
) -> tuple[list[dict[str, Any]], list[str]]:
    token = str(os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
    account_id = str(os.getenv("CLOUDFLARE_ACCOUNT_ID") or "d1215a30b84b673ef0367010b0e78c10").strip()
    gateway_id = _infer_cloudflare_gateway_id()
    if not token:
        return [], ["cloudflare_api_token_missing"]

    earliest_start = min(window.start for window in windows.values())
    latest_end = max(window.end for window in windows.values())
    records: list[dict[str, Any]] = []
    for page in range(1, 9):
        params = {
            "per_page": 50,
            "page": page,
            "order_by": "created_at",
            "order_by_direction": "desc",
        }
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai-gateway/gateways/{gateway_id}/logs?"
            f"{urlencode(params)}"
        )
        request = Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return records, [f"cloudflare_api_fetch_failed:{exc}"]

        result = payload.get("result") if isinstance(payload, dict) else []
        if not isinstance(result, list) or not result:
            break

        page_oldest_ts_ms: int | None = None
        for item in result:
            if not isinstance(item, dict):
                continue
            created_ts_ms = _extract_cloudflare_timestamp_ms({"timestamp": item.get("created_at")}, timezone_name)
            if created_ts_ms is None:
                continue
            page_oldest_ts_ms = created_ts_ms if page_oldest_ts_ms is None else min(page_oldest_ts_ms, created_ts_ms)
            if not (int(earliest_start.timestamp() * 1000) <= created_ts_ms < int(latest_end.timestamp() * 1000)):
                continue
            metadata = item.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            hermes_request_summary = _parse_hermes_request_summary(metadata.get("hermes_request"))
            record = {
                "event": "feishu.cf_ai_gateway.log",
                "timestamp": item.get("created_at"),
                "correlation_id": metadata.get("correlation_id"),
                "session_key": metadata.get("session_key"),
                "feishu_event_id": metadata.get("feishu_event_id"),
                "request_class": metadata.get("request_class"),
                "route_family": hermes_request_summary.get("rf"),
                "gateway_route_name": hermes_request_summary.get("rt"),
                "gateway_eligible": hermes_request_summary.get("ge") == "1" if "ge" in hermes_request_summary else None,
                "modality_profile": hermes_request_summary.get("mp"),
                "content_modalities": hermes_request_summary.get("mp", "").split("+") if hermes_request_summary.get("mp") else [],
                "requires_tools": hermes_request_summary.get("tb") == "1" if "tb" in hermes_request_summary else None,
                "requires_browser": hermes_request_summary.get("br") == "1" if "br" in hermes_request_summary else None,
                "requires_media_hydration": hermes_request_summary.get("mh") == "1" if "mh" in hermes_request_summary else None,
                "hermes_request": metadata.get("hermes_request"),
                "toolset": [] if hermes_request_summary.get("ts") in (None, "", "none") else hermes_request_summary.get("ts", "").split("+"),
                "reason_code": hermes_request_summary.get("rs"),
                "provider": item.get("provider"),
                "model": item.get("model"),
                "success": item.get("success"),
                "status_code": item.get("status_code"),
                "cf_ai_exec_elapsed_ms": item.get("duration"),
                "cf_cache_status": item.get("cached"),
                "cost": item.get("cost"),
                "route_version": metadata.get("route_version"),
                "provider_alias": metadata.get("provider_alias"),
                "model_catalog_version": metadata.get("model_catalog_version"),
                "feedback_score_before": metadata.get("feedback_score_before"),
                "feedback_score_after": metadata.get("feedback_score_after"),
                "ai_call_count": _coerce_optional_int(metadata.get("ai_call_count")),
                "cache_eligible": _coerce_optional_bool(metadata.get("cache_eligible")),
                "capability_match": _coerce_optional_bool(metadata.get("capability_match")),
                "preferred_model_selected": _coerce_optional_bool(metadata.get("preferred_model_selected")),
                "__timestamp_ms": created_ts_ms,
                "__source_path": f"cloudflare_api:{gateway_id}",
            }
            records.append(record)
        if page_oldest_ts_ms is not None and page_oldest_ts_ms < int(earliest_start.timestamp() * 1000):
            break
    return records, []


def _build_cloudflare_summary(
    *,
    repo_root: Path,
    explicit_paths: list[str],
    windows: dict[str, AnalysisWindow],
    timezone_name: str,
    session_key_map: dict[str, str],
    event_id_map: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    log_paths = _discover_cloudflare_log_paths(repo_root, explicit_paths)
    records: list[dict[str, Any]] = []
    gaps: list[str] = []
    if log_paths:
        records, gaps = _read_cloudflare_records(log_paths, timezone_name)
    api_records, api_gaps = _fetch_cloudflare_ai_gateway_records(
        timezone_name=timezone_name,
        windows=windows,
    )
    worker_records, worker_gaps = _fetch_cloudflare_worker_observability_records(
        repo_root=repo_root,
        timezone_name=timezone_name,
        windows=windows,
    )
    records.extend(api_records)
    records.extend(worker_records)
    gaps.extend(api_gaps)
    gaps.extend(worker_gaps)
    if not log_paths and not api_records and not worker_records:
        return (
            {
                "status": "missing",
                "join_level": "none",
                "log_paths": [],
                "window_summaries": {},
                "notes": [
                    "No Cloudflare historical log files were discovered locally, and neither the AI Gateway API nor Worker Observability API returned usable records."
                ],
            },
            ["cloudflare_logs_missing"] + gaps,
        )
    if not records:
        return (
            {
                "status": "degraded",
                "join_level": "none",
                "log_paths": [str(path) for path in log_paths],
                "window_summaries": {},
                "notes": ["Cloudflare sources were found, but no parseable event records were available."],
            },
            gaps + ["cloudflare_logs_unparseable"],
        )

    matched_records = 0
    join_level = "interval_only"
    for record in records:
        matched_event_id = (
            str(record.get("feishu_event_id") or "").strip()
            or str(record.get("event_id") or "").strip()
            or str(record.get("derived_event_id") or "").strip()
            or _event_id_from_correlation_id(record.get("correlation_id"))
        )
        if matched_event_id and matched_event_id in event_id_map:
            record["matched_session_id"] = event_id_map[matched_event_id]
            record["matched_event_id"] = matched_event_id
            matched_records += 1
            join_level = "event_level"
            continue
        session_key = str(record.get("session_key") or "").strip()
        if session_key and session_key in session_key_map:
            record["matched_session_id"] = session_key_map[session_key]
            matched_records += 1
            join_level = "event_level"

    window_summaries: dict[str, Any] = {}
    bootstrapped_worker_fact_count = 0
    for label, window in windows.items():
        window_records = [
            record
            for record in records
            if isinstance(record.get("__timestamp_ms"), int) and window.start_ms <= int(record["__timestamp_ms"]) < window.end_ms
        ]
        event_counts: dict[str, int] = defaultdict(int)
        exec_latencies: list[float] = []
        send_latencies: list[float] = []
        gateway_costs: list[Decimal] = []
        session_model_elapsed: dict[str, float] = defaultdict(float)
        matched_session_gateway_costs: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        worker_session_facts: dict[str, dict[str, Any]] = {}
        read_event_facts: list[dict[str, Any]] = []
        failures = 0
        for record in window_records:
            event = str(record.get("event") or "").strip()
            event_counts[event] += 1
            if isinstance(record.get("cf_ai_exec_elapsed_ms"), (int, float)):
                exec_latencies.append(float(record["cf_ai_exec_elapsed_ms"]))
                matched_session_id = str(record.get("matched_session_id") or "").strip()
                if matched_session_id:
                    session_model_elapsed[matched_session_id] += float(record["cf_ai_exec_elapsed_ms"])
            if isinstance(record.get("cf_send_elapsed_ms"), (int, float)):
                send_latencies.append(float(record["cf_send_elapsed_ms"]))
            cost_value = record.get("cost")
            if cost_value not in (None, ""):
                gateway_cost = _decimal(cost_value)
                if gateway_cost >= 0:
                    gateway_costs.append(gateway_cost)
                    matched_session_id = str(record.get("matched_session_id") or "").strip()
                    if matched_session_id:
                        matched_session_gateway_costs[matched_session_id] += gateway_cost
            if event.endswith(".error") or event.endswith(".failed") or "fallback" in event:
                failures += 1
            if event == "feishu.message_read.accepted":
                read_event_facts.append(
                    {
                        "event": event,
                        "event_id": str(record.get("event_id") or "").strip(),
                        "correlation_id": str(record.get("correlation_id") or "").strip(),
                        "__timestamp_ms": record.get("__timestamp_ms"),
                        "ts": record.get("ts"),
                        "read_time": record.get("read_time"),
                        "message_id_list": list(record.get("message_id_list") or []),
                    }
                )
            worker_event_id = (
                str(record.get("event_id") or "").strip()
                or str(record.get("derived_event_id") or "").strip()
                or _event_id_from_correlation_id(record.get("correlation_id"))
            )
            if not worker_event_id:
                continue
            fact = worker_session_facts.setdefault(
                worker_event_id,
                {
                    "event_id": worker_event_id,
                    "correlation_id": str(record.get("correlation_id") or "").strip(),
                    "session_key": str(record.get("session_key") or "").strip(),
                    "message_id": str(record.get("message_id") or "").strip(),
                    "execution_mode": "",
                    "route_hint": "",
                    "route_version": "",
                    "provider_alias": "",
                    "model_catalog_version": "",
                    "external_exec_candidate": None,
                    "request_class": "",
                    "content_modalities": [],
                    "route_family": "",
                    "gateway_route_name": "",
                    "gateway_eligible": None,
                    "cache_eligible": None,
                    "cache_status": "",
                    "requires_tools": None,
                    "requires_browser": None,
                    "requires_media_hydration": None,
                    "toolset": [],
                    "modality_profile": "",
                    "gateway_error_class": "",
                    "misroute_detected": False,
                    "dfmea_failure_mode": "",
                    "dfmea_control_phase": "",
                    "dfmea_detection_signal": "",
                    "dfmea_control_action": "",
                    "dfmea_severity": None,
                    "reply_visible_proxy": None,
                    "feedback_score_before": None,
                    "feedback_score_after": None,
                    "ai_call_count": None,
                    "capability_match": None,
                    "preferred_model_selected": None,
                    "reply_send_message_id": None,
                    "reply_send_message_ids": [],
                    "t0_ms": None,
                    "t3_ms": None,
                    "ai_model_elapsed_ms": None,
                    "cloudflare_ai_cost_usd": None,
                    "cloudflare_send_elapsed_ms": None,
                    "event_names": [],
                },
            )
            fact["event_names"].append(event)
            if not fact.get("correlation_id"):
                fact["correlation_id"] = str(record.get("correlation_id") or "").strip()
            if not fact.get("session_key"):
                fact["session_key"] = str(record.get("session_key") or "").strip()
            if not fact.get("message_id"):
                fact["message_id"] = str(record.get("message_id") or "").strip()
            execution_mode = str(record.get("execution_mode") or "").strip()
            if execution_mode:
                fact["execution_mode"] = execution_mode
            route_hint = str(record.get("route_hint") or "").strip()
            if route_hint and not fact.get("route_hint"):
                fact["route_hint"] = route_hint
            route_version = str(record.get("route_version") or "").strip()
            if route_version and not fact.get("route_version"):
                fact["route_version"] = route_version
            provider_alias = str(record.get("provider_alias") or "").strip()
            if provider_alias and not fact.get("provider_alias"):
                fact["provider_alias"] = provider_alias
            model_catalog_version = str(record.get("model_catalog_version") or "").strip()
            if model_catalog_version and not fact.get("model_catalog_version"):
                fact["model_catalog_version"] = model_catalog_version
            if isinstance(record.get("external_exec_candidate"), bool):
                fact["external_exec_candidate"] = bool(record.get("external_exec_candidate"))
            request_class = str(record.get("request_class") or "").strip()
            if request_class and not fact.get("request_class"):
                fact["request_class"] = request_class
            route_family = str(record.get("route_family") or "").strip()
            if route_family and not fact.get("route_family"):
                fact["route_family"] = route_family
            gateway_route_name = str(record.get("gateway_route_name") or "").strip()
            if gateway_route_name and not fact.get("gateway_route_name"):
                fact["gateway_route_name"] = gateway_route_name
            if record.get("gateway_eligible") is not None and fact.get("gateway_eligible") is None:
                fact["gateway_eligible"] = bool(record.get("gateway_eligible"))
            cache_eligible = _coerce_optional_bool(record.get("cache_eligible"))
            if cache_eligible is not None and fact.get("cache_eligible") is None:
                fact["cache_eligible"] = cache_eligible
            cache_status = str(record.get("cf_cache_status") or record.get("cache_status") or "").strip()
            if cache_status and not fact.get("cache_status"):
                fact["cache_status"] = cache_status
            if record.get("requires_tools") is not None and fact.get("requires_tools") is None:
                fact["requires_tools"] = bool(record.get("requires_tools"))
            if record.get("requires_browser") is not None and fact.get("requires_browser") is None:
                fact["requires_browser"] = bool(record.get("requires_browser"))
            if record.get("requires_media_hydration") is not None and fact.get("requires_media_hydration") is None:
                fact["requires_media_hydration"] = bool(record.get("requires_media_hydration"))
            content_modalities = _coerce_string_list(record.get("content_modalities"))
            if content_modalities:
                fact["content_modalities"] = _unique_nonempty_strings(
                    list(fact.get("content_modalities") or []) + content_modalities
                )
            hermes_request = record.get("hermes_request")
            if isinstance(hermes_request, dict):
                if not fact.get("modality_profile"):
                    fact["modality_profile"] = str(hermes_request.get("modality_profile") or "").strip()
                hermes_toolset = _coerce_string_list(hermes_request.get("toolset"))
                if hermes_toolset:
                    fact["toolset"] = _unique_nonempty_strings(list(fact.get("toolset") or []) + hermes_toolset)
            toolset = _coerce_string_list(record.get("toolset"))
            if toolset:
                fact["toolset"] = _unique_nonempty_strings(list(fact.get("toolset") or []) + toolset)
            modality_profile = str(record.get("modality_profile") or "").strip()
            if modality_profile and not fact.get("modality_profile"):
                fact["modality_profile"] = modality_profile
            gateway_error_class = str(record.get("gateway_error_class") or "").strip()
            if gateway_error_class:
                fact["gateway_error_class"] = gateway_error_class
            if bool(record.get("misroute_detected")) or gateway_error_class == "misrouted_request_class":
                fact["misroute_detected"] = True
            dfmea_failure_mode = str(record.get("dfmea_failure_mode") or "").strip()
            if dfmea_failure_mode:
                fact["dfmea_failure_mode"] = dfmea_failure_mode
            dfmea_control_phase = str(record.get("dfmea_control_phase") or "").strip()
            if dfmea_control_phase:
                fact["dfmea_control_phase"] = dfmea_control_phase
            dfmea_detection_signal = str(record.get("dfmea_detection_signal") or "").strip()
            if dfmea_detection_signal:
                fact["dfmea_detection_signal"] = dfmea_detection_signal
            dfmea_control_action = str(record.get("dfmea_control_action") or "").strip()
            if dfmea_control_action:
                fact["dfmea_control_action"] = dfmea_control_action
            if isinstance(record.get("dfmea_severity"), (int, float)) and fact.get("dfmea_severity") is None:
                fact["dfmea_severity"] = int(record.get("dfmea_severity"))
            if record.get("feedback_score_before") not in (None, "") and fact.get("feedback_score_before") in (None, ""):
                fact["feedback_score_before"] = record.get("feedback_score_before")
            if record.get("feedback_score_after") not in (None, "") and fact.get("feedback_score_after") in (None, ""):
                fact["feedback_score_after"] = record.get("feedback_score_after")
            ai_call_count = _coerce_optional_int(record.get("ai_call_count"))
            if ai_call_count is not None:
                fact["ai_call_count"] = max(ai_call_count, int(fact.get("ai_call_count") or 0))
            capability_match = _coerce_optional_bool(record.get("capability_match"))
            if capability_match is not None and fact.get("capability_match") is None:
                fact["capability_match"] = capability_match
            preferred_model_selected = _coerce_optional_bool(record.get("preferred_model_selected"))
            if preferred_model_selected is not None and fact.get("preferred_model_selected") is None:
                fact["preferred_model_selected"] = preferred_model_selected
            event_ts_ms = _normalized_epoch_ms(record.get("__timestamp_ms"))
            if event in {"feishu.webhook.accepted", "feishu.workflow.start", "feishu.direct_planned.start"} and event_ts_ms is not None:
                if fact["t0_ms"] is None or event_ts_ms < int(fact["t0_ms"]):
                    fact["t0_ms"] = event_ts_ms
            if event in {"feishu.send.operation.done", "feishu.workflow.send.done", "feishu.direct_planned.done"} and event_ts_ms is not None:
                if fact["t3_ms"] is None or event_ts_ms < int(fact["t3_ms"]):
                    fact["t3_ms"] = event_ts_ms
                    fact["reply_visible_proxy"] = event
            send_message_id = str(record.get("send_message_id") or "").strip()
            if send_message_id:
                if send_message_id not in fact["reply_send_message_ids"]:
                    fact["reply_send_message_ids"].append(send_message_id)
                operation_kind = str(record.get("kind") or "").strip().lower()
                if not fact.get("reply_send_message_id") or operation_kind in {"text", "interactive", "post"}:
                    fact["reply_send_message_id"] = send_message_id
            if isinstance(record.get("cf_ai_exec_elapsed_ms"), (int, float)):
                fact["ai_model_elapsed_ms"] = max(
                    float(record["cf_ai_exec_elapsed_ms"]),
                    float(fact["ai_model_elapsed_ms"] or 0),
                )
            if cost_value not in (None, ""):
                gateway_cost = _decimal(cost_value)
                if gateway_cost >= 0:
                    fact["cloudflare_ai_cost_usd"] = _decimal_str(
                        _decimal(fact.get("cloudflare_ai_cost_usd")) + gateway_cost
                    )
            if isinstance(record.get("cf_send_elapsed_ms"), (int, float)):
                fact["cloudflare_send_elapsed_ms"] = max(
                    float(record["cf_send_elapsed_ms"]),
                    float(fact["cloudflare_send_elapsed_ms"] or 0),
                )
        window_summaries[label] = {
            "record_count": len(window_records),
            "matched_session_record_count": sum(1 for record in window_records if record.get("matched_session_id")),
            "event_counts": dict(sorted(event_counts.items())),
            "upstream_model_elapsed_ms": _summarize_numeric_series(exec_latencies),
            "cloudflare_send_elapsed_ms": _summarize_numeric_series(send_latencies),
            "cloudflare_ai_cost_usd": _summarize_decimal_series(gateway_costs),
            "failure_rate": _safe_ratio(failures, len(window_records)) if window_records else None,
            "cost_detail_status": "available" if gateway_costs else "missing",
            "gateway_request_rows": [
                {
                    "event": str(record.get("event") or "").strip(),
                    "request_class": str(record.get("request_class") or "").strip(),
                    "gateway_route_name": str(record.get("gateway_route_name") or "").strip(),
                    "route_family": str(record.get("route_family") or "").strip(),
                    "cf_cache_status": record.get("cf_cache_status"),
                    "gateway_eligible": _coerce_optional_bool(record.get("gateway_eligible")),
                    "cache_eligible": _coerce_optional_bool(record.get("cache_eligible")),
                    "ai_call_count": _coerce_optional_int(record.get("ai_call_count")),
                    "capability_match": _coerce_optional_bool(record.get("capability_match")),
                    "preferred_model_selected": _coerce_optional_bool(record.get("preferred_model_selected")),
                    "provider": record.get("provider"),
                    "model": record.get("model"),
                }
                for record in window_records
                if str(record.get("event") or "").strip() in {"feishu.cf_ai_gateway.log", "feishu.cf_ai_exec.done"}
            ],
            "matched_session_model_elapsed_ms": {
                session_id: round(duration_ms, 1) for session_id, duration_ms in sorted(session_model_elapsed.items())
            },
            "matched_session_gateway_costs_usd": {
                session_id: _decimal_str(cost) for session_id, cost in sorted(matched_session_gateway_costs.items())
            },
            "worker_session_facts": worker_session_facts,
            "read_event_facts": read_event_facts,
        }
        bootstrapped_worker_fact_count += len(worker_session_facts)

    if matched_records == 0 and bootstrapped_worker_fact_count > 0:
        join_level = "worker_event_level"

    notes = []
    if join_level == "worker_event_level":
        notes.append(
            "Cloudflare Worker Observability supplied event-level session facts directly, even though they could not be joined back to Modal trace sessions."
        )
    elif join_level != "event_level":
        notes.append("Cloudflare records lacked enough session_key overlap for reliable event-level joins; summary is interval-only.")
    if api_records:
        notes.append("Cloudflare AI Gateway logs were fetched live from the official API.")
    if worker_records:
        notes.append("Cloudflare Worker Observability events were fetched live from the official API.")
    cost_details_available = any(((payload.get("cost_detail_status") == "available") for payload in window_summaries.values()))
    if cost_details_available:
        notes.append("Cloudflare AI Gateway official per-request cost was attached from the live logs API when available.")
    else:
        notes.append("Cloudflare logs did not expose official per-request cost in the discovered artifacts.")
    return (
        {
            "status": "ok" if (matched_records or bootstrapped_worker_fact_count > 0) else "degraded",
            "join_level": join_level,
            "matched_record_count": matched_records,
            "bootstrapped_worker_fact_count": bootstrapped_worker_fact_count,
            "log_paths": [str(path) for path in log_paths]
            + (["cloudflare_api:ai_gateway"] if api_records else [])
            + (["cloudflare_api:worker_observability"] if worker_records else []),
            "window_summaries": window_summaries,
            "notes": notes,
        },
        gaps
        + (["cloudflare_interval_only_join"] if join_level not in {"event_level", "worker_event_level"} else [])
        + ([] if cost_details_available else ["cloudflare_cost_details_missing"]),
    )


def _build_hourly_cost_pk(
    windows: dict[str, AnalysisWindow],
    hourly_costs_by_window: dict[str, dict[str, Decimal]],
    session_counts_by_window: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    current_window = windows["current"]
    slot_keys = [_hour_key(slot) for slot in _iter_hour_slots(current_window.start, current_window.end)]
    compare_labels = [label for label in windows if label != "current"]
    payload: list[dict[str, Any]] = []
    current_hourly = hourly_costs_by_window.get("current", {})
    current_session_counts = session_counts_by_window.get("current", {})
    for index, hour_key in enumerate(slot_keys):
        slot_dt = datetime.fromisoformat(hour_key)
        current_cost = current_hourly.get(hour_key, Decimal("0"))
        current_sessions = current_session_counts.get(hour_key, 0)
        entry = {
            "slot_index": index,
            "hour_key": hour_key,
            "local_slot_start": slot_dt.isoformat(),
            "current": {
                "cost_usd": _decimal_str(current_cost),
                "session_count": current_sessions,
                "cost_per_session_usd": _decimal_str(current_cost / current_sessions) if current_sessions else None,
            },
        }
        for label in compare_labels:
            compare_window = windows[label]
            compare_hour_key = _hour_key(slot_dt - timedelta(days=compare_window.compare_day_shift))
            compare_cost = hourly_costs_by_window.get(label, {}).get(compare_hour_key, Decimal("0"))
            compare_sessions = session_counts_by_window.get(label, {}).get(compare_hour_key, 0)
            baseline_key = f"compare_{label.replace('-', '_')}"
            entry[baseline_key] = {
                "window_label": label,
                "hour_key": compare_hour_key,
                "cost_usd": _decimal_str(compare_cost),
                "session_count": compare_sessions,
                "cost_per_session_usd": _decimal_str(compare_cost / compare_sessions) if compare_sessions else None,
                "delta_cost_usd": _decimal_str(current_cost - compare_cost),
                "delta_cost_pct": _safe_pct_delta(current_cost, compare_cost),
            }
        payload.append(entry)
    return payload


def _compare_metric(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current_cost = _decimal(current.get("calibrated_cost_usd"))
    baseline_cost = _decimal(baseline.get("calibrated_cost_usd"))
    current_cps = _decimal(current.get("cost_per_session_usd"))
    baseline_cps = _decimal(baseline.get("cost_per_session_usd"))
    return {
        "delta_cost_usd": _decimal_str(current_cost - baseline_cost),
        "delta_cost_pct": _safe_pct_delta(current_cost, baseline_cost),
        "delta_cost_per_session_pct": _safe_pct_delta(current_cps, baseline_cps),
        "delta_invocation_count": int(current.get("invocation_count") or 0) - int(baseline.get("invocation_count") or 0),
        "delta_session_count": int(current.get("session_count") or 0) - int(baseline.get("session_count") or 0),
    }


def _snapshot_window_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    session_summary = payload.get("session_summary") or {}
    completion = session_summary.get("completion") or {}
    metrics = session_summary.get("metrics") or {}
    billing = payload.get("billing_summary") or {}
    read_metric = (metrics.get("read_receipt_ms") or {}).get("p90")
    reply_minus_ai_metric = (metrics.get("t0_to_t3_minus_model_ms") or {}).get("p90")
    t3_metric = (metrics.get("t0_to_t3_ms") or {}).get("p90")
    avg_cost = billing.get("avg_cost_per_session_usd")
    total_cost = billing.get("total_cost_usd")
    return {
        "session_count": int(completion.get("session_count") or 0),
        "read_receipt_p90_ms": round(float(read_metric), 1) if read_metric is not None else None,
        "reply_minus_ai_p90_ms": round(float(reply_minus_ai_metric), 1) if reply_minus_ai_metric is not None else None,
        "t0_to_t3_p90_ms": round(float(t3_metric), 1) if t3_metric is not None else None,
        "avg_cost_per_session_usd": round(float(avg_cost), 6) if avg_cost is not None else None,
        "total_cost_usd": round(float(total_cost), 6) if total_cost is not None else None,
    }


def _trend_label(current: Any, baseline: Any, *, lower_is_better: bool) -> str:
    if current is None or baseline is None:
        return "insufficient_data"
    try:
        current_value = Decimal(str(current))
        baseline_value = Decimal(str(baseline))
    except Exception:
        return "insufficient_data"
    delta = current_value - baseline_value
    if delta == 0:
        return "flat"
    if lower_is_better:
        return "improving" if delta < 0 else "regressing"
    return "improving" if delta > 0 else "regressing"


def _overall_window_trend(trend_map: dict[str, str]) -> str:
    improving = sum(1 for value in trend_map.values() if value == "improving")
    regressing = sum(1 for value in trend_map.values() if value == "regressing")
    if improving == 0 and regressing == 0:
        return "insufficient_data"
    if improving > regressing:
        return "improving"
    if regressing > improving:
        return "regressing"
    return "mixed"


def _build_fixed_window_rollup(window_payload_matrix: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rollup: dict[str, Any] = {}
    for label, payloads in window_payload_matrix.items():
        current_snapshot = _snapshot_window_metrics(payloads.get("current") or {})
        baselines = {
            baseline_label: _snapshot_window_metrics(payload)
            for baseline_label, payload in payloads.items()
            if baseline_label != "current"
        }
        primary_baseline = baselines.get("d-1") or next(iter(baselines.values()), {})
        trend = {
            "session_count": _trend_label(
                current_snapshot.get("session_count"),
                primary_baseline.get("session_count"),
                lower_is_better=False,
            ),
            "read_receipt_p90_ms": _trend_label(
                current_snapshot.get("read_receipt_p90_ms"),
                primary_baseline.get("read_receipt_p90_ms"),
                lower_is_better=True,
            ),
            "reply_minus_ai_p90_ms": _trend_label(
                current_snapshot.get("reply_minus_ai_p90_ms"),
                primary_baseline.get("reply_minus_ai_p90_ms"),
                lower_is_better=True,
            ),
            "t0_to_t3_p90_ms": _trend_label(
                current_snapshot.get("t0_to_t3_p90_ms"),
                primary_baseline.get("t0_to_t3_p90_ms"),
                lower_is_better=True,
            ),
            "avg_cost_per_session_usd": _trend_label(
                current_snapshot.get("avg_cost_per_session_usd"),
                primary_baseline.get("avg_cost_per_session_usd"),
                lower_is_better=True,
            ),
        }
        rollup[label] = {
            "current": current_snapshot,
            "baselines": baselines,
            "trend": {
                **trend,
                "overall": _overall_window_trend(trend),
            },
        }
    return rollup


def _build_function_cost_pk(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = list(window_payloads)
    all_functions = sorted(
        {
            function_name
            for payload in window_payloads.values()
            for function_name in (payload.get("function_summary") or {}).keys()
        }
    )
    comparisons: list[dict[str, Any]] = []
    for function_name in all_functions:
        current = (window_payloads.get("current", {}).get("function_summary") or {}).get(function_name, {"function_name": function_name})
        entry = {"function_name": function_name, "current": current}
        for label in labels:
            if label == "current":
                continue
            baseline = (window_payloads.get(label, {}).get("function_summary") or {}).get(function_name, {"function_name": function_name})
            entry[label.replace("-", "_")] = baseline
            entry[f"delta_vs_{label.replace('-', '_')}"] = _compare_metric(current, baseline)
        comparisons.append(entry)
    comparisons.sort(key=lambda item: _decimal((item.get("current") or {}).get("calibrated_cost_usd")), reverse=True)
    return {
        "comparisons": comparisons,
        "window_totals": {
            label: {
                "total_cost_usd": _decimal_str(
                    sum((_decimal(item.get("calibrated_cost_usd")) for item in (payload.get("function_summary") or {}).values()), Decimal("0"))
                ),
                "function_count": len(payload.get("function_summary") or {}),
            }
            for label, payload in window_payloads.items()
        },
    }


def _build_container_cost_pk(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    overall = {
        label: (payload.get("container_summary") or {}).get("fleet_summary") or {}
        for label, payload in window_payloads.items()
    }
    by_function_labels = sorted(
        {
            function_name
            for payload in window_payloads.values()
            for function_name in ((payload.get("container_summary") or {}).get("by_function") or {}).keys()
        }
    )
    by_function: list[dict[str, Any]] = []
    for function_name in by_function_labels:
        current = ((window_payloads.get("current", {}).get("container_summary") or {}).get("by_function") or {}).get(
            function_name,
            {"function_name": function_name},
        )
        entry = {"function_name": function_name, "current": current}
        for label, payload in window_payloads.items():
            if label == "current":
                continue
            baseline = ((payload.get("container_summary") or {}).get("by_function") or {}).get(
                function_name,
                {"function_name": function_name},
            )
            entry[label.replace("-", "_")] = baseline
        by_function.append(entry)
    return {
        "fleet_comparison": overall,
        "by_function": by_function,
        "hot_containers_current": ((window_payloads.get("current", {}).get("container_summary") or {}).get("hot_containers") or []),
    }


def _build_session_latency_pk(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    current_metrics = (window_payloads.get("current", {}).get("session_summary") or {}).get("metrics") or {}
    for metric_name, metric_summary in current_metrics.items():
        entry = {"metric_name": metric_name, "current": metric_summary}
        for label, payload in window_payloads.items():
            if label == "current":
                continue
            baseline_metrics = ((payload.get("session_summary") or {}).get("metrics") or {}).get(metric_name, {"count": 0})
            entry[label.replace("-", "_")] = baseline_metrics
            current_p50 = _decimal(metric_summary.get("p50"))
            baseline_p50 = _decimal(baseline_metrics.get("p50"))
            entry[f"delta_p50_vs_{label.replace('-', '_')}_ms"] = round(float(current_p50 - baseline_p50), 1)
            entry[f"delta_p50_vs_{label.replace('-', '_')}_pct"] = _safe_pct_delta(current_p50, baseline_p50)
        comparisons.append(entry)
    return {
        "comparisons": comparisons,
        "completion": {
            label: (payload.get("session_summary") or {}).get("completion") or {}
            for label, payload in window_payloads.items()
        },
        "slow_sessions_current": ((window_payloads.get("current", {}).get("session_summary") or {}).get("slowest_sessions") or []),
    }


def _build_idle_cost_summary(window_payload: dict[str, Any]) -> dict[str, Any]:
    hourly_costs = dict(window_payload.get("hourly_official_costs") or {})
    session_counts = dict(window_payload.get("hourly_session_counts") or {})
    idle_hour_entries = []
    for hour_key, cost in sorted(hourly_costs.items()):
        if int(session_counts.get(hour_key, 0) or 0) != 0:
            continue
        idle_hour_entries.append(
            {
                "hour_key": hour_key,
                "official_cost_usd": _decimal_str(_decimal(cost)),
            }
        )
    idle_values = [_decimal(entry["official_cost_usd"]) for entry in idle_hour_entries if entry.get("official_cost_usd") is not None]
    return {
        "idle_hour_count": len(idle_hour_entries),
        "idle_hours": idle_hour_entries,
        "summary": _summarize_decimal_series(idle_values),
    }


def _build_goal_assessment(
    window_payloads: dict[str, dict[str, Any]],
    *,
    cloudflare_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_payload = window_payloads.get("current", {})
    billing_summary = current_payload.get("billing_summary") or {}
    session_summary = current_payload.get("session_summary") or {}
    completion = session_summary.get("completion") or {}
    metrics = session_summary.get("metrics") or {}
    idle_cost = _build_idle_cost_summary(current_payload)
    session_count = int(completion.get("session_count") or 0)
    total_cost = _decimal(billing_summary.get("total_cost_usd"))
    avg_cost_per_session = (total_cost / session_count) if session_count else None
    allocation_gap_reasons = {
        str(gap.get("reason") or "").strip()
        for gap in ((current_payload.get("cost_calibration") or {}).get("allocation_gaps") or [])
        if isinstance(gap, dict)
    }
    has_function_trace_gap = "official_cost_without_matching_function_trace" in allocation_gap_reasons
    allocated_session_costs = _collect_nonnegative_decimals(
        session.get("allocated_cost_usd")
        for session in (session_summary.get("sessions") or [])
        if session.get("allocated_cost_usd") is not None
    )
    total_session_costs = _collect_nonnegative_decimals(
        session.get("total_session_cost_usd")
        for session in (session_summary.get("sessions") or [])
        if session.get("total_session_cost_usd") is not None
    )
    modal_session_cost_summary = _summarize_decimal_series(allocated_session_costs)
    total_session_cost_summary = _summarize_decimal_series(total_session_costs)
    read_receipt_mode = _read_receipt_metric_mode(metrics)
    read_receipt_p90 = _read_receipt_metric_value(metrics, "p90")
    reply_minus_ai_p90 = _decimal(((metrics.get("t0_to_t3_minus_model_ms") or {}).get("p90")))
    idle_hourly_p90 = _decimal(((idle_cost.get("summary") or {}).get("p90")))
    session_cost_p90 = _decimal(modal_session_cost_summary.get("p90"))
    total_session_cost_p90 = _decimal(total_session_cost_summary.get("p90"))
    cache_eligible_hit_rate = _decimal(
        ((_build_cache_hit_rate(cloudflare_summary or {}, eligible_only=True).get("current") or {}).get("hit_rate"))
    )
    overall_cache_hit_rate = _decimal(
        ((_build_cache_hit_rate(cloudflare_summary or {}, eligible_only=False).get("current") or {}).get("hit_rate"))
    )
    browser_single_ai_call_completion_rate = _decimal(
        ((_build_browser_single_ai_call_completion(window_payloads).get("current") or {}).get("completion_rate"))
    )
    capability_match_rate = _decimal(((_build_capability_match_rate(window_payloads).get("current") or {}).get("match_rate")))
    preferred_model_selection_accuracy = _decimal(
        ((_build_preferred_model_selection_accuracy(window_payloads).get("current") or {}).get("accuracy"))
    )

    read_actual = read_receipt_p90 if read_receipt_p90 > 0 else None
    reply_minus_ai_actual = reply_minus_ai_p90 if reply_minus_ai_p90 > 0 else None
    idle_actual = idle_hourly_p90 if idle_hourly_p90 > 0 or int(idle_cost.get("idle_hour_count") or 0) > 0 else None
    session_cost_measurement_mode = "modal_session_allocated_p90"
    if any(cost > 0 for cost in allocated_session_costs):
        session_cost_actual = session_cost_p90
    elif has_function_trace_gap:
        session_cost_actual = None
        session_cost_measurement_mode = "insufficient_function_trace"
    elif avg_cost_per_session is not None and avg_cost_per_session > 0:
        session_cost_actual = avg_cost_per_session
        session_cost_measurement_mode = "official_total_div_session_count"
    else:
        session_cost_actual = None
        session_cost_measurement_mode = ""
    total_session_cost_actual = (
        total_session_cost_p90 if any(cost > 0 for cost in total_session_costs) else None
    )

    return {
        "targets": {
            "read_receipt_ms": int(DEFAULT_GOALS["read_receipt_ms"]),
            "reply_minus_ai_ms": int(DEFAULT_GOALS["reply_minus_ai_ms"]),
            "idle_hourly_cost_usd": _decimal_str(DEFAULT_GOALS["idle_hourly_cost_usd"]),
            "cost_per_session_usd": _decimal_str(DEFAULT_GOALS["cost_per_session_usd"]),
            "cache_hit_rate": _decimal_str(DEFAULT_GOALS["cache_hit_rate"]),
            "browser_single_ai_call_completion_rate": _decimal_str(DEFAULT_GOALS["browser_single_ai_call_completion_rate"]),
            "capability_match_rate": _decimal_str(DEFAULT_GOALS["capability_match_rate"]),
            "preferred_model_selection_accuracy": _decimal_str(DEFAULT_GOALS["preferred_model_selection_accuracy"]),
        },
        "current": {
            "read_receipt_p90_ms": round(float(read_actual), 1) if read_actual is not None else None,
            "read_receipt_measurement_mode": read_receipt_mode,
            "reply_minus_ai_p90_ms": round(float(reply_minus_ai_actual), 1) if reply_minus_ai_actual is not None else None,
            "idle_hourly_p90_cost_usd": _decimal_str(idle_actual) if idle_actual is not None else None,
            "session_cost_p90_usd": _decimal_str(session_cost_actual) if session_cost_actual is not None else None,
            "session_cost_measurement_mode": session_cost_measurement_mode,
            "session_total_cost_p90_usd": _decimal_str(total_session_cost_actual) if total_session_cost_actual is not None else None,
            "avg_cost_per_session_usd": _decimal_str(avg_cost_per_session) if avg_cost_per_session is not None else None,
            "cache_eligible_hit_rate": _decimal_str(cache_eligible_hit_rate) if cache_eligible_hit_rate > 0 else None,
            "overall_cache_hit_rate": _decimal_str(overall_cache_hit_rate) if overall_cache_hit_rate > 0 else None,
            "browser_single_ai_call_completion_rate": (
                _decimal_str(browser_single_ai_call_completion_rate) if browser_single_ai_call_completion_rate > 0 else None
            ),
            "capability_match_rate": _decimal_str(capability_match_rate) if capability_match_rate > 0 else None,
            "preferred_model_selection_accuracy": (
                _decimal_str(preferred_model_selection_accuracy) if preferred_model_selection_accuracy > 0 else None
            ),
        },
        "statuses": {
            "read_receipt_under_5s": _goal_status(
                actual=read_actual,
                target=Decimal(str(DEFAULT_GOALS["read_receipt_ms"])),
            ),
            "reply_minus_ai_under_20s": _goal_status(
                actual=reply_minus_ai_actual,
                target=Decimal(str(DEFAULT_GOALS["reply_minus_ai_ms"])),
            ),
            "modal_idle_hourly_cost_under_0_005": _goal_status(
                actual=idle_actual,
                target=DEFAULT_GOALS["idle_hourly_cost_usd"],
            ),
            "session_cost_under_0_0045": _goal_status(
                actual=session_cost_actual,
                target=DEFAULT_GOALS["cost_per_session_usd"],
            ),
            "cache_hit_rate_over_0_30": _goal_status(
                actual=cache_eligible_hit_rate if cache_eligible_hit_rate > 0 else None,
                target=DEFAULT_GOALS["cache_hit_rate"],
                smaller_is_better=False,
                inclusive=True,
            ),
            "browser_single_ai_call_completion_rate_over_0_50": _goal_status(
                actual=browser_single_ai_call_completion_rate if browser_single_ai_call_completion_rate > 0 else None,
                target=DEFAULT_GOALS["browser_single_ai_call_completion_rate"],
                smaller_is_better=False,
                inclusive=True,
            ),
            "capability_match_rate_equals_1_00": _goal_status(
                actual=capability_match_rate if capability_match_rate > 0 else None,
                target=DEFAULT_GOALS["capability_match_rate"],
                smaller_is_better=False,
                inclusive=True,
            ),
            "preferred_model_selection_accuracy_over_0_95": _goal_status(
                actual=preferred_model_selection_accuracy if preferred_model_selection_accuracy > 0 else None,
                target=DEFAULT_GOALS["preferred_model_selection_accuracy"],
                smaller_is_better=False,
                inclusive=True,
            ),
        },
        "session_cost_detail": modal_session_cost_summary,
        "session_total_cost_detail": total_session_cost_summary,
        "idle_cost_detail": idle_cost,
        "notes": [
            "All target checks use current-window p90 values and strict '<' thresholds.",
            "Read receipt target uses original inbound-message read timing (T0->T2) when available; reply-send to reply-read remains a supplemental diagnostic only.",
            "Missing measurements are treated as not_met for gatekeeping, while detailed measurement mode fields remain available for diagnosis.",
            "Idle cost uses official Modal hourly billing for hours with zero observed sessions; if no idle hours are present, the gate remains not_met until data is available.",
            "Session cost target uses p90 over per-session Modal calibrated allocation only.",
            "When official Modal billing exists but function-level trace allocation is missing, session cost remains unavailable instead of falling back to total-cost/session-count.",
            "Combined session total cost remains available as a diagnostic field and may include provider billed cost and Cloudflare AI Gateway cost when present.",
        ],
    }


def _load_feishu_api_module():
    return _load_module("feishu_triparty_feishu_api_module", REPO_ROOT / "tools" / "feishu_api.py")


def _build_triparty_feishu_client(*, timeout: float = 20):
    feishu_api_module = _load_feishu_api_module()
    canonical_app_id = str(os.getenv("FEISHU_APP_ID3") or "").strip()
    canonical_app_secret = str(os.getenv("FEISHU_APP_SECRET3") or "").strip()
    if canonical_app_id and canonical_app_secret and hasattr(feishu_api_module, "FeishuOpenApiClient"):
        return feishu_api_module.FeishuOpenApiClient(
            app_id=canonical_app_id,
            app_secret=canonical_app_secret,
            timeout=timeout,
        )
    return feishu_api_module.build_feishu_client(timeout=timeout)


def _extract_feishu_read_users_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("items", "read_users"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _request_feishu_read_users_page(
    client: Any,
    message_id: str,
    *,
    page_token: str = "",
) -> dict[str, Any]:
    param_candidates: list[dict[str, str]] = []
    if page_token:
        param_candidates.append({"page_token": page_token})
        param_candidates.append({"user_id_type": "open_id", "page_token": page_token})
    else:
        param_candidates.append({})
        param_candidates.append({"user_id_type": "open_id"})

    last_error: Exception | None = None
    for params in param_candidates:
        try:
            if hasattr(client, "base_url") and hasattr(client, "get_tenant_access_token"):
                token = client.get_tenant_access_token()
                response = httpx.get(
                    f"{str(client.base_url).rstrip('/')}/open-apis/im/v1/messages/{message_id}/read_users",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params or None,
                    timeout=float(getattr(client, "timeout", 20) or 20),
                )
                try:
                    payload = response.json()
                except Exception:
                    payload = {}
                if response.status_code == 200 and payload.get("code", 0) == 0:
                    data = payload.get("data") or {}
                    return data if isinstance(data, dict) else {}
                raise RuntimeError(
                    f"feishu_read_users_error:status={response.status_code}:code={payload.get('code')}:msg={payload.get('msg')}"
                )
            return client.request_json(
                "GET",
                f"/open-apis/im/v1/messages/{message_id}/read_users",
                params=params or None,
            )
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"feishu_read_users_request_failed:{message_id}")


def _is_nonfatal_feishu_read_users_error(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "code=230012",
            "bot is not the sender of the message",
            "code=99992354",
            "id not exist",
            "not a valid {open_message_id}",
        )
    )


def _fetch_feishu_read_users_for_message_ids(message_ids: list[str]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    normalized_message_ids = _unique_nonempty_strings(message_ids)
    if not normalized_message_ids:
        return {}, []
    try:
        client = _build_triparty_feishu_client(timeout=20)
    except Exception as exc:
        return {}, [f"feishu_read_users_client_init_failed:{exc}"]

    results: dict[str, list[dict[str, Any]]] = {}
    gaps: list[str] = []
    for message_id in normalized_message_ids:
        try:
            items: list[dict[str, Any]] = []
            page_token = ""
            while True:
                payload = _request_feishu_read_users_page(client, message_id, page_token=page_token)
                items.extend(_extract_feishu_read_users_items(payload))
                if not payload.get("has_more"):
                    break
                page_token = str(payload.get("page_token") or "").strip()
                if not page_token:
                    break
            results[message_id] = items
        except Exception as exc:
            if _is_nonfatal_feishu_read_users_error(exc):
                continue
            gaps.append(f"feishu_read_users_fetch_failed:{message_id}:{exc}")
    return results, gaps


def _augment_recent_sessions_with_feishu_read_api(
    window_payloads: dict[str, dict[str, Any]],
    *,
    recent_hours: int,
    recent_min_sessions: int,
    timezone_name: str,
) -> list[str]:
    current_payload = window_payloads.get("current") or {}
    selection = _select_recent_sessions(
        current_payload,
        recent_hours=recent_hours,
        recent_min_sessions=recent_min_sessions,
    )
    sessions = list(selection.get("sessions") or [])
    unread_sessions = [
        session
        for session in sessions
        if not session.get("read_matched")
        and _unique_nonempty_strings(
            [session.get("message_id"), session.get("reply_send_message_id")] + list(session.get("reply_send_message_ids") or [])
        )
    ]
    if not unread_sessions:
        return []

    read_lookup_ids: list[str] = []
    for session in unread_sessions:
        prioritized_ids = _unique_nonempty_strings(
            [session.get("message_id"), session.get("reply_send_message_id")] + list(session.get("reply_send_message_ids") or [])
        )
        if prioritized_ids:
            read_lookup_ids.append(prioritized_ids[0])
    read_users_by_message_id, gaps = _fetch_feishu_read_users_for_message_ids(read_lookup_ids[:200])
    read_rows: list[dict[str, Any]] = []
    for message_id, items in read_users_by_message_id.items():
        timestamps = sorted(
            epoch_ms
            for epoch_ms in (_normalized_epoch_ms(item.get("timestamp")) for item in items)
            if epoch_ms is not None
        )
        if not timestamps:
            continue
        read_rows.append(
            {
                "stage": "feishu.read_users_api",
                "event_type": "im.message.message_read_v1",
                "message_id_list": [message_id],
                "read_time": timestamps[0],
                "ts": int(timestamps[0] / 1000),
            }
        )
    if read_rows:
        current_session_summary = current_payload.get("session_summary") or {}
        all_sessions = current_session_summary.get("sessions") or []
        if _apply_read_receipt_rows_to_sessions(all_sessions, read_rows, source_label="feishu_read_users_api"):
            _refresh_session_summary_payload(current_payload, timezone_name=timezone_name)
    return gaps


def _recent_window_bounds(payload: dict[str, Any], recent_hours: int) -> tuple[int, int]:
    window = payload.get("window") or {}
    end_raw = str(window.get("end") or "").strip()
    end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00")) if end_raw else datetime.now(dt_timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = end_ms - max(int(recent_hours or 0), 0) * 3600 * 1000
    return start_ms, end_ms


def _select_recent_sessions(
    payload: dict[str, Any],
    *,
    recent_hours: int,
    recent_min_sessions: int,
) -> dict[str, Any]:
    sessions = list(((payload.get("session_summary") or {}).get("sessions") or []))
    sessions = [session for session in sessions if isinstance(session.get("t0_ms"), int)]
    sessions.sort(key=lambda session: int(session.get("t0_ms") or 0))
    start_ms, end_ms = _recent_window_bounds(payload, recent_hours)
    in_range = [session for session in sessions if start_ms <= int(session.get("t0_ms") or 0) < end_ms]
    low_sample = len(in_range) < max(int(recent_min_sessions or 0), 0)
    selected = in_range
    selection_mode = f"recent_{recent_hours}h"
    if low_sample:
        selected = sorted(sessions, key=lambda session: int(session.get("t0_ms") or 0), reverse=True)[: max(int(recent_min_sessions or 0), 0)]
        selected.sort(key=lambda session: int(session.get("t0_ms") or 0))
        selection_mode = "latest_sessions_fallback"
    selected_start_ms = int(selected[0].get("t0_ms") or start_ms) if selected else start_ms
    return {
        "sessions": selected,
        "low_sample": low_sample,
        "selection_mode": selection_mode,
        "requested_start_ms": start_ms,
        "selected_start_ms": selected_start_ms,
        "end_ms": end_ms,
    }


def _build_idle_cost_summary_for_range(
    payload: dict[str, Any],
    *,
    start_ms: int,
    end_ms: int,
    timezone_name: str,
    sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    official_hourly = dict(payload.get("hourly_official_costs") or {})
    session_counts = _hourly_session_count(sessions, timezone_name)
    idle_hours = []
    for hour_key, cost in sorted(official_hourly.items()):
        try:
            hour_dt = datetime.fromisoformat(str(hour_key).replace("Z", "+00:00"))
        except ValueError:
            continue
        hour_ms = int(hour_dt.timestamp() * 1000)
        if not (start_ms <= hour_ms < end_ms):
            continue
        if int(session_counts.get(hour_key, 0) or 0) != 0:
            continue
        idle_hours.append({"hour_key": hour_key, "official_cost_usd": _decimal_str(_decimal(cost))})
    return {
        "idle_hour_count": len(idle_hours),
        "idle_hours": idle_hours,
        "summary": _summarize_decimal_series(
            _decimal(entry["official_cost_usd"]) for entry in idle_hours if entry.get("official_cost_usd") is not None
        ),
    }


def _build_goal_check_from_sessions(
    payload: dict[str, Any],
    *,
    sessions: list[dict[str, Any]],
    start_ms: int,
    end_ms: int,
    timezone_name: str,
    strict_goal_metric: str,
) -> dict[str, Any]:
    metric_name = "p90" if str(strict_goal_metric or "").strip().lower() != "p50" else "p50"
    metrics = _refresh_session_metrics(sessions)
    allocated_session_costs = _collect_nonnegative_decimals(
        session.get("allocated_cost_usd")
        for session in sessions
        if session.get("allocated_cost_usd") is not None
    )
    modal_session_cost_summary = _summarize_decimal_series(allocated_session_costs)
    idle_cost = _build_idle_cost_summary_for_range(
        payload,
        start_ms=start_ms,
        end_ms=end_ms,
        timezone_name=timezone_name,
        sessions=sessions,
    )
    read_mode = _read_receipt_metric_mode(metrics)
    read_actual = _read_receipt_metric_value(metrics, metric_name)
    reply_actual = _decimal(((metrics.get("t0_to_t3_minus_model_ms") or {}).get(metric_name)))
    idle_actual = _decimal(((idle_cost.get("summary") or {}).get(metric_name)))
    session_cost_actual = _decimal(modal_session_cost_summary.get(metric_name))
    read_value = read_actual if read_actual > 0 else None
    reply_value = reply_actual if reply_actual > 0 else None
    idle_value = idle_actual if idle_actual > 0 or int(idle_cost.get("idle_hour_count") or 0) > 0 else None
    session_cost_value = session_cost_actual if any(cost > 0 for cost in allocated_session_costs) else None
    measurement_modes = sorted(
        {
            str(session.get("reply_minus_ai_measurement_mode") or "").strip()
            for session in sessions
            if str(session.get("reply_minus_ai_measurement_mode") or "").strip()
        }
    )
    return {
        "metric": metric_name,
        "window": {
            "start": datetime.fromtimestamp(start_ms / 1000, tz=dt_timezone.utc).astimezone(_get_timezone(timezone_name)).isoformat(),
            "end": datetime.fromtimestamp(end_ms / 1000, tz=dt_timezone.utc).astimezone(_get_timezone(timezone_name)).isoformat(),
            "session_count": len(sessions),
        },
        "current": {
            "read_receipt_ms": round(float(read_value), 1) if read_value is not None else None,
            "read_receipt_measurement_mode": read_mode,
            "reply_minus_ai_ms": round(float(reply_value), 1) if reply_value is not None else None,
            "modal_idle_hourly_cost_usd": _decimal_str(idle_value) if idle_value is not None else None,
            "modal_session_cost_usd": _decimal_str(session_cost_value) if session_cost_value is not None else None,
        },
        "statuses": {
            "read_receipt_under_5s": _goal_status(actual=read_value, target=Decimal(str(DEFAULT_GOALS["read_receipt_ms"]))),
            "reply_minus_ai_under_20s": _goal_status(actual=reply_value, target=Decimal(str(DEFAULT_GOALS["reply_minus_ai_ms"]))),
            "modal_idle_hourly_cost_under_0_005": _goal_status(actual=idle_value, target=DEFAULT_GOALS["idle_hourly_cost_usd"]),
            "session_cost_under_0_0045": _goal_status(actual=session_cost_value, target=DEFAULT_GOALS["cost_per_session_usd"]),
        },
        "measurement_modes": measurement_modes,
        "idle_cost_detail": idle_cost,
        "session_cost_detail": modal_session_cost_summary,
    }


def _build_internal_exec_stage_pk(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stage_fields = (
        "gateway_run_agent_total_elapsed_ms",
        "provider_wait_elapsed_ms",
        "tool_exec_elapsed_ms",
        "hermes_overhead_elapsed_ms",
    )
    result: dict[str, Any] = {}
    for field in stage_fields:
        current_values = [
            float(session.get(field))
            for session in ((window_payloads.get("current", {}).get("session_summary") or {}).get("sessions") or [])
            if isinstance(session.get(field), (int, float))
        ]
        result[field] = {"current": _summarize_numeric_series(current_values)}
        for label, payload in window_payloads.items():
            if label == "current":
                continue
            compare_key = f"compare_{label.replace('-', '_')}"
            values = [
                float(session.get(field))
                for session in ((payload.get("session_summary") or {}).get("sessions") or [])
                if isinstance(session.get(field), (int, float))
            ]
            result[field][compare_key] = _summarize_numeric_series(values)
    return result


def _build_fallback_reason_pk(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, payload in window_payloads.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for session in ((payload.get("session_summary") or {}).get("sessions") or []):
            reason = str(session.get("fallback_reason") or "").strip() or str(session.get("route_decision_reason") or "").strip()
            if reason:
                grouped[reason].append(session)
        result[label] = [
            {
                "reason": reason,
                "session_count": len(entries),
                "modal_cost_usd": _decimal_str(
                    sum((_decimal(entry.get("allocated_cost_usd")) for entry in entries), Decimal("0"))
                ),
            }
            for reason, entries in sorted(
                grouped.items(),
                key=lambda item: (
                    sum((_decimal(entry.get("allocated_cost_usd")) for entry in item[1]), Decimal("0")),
                    len(item[1]),
                ),
                reverse=True,
            )
        ]
    return result


def _build_request_class_breakdown(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, payload in window_payloads.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for session in ((payload.get("session_summary") or {}).get("sessions") or []):
            request_class = str(session.get("request_class") or "").strip() or "unclassified"
            grouped[request_class].append(session)
        result[label] = [
            {
                "request_class": request_class,
                "session_count": len(entries),
                "gateway_eligible_count": sum(1 for entry in entries if entry.get("gateway_eligible") is True),
                "external_exec_candidate_count": sum(1 for entry in entries if entry.get("external_exec_candidate") is True),
                "modal_cost_usd": _decimal_str(
                    sum((_decimal(entry.get("allocated_cost_usd")) for entry in entries), Decimal("0"))
                ),
            }
            for request_class, entries in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
        ]
    return result


def _build_misroute_breakdown(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, payload in window_payloads.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for session in ((payload.get("session_summary") or {}).get("sessions") or []):
            misroute_detected = bool(session.get("misroute_detected")) or str(session.get("gateway_error_class") or "").strip() == "misrouted_request_class"
            if not misroute_detected:
                continue
            reason = str(session.get("gateway_error_class") or "").strip() or str(session.get("fallback_reason") or "").strip() or "misrouted_request_class"
            grouped[reason].append(session)
        result[label] = [
            {
                "reason": reason,
                "session_count": len(entries),
                "request_classes": sorted({str(entry.get("request_class") or "").strip() or "unclassified" for entry in entries}),
                "session_ids": [str(entry.get("session_id") or "").strip() for entry in entries[:20]],
            }
            for reason, entries in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
        ]
    return result


def _build_tool_image_path_breakdown(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tracked_classes = {
        "tool_browser",
        "tool_non_browser",
        "image_understanding",
        "image_generation",
        "media_hydration",
        "file_or_attachment",
    }
    for label, payload in window_payloads.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for session in ((payload.get("session_summary") or {}).get("sessions") or []):
            request_class = str(session.get("request_class") or "").strip()
            if request_class not in tracked_classes:
                continue
            path_key = "|".join(
                [
                    request_class,
                    str(session.get("route_family") or "").strip() or "unknown_route_family",
                    str(session.get("gateway_route_name") or "").strip() or "modal_only",
                ]
            )
            grouped[path_key].append(session)
        result[label] = [
            {
                "request_class": path_key.split("|")[0],
                "route_family": path_key.split("|")[1],
                "gateway_route_name": path_key.split("|")[2],
                "session_count": len(entries),
                "requires_browser_count": sum(1 for entry in entries if entry.get("requires_browser") is True),
                "requires_tools_count": sum(1 for entry in entries if entry.get("requires_tools") is True),
            }
            for path_key, entries in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
        ]
    return result


def _build_gateway_eligible_accuracy(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    gateway_expected_classes = {"text_plain", "text_coding"}
    for label, payload in window_payloads.items():
        sessions = list(((payload.get("session_summary") or {}).get("sessions") or []))
        scored_sessions = [
            session
            for session in sessions
            if str(session.get("request_class") or "").strip()
            and session.get("gateway_eligible") is not None
        ]
        correct = 0
        for session in scored_sessions:
            request_class = str(session.get("request_class") or "").strip()
            expected = request_class in gateway_expected_classes
            actual = session.get("gateway_eligible") is True
            if expected == actual:
                correct += 1
        result[label] = {
            "sample_count": len(scored_sessions),
            "correct_count": correct,
            "accuracy": _safe_ratio(correct, len(scored_sessions)) if scored_sessions else None,
            "false_positive_count": sum(
                1
                for session in scored_sessions
                if session.get("gateway_eligible") is True
                and str(session.get("request_class") or "").strip() not in gateway_expected_classes
            ),
            "false_negative_count": sum(
                1
                for session in scored_sessions
                if session.get("gateway_eligible") is False
                and str(session.get("request_class") or "").strip() in gateway_expected_classes
            ),
        }
    return result


def _build_fallback_reason_by_request_class(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, payload in window_payloads.items():
        grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for session in ((payload.get("session_summary") or {}).get("sessions") or []):
            request_class = str(session.get("request_class") or "").strip() or "unclassified"
            reason = str(session.get("fallback_reason") or "").strip() or str(session.get("route_decision_reason") or "").strip()
            if not reason:
                continue
            grouped[request_class][reason] += 1
        result[label] = {
            request_class: [
                {"reason": reason, "session_count": count}
                for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)
            ]
            for request_class, reason_counts in sorted(grouped.items())
        }
    return result


def _build_dfmea_breakdown(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, payload in window_payloads.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for session in ((payload.get("session_summary") or {}).get("sessions") or []):
            failure_mode = str(session.get("dfmea_failure_mode") or "").strip()
            if not failure_mode:
                continue
            grouped[failure_mode].append(session)
        result[label] = [
            {
                "failure_mode": failure_mode,
                "session_count": len(entries),
                "control_phases": sorted({str(entry.get("dfmea_control_phase") or "").strip() for entry in entries if str(entry.get("dfmea_control_phase") or "").strip()}),
                "max_severity": max((int(entry.get("dfmea_severity") or 0) for entry in entries), default=0),
                "top_detection_signals": sorted({str(entry.get("dfmea_detection_signal") or "").strip() for entry in entries if str(entry.get("dfmea_detection_signal") or "").strip()}),
            }
            for failure_mode, entries in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
        ]
    return result


def _build_text_route_cache_hit_rate(cloudflare_summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    window_summaries = (cloudflare_summary or {}).get("window_summaries") or {}
    for label, summary in window_summaries.items():
        cache_rows = [
            row for row in ((summary or {}).get("gateway_request_rows") or [])
            if str(row.get("request_class") or "").strip() in {"text_plain", "text_coding"}
            or str(row.get("gateway_route_name") or "").strip() in {"text-general", "text-coding"}
        ]
        hit_count = sum(1 for row in cache_rows if row.get("cf_cache_status") in {True, "HIT", "hit"})
        result[label] = {
            "sample_count": len(cache_rows),
            "hit_count": hit_count,
            "hit_rate": _safe_ratio(hit_count, len(cache_rows)) if cache_rows else None,
        }
    return result


def _build_cache_hit_rate(cloudflare_summary: dict[str, Any], *, eligible_only: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    window_summaries = (cloudflare_summary or {}).get("window_summaries") or {}
    for label, summary in window_summaries.items():
        rows = list((summary or {}).get("gateway_request_rows") or [])
        if eligible_only:
            rows = [row for row in rows if row.get("cache_eligible") is True or row.get("gateway_eligible") is True]
        hit_count = sum(1 for row in rows if row.get("cf_cache_status") in {True, "HIT", "hit"})
        result[label] = {
            "sample_count": len(rows),
            "hit_count": hit_count,
            "hit_rate": _safe_ratio(hit_count, len(rows)) if rows else None,
        }
    return result


def _build_browser_single_ai_call_completion(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, payload in window_payloads.items():
        sessions = list(((payload.get("session_summary") or {}).get("sessions") or []))
        browser_sessions = [
            session
            for session in sessions
            if session.get("requires_browser") is True or str(session.get("route_hint") or "").strip() == "cf_browser_first"
        ]
        measurable = [session for session in browser_sessions if isinstance(session.get("ai_call_count"), int)]
        success_count = sum(
            1
            for session in measurable
            if int(session.get("ai_call_count") or 0) <= 1 and bool(session.get("reply_sent"))
        )
        result[label] = {
            "sample_count": len(measurable),
            "success_count": success_count,
            "completion_rate": _safe_ratio(success_count, len(measurable)) if measurable else None,
        }
    return result


def _build_capability_match_rate(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, payload in window_payloads.items():
        sessions = list(((payload.get("session_summary") or {}).get("sessions") or []))
        measured: list[bool] = []
        for session in sessions:
            explicit = _coerce_optional_bool(session.get("capability_match"))
            if explicit is not None:
                measured.append(explicit)
                continue
            if bool(session.get("misroute_detected")):
                measured.append(False)
        success_count = sum(1 for item in measured if item is True)
        result[label] = {
            "sample_count": len(measured),
            "success_count": success_count,
            "match_rate": _safe_ratio(success_count, len(measured)) if measured else None,
        }
    return result


def _build_preferred_model_selection_accuracy(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, payload in window_payloads.items():
        sessions = list(((payload.get("session_summary") or {}).get("sessions") or []))
        measured = [_coerce_optional_bool(session.get("preferred_model_selected")) for session in sessions]
        measured = [item for item in measured if item is not None]
        success_count = sum(1 for item in measured if item is True)
        result[label] = {
            "sample_count": len(measured),
            "success_count": success_count,
            "accuracy": _safe_ratio(success_count, len(measured)) if measured else None,
        }
    return result


def _build_modal_cost_optimization(window_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current_payload = window_payloads.get("current", {})
    sessions = list(((current_payload.get("session_summary") or {}).get("sessions") or []))
    idle_cost = _build_idle_cost_summary(current_payload)
    internal_exec_sessions = [session for session in sessions if str(session.get("execution_mode") or "").strip() == "modal_heavy_exec"]
    total_internal_exec_cost = sum((_decimal(session.get("allocated_cost_usd")) for session in internal_exec_sessions), Decimal("0"))
    provider_wait_cost = Decimal("0")
    hermes_overhead_cost = Decimal("0")
    for session in internal_exec_sessions:
        allocated_cost = _decimal(session.get("allocated_cost_usd"))
        total_elapsed = session.get("gateway_run_agent_total_elapsed_ms")
        if not isinstance(total_elapsed, (int, float)) or float(total_elapsed) <= 0:
            continue
        provider_wait = float(session.get("provider_wait_elapsed_ms") or 0)
        hermes_overhead = float(session.get("hermes_overhead_elapsed_ms") or 0)
        provider_wait_cost += allocated_cost * Decimal(str(max(provider_wait, 0.0) / float(total_elapsed)))
        hermes_overhead_cost += allocated_cost * Decimal(str(max(hermes_overhead, 0.0) / float(total_elapsed)))
    fallback_breakdown = _build_fallback_reason_pk({"current": current_payload}).get("current") or []
    return {
        "modal_session_cost_p50": (_build_goal_assessment({"current": current_payload}).get("session_cost_detail") or {}).get("p50"),
        "modal_session_cost_p90": (_build_goal_assessment({"current": current_payload}).get("session_cost_detail") or {}).get("p90"),
        "modal_idle_hourly_cost_p50": (idle_cost.get("summary") or {}).get("p50"),
        "modal_idle_hourly_cost_p90": (idle_cost.get("summary") or {}).get("p90"),
        "internal_agent_exec_provider_wait_cost_share": _decimal_str(
            (provider_wait_cost / total_internal_exec_cost) if total_internal_exec_cost > 0 else None
        ),
        "internal_agent_exec_hermes_overhead_cost_share": _decimal_str(
            (hermes_overhead_cost / total_internal_exec_cost) if total_internal_exec_cost > 0 else None
        ),
        "fallback_reason_cost_breakdown": fallback_breakdown,
        "candidate_to_modal_fallback_rate": ((current_payload.get("execution_path_summary") or {}).get("external_exec_candidate_modal_fallback_rate")),
    }


def _build_recent_window_eval(
    window_payloads: dict[str, dict[str, Any]],
    *,
    recent_hours: int,
    recent_min_sessions: int,
    timezone_name: str,
    strict_goal_metric: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_payload = window_payloads.get("current", {})
    selection = _select_recent_sessions(
        current_payload,
        recent_hours=recent_hours,
        recent_min_sessions=recent_min_sessions,
    )
    sessions = selection.get("sessions") or []
    goal_check = _build_goal_check_from_sessions(
        current_payload,
        sessions=sessions,
        start_ms=int(selection.get("selected_start_ms") or selection.get("requested_start_ms") or 0),
        end_ms=int(selection.get("end_ms") or 0),
        timezone_name=timezone_name,
        strict_goal_metric=strict_goal_metric,
    )
    return (
        {
            "selection_mode": selection.get("selection_mode"),
            "low_sample": bool(selection.get("low_sample")),
            "requested_recent_hours": int(recent_hours),
            "recent_min_sessions": int(recent_min_sessions),
            "selected_session_count": len(sessions),
            "selected_window_start": goal_check.get("window", {}).get("start"),
            "selected_window_end": goal_check.get("window", {}).get("end"),
            "metrics": _refresh_session_metrics(sessions),
            "measurement_modes": goal_check.get("measurement_modes") or [],
        },
        goal_check,
    )


def _refresh_session_metrics(sessions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for metric_name in (
        "t0_to_t1_ms",
        "t0_to_t2_ms",
        "t0_to_t3_ms",
        "reply_to_read_ms",
        "t0_to_t3_minus_model_ms",
        "t1_to_t3_ms",
        "t2_to_t3_ms",
    ):
        metrics[metric_name] = _summarize_numeric_series(
            session.get(metric_name) for session in sessions if isinstance(session.get(metric_name), (int, float))
        )
    return metrics


def _refresh_session_summary_payload(
    payload: dict[str, Any],
    *,
    timezone_name: str,
) -> None:
    session_summary = payload.get("session_summary") or {}
    sessions = session_summary.get("sessions") or []
    session_summary["metrics"] = _refresh_session_metrics(sessions)
    completion = {
        "session_count": len(sessions),
        "read_receipt_count": sum(1 for session in sessions if session.get("read_matched")),
        "reply_sent_count": sum(1 for session in sessions if session.get("reply_sent")),
        "no_read_count": sum(1 for session in sessions if not session.get("read_matched")),
        "no_reply_count": sum(1 for session in sessions if not session.get("reply_sent")),
    }
    completion["read_receipt_rate"] = _safe_ratio(completion["read_receipt_count"], completion["session_count"])
    completion["reply_sent_rate"] = _safe_ratio(completion["reply_sent_count"], completion["session_count"])
    session_summary["completion"] = completion
    session_summary["slowest_sessions"] = sorted(
        [session for session in sessions if isinstance(session.get("t0_to_t3_ms"), (int, float))],
        key=lambda session: float(session.get("t0_to_t3_ms") or 0),
        reverse=True,
    )[:10]
    payload["hourly_session_counts"] = _hourly_session_count(sessions, timezone_name)


def _apply_cloudflare_observability(
    window_payloads: dict[str, dict[str, Any]],
    cloudflare_summary: dict[str, Any],
    *,
    timezone_name: str,
) -> None:
    window_summaries = cloudflare_summary.get("window_summaries") or {}
    for label, payload in window_payloads.items():
        session_summary = payload.get("session_summary") or {}
        if not isinstance(session_summary, dict):
            session_summary = {}
        payload["session_summary"] = session_summary
        sessions = session_summary.get("sessions")
        if not isinstance(sessions, list):
            sessions = []
            session_summary["sessions"] = sessions
        session_key_map = session_summary.get("session_key_map")
        if not isinstance(session_key_map, dict):
            session_key_map = {}
            session_summary["session_key_map"] = session_key_map
        sessions_by_event_id = {
            str(session.get("event_id") or "").strip(): session
            for session in sessions
            if str(session.get("event_id") or "").strip()
        }
        cf_session_model = ((window_summaries.get(label) or {}).get("matched_session_model_elapsed_ms") or {})
        cf_session_costs = ((window_summaries.get(label) or {}).get("matched_session_gateway_costs_usd") or {})
        updated = False
        for session in sessions:
            session_id = str(session.get("session_id") or "").strip()
            model_elapsed = cf_session_model.get(session_id)
            if isinstance(model_elapsed, (int, float)):
                existing_provider_wait = session.get("provider_wait_elapsed_ms")
                existing_ai_elapsed = session.get("ai_model_elapsed_ms")
                if not isinstance(existing_provider_wait, int) and not isinstance(existing_ai_elapsed, (int, float)):
                    session["ai_model_elapsed_ms"] = round(float(model_elapsed), 1)
                t0_to_t3_ms = session.get("t0_to_t3_ms")
                effective_ai_elapsed = (
                    existing_provider_wait
                    if isinstance(existing_provider_wait, int)
                    else session.get("ai_model_elapsed_ms")
                )
                if isinstance(t0_to_t3_ms, (int, float)):
                    if isinstance(effective_ai_elapsed, (int, float)):
                        session["t0_to_t3_minus_model_ms"] = max(float(t0_to_t3_ms) - float(effective_ai_elapsed), 0.0)
                updated = True
            if session_id in cf_session_costs:
                session["cloudflare_ai_cost_usd"] = cf_session_costs[session_id]
                updated = True
        worker_session_facts = ((window_summaries.get(label) or {}).get("worker_session_facts") or {})
        for event_id, fact in worker_session_facts.items():
            session = sessions_by_event_id.get(event_id)
            t0_ms = _normalized_epoch_ms(fact.get("t0_ms"))
            t3_ms = _normalized_epoch_ms(fact.get("t3_ms"))
            if session is None:
                if t0_ms is None and t3_ms is None:
                    continue
                session_key = str(fact.get("session_key") or "").strip()
                correlation_id = str(fact.get("correlation_id") or "").strip()
                session = {
                    "session_id": _session_id_for_event(event_id, ""),
                    "event_id": event_id,
                    "message_id": str(fact.get("message_id") or "").strip(),
                    "session_key": session_key,
                    "reply_send_message_id": str(fact.get("reply_send_message_id") or "").strip(),
                    "reply_send_message_ids": list(fact.get("reply_send_message_ids") or []),
                    "history_visible_mode": HISTORICAL_T3_PROXY_LABEL,
                    "measurement_mode": "cloudflare_worker_observability",
                    "t0_ms": t0_ms,
                    "t1_ms": None,
                    "t2_ms": None,
                    "reply_read_ms": None,
                    "t3_ms": t3_ms,
                    "ack_kind": "worker_observability_only",
                    "ingress_strategy": "worker_observability_only",
                    "execution_mode": str(fact.get("execution_mode") or "").strip() or "unspecified",
                    "reply_visible_proxy": fact.get("reply_visible_proxy"),
                    "t0_to_t1_ms": None,
                    "t0_to_t2_ms": None,
                    "t0_to_t3_ms": (t3_ms - t0_ms) if t3_ms is not None and t0_ms is not None else None,
                    "t0_to_t3_minus_model_ms": None,
                    "t1_to_t3_ms": None,
                    "t2_to_t3_ms": None,
                    "ai_model_elapsed_ms": None,
                    "non_model_elapsed_ms": None,
                    "allocated_cost_usd": _decimal_str(Decimal("0")),
                    "provider_billed_cost_usd": None,
                    "cloudflare_ai_cost_usd": fact.get("cloudflare_ai_cost_usd"),
                    "external_exec_candidate": fact.get("external_exec_candidate"),
                    "request_class": str(fact.get("request_class") or "").strip(),
                    "content_modalities": _coerce_string_list(fact.get("content_modalities")),
                    "route_family": str(fact.get("route_family") or "").strip(),
                    "gateway_route_name": str(fact.get("gateway_route_name") or "").strip(),
                    "gateway_eligible": fact.get("gateway_eligible"),
                    "requires_tools": fact.get("requires_tools"),
                    "requires_browser": fact.get("requires_browser"),
                    "requires_media_hydration": fact.get("requires_media_hydration"),
                    "toolset": _coerce_string_list(fact.get("toolset")),
                    "modality_profile": str(fact.get("modality_profile") or "").strip(),
                    "gateway_error_class": str(fact.get("gateway_error_class") or "").strip(),
                    "misroute_detected": bool(fact.get("misroute_detected")),
                    "dfmea_failure_mode": str(fact.get("dfmea_failure_mode") or "").strip(),
                    "dfmea_control_phase": str(fact.get("dfmea_control_phase") or "").strip(),
                    "dfmea_detection_signal": str(fact.get("dfmea_detection_signal") or "").strip(),
                    "dfmea_control_action": str(fact.get("dfmea_control_action") or "").strip(),
                    "dfmea_severity": fact.get("dfmea_severity"),
                    "read_matched": False,
                    "read_match_message_id": "",
                    "read_match_source": "",
                    "read_receipt_measurement_mode": "",
                    "reply_read_match_message_id": "",
                    "reply_read_match_source": "",
                    "reply_sent": t3_ms is not None,
                    "correlation_id": correlation_id,
                }
                sessions.append(session)
                sessions_by_event_id[event_id] = session
                if session_key:
                    session_key_map[session_key] = str(session.get("session_id") or "")
                updated = True
            if session.get("measurement_mode") == "modal_internal_exec_only" and t3_ms is not None:
                session["measurement_mode"] = "cloudflare_worker_observability"
            if session.get("t0_ms") is None and t0_ms is not None:
                session["t0_ms"] = t0_ms
                updated = True
            if session.get("t3_ms") is None and t3_ms is not None:
                session["t3_ms"] = t3_ms
                session["reply_visible_proxy"] = fact.get("reply_visible_proxy")
                session["reply_sent"] = True
                updated = True
            if not str(session.get("session_key") or "").strip() and str(fact.get("session_key") or "").strip():
                session["session_key"] = str(fact.get("session_key") or "").strip()
                updated = True
            if not str(session.get("message_id") or "").strip() and str(fact.get("message_id") or "").strip():
                session["message_id"] = str(fact.get("message_id") or "").strip()
                updated = True
            fact_send_message_ids = _unique_nonempty_strings(
                [fact.get("reply_send_message_id")] + list(fact.get("reply_send_message_ids") or [])
            )
            if fact_send_message_ids:
                existing_send_message_ids = _unique_nonempty_strings(
                    [session.get("reply_send_message_id")] + list(session.get("reply_send_message_ids") or [])
                )
                merged_send_message_ids = _unique_nonempty_strings(existing_send_message_ids + fact_send_message_ids)
                if merged_send_message_ids != existing_send_message_ids:
                    session["reply_send_message_ids"] = merged_send_message_ids
                    session["reply_send_message_id"] = merged_send_message_ids[0]
                    updated = True
            execution_mode = str(fact.get("execution_mode") or "").strip()
            if execution_mode and str(session.get("execution_mode") or "").strip() in {"", "unspecified", "modal_heavy_exec"}:
                session["execution_mode"] = execution_mode
                updated = True
            if session.get("external_exec_candidate") is None and isinstance(fact.get("external_exec_candidate"), bool):
                session["external_exec_candidate"] = bool(fact.get("external_exec_candidate"))
                updated = True
            if not str(session.get("request_class") or "").strip() and str(fact.get("request_class") or "").strip():
                session["request_class"] = str(fact.get("request_class") or "").strip()
                updated = True
            if not list(session.get("content_modalities") or []) and list(fact.get("content_modalities") or []):
                session["content_modalities"] = _coerce_string_list(fact.get("content_modalities"))
                updated = True
            if not str(session.get("route_family") or "").strip() and str(fact.get("route_family") or "").strip():
                session["route_family"] = str(fact.get("route_family") or "").strip()
                updated = True
            if not str(session.get("gateway_route_name") or "").strip() and str(fact.get("gateway_route_name") or "").strip():
                session["gateway_route_name"] = str(fact.get("gateway_route_name") or "").strip()
                updated = True
            if session.get("gateway_eligible") is None and fact.get("gateway_eligible") is not None:
                session["gateway_eligible"] = bool(fact.get("gateway_eligible"))
                updated = True
            if session.get("requires_tools") is None and fact.get("requires_tools") is not None:
                session["requires_tools"] = bool(fact.get("requires_tools"))
                updated = True
            if session.get("requires_browser") is None and fact.get("requires_browser") is not None:
                session["requires_browser"] = bool(fact.get("requires_browser"))
                updated = True
            if session.get("requires_media_hydration") is None and fact.get("requires_media_hydration") is not None:
                session["requires_media_hydration"] = bool(fact.get("requires_media_hydration"))
                updated = True
            if not str(session.get("route_hint") or "").strip() and str(fact.get("route_hint") or "").strip():
                session["route_hint"] = str(fact.get("route_hint") or "").strip()
                updated = True
            if not str(session.get("route_version") or "").strip() and str(fact.get("route_version") or "").strip():
                session["route_version"] = str(fact.get("route_version") or "").strip()
                updated = True
            if not str(session.get("provider_alias") or "").strip() and str(fact.get("provider_alias") or "").strip():
                session["provider_alias"] = str(fact.get("provider_alias") or "").strip()
                updated = True
            if not str(session.get("model_catalog_version") or "").strip() and str(fact.get("model_catalog_version") or "").strip():
                session["model_catalog_version"] = str(fact.get("model_catalog_version") or "").strip()
                updated = True
            if session.get("cache_eligible") is None and fact.get("cache_eligible") is not None:
                session["cache_eligible"] = bool(fact.get("cache_eligible"))
                updated = True
            if not str(session.get("cache_status") or "").strip() and str(fact.get("cache_status") or "").strip():
                session["cache_status"] = str(fact.get("cache_status") or "").strip()
                updated = True
            if not list(session.get("toolset") or []) and list(fact.get("toolset") or []):
                session["toolset"] = _coerce_string_list(fact.get("toolset"))
                updated = True
            if not str(session.get("modality_profile") or "").strip() and str(fact.get("modality_profile") or "").strip():
                session["modality_profile"] = str(fact.get("modality_profile") or "").strip()
                updated = True
            if not str(session.get("gateway_error_class") or "").strip() and str(fact.get("gateway_error_class") or "").strip():
                session["gateway_error_class"] = str(fact.get("gateway_error_class") or "").strip()
                updated = True
            if not bool(session.get("misroute_detected")) and bool(fact.get("misroute_detected")):
                session["misroute_detected"] = True
                updated = True
            if not str(session.get("dfmea_failure_mode") or "").strip() and str(fact.get("dfmea_failure_mode") or "").strip():
                session["dfmea_failure_mode"] = str(fact.get("dfmea_failure_mode") or "").strip()
                updated = True
            if not str(session.get("dfmea_control_phase") or "").strip() and str(fact.get("dfmea_control_phase") or "").strip():
                session["dfmea_control_phase"] = str(fact.get("dfmea_control_phase") or "").strip()
                updated = True
            if not str(session.get("dfmea_detection_signal") or "").strip() and str(fact.get("dfmea_detection_signal") or "").strip():
                session["dfmea_detection_signal"] = str(fact.get("dfmea_detection_signal") or "").strip()
                updated = True
            if not str(session.get("dfmea_control_action") or "").strip() and str(fact.get("dfmea_control_action") or "").strip():
                session["dfmea_control_action"] = str(fact.get("dfmea_control_action") or "").strip()
                updated = True
            if session.get("dfmea_severity") is None and isinstance(fact.get("dfmea_severity"), int):
                session["dfmea_severity"] = int(fact.get("dfmea_severity"))
                updated = True
            if session.get("ai_model_elapsed_ms") is None and isinstance(fact.get("ai_model_elapsed_ms"), (int, float)):
                session["ai_model_elapsed_ms"] = round(float(fact["ai_model_elapsed_ms"]), 1)
                updated = True
            if session.get("cloudflare_ai_cost_usd") is None and fact.get("cloudflare_ai_cost_usd") not in (None, ""):
                session["cloudflare_ai_cost_usd"] = fact.get("cloudflare_ai_cost_usd")
                updated = True
            if session.get("feedback_score_before") in (None, "") and fact.get("feedback_score_before") not in (None, ""):
                session["feedback_score_before"] = fact.get("feedback_score_before")
                updated = True
            if session.get("feedback_score_after") in (None, "") and fact.get("feedback_score_after") not in (None, ""):
                session["feedback_score_after"] = fact.get("feedback_score_after")
                updated = True
            if session.get("ai_call_count") is None and isinstance(fact.get("ai_call_count"), int):
                session["ai_call_count"] = int(fact.get("ai_call_count"))
                updated = True
            if session.get("capability_match") is None and fact.get("capability_match") is not None:
                session["capability_match"] = bool(fact.get("capability_match"))
                updated = True
            if session.get("preferred_model_selected") is None and fact.get("preferred_model_selected") is not None:
                session["preferred_model_selected"] = bool(fact.get("preferred_model_selected"))
                updated = True
            t0_current = session.get("t0_ms")
            t3_current = session.get("t3_ms")
            if isinstance(t0_current, int) and isinstance(t3_current, int):
                session["t0_to_t3_ms"] = t3_current - t0_current
                ai_model_elapsed = session.get("ai_model_elapsed_ms")
                if isinstance(ai_model_elapsed, (int, float)):
                    session["t0_to_t3_minus_model_ms"] = max(float(session["t0_to_t3_ms"]) - float(ai_model_elapsed), 0.0)
                updated = True
        read_event_facts = ((window_summaries.get(label) or {}).get("read_event_facts") or [])
        if _apply_read_receipt_rows_to_sessions(sessions, list(read_event_facts), source_label="cloudflare_worker_message_read"):
            updated = True
        if updated:
            _rebuild_session_total_costs(sessions)
            _refresh_session_summary_payload(payload, timezone_name=timezone_name)


def _build_optimization_analysis(
    *,
    window_payloads: dict[str, dict[str, Any]],
    hourly_cost_pk: list[dict[str, Any]],
) -> dict[str, Any]:
    current_payload = window_payloads.get("current", {})
    current_functions = current_payload.get("function_summary") or {}
    current_container_summary = current_payload.get("container_summary") or {}
    current_session_summary = current_payload.get("session_summary") or {}

    cost_hotspots = []
    for entry in sorted(hourly_cost_pk, key=lambda row: _decimal((row.get("current") or {}).get("cost_usd")), reverse=True)[:5]:
        compare_keys = [key for key in entry if key.startswith("compare_")]
        max_delta = None
        for key in compare_keys:
            delta = _decimal(((entry.get(key) or {}).get("delta_cost_usd")))
            if max_delta is None or delta > max_delta:
                max_delta = delta
        cost_hotspots.append(
            {
                "hour_key": entry.get("hour_key"),
                "current_cost_usd": (entry.get("current") or {}).get("cost_usd"),
                "current_session_count": (entry.get("current") or {}).get("session_count"),
                "largest_delta_cost_usd": _decimal_str(max_delta) if isinstance(max_delta, Decimal) else None,
            }
        )

    expensive_functions = []
    for function_name, summary in sorted(
        current_functions.items(),
        key=lambda item: (_decimal(item[1].get("cost_per_session_usd")), _decimal(item[1].get("calibrated_cost_usd"))),
        reverse=True,
    ):
        baseline_values = []
        for label, payload in window_payloads.items():
            if label == "current":
                continue
            baseline = (payload.get("function_summary") or {}).get(function_name, {})
            if baseline.get("cost_per_session_usd") is not None:
                baseline_values.append(_decimal(baseline.get("cost_per_session_usd")))
        baseline_avg = sum(baseline_values, Decimal("0")) / len(baseline_values) if baseline_values else None
        expensive_functions.append(
            {
                "function_name": function_name,
                "current_cost_per_session_usd": summary.get("cost_per_session_usd"),
                "current_total_cost_usd": summary.get("calibrated_cost_usd"),
                "baseline_avg_cost_per_session_usd": _decimal_str(baseline_avg) if baseline_avg is not None else None,
                "cost_per_session_delta_pct": _safe_pct_delta(_decimal(summary.get("cost_per_session_usd")), baseline_avg),
                "cold_start_rate": summary.get("cold_start_rate"),
            }
        )
    expensive_functions = expensive_functions[:5]

    container_findings = []
    for container in (current_container_summary.get("hot_containers") or [])[:5]:
        issue_tags = []
        if float(container.get("cold_start_rate") or 0) >= 0.5:
            issue_tags.append("cold_start_heavy")
        if bool(container.get("single_use_container")):
            issue_tags.append("single_use")
        container_findings.append(
            {
                "worker_boot_id": container.get("worker_boot_id"),
                "function_name": container.get("function_name"),
                "invocation_count": container.get("invocation_count"),
                "total_cost_usd": container.get("total_cost_usd"),
                "cold_start_rate": container.get("cold_start_rate"),
                "issue_tags": issue_tags,
            }
        )

    current_t3 = _decimal(((current_session_summary.get("metrics") or {}).get("t0_to_t3_ms") or {}).get("p90"))
    baseline_t3_values = []
    baseline_cost_values = []
    for label, payload in window_payloads.items():
        if label == "current":
            continue
        baseline_t3 = _decimal((((payload.get("session_summary") or {}).get("metrics") or {}).get("t0_to_t3_ms") or {}).get("p90"))
        baseline_cost = _decimal((((payload.get("billing_summary") or {}).get("total_cost_usd"))))
        if baseline_t3 > 0:
            baseline_t3_values.append(baseline_t3)
        if baseline_cost >= 0:
            baseline_cost_values.append(baseline_cost)
    baseline_t3_avg = sum(baseline_t3_values, Decimal("0")) / len(baseline_t3_values) if baseline_t3_values else None
    baseline_cost_avg = sum(baseline_cost_values, Decimal("0")) / len(baseline_cost_values) if baseline_cost_values else None
    current_cost_total = _decimal((current_payload.get("billing_summary") or {}).get("total_cost_usd"))

    current_measurement_modes = {
        str(session.get("measurement_mode") or "").strip()
        for session in (current_session_summary.get("sessions") or [])
        if str(session.get("measurement_mode") or "").strip()
    }
    comparable_latency_windows = 0
    verdict = "insufficient_data"
    if baseline_t3_avg is not None and baseline_cost_avg is not None and baseline_t3_avg > 0:
        comparable_latency_windows = len(baseline_t3_values)
        latency_delta_pct = _safe_pct_delta(current_t3, baseline_t3_avg)
        cost_delta_pct = _safe_pct_delta(current_cost_total, baseline_cost_avg)
        if latency_delta_pct is not None and cost_delta_pct is not None:
            if latency_delta_pct <= -10 and cost_delta_pct <= 10:
                verdict = "optimization_validated"
            elif latency_delta_pct >= 0 and cost_delta_pct > 10:
                verdict = "optimization_not_validated"
            else:
                verdict = "optimization_mixed"
    if verdict == "optimization_validated" and (
        comparable_latency_windows < 2 or current_measurement_modes == {"modal_internal_exec_only"}
    ):
        verdict = "optimization_partially_validated"

    return {
        "cost_hotspots": cost_hotspots,
        "function_efficiency": expensive_functions,
        "container_efficiency": container_findings,
        "latency_bottlenecks": {
            "t0_to_t1_ms": ((current_session_summary.get("metrics") or {}).get("t0_to_t1_ms") or {}),
            "t0_to_t2_ms": ((current_session_summary.get("metrics") or {}).get("t0_to_t2_ms") or {}),
            "t0_to_t3_ms": ((current_session_summary.get("metrics") or {}).get("t0_to_t3_ms") or {}),
            "t0_to_t3_minus_model_ms": ((current_session_summary.get("metrics") or {}).get("t0_to_t3_minus_model_ms") or {}),
        },
        "execution_path": (current_payload.get("execution_path_summary") or {}),
        "roi_assessment": {
            "verdict": verdict,
            "confidence": (
                "partial"
                if comparable_latency_windows < 2 or current_measurement_modes == {"modal_internal_exec_only"}
                else "high"
            ),
            "current_measurement_modes": sorted(current_measurement_modes),
            "comparable_latency_window_count": comparable_latency_windows,
            "current_total_cost_usd": _decimal_str(current_cost_total),
            "baseline_avg_total_cost_usd": _decimal_str(baseline_cost_avg) if baseline_cost_avg is not None else None,
            "current_t0_to_t3_p90_ms": ((current_session_summary.get("metrics") or {}).get("t0_to_t3_ms") or {}).get("p90"),
            "baseline_avg_t0_to_t3_p90_ms": round(float(baseline_t3_avg), 1) if baseline_t3_avg is not None else None,
            "cost_delta_pct": _safe_pct_delta(current_cost_total, baseline_cost_avg),
            "latency_delta_pct": _safe_pct_delta(current_t3, baseline_t3_avg),
        },
        "notes": [
            "Optimization analysis emphasizes cost hotspots, cost-per-session regressions, cold-start waste, and whether added spend bought back latency.",
            f"T3 uses the historical proxy mode `{HISTORICAL_T3_PROXY_LABEL}` and should be interpreted as send-success visibility, not client render completion.",
        ],
    }


def _render_markdown_summary(*, report: dict[str, Any]) -> str:
    def _fixed_window_line(window_label: str) -> str:
        window_payload = fixed_window_rollup.get(window_label) or {}
        current_payload = window_payload.get("current") or {}
        baseline_payload = (window_payload.get("baselines") or {}).get("d-1") or {}
        trend_payload = window_payload.get("trend") or {}
        return (
            f"- {window_label} overall trend: `{trend_payload.get('overall')}` "
            f"current sessions `{current_payload.get('session_count')}` "
            f"vs d-1 `{baseline_payload.get('session_count')}`"
        )

    current_window = (report.get("windows") or {}).get("current") or {}
    billing = (report.get("window_details") or {}).get("current", {}).get("billing_summary") or {}
    current_sessions = (((report.get("window_details") or {}).get("current", {}).get("session_summary") or {}).get("completion") or {})
    current_t3 = ((((report.get("window_details") or {}).get("current", {}).get("session_summary") or {}).get("metrics") or {}).get("t0_to_t3_ms") or {})
    current_t3_minus_ai = (
        (((report.get("window_details") or {}).get("current", {}).get("session_summary") or {}).get("metrics") or {}).get("t0_to_t3_minus_model_ms")
        or {}
    )
    optimization = report.get("optimization_analysis") or {}
    goals = report.get("goal_assessment") or {}
    recent_goals = report.get("goal_check_recent") or {}
    recent_current = recent_goals.get("current") or {}
    recent_statuses = recent_goals.get("statuses") or {}
    recent_window_eval = report.get("recent_window_eval") or {}
    fixed_window_rollup = report.get("fixed_window_rollup") or {}
    hottest_hour = ((optimization.get("cost_hotspots") or [None])[0]) or {}
    hottest_function = ((optimization.get("function_efficiency") or [None])[0]) or {}
    verdict = ((optimization.get("roi_assessment") or {}).get("verdict")) or "insufficient_data"
    current_goals = goals.get("current") or {}
    statuses = goals.get("statuses") or {}
    lines = [
        "# Feishu-Cloudflare-Modal 16h PK Report",
        "",
        f"- Current window: `{current_window.get('start')}` -> `{current_window.get('end')}`",
        f"- Historical T3 mode: `{report.get('history_visible_mode')}`",
        f"- Current official Modal cost: `${billing.get('total_cost_usd')}`",
        f"- Current session count: `{current_sessions.get('session_count')}`",
        f"- Current T0->T3 p90: `{current_t3.get('p90')}` ms",
        f"- Current T0->T3 minus AI p90: `{current_t3_minus_ai.get('p90')}` ms",
        f"- Optimization verdict: `{verdict}`",
        f"- Recent evaluation mode: `{recent_window_eval.get('selection_mode')}` low_sample=`{recent_window_eval.get('low_sample')}` sessions=`{recent_window_eval.get('selected_session_count')}`",
        "",
        "## Fixed Windows",
        _fixed_window_line("1h"),
        _fixed_window_line("24h"),
        _fixed_window_line("72h"),
        "",
        "## Goal Check",
        f"- Read receipt <5s: `{statuses.get('read_receipt_under_5s')}` (current p90 `{current_goals.get('read_receipt_p90_ms')}` ms)",
        f"- Reply minus AI <20s: `{statuses.get('reply_minus_ai_under_20s')}` (current p90 `{current_goals.get('reply_minus_ai_p90_ms')}` ms)",
        f"- Modal idle hourly cost <0.005: `{statuses.get('modal_idle_hourly_cost_under_0_005')}` (current p90 `${current_goals.get('idle_hourly_p90_cost_usd')}`)",
        f"- Session cost <0.0045: `{statuses.get('session_cost_under_0_0045')}` (current p90 `${current_goals.get('session_cost_p90_usd')}`, avg `${current_goals.get('avg_cost_per_session_usd')}`)",
        f"- AI Gateway cache eligible hit rate >30%: `{statuses.get('cache_hit_rate_over_0_30')}` (current `{current_goals.get('cache_eligible_hit_rate')}`)",
        f"- Browser single AI completion >50%: `{statuses.get('browser_single_ai_call_completion_rate_over_0_50')}` (current `{current_goals.get('browser_single_ai_call_completion_rate')}`)",
        f"- Capability match =100%: `{statuses.get('capability_match_rate_equals_1_00')}` (current `{current_goals.get('capability_match_rate')}`)",
        f"- Preferred model accuracy >=95%: `{statuses.get('preferred_model_selection_accuracy_over_0_95')}` (current `{current_goals.get('preferred_model_selection_accuracy')}`)",
        "",
        "## Recent Goal Check",
        f"- Recent read receipt <5s: `{recent_statuses.get('read_receipt_under_5s')}` ({recent_goals.get('metric')} `{recent_current.get('read_receipt_ms')}` ms)",
        f"- Recent reply minus AI <20s: `{recent_statuses.get('reply_minus_ai_under_20s')}` ({recent_goals.get('metric')} `{recent_current.get('reply_minus_ai_ms')}` ms)",
        f"- Recent Modal idle hourly cost <0.005: `{recent_statuses.get('modal_idle_hourly_cost_under_0_005')}` ({recent_goals.get('metric')} `${recent_current.get('modal_idle_hourly_cost_usd')}`)",
        f"- Recent Modal session cost <0.0045: `{recent_statuses.get('session_cost_under_0_0045')}` ({recent_goals.get('metric')} `${recent_current.get('modal_session_cost_usd')}`)",
        "",
        "## Cost Hotspots",
        f"- Top hour: `{hottest_hour.get('hour_key')}` cost `${hottest_hour.get('current_cost_usd')}` with `{hottest_hour.get('current_session_count')}` sessions",
        f"- Highest function cost/session: `{hottest_function.get('function_name')}` => `${hottest_function.get('current_cost_per_session_usd')}` per session",
        "",
        "## Cloudflare",
        f"- Status: `{(report.get('cloudflare_summary') or {}).get('status')}`",
        f"- Join level: `{(report.get('cloudflare_summary') or {}).get('join_level')}`",
        "",
        "## Data Gaps",
    ]
    data_gaps = report.get("data_gaps") or []
    if data_gaps:
        lines.extend([f"- {gap}" for gap in data_gaps[:10]])
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def _serialize_for_json(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _decimal_str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    return value


def _build_window_report(
    *,
    window: AnalysisWindow,
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    timezone_name: str,
    event_type: str,
    experiment_label: str,
    app_name_filter: str,
    snapshot_profile: str,
    include_duplicates: bool,
    history_visible_mode: str,
    billing_descriptions: set[str],
    cost_report_module,
    modal_module,
) -> dict[str, Any]:
    perf_summary = _build_window_perf_summary(
        modal_module,
        rows,
        window=window,
        event_type=event_type,
        experiment_label=experiment_label,
        app_name_filter=app_name_filter,
        snapshot_profile=snapshot_profile,
        include_duplicates=include_duplicates,
    )
    billing_rows = cost_report_module._run_modal_billing_report(window.start, window.end)
    billing_summary = cost_report_module._build_billing_summary(billing_rows, billing_descriptions)
    cost_items = _build_cost_items(
        rows,
        window=window,
        timezone_name=timezone_name,
        cost_report_module=cost_report_module,
        modal_module=modal_module,
    )
    official_hourly_costs = _build_official_hourly_costs(
        billing_summary,
        window=window,
        timezone_name=timezone_name,
    )
    cost_calibration = _calibrate_cost_items_by_hour(cost_items, official_hourly_costs=official_hourly_costs)
    session_summary = _build_session_timelines(
        all_rows,
        window=window,
        event_type=event_type,
        include_duplicates=include_duplicates,
        history_visible_mode=history_visible_mode,
    )
    _attach_session_costs(cost_calibration["items"], session_summary["sessions"])
    _rebuild_session_total_costs(session_summary["sessions"])
    return {
        "window": {
            "label": window.label,
            "compare_day_shift": window.compare_day_shift,
            "start": _to_iso(window.start),
            "end": _to_iso(window.end),
            "hours": window.hours,
        },
        "perf_summary": perf_summary,
        "billing_summary": billing_summary,
        "cost_calibration": cost_calibration,
        "function_summary": _summarize_functions(cost_calibration["items"]),
        "container_summary": _summarize_containers(cost_calibration["items"]),
        "session_summary": session_summary,
        "execution_path_summary": _build_execution_path_summary(rows),
        "hourly_official_costs": official_hourly_costs,
        "hourly_session_counts": _hourly_session_count(session_summary["sessions"], timezone_name),
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    _apply_default_proxy_env()
    parser = argparse.ArgumentParser(description="Build a Feishu-Cloudflare-Modal 16h PK report with cost and latency optimization analysis.")
    parser.add_argument("--app", default="hermes-agent")
    parser.add_argument("--hours", type=int, default=16)
    parser.add_argument("--compare-days", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--event-type", default="im.message.receive_v1")
    parser.add_argument("--history-visible-mode", default=HISTORICAL_T3_PROXY_LABEL)
    parser.add_argument("--billing-description", action="append", default=[])
    parser.add_argument("--trace-limit", type=int, default=200000)
    parser.add_argument("--experiment-label", default="")
    parser.add_argument("--snapshot-profile", default="")
    parser.add_argument("--app-name-filter", default="")
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument("--cloudflare-log", action="append", default=[])
    parser.add_argument("--trace-file", default="")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--now", default="")
    parser.add_argument("--recent-hours", type=int, default=3)
    parser.add_argument("--recent-min-sessions", type=int, default=20)
    parser.add_argument("--strict-goal-metric", default="p90")
    args = parser.parse_args()

    now = _parse_now(args.now or None, args.timezone)
    windows = _build_analysis_windows(
        hours=args.hours,
        compare_days=args.compare_days,
        timezone_name=args.timezone,
        now=now,
    )
    cost_report_module = _load_cost_report_module()
    modal_module = _load_modal_module()
    trace_file = Path(args.trace_file).resolve() if str(args.trace_file or "").strip() else None
    all_rows = (
        _read_trace_rows_from_file(trace_file, limit=0)
        if trace_file is not None and trace_file.exists()
        else _read_trace_rows(args.app, args.trace_limit)
    )
    billing_descriptions = _match_billing_descriptions(args.app, args.billing_description)

    window_payloads: dict[str, dict[str, Any]] = {}
    data_gaps: list[str] = []
    for label, window in windows.items():
        window_rows = _filter_trace_rows(all_rows, window)
        window_payload = _build_window_report(
            window=window,
            rows=window_rows,
            all_rows=all_rows,
            timezone_name=args.timezone,
            event_type=args.event_type,
            experiment_label=args.experiment_label,
            app_name_filter=args.app_name_filter,
            snapshot_profile=args.snapshot_profile,
            include_duplicates=args.include_duplicates,
            history_visible_mode=args.history_visible_mode,
            billing_descriptions=billing_descriptions,
            cost_report_module=cost_report_module,
            modal_module=modal_module,
        )
        window_payloads[label] = window_payload
        data_gaps.extend(
            gap.get("reason") if isinstance(gap, dict) else str(gap)
            for gap in (window_payload.get("cost_calibration") or {}).get("allocation_gaps") or []
        )

    combined_session_key_map: dict[str, str] = {}
    combined_event_id_map: dict[str, str] = {}
    for payload in window_payloads.values():
        combined_session_key_map.update((payload.get("session_summary") or {}).get("session_key_map") or {})
        for session in ((payload.get("session_summary") or {}).get("sessions") or []):
            event_id = str(session.get("event_id") or "").strip()
            session_id = str(session.get("session_id") or "").strip()
            if event_id and session_id:
                combined_event_id_map[event_id] = session_id

    cloudflare_summary, cloudflare_gaps = _build_cloudflare_summary(
        repo_root=REPO_ROOT,
        explicit_paths=args.cloudflare_log,
        windows=windows,
        timezone_name=args.timezone,
        session_key_map=combined_session_key_map,
        event_id_map=combined_event_id_map,
    )
    data_gaps.extend(cloudflare_gaps)
    _apply_cloudflare_observability(
        window_payloads,
        cloudflare_summary,
        timezone_name=args.timezone,
    )
    data_gaps.extend(
        _augment_recent_sessions_with_feishu_read_api(
            window_payloads,
            recent_hours=args.recent_hours,
            recent_min_sessions=args.recent_min_sessions,
            timezone_name=args.timezone,
        )
    )

    hourly_cost_pk = _build_hourly_cost_pk(
        windows,
        {label: payload.get("hourly_official_costs") or {} for label, payload in window_payloads.items()},
        {label: payload.get("hourly_session_counts") or {} for label, payload in window_payloads.items()},
    )
    function_cost_pk = _build_function_cost_pk(window_payloads)
    container_cost_pk = _build_container_cost_pk(window_payloads)
    session_latency_pk = _build_session_latency_pk(window_payloads)
    internal_exec_stage_pk = _build_internal_exec_stage_pk(window_payloads)
    fallback_reason_pk = _build_fallback_reason_pk(window_payloads)
    dfmea_breakdown = _build_dfmea_breakdown(window_payloads)
    request_class_breakdown = _build_request_class_breakdown(window_payloads)
    misroute_breakdown = _build_misroute_breakdown(window_payloads)
    tool_image_path_breakdown = _build_tool_image_path_breakdown(window_payloads)
    gateway_eligible_accuracy = _build_gateway_eligible_accuracy(window_payloads)
    fallback_reason_by_request_class = _build_fallback_reason_by_request_class(window_payloads)
    text_route_cache_hit_rate = _build_text_route_cache_hit_rate(cloudflare_summary)
    cache_hit_rate = _build_cache_hit_rate(cloudflare_summary, eligible_only=False)
    cache_eligible_hit_rate = _build_cache_hit_rate(cloudflare_summary, eligible_only=True)
    browser_single_ai_call_completion = _build_browser_single_ai_call_completion(window_payloads)
    capability_match_rate = _build_capability_match_rate(window_payloads)
    preferred_model_selection_accuracy = _build_preferred_model_selection_accuracy(window_payloads)
    optimization_analysis = _build_optimization_analysis(
        window_payloads=window_payloads,
        hourly_cost_pk=hourly_cost_pk,
    )
    goal_assessment = _build_goal_assessment(window_payloads, cloudflare_summary=cloudflare_summary)
    modal_cost_optimization = _build_modal_cost_optimization(window_payloads)
    recent_window_eval, goal_check_recent = _build_recent_window_eval(
        window_payloads,
        recent_hours=args.recent_hours,
        recent_min_sessions=args.recent_min_sessions,
        timezone_name=args.timezone,
        strict_goal_metric=args.strict_goal_metric,
    )
    fixed_window_payload_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for hours in FIXED_WINDOW_HOURS:
        fixed_windows = _build_analysis_windows(
            hours=hours,
            compare_days=args.compare_days,
            timezone_name=args.timezone,
            now=now,
        )
        fixed_payloads: dict[str, dict[str, Any]] = {}
        for label, window in fixed_windows.items():
            fixed_rows = _filter_trace_rows(all_rows, window)
            fixed_payloads[label] = _build_window_report(
                window=window,
                rows=fixed_rows,
                all_rows=all_rows,
                timezone_name=args.timezone,
                event_type=args.event_type,
                experiment_label=args.experiment_label,
                app_name_filter=args.app_name_filter,
                snapshot_profile=args.snapshot_profile,
                include_duplicates=args.include_duplicates,
                history_visible_mode=args.history_visible_mode,
                billing_descriptions=billing_descriptions,
                cost_report_module=cost_report_module,
                modal_module=modal_module,
            )
        fixed_window_payload_matrix[f"{hours}h"] = fixed_payloads
    fixed_window_rollup = _build_fixed_window_rollup(fixed_window_payload_matrix)

    assumptions = [
        f"Timezone is fixed to `{args.timezone}`.",
        f"Historical T3 uses `{args.history_visible_mode}` and represents send-success visibility rather than confirmed client render completion.",
        "D-1 and D-2 comparisons use the same local-time window shifted by full calendar days.",
        "Stage costs are first estimated from reserved CPU/memory shapes and then calibrated to official Modal hourly billing within each hour bucket.",
        "Cloudflare analysis downgrades to interval-only when event-level join keys are absent or insufficient.",
        f"Goal validation for fresh data uses `{args.strict_goal_metric}` over the recent-window selection (`--recent-hours {args.recent_hours}`, fallback to latest {args.recent_min_sessions} sessions when needed).",
    ]
    report = {
        "generated_at": _to_iso(now),
        "history_visible_mode": args.history_visible_mode,
        "windows": {
            label: {
                "label": payload["window"]["label"],
                "compare_day_shift": payload["window"]["compare_day_shift"],
                "start": payload["window"]["start"],
                "end": payload["window"]["end"],
                "hours": payload["window"]["hours"],
            }
            for label, payload in window_payloads.items()
        },
        "window_details": window_payloads,
        "hourly_cost_pk": hourly_cost_pk,
        "function_cost_pk": function_cost_pk,
        "container_cost_pk": container_cost_pk,
        "session_latency_pk": session_latency_pk,
        "internal_exec_stage_pk": internal_exec_stage_pk,
        "fallback_reason_pk": fallback_reason_pk,
        "dfmea_breakdown": dfmea_breakdown,
        "request_class_breakdown": request_class_breakdown,
        "misroute_breakdown": misroute_breakdown,
        "tool_image_path_breakdown": tool_image_path_breakdown,
        "gateway_eligible_accuracy": gateway_eligible_accuracy,
        "fallback_reason_by_request_class": fallback_reason_by_request_class,
        "text_route_cache_hit_rate": text_route_cache_hit_rate,
        "cache_hit_rate": cache_hit_rate,
        "cache_eligible_hit_rate": cache_eligible_hit_rate,
        "browser_single_ai_call_completion": browser_single_ai_call_completion,
        "capability_match_rate": capability_match_rate,
        "preferred_model_selection_accuracy": preferred_model_selection_accuracy,
        "cloudflare_summary": cloudflare_summary,
        "optimization_analysis": optimization_analysis,
        "modal_cost_optimization": modal_cost_optimization,
        "recent_window_eval": recent_window_eval,
        "goal_assessment": goal_assessment,
        "goal_check_recent": goal_check_recent,
        "fixed_window_rollup": fixed_window_rollup,
        "assumptions": assumptions,
        "data_gaps": sorted({gap for gap in data_gaps if str(gap).strip()}),
    }

    markdown = _render_markdown_summary(report=report)
    json_out = Path(args.json_out).resolve()
    markdown_out = Path(args.markdown_out).resolve()
    _write_text(json_out, json.dumps(_serialize_for_json(report), ensure_ascii=False, indent=2))
    _write_text(markdown_out, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
