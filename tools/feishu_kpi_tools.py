"""Feishu KPI reporting tools backed by the triparty PK report script."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result

DEFAULT_CF_ACCOUNT_ID = "d1215a30b84b673ef0367010b0e78c10"
DEFAULT_ANALYTICS_DATASET = "hermes_feishu_gateway_events"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_REPORT_TIMEOUT_SECONDS = 600
DEFAULT_LOCAL_PROXY_URL = "http://127.0.0.1:12334"


def _schema(name: str, description: str, properties: dict[str, Any], required: Iterable[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(required or []),
        },
    }


FEISHU_KPI_REPORT_SCHEMA = _schema(
    "feishu_kpi_report",
    "Generate a Feishu-CF-Modal-Hermes KPI report and return key goals, paths, and a concise summary.",
    {
        "hours": {"type": "integer", "description": "Current analysis window in hours.", "default": 24},
        "compare_days": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Historical day offsets to compare against, e.g. [1, 2, 7].",
            "default": [1, 2],
        },
        "timezone": {"type": "string", "description": "IANA timezone name.", "default": DEFAULT_TIMEZONE},
        "recent_hours": {"type": "integer", "description": "Recent validation window in hours.", "default": 3},
        "recent_min_sessions": {
            "type": "integer",
            "description": "Minimum sessions for recent validation before falling back to the latest sessions.",
            "default": 20,
        },
        "strict_goal_metric": {
            "type": "string",
            "description": "Goal metric for fresh validation, usually p90 or p50.",
            "default": "p90",
        },
        "include_analytics_snapshot": {
            "type": "boolean",
            "description": "Also query the Workers Analytics Engine SQL API for a quick event-count snapshot.",
            "default": True,
        },
        "session_cost_source_policy": {
            "type": "string",
            "description": "How to resolve session-cost truth when Modal per-session trace allocation is incomplete.",
            "default": "strict",
        },
    },
)


def check_feishu_kpi_available() -> bool:
    return (_repo_root() / "scripts" / "feishu_triparty_pk_report.py").exists()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reports_dir() -> Path:
    path = get_hermes_home() / "reports" / "feishu-kpi"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _coerce_compare_days(value: Any) -> list[int]:
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(",", " ").split()]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = []
    result: list[int] = []
    for item in raw_items:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return result or [1, 2]


def _report_paths() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = _reports_dir() / f"feishu-kpi-{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".md")


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (TypeError, ValueError, InvalidOperation):
        return None


def _decimal_ratio_str(numerator: Any, denominator: Any) -> str | None:
    numerator_decimal = _decimal(numerator)
    denominator_decimal = _decimal(denominator)
    if numerator_decimal is None or denominator_decimal is None or denominator_decimal == 0:
        return None
    return format((numerator_decimal / denominator_decimal).quantize(Decimal("0.00000001")), "f")


def _goal_status(actual: str | None, threshold: str) -> str:
    actual_decimal = _decimal(actual)
    threshold_decimal = _decimal(threshold)
    if actual_decimal is None or threshold_decimal is None:
        return "not_met"
    return "met" if actual_decimal >= threshold_decimal else "not_met"


def _proxy_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if env.get("HTTP_PROXY") or env.get("HTTPS_PROXY") or env.get("ALL_PROXY"):
        return env
    proxy_url = str(env.get("HERMES_LOCAL_PROXY_URL") or DEFAULT_LOCAL_PROXY_URL).strip()
    host = "127.0.0.1"
    port = 12334
    try:
        parsed = proxy_url.removeprefix("http://").removeprefix("https://")
        host_part, port_part = parsed.split(":", 1)
        host = host_part.strip(" /") or host
        port = int(port_part.split("/", 1)[0])
    except Exception:
        return env
    try:
        with socket.create_connection((host, port), timeout=0.25):
            env.setdefault("HTTP_PROXY", proxy_url)
            env.setdefault("HTTPS_PROXY", proxy_url)
            env.setdefault("ALL_PROXY", proxy_url)
    except OSError:
        pass
    return env


def _tail(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_pk_report(
    *,
    hours: int,
    compare_days: list[int],
    timezone: str,
    recent_hours: int,
    recent_min_sessions: int,
    strict_goal_metric: str,
    session_cost_source_policy: str,
) -> dict[str, Any]:
    script_path = _repo_root() / "scripts" / "feishu_triparty_pk_report.py"
    json_out, markdown_out = _report_paths()
    command = [
        sys.executable,
        str(script_path),
        "--hours",
        str(hours),
        "--timezone",
        timezone,
        "--recent-hours",
        str(recent_hours),
        "--recent-min-sessions",
        str(recent_min_sessions),
        "--strict-goal-metric",
        strict_goal_metric,
        "--session-cost-source-policy",
        str(session_cost_source_policy or "strict"),
        "--json-out",
        str(json_out),
        "--markdown-out",
        str(markdown_out),
        "--compare-days",
        *[str(item) for item in compare_days],
    ]
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=DEFAULT_REPORT_TIMEOUT_SECONDS,
        encoding="utf-8",
        errors="replace",
        env=_proxy_env(os.environ.copy()),
    )
    if completed.returncode != 0:
        return {
            "success": False,
            "error": "feishu_kpi_report_subprocess_failed",
            "returncode": completed.returncode,
            "stdout_tail": _tail(completed.stdout or ""),
            "stderr_tail": _tail(completed.stderr or ""),
            "command": " ".join(shlex.quote(part) for part in command),
        }
    try:
        report = json.loads(json_out.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "success": False,
            "error": f"feishu_kpi_report_parse_failed:{exc}",
            "stdout_tail": _tail(completed.stdout or ""),
            "stderr_tail": _tail(completed.stderr or ""),
            "json_out": str(json_out),
            "markdown_out": str(markdown_out),
        }
    return {
        "success": True,
        "report": report,
        "stdout_tail": _tail(completed.stdout or ""),
        "stderr_tail": _tail(completed.stderr or ""),
        "json_out": str(json_out),
        "markdown_out": str(markdown_out),
        "command": " ".join(shlex.quote(part) for part in command),
    }


def _run_json_script(
    *,
    script_relative_path: str,
    args: list[str] | None = None,
    timeout_seconds: int = DEFAULT_REPORT_TIMEOUT_SECONDS,
    ok_returncodes: set[int] | None = None,
) -> dict[str, Any]:
    script_path = _repo_root() / script_relative_path
    command = [sys.executable, str(script_path), *(args or [])]
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        encoding="utf-8",
        errors="replace",
        env=_proxy_env(os.environ.copy()),
    )
    payload = None
    stdout_text = completed.stdout or ""
    try:
        payload = json.loads(stdout_text)
    except Exception:
        payload = None
    allowed = ok_returncodes or {0}
    return {
        "success": completed.returncode in allowed and payload is not None,
        "returncode": completed.returncode,
        "payload": payload,
        "stdout_tail": _tail(stdout_text),
        "stderr_tail": _tail(completed.stderr or ""),
        "command": " ".join(shlex.quote(part) for part in command),
    }


def _run_chain_status(*, artifacts_dir: str | None = None) -> dict[str, Any]:
    args: list[str] = []
    if artifacts_dir:
        args.extend(["--artifacts-dir", str(artifacts_dir)])
    result = _run_json_script(
        script_relative_path="scripts/feishu_chain_status.py",
        args=args,
        timeout_seconds=120,
        ok_returncodes={0},
    )
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    return {
        **result,
        "status": payload.get("status"),
        "blocker": payload.get("blocker"),
        "session": payload.get("session"),
    }


def _run_perf_cost_report(*, since_hours: int | float = 24) -> dict[str, Any]:
    result = _run_json_script(
        script_relative_path="scripts/feishu_perf_cost_report.py",
        args=["--since-hours", str(since_hours)],
        timeout_seconds=DEFAULT_REPORT_TIMEOUT_SECONDS,
        ok_returncodes={0, 2},
    )
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    return {
        **result,
        "status": payload.get("status"),
        "blocker": payload.get("blocker"),
        "app": payload.get("app"),
    }


def _analytics_sql_request(query: str) -> dict[str, Any]:
    token = str(os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
    account_id = str(os.getenv("CLOUDFLARE_ACCOUNT_ID") or DEFAULT_CF_ACCOUNT_ID).strip()
    if not token:
        return {"available": False, "reason": "missing_cloudflare_api_token"}
    dataset = str(os.getenv("HERMES_FEISHU_ANALYTICS_DATASET") or DEFAULT_ANALYTICS_DATASET).strip()
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/analytics_engine/sql"
    request = urllib.request.Request(
        url,
        data=query.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "text/plain",
            "Accept": "application/json",
        },
        method="POST",
    )
    proxy_env = _proxy_env()
    proxy_map: dict[str, str] = {}
    http_proxy = str(proxy_env.get("HTTP_PROXY") or proxy_env.get("ALL_PROXY") or "").strip()
    https_proxy = str(proxy_env.get("HTTPS_PROXY") or proxy_env.get("ALL_PROXY") or "").strip()
    if http_proxy:
        proxy_map["http"] = http_proxy
    if https_proxy:
        proxy_map["https"] = https_proxy
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler(proxy_map))
        if proxy_map
        else urllib.request.build_opener()
    )
    try:
        with opener.open(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"available": False, "reason": f"http_{exc.code}", "body": _tail(body, 500)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"request_failed:{exc}"}
    return {"available": True, "dataset": dataset, "payload": payload}


def _extract_first_scalar(sql_result: dict[str, Any], field: str) -> int | float | str | None:
    payload = (sql_result or {}).get("payload") or {}
    rows = payload.get("data") or []
    if not rows or not isinstance(rows, list) or not isinstance(rows[0], dict):
        return None
    return rows[0].get(field)


def _query_analytics_snapshot(hours: int) -> dict[str, Any]:
    dataset = str(os.getenv("HERMES_FEISHU_ANALYTICS_DATASET") or DEFAULT_ANALYTICS_DATASET).strip()
    event_counts_query = f"""
SELECT
  index1 AS event_name,
  SUM(_sample_interval) AS sample_count
FROM {dataset}
WHERE timestamp > NOW() - INTERVAL '{int(hours)}' HOUR
GROUP BY event_name
ORDER BY sample_count DESC
LIMIT 12
FORMAT JSON
""".strip()
    base_where = (
        f"FROM {dataset} "
        f"WHERE index1 = 'feishu.cf_ai_exec.done' "
        f"AND timestamp > NOW() - INTERVAL '{int(hours)}' HOUR"
    )
    routing_queries = {
        "exec_sample_count": f"SELECT SUM(_sample_interval) AS value {base_where} FORMAT JSON",
        "cache_eligible_weight": f"SELECT SUM(_sample_interval) AS value {base_where} AND double5 = 1 FORMAT JSON",
        "cache_hit_weight": f"SELECT SUM(_sample_interval) AS value {base_where} AND double5 = 1 AND blob10 = 'HIT' FORMAT JSON",
        "capability_match_weight": f"SELECT SUM(_sample_interval) AS value {base_where} AND double6 = 1 FORMAT JSON",
        "preferred_model_weight": f"SELECT SUM(_sample_interval) AS value {base_where} AND double7 = 1 FORMAT JSON",
        "single_ai_call_weight": f"SELECT SUM(_sample_interval) AS value {base_where} AND double3 = 1 FORMAT JSON",
    }
    routing_queries_result = {
        key: _analytics_sql_request(query) for key, query in routing_queries.items()
    }
    routing_summary = {
        "available": all(item.get("available") for item in routing_queries_result.values()),
        "queries": routing_queries_result,
        "totals": {
            key: _extract_first_scalar(value, "value")
            for key, value in routing_queries_result.items()
        },
    }
    return {
        "event_counts": _analytics_sql_request(event_counts_query),
        "routing_summary": routing_summary,
    }


def _status_line(label: str, value: Any, status: str | None) -> str:
    normalized = str(status or "unknown").strip().lower()
    prefix = {"met": "[met]", "not_met": "[not_met]"}.get(normalized, "[unknown]")
    return f"{prefix} {label}: {value}"


def _extract_summary(report: dict[str, Any], hours: int) -> dict[str, Any]:
    goals = report.get("goal_assessment") or {}
    current = goals.get("current") or {}
    statuses = dict(goals.get("statuses") or {})
    recent = report.get("goal_check_recent") or {}
    recent_current = recent.get("current") or {}
    summary = {
        "hours": int(hours),
        "session_cost_source_policy": report.get("session_cost_source_policy"),
        "statuses": statuses,
        "data_gaps": report.get("data_gaps") or [],
        "read_receipt_p90_ms": current.get("read_receipt_p90_ms"),
        "reply_minus_ai_p90_ms": current.get("reply_minus_ai_p90_ms"),
        "session_cost_p90_usd": current.get("session_cost_p90_usd"),
        "session_cost_measurement_mode": current.get("session_cost_measurement_mode"),
        "session_cost_truth_status": current.get("session_cost_truth_status"),
        "session_total_cost_p90_usd": current.get("session_total_cost_p90_usd"),
        "idle_hourly_p90_cost_usd": current.get("idle_hourly_p90_cost_usd"),
        "cache_eligible_hit_rate": current.get("cache_eligible_hit_rate"),
        "capability_match_rate": current.get("capability_match_rate"),
        "preferred_model_selection_accuracy": current.get("preferred_model_selection_accuracy"),
        "browser_single_ai_call_completion_rate": current.get("browser_single_ai_call_completion_rate"),
        "recent_read_receipt_ms": recent_current.get("read_receipt_ms"),
        "recent_reply_minus_ai_ms": recent_current.get("reply_minus_ai_ms"),
    }
    data_gaps = {str(item) for item in (summary.get("data_gaps") or [])}
    if "modal_billing_report_failed:current" in data_gaps:
        summary["idle_hourly_p90_cost_usd"] = None
        summary["statuses"] = {**statuses, "modal_idle_hourly_cost_under_0_005": "not_met"}
    return summary


def _apply_analytics_snapshot_fallbacks(summary: dict[str, Any], analytics_snapshot: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = analytics_snapshot or {}
    routing = (snapshot.get("routing_summary") or {}).get("totals") or {}
    exec_count = routing.get("exec_sample_count")
    cache_eligible = routing.get("cache_eligible_weight")
    cache_hit = routing.get("cache_hit_weight")
    capability = routing.get("capability_match_weight")
    preferred = routing.get("preferred_model_weight")
    single_ai = routing.get("single_ai_call_weight")

    if summary.get("cache_eligible_hit_rate") is None:
        summary["cache_eligible_hit_rate"] = _decimal_ratio_str(cache_hit, cache_eligible)
    if summary.get("capability_match_rate") is None:
        summary["capability_match_rate"] = _decimal_ratio_str(capability, exec_count)
    if summary.get("preferred_model_selection_accuracy") is None:
        summary["preferred_model_selection_accuracy"] = _decimal_ratio_str(preferred, exec_count)
    if summary.get("browser_single_ai_call_completion_rate") is None:
        summary["browser_single_ai_call_completion_rate"] = _decimal_ratio_str(single_ai, exec_count)

    statuses = dict(summary.get("statuses") or {})
    statuses["cache_hit_rate_over_0_30"] = _goal_status(summary.get("cache_eligible_hit_rate"), "0.30")
    statuses["capability_match_rate_equals_1_00"] = _goal_status(summary.get("capability_match_rate"), "1.00")
    statuses["preferred_model_selection_accuracy_over_0_95"] = _goal_status(
        summary.get("preferred_model_selection_accuracy"),
        "0.95",
    )
    statuses["browser_single_ai_call_completion_rate_over_0_50"] = _goal_status(
        summary.get("browser_single_ai_call_completion_rate"),
        "0.50",
    )
    summary["statuses"] = statuses
    return summary


def format_feishu_kpi_gateway_summary(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return (
            "Feishu KPI report failed.\n"
            f"Error: {result.get('error')}\n"
            f"stderr: {result.get('stderr_tail') or 'n/a'}"
        )
    summary = result.get("summary") or {}
    statuses = summary.get("statuses") or {}
    lines = [
        f"Feishu-CF-Modal-Hermes KPI ({summary.get('hours', '?')}h)",
        _status_line("Read receipt <5s", summary.get("read_receipt_p90_ms"), statuses.get("read_receipt_under_5s")),
        _status_line("Reply minus AI <20s", summary.get("reply_minus_ai_p90_ms"), statuses.get("reply_minus_ai_under_20s")),
        _status_line("Session cost <0.0045 USD", summary.get("session_cost_p90_usd"), statuses.get("session_cost_under_0_0045")),
        f"Session cost mode: {summary.get('session_cost_measurement_mode')} ({summary.get('session_cost_truth_status')})",
        _status_line(
            "Idle hourly cost <0.005 USD",
            summary.get("idle_hourly_p90_cost_usd"),
            statuses.get("modal_idle_hourly_cost_under_0_005"),
        ),
        _status_line(
            "AI Gateway cache hit >30%",
            summary.get("cache_eligible_hit_rate"),
            statuses.get("cache_hit_rate_over_0_30"),
        ),
        _status_line(
            "Capability match =100%",
            summary.get("capability_match_rate"),
            statuses.get("capability_match_rate_equals_1_00"),
        ),
        _status_line(
            "Preferred model >95%",
            summary.get("preferred_model_selection_accuracy"),
            statuses.get("preferred_model_selection_accuracy_over_0_95"),
        ),
        _status_line(
            "Browser single AI call >50%",
            summary.get("browser_single_ai_call_completion_rate"),
            statuses.get("browser_single_ai_call_completion_rate_over_0_50"),
        ),
        f"Recent window read receipt: {summary.get('recent_read_receipt_ms')}",
        f"Recent window reply minus AI: {summary.get('recent_reply_minus_ai_ms')}",
        f"JSON: {result.get('json_out')}",
        f"Markdown: {result.get('markdown_out')}",
    ]
    analytics = result.get("analytics_snapshot") or {}
    event_counts = ((analytics.get("event_counts") or {}).get("payload") or {}).get("data") or []
    if event_counts:
        rendered = ", ".join(
            f"{item.get('event_name')}={item.get('sample_count')}" for item in event_counts[:5] if isinstance(item, dict)
        )
        if rendered:
            lines.append(f"AE events: {rendered}")
    data_gaps = summary.get("data_gaps") or []
    if data_gaps:
        lines.append(f"Data gaps: {', '.join(str(item) for item in data_gaps[:4])}")
    return "\n".join(lines)


def generate_feishu_kpi_report(
    *,
    hours: int = 24,
    compare_days: list[int] | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    recent_hours: int = 3,
    recent_min_sessions: int = 20,
    strict_goal_metric: str = "p90",
    include_analytics_snapshot: bool = True,
    session_cost_source_policy: str = "strict",
) -> dict[str, Any]:
    hours = _coerce_int(hours, 24, 1, 168)
    compare_days = _coerce_compare_days(compare_days)
    recent_hours = _coerce_int(recent_hours, 3, 1, 72)
    recent_min_sessions = _coerce_int(recent_min_sessions, 20, 1, 200)
    timezone = str(timezone or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    strict_goal_metric = str(strict_goal_metric or "p90").strip() or "p90"

    result = _run_pk_report(
        hours=hours,
        compare_days=compare_days,
        timezone=timezone,
        recent_hours=recent_hours,
        recent_min_sessions=recent_min_sessions,
        strict_goal_metric=strict_goal_metric,
        session_cost_source_policy=str(session_cost_source_policy or "strict"),
    )
    if not result.get("success"):
        return result

    report = result["report"]
    summary = _extract_summary(report, hours)
    analytics_snapshot = _query_analytics_snapshot(hours) if include_analytics_snapshot else None
    summary = _apply_analytics_snapshot_fallbacks(summary, analytics_snapshot)
    return {
        **result,
        "summary": summary,
        "analytics_snapshot": analytics_snapshot,
    }


def _build_execution_blockers(
    summary: dict[str, Any],
    chain_status: dict[str, Any] | None,
    perf_cost_report: dict[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    statuses = summary.get("statuses") or {}
    data_gaps = {str(item) for item in (summary.get("data_gaps") or [])}
    chain_payload = chain_status or {}
    perf_payload = perf_cost_report or {}

    chain_blocker = str(chain_payload.get("blocker") or "").strip()
    if chain_blocker and chain_blocker != "none":
        blockers.append(f"chain:{chain_blocker}")
    if statuses.get("read_receipt_under_5s") != "met":
        blockers.append("kpi:read_receipt")
    if statuses.get("session_cost_under_0_0045") != "met":
        blockers.append("kpi:session_cost")
    if statuses.get("modal_idle_hourly_cost_under_0_005") != "met":
        blockers.append("kpi:idle_cost")
    if statuses.get("cache_hit_rate_over_0_30") != "met":
        blockers.append("kpi:cache_hit_rate")
    if statuses.get("browser_single_ai_call_completion_rate_over_0_50") != "met":
        blockers.append("kpi:browser_single_ai")
    if statuses.get("capability_match_rate_equals_1_00") != "met":
        blockers.append("kpi:capability_match")
    if statuses.get("preferred_model_selection_accuracy_over_0_95") != "met":
        blockers.append("kpi:preferred_model")
    if "official_cost_without_matching_function_trace" in data_gaps:
        blockers.append("gap:official_cost_without_matching_function_trace")

    perf_status = str(perf_payload.get("status") or "").strip().lower()
    perf_blocker = str(perf_payload.get("blocker") or "").strip()
    if perf_status == "blocked" and perf_blocker:
        blockers.append(f"cost_report:{perf_blocker}")

    deduped: list[str] = []
    for blocker in blockers:
        if blocker not in deduped:
            deduped.append(blocker)
    return deduped


def _build_execution_actions(
    summary: dict[str, Any],
    chain_status: dict[str, Any] | None,
    perf_cost_report: dict[str, Any] | None,
) -> list[str]:
    statuses = summary.get("statuses") or {}
    data_gaps = {str(item) for item in (summary.get("data_gaps") or [])}
    actions: list[str] = []

    if str((chain_status or {}).get("blocker") or "").strip() not in {"", "none"}:
        actions.append("先恢复 Feishu 主链路到可稳定生成新 workflow 并成功回消息，再继续做 KPI 复测。")
    if statuses.get("read_receipt_under_5s") != "met":
        actions.append("验证 app3 的 `message_read` 回调是否进入统计链路，并补一批真实读回执样本，清掉 read_receipt 数据盲区。")
    if "official_cost_without_matching_function_trace" in data_gaps or str((perf_cost_report or {}).get("status") or "") == "blocked":
        actions.append("补齐 Modal cross-function trace 持久化，或明确把 Workers Analytics Engine 设为长期成本真相源。")
    if statuses.get("cache_hit_rate_over_0_30") != "met":
        actions.append("基于最新 Worker cache policy 连续发送重复无状态 DM 样本，验证 cache eligible hit rate 是否能超过 30%。")
    if statuses.get("modal_idle_hourly_cost_under_0_005") != "met":
        actions.append("等待最新 Modal downsizing 满 24h/72h 窗口后重跑成本报表，确认空闲小时成本是否真正回落。")
    if statuses.get("browser_single_ai_call_completion_rate_over_0_50") != "met":
        actions.append("补真实浏览器重任务样本，确认分类预处理和单次 AI 调用达成率。")
    if statuses.get("capability_match_rate_equals_1_00") != "met" or statuses.get("preferred_model_selection_accuracy_over_0_95") != "met":
        actions.append("补充 Cloudflare 路由样本并复测能力匹配率与优选模型准确率，避免继续依赖空样本。")

    deduped: list[str] = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)
    return deduped[:5]


def generate_feishu_kpi_execution_snapshot(
    *,
    hours: int = 24,
    compare_days: list[int] | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    recent_hours: int = 3,
    recent_min_sessions: int = 20,
    strict_goal_metric: str = "p90",
    include_analytics_snapshot: bool = True,
    artifacts_dir: str | None = None,
    session_cost_source_policy: str = "strict",
) -> dict[str, Any]:
    kpi_report = generate_feishu_kpi_report(
        hours=hours,
        compare_days=compare_days,
        timezone=timezone,
        recent_hours=recent_hours,
        recent_min_sessions=recent_min_sessions,
        strict_goal_metric=strict_goal_metric,
        include_analytics_snapshot=include_analytics_snapshot,
        session_cost_source_policy=session_cost_source_policy,
    )
    if not kpi_report.get("success"):
        return {
            **kpi_report,
            "status": "failed",
            "priority_blockers": [str(kpi_report.get("error") or "feishu_kpi_report_failed")],
            "next_actions": [],
        }

    summary = kpi_report.get("summary") or {}
    chain_status_result = _run_chain_status(artifacts_dir=artifacts_dir)
    perf_cost_result = _run_perf_cost_report(since_hours=hours)
    chain_status = (
        chain_status_result.get("payload") if isinstance(chain_status_result.get("payload"), dict) else chain_status_result
    )
    perf_cost_report = (
        perf_cost_result.get("payload") if isinstance(perf_cost_result.get("payload"), dict) else perf_cost_result
    )
    priority_blockers = _build_execution_blockers(summary, chain_status, perf_cost_report)
    next_actions = _build_execution_actions(summary, chain_status, perf_cost_report)

    return {
        "success": True,
        "status": "ok" if not priority_blockers else "attention",
        "hours": summary.get("hours", hours),
        "summary": summary,
        "kpi_report": kpi_report,
        "chain_status": chain_status,
        "perf_cost_report": perf_cost_report,
        "priority_blockers": priority_blockers,
        "next_actions": next_actions,
        "json_out": kpi_report.get("json_out"),
        "markdown_out": kpi_report.get("markdown_out"),
    }


def format_feishu_kpi_execution_summary(snapshot: dict[str, Any]) -> str:
    if not snapshot.get("success"):
        return (
            "Feishu KPI execution snapshot failed.\n"
            f"Error: {snapshot.get('error') or 'unknown'}"
        )
    summary = snapshot.get("summary") or {}
    statuses = summary.get("statuses") or {}
    chain_status = snapshot.get("chain_status") or {}
    perf_cost_report = snapshot.get("perf_cost_report") or {}
    lines = [
        f"Feishu KPI execution snapshot ({snapshot.get('hours', '?')}h)",
        f"Snapshot status: {snapshot.get('status')}",
        _status_line("Read receipt <5s", summary.get("read_receipt_p90_ms"), statuses.get("read_receipt_under_5s")),
        _status_line("Reply minus AI <20s", summary.get("reply_minus_ai_p90_ms"), statuses.get("reply_minus_ai_under_20s")),
        _status_line("Session cost <0.0045 USD", summary.get("session_cost_p90_usd"), statuses.get("session_cost_under_0_0045")),
        _status_line(
            "Idle hourly cost <0.005 USD",
            summary.get("idle_hourly_p90_cost_usd"),
            statuses.get("modal_idle_hourly_cost_under_0_005"),
        ),
        _status_line(
            "AI Gateway cache hit >30%",
            summary.get("cache_eligible_hit_rate"),
            statuses.get("cache_hit_rate_over_0_30"),
        ),
        f"Chain status: {chain_status.get('status')} blocker={chain_status.get('blocker')}",
        f"Cost report: {perf_cost_report.get('status')} blocker={perf_cost_report.get('blocker')}",
        f"PK report JSON: {snapshot.get('json_out')}",
        f"PK report Markdown: {snapshot.get('markdown_out')}",
    ]
    blockers = snapshot.get("priority_blockers") or []
    if blockers:
        lines.append("Priority blockers:")
        lines.extend(f"- {item}" for item in blockers[:6])
    next_actions = snapshot.get("next_actions") or []
    if next_actions:
        lines.append("Next actions:")
        lines.extend(f"- {item}" for item in next_actions[:5])
    return "\n".join(lines)


def feishu_kpi_report_tool(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        result = generate_feishu_kpi_report(
            hours=args.get("hours", 24),
            compare_days=args.get("compare_days"),
            timezone=args.get("timezone", DEFAULT_TIMEZONE),
            recent_hours=args.get("recent_hours", 3),
            recent_min_sessions=args.get("recent_min_sessions", 20),
            strict_goal_metric=args.get("strict_goal_metric", "p90"),
            include_analytics_snapshot=bool(args.get("include_analytics_snapshot", True)),
            session_cost_source_policy=args.get("session_cost_source_policy", "strict"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return tool_error(f"feishu_kpi_report_failed:{exc}")
    if not result.get("success"):
        return tool_error(result.get("error") or "feishu_kpi_report_failed", **result)
    return tool_result(
        success=True,
        summary=result.get("summary"),
        analytics_snapshot=result.get("analytics_snapshot"),
        json_out=result.get("json_out"),
        markdown_out=result.get("markdown_out"),
        gateway_summary=format_feishu_kpi_gateway_summary(result),
    )


registry.register(
    name="feishu_kpi_report",
    toolset="feishu",
    schema=FEISHU_KPI_REPORT_SCHEMA,
    handler=lambda args, **kwargs: feishu_kpi_report_tool(args, **kwargs),
    check_fn=check_feishu_kpi_available,
    description="Generate the Feishu-CF-Modal-Hermes KPI report and quick summary.",
    emoji="chart",
)
