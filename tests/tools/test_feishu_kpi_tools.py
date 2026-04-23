from __future__ import annotations

import json
from pathlib import Path

from tools.feishu_kpi_tools import (
    format_feishu_kpi_gateway_summary,
    format_feishu_kpi_execution_summary,
    generate_feishu_kpi_execution_snapshot,
    generate_feishu_kpi_report,
)


def test_generate_feishu_kpi_report_returns_summary(monkeypatch, tmp_path):
    sample_report = {
        "goal_assessment": {
            "current": {
                "read_receipt_p90_ms": 4100.0,
                "reply_minus_ai_p90_ms": 12000.0,
                "session_cost_p90_usd": "0.00430000",
                "session_cost_measurement_mode": "modal_session_allocated_p90",
                "session_cost_truth_status": "authoritative",
                "idle_hourly_p90_cost_usd": "0.00200000",
                "cache_eligible_hit_rate": "0.50000000",
                "capability_match_rate": "1.00000000",
                "preferred_model_selection_accuracy": "1.00000000",
                "browser_single_ai_call_completion_rate": "0.75000000",
            },
            "statuses": {
                "read_receipt_under_5s": "met",
                "reply_minus_ai_under_20s": "met",
                "session_cost_under_0_0045": "met",
                "modal_idle_hourly_cost_under_0_005": "met",
                "cache_hit_rate_over_0_30": "met",
                "capability_match_rate_equals_1_00": "met",
                "preferred_model_selection_accuracy_over_0_95": "met",
                "browser_single_ai_call_completion_rate_over_0_50": "met",
            },
        },
        "goal_check_recent": {
            "current": {
                "read_receipt_ms": 3900.0,
                "reply_minus_ai_ms": 11000.0,
            }
        },
        "data_gaps": [],
    }

    def _fake_run_pk_report(**_kwargs):
        return {
            "success": True,
            "report": sample_report,
            "json_out": str(tmp_path / "sample.json"),
            "markdown_out": str(tmp_path / "sample.md"),
            "stdout_tail": "",
            "stderr_tail": "",
            "command": "python scripts/feishu_triparty_pk_report.py",
        }

    monkeypatch.setattr("tools.feishu_kpi_tools._run_pk_report", _fake_run_pk_report)
    monkeypatch.setattr(
        "tools.feishu_kpi_tools._query_analytics_snapshot",
        lambda hours: {"event_counts": {"payload": {"data": [{"event_name": "feishu.cf_ai_exec.done", "sample_count": 5}]}}},
    )

    result = generate_feishu_kpi_report(hours=24, include_analytics_snapshot=True)

    assert result["success"] is True
    assert result["summary"]["hours"] == 24
    assert result["summary"]["read_receipt_p90_ms"] == 4100.0
    assert result["summary"]["capability_match_rate"] == "1.00000000"
    assert result["summary"]["session_cost_measurement_mode"] == "modal_session_allocated_p90"
    assert result["analytics_snapshot"]["event_counts"]["payload"]["data"][0]["sample_count"] == 5


def test_format_feishu_kpi_gateway_summary_includes_paths_and_statuses():
    result = {
        "success": True,
        "json_out": str(Path("C:/tmp/report.json")),
        "markdown_out": str(Path("C:/tmp/report.md")),
        "summary": {
            "hours": 24,
            "statuses": {
                "read_receipt_under_5s": "met",
                "reply_minus_ai_under_20s": "met",
                "session_cost_under_0_0045": "not_met",
                "modal_idle_hourly_cost_under_0_005": "met",
                "cache_hit_rate_over_0_30": "met",
                "capability_match_rate_equals_1_00": "met",
                "preferred_model_selection_accuracy_over_0_95": "met",
                "browser_single_ai_call_completion_rate_over_0_50": "not_met",
            },
            "read_receipt_p90_ms": 4200.0,
            "reply_minus_ai_p90_ms": 15000.0,
            "session_cost_p90_usd": "0.00480000",
            "session_cost_measurement_mode": "insufficient_function_trace",
            "session_cost_truth_status": "unavailable",
            "idle_hourly_p90_cost_usd": "0.00210000",
            "cache_eligible_hit_rate": "0.35000000",
            "capability_match_rate": "1.00000000",
            "preferred_model_selection_accuracy": "0.98000000",
            "browser_single_ai_call_completion_rate": "0.45000000",
            "recent_read_receipt_ms": 4100.0,
            "recent_reply_minus_ai_ms": 14000.0,
            "data_gaps": ["cloudflare_cost_details_missing"],
        },
        "analytics_snapshot": {
            "event_counts": {
                "payload": {
                    "data": [
                        {"event_name": "feishu.webhook.accepted", "sample_count": 12},
                        {"event_name": "feishu.cf_ai_exec.done", "sample_count": 9},
                    ]
                }
            }
        },
    }

    text = format_feishu_kpi_gateway_summary(result)

    assert "Feishu-CF-Modal-Hermes KPI (24h)" in text
    assert "[met] Read receipt <5s: 4200.0" in text
    assert "[not_met] Session cost <0.0045 USD: 0.00480000" in text
    assert "Session cost mode: insufficient_function_trace (unavailable)" in text
    assert "JSON:" in text and "report.json" in text
    assert "[met] AI Gateway cache hit >30%: 0.35000000" in text
    assert "AE events: feishu.webhook.accepted=12, feishu.cf_ai_exec.done=9" in text
    assert "Data gaps: cloudflare_cost_details_missing" in text


def test_generate_feishu_kpi_report_backfills_gateway_metrics_from_analytics_snapshot(monkeypatch, tmp_path):
    sample_report = {
        "goal_assessment": {
            "current": {
                "read_receipt_p90_ms": 500.0,
                "reply_minus_ai_p90_ms": 6000.0,
                "session_cost_p90_usd": "0.01000000",
                "idle_hourly_p90_cost_usd": "0.00400000",
                "cache_eligible_hit_rate": None,
                "capability_match_rate": None,
                "preferred_model_selection_accuracy": None,
                "browser_single_ai_call_completion_rate": None,
            },
            "statuses": {
                "read_receipt_under_5s": "met",
                "reply_minus_ai_under_20s": "met",
                "session_cost_under_0_0045": "not_met",
                "modal_idle_hourly_cost_under_0_005": "met",
                "cache_hit_rate_over_0_30": "not_met",
                "capability_match_rate_equals_1_00": "not_met",
                "preferred_model_selection_accuracy_over_0_95": "not_met",
                "browser_single_ai_call_completion_rate_over_0_50": "not_met",
            },
        },
        "goal_check_recent": {"current": {"read_receipt_ms": 480.0, "reply_minus_ai_ms": 5500.0}},
        "data_gaps": [],
    }

    def _fake_run_pk_report(**_kwargs):
        return {
            "success": True,
            "report": sample_report,
            "json_out": str(tmp_path / "sample.json"),
            "markdown_out": str(tmp_path / "sample.md"),
            "stdout_tail": "",
            "stderr_tail": "",
            "command": "python scripts/feishu_triparty_pk_report.py",
        }

    monkeypatch.setattr("tools.feishu_kpi_tools._run_pk_report", _fake_run_pk_report)
    monkeypatch.setattr(
        "tools.feishu_kpi_tools._query_analytics_snapshot",
        lambda hours: {
            "routing_summary": {
                "totals": {
                    "exec_sample_count": "2",
                    "cache_eligible_weight": "2",
                    "cache_hit_weight": "1",
                    "capability_match_weight": "2",
                    "preferred_model_weight": "2",
                    "single_ai_call_weight": "2",
                }
            }
        },
    )

    result = generate_feishu_kpi_report(hours=24, include_analytics_snapshot=True)

    assert result["summary"]["cache_eligible_hit_rate"] == "0.50000000"
    assert result["summary"]["capability_match_rate"] == "1.00000000"
    assert result["summary"]["preferred_model_selection_accuracy"] == "1.00000000"
    assert result["summary"]["browser_single_ai_call_completion_rate"] == "1.00000000"
    assert result["summary"]["statuses"]["cache_hit_rate_over_0_30"] == "met"
    assert result["summary"]["statuses"]["capability_match_rate_equals_1_00"] == "met"


def test_generate_feishu_kpi_report_marks_idle_cost_not_met_when_current_billing_window_failed(monkeypatch, tmp_path):
    sample_report = {
        "goal_assessment": {
            "current": {
                "read_receipt_p90_ms": None,
                "reply_minus_ai_p90_ms": 5000.0,
                "session_cost_p90_usd": None,
                "idle_hourly_p90_cost_usd": "0.00000000",
                "cache_eligible_hit_rate": "0.00000000",
            },
            "statuses": {
                "read_receipt_under_5s": "not_met",
                "reply_minus_ai_under_20s": "met",
                "session_cost_under_0_0045": "not_met",
                "modal_idle_hourly_cost_under_0_005": "met",
                "cache_hit_rate_over_0_30": "not_met",
                "capability_match_rate_equals_1_00": "met",
                "preferred_model_selection_accuracy_over_0_95": "met",
                "browser_single_ai_call_completion_rate_over_0_50": "met",
            },
        },
        "goal_check_recent": {"current": {"read_receipt_ms": None, "reply_minus_ai_ms": 5000.0}},
        "data_gaps": ["modal_billing_report_failed:current"],
    }

    def _fake_run_pk_report(**_kwargs):
        return {
            "success": True,
            "report": sample_report,
            "json_out": str(tmp_path / "sample.json"),
            "markdown_out": str(tmp_path / "sample.md"),
            "stdout_tail": "",
            "stderr_tail": "",
            "command": "python scripts/feishu_triparty_pk_report.py",
        }

    monkeypatch.setattr("tools.feishu_kpi_tools._run_pk_report", _fake_run_pk_report)
    monkeypatch.setattr("tools.feishu_kpi_tools._query_analytics_snapshot", lambda hours: None)

    result = generate_feishu_kpi_report(hours=24, include_analytics_snapshot=False)

    assert result["summary"]["idle_hourly_p90_cost_usd"] is None
    assert result["summary"]["statuses"]["modal_idle_hourly_cost_under_0_005"] == "not_met"


def test_analytics_sql_request_works_without_proxy(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setattr("tools.feishu_kpi_tools._proxy_env", lambda base_env=None: {})

    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": [{"value": 1}]}).encode("utf-8")

    class _Opener:
        def open(self, request, timeout=0):
            captured["timeout"] = timeout
            captured["url"] = request.full_url
            return _Response()

    def _fake_build_opener(*handlers):
        captured["handler_count"] = len(handlers)
        return _Opener()

    monkeypatch.setattr("tools.feishu_kpi_tools.urllib.request.build_opener", _fake_build_opener)

    from tools.feishu_kpi_tools import _analytics_sql_request

    result = _analytics_sql_request("SELECT 1 FORMAT JSON")

    assert result["available"] is True
    assert result["payload"]["data"][0]["value"] == 1
    assert captured["handler_count"] == 0
    assert captured["timeout"] == 30


def test_analytics_sql_request_filters_missing_proxy_values(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.setattr(
        "tools.feishu_kpi_tools._proxy_env",
        lambda base_env=None: {"HTTP_PROXY": "http://127.0.0.1:12334"},
    )

    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": []}).encode("utf-8")

    class _Opener:
        def open(self, request, timeout=0):
            return _Response()

    def _fake_build_opener(*handlers):
        captured["handler_count"] = len(handlers)
        captured["proxies"] = getattr(handlers[0], "proxies", {}) if handlers else {}
        return _Opener()

    monkeypatch.setattr("tools.feishu_kpi_tools.urllib.request.build_opener", _fake_build_opener)

    from tools.feishu_kpi_tools import _analytics_sql_request

    result = _analytics_sql_request("SELECT 1 FORMAT JSON")

    assert result["available"] is True
    assert captured["handler_count"] == 1
    assert captured["proxies"] == {"http": "http://127.0.0.1:12334"}


def test_generate_feishu_kpi_execution_snapshot_collects_blockers(monkeypatch):
    monkeypatch.setattr(
        "tools.feishu_kpi_tools.generate_feishu_kpi_report",
        lambda **_kwargs: {
            "success": True,
            "json_out": "C:/tmp/report.json",
            "markdown_out": "C:/tmp/report.md",
            "summary": {
                "hours": 24,
                "statuses": {
                    "read_receipt_under_5s": "not_met",
                    "reply_minus_ai_under_20s": "met",
                    "session_cost_under_0_0045": "not_met",
                    "modal_idle_hourly_cost_under_0_005": "not_met",
                    "cache_hit_rate_over_0_30": "not_met",
                    "capability_match_rate_equals_1_00": "met",
                    "preferred_model_selection_accuracy_over_0_95": "met",
                    "browser_single_ai_call_completion_rate_over_0_50": "not_met",
                },
                "data_gaps": ["official_cost_without_matching_function_trace"],
                "read_receipt_p90_ms": None,
                "reply_minus_ai_p90_ms": 5036.0,
                "session_cost_p90_usd": None,
                "idle_hourly_p90_cost_usd": "0.01023596",
                "cache_eligible_hit_rate": None,
            },
        },
    )
    monkeypatch.setattr(
        "tools.feishu_kpi_tools._run_chain_status",
        lambda **_kwargs: {"payload": {"status": "ok", "blocker": "none"}},
    )
    monkeypatch.setattr(
        "tools.feishu_kpi_tools._run_perf_cost_report",
        lambda **_kwargs: {
            "payload": {
                "status": "blocked",
                "blocker": "modal_debug_function_missing",
            }
        },
    )

    result = generate_feishu_kpi_execution_snapshot(hours=24)

    assert result["status"] == "attention"
    assert "kpi:read_receipt" in result["priority_blockers"]
    assert "gap:official_cost_without_matching_function_trace" in result["priority_blockers"]
    assert "cost_report:modal_debug_function_missing" in result["priority_blockers"]
    assert any("message_read" in item for item in result["next_actions"])
    assert any("Workers Analytics Engine" in item for item in result["next_actions"])


def test_format_feishu_kpi_execution_summary_renders_paths_and_actions():
    text = format_feishu_kpi_execution_summary(
        {
            "success": True,
            "hours": 24,
            "status": "attention",
            "json_out": "C:/tmp/report.json",
            "markdown_out": "C:/tmp/report.md",
            "summary": {
                "statuses": {
                    "read_receipt_under_5s": "met",
                    "reply_minus_ai_under_20s": "met",
                    "session_cost_under_0_0045": "not_met",
                    "modal_idle_hourly_cost_under_0_005": "not_met",
                    "cache_hit_rate_over_0_30": "not_met",
                },
                "read_receipt_p90_ms": 528.3,
                "reply_minus_ai_p90_ms": 7594.1,
                "session_cost_p90_usd": None,
                "idle_hourly_p90_cost_usd": "0.01024489",
                "cache_eligible_hit_rate": "0.00000000",
            },
            "chain_status": {"status": "ok", "blocker": "none"},
            "perf_cost_report": {"status": "blocked", "blocker": "modal_debug_function_missing"},
            "priority_blockers": ["kpi:session_cost", "cost_report:modal_debug_function_missing"],
            "next_actions": ["补齐 Modal cross-function trace 持久化。"],
        }
    )

    assert "Feishu KPI execution snapshot (24h)" in text
    assert "Chain status: ok blocker=none" in text
    assert "Cost report: blocked blocker=modal_debug_function_missing" in text
    assert "PK report JSON: C:/tmp/report.json" in text
    assert "- kpi:session_cost" in text
    assert "- 补齐 Modal cross-function trace 持久化。" in text
