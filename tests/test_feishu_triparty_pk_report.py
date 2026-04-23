import importlib.util
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "feishu_triparty_pk_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("feishu_triparty_pk_report_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _tz(module):
    try:
        return ZoneInfo("Asia/Shanghai")
    except Exception:
        return module._get_timezone("Asia/Shanghai")


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def test_build_analysis_windows_keeps_same_local_span():
    module = _load_module()
    tz = _tz(module)
    now = datetime(2026, 4, 15, 20, 30, tzinfo=tz)

    windows = module._build_analysis_windows(
        hours=16,
        compare_days=[1, 2],
        timezone_name="Asia/Shanghai",
        now=now,
    )

    assert windows["current"].start.isoformat() == "2026-04-15T04:30:00+08:00"
    assert windows["d-1"].start.isoformat() == "2026-04-14T04:30:00+08:00"
    assert windows["d-2"].start.isoformat() == "2026-04-13T04:30:00+08:00"
    assert windows["current"].hours == windows["d-1"].hours == windows["d-2"].hours == 16.0


def test_apply_default_proxy_env_sets_local_proxy_when_reachable(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.setenv("HERMES_LOCAL_PROXY_URL", "http://127.0.0.1:12334")

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(module.socket, "create_connection", lambda *args, **kwargs: _Conn())

    module._apply_default_proxy_env()

    assert module.os.environ["HTTP_PROXY"] == "http://127.0.0.1:12334"
    assert module.os.environ["HTTPS_PROXY"] == "http://127.0.0.1:12334"
    assert module.os.environ["ALL_PROXY"] == "http://127.0.0.1:12334"


def test_build_session_timelines_matches_message_read_and_success_send_proxy():
    module = _load_module()
    tz = _tz(module)
    start = datetime(2026, 4, 15, 4, 30, tzinfo=tz)
    end = start + timedelta(hours=16)
    window = module.AnalysisWindow(label="current", compare_day_shift=0, start=start, end=end)

    t0 = start + timedelta(hours=1)
    rows = [
        {
            "ts": _ts(t0),
            "stage": "webhook.accepted",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "message_id": "om_in_1",
        },
        {
            "ts": _ts(t0 + timedelta(milliseconds=1)),
            "stage": "webhook.ack",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "message_id": "om_in_1",
            "ack_kind": "queued_message",
            "ack_elapsed_ms": 500,
        },
        {
            "ts": _ts(t0 + timedelta(milliseconds=2)),
            "stage": "internal.agent_plan.done",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "message_id": "om_in_1",
            "external_exec_candidate": True,
            "request_class": "text_plain",
            "content_modalities": ["text"],
            "route_family": "gateway_text",
            "gateway_route_name": "affiliate-general",
            "gateway_eligible": True,
            "requires_tools": False,
            "requires_browser": False,
            "requires_media_hydration": False,
            "toolset": [],
            "modality_profile": "text",
        },
        {
            "ts": _ts(t0 + timedelta(seconds=20)),
            "stage": "gateway.send.done",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "message_id": "om_in_1",
            "session_key": "sess_1",
            "send_success": False,
        },
        {
            "ts": _ts(t0 + timedelta(seconds=25)),
            "stage": "gateway.send.done",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "message_id": "om_in_1",
            "send_message_id": "om_out_1",
            "session_key": "sess_1",
            "send_success": True,
        },
        {
            "ts": _ts(t0 + timedelta(seconds=24)),
            "stage": "internal.agent_exec.done",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "message_id": "om_in_1",
            "agent_model_elapsed_ms": 6000,
            "agent_non_model_elapsed_ms": 19000,
            "provider_usage_totals": {"billed_cost_usd": 0.0012},
        },
        {
            "ts": _ts(t0 + timedelta(seconds=31)),
            "stage": "webhook.message_read",
            "event_type": "im.message.message_read_v1",
            "event_id": "read_evt_1",
            "read_time": _ms(t0 + timedelta(seconds=4)),
            "message_id_list": ["om_in_1"],
        },
        {
            "ts": _ts(t0 + timedelta(seconds=33)),
            "stage": "webhook.message_read",
            "event_type": "im.message.message_read_v1",
            "event_id": "read_evt_1_reply",
            "read_time": _ms(t0 + timedelta(seconds=29)),
            "message_id_list": ["om_out_1"],
        },
        {
            "ts": _ts(t0 + timedelta(hours=1)),
            "stage": "webhook.accepted",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_2",
            "message_id": "om_in_2",
        },
        {
            "ts": _ts(t0 + timedelta(hours=1, milliseconds=1)),
            "stage": "webhook.ack",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_2",
            "message_id": "om_in_2",
            "ack_kind": "queued_message",
            "ack_elapsed_ms": 300,
        },
        {
            "ts": _ts(t0 + timedelta(hours=1, seconds=8)),
            "stage": "webhook.message_read",
            "event_type": "im.message.message_read_v1",
            "event_id": "read_evt_2",
            "read_time": _ms(t0 + timedelta(hours=1, seconds=7)),
            "message_id_list": ["om_bot_reply_only"],
        },
    ]

    result = module._build_session_timelines(
        rows,
        window=window,
        event_type="im.message.receive_v1",
        include_duplicates=False,
        history_visible_mode="send_success_proxy",
    )

    by_id = {item["session_id"]: item for item in result["sessions"]}
    assert by_id["evt_1"]["t2_ms"] == _ms(t0 + timedelta(seconds=4))
    assert by_id["evt_1"]["t3_ms"] == _ts(t0 + timedelta(seconds=25)) * 1000
    assert by_id["evt_1"]["reply_send_message_id"] == "om_out_1"
    assert by_id["evt_1"]["reply_to_read_ms"] == 4000
    assert by_id["evt_1"]["read_match_source"] == "inbound_message_id"
    assert by_id["evt_1"]["reply_read_match_source"] == "webhook_message_read"
    assert by_id["evt_1"]["t0_to_t2_ms"] == 4500
    assert by_id["evt_1"]["t0_to_t3_minus_model_ms"] == 19500
    assert by_id["evt_1"]["provider_billed_cost_usd"] == 0.0012
    assert by_id["evt_1"]["reply_visible_proxy"] == "gateway.send.done:send_success=true"
    assert by_id["evt_1"]["request_class"] == "text_plain"
    assert by_id["evt_1"]["route_family"] == "gateway_text"
    assert by_id["evt_1"]["gateway_route_name"] == "affiliate-general"
    assert by_id["evt_1"]["gateway_eligible"] is True
    assert by_id["evt_1"]["content_modalities"] == ["text"]
    assert by_id["evt_2"]["t2_ms"] is None
    assert by_id["evt_2"]["t3_ms"] is None


def test_build_session_timelines_prefers_provider_wait_over_legacy_model_proxy():
    module = _load_module()
    tz = _tz(module)
    start = datetime(2026, 4, 15, 4, 30, tzinfo=tz)
    end = start + timedelta(hours=16)
    window = module.AnalysisWindow(label="current", compare_day_shift=0, start=start, end=end)
    t0 = start + timedelta(hours=2)
    rows = [
        {
            "ts": _ts(t0),
            "stage": "webhook.accepted",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_wait",
            "message_id": "om_wait_1",
        },
        {
            "ts": _ts(t0 + timedelta(seconds=25)),
            "stage": "internal.agent_exec.done",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_wait",
            "message_id": "om_wait_1",
            "agent_model_elapsed_ms": 9000,
            "provider_wait_elapsed_ms": 4000,
            "provider_wait_measurement_mode": "provider_wait_observed",
        },
    ]

    result = module._build_session_timelines(
        rows,
        window=window,
        event_type="im.message.receive_v1",
        include_duplicates=False,
        history_visible_mode="send_success_proxy",
    )

    session = result["sessions"][0]
    assert session["ai_model_elapsed_ms"] == 4000
    assert session["provider_wait_elapsed_ms"] == 4000
    assert session["reply_minus_ai_measurement_mode"] == "provider_wait_observed"
    assert session["t0_to_t3_minus_model_ms"] == 21000


def test_calibrate_cost_items_by_hour_matches_official_hourly_total():
    module = _load_module()
    items = [
        {"hour_key": "2026-04-15T08:00:00+08:00", "raw_estimated_cost_usd": Decimal("1.00")},
        {"hour_key": "2026-04-15T08:00:00+08:00", "raw_estimated_cost_usd": Decimal("2.00")},
        {"hour_key": "2026-04-15T09:00:00+08:00", "raw_estimated_cost_usd": Decimal("1.50")},
    ]
    official = {
        "2026-04-15T08:00:00+08:00": Decimal("6.00"),
        "2026-04-15T09:00:00+08:00": Decimal("3.00"),
    }

    result = module._calibrate_cost_items_by_hour(items, official_hourly_costs=official)

    calibrated_total = sum(item["calibrated_cost_usd"] for item in result["items"] if item["calibrated_cost_usd"] is not None)
    assert calibrated_total == Decimal("9.00")
    assert result["calibration_ratio_by_hour"]["2026-04-15T08:00:00+08:00"] == "2.00000000"
    assert result["calibration_ratio_by_hour"]["2026-04-15T09:00:00+08:00"] == "2.00000000"


def test_calibrate_cost_items_by_hour_uses_raw_estimate_when_official_hour_is_missing():
    module = _load_module()
    items = [
        {"hour_key": "2026-04-15T10:00:00+08:00", "raw_estimated_cost_usd": Decimal("1.25")},
    ]
    official = {
        "2026-04-15T10:00:00+08:00": Decimal("0"),
    }

    result = module._calibrate_cost_items_by_hour(items, official_hourly_costs=official)

    assert result["items"][0]["calibrated_cost_usd"] == Decimal("1.25")
    assert result["calibration_ratio_by_hour"]["2026-04-15T10:00:00+08:00"] == "1.00000000"
    assert result["allocation_gaps"][0]["reason"] == "official_cost_missing_using_raw_estimate"


def test_request_class_breakdowns_and_text_cache_hit_rate():
    module = _load_module()
    window_payloads = {
        "current": {
            "session_summary": {
                "sessions": [
                    {
                        "session_id": "evt_text",
                        "request_class": "text_plain",
                        "gateway_eligible": True,
                        "external_exec_candidate": True,
                        "allocated_cost_usd": "0.0010",
                        "fallback_reason": "",
                        "route_decision_reason": "plain_text_without_attachments_or_browser",
                    },
                    {
                        "session_id": "evt_image",
                        "request_class": "image_understanding",
                        "gateway_eligible": False,
                        "external_exec_candidate": False,
                        "allocated_cost_usd": "0.0020",
                        "fallback_reason": "media_hydration_required",
                        "route_decision_reason": "media_hydration_required",
                        "route_family": "modal_runtime",
                        "gateway_route_name": "",
                        "requires_browser": False,
                        "requires_tools": False,
                        "misroute_detected": True,
                        "gateway_error_class": "misrouted_request_class",
                    },
                ]
            }
        }
    }

    request_class_breakdown = module._build_request_class_breakdown(window_payloads)
    misroute_breakdown = module._build_misroute_breakdown(window_payloads)
    tool_image_path_breakdown = module._build_tool_image_path_breakdown(window_payloads)
    gateway_eligible_accuracy = module._build_gateway_eligible_accuracy(window_payloads)
    fallback_reason_by_request_class = module._build_fallback_reason_by_request_class(window_payloads)
    dfmea_breakdown = module._build_dfmea_breakdown(
        {
            "current": {
                "session_summary": {
                    "sessions": [
                        {
                            "session_id": "evt_dfmea",
                            "dfmea_failure_mode": "gateway_misroute",
                            "dfmea_control_phase": "classification",
                            "dfmea_detection_signal": "strict_request_class_route_guard",
                            "dfmea_severity": 9,
                        }
                    ]
                }
            }
        }
    )
    text_route_cache_hit_rate = module._build_text_route_cache_hit_rate(
        {
            "window_summaries": {
                "current": {
                    "gateway_request_rows": [
                        {"request_class": "text_plain", "gateway_route_name": "affiliate-general", "cf_cache_status": True},
                        {"request_class": "text_coding", "gateway_route_name": "affiliate-coding", "cf_cache_status": False},
                        {"request_class": "image_understanding", "gateway_route_name": "image-understanding", "cf_cache_status": True},
                    ]
                }
            }
        }
    )

    assert request_class_breakdown["current"][0]["request_class"] == "text_plain"
    assert misroute_breakdown["current"][0]["reason"] == "misrouted_request_class"
    assert tool_image_path_breakdown["current"][0]["request_class"] == "image_understanding"
    assert gateway_eligible_accuracy["current"]["accuracy"] == 1.0
    assert fallback_reason_by_request_class["current"]["image_understanding"][0]["reason"] == "media_hydration_required"
    assert dfmea_breakdown["current"][0]["failure_mode"] == "gateway_misroute"
    assert dfmea_breakdown["current"][0]["max_severity"] == 9
    assert text_route_cache_hit_rate["current"]["sample_count"] == 2
    assert text_route_cache_hit_rate["current"]["hit_count"] == 1


def test_build_cloudflare_summary_marks_missing_logs(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    tz = _tz(module)
    now = datetime(2026, 4, 15, 20, 30, tzinfo=tz)
    windows = module._build_analysis_windows(
        hours=16,
        compare_days=[1, 2],
        timezone_name="Asia/Shanghai",
        now=now,
    )

    summary, gaps = module._build_cloudflare_summary(
        repo_root=tmp_path,
        explicit_paths=[],
        windows=windows,
        timezone_name="Asia/Shanghai",
        session_key_map={},
        event_id_map={},
    )

    assert summary["status"] == "missing"
    assert "cloudflare_logs_missing" in gaps
    assert summary["join_level"] == "none"


def test_build_cloudflare_summary_promotes_worker_fact_event_level(tmp_path, monkeypatch):
    module = _load_module()
    tz = _tz(module)
    now = datetime(2026, 4, 15, 20, 30, tzinfo=tz)
    windows = module._build_analysis_windows(
        hours=16,
        compare_days=[1, 2],
        timezone_name="Asia/Shanghai",
        now=now,
    )
    monkeypatch.setattr(module, "_discover_cloudflare_log_paths", lambda repo_root, explicit_paths: [])
    monkeypatch.setattr(module, "_read_cloudflare_records", lambda paths, timezone_name: ([], []))
    monkeypatch.setattr(module, "_fetch_cloudflare_ai_gateway_records", lambda timezone_name, windows: ([], []))
    worker_ts = _ms(datetime(2026, 4, 15, 19, 0, tzinfo=tz))
    monkeypatch.setattr(
        module,
        "_fetch_cloudflare_worker_observability_records",
        lambda repo_root, timezone_name, windows: (
            [
                {
                    "event": "feishu.webhook.accepted",
                    "event_id": "evt_worker_only",
                    "session_key": "agent:main:feishu:dm:chat_worker",
                    "message_id": "om_in_worker",
                    "__timestamp_ms": worker_ts,
                }
            ],
            [],
        ),
    )

    summary, gaps = module._build_cloudflare_summary(
        repo_root=tmp_path,
        explicit_paths=[],
        windows=windows,
        timezone_name="Asia/Shanghai",
        session_key_map={},
        event_id_map={},
    )

    assert summary["status"] == "ok"
    assert summary["join_level"] == "worker_event_level"
    assert summary["bootstrapped_worker_fact_count"] == 1
    assert "cloudflare_interval_only_join" not in gaps


def test_build_cloudflare_summary_uses_cf_ai_exec_done_as_gateway_request_fallback(tmp_path, monkeypatch):
    module = _load_module()
    tz = _tz(module)
    now = datetime(2026, 4, 15, 20, 30, tzinfo=tz)
    windows = module._build_analysis_windows(
        hours=16,
        compare_days=[1, 2],
        timezone_name="Asia/Shanghai",
        now=now,
    )
    monkeypatch.setattr(module, "_discover_cloudflare_log_paths", lambda repo_root, explicit_paths: [])
    monkeypatch.setattr(module, "_read_cloudflare_records", lambda paths, timezone_name: ([], []))
    monkeypatch.setattr(module, "_fetch_cloudflare_ai_gateway_records", lambda timezone_name, windows: ([], []))
    worker_ts = _ms(datetime(2026, 4, 15, 19, 0, tzinfo=tz))
    monkeypatch.setattr(
        module,
        "_fetch_cloudflare_worker_observability_records",
        lambda repo_root, timezone_name, windows: (
            [
                {
                    "event": "feishu.cf_ai_exec.done",
                    "event_id": "evt_cf_exec",
                    "correlation_id": "feishu:chat:evt_cf_exec",
                    "request_class": "text_plain",
                    "route_family": "gateway_text",
                    "gateway_route_name": "affiliate-general",
                    "gateway_eligible": True,
                    "cf_cache_status": "HIT",
                    "cache_eligible": True,
                    "ai_call_count": 1,
                    "capability_match": True,
                    "preferred_model_selected": True,
                    "provider": "openrouter",
                    "model": "openai/gpt-4.1-mini",
                    "__timestamp_ms": worker_ts,
                }
            ],
            [],
        ),
    )

    summary, gaps = module._build_cloudflare_summary(
        repo_root=tmp_path,
        explicit_paths=[],
        windows=windows,
        timezone_name="Asia/Shanghai",
        session_key_map={},
        event_id_map={},
    )

    assert "cloudflare_cost_details_missing" in gaps
    rows = summary["window_summaries"]["current"]["gateway_request_rows"]
    assert len(rows) == 1
    assert rows[0]["event"] == "feishu.cf_ai_exec.done"
    assert rows[0]["cf_cache_status"] == "HIT"
    assert rows[0]["cache_eligible"] is True
    assert rows[0]["ai_call_count"] == 1
    assert rows[0]["capability_match"] is True
    assert rows[0]["preferred_model_selected"] is True


def test_build_fixed_window_rollup_outputs_1h_24h_72h():
    module = _load_module()
    payload_matrix = {
        "1h": {
            "current": {
                "session_summary": {
                    "completion": {"session_count": 8},
                    "metrics": {
                        "read_receipt_ms": {"p90": 4100},
                        "t0_to_t3_minus_model_ms": {"p90": 15000},
                        "t0_to_t3_ms": {"p90": 22000},
                    },
                },
                "billing_summary": {"avg_cost_per_session_usd": 0.0039, "total_cost_usd": 0.0312},
            },
            "d-1": {
                "session_summary": {
                    "completion": {"session_count": 6},
                    "metrics": {
                        "read_receipt_ms": {"p90": 5200},
                        "t0_to_t3_minus_model_ms": {"p90": 19000},
                        "t0_to_t3_ms": {"p90": 26000},
                    },
                },
                "billing_summary": {"avg_cost_per_session_usd": 0.0048, "total_cost_usd": 0.0288},
            },
        },
        "24h": {
            "current": {
                "session_summary": {"completion": {"session_count": 80}, "metrics": {"read_receipt_ms": {"p90": 4300}}},
                "billing_summary": {"avg_cost_per_session_usd": 0.0041, "total_cost_usd": 0.328},
            },
            "d-1": {
                "session_summary": {"completion": {"session_count": 75}, "metrics": {"read_receipt_ms": {"p90": 4700}}},
                "billing_summary": {"avg_cost_per_session_usd": 0.0046, "total_cost_usd": 0.345},
            },
        },
        "72h": {
            "current": {
                "session_summary": {"completion": {"session_count": 210}, "metrics": {"read_receipt_ms": {"p90": 4500}}},
                "billing_summary": {"avg_cost_per_session_usd": 0.0042, "total_cost_usd": 0.882},
            },
            "d-1": {
                "session_summary": {"completion": {"session_count": 190}, "metrics": {"read_receipt_ms": {"p90": 4300}}},
                "billing_summary": {"avg_cost_per_session_usd": 0.0040, "total_cost_usd": 0.76},
            },
        },
    }

    result = module._build_fixed_window_rollup(payload_matrix)

    assert sorted(result) == ["1h", "24h", "72h"]
    assert result["1h"]["current"]["session_count"] == 8
    assert result["1h"]["baselines"]["d-1"]["session_count"] == 6
    assert result["1h"]["trend"]["overall"] == "improving"
    assert result["24h"]["trend"]["session_count"] == "improving"
    assert result["72h"]["trend"]["read_receipt_p90_ms"] == "regressing"


def test_read_cloudflare_records_parses_wrangler_tail_json_logs(tmp_path):
    module = _load_module()
    log_path = tmp_path / ".tmp-wrangler-tail-live.jsonl"
    log_path.write_text(
        '\n'.join(
            [
                '{"wallTime":1,"cpuTime":0,"scriptName":"hermes-feishu-gateway","scriptVersion":{"id":"v1"},"executionModel":"stateless","logs":[{"message":["{\\"event\\":\\"feishu.direct_planned.plan.done\\",\\"correlation_id\\":\\"feishu:chat:evt_1\\",\\"route_hint\\":\\"modal_heavy_exec\\",\\"execution_mode\\":\\"deferred_reconcile\\",\\"external_exec_candidate\\":true}"],"level":"log","timestamp":1776319666054}],"eventTimestamp":1776319654733,"event":{"request":{"url":"https://hermes.isuyee.com/","method":"POST"},"response":{"status":200}}}',
            ]
        ),
        encoding="utf-8",
    )

    records, gaps = module._read_cloudflare_records([log_path], "Asia/Shanghai")

    assert gaps == []
    assert len(records) == 1
    assert records[0]["event"] == "feishu.direct_planned.plan.done"
    assert records[0]["correlation_id"] == "feishu:chat:evt_1"
    assert records[0]["request_url"] == "https://hermes.isuyee.com/"
    assert records[0]["response_status"] == 200
    assert records[0]["script_name"] == "hermes-feishu-gateway"
    assert records[0]["__timestamp_ms"] == 1776319666054


def test_read_cloudflare_records_parses_multiline_wrangler_tail_json_logs(tmp_path):
    module = _load_module()
    log_path = tmp_path / ".tmp-wrangler-tail-live.jsonl"
    log_path.write_text(
        '{\n'
        '  "wallTime": 1,\n'
        '  "cpuTime": 0,\n'
        '  "scriptName": "hermes-feishu-gateway",\n'
        '  "scriptVersion": {"id": "v2"},\n'
        '  "executionModel": "stateless",\n'
        '  "logs": [\n'
        '    {\n'
        '      "message": [\n'
        '        "{\\"event\\":\\"feishu.direct_planned.done\\",\\"correlation_id\\":\\"feishu:chat:evt_2\\",\\"direct_exec_elapsed_ms\\":3336}"\n'
        '      ],\n'
        '      "level": "log",\n'
        '      "timestamp": 1776326689000\n'
        '    }\n'
        '  ],\n'
        '  "eventTimestamp": 1776326688000,\n'
        '  "event": {\n'
        '    "request": {"url": "https://hermes-feishu-gateway.suyee88.workers.dev", "method": "POST"},\n'
        '    "response": {"status": 200}\n'
        '  }\n'
        '}\n',
        encoding="utf-8",
    )

    records, gaps = module._read_cloudflare_records([log_path], "Asia/Shanghai")

    assert gaps == []
    assert len(records) == 1
    assert records[0]["event"] == "feishu.direct_planned.done"
    assert records[0]["correlation_id"] == "feishu:chat:evt_2"
    assert records[0]["direct_exec_elapsed_ms"] == 3336
    assert records[0]["response_status"] == 200
    assert records[0]["__timestamp_ms"] == 1776326689000


def test_build_optimization_analysis_surfaces_cost_regression():
    module = _load_module()
    hourly_cost_pk = [
        {
            "hour_key": "2026-04-15T08:00:00+08:00",
            "current": {"cost_usd": "9.00000000", "session_count": 2},
            "compare_d_1": {"delta_cost_usd": "4.00000000"},
            "compare_d_2": {"delta_cost_usd": "5.00000000"},
        }
    ]
    window_payloads = {
        "current": {
            "billing_summary": {"total_cost_usd": "18.00000000"},
            "function_summary": {
                "feishu_background_exec_worker": {
                    "cost_per_session_usd": "5.00000000",
                    "calibrated_cost_usd": "12.00000000",
                    "cold_start_rate": 0.7,
                }
            },
            "container_summary": {
                "hot_containers": [
                    {
                        "worker_boot_id": "boot-1",
                        "function_name": "feishu_background_exec_worker",
                        "invocation_count": 1,
                        "total_cost_usd": "4.50000000",
                        "cold_start_rate": 1.0,
                        "single_use_container": True,
                    }
                ]
            },
            "session_summary": {"metrics": {"t0_to_t3_ms": {"p50": 12000.0, "p90": 12000.0}, "t0_to_t1_ms": {}, "t0_to_t2_ms": {}}},
        },
        "d-1": {
            "billing_summary": {"total_cost_usd": "10.00000000"},
            "function_summary": {"feishu_background_exec_worker": {"cost_per_session_usd": "2.00000000"}},
            "session_summary": {"metrics": {"t0_to_t3_ms": {"p50": 9000.0, "p90": 9000.0}}},
        },
        "d-2": {
            "billing_summary": {"total_cost_usd": "9.00000000"},
            "function_summary": {"feishu_background_exec_worker": {"cost_per_session_usd": "2.50000000"}},
            "session_summary": {"metrics": {"t0_to_t3_ms": {"p50": 8500.0, "p90": 8500.0}}},
        },
    }

    result = module._build_optimization_analysis(
        window_payloads=window_payloads,
        hourly_cost_pk=hourly_cost_pk,
    )

    assert result["function_efficiency"][0]["function_name"] == "feishu_background_exec_worker"
    assert result["container_efficiency"][0]["issue_tags"] == ["cold_start_heavy", "single_use"]
    assert result["roi_assessment"]["verdict"] == "optimization_not_validated"


def test_build_recent_window_eval_falls_back_to_latest_sessions_when_recent_sample_is_small():
    module = _load_module()
    tz = _tz(module)
    end = datetime(2026, 4, 16, 16, 0, tzinfo=tz)
    sessions = [
        {
            "session_id": "old_1",
            "t0_ms": _ms(end - timedelta(hours=5)),
            "t0_to_t2_ms": 1000,
            "t0_to_t3_minus_model_ms": 2000,
            "allocated_cost_usd": "0.0010",
            "reply_minus_ai_measurement_mode": "provider_wait_observed",
        },
        {
            "session_id": "old_2",
            "t0_ms": _ms(end - timedelta(hours=4, minutes=30)),
            "t0_to_t2_ms": 1100,
            "t0_to_t3_minus_model_ms": 2100,
            "allocated_cost_usd": "0.0011",
            "reply_minus_ai_measurement_mode": "provider_wait_observed",
        },
        {
            "session_id": "recent_1",
            "t0_ms": _ms(end - timedelta(minutes=20)),
            "t0_to_t2_ms": 900,
            "t0_to_t3_minus_model_ms": 1800,
            "allocated_cost_usd": "0.0009",
            "reply_minus_ai_measurement_mode": "provider_wait_observed",
        },
    ]
    payload = {
        "window": {
            "start": (end - timedelta(hours=16)).isoformat(),
            "end": end.isoformat(),
            "hours": 16.0,
        },
        "session_summary": {"sessions": sessions},
        "hourly_official_costs": {},
    }

    recent_window_eval, goal_check_recent = module._build_recent_window_eval(
        {"current": payload},
        recent_hours=1,
        recent_min_sessions=2,
        timezone_name="Asia/Shanghai",
        strict_goal_metric="p90",
    )

    assert recent_window_eval["selection_mode"] == "latest_sessions_fallback"
    assert recent_window_eval["low_sample"] is True
    assert recent_window_eval["selected_session_count"] == 2
    assert goal_check_recent["current"]["reply_minus_ai_ms"] == 2070.0


def test_build_goal_assessment_reports_idle_and_session_cost_targets():
    module = _load_module()
    window_payloads = {
        "current": {
            "billing_summary": {"total_cost_usd": "0.00860000"},
            "session_summary": {
                "completion": {"session_count": 2},
                "sessions": [
                    {
                        "allocated_cost_usd": "0.00410000",
                        "capability_match": True,
                        "preferred_model_selected": True,
                    },
                    {
                        "allocated_cost_usd": "0.00440000",
                        "capability_match": True,
                        "preferred_model_selected": True,
                    },
                ],
                "metrics": {
                    "t0_to_t2_ms": {"p50": 4200.0, "p90": 4200.0},
                    "t0_to_t3_minus_model_ms": {"p50": 12000.0, "p90": 12000.0},
                },
            },
            "hourly_official_costs": {
                "2026-04-15T08:00:00+08:00": Decimal("0.00200000"),
                "2026-04-15T09:00:00+08:00": Decimal("0.01000000"),
            },
            "hourly_session_counts": {
                "2026-04-15T08:00:00+08:00": 0,
                "2026-04-15T09:00:00+08:00": 2,
            },
        }
    }

    cloudflare_summary = {
        "window_summaries": {
            "current": {
                "gateway_request_rows": [
                    {
                        "request_class": "text_plain",
                        "gateway_route_name": "affiliate-general",
                        "cf_cache_status": True,
                        "gateway_eligible": True,
                        "cache_eligible": True,
                        "success": True,
                        "status_code": 200,
                    },
                    {
                        "request_class": "text_coding",
                        "gateway_route_name": "affiliate-coding",
                        "cf_cache_status": False,
                        "gateway_eligible": True,
                        "cache_eligible": True,
                        "success": True,
                        "status_code": 200,
                    },
                ]
            }
        }
    }

    result = module._build_goal_assessment(window_payloads, cloudflare_summary=cloudflare_summary)

    assert result["statuses"]["read_receipt_under_5s"] == "met"
    assert result["statuses"]["reply_minus_ai_under_20s"] == "met"
    assert result["statuses"]["modal_idle_hourly_cost_under_0_005"] == "met"
    assert result["statuses"]["session_cost_under_0_0045"] == "met"
    assert result["statuses"]["cache_hit_rate_over_0_30"] == "met"
    assert result["statuses"]["gateway_error_rate_equals_0"] == "met"
    assert result["statuses"]["rate_limit_triggered_fallback_count_equals_0"] == "not_met"
    assert result["statuses"]["fallback_once_success_rate_equals_1_00"] == "not_met"
    assert result["statuses"]["browser_preprocess_accuracy_equals_1_00"] == "not_met"
    assert result["statuses"]["capability_match_rate_equals_1_00"] == "met"
    assert result["statuses"]["preferred_model_selection_accuracy_over_0_95"] == "met"
    assert result["statuses"]["browser_single_ai_call_completion_rate_over_0_50"] == "not_met"
    assert result["statuses"]["route_decision_explainable_rate_equals_1_00"] == "not_met"
    assert result["current"]["read_receipt_p90_ms"] == 4200.0
    assert result["current"]["reply_minus_ai_p90_ms"] == 12000.0
    assert result["current"]["idle_hourly_p90_cost_usd"] == "0.00200000"
    assert result["current"]["session_cost_p90_usd"] == "0.00437000"
    assert result["current"]["avg_cost_per_session_usd"] == "0.00430000"
    assert result["current"]["cache_eligible_hit_rate"] == "0.50000000"
    assert result["current"]["gateway_error_rate"] == "0.00000000"
    assert result["current"]["capability_match_rate"] == "1.00000000"
    assert result["current"]["preferred_model_selection_accuracy"] == "1.00000000"


def test_build_goal_assessment_keeps_session_target_modal_only_when_cloudflare_cost_exists():
    module = _load_module()
    window_payloads = {
        "current": {
            "billing_summary": {"total_cost_usd": "0.00820000"},
            "session_summary": {
                "completion": {"session_count": 2},
                "sessions": [
                    {
                        "allocated_cost_usd": "0.00410000",
                        "cloudflare_ai_cost_usd": "0.00030000",
                        "total_session_cost_usd": "0.00440000",
                    },
                    {
                        "allocated_cost_usd": "0.00430000",
                        "cloudflare_ai_cost_usd": "0.00050000",
                        "total_session_cost_usd": "0.00480000",
                    },
                ],
                "metrics": {
                    "t0_to_t2_ms": {"p50": 4100.0, "p90": 4100.0},
                    "t0_to_t3_minus_model_ms": {"p50": 11000.0, "p90": 11000.0},
                },
            },
            "hourly_official_costs": {
                "2026-04-15T08:00:00+08:00": Decimal("0.00200000"),
                "2026-04-15T09:00:00+08:00": Decimal("0.01000000"),
            },
            "hourly_session_counts": {
                "2026-04-15T08:00:00+08:00": 0,
                "2026-04-15T09:00:00+08:00": 2,
            },
        }
    }

    result = module._build_goal_assessment(window_payloads)

    assert result["statuses"]["session_cost_under_0_0045"] == "met"
    assert result["current"]["session_cost_p90_usd"] == "0.00428000"
    assert result["current"]["session_total_cost_p90_usd"] == "0.00476000"


def test_build_goal_assessment_prefers_inbound_message_read_metric():
    module = _load_module()
    window_payloads = {
        "current": {
            "billing_summary": {"total_cost_usd": "0.00200000"},
            "session_summary": {
                "completion": {"session_count": 2},
                "sessions": [
                    {"allocated_cost_usd": "0.00100000"},
                    {"allocated_cost_usd": "0.00100000"},
                ],
                "metrics": {
                    "t0_to_t2_ms": {"p50": 12000.0, "p90": 12000.0, "count": 2},
                    "reply_to_read_ms": {"p50": 3200.0, "p90": 4200.0, "count": 2},
                    "t0_to_t3_minus_model_ms": {"p50": 9000.0, "p90": 9000.0, "count": 2},
                },
            },
            "hourly_official_costs": {"2026-04-15T09:00:00+08:00": Decimal("0.00100000")},
            "hourly_session_counts": {"2026-04-15T09:00:00+08:00": 2},
        }
    }

    result = module._build_goal_assessment(window_payloads)

    assert result["current"]["read_receipt_p90_ms"] == 12000.0
    assert result["current"]["read_receipt_measurement_mode"] == "inbound_message_read"
    assert result["statuses"]["read_receipt_under_5s"] == "not_met"


def test_row_ts_ms_falls_back_to_cloudflare_timestamp_ms():
    module = _load_module()

    assert module._row_ts_ms({"__timestamp_ms": 1776770156863}) == 1776770156863


def test_build_cloudflare_summary_preserves_message_read_event_timestamp_for_join(tmp_path, monkeypatch):
    module = _load_module()
    tz = _tz(module)
    start = datetime(2026, 4, 21, 20, 0, tzinfo=tz)
    windows = module._build_analysis_windows(
        hours=1,
        compare_days=[],
        timezone_name="Asia/Shanghai",
        now=start + timedelta(hours=1),
    )
    monkeypatch.setattr(module, "_discover_cloudflare_log_paths", lambda repo_root, explicit_paths: [])
    monkeypatch.setattr(module, "_read_cloudflare_records", lambda paths, timezone_name: ([], []))
    monkeypatch.setattr(module, "_fetch_cloudflare_ai_gateway_records", lambda timezone_name, windows: ([], []))
    monkeypatch.setattr(
        module,
        "_fetch_cloudflare_worker_observability_records",
        lambda repo_root, timezone_name, windows: (
            [
                {
                    "__timestamp_ms": _ms(start + timedelta(minutes=5)),
                    "event": "feishu.message_read.accepted",
                    "event_id": "read_evt_1",
                    "correlation_id": "feishu::read_evt_1",
                    "read_time": "",
                    "message_id_list": ["om_reply_1"],
                }
            ],
            [],
        ),
    )

    summary, gaps = module._build_cloudflare_summary(
        repo_root=tmp_path,
        explicit_paths=[],
        windows=windows,
        timezone_name="Asia/Shanghai",
        session_key_map={},
        event_id_map={},
    )
    read_fact = summary["window_summaries"]["current"]["read_event_facts"][0]

    assert read_fact["__timestamp_ms"] == _ms(start + timedelta(minutes=5))
    assert read_fact["ts"] is None
    assert "cloudflare_logs_missing" not in gaps


def test_build_goal_assessment_falls_back_to_official_average_session_cost_when_allocations_are_zero():
    module = _load_module()
    window_payloads = {
        "current": {
            "billing_summary": {"total_cost_usd": "0.06800000"},
            "cost_calibration": {"allocation_gaps": []},
            "session_summary": {
                "completion": {"session_count": 10},
                "sessions": [
                    {"allocated_cost_usd": "0.00000000"},
                    {"allocated_cost_usd": "0.00000000"},
                ],
                "metrics": {
                    "t0_to_t3_minus_model_ms": {"p90": 8000.0, "count": 10},
                },
            },
            "hourly_official_costs": {"2026-04-15T09:00:00+08:00": Decimal("0.01000000")},
            "hourly_session_counts": {"2026-04-15T09:00:00+08:00": 10},
        }
    }

    result = module._build_goal_assessment(window_payloads)

    assert result["current"]["session_cost_measurement_mode"] == "official_total_div_session_count"
    assert result["current"]["session_cost_p90_usd"] == "0.00680000"
    assert result["statuses"]["session_cost_under_0_0045"] == "not_met"


def test_build_goal_assessment_keeps_session_cost_unavailable_when_function_trace_is_missing():
    module = _load_module()
    window_payloads = {
        "current": {
            "billing_summary": {"total_cost_usd": "0.06800000"},
            "cost_calibration": {
                "allocation_gaps": [
                    {"hour_key": "2026-04-15T09:00:00+08:00", "reason": "official_cost_without_matching_function_trace"}
                ]
            },
            "session_summary": {
                "completion": {"session_count": 10},
                "sessions": [
                    {"allocated_cost_usd": "0.00000000"},
                    {"allocated_cost_usd": "0.00000000"},
                ],
                "metrics": {
                    "t0_to_t3_minus_model_ms": {"p90": 8000.0, "count": 10},
                },
            },
            "hourly_official_costs": {"2026-04-15T09:00:00+08:00": Decimal("0.01000000")},
            "hourly_session_counts": {"2026-04-15T09:00:00+08:00": 10},
        }
    }

    result = module._build_goal_assessment(window_payloads)

    assert result["current"]["session_cost_measurement_mode"] == "insufficient_function_trace"
    assert result["current"]["session_cost_p90_usd"] is None
    assert result["statuses"]["session_cost_under_0_0045"] == "not_met"


def test_build_goal_assessment_can_use_blended_total_policy_when_trace_is_missing():
    module = _load_module()
    window_payloads = {
        "current": {
            "billing_summary": {"total_cost_usd": "0.06800000"},
            "cost_calibration": {
                "allocation_gaps": [
                    {"hour_key": "2026-04-15T09:00:00+08:00", "reason": "official_cost_without_matching_function_trace"}
                ]
            },
            "session_summary": {
                "completion": {"session_count": 2},
                "sessions": [
                    {"allocated_cost_usd": "0.00000000", "total_session_cost_usd": "0.00350000"},
                    {"allocated_cost_usd": "0.00000000", "total_session_cost_usd": "0.00420000"},
                ],
                "metrics": {
                    "t0_to_t3_minus_model_ms": {"p90": 8000.0, "count": 2},
                },
            },
            "hourly_official_costs": {"2026-04-15T09:00:00+08:00": Decimal("0.01000000")},
            "hourly_session_counts": {"2026-04-15T09:00:00+08:00": 2},
        }
    }

    result = module._build_goal_assessment(window_payloads, session_cost_source_policy="blended_total")

    assert result["current"]["session_cost_source_policy"] == "blended_total"
    assert result["current"]["session_cost_measurement_mode"] == "session_total_cost_p90"
    assert result["current"]["session_cost_truth_status"] == "partial"
    assert result["current"]["session_cost_p90_usd"] == "0.00413000"


def test_build_goal_assessment_can_force_official_average_policy_when_trace_is_missing():
    module = _load_module()
    window_payloads = {
        "current": {
            "billing_summary": {"total_cost_usd": "0.06800000"},
            "cost_calibration": {
                "allocation_gaps": [
                    {"hour_key": "2026-04-15T09:00:00+08:00", "reason": "official_cost_without_matching_function_trace"}
                ]
            },
            "session_summary": {
                "completion": {"session_count": 10},
                "sessions": [
                    {"allocated_cost_usd": "0.00000000"},
                    {"allocated_cost_usd": "0.00000000"},
                ],
                "metrics": {
                    "t0_to_t3_minus_model_ms": {"p90": 8000.0, "count": 10},
                },
            },
            "hourly_official_costs": {"2026-04-15T09:00:00+08:00": Decimal("0.01000000")},
            "hourly_session_counts": {"2026-04-15T09:00:00+08:00": 10},
        }
    }

    result = module._build_goal_assessment(window_payloads, session_cost_source_policy="official_average")

    assert result["current"]["session_cost_source_policy"] == "official_average"
    assert result["current"]["session_cost_measurement_mode"] == "official_total_div_session_count_forced"
    assert result["current"]["session_cost_truth_status"] == "estimated"
    assert result["current"]["session_cost_p90_usd"] == "0.00680000"


def test_build_goal_assessment_treats_missing_gate_data_as_not_met():
    module = _load_module()
    result = module._build_goal_assessment({"current": {"billing_summary": {}, "session_summary": {}}})

    assert result["statuses"]["read_receipt_under_5s"] == "not_met"
    assert result["statuses"]["reply_minus_ai_under_20s"] == "not_met"
    assert result["statuses"]["modal_idle_hourly_cost_under_0_005"] == "not_met"
    assert result["statuses"]["session_cost_under_0_0045"] == "not_met"
    assert result["statuses"]["cache_hit_rate_over_0_30"] == "not_met"
    assert result["statuses"]["gateway_error_rate_equals_0"] == "not_met"
    assert result["statuses"]["rate_limit_triggered_fallback_count_equals_0"] == "not_met"
    assert result["statuses"]["fallback_once_success_rate_equals_1_00"] == "not_met"
    assert result["statuses"]["browser_preprocess_accuracy_equals_1_00"] == "not_met"
    assert result["statuses"]["browser_single_ai_call_completion_rate_over_0_50"] == "not_met"
    assert result["statuses"]["capability_match_rate_equals_1_00"] == "not_met"
    assert result["statuses"]["preferred_model_selection_accuracy_over_0_95"] == "not_met"
    assert result["statuses"]["route_decision_explainable_rate_equals_1_00"] == "not_met"


def test_build_goal_check_from_sessions_keeps_session_cost_not_met_when_allocations_are_zero():
    module = _load_module()
    tz = _tz(module)
    end = datetime(2026, 4, 16, 18, 0, tzinfo=tz)
    payload = {
        "window": {
            "start": (end - timedelta(hours=16)).isoformat(),
            "end": end.isoformat(),
            "hours": 16.0,
        },
        "session_summary": {"sessions": []},
        "hourly_official_costs": {"2026-04-16T17:00:00+08:00": Decimal("0.01000000")},
    }
    sessions = [
        {
            "session_id": "evt_recent",
            "t0_ms": _ms(end - timedelta(minutes=10)),
            "allocated_cost_usd": "0.00000000",
            "t0_to_t3_minus_model_ms": 8000.0,
        }
    ]

    result = module._build_goal_check_from_sessions(
        payload,
        sessions=sessions,
        start_ms=_ms(end - timedelta(hours=1)),
        end_ms=_ms(end),
        timezone_name="Asia/Shanghai",
        strict_goal_metric="p90",
    )

    assert result["current"]["modal_session_cost_usd"] is None
    assert result["statuses"]["session_cost_under_0_0045"] == "not_met"


def test_browser_single_ai_call_completion_and_preferred_model_metrics():
    module = _load_module()
    window_payloads = {
        "current": {
            "session_summary": {
                "sessions": [
                    {
                        "session_id": "browser_ok",
                        "requires_browser": True,
                        "route_hint": "cf_browser_first",
                        "ai_call_count": 1,
                        "reply_sent": True,
                        "capability_match": True,
                        "preferred_model_selected": True,
                    },
                    {
                        "session_id": "browser_retry",
                        "requires_browser": True,
                        "route_hint": "cf_browser_first",
                        "ai_call_count": 2,
                        "reply_sent": True,
                        "capability_match": False,
                        "preferred_model_selected": False,
                    },
                ]
            }
        }
    }

    browser_summary = module._build_browser_single_ai_call_completion(window_payloads)
    capability_summary = module._build_capability_match_rate(window_payloads)
    preferred_summary = module._build_preferred_model_selection_accuracy(window_payloads)

    assert browser_summary["current"]["sample_count"] == 2
    assert browser_summary["current"]["success_count"] == 1
    assert browser_summary["current"]["completion_rate"] == 0.5
    assert capability_summary["current"]["sample_count"] == 2
    assert capability_summary["current"]["match_rate"] == 0.5
    assert preferred_summary["current"]["sample_count"] == 2
    assert preferred_summary["current"]["accuracy"] == 0.5


def test_gateway_fallback_browser_and_route_kpi_builders():
    module = _load_module()
    window_payloads = {
        "current": {
            "session_summary": {
                "sessions": [
                    {
                        "session_id": "evt_rate_once",
                        "route_hint": "modal_heavy_exec",
                        "request_class": "text_plain",
                        "route_decision_reason": "429_rate_limit_recoverable",
                        "fallback_reason": "rate_limited",
                        "gateway_error_class": "rate_limited",
                        "ai_call_count": 2,
                        "reply_sent": True,
                    },
                    {
                        "session_id": "evt_rate_multi",
                        "route_hint": "modal_heavy_exec",
                        "request_class": "text_plain",
                        "route_decision_reason": "rate_limit_backoff",
                        "fallback_reason": "rate_limited",
                        "gateway_error_class": "rate_limited",
                        "ai_call_count": 3,
                        "reply_sent": True,
                    },
                    {
                        "session_id": "evt_browser_ok",
                        "route_hint": "cf_browser_first",
                        "request_class": "tool_browser",
                        "requires_browser": True,
                        "capability_match": True,
                        "route_decision_reason": "browser_required",
                    },
                    {
                        "session_id": "evt_browser_bad",
                        "route_hint": "modal_heavy_exec",
                        "request_class": "tool_browser",
                        "requires_browser": True,
                        "capability_match": False,
                        "route_decision_reason": "",
                    },
                    {
                        "session_id": "evt_plain_explainable",
                        "route_hint": "modal_heavy_exec",
                        "request_class": "text_plain",
                        "route_decision_reason": "plain_text_without_attachments_or_browser",
                    },
                    {
                        "session_id": "evt_plain_unexplained",
                        "route_hint": "modal_heavy_exec",
                        "request_class": "text_plain",
                        "route_decision_reason": "",
                    },
                ]
            }
        }
    }
    cloudflare_summary = {
        "window_summaries": {
            "current": {
                "gateway_request_rows": [
                    {"success": True, "status_code": 200},
                    {"success": False, "status_code": 429},
                    {"status_code": 500},
                ]
            }
        }
    }

    gateway_error_rate = module._build_gateway_error_rate(cloudflare_summary)
    rate_limit_triggered_fallback = module._build_rate_limit_triggered_fallback_count(window_payloads)
    fallback_once_success_rate = module._build_fallback_once_success_rate(window_payloads)
    browser_preprocess_accuracy = module._build_browser_preprocess_accuracy(window_payloads)
    route_decision_explainable_rate = module._build_route_decision_explainable_rate(window_payloads)

    assert gateway_error_rate["current"]["sample_count"] == 3
    assert gateway_error_rate["current"]["error_count"] == 2
    assert gateway_error_rate["current"]["error_rate"] == 0.666667

    assert rate_limit_triggered_fallback["current"]["sample_count"] == 6
    assert rate_limit_triggered_fallback["current"]["triggered_fallback_count"] == 2

    assert fallback_once_success_rate["current"]["sample_count"] == 2
    assert fallback_once_success_rate["current"]["measured_count"] == 2
    assert fallback_once_success_rate["current"]["success_count"] == 1
    assert fallback_once_success_rate["current"]["multi_fallback_success_count"] == 1
    assert fallback_once_success_rate["current"]["success_rate"] == 0.5

    assert browser_preprocess_accuracy["current"]["sample_count"] == 2
    assert browser_preprocess_accuracy["current"]["success_count"] == 1
    assert browser_preprocess_accuracy["current"]["accuracy"] == 0.5

    assert route_decision_explainable_rate["current"]["sample_count"] == 6
    assert route_decision_explainable_rate["current"]["explainable_count"] == 4
    assert route_decision_explainable_rate["current"]["explainable_rate"] == 0.666667


def test_apply_cloudflare_observability_adds_worker_only_sessions():
    module = _load_module()
    tz = module._get_timezone("Asia/Shanghai")
    session_summary = {
        "sessions": [
            {
                "session_id": "evt_existing",
                "event_id": "evt_existing",
                "message_id": "",
                "session_key": "",
                "history_visible_mode": "send_success_proxy",
                "measurement_mode": "modal_internal_exec_only",
                "t0_ms": _ms(datetime(2026, 4, 15, 8, 0, tzinfo=tz)),
                "t1_ms": None,
                "t2_ms": None,
                "t3_ms": _ms(datetime(2026, 4, 15, 8, 0, 20, tzinfo=tz)),
                "ack_kind": "unspecified",
                "ingress_strategy": "unspecified",
                "execution_mode": "modal_heavy_exec",
                "reply_visible_proxy": "internal.agent_exec.done",
                "t0_to_t1_ms": None,
                "t0_to_t2_ms": None,
                "t0_to_t3_ms": 20000,
                "t0_to_t3_minus_model_ms": None,
                "t1_to_t3_ms": None,
                "t2_to_t3_ms": None,
                "ai_model_elapsed_ms": None,
                "non_model_elapsed_ms": None,
                "provider_billed_cost_usd": None,
                "external_exec_candidate": None,
                "read_matched": False,
                "reply_sent": True,
            }
        ],
        "metrics": {},
        "completion": {},
        "slowest_sessions": [],
        "session_key_map": {},
    }
    window_payloads = {
        "current": {
            "session_summary": session_summary,
            "hourly_session_counts": {},
        }
    }
    cloudflare_summary = {
        "window_summaries": {
            "current": {
                "matched_session_model_elapsed_ms": {"evt_existing": 9000.0},
                "worker_session_facts": {
                    "evt_cf_only": {
                        "event_id": "evt_cf_only",
                        "correlation_id": "feishu:chat:evt_cf_only",
                        "session_key": "agent:main:feishu:dm:chat",
                        "message_id": "om_in_cf_only",
                        "execution_mode": "deferred_reconcile",
                        "external_exec_candidate": True,
                        "cache_eligible": True,
                        "cache_status": "HIT",
                        "reply_visible_proxy": "feishu.workflow.send.done",
                        "t0_ms": _ms(datetime(2026, 4, 15, 9, 0, tzinfo=tz)),
                        "t3_ms": _ms(datetime(2026, 4, 15, 9, 0, 12, tzinfo=tz)),
                        "ai_model_elapsed_ms": 4000.0,
                        "cloudflare_ai_cost_usd": "0.00020000",
                    }
                },
            }
        }
    }

    module._apply_cloudflare_observability(
        window_payloads,
        cloudflare_summary,
        timezone_name="Asia/Shanghai",
    )

    sessions = window_payloads["current"]["session_summary"]["sessions"]
    by_id = {item["event_id"]: item for item in sessions}
    assert len(sessions) == 2
    assert by_id["evt_existing"]["t0_to_t3_minus_model_ms"] == 11000.0
    assert by_id["evt_cf_only"]["measurement_mode"] == "cloudflare_worker_observability"
    assert by_id["evt_cf_only"]["message_id"] == "om_in_cf_only"
    assert by_id["evt_cf_only"]["reply_visible_proxy"] == "feishu.workflow.send.done"
    assert by_id["evt_cf_only"]["allocated_cost_usd"] == "0.00000000"
    assert by_id["evt_cf_only"]["cache_eligible"] is True
    assert by_id["evt_cf_only"]["cache_status"] == "HIT"
    assert by_id["evt_cf_only"]["cloudflare_ai_cost_usd"] == "0.00020000"
    assert by_id["evt_cf_only"]["t0_to_t3_ms"] == 12000
    assert by_id["evt_cf_only"]["t0_to_t3_minus_model_ms"] == 8000.0
    assert window_payloads["current"]["session_summary"]["completion"]["session_count"] == 2
    assert window_payloads["current"]["hourly_session_counts"]["2026-04-15T09:00:00+08:00"] == 1


def test_apply_cloudflare_observability_keeps_provider_wait_as_reply_minus_ai_source():
    module = _load_module()
    tz = module._get_timezone("Asia/Shanghai")
    session_summary = {
        "sessions": [
            {
                "session_id": "evt_existing",
                "event_id": "evt_existing",
                "message_id": "om_in_existing",
                "session_key": "agent:main:feishu:dm:chat_existing",
                "history_visible_mode": "send_success_proxy",
                "measurement_mode": "modal_internal_exec_only",
                "t0_ms": _ms(datetime(2026, 4, 15, 8, 0, tzinfo=tz)),
                "t1_ms": None,
                "t2_ms": None,
                "t3_ms": _ms(datetime(2026, 4, 15, 8, 1, tzinfo=tz)),
                "reply_to_read_ms": None,
                "ack_kind": "unspecified",
                "ingress_strategy": "unspecified",
                "execution_mode": "modal_heavy_exec",
                "reply_visible_proxy": "internal.agent_exec.done",
                "t0_to_t1_ms": None,
                "t0_to_t2_ms": None,
                "t0_to_t3_ms": 60000,
                "t0_to_t3_minus_model_ms": 12000.0,
                "t1_to_t3_ms": None,
                "t2_to_t3_ms": None,
                "ai_model_elapsed_ms": 48000,
                "provider_wait_elapsed_ms": 48000,
                "non_model_elapsed_ms": None,
                "provider_billed_cost_usd": None,
                "external_exec_candidate": None,
                "read_matched": False,
                "reply_sent": True,
            }
        ],
        "metrics": {},
        "completion": {},
        "slowest_sessions": [],
        "session_key_map": {},
    }
    window_payloads = {"current": {"session_summary": session_summary, "hourly_session_counts": {}}}
    cloudflare_summary = {
        "window_summaries": {
            "current": {
                "matched_session_model_elapsed_ms": {"evt_existing": 120000.0},
                "worker_session_facts": {},
            }
        }
    }

    module._apply_cloudflare_observability(
        window_payloads,
        cloudflare_summary,
        timezone_name="Asia/Shanghai",
    )

    session = window_payloads["current"]["session_summary"]["sessions"][0]
    assert session["ai_model_elapsed_ms"] == 48000
    assert session["provider_wait_elapsed_ms"] == 48000
    assert session["t0_to_t3_minus_model_ms"] == 12000.0


def test_apply_cloudflare_observability_initializes_empty_session_summary():
    module = _load_module()
    tz = module._get_timezone("Asia/Shanghai")
    window_payloads = {
        "current": {
            "session_summary": {},
            "hourly_session_counts": {},
        }
    }
    cloudflare_summary = {
        "window_summaries": {
            "current": {
                "matched_session_model_elapsed_ms": {},
                "worker_session_facts": {
                    "evt_cf_bootstrap": {
                        "event_id": "evt_cf_bootstrap",
                        "correlation_id": "feishu:chat:evt_cf_bootstrap",
                        "session_key": "agent:main:feishu:dm:chat_bootstrap",
                        "message_id": "om_in_bootstrap",
                        "execution_mode": "deferred_reconcile",
                        "external_exec_candidate": True,
                        "reply_visible_proxy": "feishu.direct_planned.done",
                        "t0_ms": _ms(datetime(2026, 4, 15, 10, 0, tzinfo=tz)),
                        "t3_ms": _ms(datetime(2026, 4, 15, 10, 0, 8, tzinfo=tz)),
                        "ai_model_elapsed_ms": 3000.0,
                    }
                },
            }
        }
    }

    module._apply_cloudflare_observability(
        window_payloads,
        cloudflare_summary,
        timezone_name="Asia/Shanghai",
    )

    session_summary = window_payloads["current"]["session_summary"]
    sessions = session_summary["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["event_id"] == "evt_cf_bootstrap"
    assert sessions[0]["message_id"] == "om_in_bootstrap"
    assert sessions[0]["measurement_mode"] == "cloudflare_worker_observability"
    assert sessions[0]["allocated_cost_usd"] == "0.00000000"
    assert sessions[0]["t0_to_t3_ms"] == 8000
    assert sessions[0]["t0_to_t3_minus_model_ms"] == 5000.0
    assert session_summary["completion"]["session_count"] == 1
    assert session_summary["session_key_map"]["agent:main:feishu:dm:chat_bootstrap"] == sessions[0]["session_id"]


def test_augment_recent_sessions_with_feishu_read_api_prefers_inbound_message_id(monkeypatch):
    module = _load_module()
    tz = _tz(module)
    end = datetime(2026, 4, 16, 18, 0, tzinfo=tz)
    payload = {
        "window": {
            "start": (end - timedelta(hours=16)).isoformat(),
            "end": end.isoformat(),
            "hours": 16.0,
        },
        "session_summary": {
            "sessions": [
                {
                    "session_id": "evt_recent",
                    "event_id": "evt_recent",
                    "message_id": "om_in_recent",
                    "reply_send_message_id": "om_out_recent",
                    "reply_send_message_ids": ["om_out_recent"],
                    "t0_ms": _ms(end - timedelta(minutes=10)),
                    "t3_ms": _ms(end - timedelta(minutes=9, seconds=57)),
                    "read_matched": False,
                    "reply_sent": True,
                }
            ],
            "completion": {"session_count": 1},
            "metrics": {},
            "session_key_map": {},
        },
        "hourly_official_costs": {},
    }
    window_payloads = {"current": payload}

    monkeypatch.setattr(
        module,
        "_fetch_feishu_read_users_for_message_ids",
        lambda message_ids: (
            {"om_in_recent": [{"timestamp": str(_ms(end - timedelta(minutes=9, seconds=58)))}]},
            [],
        ),
    )

    gaps = module._augment_recent_sessions_with_feishu_read_api(
        window_payloads,
        recent_hours=3,
        recent_min_sessions=20,
        timezone_name="Asia/Shanghai",
    )

    session = window_payloads["current"]["session_summary"]["sessions"][0]
    assert gaps == []
    assert session["read_matched"] is True
    assert session["t2_ms"] == _ms(end - timedelta(minutes=9, seconds=58))
    assert session["reply_to_read_ms"] is None
    assert session["read_match_source"] == "feishu_read_users_api"
    assert session["read_receipt_measurement_mode"] == "feishu_read_users_api"


def test_augment_recent_sessions_with_feishu_read_api_queries_only_primary_message_id(monkeypatch):
    module = _load_module()
    tz = _tz(module)
    end = datetime(2026, 4, 16, 18, 0, tzinfo=tz)
    payload = {
        "window": {
            "start": (end - timedelta(hours=16)).isoformat(),
            "end": end.isoformat(),
            "hours": 16.0,
        },
        "session_summary": {
            "sessions": [
                {
                    "session_id": "evt_recent",
                    "event_id": "evt_recent",
                    "message_id": "om_in_recent",
                    "reply_send_message_id": "om_out_recent",
                    "reply_send_message_ids": ["om_out_recent", "om_out_recent_2"],
                    "t0_ms": _ms(end - timedelta(minutes=10)),
                    "t3_ms": _ms(end - timedelta(minutes=9, seconds=57)),
                    "read_matched": False,
                    "reply_sent": True,
                }
            ],
            "completion": {"session_count": 1},
            "metrics": {},
            "session_key_map": {},
        },
        "hourly_official_costs": {},
    }
    window_payloads = {"current": payload}
    captured_ids: list[str] = []

    def _fake_fetch(message_ids):
        captured_ids.extend(message_ids)
        return {}, []

    monkeypatch.setattr(module, "_fetch_feishu_read_users_for_message_ids", _fake_fetch)

    module._augment_recent_sessions_with_feishu_read_api(
        window_payloads,
        recent_hours=3,
        recent_min_sessions=20,
        timezone_name="Asia/Shanghai",
    )

    assert captured_ids == ["om_in_recent"]


def test_fetch_feishu_read_users_without_user_id_type_prefers_read_users_shape(monkeypatch):
    module = _load_module()

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object] | None] = []

        def request_json(self, method: str, path: str, *, params=None, **_kwargs):
            self.calls.append(params)
            assert method == "GET"
            assert path.endswith("/om_target/read_users")
            assert params in (None, {})
            return {
                "read_users": [
                    {"timestamp": "1710000000000", "reader_id": {"open_id": "ou_reader"}},
                ],
                "has_more": False,
            }

    client = FakeClient()
    fake_module = type(
        "FakeFeishuApiModule",
        (),
        {"build_feishu_client": staticmethod(lambda timeout=20: client)},
    )
    monkeypatch.setattr(module, "_load_feishu_api_module", lambda: fake_module)

    result, gaps = module._fetch_feishu_read_users_for_message_ids(["om_target"])

    assert gaps == []
    assert result == {
        "om_target": [
            {"timestamp": "1710000000000", "reader_id": {"open_id": "ou_reader"}},
        ]
    }
    assert client.calls == [None]


def test_fetch_feishu_read_users_falls_back_to_open_id_param_on_legacy_api(monkeypatch):
    module = _load_module()

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object] | None] = []

        def request_json(self, method: str, path: str, *, params=None, **_kwargs):
            self.calls.append(params)
            assert method == "GET"
            assert path.endswith("/om_target/read_users")
            if params in (None, {}):
                raise RuntimeError("400 Bad Request")
            assert params == {"user_id_type": "open_id"}
            return {
                "items": [
                    {"timestamp": "1710000005000", "reader_id": {"open_id": "ou_reader"}},
                ],
                "has_more": False,
            }

    client = FakeClient()
    fake_module = type(
        "FakeFeishuApiModule",
        (),
        {"build_feishu_client": staticmethod(lambda timeout=20: client)},
    )
    monkeypatch.setattr(module, "_load_feishu_api_module", lambda: fake_module)

    result, gaps = module._fetch_feishu_read_users_for_message_ids(["om_target"])

    assert gaps == []
    assert result == {
        "om_target": [
            {"timestamp": "1710000005000", "reader_id": {"open_id": "ou_reader"}},
        ]
    }
    assert client.calls == [None, {"user_id_type": "open_id"}]


def test_fetch_feishu_read_users_ignores_nonfatal_message_id_errors(monkeypatch):
    module = _load_module()

    class FakeClient:
        def request_json(self, method: str, path: str, *, params=None, **_kwargs):
            assert method == "GET"
            assert path.endswith("/om_target/read_users")
            assert params == {"user_id_type": "open_id"}
            raise RuntimeError("code=230012 Bot is NOT the sender of the message.")

    monkeypatch.setattr(module, "_build_triparty_feishu_client", lambda timeout=20: FakeClient())

    result, gaps = module._fetch_feishu_read_users_for_message_ids(["om_target"])

    assert result == {}
    assert gaps == []
