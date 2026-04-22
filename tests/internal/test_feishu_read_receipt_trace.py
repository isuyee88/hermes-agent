from __future__ import annotations

import time
from pathlib import Path

from internal.feishu.trace import (
    build_message_read_correlation_trace_extras,
    correlate_message_read_event,
    read_recent_trace_rows,
)


def test_correlate_message_read_event_matches_inbound_and_reply(tmp_path: Path) -> None:
    trace_path = tmp_path / "feishu_trace.jsonl"
    now = int(time.time())
    trace_path.write_text(
        "\n".join(
            [
                (
                    '{"ts": %d, "stage": "webhook.accepted", "event_type": "im.message.receive_v1", '
                    '"event_id": "evt_in", "message_id": "om_in_1", "chat_id": "oc_demo"}'
                )
                % (now - 10),
                (
                    '{"ts": %d, "stage": "gateway.send.done", "event_type": "im.message.receive_v1", '
                    '"event_id": "evt_in", "message_id": "om_in_1", "send_message_id": "om_out_1", '
                    '"send_success": true, "session_key": "agent:main:feishu:group:oc_demo:ou_demo", '
                    '"correlation_id": "feishu:oc_demo:evt_in", "chat_id": "oc_demo", "actor_id": "ou_demo"}'
                )
                % (now - 5)
            ]
        ),
        encoding="utf-8",
    )
    read_event = {
        "event_id": "evt_read",
        "reader_open_id": "ou_reader",
        "read_time": 1_712_970_029_000,
        "message_id_list": ["om_in_1", "om_out_1"],
    }

    result = correlate_message_read_event(read_event, trace_path=trace_path)

    assert result["matched_count"] == 2
    by_id = {item["read_message_id"]: item for item in result["matches"]}
    assert by_id["om_in_1"]["matched_kind"] == "inbound"
    assert by_id["om_in_1"]["read_match_source"] == "inbound_message_id"
    assert by_id["om_out_1"]["matched_kind"] == "reply"
    assert by_id["om_out_1"]["read_match_source"] == "reply_send_message_id"
    assert by_id["om_out_1"]["session_key"] == "agent:main:feishu:group:oc_demo:ou_demo"


def test_build_message_read_correlation_trace_extras_includes_latency() -> None:
    read_event = {
        "reader_open_id": "ou_reader",
        "read_time": 1_712_970_029_000,
    }
    match = {
        "read_message_id": "om_out_1",
        "matched_kind": "reply",
        "read_match_source": "reply_send_message_id",
        "matched_stage": "gateway.send.done",
        "matched_event_id": "evt_in",
        "matched_chat_id": "oc_demo",
        "session_key": "agent:main:feishu:group:oc_demo:ou_demo",
        "inbound_message_id": "om_in_1",
        "reply_send_message_id": "om_out_1",
        "reply_send_ts_ms": 1_712_970_025_000,
    }

    extras = build_message_read_correlation_trace_extras(read_event, match)

    assert extras["read_latency_ms"] == 4000
    assert extras["reply_send_message_id"] == "om_out_1"
    assert extras["inbound_message_id"] == "om_in_1"


def test_read_recent_trace_rows_filters_old_rows(tmp_path: Path) -> None:
    trace_path = tmp_path / "feishu_trace.jsonl"
    now = int(time.time())
    trace_path.write_text(
        "\n".join(
            [
                '{"ts": %d, "stage": "webhook.accepted", "message_id": "om_old"}' % (now - 200_000),
                '{"ts": %d, "stage": "webhook.accepted", "message_id": "om_new"}' % now,
            ]
        ),
        encoding="utf-8",
    )

    rows = read_recent_trace_rows(trace_path=trace_path, limit=10, lookback_seconds=3600)

    assert [row["message_id"] for row in rows] == ["om_new"]
