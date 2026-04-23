from __future__ import annotations

from scripts.feishu_read_receipt_probe import (
    _extract_read_users_items,
    _filter_recent_app_messages,
    _summarize_message,
)


def test_extract_read_users_items_supports_items_and_read_users_keys():
    assert _extract_read_users_items({"data": {"items": [{"reader_id": "ou_1"}]}}) == [{"reader_id": "ou_1"}]
    assert _extract_read_users_items({"data": {"read_users": [{"reader_id": "ou_2"}]}}) == [{"reader_id": "ou_2"}]


def test_filter_recent_app_messages_keeps_only_current_app_messages():
    messages = [
        {
            "message_id": "om_current",
            "create_time": "9999999999999",
            "sender": {"sender_type": "app", "sender_id": "cli_app3"},
        },
        {
            "message_id": "om_other_app",
            "create_time": "9999999999999",
            "sender": {"sender_type": "app", "sender_id": "cli_other"},
        },
        {
            "message_id": "om_user",
            "create_time": "9999999999999",
            "sender": {"sender_type": "user", "sender_id": "ou_user"},
        },
    ]

    result = _filter_recent_app_messages(messages, current_app_id="cli_app3", window_minutes=180)

    assert [item["message_id"] for item in result] == ["om_current"]


def test_summarize_message_reports_first_read_latency():
    message = {
        "message_id": "om_1",
        "create_time": "1000",
        "msg_type": "text",
        "body": {"content": '{"text":"hello"}'},
    }
    read_users = [
        {"reader_id": "ou_a", "read_time": "2400"},
        {"reader_id": "ou_b", "read_time": "1800"},
    ]

    summary = _summarize_message(message, read_users)

    assert summary["read_user_count"] == 2
    assert summary["first_read_time_ms"] == 1800
    assert summary["first_read_latency_ms"] == 800
    assert summary["reader_open_ids"] == ["ou_a", "ou_b"]
