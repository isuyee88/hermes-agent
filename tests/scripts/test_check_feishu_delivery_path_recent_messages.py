from __future__ import annotations

from scripts import check_feishu_delivery_path as delivery_path
from scripts.check_feishu_delivery_path import _analyze_recent_app_messages, _analyze_recent_user_messages


def test_analyze_recent_user_messages_flags_messages_targeting_other_bot():
    result = _analyze_recent_user_messages(
        [
            {
                "create_time": "1700000001000",
                "message_id": "m1",
                "msg_type": "text",
                "body": {"content": '{"text":"hello @_user_1"}'},
                "sender": {"sender_type": "user"},
                "mentions": [{"id": "ou_other_bot"}],
            }
        ],
        bot_open_id="ou_current_bot",
        since_ms=1700000000000,
        known_bots_by_open_id={
            "ou_other_bot": delivery_path.BotIdentity(
                app_id_env="FEISHU_APP_ID2",
                app_secret_env="FEISHU_APP_SECRET2",
                app_id="cli_other",
                app_name="Other Hermes",
                bot_open_id="ou_other_bot",
            )
        },
    )

    assert result["recent_user_message_count"] == 1
    assert result["recent_user_messages_targeting_current_bot"] == 0
    assert result["recent_user_messages_without_mentions"] == 0
    assert result["recent_user_messages_targeting_other_bot_ids"] == [
        {
            "open_id": "ou_other_bot",
            "known_in_env": True,
            "app_id": "cli_other",
            "app_name": "Other Hermes",
            "app_id_env": "FEISHU_APP_ID2",
            "count": 1,
        }
    ]


def test_analyze_recent_user_messages_counts_targeted_and_plain_messages():
    result = _analyze_recent_user_messages(
        [
            {
                "create_time": "1700000001000",
                "message_id": "m1",
                "msg_type": "text",
                "body": {"content": '{"text":"hello"}'},
                "sender": {"sender_type": "user"},
                "mentions": [],
            },
            {
                "create_time": "1700000002000",
                "message_id": "m2",
                "msg_type": "text",
                "body": {"content": '{"text":"hello @_user_1"}'},
                "sender": {"sender_type": "user"},
                "mentions": [{"id": "ou_current_bot"}],
            },
        ],
        bot_open_id="ou_current_bot",
        since_ms=1700000000000,
    )

    assert result["recent_user_message_count"] == 2
    assert result["recent_user_messages_targeting_current_bot"] == 1
    assert result["recent_user_messages_without_mentions"] == 1
    assert result["recent_user_messages_targeting_other_bot_ids"] == []


def test_analyze_recent_app_messages_flags_other_replying_app():
    result = _analyze_recent_app_messages(
        [
            {
                "create_time": "1700000001000",
                "message_id": "m1",
                "msg_type": "post",
                "body": {"content": '{"text":"reply"}'},
                "sender": {"sender_type": "app", "id_type": "app_id", "id": "cli_other"},
            }
        ],
        current_app_id="cli_current",
        since_ms=1700000000000,
        known_apps_by_id={
            "cli_other": delivery_path.BotIdentity(
                app_id_env="FEISHU_APP_ID2",
                app_secret_env="FEISHU_APP_SECRET2",
                app_id="cli_other",
                app_name="Other Hermes",
                bot_open_id="ou_other_bot",
            )
        },
    )

    assert result["recent_app_message_count"] == 1
    assert result["recent_app_messages_from_current_app"] == 0
    assert result["recent_app_messages_from_other_app_ids"] == [
        {
            "app_id": "cli_other",
            "known_in_env": True,
            "app_name": "Other Hermes",
            "app_id_env": "FEISHU_APP_ID2",
            "bot_open_id": "ou_other_bot",
            "count": 1,
        }
    ]
    assert result["latest_recent_app_message_sender_app_id"] == "cli_other"


def test_inspect_recent_chat_activity_prioritizes_other_replying_app_blocker(monkeypatch):
    monkeypatch.setattr(
        delivery_path,
        "_iter_chat_messages",
        lambda token, chat_id: [
            {
                "create_time": "1700000001000",
                "message_id": "user-1",
                "msg_type": "text",
                "body": {"content": '{"text":"hello"}'},
                "sender": {"sender_type": "user"},
                "mentions": [],
            },
            {
                "create_time": "1700000002000",
                "message_id": "app-1",
                "msg_type": "post",
                "body": {"content": '{"text":"reply"}'},
                "sender": {"sender_type": "app", "id_type": "app_id", "id": "cli_other"},
            },
        ],
    )
    monkeypatch.setattr(delivery_path.time, "time", lambda: 1700000002)

    result = delivery_path._inspect_recent_chat_activity(
        "tenant-token",
        [{"chat_id": "oc_1", "name": "Group 1"}],
        current_app_id="cli_current",
        bot_open_id="ou_current_bot",
        window_minutes=1,
        known_bots_by_open_id={},
        known_apps_by_id={
            "cli_current": delivery_path.BotIdentity(
                app_id_env="FEISHU_APP_ID3",
                app_secret_env="FEISHU_APP_SECRET3",
                app_id="cli_current",
                app_name="Current Hermes",
                bot_open_id="ou_current_bot",
            ),
            "cli_other": delivery_path.BotIdentity(
                app_id_env="FEISHU_APP_ID2",
                app_secret_env="FEISHU_APP_SECRET2",
                app_id="cli_other",
                app_name="Other Hermes",
                bot_open_id="ou_other_bot",
            ),
        },
    )

    assert result["total_recent_user_messages"] == 1
    assert result["total_recent_app_messages"] == 1
    assert result["total_recent_app_messages_from_current_app"] == 0
    assert result["total_recent_app_messages_from_other_app_ids"] == [
        {
            "app_id": "cli_other",
            "known_in_env": True,
            "app_name": "Other Hermes",
            "app_id_env": "FEISHU_APP_ID2",
            "bot_open_id": "ou_other_bot",
            "count": 1,
        }
    ]
    assert result["primary_blocker"] == "recent_messages_answered_by_other_app"


def test_iter_chat_messages_requests_recent_messages_first(monkeypatch):
    captured_params = []

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"items": [], "has_more": False}}

    def _fake_get(url, *, headers, params, timeout):
        captured_params.append(params.copy())
        return _FakeResponse()

    monkeypatch.setattr(delivery_path.requests, "get", _fake_get)

    result = delivery_path._iter_chat_messages("tenant-token", "oc_1")

    assert result == []
    assert captured_params
    assert captured_params[0]["sort_type"] == "ByCreateTimeDesc"


def test_filter_visible_chats_restricts_to_target_chat():
    result = delivery_path._filter_visible_chats(
        [
            {"chat_id": "oc_keep", "name": "Keep"},
            {"chat_id": "oc_drop", "name": "Drop"},
        ],
        "oc_keep",
    )

    assert result == [{"chat_id": "oc_keep", "name": "Keep"}]
