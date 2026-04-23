from __future__ import annotations

from scripts.check_feishu_delivery_path import (
    _build_app_repair_actions,
    _build_expected_webhook_url,
    _summarize_chat_topology_snapshots,
    _summarize_matrix,
    _summarize_published_app_versions,
)


def test_build_expected_webhook_url_appends_webhook_path_for_bare_base_url():
    assert _build_expected_webhook_url("https://hermes.example.com") == "https://hermes.example.com/feishu/webhook"


def test_build_expected_webhook_url_preserves_existing_explicit_callback_path():
    assert (
        _build_expected_webhook_url("https://hermes.example.com/custom/callback")
        == "https://hermes.example.com/custom/callback"
    )


def test_summarize_published_app_versions_uses_latest_publish_time_and_flags_missing_callbacks():
    result = _summarize_published_app_versions(
        [
            {
                "app_id": "cli_old",
                "app_name": "Hermes Old",
                "publish_time": "100",
                "create_time": "90",
                "event_infos": [{"event_type": "im.message.receive_v1"}],
            },
            {
                "app_id": "cli_new",
                "app_name": "Hermes New",
                "publish_time": "200",
                "create_time": "150",
                "event_infos": [
                    {"event_type": "im.message.receive_v1"},
                    {"event_type": "application.bot.menu_v6"},
                ],
                "remark": {"visibility": {"visible_list": {"open_ids": ["ou_user_1"]}}},
            },
        ],
        [
            "im.message.receive_v1",
            "im.message.message_read_v1",
            "application.bot.menu_v6",
        ],
    )

    assert result["available"] is True
    assert result["latest_app_id"] == "cli_new"
    assert result["message_receive_enabled"] is True
    assert result["message_read_enabled"] is False
    assert result["missing_required_callbacks"] == ["im.message.message_read_v1"]
    assert result["latest_visibility_open_ids"] == ["ou_user_1"]


def test_summarize_chat_topology_snapshots_flags_multiple_bots_and_incomplete_members():
    result = _summarize_chat_topology_snapshots(
        [
            {
                "detail": {"ok": True, "bot_count": 8, "user_count": 1},
                "members": {"ok": True, "member_total": 1, "returned_count": 1},
                "blockers": [
                    "multiple_bots_in_target_chat",
                    "chat_member_view_incomplete",
                    "user_scope_missing_chat_members_read",
                ],
            }
        ]
    )

    assert result["chat_count"] == 1
    assert result["total_known_bots"] == 8
    assert result["total_known_users"] == 1
    assert result["chats_with_multiple_bots"] == 1
    assert result["chats_with_incomplete_members"] == 1
    assert result["chats_with_user_scope_missing_chat_members_read"] == 1


def test_summarize_matrix_reports_multiple_bots_when_no_stronger_signal_exists():
    status, summary = _summarize_matrix(
        [
            {
                "app_id": "cli_app3",
                "visible_chat_count": 1,
                "delivery_ready": False,
                "chat_topology": {
                    "chats_with_multiple_bots": 1,
                    "total_known_bots": 8,
                },
                "recent_chat_activity": {"primary_blocker": ""},
                "published_version_summary": {"missing_required_callbacks": []},
                "primary_blocker": "",
                "delivery_health": {"issues": [{"code": "callback_url_mismatch"}]},
            }
        ]
    )

    assert status == "error"
    assert "8" in summary
    assert "single-bot Hermes routing" in summary


def test_build_app_repair_actions_prioritizes_message_read_and_multi_bot_repairs():
    actions = _build_app_repair_actions(
        {
            "app_id": "cli_app3",
            "app_name": "Hermes",
            "published_version_summary": {
                "missing_required_callbacks": [
                    "im.message.message_read_v1",
                    "card.action.trigger",
                    "application.bot.menu_v6",
                ]
            },
            "delivery_health": {
                "recommendation_actions": ["将飞书应用事件订阅请求地址设置为 `https://hermes.example.com/feishu/webhook`。"]
            },
            "recent_chat_activity": {
                "primary_blocker": "recent_messages_answered_by_other_app",
                "total_recent_app_messages_from_other_app_ids": [
                    {"app_id": "cli_other", "app_name": "Other Hermes", "known_in_env": True}
                ],
            },
            "chat_topology": {
                "chats_with_multiple_bots": 1,
                "chats_with_expired_user_token": 1,
                "chats_with_user_scope_missing_chat_members_read": 1,
                "chats_with_incomplete_members": 1,
            },
        }
    )

    codes = [item["code"] for item in actions]
    assert codes[:3] == [
        "publish_card_action_callback",
        "publish_message_read_callback",
        "reduce_multi_bot_contention",
    ]
    assert "publish_card_action_callback" in codes
    assert "publish_bot_menu_callback" in codes
    assert "stop_other_replying_app" in codes
    assert "refresh_user_access_token" in codes
    assert "grant_user_chat_member_scope" in codes
    assert "rerun_member_audit" in codes


def test_summarize_matrix_reports_user_scope_gap_when_member_audit_lacks_chat_read_scope():
    status, summary = _summarize_matrix(
        [
            {
                "app_id": "cli_app3",
                "visible_chat_count": 1,
                "delivery_ready": True,
                "chat_topology": {
                    "chats_with_multiple_bots": 0,
                    "chats_with_user_scope_missing_chat_members_read": 1,
                },
                "recent_chat_activity": {"primary_blocker": ""},
                "published_version_summary": {"missing_required_callbacks": []},
                "primary_blocker": "",
                "delivery_health": {"issues": []},
            }
        ]
    )

    assert status == "error"
    assert "im:chat.members:read" in summary


def test_summarize_matrix_reports_published_card_action_gap():
    status, summary = _summarize_matrix(
        [
            {
                "app_id": "cli_app3",
                "visible_chat_count": 1,
                "delivery_ready": False,
                "chat_topology": {
                    "chats_with_multiple_bots": 0,
                },
                "recent_chat_activity": {"primary_blocker": ""},
                "published_version_summary": {"missing_required_callbacks": ["card.action.trigger"]},
                "primary_blocker": "published_version_missing_card_action_callback",
                "delivery_health": {
                    "issues": [
                        {
                            "code": "missing_callbacks",
                            "message": "Feishu app is missing required callbacks: card.action.trigger",
                        }
                    ]
                },
            }
        ]
    )

    assert status == "error"
    assert "card.action.trigger" in summary


def test_summarize_matrix_reports_published_menu_gap():
    status, summary = _summarize_matrix(
        [
            {
                "app_id": "cli_app3",
                "visible_chat_count": 1,
                "delivery_ready": False,
                "chat_topology": {
                    "chats_with_multiple_bots": 0,
                },
                "recent_chat_activity": {"primary_blocker": ""},
                "published_version_summary": {"missing_required_callbacks": ["application.bot.menu_v6"]},
                "primary_blocker": "published_version_missing_bot_menu_callback",
                "delivery_health": {
                    "issues": [
                        {
                            "code": "missing_callbacks",
                            "message": "Feishu app is missing required callbacks: application.bot.menu_v6",
                        }
                    ]
                },
            }
        ]
    )

    assert status == "error"
    assert "application.bot.menu_v6" in summary
