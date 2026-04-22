from __future__ import annotations

from internal.feishu.delivery_diagnostics import analyze_feishu_delivery_config, sanitize_callback_info


def test_analyze_feishu_delivery_config_detects_websocket_callback_mismatch():
    result = analyze_feishu_delivery_config(
        runtime_mode="webhook",
        callback_info={
            "callback_type": "websocket",
            "subscribed_callbacks": ["card.action.trigger"],
        },
        expected_webhook_url="https://hermes.example.com/feishu/webhook",
    )

    assert result["status"] == "error"
    assert result["mode_matches"] is False
    assert "im.message.receive_v1" in result["missing_callbacks"]
    assert any(issue["code"] == "callback_type_mismatch" for issue in result["issues"])
    assert any(issue["code"] == "missing_callbacks" for issue in result["issues"])


def test_analyze_feishu_delivery_config_accepts_matching_webhook_configuration():
    result = analyze_feishu_delivery_config(
        runtime_mode="webhook",
        callback_info={
            "callback_type": "webhook",
            "callback_url": "https://hermes.example.com/feishu/webhook",
            "subscribed_callbacks": [
                "im.message.receive_v1",
                "im.message.message_read_v1",
                "card.action.trigger",
                "application.bot.menu_v6",
            ],
        },
        expected_webhook_url="https://hermes.example.com/feishu/webhook",
    )

    assert result["status"] == "ok"
    assert result["mode_matches"] is True
    assert result["webhook_url_matches"] is True
    assert result["missing_callbacks"] == []


def test_analyze_feishu_delivery_config_accepts_legacy_root_webhook_url():
    result = analyze_feishu_delivery_config(
        runtime_mode="webhook",
        callback_info={
            "callback_type": "webhook",
            "callback_url": "https://hermes.example.com/",
            "subscribed_callbacks": [
                "im.message.receive_v1",
                "im.message.message_read_v1",
                "card.action.trigger",
                "application.bot.menu_v6",
            ],
        },
        expected_webhook_url="https://hermes.example.com/feishu/webhook",
    )

    assert result["status"] == "ok"
    assert result["webhook_url_matches"] is True


def test_sanitize_callback_info_masks_secret_fields():
    sanitized = sanitize_callback_info(
        {
            "callback_type": "webhook",
            "callback_url": "https://hermes.example.com/feishu/webhook",
            "verification_token": "verify-token-123",
            "encrypt_key": "encrypt-key-456",
            "subscribed_callbacks": ["im.message.receive_v1", "card.action.trigger"],
        }
    )

    assert sanitized["callback_type"] == "webhook"
    assert sanitized["callback_url"] == "https://hermes.example.com/feishu/webhook"
    assert sanitized["verification_token"] == "<masked len=16>"
    assert sanitized["encrypt_key"] == "<masked len=15>"
