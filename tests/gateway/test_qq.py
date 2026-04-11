"""Tests for the QQ official bot webhook adapter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides


class TestPlatformEnum:
    def test_qq_in_platform_enum(self):
        assert Platform.QQ.value == "qq"


class TestConfigEnvOverrides:
    def test_qq_config_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("QQ_APP_ID", "app-123")
        monkeypatch.setenv("QQ_APP_SECRET", "secret-456")
        monkeypatch.setenv("QQ_CONNECTION_MODE", "webhook")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.QQ in config.platforms
        assert config.platforms[Platform.QQ].enabled is True
        assert config.platforms[Platform.QQ].extra["app_id"] == "app-123"
        assert config.platforms[Platform.QQ].extra["connection_mode"] == "webhook"
        assert Platform.QQ in config.get_connected_platforms()


def _make_adapter():
    from gateway.platforms.qq import QQAdapter

    config = PlatformConfig(
        enabled=True,
        extra={
            "app_id": "app-123",
            "app_secret": "secret-456",
            "connection_mode": "webhook",
        },
    )
    return QQAdapter(config)


class TestValidationFlow:
    def test_op13_validation_returns_signature(self):
        adapter = _make_adapter()

        result = asyncio.run(
            adapter.handle_webhook_payload(
                {"op": 13, "d": {"plain_token": "plain-xyz", "event_ts": "1725442341"}},
                headers={"X-Bot-Appid": "app-123"},
            )
        )

        assert result["plain_token"] == "plain-xyz"
        assert len(result["signature"]) == 128


class TestInboundEventNormalization:
    def test_group_at_message_is_normalized_and_dispatched(self, monkeypatch):
        import gateway.platforms.qq as qq_module

        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        monkeypatch.setattr(adapter, "_get_api", lambda: object())

        class FakeGroupMessage:
            def __init__(self, api, event_id, data):
                del api, event_id
                self.id = data["id"]
                self.content = data["content"]
                self.group_openid = data["group_openid"]
                self.timestamp = data["timestamp"]
                self.author = SimpleNamespace(member_openid=data["author"]["member_openid"])
                self.message_reference = SimpleNamespace(message_id=None)

        monkeypatch.setattr(qq_module, "GroupMessage", FakeGroupMessage)

        result = asyncio.run(
            adapter.handle_webhook_payload(
                {
                    "op": 0,
                    "id": "evt-1",
                    "t": "GROUP_AT_MESSAGE_CREATE",
                    "d": {
                        "id": "msg-1",
                        "content": "<@!bot> hello hermes",
                        "group_openid": "group-openid-1",
                        "timestamp": "1725442341",
                        "author": {"member_openid": "member-openid-1"},
                    },
                },
                headers={"X-Bot-Appid": "app-123"},
            )
        )

        assert result == {"op": 12}
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello hermes"
        assert event.source.chat_id == "group:group-openid-1"
        assert event.source.user_id == "member-openid-1"
        assert event.message_id == "msg-1"

    def test_duplicate_event_is_acked_without_dispatch(self, monkeypatch):
        import gateway.platforms.qq as qq_module

        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        monkeypatch.setattr(adapter, "_get_api", lambda: object())

        class FakeC2CMessage:
            def __init__(self, api, event_id, data):
                del api, event_id
                self.id = data["id"]
                self.content = data["content"]
                self.timestamp = data["timestamp"]
                self.author = SimpleNamespace(user_openid=data["author"]["user_openid"])
                self.message_reference = SimpleNamespace(message_id=None)

        monkeypatch.setattr(qq_module, "C2CMessage", FakeC2CMessage)

        payload = {
            "op": 0,
            "id": "evt-dup",
            "t": "C2C_MESSAGE_CREATE",
            "d": {
                "id": "msg-2",
                "content": "hello",
                "timestamp": "1725442341",
                "author": {"user_openid": "user-openid-1"},
            },
        }

        first = asyncio.run(adapter.handle_webhook_payload(payload, headers={"X-Bot-Appid": "app-123"}))
        second = asyncio.run(adapter.handle_webhook_payload(payload, headers={"X-Bot-Appid": "app-123"}))

        assert first == {"op": 12}
        assert second == {"op": 12}
        assert adapter.handle_message.await_count == 1


class TestOutboundSend:
    def test_send_group_message_uses_passive_reply_sequence(self):
        adapter = _make_adapter()

        class FakeAPI:
            def __init__(self):
                self.calls = []

            async def post_group_message(self, **kwargs):
                self.calls.append(kwargs)
                return {"id": f"out-{len(self.calls)}"}

        fake_api = FakeAPI()
        adapter._api = fake_api

        result = asyncio.run(adapter.send("group:group-openid-1", "hello", reply_to="msg-1"))

        assert result.success is True
        assert result.message_id == "out-1"
        assert fake_api.calls[0]["group_openid"] == "group-openid-1"
        assert fake_api.calls[0]["msg_id"] == "msg-1"
        assert fake_api.calls[0]["msg_seq"] == 1
