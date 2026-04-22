from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/kpi 24 --recent-hours 6", platform=Platform.FEISHU):
    source = SessionSource(
        platform=platform,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    runner.session_store = MagicMock()

    from gateway.hooks import HookRegistry

    runner.hooks = HookRegistry()
    return runner


@pytest.mark.asyncio
async def test_handle_kpi_command_formats_gateway_summary():
    runner = _make_runner()
    event = _make_event()

    with patch(
        "tools.feishu_kpi_tools.generate_feishu_kpi_report",
        return_value={"success": True, "summary": {"hours": 24}, "json_out": "a.json", "markdown_out": "a.md"},
    ) as generate_mock, patch(
        "tools.feishu_kpi_tools.format_feishu_kpi_gateway_summary",
        return_value="kpi summary",
    ):
        result = await runner._handle_kpi_command(event)

    assert result == "kpi summary"
    generate_mock.assert_called_once_with(
        hours=24,
        compare_days=[1, 2],
        timezone="Asia/Shanghai",
        recent_hours=6,
        recent_min_sessions=20,
        include_analytics_snapshot=True,
    )
