from __future__ import annotations

import pytest
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = GatewayConfig()
    runner.config.session = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.session_store = None
    runner._session_model_overrides = {}
    runner._session_model_list_offsets = {}
    runner._pending_model_notes = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._session_route_ttl_seconds = 7200
    runner._load_model_runtime_config = lambda: (
        "openrouter/default-model",
        "openrouter",
        "https://openrouter.ai/api/v1",
        "",
        None,
    )
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_test",
            chat_type="dm",
            user_id="user-1",
        ),
    )


@pytest.mark.asyncio
async def test_provider_command_filters_to_openrouter_and_nvidia(monkeypatch):
    import hermes_cli.models as cli_models

    runner = _make_runner()
    event = _make_event("/provider")
    session_key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[session_key] = {
        "model": "qwen/qwq-32b",
        "provider": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "nvapi-test",
        "api_mode": "chat_completions",
    }

    monkeypatch.setattr(
        cli_models,
        "list_available_providers",
        lambda: [
            {"id": "openrouter", "label": "OpenRouter", "aliases": [], "authenticated": True},
            {"id": "nvidia", "label": "NVIDIA", "aliases": [], "authenticated": True},
            {"id": "anthropic", "label": "Anthropic", "aliases": [], "authenticated": True},
        ],
    )

    text = await runner._handle_provider_command(event)

    assert "`openrouter`" in text
    assert "`nvidia`" in text
    assert "`anthropic`" not in text
    assert "Manual lock: `yes`" in text
    assert "Current provider:" in text
    assert "(`nvidia`)" in text


@pytest.mark.asyncio
async def test_model_command_text_overview_only_shows_chat_providers(monkeypatch):
    import agent.models_dev as models_dev
    import hermes_cli.models as cli_models

    runner = _make_runner()
    event = _make_event("/model")

    monkeypatch.setattr(
        cli_models,
        "list_available_providers",
        lambda: [
            {"id": "openrouter", "label": "OpenRouter", "aliases": [], "authenticated": True},
            {"id": "nvidia", "label": "NVIDIA", "aliases": [], "authenticated": False},
            {"id": "anthropic", "label": "Anthropic", "aliases": [], "authenticated": True},
        ],
    )
    monkeypatch.setattr(
        cli_models,
        "OPENROUTER_MODELS",
        [
            ("minimax/minimax-m2.5:free", "recommended"),
            ("google/gemma-3-27b-it:free", "free"),
        ],
    )
    monkeypatch.setattr(
        models_dev,
        "list_agentic_models",
        lambda provider: ["qwen/qwq-32b", "meta/llama-3.1-70b-instruct"] if provider == "nvidia" else [],
    )
    monkeypatch.setattr(models_dev, "list_provider_models", lambda provider: [])

    text = await runner._handle_model_command(event)

    assert "Current provider: OpenRouter (`openrouter`)" in text
    assert "**OpenRouter** `--provider openrouter`" in text
    assert "**Nvidia** `--provider nvidia`" in text
    assert "anthropic" not in text.lower()
    assert "Expanded: `/model list --provider openrouter`" in text
    assert "Expanded: `/model list --provider nvidia`" in text


@pytest.mark.asyncio
async def test_get_chat_provider_records_keeps_nvidia_when_env_key_present(monkeypatch):
    import hermes_cli.models as cli_models

    runner = _make_runner()

    monkeypatch.setattr(
        cli_models,
        "list_available_providers",
        lambda: [
            {"id": "openrouter", "label": "OpenRouter", "aliases": [], "authenticated": True},
        ],
    )
    monkeypatch.setattr(
        GatewayRunner,
        "_get_chat_curated_model_items",
        lambda self, provider: [("qwen/qwq-32b", "recommended")] if provider == "nvidia" else [],
    )
    monkeypatch.setattr(
        GatewayRunner,
        "_get_chat_expanded_model_ids",
        lambda self, provider: ["qwen/qwq-32b", "meta/llama-3.1-70b-instruct"] if provider == "nvidia" else [],
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    records = runner._get_chat_provider_records(
        session_key="agent:main:feishu:dm:oc_test",
        current_model="qwen/qwq-32b",
        current_provider="openrouter",
        authenticated_only=True,
        max_models=0,
    )

    assert any(record["slug"] == "nvidia" for record in records)
    nvidia = next(record for record in records if record["slug"] == "nvidia")
    assert nvidia["authenticated"] is True
    assert "qwen/qwq-32b" in nvidia["models"]


@pytest.mark.asyncio
async def test_model_provider_catalog_shows_curated_models(monkeypatch):
    import agent.models_dev as models_dev
    import hermes_cli.models as cli_models

    runner = _make_runner()
    event = _make_event("/model --provider openrouter")

    monkeypatch.setattr(
        cli_models,
        "OPENROUTER_MODELS",
        [
            ("minimax/minimax-m2.5:free", "recommended"),
            ("google/gemma-3-27b-it:free", "free"),
        ],
    )
    monkeypatch.setattr(models_dev, "list_agentic_models", lambda provider: [])
    monkeypatch.setattr(models_dev, "list_provider_models", lambda provider: [])

    text = await runner._handle_model_command(event)

    assert "**OpenRouter models** (curated, page 1/1)" in text
    assert "`minimax/minimax-m2.5:free` - recommended" in text
    assert "`google/gemma-3-27b-it:free` - free" in text
    assert "Expanded list: `/model list --provider openrouter`" in text


@pytest.mark.asyncio
async def test_model_list_and_more_paginate_expanded_catalog(monkeypatch):
    runner = _make_runner()
    expanded = [f"openrouter/model-{index}" for index in range(1, 11)]

    monkeypatch.setattr(
        GatewayRunner,
        "_get_chat_expanded_model_ids",
        lambda self, provider: expanded if provider == "openrouter" else [],
    )

    first_page = await runner._handle_model_command(_make_event("/model list --provider openrouter"))
    second_page = await runner._handle_model_command(_make_event("/model more --provider openrouter"))

    assert "page 1/2" in first_page
    assert "`openrouter/model-1`" in first_page
    assert "`openrouter/model-8`" in first_page
    assert "`openrouter/model-9`" not in first_page
    assert "More: `/model more --provider openrouter`" in first_page

    assert "page 2/2" in second_page
    assert "`openrouter/model-9`" in second_page
    assert "`openrouter/model-10`" in second_page
    assert "`openrouter/model-1`" not in second_page


@pytest.mark.asyncio
async def test_feishu_model_picker_open_id_target_updates_real_dm_session(monkeypatch):
    import hermes_cli.model_switch as model_switch

    runner = _make_runner()
    event = MessageEvent(
        text="/model",
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="ou_user",
            chat_type="dm",
            user_id="ou_user",
            user_name="Alice",
        ),
    )

    class _Adapter:
        async def send_model_picker(
            self,
            *,
            chat_id,
            providers,
            current_model,
            current_provider,
            session_key,
            on_model_selected,
            metadata,
        ):
            assert chat_id == "ou_user"
            assert metadata["receive_id_type"] == "open_id"
            await on_model_selected("oc_real_chat", "qwen/qwq-32b", "nvidia")
            return type("Result", (), {"success": True})()

    runner.adapters = {Platform.FEISHU: _Adapter()}
    monkeypatch.setattr(
        runner,
        "_get_chat_provider_records",
        lambda **_kwargs: [
            {"slug": "nvidia", "name": "NVIDIA", "models": ["qwen/qwq-32b"], "total_models": 1, "is_current": False}
        ],
    )
    monkeypatch.setattr(
        runner,
        "_load_model_runtime_config",
        lambda: ("openrouter/default-model", "openrouter", "https://openrouter.ai/api/v1", "", None),
    )
    monkeypatch.setattr(
        model_switch,
        "switch_model",
        lambda **_kwargs: SimpleNamespace(
            success=True,
            error_message="",
            new_model="qwen/qwq-32b",
            target_provider="nvidia",
            api_key="nvapi-test",
            base_url="https://integrate.api.nvidia.com/v1",
            api_mode="chat_completions",
            provider_label="NVIDIA",
            model_info=None,
        ),
    )

    text = await runner._handle_model_command(event)

    assert text is None
    actual_session_key = build_session_key(
        SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_real_chat",
            chat_type="dm",
            user_id="ou_user",
            user_name="Alice",
        )
    )
    assert actual_session_key in runner._session_model_overrides
    assert runner._session_model_overrides[actual_session_key]["model"] == "qwen/qwq-32b"
    assert runner._session_model_overrides[actual_session_key]["provider"] == "nvidia"
