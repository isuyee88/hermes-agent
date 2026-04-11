from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


def test_session_store_persists_route_metadata(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="user-1",
    )
    entry = store.get_or_create_session(source)

    route_lease = {
        "provider": "openrouter",
        "model": "qwen/qwen3-coder:free",
        "base_url": "https://openrouter.ai/api/v1",
        "lease_expires_at": 9999999999,
        "selection_reason": "fresh_select",
    }
    route_debug = {"last_route_selection": "sticky_hit"}
    route_metrics = {"sticky_hit": 2}

    store.update_session(
        entry.session_key,
        route_lease=route_lease,
        route_debug=route_debug,
        route_metrics=route_metrics,
    )

    reloaded = SessionStore(tmp_path / "sessions", GatewayConfig())
    reloaded.list_sessions()
    loaded_entry = reloaded._entries[entry.session_key]

    assert loaded_entry.route_lease == route_lease
    assert loaded_entry.route_debug == route_debug
    assert loaded_entry.route_metrics == route_metrics


def test_resolve_turn_agent_config_prefers_active_route_lease():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = SimpleNamespace(
        _entries={
            "telegram:123": SimpleNamespace(
                route_lease={
                    "provider": "openrouter",
                    "model": "google/gemma-3-27b-it:free",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_mode": "chat_completions",
                    "lease_expires_at": 9999999999,
                },
                route_metrics={},
            )
        }
    )

    turn = runner._resolve_turn_agent_config(
        "hello",
        "openrouter/free",
        {
            "api_key": "sk-or-test",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "_session_key": "telegram:123",
        },
    )

    assert turn["route_selection"] == "sticky_hit"
    assert turn["model"] == "google/gemma-3-27b-it:free"
    assert turn["runtime"]["provider"] == "openrouter"


def test_resolve_turn_agent_config_prefers_session_override_over_lease():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "telegram:123": {
            "model": "openrouter/custom-picked-model",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-test",
            "api_mode": "chat_completions",
        }
    }
    runner.session_store = SimpleNamespace(
        _entries={
            "telegram:123": SimpleNamespace(
                route_lease={
                    "provider": "openrouter",
                    "model": "stale/model",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_mode": "chat_completions",
                    "lease_expires_at": 9999999999,
                },
                route_metrics={},
            )
        }
    )

    turn = runner._resolve_turn_agent_config(
        "hello",
        "openrouter/free",
        {
            "api_key": "sk-or-test",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "_session_key": "telegram:123",
        },
    )

    assert turn["route_selection"] == "explicit_override"
    assert turn["model"] == "openrouter/custom-picked-model"
    assert turn["runtime"]["base_url"] == "https://openrouter.ai/api/v1"


def test_get_session_entry_eagerly_loads_store(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_test",
        chat_type="dm",
        user_id="user-1",
    )
    entry = store.get_or_create_session(source)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = SessionStore(tmp_path / "sessions", GatewayConfig())

    loaded_entry = runner._get_session_entry(entry.session_key)

    assert loaded_entry is not None
    assert loaded_entry.session_key == entry.session_key
