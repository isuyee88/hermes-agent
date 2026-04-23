import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "modal_.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("modal_personality_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_personality_card_falls_back_to_builtin_personalities(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_load_personality_names", lambda: [])

    card = module._build_personality_card("agent:main:feishu:dm:test")
    labels = [
        action.get("text", {}).get("content")
        for element in card["elements"]
        if element.get("tag") == "action"
        for action in element.get("actions", [])
    ]

    assert "Neutral" in labels
    assert "CEO" in labels
    assert "CTO" in labels


def test_personality_command_accepts_builtin_fallback_personality(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_load_personality_names", lambda: [])

    result = module._handle_command("/personality cto", "agent:main:feishu:dm:test")

    assert result["status"] == "ok"
    assert result["session_state_after"]["current_personality"] == "cto"
    assert "CTO" in result["final_response"]


def test_session_state_persists_to_shared_store(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(module, "SESSION_STATE_PATH", tmp_path / "session_state.json")
    monkeypatch.setattr(module, "modal_volume", None)
    module.SESSION_STATE.clear()

    first = module._handle_command(
        "/model moonshotai/kimi-k2.5 --provider nvidia",
        "agent:main:feishu:dm:test-persist",
    )
    assert first["session_state_after"]["current_model"] == "moonshotai/kimi-k2.5"
    assert first["session_state_after"]["current_provider"] == "nvidia"

    module.SESSION_STATE.clear()
    restored = module._get_session_state("agent:main:feishu:dm:test-persist")

    assert restored["current_model"] == "moonshotai/kimi-k2.5"
    assert restored["current_provider"] == "nvidia"


def test_model_command_uses_registry_intro(monkeypatch):
    module = _load_module()
    monkeypatch.setitem(
        sys.modules,
        "tools.feishu_api",
        types.SimpleNamespace(
            load_feishu_model_registry=lambda force_refresh=False: {
                "status": "ok",
                "entries": [
                    {
                        "provider": "nvidia",
                        "model": "moonshotai/kimi-k2.5",
                        "introduction": "Long-context Chinese reasoning model for fast analysis.",
                    }
                ],
            }
        ),
    )

    result = module._handle_command("/model moonshotai/kimi-k2.5 --provider nvidia", "agent:main:feishu:dm:test")

    assert "Model switched to `moonshotai/kimi-k2.5`" in result["final_response"]
    assert "Introduction: Long-context Chinese reasoning model for fast analysis." in result["final_response"]


def test_model_hub_card_uses_registry_model_ids(monkeypatch):
    module = _load_module()
    monkeypatch.setitem(
        sys.modules,
        "tools.feishu_api",
        types.SimpleNamespace(
            load_feishu_model_registry=lambda force_refresh=False: {
                "status": "ok",
                "entries": [
                    {"provider": "openrouter", "model": "openai/gpt-5.4-mini", "selection_hint": "recommended"},
                    {"provider": "nvidia", "model": "moonshotai/kimi-k2.5"},
                ],
            }
        ),
    )

    card = module._build_model_hub_card()
    labels = [
        action.get("text", {}).get("content")
        for element in card["elements"]
        if element.get("tag") == "action"
        for action in element.get("actions", [])
    ]

    assert "openai/gpt-5.4-mini" in labels
    assert "moonshotai/kimi-k2.5" in labels


def test_generated_reply_result_uses_model_output(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_generate_session_reply",
        lambda message_text, session_key: {
            "text": f"分析结果: {message_text}",
            "provider": "openrouter",
            "model": "openai/gpt-5.4-mini",
            "ai_call_count": 1,
        },
    )

    result = module._build_generated_reply_result("请分析这个问题", "agent:main:feishu:dm:test", route_hint="modal_heavy_exec")

    assert result["final_response"] == "分析结果: 请分析这个问题"
    assert "我收到了你的消息" not in result["final_response"]
