import importlib.util
import sys
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
