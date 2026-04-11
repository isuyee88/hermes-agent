import importlib.util
import asyncio
import json
import os
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "modal_.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("modal_deployment_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


async def _async_return(value):
    return value


def test_validate_bearer_token():
    module = _load_module()
    assert module._validate_bearer_token("Bearer token-123", "token-123") is True
    assert module._validate_bearer_token("Bearer wrong", "token-123") is False
    assert module._validate_bearer_token(None, "token-123") is False
    assert module._validate_bearer_token(None, None) is True


def test_extract_tool_names_handles_dict_payloads():
    module = _load_module()
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "web_search"}},
                {"function": {"name": "read_file"}},
            ],
        }
    ]
    assert module._extract_tool_names(messages) == ["web_search", "read_file"]


def test_session_state_round_trip(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    messages = [{"role": "user", "content": "hello"}]
    module._save_session_state("telegram:42", "session-1", messages)
    loaded = module._load_session_state("telegram:42")
    assert loaded["session_id"] == "session-1"
    assert loaded["messages"] == messages


def test_session_state_persists_route_lease(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    route_lease = module._build_route_lease(
        {
            "provider": "openrouter",
            "model": "qwen/qwen3-coder:free",
            "base_url": "https://openrouter.ai/api/v1",
        },
        selection_reason="fresh_select",
        selected_at=123,
        last_success_at=456,
    )
    module._save_session_state(
        "telegram:42",
        "session-1",
        [{"role": "user", "content": "hello"}],
        route_lease=route_lease,
        route_debug={"last_route_selection": "fresh_select"},
        route_metrics={"fresh_select": 1},
    )

    loaded = module._load_session_state("telegram:42")
    assert loaded["route_lease"]["model"] == "qwen/qwen3-coder:free"
    assert loaded["route_debug"]["last_route_selection"] == "fresh_select"
    assert loaded["route_metrics"]["fresh_select"] == 1


def test_mark_update_seen_deduplicates(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    assert module._mark_update_seen("1001") is True
    assert module._mark_update_seen("1001") is False

    payload = json.loads(module.UPDATES_PATH.read_text(encoding="utf-8"))
    assert "1001" in payload


def test_cron_queue_claim_round_trip(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CRON_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "cron_queue_claims.json"
    module._ensure_runtime_dirs()

    job = {"id": "cron-1", "name": "Reminder", "next_run_at": "2026-04-10T10:00:00+00:00"}
    claimed, token = module._claim_due_cron_job(job, ttl_seconds=3600)
    assert claimed is True
    assert token == "cron-1:2026-04-10T10:00:00+00:00"

    claimed_again, same_token = module._claim_due_cron_job(job, ttl_seconds=3600)
    assert claimed_again is False
    assert same_token == token

    module._release_cron_job_claim("cron-1", claim_token=token)

    claimed_after_release, _ = module._claim_due_cron_job(job, ttl_seconds=3600)
    assert claimed_after_release is True


def test_chat_partition_claim_round_trip(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module._ensure_runtime_dirs()

    claimed, token = module._claim_chat_partition("feishu:oc_chat", platform="feishu", ttl_seconds=3600)
    assert claimed is True
    assert token.startswith("feishu:feishu:oc_chat:")

    claimed_again, existing = module._claim_chat_partition("feishu:oc_chat", platform="feishu", ttl_seconds=3600)
    assert claimed_again is False
    assert existing == token

    module._release_chat_partition_claim("feishu:oc_chat", claim_token=token)

    claimed_after_release, _ = module._claim_chat_partition("feishu:oc_chat", platform="feishu", ttl_seconds=3600)
    assert claimed_after_release is True


def test_process_chat_queue_commits_before_releasing_claim(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module._ensure_runtime_dirs()

    events = []

    class FakeQueue:
        def __init__(self):
            self._items = [{"payload": "one"}]

        def get_many(self, _max_items, block=False, partition=None):
            if self._items:
                items, self._items = self._items, []
                return items
            return []

    monkeypatch.setattr(module, "_get_chat_queue", lambda: FakeQueue())
    monkeypatch.setattr(module, "_claim_chat_partition", lambda partition, platform: (True, "claim-1"))
    monkeypatch.setattr(module, "_process_chat_queue_item", lambda item: {"status": "ok", "item": item})
    monkeypatch.setattr(module, "_safe_chat_queue_depth", lambda: 0)

    def _record_sync(*, reload=False, commit=False):
        if reload:
            events.append("reload")
        if commit:
            events.append("commit")

    def _record_release(partition, *, claim_token=None):
        events.append(f"release:{partition}:{claim_token}")

    monkeypatch.setattr(module, "_sync_modal_volume", _record_sync)
    monkeypatch.setattr(module, "_release_chat_partition_claim", _record_release)

    result = module._process_chat_queue_impl(platform="feishu", partition="feishu:oc_chat", max_items=1)

    assert result["processed_count"] == 1
    assert events[-2:] == ["commit", "release:feishu:oc_chat:claim-1"]


def test_sync_runtime_config_writes_official_config_path(tmp_path, monkeypatch):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"

    source = tmp_path / "config.modal.yaml"
    source.write_text("terminal:\n  backend: modal\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MODAL_CONFIG_SOURCE", str(source))
    monkeypatch.setenv("HERMES_MODAL_SYNC_CONFIG", "true")

    module._ensure_runtime_dirs()
    written_path = module._sync_runtime_config()

    assert written_path == str(module.HERMES_HOME_DIR / "config.yaml")
    assert (module.HERMES_HOME_DIR / "config.yaml").read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_sync_runtime_config_expands_model_env_placeholder(tmp_path, monkeypatch):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"

    source = tmp_path / "config.modal.yaml"
    source.write_text("model:\n  default: ${DEFAULT_MODEL}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MODAL_CONFIG_SOURCE", str(source))
    monkeypatch.setenv("HERMES_MODAL_SYNC_CONFIG", "true")
    monkeypatch.setenv("DEFAULT_MODEL", "openrouter/free")

    module._ensure_runtime_dirs()
    module._sync_runtime_config()

    target_text = (module.HERMES_HOME_DIR / "config.yaml").read_text(encoding="utf-8")
    assert "${DEFAULT_MODEL}" not in target_text
    assert "openrouter/free" in target_text


def test_memory_provider_status_reports_supermemory(monkeypatch, tmp_path):
    module = _load_module()
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "memory:\n  provider: supermemory\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm-test-key")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object() if name == "supermemory" else None)

    status = module._get_memory_provider_status()

    assert status["provider"] == "supermemory"
    assert status["configured"] is True
    assert status["api_key_configured"] is True
    assert status["sdk_available"] is True


def test_healthz_syncs_runtime_config_before_reporting_memory_provider(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"

    source = tmp_path / "config.modal.yaml"
    source.write_text("memory:\n  provider: supermemory\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(module.HERMES_HOME_DIR))
    monkeypatch.setenv("HERMES_MODAL_CONFIG_SOURCE", str(source))
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm-test-key")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object() if name == "supermemory" else None)

    client = TestClient(module.create_web_app())
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_provider"]["provider"] == "supermemory"
    assert payload["memory_provider"]["configured"] is True
    assert payload["memory_provider"]["api_key_configured"] is True
    assert payload["memory_provider"]["config_present"] is True
    assert payload["runtime_config"] == str(module.HERMES_HOME_DIR / "config.yaml")


def test_prepare_runtime_environment_syncs_supermemory_config(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"

    source = tmp_path / "supermemory.modal.json"
    source.write_text('{"profile_frequency": 3}\n', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(module.HERMES_HOME_DIR))
    monkeypatch.setenv("HERMES_MODAL_SUPERMEMORY_CONFIG_SOURCE", str(source))

    module._prepare_runtime_environment()

    target = module.HERMES_HOME_DIR / "supermemory.json"
    assert target.exists() is True
    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_runtime_settings_capture_qq_credentials(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("QQ_APP_ID", "app-12345678")
    monkeypatch.setenv("QQ_APP_SECRET", "secret-456789")

    settings = module.RuntimeSettings.from_env()
    serialized = module._serialize_settings_for_log(settings)

    assert settings.qq_app_id == "app-12345678"
    assert settings.qq_app_secret == "secret-456789"
    assert serialized["qq_app_id"].startswith("app-")
    assert serialized["qq_app_secret"].startswith("secr")


def test_runtime_settings_capture_feishu_credentials(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    monkeypatch.setenv("FEISHU_BITABLE_APP_TOKEN", "bitable-app-token")
    monkeypatch.setenv("FEISHU_BITABLE_TABLE_ID", "tbl_model_registry")
    monkeypatch.setenv("FEISHU_MODEL_REGISTRY_MIRROR_ENABLED", "true")
    monkeypatch.setenv("HERMES_FEISHU_TOOL_CAPABILITIES", "docs,files,model_registry")
    monkeypatch.setenv("HERMES_FEISHU_DEFAULT_WORKSPACE", "growth")

    settings = module.RuntimeSettings.from_env()
    serialized = module._serialize_settings_for_log(settings)

    assert settings.feishu_app_id == "cli_feishu_app"
    assert settings.feishu_app_secret == "feishu-secret-123"
    assert settings.feishu_bitable_app_token == "bitable-app-token"
    assert settings.feishu_bitable_table_id == "tbl_model_registry"
    assert settings.feishu_model_registry_mirror_enabled is True
    assert settings.feishu_tool_capabilities == ["docs", "files", "model_registry"]
    assert settings.feishu_default_workspace == "growth"
    assert serialized["feishu_app_id"].startswith("cli_")
    assert serialized["feishu_app_secret"].startswith("feis")
    assert serialized["feishu_bitable_app_token"].startswith("bita")


def test_extract_feishu_queue_context_handles_menu_event():
    module = _load_module()
    payload = {
        "header": {
            "event_id": "evt-menu-1",
            "event_type": "application.bot.menu_v6",
        },
        "event": {
            "operator": {"operator_id": {"open_id": "ou_menu_operator"}},
            "context": {"open_chat_id": "oc_menu_chat"},
        },
    }

    result = module._extract_feishu_queue_context(payload)

    assert result["platform"] == "feishu"
    assert result["partition"] == "feishu:oc_menu_chat"
    assert result["chat_id"] == "oc_menu_chat"
    assert result["actor_id"] == "ou_menu_operator"
    assert result["event_type"] == "application.bot.menu_v6"


def test_process_chat_queue_item_skips_reload_by_default(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()
    monkeypatch.delenv("HERMES_MODAL_CHAT_WORKER_RELOAD", raising=False)

    sync_calls = []

    def _record_sync(*, reload=False, commit=False):
        sync_calls.append((reload, commit))

    monkeypatch.setattr(module, "_prepare_runtime_environment", lambda: None)
    monkeypatch.setattr(module, "_sync_modal_volume", _record_sync)

    async def _fake_dispatch(_payload):
        return {"status": "ok"}

    monkeypatch.setattr(module, "_dispatch_telegram_update", _fake_dispatch)

    result = module._process_chat_queue_item(
        {
            "platform": "telegram",
            "partition": "telegram:42",
            "payload": {"update_id": 1, "message": {"chat": {"id": 42}, "from": {"id": 7}}},
            "metadata": {},
        }
    )

    assert result["status"] == "ok"
    assert sync_calls == []


def test_chat_partition_claim_skips_reload_by_default(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module._ensure_runtime_dirs()
    monkeypatch.delenv("HERMES_MODAL_CHAT_CLAIMS_RELOAD", raising=False)

    sync_calls = []

    def _record_sync(*, reload=False, commit=False):
        sync_calls.append((reload, commit))

    monkeypatch.setattr(module, "_sync_modal_volume", _record_sync)

    claimed, token = module._claim_chat_partition("feishu:oc_chat", platform="feishu", ttl_seconds=3600)

    assert claimed is True
    assert token.startswith("feishu:feishu:oc_chat:")
    assert sync_calls == [(False, True)]


def test_cron_claim_skips_reload_by_default(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CRON_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "cron_queue_claims.json"
    module._ensure_runtime_dirs()
    monkeypatch.delenv("HERMES_MODAL_CRON_CLAIMS_RELOAD", raising=False)

    sync_calls = []

    def _record_sync(*, reload=False, commit=False):
        sync_calls.append((reload, commit))

    monkeypatch.setattr(module, "_sync_modal_volume", _record_sync)

    claimed, token = module._claim_due_cron_job(
        {"id": "cron-1", "name": "Reminder", "next_run_at": "2026-04-10T10:00:00+00:00"},
        ttl_seconds=3600,
    )

    assert claimed is True
    assert token == "cron-1:2026-04-10T10:00:00+00:00"
    assert sync_calls == [(False, True)]


def test_debug_feishu_capabilities_and_registry_helpers(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_prepare_runtime_environment", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "tools.feishu_api",
        types.SimpleNamespace(
            get_feishu_capability_snapshot=lambda probe=False: {"configured": True, "probe": probe},
            load_feishu_model_registry=lambda force_refresh=False: {
                "status": "ok",
                "entries": [{"provider": "openrouter", "model": "demo"}],
                "generated_at": 123,
            },
        ),
    )

    capabilities = module._build_feishu_capabilities_debug_state(probe=True)
    registry = module._build_feishu_model_registry_debug_state(force_refresh=True)

    assert capabilities == {"configured": True, "probe": True}
    assert registry["status"] == "ok"
    assert registry["entry_count"] == 1


def test_health_check_reports_feishu_configured(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    client = TestClient(module.create_web_app())
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["feishu_configured"] is True


def test_feishu_webhook_deduplicates_event_id(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_EVENTS_PATH = module.DATA_ROOT / "feishu_events.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    assert module._mark_feishu_event_seen("evt-1") is True
    assert module._mark_feishu_event_seen("evt-1") is False


def test_build_feishu_menu_manifest_contains_model_picker():
    module = _load_module()
    payload = module._build_feishu_menu_manifest()

    assert payload["platform"] == "feishu"
    assert "model_picker" in payload["supported_event_keys"]
    assert payload["menu_items"][0]["event_key"] == "model_picker"


def test_runtime_api_config_infers_openrouter_defaults(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HERMES_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    provider, base_url, api_key = module._pick_runtime_api_config()

    assert provider == "openrouter"
    assert base_url == "https://openrouter.ai/api/v1"
    assert api_key == "sk-or-test"


def test_desired_telegram_webhook_url_prefers_explicit_env(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.com/custom/hook")
    monkeypatch.setenv("HERMES_PUBLIC_BASE_URL", "https://ignored.example.com")

    assert module._desired_telegram_webhook_url() == "https://example.com/custom/hook"


def test_desired_telegram_webhook_url_derives_from_public_base(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("HERMES_PUBLIC_BASE_URL", "https://example.com/")

    assert module._desired_telegram_webhook_url() == "https://example.com/telegram/webhook"


def test_telegram_bot_token_format_validator():
    module = _load_module()
    assert module._is_valid_telegram_bot_token_format("123456789:ABCdef_ghi-JKLmnopQRSTUvwxYZ")
    assert module._is_valid_telegram_bot_token_format("AAE1kxRtD3uvpjI1Y_Xkg-zrv96fDLeuN_4") is False


def test_prepare_runtime_environment_defaults_terminal_to_local(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("HERMES_MODAL_TERMINAL_ENV", raising=False)

    module._prepare_runtime_environment()

    assert os.environ["TERMINAL_ENV"] == "local"


def test_prepare_runtime_environment_enables_project_plugins_by_default(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

    module._prepare_runtime_environment()

    assert os.environ["HERMES_ENABLE_PROJECT_PLUGINS"] == "true"


def test_prepare_runtime_environment_overrides_stale_modal_terminal_env(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.delenv("HERMES_MODAL_TERMINAL_ENV", raising=False)

    module._prepare_runtime_environment()

    assert os.environ["TERMINAL_ENV"] == "local"


def test_runtime_model_name_normalizes_openrouter_anthropic_snapshot():
    module = _load_module()
    assert (
        module._resolve_runtime_model_name(
            "anthropic/claude-sonnet-4-20250514",
            "openrouter",
        )
        == "anthropic/claude-sonnet-4"
    )
    assert (
        module._resolve_runtime_model_name(
            "anthropic/claude-sonnet-4-20250514",
            "anthropic",
        )
        == "anthropic/claude-sonnet-4-20250514"
    )


def test_invalid_model_error_triggers_refresh_retry(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("DEFAULT_MODEL", "broken-model")

    refresh_calls = []

    def _fake_refresh_and_select_valid_route(*, preferred_provider=None, failed_route=None):
        refresh_calls.append((preferred_provider, failed_route))
        return {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-test",
            "model": "openrouter/free",
        }

    monkeypatch.setattr(module, "_refresh_and_select_valid_route", _fake_refresh_and_select_valid_route)

    init_models = []

    class FakeAgent:
        def __init__(self, **kwargs):
            init_models.append(kwargs["model"])
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *_args, **_kwargs):
            if len(init_models) == 1:
                return {
                    "error": "broken-model is not a valid model ID",
                    "messages": [],
                    "completed": False,
                    "interrupted": False,
                    "api_calls": 1,
                }
            return {
                "final_response": "PONG",
                "messages": [{"role": "assistant", "content": "PONG"}],
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))

    result = module._run_agent_task_impl("Reply with exactly PONG.", session_key="retry-test")

    assert result["status"] == "success"
    assert result["output"] == "PONG"
    assert result["retried_after_model_refresh"] is True
    assert result["refreshed_model"] == "openrouter/free"
    assert init_models == ["broken-model", "openrouter/free"]
    assert refresh_calls and refresh_calls[0][1]["model"] == "broken-model"


def test_session_route_lease_sticky_hit(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("DEFAULT_MODEL", "openrouter/free")
    route_lease = module._build_route_lease(
        {
            "provider": "openrouter",
            "model": "google/gemma-3-27b-it:free",
            "base_url": "https://openrouter.ai/api/v1",
        },
        selection_reason="fresh_select",
    )
    module._save_session_state(
        "sticky-session",
        "session-sticky",
        [{"role": "user", "content": "hi"}],
        route_lease=route_lease,
    )

    def _fail_if_resolved(*_args, **_kwargs):
        raise AssertionError("sticky route should bypass fresh route selection")

    monkeypatch.setattr(module, "_resolve_primary_route", _fail_if_resolved)

    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            seen["model"] = kwargs["model"]
            seen["trace_session_key"] = kwargs["trace_session_key"]
            seen["trace_metadata"] = kwargs["trace_metadata"]
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *_args, **_kwargs):
            return {
                "final_response": "OK",
                "messages": [{"role": "assistant", "content": "OK"}],
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
            }

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))

    result = module._run_agent_task_impl("Reply with OK", session_key="sticky-session")

    assert result["route_selection"] == "sticky_hit"
    assert seen["model"] == "google/gemma-3-27b-it:free"
    assert seen["trace_session_key"] == "sticky-session"
    assert seen["trace_metadata"]["route_selection"] == "sticky_hit"


def test_explicit_model_override_updates_route_lease(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    module._save_session_state(
        "override-session",
        "session-override",
        [],
        route_lease=module._build_route_lease(
            {
                "provider": "openrouter",
                "model": "old/model",
                "base_url": "https://openrouter.ai/api/v1",
            },
            selection_reason="fresh_select",
        ),
    )

    monkeypatch.setattr(
        module,
        "_resolve_primary_route",
        lambda _settings, model_name=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-test",
            "model": model_name or "fallback-model",
        },
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *_args, **_kwargs):
            return {
                "final_response": "OVERRIDE_OK",
                "messages": [{"role": "assistant", "content": "OVERRIDE_OK"}],
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
            }

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))

    result = module._run_agent_task_impl(
        "Reply with OVERRIDE_OK",
        session_key="override-session",
        model_name="openrouter/custom-new-model",
    )

    assert result["route_selection"] == "explicit_override"
    loaded = module._load_session_state("override-session")
    assert loaded["route_lease"]["model"] == "openrouter/custom-new-model"
    assert loaded["route_lease"]["selection_reason"] == "explicit_override"


def test_sync_runtime_config_materializes_dynamic_free_route(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"

    source = tmp_path / "config.modal.yaml"
    source.write_text("model:\n  default: openrouter/free\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MODAL_CONFIG_SOURCE", str(source))
    monkeypatch.setenv("HERMES_MODAL_SYNC_CONFIG", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    monkeypatch.setattr(
        module,
        "_refresh_free_model_routes",
        lambda force=False: {
            "refreshed_at": 123,
            "providers": {
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "candidates": ["moonshotai/kimi-k2:free"],
                },
                "nvidia": {
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "candidates": ["meta/llama-3.1-8b-instruct"],
                },
            },
        },
    )

    module._ensure_runtime_dirs()
    module._sync_runtime_config()

    target_text = (module.HERMES_HOME_DIR / "config.yaml").read_text(encoding="utf-8")
    assert "default: moonshotai/kimi-k2:free" in target_text
    assert "provider: openrouter" in target_text
    assert "fallback_providers:" in target_text
    assert "provider: custom" in target_text
    assert "model: meta/llama-3.1-8b-instruct" in target_text


def test_transient_error_triggers_retry_route_refresh(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("DEFAULT_MODEL", "openrouter/free")

    refresh_calls = []

    def _fake_refresh_and_select_valid_route(*, preferred_provider=None, failed_route=None):
        refresh_calls.append((preferred_provider, failed_route))
        return {
            "provider": "nvidia",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": "nvapi-test",
            "model": "meta/llama-3.1-8b-instruct",
        }

    monkeypatch.setattr(module, "_refresh_and_select_valid_route", _fake_refresh_and_select_valid_route)
    monkeypatch.setattr(
        module,
        "_select_dynamic_primary_route",
        lambda force_refresh=False: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-test",
            "model": "openrouter/free",
        },
    )

    init_models = []

    class FakeAgent:
        def __init__(self, **kwargs):
            init_models.append((kwargs["provider"], kwargs["model"]))
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *_args, **_kwargs):
            if len(init_models) == 1:
                return {
                    "error": "ReadTimeout while waiting for provider response",
                    "messages": [],
                    "completed": False,
                    "interrupted": True,
                    "api_calls": 1,
                }
            return {
                "final_response": "FAST_OK",
                "messages": [{"role": "assistant", "content": "FAST_OK"}],
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))

    result = module._run_agent_task_impl("只回复 FAST_OK", session_key="transient-retry-test")

    assert result["status"] == "success"
    assert result["output"] == "FAST_OK"
    assert result["retried_after_model_refresh"] is True
    assert result["refreshed_provider"] == "nvidia"
    assert init_models == [
        ("openrouter", "openrouter/free"),
        ("nvidia", "meta/llama-3.1-8b-instruct"),
    ]
    assert refresh_calls and refresh_calls[0][0] == "nvidia"


def test_model_routing_debug_state_includes_session_route_summaries(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    module._save_session_state(
        "debug-session",
        "session-debug",
        [],
        route_lease=module._build_route_lease(
            {
                "provider": "openrouter",
                "model": "qwen/qwen3-coder:free",
                "base_url": "https://openrouter.ai/api/v1",
            },
            selection_reason="fresh_select",
        ),
        route_debug={"last_route_selection": "sticky_hit"},
        route_metrics={"sticky_hit": 2, "fresh_select": 1},
    )

    payload = module._build_model_routing_debug_state(force_refresh=False, allow_network=False)

    assert payload["session_route_metrics"]["sessions_with_route_lease"] >= 1
    assert payload["recent_session_routes"][0]["session_key"] == "debug-session"
    assert payload["recent_session_routes"][0]["route_selection"] == "sticky_hit"


def test_debug_session_route_state_returns_ttl(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    module._save_session_state(
        "route-state-session",
        "session-route-state",
        [],
        route_lease=module._build_route_lease(
            {
                "provider": "openrouter",
                "model": "openrouter/free",
                "base_url": "https://openrouter.ai/api/v1",
            },
            selection_reason="fresh_select",
        ),
    )

    payload = module._debug_session_route_state("route-state-session")

    assert payload["status"] == "ok"
    assert payload["lease_active"] is True
    assert payload["lease_ttl_remaining_seconds"] > 0


def test_run_agent_task_reports_supermemory_tool_usage(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("DEFAULT_MODEL", "openrouter/free")

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *_args, **_kwargs):
            return {
                "final_response": "Supermemory available",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "supermemory_profile"}},
                        ],
                    },
                    {"role": "assistant", "content": "Supermemory available"},
                ],
                "completed": True,
                "interrupted": False,
                "api_calls": 2,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))

    result = module._run_agent_task_impl(
        "Use the supermemory_profile tool, then answer briefly.",
        session_key="supermemory-tool-test",
    )

    assert result["status"] == "success"
    assert result["output"] == "Supermemory available"
    assert result["tool_summary"] == ["supermemory_profile"]


def test_run_agent_task_includes_provider_usage_metadata(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("DEFAULT_MODEL", "openrouter/free")

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *_args, **_kwargs):
            return {
                "final_response": "usage ready",
                "messages": [{"role": "assistant", "content": "usage ready"}],
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "provider_usage": {
                    "generation_id": "gen_123",
                    "cost": 0.0012,
                    "upstream_inference_cost": 0.0009,
                    "cache_discount": 0.0003,
                },
                "provider_usage_totals": {
                    "billed_cost_usd": 0.0012,
                    "upstream_inference_cost_usd": 0.0009,
                    "cache_discount_usd": 0.0003,
                },
            }

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))

    result = module._run_agent_task_impl("Report provider usage", session_key="usage-meta-test")

    assert result["provider_usage"]["generation_id"] == "gen_123"
    assert result["provider_usage_totals"]["billed_cost_usd"] == 0.0012


def test_run_agent_task_pins_response_model_into_route_lease(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("DEFAULT_MODEL", "openrouter/free")

    seen_models = []

    class FakeAgent:
        def __init__(self, **kwargs):
            seen_models.append(kwargs["model"])
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *_args, **_kwargs):
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "provider_usage": {
                    "response_model": "google/gemma-3-27b-it:free",
                },
            }

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))

    first = module._run_agent_task_impl("Reply ok", session_key="pin-response-model")
    second = module._run_agent_task_impl("Reply ok again", session_key="pin-response-model")

    assert first["route_selection"] == "fresh_select"
    assert second["route_selection"] == "sticky_hit"
    assert seen_models == ["openrouter/free", "google/gemma-3-27b-it:free"]
    assert module._load_session_state("pin-response-model")["route_lease"]["model"] == "google/gemma-3-27b-it:free"


def test_debug_gateway_session_state_loads_persisted_entries(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(module.HERMES_HOME_DIR))
    module._ensure_runtime_dirs()

    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore

    store = SessionStore(module.HERMES_HOME_DIR / "sessions", GatewayConfig())
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_test",
        chat_type="dm",
        user_id="user-1",
    )
    entry = store.get_or_create_session(source)

    result = module._debug_gateway_session_state(entry.session_key)

    assert result["session_count"] >= 1
    assert result["entry"]["session_key"] == entry.session_key


def test_maybe_sync_telegram_webhook_skips_redundant_set(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.TELEGRAM_WEBHOOK_SYNC_STATE_PATH = module.DATA_ROOT / "telegram_webhook_sync.json"
    module._ensure_runtime_dirs()

    settings = module.RuntimeSettings(
        model="openrouter/free",
        max_turns=16,
        max_tokens=None,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        bearer_token=None,
        telegram_bot_token="123456789:ABCdef_ghi-JKLmnopQRSTUvwxYZ",
        telegram_webhook_secret="secret",
        telegram_webhook_url="https://example.com/telegram/webhook",
        telegram_send_ack=False,
        feishu_app_id=None,
        feishu_app_secret=None,
        feishu_domain="feishu",
        feishu_connection_mode="webhook",
        feishu_verification_token=None,
        feishu_encrypt_key=None,
        feishu_bitable_app_token=None,
        feishu_bitable_table_id=None,
        feishu_model_registry_mirror_enabled=False,
        feishu_tool_capabilities=None,
        feishu_default_workspace=None,
        qq_app_id=None,
        qq_app_secret=None,
        nvidia_api_key=None,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        enabled_toolsets=[],
        disabled_toolsets=[],
    )

    async def _fake_status(_settings, *, ensure_registered=False, drop_pending_updates=False):
        assert ensure_registered is False
        return {
            "configured": True,
            "registered_url": "https://example.com/telegram/webhook",
            "expected_url": "https://example.com/telegram/webhook",
            "matches_expected": True,
        }

    set_calls = []

    async def _fake_set(*_args, **_kwargs):
        set_calls.append(True)
        return {"ok": True}

    monkeypatch.setattr(module, "_get_telegram_webhook_status", _fake_status)
    monkeypatch.setattr(module, "_set_telegram_webhook", _fake_set)

    result = asyncio.run(module._maybe_sync_telegram_webhook(settings))

    assert result["matches_expected"] is True
    assert set_calls == []


def test_maybe_sync_telegram_webhook_respects_429_backoff(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.TELEGRAM_WEBHOOK_SYNC_STATE_PATH = module.DATA_ROOT / "telegram_webhook_sync.json"
    module._ensure_runtime_dirs()

    settings = module.RuntimeSettings(
        model="openrouter/free",
        max_turns=16,
        max_tokens=None,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        bearer_token=None,
        telegram_bot_token="123456789:ABCdef_ghi-JKLmnopQRSTUvwxYZ",
        telegram_webhook_secret="secret",
        telegram_webhook_url="https://example.com/telegram/webhook",
        telegram_send_ack=False,
        feishu_app_id=None,
        feishu_app_secret=None,
        feishu_domain="feishu",
        feishu_connection_mode="webhook",
        feishu_verification_token=None,
        feishu_encrypt_key=None,
        feishu_bitable_app_token=None,
        feishu_bitable_table_id=None,
        feishu_model_registry_mirror_enabled=False,
        feishu_tool_capabilities=None,
        feishu_default_workspace=None,
        qq_app_id=None,
        qq_app_secret=None,
        nvidia_api_key=None,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        enabled_toolsets=[],
        disabled_toolsets=[],
    )

    async def _fake_status(_settings, *, ensure_registered=False, drop_pending_updates=False):
        return {
            "configured": True,
            "registered_url": "",
            "expected_url": "https://example.com/telegram/webhook",
            "matches_expected": False,
        }

    response = types.SimpleNamespace(
        headers={},
        json=lambda: {"parameters": {"retry_after": 42}},
    )
    exc = Exception(
        "Client error '429 Too Many Requests' for url 'https://api.telegram.org/bot123/setWebhook'"
    )
    setattr(exc, "response", response)

    async def _fake_set(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(module, "_get_telegram_webhook_status", _fake_status)
    monkeypatch.setattr(module, "_set_telegram_webhook", _fake_set)

    result = asyncio.run(module._maybe_sync_telegram_webhook(settings))

    assert result["sync_rate_limited"] is True
    assert result["retry_after_seconds"] == 42
    saved_state = module._load_json_file(module.TELEGRAM_WEBHOOK_SYNC_STATE_PATH, {})
    assert int(saved_state["next_retry_at"]) >= int(saved_state["last_attempt_at"]) + 42


def test_validate_tavily_integration_impl(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    fake_tools_package = types.ModuleType("tools")
    fake_tools_package.__path__ = []  # mark as package-like for nested imports used during runtime prep

    fake_web_tools = types.ModuleType("tools.web_tools")
    fake_web_tools._get_backend = lambda: "tavily"
    fake_web_tools._is_backend_available = lambda backend: backend == "tavily"
    fake_web_tools.web_search_tool = lambda _query, limit=3: json.dumps(
        {
            "success": True,
            "data": {
                "web": [
                    {"title": "Tavily", "url": "https://tavily.com/", "description": "Official"},
                ][:limit]
            },
        }
    )
    fake_web_tools.web_extract_tool = lambda _urls, use_llm_processing=False: _async_return(
        json.dumps(
            {
                "results": [
                    {"url": "https://tavily.com/", "title": "Tavily", "content": "Official site"},
                ]
            }
        )
    )
    fake_web_tools.web_crawl_tool = lambda _url, _instructions, use_llm_processing=False: _async_return(
        json.dumps(
            {
                "results": [
                    {"url": "https://tavily.com/", "title": "Tavily", "content": "Crawled content"},
                ]
            }
        )
    )
    fake_tools_package.web_tools = fake_web_tools
    monkeypatch.setitem(sys.modules, "tools", fake_tools_package)
    monkeypatch.setitem(sys.modules, "tools.web_tools", fake_web_tools)

    result = module._validate_tavily_integration_impl()

    assert result["integration"] == "tavily"
    assert result["backend"] == "tavily"
    assert result["backend_available"] is True
    assert result["env_configured"] is True
    assert result["search"]["success"] is True
    assert result["search"]["top_result"]["url"] == "https://tavily.com/"
    assert result["extract"]["success"] is True
    assert result["crawl"]["success"] is True


def test_qq_webhook_route_preserves_official_ack_shape(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("QQ_APP_ID", "app-12345678")
    monkeypatch.setenv("QQ_APP_SECRET", "secret-456789")

    async def _fake_dispatch(payload, *, headers=None):
        assert payload == {"op": 0}
        assert headers == {"X-Bot-Appid": "app-12345678"}
        return {"op": 12}

    monkeypatch.setattr(module, "_dispatch_qq_update", _fake_dispatch)

    client = TestClient(module.create_web_app())
    response = client.post(
        "/qq/webhook",
        json={"op": 0},
        headers={"x-bot-appid": "app-12345678"},
    )

    assert response.status_code == 200
    assert response.json() == {"op": 12}


def test_feishu_webhook_route_preserves_json_response(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")

    async def _fake_dispatch(_request):
        return module.Response(
            content=b'{"challenge":"ok"}',
            status_code=200,
            headers={"content-type": "application/json; charset=utf-8"},
        )

    monkeypatch.setattr(module, "_dispatch_feishu_update", _fake_dispatch)

    client = TestClient(module.create_web_app())
    response = client.post(
        "/feishu/webhook",
        json={"type": "url_verification", "challenge": "ok"},
    )

    assert response.status_code == 200
    assert response.json() == {"challenge": "ok"}


def test_dispatch_feishu_update_processes_message_synchronously(monkeypatch):
    module = _load_module()
    handled = {"spawned": None, "enqueued": None}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": "evt_sync",
            "token": "verify-token",
        },
        "event": {"message": {"message_id": "om_sync"}},
    }

    def _fake_enqueue(*, platform, partition, payload, metadata):
        handled["enqueued"] = {
            "platform": platform,
            "partition": partition,
            "payload": payload,
            "metadata": metadata,
        }
        return {"status": "enqueued", "queue_depth": 1}

    async def _fake_spawn_aio(**kwargs):
        handled["spawned"] = kwargs

    monkeypatch.setattr(module, "_enqueue_chat_event", _fake_enqueue)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)
    monkeypatch.setattr(module, "_safe_chat_queue_depth", lambda: 1)

    async def _fake_parse(_request):
        return object(), payload

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)

    monkeypatch.setattr(
        module,
        "process_chat_queue",
        types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_fake_spawn_aio)),
        raising=False,
    )

    client = TestClient(module.create_web_app())
    response = client.post(
        "/feishu/webhook",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json() == {"code": 0, "msg": "accepted"}
    assert handled["enqueued"]["payload"]["event"]["message"]["message_id"] == "om_sync"
    assert handled["spawned"]["platform"] == "feishu"
    assert handled["spawned"]["partition"] == handled["enqueued"]["partition"]


def test_validate_feishu_webhook_impl_builds_signed_encrypted_request(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", "verify-token")
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", "encrypt-key")
    monkeypatch.setenv("HERMES_PUBLIC_BASE_URL", "https://example.com")

    captured = {}

    class FakeResponse:
        status_code = 200
        text = '{"challenge":"feishu-encrypted-selftest-ok"}'

    class FakeClient:
        def __init__(self, timeout):
            assert timeout == 20

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *, content, headers):
            captured["url"] = url
            captured["body"] = content.decode("utf-8")
            captured["headers"] = headers
            return FakeResponse()

    fake_httpx = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = module._validate_feishu_webhook_impl()

    assert result["status"] == "ok"
    assert result["status_code"] == 200
    assert result["webhook_url"] == "https://example.com/feishu/webhook"
    assert result["response"] == {"challenge": "feishu-encrypted-selftest-ok"}
    assert captured["url"] == "https://example.com/feishu/webhook"
    assert "x-lark-signature" in captured["headers"]
    assert '"encrypt"' in captured["body"]


def test_approve_pairing_impl_returns_approved_user(monkeypatch):
    module = _load_module()

    class FakeStore:
        def list_pending(self, platform):
            assert platform == "feishu"
            return [{"platform": "feishu", "code": "HK2DHZ3A"}]

        def approve_code(self, platform, code):
            assert platform == "feishu"
            assert code == "HK2DHZ3A"
            return {"user_id": "ou_xxx", "user_name": "suyee"}

        def list_approved(self, platform):
            assert platform == "feishu"
            return [{"platform": "feishu", "user_id": "ou_xxx", "user_name": "suyee"}]

    monkeypatch.setitem(sys.modules, "gateway.pairing", types.SimpleNamespace(PairingStore=FakeStore))

    result = module._approve_pairing_impl("feishu", "hk2dhz3a")

    assert result["status"] == "approved"
    assert result["approved_user"]["user_id"] == "ou_xxx"
    assert result["code"] == "HK2DHZ3A"


def test_healthz_reports_telegram_webhook_status(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token-123")
    monkeypatch.setenv("HERMES_PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_AUTO_SYNC", "false")

    async def _fake_status(_settings, *, ensure_registered=False, drop_pending_updates=False):
        assert ensure_registered is False
        assert drop_pending_updates is False
        return {
            "configured": True,
            "expected_url": "https://example.com/telegram/webhook",
            "registered_url": "https://example.com/telegram/webhook",
            "matches_expected": True,
        }

    monkeypatch.setattr(module, "_get_telegram_webhook_status", _fake_status)

    client = TestClient(module.create_web_app())
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["telegram_configured"] is True
    assert payload["telegram_webhook"]["matches_expected"] is True


def test_healthz_reports_invalid_telegram_token_format(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "AAE1kxRtD3uvpjI1Y_Xkg-zrv96fDLeuN_4")
    monkeypatch.setenv("HERMES_PUBLIC_BASE_URL", "https://example.com")

    client = TestClient(module.create_web_app())
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["telegram_webhook"]["token_format_valid"] is False
    assert payload["telegram_webhook"]["reason"] == "telegram_bot_token_invalid_format"


def test_dispatch_telegram_update_processes_text_synchronously(monkeypatch):
    module = _load_module()
    import gateway.session as gateway_session

    handled = {}

    class FakeAdapter:
        def __init__(self):
            self._bot = object()
            self._active_sessions = {}
            self.config = types.SimpleNamespace(extra={})

        def _should_process_message(self, message):
            return bool(message.text)

        def _build_message_event(self, message, _message_type):
            handled["raw_text"] = message.text
            return types.SimpleNamespace(
                text=message.text,
                source=types.SimpleNamespace(
                    platform=types.SimpleNamespace(value="telegram"),
                    chat_id="6379576758",
                    chat_type="dm",
                    user_id="6379576758",
                    thread_id=None,
                ),
            )

        def _clean_bot_trigger_text(self, text):
            return text.replace("@suyeeagentbot", "").strip()

        async def _process_message_background(self, event, _session_key):
            handled["final_text"] = event.text

    fake_runtime = types.SimpleNamespace(adapter=FakeAdapter())

    async def _fake_get_runtime():
        return fake_runtime

    class FakeMessage:
        def __init__(self, text):
            self.text = text

    class FakeUpdate:
        def __init__(self, message):
            self.message = message
            self.edited_message = None
            self.channel_post = None
            self.edited_channel_post = None
            self.callback_query = None

        @classmethod
        def de_json(cls, payload, _bot):
            return cls(FakeMessage(payload["message"]["text"]))

    monkeypatch.setattr(module, "_get_telegram_gateway_runtime", _fake_get_runtime)
    monkeypatch.setitem(sys.modules, "telegram", types.SimpleNamespace(Update=FakeUpdate))
    monkeypatch.setattr(
        gateway_session,
        "build_session_key",
        lambda source, **_kwargs: f"{source.platform.value}:{source.chat_id}",
    )

    result = asyncio.run(
        module._dispatch_telegram_update(
            {"message": {"text": "@suyeeagentbot  请只回复 SELFTEST_OK"}}
        )
    )

    assert result == {"status": "accepted", "kind": "text"}
    assert handled["raw_text"] == "@suyeeagentbot  请只回复 SELFTEST_OK"
    assert handled["final_text"] == "请只回复 SELFTEST_OK"


def test_modal_source_includes_bundled_skill_directories():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '.add_local_dir("skills", remote_path="/root/skills", copy=True)' in source
    assert '.add_local_dir("optional-skills", remote_path="/root/optional-skills", copy=True)' in source
    assert '.add_local_dir("acp_registry", remote_path="/root/acp_registry", copy=True)' in source


def test_modal_source_supports_project_plugins_and_mcp_runtimes():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '.apt_install("nodejs", "npm")' in source
    assert '"uv>=0.7.0,<1"' in source
    assert '"feishu"' in source
    assert '"/feishu/webhook"' in source
    assert 'def debug_feishu_runtime()' in source
    assert 'def debug_feishu_menu_config()' in source
    assert 'def debug_model_routing_state(' in source
    assert 'Path(".hermes/plugins").is_dir()' in source
    assert 'remote_path="/root/.hermes/plugins"' in source
    assert 'def validate_tavily_integration()' in source
    assert 'modal.Queue.from_name(DEFAULT_CHAT_QUEUE_NAME' in source
    assert 'modal.Queue.from_name(DEFAULT_CRON_QUEUE_NAME' in source
    assert 'schedule=modal.Period(minutes=1)' in source
