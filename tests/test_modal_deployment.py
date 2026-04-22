import importlib.util
import asyncio
import json
import os
import sys
import time
import types
from pathlib import Path

from fastapi.testclient import TestClient
import yaml


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


def test_validate_feishu_internal_bearer_token_accepts_secondary_app_derivation(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FEISHU_APP_ID2", "cli_secondary_app")
    monkeypatch.setenv("FEISHU_APP_SECRET2", "secondary-secret-123")

    settings = module.RuntimeSettings(
        model="openrouter/free",
        max_turns=16,
        max_tokens=None,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        bearer_token=None,
        feishu_internal_bearer_token="fi_primary",
        telegram_bot_token=None,
        telegram_webhook_secret=None,
        telegram_webhook_url=None,
        telegram_send_ack=False,
        feishu_app_id="cli_primary_app",
        feishu_app_secret="primary-secret-123",
        feishu_domain="feishu",
        feishu_connection_mode="webhook",
        feishu_verification_token=None,
        feishu_encrypt_key=None,
        feishu_bitable_app_token=None,
        feishu_bitable_table_id=None,
        feishu_model_registry_mirror_enabled=False,
        feishu_tool_capabilities=[],
        feishu_default_workspace=None,
        qq_app_id=None,
        qq_app_secret=None,
        nvidia_api_key=None,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        enabled_toolsets=[],
        disabled_toolsets=[],
    )

    secondary_token = module._derive_feishu_internal_bearer_token(
        app_id="cli_secondary_app",
        app_secret="secondary-secret-123",
    )

    assert module._validate_feishu_internal_bearer_token(f"Bearer {secondary_token}", settings) is True
    assert module._validate_feishu_internal_bearer_token("Bearer wrong", settings) is False


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


def test_extract_feishu_inline_fast_command_from_text_payload():
    module = _load_module()
    payload = {
        "event": {
            "message": {
                "message_type": "text",
                "content": json.dumps({"text": "/model moonshotai/kimi-k2.5 --provider openrouter"}),
            }
        }
    }

    assert module._extract_feishu_inline_fast_command(payload) == "model"


def test_extract_feishu_inline_fast_command_ignores_leading_mentions():
    module = _load_module()
    payload = {
        "event": {
            "message": {
                "message_type": "text",
                "content": json.dumps({"text": "<at user_id=\"ou_x\">Hermes</at> /provider"}),
            }
        }
    }

    assert module._extract_feishu_inline_fast_command(payload) == "provider"


def test_extract_feishu_trace_token_from_text_payload():
    module = _load_module()
    payload = {
        "event": {
            "message": {
                "message_type": "text",
                "content": json.dumps({"text": "hello world [trace:fev_123abc]"}),
            }
        }
    }

    assert module._extract_feishu_trace_token(payload) == "fev_123abc"


def test_infer_feishu_internal_route_hint_prefers_fast_control_for_commands():
    module = _load_module()

    route_hint = module._infer_feishu_internal_route_hint(
        {
            "task_kind": "command",
            "message_type": "command",
            "text": "/model openai/gpt-4.1",
        }
    )

    assert route_hint == "fast_control"


def test_infer_feishu_internal_route_hint_marks_browser_requests():
    module = _load_module()

    route_hint = module._infer_feishu_internal_route_hint(
        {
            "task_kind": "text",
            "message_type": "text",
            "text": "请打开 https://example.com 并截图给我",
        }
    )

    assert route_hint == "cf_browser_first"


def test_extract_feishu_trace_context_preserves_gateway_classification_fields():
    module = _load_module()

    payload = {
        "correlation_id": "feishu:chat:evt_123",
        "session_key": "agent:main:feishu:group:chat:user",
        "event_type": "im.message.receive_v1",
        "event_id": "evt_123",
        "message_id": "om_123",
        "chat_id": "oc_123",
        "route_hint": "modal_heavy_exec",
        "task_kind": "text",
        "request_class": "text_plain",
        "route_family": "gateway_text",
        "gateway_route_name": "affiliate-general",
        "gateway_eligible": True,
        "requires_tools": False,
        "requires_browser": False,
        "requires_media_hydration": False,
        "requires_modal_runtime": False,
        "content_modalities": ["text"],
        "toolset": [],
        "modality_profile": "text",
        "reason_code": "plain_text_without_attachments_or_browser",
        "site_category": "site_content",
        "site_intent": "docs",
        "target_domain": "developers.cloudflare.com",
        "target_url": "https://developers.cloudflare.com/ai-gateway/",
        "site_prefetch": {
            "mode": "browser_markdown",
            "status": "completed",
        },
        "site_prefetch_direct_navigation": True,
        "site_prefetch_direct_mode": "content_direct",
        "browser_target_domain": "developers.cloudflare.com",
        "estimated_cost_usd": 0.0123,
        "event": {
            "message": {
                "message_id": "om_from_event",
                "chat_type": "group",
            }
        },
    }

    result = module._extract_feishu_trace_context(payload)

    assert result["correlation_id"] == "feishu:chat:evt_123"
    assert result["session_key"] == "agent:main:feishu:group:chat:user"
    assert result["message_id"] == "om_123"
    assert result["chat_id"] == "oc_123"
    assert result["request_class"] == "text_plain"
    assert result["route_family"] == "gateway_text"
    assert result["gateway_route_name"] == "affiliate-general"
    assert result["site_category"] == "site_content"
    assert result["site_intent"] == "docs"
    assert result["site_prefetch_mode"] == "browser_markdown"
    assert result["site_prefetch_status"] == "completed"
    assert result["site_prefetch_direct_navigation"] is True
    assert result["site_prefetch_direct_mode"] == "content_direct"
    assert result["browser_target_domain"] == "developers.cloudflare.com"
    assert result["estimated_cost_usd"] == 0.0123


def test_extract_feishu_trace_context_reads_site_prefetch_from_ingress_meta():
    module = _load_module()

    payload = {
        "event_type": "im.message.receive_v1",
        "event_id": "evt_ingress",
        "message_id": "om_ingress",
        "_hermes_ingress": {
            "site_prefetch": {
                "mode": "playwright_preflight",
                "status": "completed",
                "confidence": 0.8,
            }
        },
        "event": {
            "message": {
                "message_id": "om_ingress",
                "chat_type": "p2p",
            }
        },
    }

    result = module._extract_feishu_trace_context(payload)

    assert result["site_prefetch_mode"] == "playwright_preflight"
    assert result["site_prefetch_status"] == "completed"
    assert result["site_prefetch_confidence"] == 0.8


def test_build_feishu_internal_result_includes_action_plan_and_flags():
    module = _load_module()

    result = module._build_feishu_internal_result(
        status="ok",
        route_hint="fast_control",
        execution_mode="control_complete",
        session_state_before={"current_model": "a"},
        session_state_after={"current_model": "b"},
        send_plan=[{"kind": "text", "content": "hello"}],
        final_response="hello",
        reconcile_required=False,
        browser_fallback_allowed=False,
        action="dispatch_command",
    )

    assert result["route_hint"] == "fast_control"
    assert result["execution_mode"] == "control_complete"
    assert result["send_plan"][0]["kind"] == "text"
    assert result["action_plan"][0]["content"] == "hello"
    assert result["reconcile_required"] is False


def test_build_feishu_internal_plan_marks_plain_text_as_external_exec_candidate():
    module = _load_module()

    result = module._build_feishu_internal_plan(
        {
            "task_kind": "text",
            "message_type": "text",
            "text": "请帮我总结一下这周的工作重点",
            "attachment_refs": [],
        },
        session_state_before={
            "current_model": "openrouter/free",
            "current_provider": "openrouter",
            "route_debug": {
                "last_model": "openai/gpt-oss-20b:free",
                "last_provider": "openrouter",
            },
        },
    )

    assert result["route_hint"] == "modal_heavy_exec"
    assert result["execution_mode"] == "deferred_reconcile"
    assert result["external_exec_candidate"] is True
    assert result["reconcile_required"] is True
    assert result["provider_plan"]["mode"] == "cloudflare_workflow_candidate"
    assert result["provider_plan"]["model"] == "mistralai/mistral-small-3.1-24b-instruct"
    assert result["provider_plan"]["session_model"] == "openrouter/free"
    assert result["provider_plan"]["fallback_model"] == "deepseek/deepseek-chat-v3-0324"
    assert result["provider_plan"]["fallback_provider"] == "openrouter"
    assert result["provider_plan"]["request_timeout_ms"] == 18000
    assert result["provider_plan"]["max_attempts"] == 2
    assert result["provider_plan"]["cache_mode"] == "ttl"
    assert result["provider_plan"]["cache_scope"] == "chat"
    assert result["provider_plan"]["cache_ttl_seconds"] == 300


def test_build_feishu_internal_plan_rewrites_free_llm_request_model_to_paid_exec_model():
    module = _load_module()

    result = module._build_feishu_internal_plan(
        {
            "task_kind": "text",
            "message_type": "text",
            "text": "please summarize this thread",
            "attachment_refs": [],
        },
        session_state_before={
            "current_model": "openrouter/free",
            "current_provider": "openrouter",
            "route_debug": {
                "last_model": "openai/gpt-oss-20b:free",
                "last_provider": "openrouter",
            },
        },
        llm_request={
            "model": "openrouter/free",
            "messages": [{"role": "user", "content": "please summarize this thread"}],
        },
    )

    assert result["external_exec_candidate"] is True
    assert result["provider_plan"]["model"] == "mistralai/mistral-small-3.1-24b-instruct"
    assert result["llm_request"]["model"] == "mistralai/mistral-small-3.1-24b-instruct"


def test_build_feishu_internal_plan_keeps_browser_requests_in_modal_exec():
    module = _load_module()

    result = module._build_feishu_internal_plan(
        {
            "task_kind": "text",
            "message_type": "text",
            "text": "请打开 https://example.com 并截图给我",
            "attachment_refs": [],
        }
    )

    assert result["route_hint"] == "cf_browser_first"
    assert result["execution_mode"] == "cf_browser_first"
    assert result["external_exec_candidate"] is False
    assert result["reconcile_required"] is False


def test_extract_feishu_message_read_event_info():
    module = _load_module()
    payload = {
        "header": {
            "event_type": "im.message.message_read_v1",
            "event_id": "evt_read_123",
        },
        "event": {
            "reader": {
                "reader_id": {
                    "open_id": "ou_reader_123",
                    "user_id": "reader_user_123",
                    "union_id": "union_reader_123",
                },
                "tenant_key": "tenant_key_123",
                "read_time": "1712970000",
            },
            "message_id_list": ["om_1", "om_2"],
        },
    }

    result = module._extract_feishu_message_read_event_info(payload)

    assert result["event_type"] == "im.message.message_read_v1"
    assert result["event_id"] == "evt_read_123"
    assert result["reader_open_id"] == "ou_reader_123"
    assert result["reader_user_id"] == "reader_user_123"
    assert result["reader_union_id"] == "union_reader_123"
    assert result["tenant_key"] == "tenant_key_123"
    assert result["read_time"] == 1712970000
    assert result["message_id_list"] == ["om_1", "om_2"]
    assert result["message_count"] == 2


def test_resolve_feishu_request_started_at_ms_prefers_valid_epoch_over_perf_counter():
    module = _load_module()
    payload = module._with_feishu_internal_meta(
        {"event": {"message": {"message_id": "om_1"}}},
        ack_reaction_requested_at_ms=1712970000000,
    )

    assert module._resolve_feishu_request_started_at_ms(payload, 123456789) == 1712970000000


def test_resolve_feishu_request_started_at_ms_uses_explicit_valid_epoch():
    module = _load_module()
    payload = module._with_feishu_internal_meta(
        {"event": {"message": {"message_id": "om_1"}}},
        ack_reaction_requested_at_ms=1712970000000,
    )

    assert module._resolve_feishu_request_started_at_ms(payload, 1712971234567) == 1712971234567


def test_extract_feishu_internal_request_meta_reads_gateway_headers():
    module = _load_module()

    result = module._extract_feishu_internal_request_meta(
        {
            "x-hermes-gateway-hop": "cloudflare-worker",
            "x-hermes-gateway-script": "hermes-feishu-gateway",
            "x-hermes-correlation-id": "feishu:chat:evt_1",
            "x-hermes-event-id": "evt_1",
            "x-hermes-session-key": "agent:main:feishu:dm:oc_1",
        }
    )

    assert result["gateway_hop"] == "cloudflare-worker"
    assert result["gateway_script"] == "hermes-feishu-gateway"
    assert result["gateway_correlation_id"] == "feishu:chat:evt_1"
    assert result["gateway_event_id"] == "evt_1"
    assert result["gateway_session_key"] == "agent:main:feishu:dm:oc_1"


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


def test_chat_partition_claim_can_take_over_stale_claim(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module.DEFAULT_CHAT_QUEUE_STALE_CLAIM_TAKEOVER_SECONDS = 30
    module._ensure_runtime_dirs()

    old_token = "claim-old"
    module._save_chat_queue_claims(
        {
            "feishu:oc_chat": {
                "claim_token": old_token,
                "claimed_at": int(time.time()) - 120,
                "platform": "feishu",
                "status": "claimed",
            }
        }
    )

    claimed, token = module._claim_chat_partition("feishu:oc_chat", platform="feishu", ttl_seconds=3600)

    assert claimed is True
    assert token != old_token


def test_feishu_bot_identity_cache_round_trip(tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.FEISHU_BOT_IDENTITY_CACHE_PATH = module.DATA_ROOT / "feishu_bot_identity_cache.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    assert module._get_cached_feishu_bot_identity("cli_app") == {}

    asyncio.run(module._cache_feishu_bot_identity_async("cli_app", bot_name="Hermes"))

    cached = module._get_cached_feishu_bot_identity("cli_app")
    assert cached["bot_name"] == "Hermes"
    assert cached["updated_at"] > 0


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
    monkeypatch.setattr(
        module,
        "_claim_chat_partition",
        lambda partition, platform, claim_token=None: (True, claim_token or "claim-1"),
    )
    async def _fake_process_items(items, **kwargs):
        return [{"status": "ok", "item": item} for item in items]

    monkeypatch.setattr(module, "_process_chat_queue_items_async", _fake_process_items)
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
    assert result["worker_boot_id"].startswith("inline-")
    assert result["container_reused"] is False
    assert result["batch_size"] == 1
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


def test_config_modal_seeds_ceo_personality_defaults():
    payload = yaml.safe_load((REPO_ROOT / "config.modal.yaml").read_text(encoding="utf-8"))

    assert payload["agent"]["personalities"]["ceo"]["description"].startswith("CEO")
    assert "Tone:" in payload["agent"]["system_prompt"]


def test_config_modal_enables_startup_browser_and_toolset_defaults():
    payload = yaml.safe_load((REPO_ROOT / "config.modal.yaml").read_text(encoding="utf-8"))

    assert payload["browser"]["cloud_provider"] == "local"
    assert payload["browser"]["command_timeout"] == 30
    assert payload["skills"]["startup"] == ["affiliate-os", "browser-ops"]
    assert payload["skills"]["platform_startup"]["cli"] == ["automation-os", "ceo-os"]
    assert payload["skills"]["platform_startup"]["feishu"] == ["feishu-workbench", "affiliate-os", "browser-ops"]
    assert payload["platform_toolsets"]["cli"] == ["founder-max", "feishu", "plugin_startup_ops"]
    assert payload["platform_toolsets"]["feishu"] == ["collab-safe", "feishu"]
    assert payload["platform_toolsets"]["api_server"] == ["cto-max", "plugin_startup_ops"]


def test_config_modal_seeds_affiliate_operator_personalities():
    payload = yaml.safe_load((REPO_ROOT / "config.modal.yaml").read_text(encoding="utf-8"))
    personalities = payload["agent"]["personalities"]

    for name in ("content", "seo", "ads", "bd", "ops", "finance"):
        assert name in personalities
        assert personalities[name]["description"]


def test_runtime_bootstrap_debug_state_reports_startup_skills_and_project_plugins(
    tmp_path, monkeypatch
):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"

    repo_root = tmp_path / "repo"
    app_dir = repo_root / "hermes-agent"
    app_dir.mkdir(parents=True)
    plugin_dir = repo_root / ".hermes" / "plugins" / "repo_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "repo_plugin",
                "version": "0.1.0",
                "description": "repo scoped plugin",
                "provides_hooks": ["on_session_start"],
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        'def register(ctx):\n    ctx.register_hook("on_session_start", lambda **kw: None)\n',
        encoding="utf-8",
    )

    config_source = tmp_path / "config.modal.yaml"
    config_source.write_text(
        yaml.safe_dump(
            {
                "model": {"default": "openrouter/free"},
                "agent": {
                    "personality": "ceo",
                    "personalities": {
                        "ceo": {
                            "description": "CEO operator",
                            "system_prompt": "Run the company.",
                        }
                    },
                },
                "skills": {
                    "startup": ["affiliate-os", "browser-ops"],
                    "platform_startup": {
                        "cli": ["automation-os", "ceo-os"],
                        "feishu": ["affiliate-os", "browser-ops"],
                    },
                },
                "plugins": {"enable_project": True},
                "platform_toolsets": {
                    "cli": ["founder-max", "plugin_startup_ops"],
                    "feishu": ["collab-safe", "feishu"],
                    "api_server": ["cto-max"],
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_MODAL_CONFIG_SOURCE", str(config_source))
    monkeypatch.setenv("HERMES_MODAL_SYNC_CONFIG", "true")
    monkeypatch.setenv("HERMES_HOME", str(module.HERMES_HOME_DIR))
    monkeypatch.chdir(app_dir)

    import hermes_cli.plugins as plugins_mod

    plugins_mod._plugin_manager = None
    try:
        payload = module._build_runtime_bootstrap_debug_state(("cli", "feishu"))
    finally:
        plugins_mod._plugin_manager = None

    assert payload["config_loaded"] is True
    assert payload["agent"]["default_personality"] == "ceo"
    assert "ceo" in payload["agent"]["available_personalities"]
    assert payload["startup_skills"]["global"] == ["affiliate-os", "browser-ops"]
    assert payload["startup_skills"]["platforms"]["cli"] == [
        "affiliate-os",
        "browser-ops",
        "automation-os",
        "ceo-os",
    ]
    assert payload["project_plugins"]["search_dir"].endswith(str(Path(".hermes") / "plugins"))
    assert payload["project_plugins"]["search_dir_exists"] is True
    assert any(plugin["name"] == "repo_plugin" and plugin["enabled"] for plugin in payload["plugins"]["loaded"])
    assert payload["platform_toolsets"]["cli"]["configured_toolsets"] == ["founder-max", "plugin_startup_ops"]
    assert payload["platform_toolsets"]["feishu"]["configured_toolsets"] == ["collab-safe", "feishu"]
    assert "browser" in payload["platform_toolsets"]["cli"]["effective_toolsets"]
    assert payload["platform_toolsets"]["cli"]["resolved_tool_count"] > 0


def test_sync_runtime_config_preserves_existing_mcp_servers_without_feishu_injection(tmp_path, monkeypatch):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"

    source = tmp_path / "config.modal.yaml"
    source.write_text(
        "mcp_servers:\n"
        "  github:\n"
        "    command: npx\n"
        "    args: [github-mcp]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MODAL_CONFIG_SOURCE", str(source))
    monkeypatch.setenv("HERMES_MODAL_SYNC_CONFIG", "true")
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test_feishu")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test_feishu")

    module._ensure_runtime_dirs()
    written_path = Path(module._sync_runtime_config())
    payload = yaml.safe_load(written_path.read_text(encoding="utf-8"))

    assert payload["mcp_servers"] == {"github": {"command": "npx", "args": ["github-mcp"]}}


def test_modal_official_parity_state_reports_modal_exceptions(monkeypatch):
    module = _load_module()

    monkeypatch.delenv("HERMES_ENABLED_TOOLSETS", raising=False)
    monkeypatch.delenv("HERMES_DISABLED_TOOLSETS", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")

    original_find_spec = module.importlib.util.find_spec

    def _fake_find_spec(name):
        if name in {
            "plugins.memory.honcho",
            "tools.homeassistant_tool",
            "botpy",
        }:
            return object()
        return original_find_spec(name)

    monkeypatch.setattr(module.importlib.util, "find_spec", _fake_find_spec)

    payload = module._build_modal_official_parity_state()

    assert payload["features"]["core_agent_loop"]["status"] == "supported"
    assert payload["features"]["browser_tools"]["status"] == "supported"
    assert payload["features"]["honcho_memory_provider"]["status"] == "supported"
    assert payload["platforms"]["telegram"]["status"] == "supported"
    assert payload["platforms"]["feishu"]["status"] == "supported"
    assert payload["platforms"]["qq"]["status"] == "supported"
    assert payload["platforms"]["discord"]["status"] == "unsupported"
    assert payload["platforms"]["matrix"]["status"] == "unsupported"
    assert "voice" in payload["default_disabled_toolsets"]
    assert "rl" in payload["default_disabled_toolsets"]


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


def test_prepare_runtime_environment_syncs_bundled_skills_once(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._RUNTIME_SKILLS_SYNCED = False

    sync_calls = []

    def _fake_sync_skills(*, quiet: bool = False):
        sync_calls.append({"quiet": quiet})
        skills_dir = module.HERMES_HOME_DIR / "skills" / "productivity" / "feishu-workbench"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text("# test\n", encoding="utf-8")
        return {"copied": ["feishu-workbench"]}

    import types as _types

    monkeypatch.setenv("HERMES_HOME", str(module.HERMES_HOME_DIR))
    monkeypatch.setitem(sys.modules, "tools.skills_sync", _types.SimpleNamespace(sync_skills=_fake_sync_skills))

    module._prepare_runtime_environment()
    module._prepare_runtime_environment()

    assert sync_calls == [{"quiet": True}]
    assert (module.HERMES_HOME_DIR / "skills" / "productivity" / "feishu-workbench" / "SKILL.md").exists()


def test_prepare_runtime_environment_keeps_feishu_skills_toolset_enabled_by_default(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._RUNTIME_SKILLS_SYNCED = True

    monkeypatch.delenv("HERMES_FEISHU_DISABLED_TOOLSETS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(module.HERMES_HOME_DIR))

    module._prepare_runtime_environment()

    assert "skills" not in os.environ["HERMES_FEISHU_DISABLED_TOOLSETS"].split(",")


def test_resolve_camofox_launch_command_prefers_installed_binary(monkeypatch):
    module = _load_module()

    def _fake_which(name: str):
        if name == "camofox-browser":
            return "/usr/local/bin/camofox-browser"
        return None

    monkeypatch.setattr(module.shutil, "which", _fake_which)

    assert module._resolve_camofox_launch_command() == ["/usr/local/bin/camofox-browser"]


def test_resolve_camofox_launch_command_falls_back_to_npx(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    assert module._resolve_camofox_launch_command() == ["npx", "--yes", "@askjo/camofox-browser"]


def test_ensure_camofox_server_starts_local_process_when_needed(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module._CAMOFOX_SERVER_PROCESS = None

    monkeypatch.setenv("CAMOFOX_URL", "http://127.0.0.1:9377")
    monkeypatch.setattr(module, "_resolve_camofox_launch_command", lambda: ["camofox-browser"])

    readiness_checks = iter([False, False, True])
    monkeypatch.setattr(module, "_is_camofox_healthcheck_ready", lambda _url: next(readiness_checks))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 4321

        def poll(self):
            return None

    def _fake_popen(cmd, cwd, env, stdin, stdout, stderr, start_new_session):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return _FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)

    module._ensure_camofox_server()

    assert captured["cmd"] == ["camofox-browser"]
    assert captured["cwd"] == "/root"
    assert captured["env"]["CAMOFOX_PORT"] == "9377"
    assert (module.DATA_ROOT / "logs" / "camofox.log").exists()


def test_ensure_camofox_server_skips_remote_url(monkeypatch):
    module = _load_module()
    module._CAMOFOX_SERVER_PROCESS = None
    monkeypatch.setenv("CAMOFOX_URL", "https://browser.example.com")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not start")))

    module._ensure_camofox_server()


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


def test_runtime_settings_derive_feishu_internal_bearer(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HERMES_FEISHU_INTERNAL_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("HERMES_WEBHOOK_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")

    settings = module.RuntimeSettings.from_env()

    expected = module._derive_feishu_internal_bearer_token(
        app_id="cli_feishu_app",
        app_secret="feishu-secret-123",
    )
    assert settings.feishu_internal_bearer_token == expected


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
    assert result["partition"] == "feishu:control:oc_menu_chat"
    assert result["lane"] == "control"
    assert result["chat_id"] == "oc_menu_chat"
    assert result["actor_id"] == "ou_menu_operator"
    assert result["event_type"] == "application.bot.menu_v6"


def test_extract_feishu_queue_context_handles_bot_entered_user_shape():
    module = _load_module()
    payload = {
        "header": {
            "event_id": "evt-entered-1",
            "event_type": "im.chat.access_event.bot_p2p_chat_entered_v1",
        },
        "event": {
            "user_id": {"open_id": "ou_entered_user"},
            "open_chat_id": "oc_entered_chat",
        },
    }

    result = module._extract_feishu_queue_context(payload)

    assert result["platform"] == "feishu"
    assert result["partition"] == "feishu:control:oc_entered_chat"
    assert result["lane"] == "control"
    assert result["chat_id"] == "oc_entered_chat"
    assert result["actor_id"] == "ou_entered_user"
    assert result["event_type"] == "im.chat.access_event.bot_p2p_chat_entered_v1"

    warmup = module._extract_feishu_warmup_context(payload)
    assert warmup["lane"] == "chat_light"
    assert warmup["partition"] == "feishu:chat_light:oc_entered_chat"


def test_build_feishu_internal_source_falls_back_to_raw_control_event_actor():
    module = _load_module()

    source = module._build_feishu_internal_source(
        {
            "chat_type": "group",
            "raw_message": {
                "event": {
                    "context": {"open_chat_id": "oc_control_chat"},
                    "operator": {"open_id": "ou_control_user"},
                }
            },
        }
    )

    assert source.chat_id == "oc_control_chat"
    assert source.user_id == "ou_control_user"
    assert source.user_name == "ou_control_user"


def test_classify_feishu_chat_lane_prefers_chat_light_for_short_text():
    module = _load_module()
    payload = {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": "evt_light_1",
        },
        "event": {
            "message": {
                "message_type": "text",
                "content": json.dumps({"text": "hello"}),
            }
        },
    }

    assert module._classify_feishu_chat_lane(payload) == "chat_light"
    context = module._extract_feishu_queue_context(payload)
    assert context["lane"] == "chat_light"
    assert context["partition"] == "feishu:chat_light:unknown"


def test_classify_feishu_chat_lane_marks_non_text_messages_heavy():
    module = _load_module()
    payload = {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": "evt_heavy_1",
        },
        "event": {
            "open_chat_id": "oc_heavy_chat",
            "message": {
                "message_type": "image",
                "content": json.dumps({"image_key": "img_123"}),
            },
        },
    }

    assert module._classify_feishu_chat_lane(payload) == "chat_heavy"
    context = module._extract_feishu_queue_context(payload)
    assert context["lane"] == "chat_heavy"
    assert context["partition"] == "feishu:chat_heavy:oc_heavy_chat"


def test_process_chat_queue_warmup_snapshot_is_consumed_once(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_EVENTS_PATH = module.DATA_ROOT / "feishu_events.json"
    module.FEISHU_TRACE_PATH = module.DATA_ROOT / "feishu_trace.jsonl"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module.CHAT_QUEUE_WARMUPS_PATH = module.DATA_ROOT / "chat_queue_warmups.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setattr(module, "_sync_modal_volume", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_safe_chat_queue_depth", lambda: 0)
    monkeypatch.setattr(module, "_claim_chat_partition", lambda *args, **kwargs: (False, "claimed"))

    warm_context = {
        "worker_boot_id": "boot-warm",
        "worker_started_at": 111,
        "enter_elapsed_ms": 25,
        "runtime_prepare_elapsed_ms": 25,
        "chat_id": "oc_entered_chat",
        "actor_id": "ou_entered_user",
        "event_id": "evt-entered-1",
        "event_type": "im.chat.access_event.bot_p2p_chat_entered_v1",
        "started_at_ms": "1000",
    }
    warm_result = module._process_chat_queue_impl(
        platform="feishu",
        partition="feishu:oc_entered_chat",
        max_items=0,
        worker_context=warm_context,
        runtime_prepared=True,
    )

    assert warm_result["status"] == "warmed"
    assert warm_result["warmup_status"] == "ready"

    cold_result = module._process_chat_queue_impl(
        platform="feishu",
        partition="feishu:oc_entered_chat",
        max_items=1,
        worker_context={"worker_boot_id": "boot-real"},
        runtime_prepared=True,
    )

    assert cold_result["status"] == "skipped"
    assert cold_result["reason"] == "already_claimed"
    assert cold_result["warmup_status"] == "ready"
    assert cold_result["warmup_actor_id"] == "ou_entered_user"
    assert cold_result["warmup_chat_id"] == "oc_entered_chat"
    assert cold_result["warmup_same_container"] is False

    next_result = module._process_chat_queue_impl(
        platform="feishu",
        partition="feishu:oc_entered_chat",
        max_items=1,
        worker_context={"worker_boot_id": "boot-next"},
        runtime_prepared=True,
    )

    assert next_result["status"] == "skipped"
    assert next_result["reason"] == "already_claimed"
    assert "warmup_status" not in next_result


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


def test_process_chat_queue_batch_reuses_single_async_run(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module._ensure_runtime_dirs()

    class FakeQueue:
        def __init__(self):
            self._polls = 0

        def get_many(self, _max_items, block=False, timeout=None, partition=None):
            self._polls += 1
            if self._polls == 1:
                return [{"payload": "one"}, {"payload": "two"}]
            return []

    asyncio_runs = []

    async def _fake_process_items(items, **kwargs):
        return [{"status": "ok", "payload": item["payload"]} for item in items]

    real_asyncio_run = module.asyncio.run

    def _record_run(coro):
        asyncio_runs.append(type(coro).__name__)
        return real_asyncio_run(coro)

    monkeypatch.setattr(module, "_get_chat_queue", lambda: FakeQueue())
    monkeypatch.setattr(
        module,
        "_claim_chat_partition",
        lambda partition, platform, claim_token=None: (True, claim_token or "claim-1"),
    )
    monkeypatch.setattr(module, "_process_chat_queue_items_async", _fake_process_items)
    monkeypatch.setattr(module, "_safe_chat_queue_depth", lambda: 0)
    monkeypatch.setattr(module.asyncio, "run", _record_run)
    monkeypatch.setattr(module, "_sync_modal_volume", lambda **kwargs: None)
    monkeypatch.setattr(module, "_release_chat_partition_claim", lambda *args, **kwargs: None)

    result = module._process_chat_queue_impl(platform="feishu", partition="feishu:oc_chat", max_items=8)

    assert result["processed_count"] == 2
    assert result["batch_size"] == 2
    assert asyncio_runs == ["coroutine"]


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


def test_debug_feishu_sync_state_helper(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_SYNC_STATE_PATH = module.DATA_ROOT / "feishu_sync_state.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()
    monkeypatch.setattr(module, "_prepare_runtime_environment", lambda: None)
    monkeypatch.setenv("FEISHU_BITABLE_WIKI_TOKEN", "wiki_token_123")
    monkeypatch.setenv("FEISHU_BITABLE_TABLE_ID", "tbl_123")
    monkeypatch.setenv("FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS", "900")
    module._save_feishu_sync_state(
        {
            "last_attempt_at": 100,
            "last_success_at": 120,
            "last_status": "ok",
            "last_sync": {"status": "ok", "mirrored": True, "created": 2},
            "schema": {"status": "ok", "table_id": "tbl_123"},
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.feishu_api",
        types.SimpleNamespace(
            build_feishu_client=lambda: object(),
            resolve_bitable_target=lambda _args, _client: ("app_token_123", "tbl_123"),
        ),
    )

    payload = module._build_feishu_sync_state_debug_state()

    assert payload["configured"] is True
    assert payload["sync_interval_seconds"] == 900
    assert payload["schema"]["table_id"] == "tbl_123"
    assert payload["resolved_target"]["table_id"] == "tbl_123"


def test_should_prepare_feishu_registry_schema_on_startup_skips_recent_ok_schema(monkeypatch):
    module = _load_module()
    recent_checked_at = int(time.time()) - 60
    monkeypatch.setattr(module, "DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS", 3600)

    should_prepare = module._should_prepare_feishu_registry_schema_on_startup(
        {
            "schema": {
                "status": "ok",
                "checked_at": recent_checked_at,
                "missing_required_fields": [],
                "missing_views": [],
            }
        }
    )

    assert should_prepare is False


def test_should_prepare_feishu_registry_schema_on_startup_rechecks_stale_schema(monkeypatch):
    module = _load_module()
    stale_checked_at = int(time.time()) - 7200
    monkeypatch.setattr(module, "DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS", 3600)

    should_prepare = module._should_prepare_feishu_registry_schema_on_startup(
        {
            "schema": {
                "status": "ok",
                "checked_at": stale_checked_at,
                "missing_required_fields": [],
                "missing_views": [],
            }
        }
    )

    assert should_prepare is True


def _obsolete_test_removed_feishu_debug_helper(monkeypatch):
    pass


def _obsolete_test_removed_feishu_parity_helper(monkeypatch, tmp_path):
    pass

def test_debug_feishu_runtime_uses_api_only_surface(monkeypatch):
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'def debug_feishu_runtime()' in source
    assert '"feishu_sync": _build_feishu_sync_state_debug_state()' in source
    assert '"memory_snapshots": {' in source


def test_feishu_platform_hint_explicitly_mentions_native_workspace_tools():
    from agent.prompt_builder import PLATFORM_HINTS

    hint = PLATFORM_HINTS["feishu"]

    assert "feishu_*" in hint
    assert "browser_*" in hint
    assert "Docs" in hint
    assert "Sheets" in hint
    assert "Bitable" in hint
    assert "instead of claiming that Feishu access is unavailable" in hint


def test_chat_queue_memory_snapshot_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_MODAL_CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED", raising=False)
    module = _load_module()

    assert module.CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED is False


def test_feishu_snapshot_defaults_are_cost_first(monkeypatch):
    monkeypatch.delenv("HERMES_MODAL_WEB_APP_MEMORY_SNAPSHOT_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_MODAL_FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_MODAL_FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED", raising=False)
    module = _load_module()

    assert module.WEB_APP_MEMORY_SNAPSHOT_ENABLED is False
    assert module.FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED is False
    assert module.FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED is False


def test_feishu_model_registry_heartbeat_impl_respects_next_due(monkeypatch):
    module = _load_module()
    future_due = int(time.time()) + 120
    monkeypatch.setenv("FEISHU_MODEL_REGISTRY_MIRROR_ENABLED", "true")
    monkeypatch.setenv("FEISHU_BITABLE_APP_TOKEN", "app_token_123")
    monkeypatch.setenv("FEISHU_BITABLE_TABLE_ID", "tbl_123")
    monkeypatch.setattr(
        module,
        "_load_feishu_sync_state",
        lambda: {"next_due_at": future_due},
    )

    payload = module._feishu_model_registry_heartbeat_impl()

    assert payload["status"] == "skipped"
    assert payload["reason"] == "not_due"
    assert payload["next_due_at"] == future_due


def test_maintenance_heartbeat_disabled_by_default(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HERMES_MODAL_MAINTENANCE_HEARTBEAT_ENABLED", raising=False)

    assert module._maintenance_heartbeat_is_enabled() is False


def test_maintenance_heartbeat_impl_combines_cron_and_feishu(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_cron_scheduler_tick_impl",
        lambda *, enqueue_limit, worker_count: {
            "status": "ok",
            "enqueue_limit": enqueue_limit,
            "worker_count": worker_count,
        },
    )
    monkeypatch.setattr(
        module,
        "_feishu_model_registry_heartbeat_impl",
        lambda: {"status": "skipped", "reason": "mirror_disabled"},
    )

    payload = module._maintenance_heartbeat_impl(enqueue_limit=3, worker_count=2)

    assert payload["status"] == "ok"
    assert payload["cron"]["enqueue_limit"] == 3
    assert payload["cron"]["worker_count"] == 2
    assert payload["feishu_registry"]["reason"] == "mirror_disabled"


def test_bootstrap_chat_queue_worker_context_preloads_local_state(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_prepare_runtime_environment", lambda: None)
    monkeypatch.setattr(module, "_load_routing_state", lambda: {"refreshed_at": 123456})

    fake_tools_module = types.SimpleNamespace(
        load_feishu_model_registry=lambda force_refresh=False: {
            "source": "routing_state",
            "entries": [{"provider": "openrouter", "model": "openrouter/free"}],
        }
    )
    monkeypatch.setitem(sys.modules, "tools.feishu_api", fake_tools_module)

    payload = module._bootstrap_chat_queue_worker_context()

    assert payload["worker_boot_id"]
    assert payload["worker_started_at"] > 0
    assert payload["runtime_prepare_elapsed_ms"] >= 0
    assert payload["enter_elapsed_ms"] >= payload["runtime_prepare_elapsed_ms"]
    assert payload["routing_state_refreshed_at"] == 123456
    assert payload["model_registry_entry_count"] == 1
    assert payload["model_registry_source"] == "routing_state"
    assert payload["container_reused"] is False


def test_cron_scheduler_tick_batches_jobs_per_worker(monkeypatch):
    module = _load_module()
    spawned = []

    monkeypatch.setattr(
        module,
        "_enqueue_due_cron_jobs_impl",
        lambda limit: {"status": "ok", "queue_depth": 5, "enqueued_count": 5},
    )
    monkeypatch.setattr(module, "_safe_cron_queue_depth", lambda: 5)
    monkeypatch.setattr(module, "modal", object())
    monkeypatch.setattr(
        module,
        "process_cron_queue",
        types.SimpleNamespace(spawn=lambda **kwargs: spawned.append(kwargs)),
        raising=False,
    )

    payload = module._cron_scheduler_tick_impl(enqueue_limit=8, worker_count=2)

    assert payload["spawned_workers"] == 2
    assert payload["jobs_per_worker"] == 3
    assert spawned == [{"max_jobs": 3}, {"max_jobs": 3}]


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
    assert "personality_picker" in payload["supported_event_keys"]
    assert "skill_combo_picker" in payload["supported_event_keys"]
    assert "command_center" in payload["supported_event_keys"]
    assert payload["menu_items"][0]["event_key"] == "model_picker"


def test_build_feishu_local_operator_cards_expose_personality_and_combo_actions():
    module = _load_module()

    personality_card = module._build_feishu_personality_card()
    combo_card = module._build_feishu_skill_combo_card()
    command_card = module._build_feishu_command_center_card()

    personality_actions = [
        action.get("value", {})
        for element in personality_card["elements"]
        if element.get("tag") == "action"
        for action in element.get("actions", [])
    ]
    combo_actions = [
        action.get("value", {})
        for element in combo_card["elements"]
        if element.get("tag") == "action"
        for action in element.get("actions", [])
    ]
    command_actions = [
        action.get("value", {})
        for element in command_card["elements"]
        if element.get("tag") == "action"
        for action in element.get("actions", [])
    ]

    assert any(value.get("hermes_action") == "personality_set" for value in personality_actions)
    assert any(value.get("hermes_action") == "skill_combo_apply" for value in combo_actions)
    assert any(value.get("hermes_action") == "command_run" for value in command_actions)
    assert any(value.get("hermes_action") == "registry_close_card" for value in personality_actions)
    assert any(value.get("hermes_action") == "registry_close_card" for value in combo_actions)
    assert any(value.get("hermes_action") == "registry_close_card" for value in command_actions)
    assert any(value.get("command_text") == "/help" for value in command_actions)
    assert any("`/browser [connect|disconnect|status]`" in element.get("content", "") for element in command_card["elements"] if element.get("tag") == "markdown")


def test_runtime_api_config_defaults_to_cloudflare_gateway_for_openrouter(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HERMES_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    provider, base_url, api_key = module._pick_runtime_api_config()

    assert provider == "openrouter"
    assert base_url == "https://gateway.ai.cloudflare.com/v1/d1215a30b84b673ef0367010b0e78c10/affiliate-manager/compat"
    assert api_key == "sk-or-test"


def test_runtime_api_config_prefers_cloudflare_gateway_when_enabled(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HERMES_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_INFERENCE_USE_CLOUDFLARE_AI_GATEWAY", "true")
    monkeypatch.setenv(
        "CLOUDFLARE_AI_GATEWAY_BASE_URL",
        "https://gateway.ai.cloudflare.com/v1/acct/gateway/compat/chat/completions",
    )
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")

    provider, base_url, api_key = module._pick_runtime_api_config()

    assert provider == "openrouter"
    assert base_url == "https://gateway.ai.cloudflare.com/v1/acct/gateway/compat"
    assert api_key == "cf-token"


def test_runtime_api_config_ignores_direct_openrouter_base_url_for_gateway_providers(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HERMES_PROVIDER", raising=False)
    monkeypatch.setenv("HERMES_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv(
        "CLOUDFLARE_AI_GATEWAY_BASE_URL",
        "https://gateway.ai.cloudflare.com/v1/acct/gateway/compat/chat/completions",
    )

    provider, base_url, api_key = module._pick_runtime_api_config()

    assert provider == "openrouter"
    assert base_url == "https://gateway.ai.cloudflare.com/v1/acct/gateway/compat"
    assert api_key == "sk-or-test"


def test_runtime_api_config_routes_remote_providers_through_cloudflare_gateway(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("HERMES_PROVIDER", "anthropic")
    monkeypatch.setenv("HERMES_BASE_URL", "https://api.anthropic.com")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv(
        "CLOUDFLARE_AI_GATEWAY_BASE_URL",
        "https://gateway.ai.cloudflare.com/v1/acct/gateway/compat/chat/completions",
    )
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    provider, base_url, api_key = module._pick_runtime_api_config()

    assert provider == "anthropic"
    assert base_url == "https://gateway.ai.cloudflare.com/v1/acct/gateway/compat"
    assert api_key == "cf-token"


def test_runtime_api_config_keeps_bypass_providers_direct(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("HERMES_PROVIDER", "custom")
    monkeypatch.setenv("HERMES_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("HERMES_API_KEY", "custom-key")
    monkeypatch.setenv(
        "CLOUDFLARE_AI_GATEWAY_BASE_URL",
        "https://gateway.ai.cloudflare.com/v1/acct/gateway/compat/chat/completions",
    )
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")

    provider, base_url, api_key = module._pick_runtime_api_config()

    assert provider == "custom"
    assert base_url == "https://example.com/v1"
    assert api_key == "custom-key"


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


def test_candidate_routes_from_state_prefers_runtime_binding(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_resolve_provider_runtime_binding",
        lambda provider_name: {
            "provider": provider_name,
            "base_url": "https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
            "api_key": "cf-token",
            "api_mode": "chat_completions",
        } if provider_name == "openrouter" else None,
    )

    routes = module._candidate_routes_from_state(
        {
            "providers": {
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "candidates": ["openrouter/free"],
                }
            }
        }
    )

    assert routes[0]["provider"] == "openrouter"
    assert routes[0]["base_url"] == "https://gateway.ai.cloudflare.com/v1/acct/gw/compat"
    assert routes[0]["api_key"] == "cf-token"


def test_route_from_settings_prefers_runtime_binding_for_remote_providers(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_resolve_provider_runtime_binding",
        lambda provider_name: {
            "provider": provider_name,
            "base_url": "https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
            "api_key": "cf-token",
            "api_mode": "chat_completions",
        } if provider_name == "anthropic" else None,
    )
    settings = types.SimpleNamespace(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_key="anthropic-key",
        model="claude-3-7-sonnet",
    )

    route = module._route_from_settings(settings, "claude-3-7-sonnet")

    assert route["provider"] == "anthropic"
    assert route["base_url"] == "https://gateway.ai.cloudflare.com/v1/acct/gw/compat"
    assert route["api_key"] == "cf-token"


def test_candidate_routes_from_state_skips_direct_provider_routes_without_runtime_binding(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_resolve_provider_runtime_binding", lambda provider_name: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    routes = module._candidate_routes_from_state(
        {
            "providers": {
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "candidates": ["openrouter/free"],
                }
            }
        }
    )

    assert routes == []


def test_hydrate_route_from_lease_prefers_runtime_binding(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_resolve_provider_runtime_binding",
        lambda provider_name: {
            "provider": provider_name,
            "base_url": "https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
            "api_key": "cf-token",
            "api_mode": "chat_completions",
        } if provider_name == "nvidia" else None,
    )
    settings = types.SimpleNamespace(provider="openrouter", api_key="sk-or-test")

    hydrated = module._hydrate_route_from_lease(
        settings,
        {
            "provider": "nvidia",
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "lease_expires_at": 9999999999,
        },
    )

    assert hydrated["provider"] == "nvidia"
    assert hydrated["base_url"] == "https://gateway.ai.cloudflare.com/v1/acct/gw/compat"
    assert hydrated["api_key"] == "cf-token"


def test_hydrate_route_from_lease_prefers_runtime_binding_for_gateway_remote_provider(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_resolve_provider_runtime_binding",
        lambda provider_name: {
            "provider": provider_name,
            "base_url": "https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
            "api_key": "cf-token",
            "api_mode": "chat_completions",
        } if provider_name == "anthropic" else None,
    )
    settings = types.SimpleNamespace(provider="openrouter", api_key="sk-or-test")

    hydrated = module._hydrate_route_from_lease(
        settings,
        {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet",
            "base_url": "https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
            "lease_expires_at": 9999999999,
        },
    )

    assert hydrated["provider"] == "anthropic"
    assert hydrated["base_url"] == "https://gateway.ai.cloudflare.com/v1/acct/gw/compat"
    assert hydrated["api_key"] == "cf-token"


def test_route_api_key_source_uses_cloudflare_token_for_gateway_routes():
    module = _load_module()

    assert (
        module._route_api_key_source(
            "anthropic",
            "https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
        )
        == "CLOUDFLARE_API_TOKEN"
    )


def test_probe_provider_request_metadata_impl_uses_runtime_binding_and_model_alias(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_resolve_provider_runtime_binding",
        lambda provider_name: {
            "provider": provider_name,
            "base_url": "https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
            "api_key": "cf-token",
            "api_mode": "chat_completions",
        } if provider_name == "openrouter" else None,
    )

    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run_conversation(self, *_args, **_kwargs):
            return {
                "final_response": "ok",
                "completed": True,
                "provider_usage": {"response_model": "openrouter/free"},
                "provider_usage_totals": {},
            }

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))

    result = module._probe_provider_request_metadata_impl(
        provider_name="openrouter",
        model="openrouter/free",
        prompt="reply with ok",
        max_tokens=32,
    )

    assert seen["base_url"] == "https://gateway.ai.cloudflare.com/v1/acct/gw/compat"
    assert seen["api_key"] == "cf-token"
    assert seen["provider"] == "openrouter"
    assert result["cloudflare_ai_gateway"] is True


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
    source.write_text("model:\n  default: free\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MODAL_CONFIG_SOURCE", str(source))
    monkeypatch.setenv("HERMES_MODAL_SYNC_CONFIG", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setattr(
        module,
        "_resolve_provider_runtime_binding",
        lambda provider_name: {
            "provider": provider_name,
            "base_url": "https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
            "api_key": "cf-token" if provider_name == "openrouter" else "nvapi-test",
            "api_mode": "chat_completions",
        } if provider_name in {"openrouter", "nvidia"} else None,
    )

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
    assert "base_url: https://gateway.ai.cloudflare.com/v1/acct/gw/compat" in target_text
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
                    "provider_request_id": "req_456",
                    "provider_http_status_code": 202,
                    "provider_async_poll_supported": True,
                    "provider_async_request_id": "nv_req_789",
                    "provider_async_poll_url": "https://integrate.api.nvidia.com/v1/status/nv_req_789",
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
    assert result["provider_usage"]["provider_async_request_id"] == "nv_req_789"
    assert result["provider_usage"]["provider_async_poll_supported"] is True


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


def _obsolete_test_validate_feishu_workbench_call_retries_with_stronger_model(monkeypatch, tmp_path):
    pass


def _obsolete_test_validate_feishu_workbench_call_continues_after_provider_403_exception(monkeypatch, tmp_path):
    pass


def _obsolete_test_validate_feishu_workbench_call_prefers_explicit_mcp_toolset(monkeypatch, tmp_path):
    pass

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
        feishu_internal_bearer_token=None,
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
        feishu_internal_bearer_token=None,
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


def test_spawn_feishu_event_handoff_async_schedules_worker(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_EVENTS_PATH = module.DATA_ROOT / "feishu_events.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module._ensure_runtime_dirs()
    handled = {"spawn_payload": None}
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

    async def _fake_spawn_aio(*, payload, warmup_only=False, warmup_context=None):
        handled["spawn_payload"] = payload
        handled["warmup_only"] = warmup_only
        handled["warmup_context"] = warmup_context

    monkeypatch.setattr(
        module,
        "process_feishu_event",
        types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_fake_spawn_aio)),
        raising=False,
    )

    result = asyncio.run(
        module._spawn_feishu_event_handoff_async(
            payload=payload,
            context={"partition": "feishu:chat_light:unknown", "lane": "chat_light"},
        )
    )

    assert result["status"] == "scheduled"
    assert result["reason"] == "process_feishu_event_spawned"
    assert handled["spawn_payload"]["event"]["message"]["message_id"] == "om_sync"
    assert handled["warmup_only"] is False
    assert handled["warmup_context"] is None


def test_feishu_message_webhook_uses_inline_enqueue_and_direct_worker_spawn(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_EVENTS_PATH = module.DATA_ROOT / "feishu_events.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module._ensure_runtime_dirs()
    handled = {"enqueued": None, "spawned": None, "ack_reaction_inline": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": "evt_sync_fallback",
            "token": "verify-token",
        },
        "event": {"message": {"message_id": "om_sync_fallback"}},
    }

    async def _fake_parse(_request):
        return object(), payload

    async def _fake_enqueue_async(*, platform, partition, payload, metadata, include_queue_depth=True):
        handled["enqueued"] = {
            "platform": platform,
            "partition": partition,
            "payload": payload,
            "metadata": metadata,
            "include_queue_depth": include_queue_depth,
        }
        return {"status": "enqueued", "partition": partition, "queue_depth": None, **metadata}

    async def _fake_spawn_async(**kwargs):
        handled["spawned"] = kwargs
        return {"status": "scheduled", **kwargs}

    async def _fake_inline_ack_reaction(*, payload, request_started_at_ms=None):
        handled["ack_reaction_inline"] = request_started_at_ms is not None
        return {
            "status": "ok",
            "reason": "inline",
            "total_elapsed_ms": 12,
            "message_id": payload["event"]["message"]["message_id"],
        }

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)
    monkeypatch.setattr(module, "DEFAULT_FEISHU_ACK_REACTION_MODE", "inline")
    monkeypatch.setattr(
        module,
        "process_feishu_event",
        types.SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(module, "_enqueue_chat_event_async", _fake_enqueue_async)
    monkeypatch.setattr(module, "_spawn_chat_queue_worker_optimistic_async", _fake_spawn_async)
    monkeypatch.setattr(module, "_add_feishu_ack_reaction_inline_async", _fake_inline_ack_reaction)

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"code": 0, "msg": "accepted"}
    assert handled["ack_reaction_inline"] is True
    assert handled["enqueued"]["partition"] == "feishu:chat_light:unknown"
    assert handled["spawned"]["partition"] == "feishu:chat_light:unknown"


def test_spawn_feishu_ack_reaction_async_requires_message_id(monkeypatch):
    module = _load_module()

    async def _run():
        return await module._spawn_feishu_ack_reaction_async(
            payload={
                "header": {"event_type": "im.message.receive_v1", "event_id": "evt_missing_msg"},
                "event": {"message": {}},
            },
        )

    result = asyncio.run(_run())

    assert result["status"] == "skipped"
    assert result["reason"] == "missing_message_id"


def test_add_feishu_ack_reaction_from_payload_records_success(monkeypatch):
    module = _load_module()
    traces = []
    fake_client = types.SimpleNamespace(
        request_json=lambda method, path, json_body=None, **kwargs: {
            "reaction_id": "reaction-om_ack_bg",
            "method": method,
            "path": path,
            "json_body": json_body,
            "kwargs": kwargs,
        }
    )

    monkeypatch.setitem(
        sys.modules,
        "tools.feishu_api",
        types.SimpleNamespace(build_feishu_client=lambda **kwargs: fake_client),
    )
    monkeypatch.setattr(module, "_append_feishu_trace", lambda stage, payload, **extra: traces.append((stage, extra)))

    result = module._add_feishu_ack_reaction_from_payload(
        {
            "header": {"event_type": "im.message.receive_v1", "event_id": "evt_ack_bg"},
            "event": {"message": {"message_id": "om_ack_bg", "chat_id": "oc_chat"}},
        },
        request_started_at_ms=int(time.time() * 1000),
        worker_context={"worker_boot_id": "ack-worker-1", "container_reused": False},
    )

    assert result["status"] == "ok"
    assert result["message_id"] == "om_ack_bg"
    assert result["reaction_id"] == "reaction-om_ack_bg"
    assert result["worker_boot_id"] == "ack-worker-1"
    assert any(stage == "webhook.ack_reaction" for stage, _extra in traces)


def test_spawn_feishu_event_handoff_async_times_out(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS", 0.01)

    async def _fake_spawn_aio(**kwargs):
        await asyncio.sleep(0.2)

    monkeypatch.setattr(
        module,
        "process_feishu_event",
        types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_fake_spawn_aio)),
        raising=False,
    )

    result = asyncio.run(
        module._spawn_feishu_event_handoff_async(
            payload={"header": {"event_type": "im.message.receive_v1", "event_id": "evt_timeout"}},
            context={"partition": "feishu:chat_light:oc_timeout", "lane": "chat_light"},
        )
    )

    assert result["status"] == "timeout"
    assert result["reason"] == "process_feishu_event_spawn_timeout"
    assert result["handoff_schedule_wait_elapsed_ms"] >= result["handoff_wait_elapsed_ms"]


def test_feishu_webhook_fast_command_bypasses_chat_queue(monkeypatch):
    module = _load_module()
    handled = {"dispatched": None, "enqueued": False, "spawned": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": "evt_fast_model",
            "token": "verify-token",
        },
        "event": {
            "message": {
                "message_id": "om_fast_model",
                "message_type": "text",
                "content": json.dumps({"text": "/model"}),
            }
        },
    }

    async def _fake_parse(_request):
        return object(), payload

    async def _fake_dispatch(raw_payload, await_background_tasks=False):
        handled["dispatched"] = {
            "payload": raw_payload,
            "await_background_tasks": await_background_tasks,
        }

    async def _fake_enqueue(**kwargs):
        handled["enqueued"] = True
        return {"status": "enqueued", "queue_depth": 1}

    async def _fake_spawn(**kwargs):
        handled["spawned"] = kwargs

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_dispatch_feishu_payload", _fake_dispatch)
    monkeypatch.setattr(module, "_enqueue_chat_event_async", _fake_enqueue)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)
    monkeypatch.setattr(
        module,
        "process_chat_queue",
        types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_fake_spawn)),
        raising=False,
    )

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"code": 0, "msg": "accepted"}
    assert handled["dispatched"]["payload"]["event"]["message"]["message_id"] == "om_fast_model"
    assert handled["dispatched"]["await_background_tasks"] is True
    assert handled["enqueued"] is False
    assert handled["spawned"] is False


def test_feishu_webhook_bot_p2p_chat_entered_warms_chat_worker_without_queue(monkeypatch):
    module = _load_module()
    handled = {"warmup": None, "enqueued": False, "spawned": False, "dispatched": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "im.chat.access_event.bot_p2p_chat_entered_v1",
            "event_id": "evt_warmup_1",
            "token": "verify-token",
        },
        "event": {
            "operator": {
                "operator_id": {
                    "open_id": "ou_entered_123",
                }
            },
            "context": {
                "open_chat_id": "oc_entered_123",
            },
        },
    }

    async def _fake_parse(_request):
        return object(), payload

    async def _fake_warmup(**kwargs):
        handled["warmup"] = kwargs
        return {"status": "scheduled", "spawned": True, **kwargs}

    async def _fake_enqueue(**kwargs):
        handled["enqueued"] = kwargs
        return {"status": "enqueued", "queue_depth": 1}

    async def _fake_spawn(**kwargs):
        handled["spawned"] = kwargs
        return {"status": "scheduled", **kwargs}

    async def _fake_dispatch(*_args, **_kwargs):
        handled["dispatched"] = True

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_spawn_feishu_ingress_warmup_async", _fake_warmup)
    monkeypatch.setattr(module, "_enqueue_chat_event_async", _fake_enqueue)
    monkeypatch.setattr(module, "_spawn_chat_queue_worker_async", _fake_spawn)
    monkeypatch.setattr(module, "_dispatch_feishu_payload", _fake_dispatch)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"code": 0, "msg": "accepted"}
    assert handled["warmup"]["warmup_context"]["partition"] == "feishu:chat_light:oc_entered_123"
    assert handled["warmup"]["warmup_context"]["actor_id"] == "ou_entered_123"
    assert handled["warmup"]["payload"]["header"]["event_id"] == "evt_warmup_1"


def test_coalesce_feishu_chat_queue_items_keeps_latest_message_for_partition(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    items = [
        {
            "platform": "feishu",
            "partition": "feishu:chat_light:oc_chat",
            "metadata": {"event_type": "im.message.receive_v1", "event_id": "evt_old"},
            "enqueued_at_ms": 999000,
        },
        {
            "platform": "feishu",
            "partition": "feishu:chat_light:oc_chat",
            "metadata": {"event_type": "im.message.receive_v1", "event_id": "evt_new"},
            "enqueued_at_ms": 1000000,
        },
    ]

    result = module._coalesce_feishu_chat_queue_items(items)

    assert len(result) == 1
    assert result[0]["metadata"]["event_id"] == "evt_new"


def test_coalesce_feishu_chat_queue_items_skips_stale_message(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(module, "DEFAULT_FEISHU_MESSAGE_QUEUE_MAX_AGE_SECONDS", 60)
    items = [
        {
            "platform": "feishu",
            "partition": "feishu:chat_light:oc_chat",
            "metadata": {"event_type": "im.message.receive_v1", "event_id": "evt_stale"},
            "enqueued_at_ms": 800000,
        }
    ]

    result = module._coalesce_feishu_chat_queue_items(items)

    assert result == []


def test_resolve_feishu_message_ingress_strategy_prefers_supported_env(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY", "spawn_process_feishu_message_inline")
    payload = {"header": {"event_id": "evt_1"}}
    context = {"event_id": "evt_1", "chat_id": "oc_chat"}

    result = module._resolve_feishu_message_ingress_strategy(payload, context)

    assert result == "spawn_process_feishu_message_inline"


def test_resolve_feishu_message_ingress_strategy_prefers_inline_for_light_p2p_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_FEISHU_MESSAGE_INGRESS_STRATEGY", raising=False)
    module = _load_module()
    monkeypatch.setattr(module, "DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY", "inline_enqueue_spawn")
    payload = {
        "header": {"event_id": "evt_light", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {
                "chat_id": "oc_chat",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": "测速"}),
            }
        },
    }
    context = {"event_id": "evt_light", "chat_id": "oc_chat", "lane": "chat_light", "event_type": "im.message.receive_v1"}

    result = module._resolve_feishu_message_ingress_strategy(payload, context)

    assert result == "spawn_process_feishu_message_inline"


def test_resolve_feishu_message_ingress_strategy_respects_explicit_queue_env(monkeypatch):
    monkeypatch.setenv("HERMES_FEISHU_MESSAGE_INGRESS_STRATEGY", "inline_enqueue_spawn")
    module = _load_module()
    payload = {
        "header": {"event_id": "evt_forced_queue", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {
                "chat_id": "oc_chat",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": "测速"}),
            }
        },
    }
    context = {
        "event_id": "evt_forced_queue",
        "chat_id": "oc_chat",
        "lane": "chat_light",
        "event_type": "im.message.receive_v1",
    }

    result = module._resolve_feishu_message_ingress_strategy(payload, context)

    assert result == "inline_enqueue_spawn"


def test_resolve_feishu_message_ingress_strategy_legacy_env_still_allows_inline_for_light_p2p(monkeypatch):
    monkeypatch.setenv("HERMES_FEISHU_MESSAGE_INGRESS_STRATEGY", "spawn_process_feishu_event")
    module = _load_module()
    payload = {
        "header": {"event_id": "evt_legacy_inline", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {
                "chat_id": "oc_chat",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": "测速"}),
            }
        },
    }
    context = {
        "event_id": "evt_legacy_inline",
        "chat_id": "oc_chat",
        "lane": "chat_light",
        "event_type": "im.message.receive_v1",
    }

    result = module._resolve_feishu_message_ingress_strategy(payload, context)

    assert result == "spawn_process_feishu_message_inline"


def test_resolve_feishu_message_ingress_strategy_aliases_legacy_spawn_to_inline(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY", "spawn_process_feishu_event")
    payload = {"header": {"event_id": "evt_legacy"}}
    context = {"event_id": "evt_legacy", "chat_id": "oc_chat"}

    result = module._resolve_feishu_message_ingress_strategy(payload, context)

    assert result == "inline_enqueue_spawn"


def test_resolve_feishu_message_ingress_strategy_falls_back_to_final_strategy(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY", "unsupported-strategy")
    payload = {"header": {"event_id": "evt_ab"}}
    context = {"event_id": "evt_ab", "chat_id": "oc_chat"}

    result = module._resolve_feishu_message_ingress_strategy(payload, context)

    assert result == "inline_enqueue_spawn"


def test_spawn_chat_queue_worker_async_skips_duplicate_partition(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_EVENTS_PATH = module.DATA_ROOT / "feishu_events.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    spawned = []

    async def _fake_spawn_aio(**kwargs):
        spawned.append(kwargs)

    monkeypatch.setattr(
        module,
        "process_chat_queue",
        types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_fake_spawn_aio)),
        raising=False,
    )

    first = asyncio.run(
        module._spawn_chat_queue_worker_async(
            platform="feishu",
            partition="feishu:test-partition",
            max_items=2,
        )
    )
    second = asyncio.run(
        module._spawn_chat_queue_worker_async(
            platform="feishu",
            partition="feishu:test-partition",
            max_items=2,
        )
    )

    assert first["status"] == "scheduled"
    assert second["status"] == "skipped"
    assert second["reason"] in {"scheduled", "claimed"}
    assert len(spawned) == 1


def test_spawn_chat_queue_worker_optimistic_async_skips_recent_active_claim(monkeypatch):
    module = _load_module()
    spawned = []

    async def _fake_spawn_aio(**kwargs):
        spawned.append(kwargs)

    monkeypatch.setattr(
        module,
        "process_chat_queue",
        types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_fake_spawn_aio)),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_peek_chat_partition_claim_async",
        lambda *args, **kwargs: _async_return(
            {
                "claim_token": "claim-active-1",
                "claimed_at": int(time.time()),
                "status": "claimed",
                "platform": "feishu",
            }
        ),
    )
    monkeypatch.setattr(module, "DEFAULT_CHAT_QUEUE_ACTIVE_CLAIM_SKIP_SECONDS", 30)

    result = asyncio.run(
        module._spawn_chat_queue_worker_optimistic_async(
            platform="feishu",
            partition="feishu:test-active-claim",
            max_items=2,
        )
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "active_claimed"
    assert result["claim_token"] == "claim-active-1"
    assert spawned == []


def test_peek_chat_partition_claim_async_refreshes_volume_on_miss(monkeypatch):
    module = _load_module()
    loads = {"count": 0}
    reloads = {"count": 0}

    def _fake_load_claims():
        loads["count"] += 1
        if loads["count"] == 1:
            return {}
        return {
            "feishu:test-refresh": {
                "claim_token": "claim-refresh-1",
                "claimed_at": int(time.time()),
                "status": "claimed",
                "platform": "feishu",
            }
        }

    async def _fake_sync_modal_volume_async(*, reload=False, commit=False):
        if reload:
            reloads["count"] += 1

    monkeypatch.setattr(module, "_should_reload_modal_volume_for_claims", lambda kind: False)
    monkeypatch.setattr(module, "_load_chat_queue_claims", _fake_load_claims)
    monkeypatch.setattr(module, "_prune_chat_queue_claims", lambda claims, ttl_seconds=None: claims)
    monkeypatch.setattr(module, "_sync_modal_volume_async", _fake_sync_modal_volume_async)

    result = asyncio.run(
        module._peek_chat_partition_claim_async(
            "feishu:test-refresh",
            refresh_on_miss=True,
        )
    )

    assert result["claim_token"] == "claim-refresh-1"
    assert reloads["count"] == 1
    assert loads["count"] == 2


def test_spawn_chat_queue_worker_optimistic_async_skips_recent_spawn_gate(monkeypatch):
    module = _load_module()
    spawned = []

    async def _fake_spawn_aio(**kwargs):
        spawned.append(kwargs)

    monkeypatch.setattr(
        module,
        "process_chat_queue",
        types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_fake_spawn_aio)),
        raising=False,
    )
    monkeypatch.setattr(module, "DEFAULT_FEISHU_RECENT_SPAWN_SKIP_SECONDS", 12.0)
    monkeypatch.setattr(module, "_RECENT_CHAT_WORKER_SPAWNS", {"feishu:test-recent-gate": time.monotonic()})

    result = asyncio.run(
        module._spawn_chat_queue_worker_optimistic_async(
            platform="feishu",
            partition="feishu:test-recent-gate",
            max_items=2,
        )
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "recent_spawn_gate"
    assert spawned == []


def test_spawn_chat_queue_warmup_async_uses_waiting_worker(monkeypatch):
    module = _load_module()
    captured = {}

    async def _fake_spawn(**kwargs):
        captured.update(kwargs)
        return {"status": "scheduled", "spawned": True, **kwargs}

    monkeypatch.setattr(module, "_spawn_chat_queue_worker_async", _fake_spawn)

    result = asyncio.run(
        module._spawn_chat_queue_warmup_async(
            platform="feishu",
            partition="feishu:oc_waiting_123",
            reason="bot_p2p_chat_entered",
            metadata={
                "event_id": "evt_waiting_1",
                "event_type": "im.chat.access_event.bot_p2p_chat_entered_v1",
                "actor_id": "ou_waiting_123",
                "chat_id": "oc_waiting_123",
                "started_at_ms": "1234567890",
            },
        )
    )

    assert result["status"] == "scheduled"
    assert captured["partition"] == "feishu:oc_waiting_123"
    assert captured["max_items"] == 1
    assert captured["warmup_wait_seconds"] == module.DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS
    assert captured["warmup_metadata"]["actor_id"] == "ou_waiting_123"


def test_process_chat_queue_warmup_wait_hits_first_message(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_EVENTS_PATH = module.DATA_ROOT / "feishu_events.json"
    module.FEISHU_TRACE_PATH = module.DATA_ROOT / "feishu_trace.jsonl"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module.CHAT_QUEUE_WARMUPS_PATH = module.DATA_ROOT / "chat_queue_warmups.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setattr(module, "_sync_modal_volume", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_safe_chat_queue_depth", lambda: 0)
    monkeypatch.setattr(module, "_claim_chat_partition", lambda *args, **kwargs: (True, "claim-1"))
    monkeypatch.setattr(module, "_release_chat_partition_claim", lambda *args, **kwargs: None)

    class FakeQueue:
        def __init__(self):
            self.calls = []
            self._returned_once = False

        def get_many(self, max_items, block=False, timeout=None, partition=None):
            self.calls.append(
                {
                    "max_items": max_items,
                    "block": block,
                    "timeout": timeout,
                    "partition": partition,
                }
            )
            if self._returned_once:
                return []
            self._returned_once = True
            return [
                {
                    "platform": "feishu",
                    "partition": partition,
                    "payload": {"header": {"event_id": "evt_msg_1", "event_type": "im.message.receive_v1"}},
                    "metadata": {"event_id": "evt_msg_1", "event_type": "im.message.receive_v1"},
                    "enqueued_at_ms": 1234567900,
                }
            ]

    fake_queue = FakeQueue()
    monkeypatch.setattr(module, "_get_chat_queue", lambda: fake_queue)

    async def _fake_process(items, worker_context=None, runtime_prepared=False):
        return [
            {
                "status": "ok",
                "items": len(items),
                "warmup_status": dict(worker_context or {}).get("warmup_status"),
                "warmup_same_container": dict(worker_context or {}).get("warmup_same_container"),
            }
        ]

    monkeypatch.setattr(module, "_process_chat_queue_items_async", _fake_process)

    result = module._process_chat_queue_impl(
        platform="feishu",
        partition="feishu:oc_waiting_123",
        max_items=1,
        claim_token="claim-1",
        worker_context={"worker_boot_id": "boot-waiting"},
        runtime_prepared=True,
        warmup_wait_seconds=9,
        warmup_metadata={
            "event_id": "evt_waiting_1",
            "event_type": "im.chat.access_event.bot_p2p_chat_entered_v1",
            "actor_id": "ou_waiting_123",
            "chat_id": "oc_waiting_123",
            "started_at_ms": "1234567890",
        },
    )

    assert result["status"] == "ok"
    assert result["processed_count"] == 1
    assert result["warmup_status"] == "hit"
    assert result["warmup_same_container"] is True
    assert fake_queue.calls[0]["timeout"] == 9.0


def test_spawn_chat_queue_worker_async_uses_async_volume_sync(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_EVENTS_PATH = module.DATA_ROOT / "feishu_events.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module.CHAT_QUEUE_CLAIMS_PATH = module.DATA_ROOT / "chat_queue_claims.json"
    module._ensure_runtime_dirs()

    sync_calls = []
    async_calls = []
    spawned = []

    async def _fake_spawn_aio(**kwargs):
        spawned.append(kwargs)

    async def _fake_sync_async(*, reload=False, commit=False):
        async_calls.append((reload, commit))

    def _fail_sync(*, reload=False, commit=False):
        sync_calls.append((reload, commit))
        raise AssertionError("sync volume helper should not be used in async path")

    monkeypatch.setattr(module, "_sync_modal_volume", _fail_sync)
    monkeypatch.setattr(module, "_sync_modal_volume_async", _fake_sync_async)
    monkeypatch.setattr(
        module,
        "process_chat_queue",
        types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_fake_spawn_aio)),
        raising=False,
    )

    result = asyncio.run(
        module._spawn_chat_queue_worker_async(
            platform="feishu",
            partition="feishu:test-async-volume",
            max_items=2,
        )
    )

    assert result["status"] == "scheduled"
    assert spawned and spawned[0]["partition"] == "feishu:test-async-volume"
    assert async_calls == [(False, True)]
    assert sync_calls == []


def test_telegram_webhook_fast_command_bypasses_chat_queue(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABCdefGHIjklMNOpqrSTUvwxYZ")
    handled = {"dispatched": None, "enqueued": False, "spawned": False}

    async def _fake_dispatch(update):
        handled["dispatched"] = update
        return {"status": "accepted", "kind": "command"}

    async def _fake_enqueue(**kwargs):
        handled["enqueued"] = True
        return {"status": "enqueued", "queue_depth": 1}

    async def _fake_spawn(**kwargs):
        handled["spawned"] = kwargs

    monkeypatch.setattr(module, "_dispatch_telegram_update", _fake_dispatch)
    monkeypatch.setattr(module, "_enqueue_chat_event_async", _fake_enqueue)
    monkeypatch.setattr(
        module,
        "_spawn_chat_queue_worker_async",
        _fake_spawn,
    )
    monkeypatch.setattr(module, "_mark_update_seen", lambda _update_id: True)

    client = TestClient(module.create_web_app())
    response = client.post(
        "/telegram/webhook",
        json={
            "update_id": 1001,
            "message": {
                "text": "/provider",
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 7, "username": "tester"},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "inline_fast_command"
    assert response.json()["command"] == "provider"
    assert handled["dispatched"]["message"]["text"] == "/provider"
    assert handled["enqueued"] is False
    assert handled["spawned"] is False


def test_feishu_menu_event_bypasses_chat_queue(monkeypatch):
    module = _load_module()
    handled = {"dispatched": None, "enqueued": False, "spawned": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "application.bot.menu_v6",
            "event_id": "evt_menu_inline",
            "token": "verify-token",
        },
        "event": {
            "event_key": "model_picker",
            "operator": {"operator_id": {"open_id": "ou_user"}},
        },
    }

    async def _fake_parse(_request):
        return object(), payload

    async def _fake_dispatch(raw_payload, await_background_tasks=False):
        handled["dispatched"] = {
            "payload": raw_payload,
            "await_background_tasks": await_background_tasks,
        }

    def _fake_spawn_background(**kwargs):
        handled["spawned"] = kwargs
        return True

    async def _fake_enqueue(**kwargs):
        handled["enqueued"] = True
        return {"status": "enqueued", "queue_depth": 1}

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_dispatch_feishu_payload", _fake_dispatch)
    monkeypatch.setattr(module, "_enqueue_chat_event_async", _fake_enqueue)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)
    monkeypatch.setattr(module, "_schedule_chat_queue_worker_background", _fake_spawn_background)

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"code": 0, "msg": "accepted"}
    assert handled["dispatched"]["payload"]["header"]["event_id"] == "evt_menu_inline"
    assert handled["dispatched"]["await_background_tasks"] is True
    assert handled["enqueued"] is False
    assert handled["spawned"] is False


def test_feishu_menu_event_prefers_local_registry_card(monkeypatch):
    module = _load_module()
    handled = {"local_menu": False, "dispatched": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "application.bot.menu_v6",
            "event_id": "evt_menu_local",
            "token": "verify-token",
        },
        "event": {
            "event_key": "provider_nvidia_featured",
            "operator": {"operator_id": {"open_id": "ou_user"}},
            "context": {"open_chat_id": "oc_chat"},
        },
    }

    async def _fake_parse(_request):
        return None, payload

    async def _fake_local_menu(_payload):
        handled["local_menu"] = True
        return True

    async def _fake_dispatch(*_args, **_kwargs):
        handled["dispatched"] = True

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_send_feishu_local_registry_menu_card", _fake_local_menu)
    monkeypatch.setattr(module, "_dispatch_feishu_payload", _fake_dispatch)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"code": 0, "msg": "accepted"}
    assert handled["local_menu"] is True
    assert handled["dispatched"] is False


def test_feishu_card_action_inline_ack_does_not_wait_for_background(monkeypatch):
    module = _load_module()
    handled = {"dispatched": None, "enqueued": False, "spawned": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "card.action.trigger",
            "event_id": "evt_card_inline",
            "token": "verify-token",
        },
        "event": {
            "context": {"open_chat_id": "oc_chat"},
            "operator": {"open_id": "ou_user"},
            "action": {"tag": "button", "value": {"hermes_action": "model_picker_cancel", "picker_id": "fp1"}},
        },
    }

    async def _fake_parse(_request):
        return object(), payload

    async def _fake_dispatch(raw_payload, await_background_tasks=False):
        handled["dispatched"] = {
            "payload": raw_payload,
            "await_background_tasks": await_background_tasks,
        }

    def _fake_spawn_background(**kwargs):
        handled["spawned"] = kwargs
        return True

    async def _fake_enqueue(**kwargs):
        handled["enqueued"] = True
        return {"status": "enqueued", "queue_depth": 1}

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_dispatch_feishu_payload", _fake_dispatch)
    monkeypatch.setattr(module, "_enqueue_chat_event_async", _fake_enqueue)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)
    monkeypatch.setattr(module, "_schedule_chat_queue_worker_background", _fake_spawn_background)

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"toast": {"type": "info", "content": "已收到，正在处理"}}
    assert handled["dispatched"] is None
    assert handled["enqueued"] is True
    assert handled["spawned"]["platform"] == "feishu"
    assert handled["spawned"]["partition"] == "feishu:control:oc_chat"


def test_feishu_registry_close_card_prefers_local_delete(monkeypatch):
    module = _load_module()
    handled = {"closed": False, "dispatched": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "card.action.trigger",
            "event_id": "evt_card_close",
            "token": "verify-token",
        },
        "event": {
            "context": {"open_chat_id": "oc_chat", "open_message_id": "om_close_123"},
            "operator": {"open_id": "ou_user"},
            "action": {"tag": "button", "value": {"hermes_action": "registry_close_card"}},
        },
    }

    async def _fake_parse(_request):
        return None, payload

    async def _fake_close(_payload):
        handled["closed"] = True
        return True

    async def _fake_dispatch(*_args, **_kwargs):
        handled["dispatched"] = True

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_close_feishu_card_from_payload", _fake_close)
    monkeypatch.setattr(module, "_dispatch_feishu_payload", _fake_dispatch)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"toast": {"type": "info", "content": "已关闭"}}
    assert handled["closed"] is True
    assert handled["dispatched"] is False


def test_feishu_registry_switch_card_queues_background_work(monkeypatch):
    module = _load_module()
    handled = {"queued": False, "dispatched": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "card.action.trigger",
            "event_id": "evt_card_switch",
            "token": "verify-token",
        },
        "event": {
            "context": {"open_chat_id": "oc_chat", "open_message_id": "om_switch_123"},
            "operator": {"open_id": "ou_user"},
            "action": {
                "tag": "button",
                "value": {
                    "hermes_action": "registry_switch_model",
                    "provider": "nvidia",
                    "model": "moonshotai/kimi-k2.5",
                },
            },
        },
    }

    async def _fake_parse(_request):
        return None, payload

    async def _fake_enqueue(_payload):
        handled["queued"] = True
        return {"status": "enqueued", "partition": "feishu:control:oc_chat", "queue_depth": 0, "lane": "control"}

    async def _fake_dispatch(*_args, **_kwargs):
        handled["dispatched"] = True

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_enqueue_feishu_card_action_for_background", _fake_enqueue)
    monkeypatch.setattr(module, "_dispatch_feishu_payload", _fake_dispatch)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"toast": {"type": "info", "content": "已收到，正在处理"}}
    assert handled["queued"] is True
    assert handled["dispatched"] is False


def test_feishu_message_read_event_fast_ack_bypasses_dispatch(monkeypatch):
    module = _load_module()
    module.DATA_ROOT = Path.cwd() / ".tmp-pytest" / "feishu-message-read"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_EVENTS_PATH = module.DATA_ROOT / "feishu_events.json"
    module.FEISHU_TRACE_PATH = module.DATA_ROOT / "feishu_trace.jsonl"
    module.HERMES_HOME_DIR = module.DATA_ROOT / "home"
    module._ensure_runtime_dirs()
    handled = {"dispatched": False, "enqueued": False}
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    payload = {
        "header": {
            "event_type": "im.message.message_read_v1",
            "event_id": "evt_read_fast",
            "token": "verify-token",
        },
        "event": {
            "reader": {
                "reader_id": {"open_id": "ou_reader_fast"},
                "read_time": "1712970001",
            },
            "message_id_list": ["om_read_1"],
        },
    }

    async def _fake_parse(_request):
        return None, payload

    async def _fake_dispatch(*_args, **_kwargs):
        handled["dispatched"] = True

    async def _fake_enqueue(**_kwargs):
        handled["enqueued"] = True
        return {"status": "enqueued"}

    monkeypatch.setattr(module, "_parse_feishu_webhook_request", _fake_parse)
    monkeypatch.setattr(module, "_dispatch_feishu_payload", _fake_dispatch)
    monkeypatch.setattr(module, "_enqueue_chat_event_async", _fake_enqueue)
    monkeypatch.setattr(module, "_mark_feishu_event_seen", lambda _event_id: True)

    client = TestClient(module.create_web_app())
    response = client.post("/feishu/webhook", json=payload)

    assert response.status_code == 200
    assert response.json() == {"code": 0, "msg": "accepted"}
    assert handled["dispatched"] is False
    assert handled["enqueued"] is False


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
    monkeypatch.setattr(module, "_encrypt_feishu_payload", lambda _key, _payload: "encrypted-payload")

    result = module._validate_feishu_webhook_impl()

    assert result["status"] == "ok"
    assert result["status_code"] == 200
    assert result["webhook_url"] == "https://example.com/feishu/webhook"
    assert result["response"] == {"challenge": "feishu-encrypted-selftest-ok"}
    assert captured["url"] == "https://example.com/feishu/webhook"
    assert "x-lark-signature" in captured["headers"]
    assert '"encrypt"' in captured["body"]


def test_validate_feishu_message_ingress_impl_builds_signed_message_request(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret-123")
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", "verify-token")
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", "encrypt-key")
    monkeypatch.setenv("HERMES_PUBLIC_BASE_URL", "https://example.com")

    captured = {}

    class FakeResponse:
        status_code = 200
        text = '{"code":0,"msg":"accepted"}'

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

    result = module._validate_feishu_message_ingress_impl(message_text="probe")
    body = json.loads(captured["body"])

    assert result["status"] == "ok"
    assert result["status_code"] == 200
    assert result["webhook_url"] == "https://example.com/feishu/webhook"
    assert result["response"] == {"code": 0, "msg": "accepted"}
    assert result["event_id"].startswith("evt_selftest_")
    assert result["message_id"].startswith("om_selftest_")
    assert body["header"]["event_type"] == "im.message.receive_v1"
    assert body["header"]["token"] == "verify-token"
    assert json.loads(body["event"]["message"]["content"]) == {"text": "probe"}
    assert "x-lark-signature" in captured["headers"]


def test_append_feishu_trace_includes_experiment_metadata(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_TRACE_PATH = module.DATA_ROOT / "feishu_trace.jsonl"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setattr(module, "APP_NAME", "hermes-agent-ab")
    monkeypatch.setattr(module, "WEB_APP_MEMORY_SNAPSHOT_ENABLED", True)
    monkeypatch.setattr(module, "FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED", True)
    monkeypatch.setattr(module, "FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED", False)
    monkeypatch.setattr(module, "CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED", False)
    monkeypatch.setenv("HERMES_FEISHU_PERF_EXPERIMENT_LABEL", "current")

    payload = {
        "header": {"event_type": "im.message.receive_v1", "event_id": "evt_meta_1"},
        "event": {"message": {"message_id": "om_meta_1", "chat_id": "oc_meta_1", "chat_type": "p2p"}},
    }

    module._append_feishu_trace("webhook.accepted", payload, ack_elapsed_ms=123)
    rows = module._read_feishu_trace(limit=10)

    assert rows[-1]["app_name"] == "hermes-agent-ab"
    assert rows[-1]["experiment_label"] == "current"
    assert rows[-1]["snapshot_profile"] == "web1_ingress1_ack0_chat0"
    assert rows[-1]["snapshot_flags"]["web_app_enabled"] is True
    assert rows[-1]["snapshot_flags"]["chat_queue_enabled"] is False


def test_append_feishu_trace_captures_route_observability_fields(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.FEISHU_TRACE_PATH = module.DATA_ROOT / "feishu_trace.jsonl"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()

    monkeypatch.setenv("HERMES_FEISHU_PERF_EXPERIMENT_LABEL", "current")

    payload = {
        "header": {"event_type": "im.message.message_read_v1", "event_id": "evt_meta_read"},
        "event": {
            "reader": {"reader_id": {"open_id": "ou_reader_meta"}, "read_time": "1712971234"},
            "message_id_list": ["om_read_meta_1", "om_read_meta_2"],
        },
        "_hermes_gateway_meta": {
            "route_version": "cf_route_policy_v1",
            "provider_alias": "openrouter",
            "cache_eligible": False,
            "cache_status": "bypass",
            "capability_match": True,
            "preferred_model_selected": True,
            "model_catalog_version": "catalog-2026-04-19",
            "feedback_score_before": 82.5,
            "feedback_score_after": 91.0,
            "gateway_error_class": "rate_limited",
            "misroute_detected": False,
        },
        "fallback_reason": "rate_exhausted",
    }

    module._append_feishu_trace("webhook.message_read", payload)
    rows = module._read_feishu_trace(limit=10)
    row = rows[-1]

    assert row["message_id"] == "om_read_meta_1"
    assert row["route_version"] == "cf_route_policy_v1"
    assert row["provider_alias"] == "openrouter"
    assert row["cache_eligible"] is False
    assert row["cache_status"] == "bypass"
    assert row["capability_match"] is True
    assert row["preferred_model_selected"] is True
    assert row["model_catalog_version"] == "catalog-2026-04-19"
    assert row["feedback_score_before"] == 82.5
    assert row["feedback_score_after"] == 91.0
    assert row["gateway_error_class"] == "rate_limited"
    assert row["fallback_reason"] == "rate_exhausted"


def test_internal_agent_exec_normalizes_gateway_error_class():
    from internal.feishu.executor import _normalize_gateway_error_class

    assert _normalize_gateway_error_class("auth_or_secret_missing") == "provider_permission_denied"
    assert _normalize_gateway_error_class("provider_model_invalid") == "provider_model_not_found"
    assert _normalize_gateway_error_class("rate_exhausted") == "rate_limited"
    assert _normalize_gateway_error_class("provider_timeout") == "timeout"
    assert _normalize_gateway_error_class("config_hard_fail") == "payload_incompatible"
    assert _normalize_gateway_error_class("provider_http_error") == "upstream_5xx"
    assert _normalize_gateway_error_class("misrouted_request_class") == "misrouted_request_class"
    assert _normalize_gateway_error_class("", error_text="model catalog stale after sync") == "catalog_stale"


def test_build_feishu_perf_summary_from_rows_aggregates_window_metrics():
    module = _load_module()
    now_ts = int(time.time())
    rows = [
        {
            "ts": now_ts - 10,
            "stage": "webhook.ack",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "app_name": "hermes-agent",
            "experiment_label": "current",
            "snapshot_profile": "web1_ingress1_ack1_chat0",
            "ack_kind": "queued_message",
            "ingress_strategy": "spawn_process_feishu_message_inline",
            "route_hint": "cf_browser_first",
            "route_version": "cf_route_policy_v1",
            "request_class": "text_plain",
            "site_category": "site_content",
            "site_intent": "docs",
            "target_domain": "developers.cloudflare.com",
            "site_skill_name": "site.cloudflare-developers-docs",
            "site_prefetch_mode": "browser_markdown",
            "site_prefetch_status": "completed",
            "site_prefetch_direct_navigation": True,
            "modal_avoided": True,
            "browser_backend_selected": "local",
            "provider_alias": "openrouter",
            "fallback_reason": "plain_text_without_attachments_or_browser",
            "gateway_error_class": "none",
            "cache_status": "hit",
            "cache_eligible": True,
            "capability_match": True,
            "preferred_model_selected": True,
            "model_catalog_version": "catalog-2026-04-19",
            "feedback_score_before": 82.5,
            "feedback_score_after": 91.0,
            "ai_call_count": 1,
            "ack_elapsed_ms": 300,
            "phase_timings": {"dedupe_elapsed_ms": 12, "ack_reaction_schedule_elapsed_ms": 40},
        },
        {
            "ts": now_ts - 10,
            "stage": "feishu.site_prefetch.cache_hit",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "app_name": "hermes-agent",
            "experiment_label": "current",
            "snapshot_profile": "web1_ingress1_ack1_chat0",
        },
        {
            "ts": now_ts - 10,
            "stage": "webhook.response_sent",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "app_name": "hermes-agent",
            "experiment_label": "current",
            "snapshot_profile": "web1_ingress1_ack1_chat0",
            "ack_kind": "queued_message",
            "response_elapsed_ms": 320,
        },
        {
            "ts": now_ts - 9,
            "stage": "dispatch.done",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "app_name": "hermes-agent",
            "experiment_label": "current",
            "snapshot_profile": "web1_ingress1_ack1_chat0",
            "phase_timings": {"message_handle_elapsed_ms": 900, "background_tasks_elapsed_ms": 4100},
        },
        {
            "ts": now_ts - 8,
            "stage": "inline_message.done",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "app_name": "hermes-agent",
            "experiment_label": "current",
            "snapshot_profile": "web1_ingress1_ack1_chat0",
            "worker_boot_id": "boot-1",
            "container_reused": True,
            "inline_elapsed_ms": 5100,
            "execution_mode": "inline_to_background",
            "handoff_reason": "must_ai_default",
        },
        {
            "ts": now_ts - 8,
            "stage": "background_exec.done",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_1",
            "app_name": "hermes-agent",
            "experiment_label": "current",
            "snapshot_profile": "web1_ingress1_ack1_chat0",
            "worker_boot_id": "boot-bg-1",
            "container_reused": False,
            "worker_elapsed_ms": 4900,
            "background_send_elapsed_ms": 4100,
            "execution_mode": "inline_to_background",
            "handoff_reason": "must_ai_default",
        },
        {
            "ts": now_ts - 7,
            "stage": "webhook.ack",
            "event_type": "im.message.receive_v1",
            "event_id": "evt_2",
            "app_name": "hermes-agent",
            "experiment_label": "current",
            "snapshot_profile": "web1_ingress1_ack1_chat0",
            "ack_kind": "duplicate",
            "ack_elapsed_ms": 50,
        },
    ]

    result = module._build_feishu_perf_summary_from_rows(
        rows,
        since_seconds=60,
        event_type="im.message.receive_v1",
        experiment_label="current",
    )

    assert result["event_count"] == 1
    assert result["duplicate_only_event_count"] == 1
    assert result["metrics"]["ack_elapsed_ms"]["avg"] == 300.0
    assert result["metrics"]["response_elapsed_ms"]["avg"] == 320.0
    assert result["metrics"]["message_handle_elapsed_ms"]["avg"] == 900.0
    assert result["metrics"]["inline_elapsed_ms"]["avg"] == 5100.0
    assert result["metrics"]["background_exec_elapsed_ms"]["avg"] == 4900.0
    assert result["metrics"]["background_send_elapsed_ms"]["avg"] == 4100.0
    assert result["by_experiment_label"]["current"] == 1
    assert result["by_execution_mode"]["inline_to_background"] == 1
    assert result["by_handoff_reason"]["must_ai_default"] == 1
    assert result["by_snapshot_profile"]["web1_ingress1_ack1_chat0"] == 1
    assert result["by_route_hint"]["cf_browser_first"] == 1
    assert result["by_route_version"]["cf_route_policy_v1"] == 1
    assert result["by_request_class"]["text_plain"] == 1
    assert result["by_provider_alias"]["openrouter"] == 1
    assert result["by_fallback_reason"]["plain_text_without_attachments_or_browser"] == 1
    assert result["by_cache_status"]["hit"] == 1
    assert result["by_model_catalog_version"]["catalog-2026-04-19"] == 1
    assert result["by_site_category"]["site_content"] == 1
    assert result["by_site_intent"]["docs"] == 1
    assert result["by_target_domain"]["developers.cloudflare.com"] == 1
    assert result["by_site_skill_name"]["site.cloudflare-developers-docs"] == 1
    assert result["by_site_prefetch_mode"]["browser_markdown"] == 1
    assert result["by_site_prefetch_status"]["completed"] == 1
    assert result["modal_avoided_by_site_prefetch"] == 1
    assert result["site_prefetch_cache_hit_rate"] == 1.0
    assert result["site_prefetch_direct_navigation_success_rate"] == 1.0
    assert result["local_browser_success_rate"] == 1.0
    assert result["cloud_escalation_rate"] == 0.0
    assert result["metrics"]["ai_call_count"]["avg"] == 1.0
    assert result["metrics"]["feedback_score_after"]["avg"] == 91.0
    assert result["per_domain_success_rate"]["developers.cloudflare.com"]["success_rate"] == 1.0
    assert result["per_domain_cost"]["developers.cloudflare.com"] == 0.0
    assert result["events"][0]["site_prefetch_mode"] == "browser_markdown"


def test_validate_feishu_native_delivery_impl_generates_and_sends_assets(monkeypatch, tmp_path):
    module = _load_module()
    module.DATA_ROOT = tmp_path / "data"
    module.SESSIONS_DIR = module.DATA_ROOT / "sessions"
    module.UPDATES_PATH = module.DATA_ROOT / "telegram_updates.json"
    module.HERMES_HOME_DIR = tmp_path / "home"
    module._ensure_runtime_dirs()
    monkeypatch.setattr(module, "_prepare_runtime_environment", lambda: None)

    sent = {}

    class FakeAdapter:
        async def send_image_file(self, *, chat_id, image_path, caption=None, **kwargs):
            sent["image"] = {
                "chat_id": chat_id,
                "image_path": image_path,
                "caption": caption,
            }
            assert Path(image_path).exists()
            return types.SimpleNamespace(success=True, message_id="om_img_123", error=None)

        async def send_document(self, *, chat_id, file_path, caption=None, **kwargs):
            sent["document"] = {
                "chat_id": chat_id,
                "file_path": file_path,
                "caption": caption,
            }
            assert Path(file_path).exists()
            return types.SimpleNamespace(success=True, message_id="om_doc_123", error=None)

    async def _fake_get_runtime():
        return types.SimpleNamespace(adapter=FakeAdapter())

    monkeypatch.setattr(module, "_get_feishu_gateway_runtime", _fake_get_runtime)

    result = module._validate_feishu_native_delivery_impl(
        target_id="ou_1234567890",
        document_format="md",
    )

    assert result["status"] == "ok"
    assert result["target_id"] == "ou_1234567890"
    assert result["image"]["success"] is True
    assert result["document"]["success"] is True
    assert sent["image"]["chat_id"] == "ou_1234567890"
    assert sent["document"]["chat_id"] == "ou_1234567890"
    assert not Path(result["image"]["path"]).exists()
    assert not Path(result["document"]["path"]).exists()


def test_validate_feishu_native_delivery_impl_requires_target(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_prepare_runtime_environment", lambda: None)
    monkeypatch.delenv("FEISHU_HOME_CHANNEL", raising=False)

    result = module._validate_feishu_native_delivery_impl()

    assert result["status"] == "error"
    assert "FEISHU_HOME_CHANNEL" in result["message"]


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


def test_modal_source_supports_project_plugins_and_api_only_feishu_runtime():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '.apt_install("curl", "ca-certificates", "gnupg", "libsecret-1-0")' in source
    assert "https://deb.nodesource.com/node_20.x" in source
    assert "/etc/machine-id" in source
    assert "/var/lib/dbus/machine-id" in source
    assert '"uv>=0.7.0,<1"' in source
    assert '"feishu"' in source
    assert '"/feishu/webhook"' in source
    assert 'def debug_feishu_runtime()' in source
    assert 'def debug_feishu_menu_config()' in source
    assert 'def debug_feishu_sync_state()' in source
    assert 'def debug_model_routing_state(' in source
    assert 'Path(".hermes/plugins").is_dir()' in source
    assert 'remote_path="/root/.hermes/plugins"' in source
    assert 'def validate_tavily_integration()' in source
    assert 'modal.Queue.from_name(DEFAULT_CHAT_QUEUE_NAME' in source
    assert 'modal.Queue.from_name(DEFAULT_CRON_QUEUE_NAME' in source
    assert 'HERMES_MODAL_MAINTENANCE_HEARTBEAT_MINUTES' in source
    assert 'schedule=maintenance_heartbeat_schedule' in source
    assert '@app.cls(' in source
    assert 'scaledown_window=DEFAULT_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS' in source
    assert 'enable_memory_snapshot=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED' in source
    assert 'enable_memory_snapshot=CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED' in source
    assert '@modal.enter(snap=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED)' in source
    assert '@modal.enter(snap=CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED)' in source
    assert 'phase_timings=' in source
    assert 'handoff_wait_elapsed_ms' in source
    assert 'handoff_schedule_wait_elapsed_ms' in source
    assert 'ingress_execution_delay_ms' in source
    assert 'ingress_enqueue_elapsed_ms' in source
    assert 'chat_worker_spawn_elapsed_ms' in source
    assert 'ingress.handoff_done' in source
    assert 'process_chat_queue = ChatQueueWorker().process' in source
    assert 'maintenance_heartbeat_enabled = _maintenance_heartbeat_is_enabled()' in source
