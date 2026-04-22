from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional


_RUNTIME_SETTINGS_HOOKS: dict[str, Any] = {}


def configure_runtime_settings(**hooks: Any) -> None:
    _RUNTIME_SETTINGS_HOOKS.clear()
    _RUNTIME_SETTINGS_HOOKS.update(hooks)


@dataclass(slots=True)
class RuntimeSettings:
    model: str
    max_turns: int
    max_tokens: Optional[int]
    provider: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    bearer_token: Optional[str]
    feishu_internal_bearer_token: Optional[str]
    telegram_bot_token: Optional[str]
    telegram_webhook_secret: Optional[str]
    telegram_webhook_url: Optional[str]
    telegram_send_ack: bool
    feishu_app_id: Optional[str]
    feishu_app_secret: Optional[str]
    feishu_domain: Optional[str]
    feishu_connection_mode: Optional[str]
    feishu_verification_token: Optional[str]
    feishu_encrypt_key: Optional[str]
    feishu_bitable_app_token: Optional[str]
    feishu_bitable_table_id: Optional[str]
    feishu_model_registry_mirror_enabled: bool
    feishu_tool_capabilities: list[str]
    feishu_default_workspace: Optional[str]
    qq_app_id: Optional[str]
    qq_app_secret: Optional[str]
    nvidia_api_key: Optional[str]
    nvidia_base_url: Optional[str]
    enabled_toolsets: list[str]
    disabled_toolsets: list[str]
    feishu_bitable_wiki_token: Optional[str] = None
    feishu_model_registry_sync_interval_seconds: int = 0

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        hooks = _RUNTIME_SETTINGS_HOOKS
        pick_runtime_api_config = hooks["pick_runtime_api_config"]
        desired_telegram_webhook_url = hooks["desired_telegram_webhook_url"]
        derive_feishu_internal_bearer_token = hooks["derive_feishu_internal_bearer_token"]
        is_truthy = hooks["is_truthy"]
        get_feishu_sync_interval_seconds = hooks["get_feishu_sync_interval_seconds"]
        split_csv = hooks["split_csv"]
        default_nvidia_base_url = hooks["default_nvidia_base_url"]
        default_disabled_toolsets = hooks["default_disabled_toolsets"]

        provider, base_url, api_key = pick_runtime_api_config()
        feishu_app_id = os.getenv("FEISHU_APP_ID")
        feishu_app_secret = os.getenv("FEISHU_APP_SECRET")
        feishu_internal_bearer_token = (
            os.getenv("HERMES_FEISHU_INTERNAL_BEARER_TOKEN")
            or os.getenv("WEBHOOK_SECRET")
            or os.getenv("HERMES_WEBHOOK_BEARER_TOKEN")
            or derive_feishu_internal_bearer_token(
                app_id=feishu_app_id,
                app_secret=feishu_app_secret,
            )
        )
        return cls(
            model=os.getenv("DEFAULT_MODEL", "openrouter/free"),
            max_turns=int(os.getenv("HERMES_MAX_TURNS", "16")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "0")) or None,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            bearer_token=os.getenv("WEBHOOK_SECRET") or os.getenv("HERMES_WEBHOOK_BEARER_TOKEN"),
            feishu_internal_bearer_token=feishu_internal_bearer_token,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET"),
            telegram_webhook_url=desired_telegram_webhook_url(),
            telegram_send_ack=os.getenv("TELEGRAM_SEND_ACK", "false").lower() in {"1", "true", "yes"},
            feishu_app_id=feishu_app_id,
            feishu_app_secret=feishu_app_secret,
            feishu_domain=os.getenv("FEISHU_DOMAIN") or "feishu",
            feishu_connection_mode=os.getenv("FEISHU_CONNECTION_MODE") or "webhook",
            feishu_verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN"),
            feishu_encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY"),
            feishu_bitable_app_token=os.getenv("FEISHU_BITABLE_APP_TOKEN"),
            feishu_bitable_wiki_token=os.getenv("FEISHU_BITABLE_WIKI_TOKEN"),
            feishu_bitable_table_id=os.getenv("FEISHU_BITABLE_TABLE_ID"),
            feishu_model_registry_mirror_enabled=is_truthy(
                os.getenv("FEISHU_MODEL_REGISTRY_MIRROR_ENABLED"),
                default=False,
            ),
            feishu_model_registry_sync_interval_seconds=get_feishu_sync_interval_seconds(),
            feishu_tool_capabilities=split_csv(os.getenv("HERMES_FEISHU_TOOL_CAPABILITIES")),
            feishu_default_workspace=os.getenv("HERMES_FEISHU_DEFAULT_WORKSPACE"),
            qq_app_id=os.getenv("QQ_APP_ID"),
            qq_app_secret=os.getenv("QQ_APP_SECRET"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY"),
            nvidia_base_url=os.getenv("NVIDIA_BASE_URL") or default_nvidia_base_url,
            enabled_toolsets=split_csv(os.getenv("HERMES_ENABLED_TOOLSETS")),
            disabled_toolsets=split_csv(os.getenv("HERMES_DISABLED_TOOLSETS"))
            or list(default_disabled_toolsets),
        )


def derive_feishu_internal_bearer_token(
    *,
    app_id: str | None,
    app_secret: str | None,
) -> str | None:
    normalized_app_id = str(app_id or "").strip()
    normalized_app_secret = str(app_secret or "").strip()
    if not normalized_app_id or not normalized_app_secret:
        return None
    digest = hashlib.sha256(
        f"hermes-feishu-internal:{normalized_app_id}:{normalized_app_secret}".encode("utf-8")
    ).hexdigest()
    return f"fi_{digest}"


def serialize_settings_for_log(
    settings: RuntimeSettings,
    *,
    mask_secret: Callable[[str | None], str | None],
) -> dict[str, Any]:
    payload = asdict(settings)
    payload["api_key"] = mask_secret(settings.api_key)
    payload["bearer_token"] = mask_secret(settings.bearer_token)
    payload["feishu_internal_bearer_token"] = mask_secret(settings.feishu_internal_bearer_token)
    payload["telegram_bot_token"] = mask_secret(settings.telegram_bot_token)
    payload["telegram_webhook_secret"] = mask_secret(settings.telegram_webhook_secret)
    payload["feishu_app_id"] = mask_secret(settings.feishu_app_id)
    payload["feishu_app_secret"] = mask_secret(settings.feishu_app_secret)
    payload["feishu_verification_token"] = mask_secret(settings.feishu_verification_token)
    payload["feishu_encrypt_key"] = mask_secret(settings.feishu_encrypt_key)
    payload["feishu_bitable_app_token"] = mask_secret(settings.feishu_bitable_app_token)
    payload["feishu_bitable_wiki_token"] = mask_secret(settings.feishu_bitable_wiki_token)
    payload["feishu_bitable_table_id"] = mask_secret(settings.feishu_bitable_table_id)
    payload["qq_app_id"] = mask_secret(settings.qq_app_id)
    payload["qq_app_secret"] = mask_secret(settings.qq_app_secret)
    payload["nvidia_api_key"] = mask_secret(settings.nvidia_api_key)
    return payload
