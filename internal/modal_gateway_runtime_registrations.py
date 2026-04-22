from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_modal_public_gateway_runtime_helpers(namespace: Namespace) -> None:
    async def _initialize_telegram_gateway_runtime(settings: Any) -> Any:
        return await namespace["_platform_initialize_telegram_gateway_runtime"](
            settings,
            runtime_cls=namespace["_TelegramGatewayRuntime"],
            logger=namespace["logger"],
        )

    async def _get_telegram_gateway_runtime() -> Any:
        if namespace["_TELEGRAM_RUNTIME"] is not None:
            return namespace["_TELEGRAM_RUNTIME"]

        async with namespace["_get_telegram_runtime_lock"]():
            if namespace["_TELEGRAM_RUNTIME"] is not None:
                return namespace["_TELEGRAM_RUNTIME"]
            namespace["_prepare_runtime_environment"]()
            settings = namespace["RuntimeSettings"].from_env()
            namespace["_TELEGRAM_RUNTIME"] = await namespace["_initialize_telegram_gateway_runtime"](settings)
            return namespace["_TELEGRAM_RUNTIME"]

    async def _initialize_feishu_gateway_runtime(settings: Any) -> Any:
        return await namespace["_platform_initialize_feishu_gateway_runtime"](
            settings,
            runtime_cls=namespace["_FeishuGatewayRuntime"],
            split_csv=namespace["_split_csv"],
            dedupe_keep_order=namespace["_dedupe_keep_order"],
            is_truthy=namespace["_is_truthy"],
            get_cached_feishu_bot_identity=namespace["_get_cached_feishu_bot_identity"],
            cache_feishu_bot_identity_async=namespace["_cache_feishu_bot_identity_async"],
            logger=namespace["logger"],
        )

    def _get_feishu_webhook_security_settings() -> tuple[str, str]:
        encrypt_key = str(namespace["os"].getenv("FEISHU_ENCRYPT_KEY", "") or "").strip()
        verification_token = str(namespace["os"].getenv("FEISHU_VERIFICATION_TOKEN", "") or "").strip()
        return encrypt_key, verification_token

    def _is_feishu_webhook_signature_valid(
        headers: Mapping[str, Any],
        body_bytes: bytes,
        *,
        encrypt_key: str,
    ) -> bool:
        return namespace["_feishu_selftest_is_signature_valid"](
            headers,
            body_bytes,
            encrypt_key=encrypt_key,
            allow_when_encrypt_key_missing=True,
            logger=namespace["logger"],
        )

    def _decrypt_feishu_webhook_payload(encrypted_payload: str, *, encrypt_key: str) -> dict[str, Any]:
        if not encrypted_payload:
            raise ValueError("encrypted webhook payload is empty")
        if not encrypt_key:
            raise ValueError("encrypt_key is required to decrypt webhook payloads")
        return namespace["_feishu_selftest_decrypt_payload"](encrypt_key, encrypted_payload)

    async def _get_feishu_gateway_runtime() -> Any:
        if namespace["_FEISHU_RUNTIME"] is not None:
            return namespace["_FEISHU_RUNTIME"]

        async with namespace["_get_feishu_runtime_lock"]():
            if namespace["_FEISHU_RUNTIME"] is not None:
                return namespace["_FEISHU_RUNTIME"]
            namespace["_prepare_runtime_environment"]()
            settings = namespace["RuntimeSettings"].from_env()
            namespace["_FEISHU_RUNTIME"] = await namespace["_initialize_feishu_gateway_runtime"](settings)
            return namespace["_FEISHU_RUNTIME"]

    async def _initialize_qq_gateway_runtime(settings: Any) -> Any:
        return namespace["_qq_initialize_gateway_runtime_bridge"](
            settings,
            runtime_cls=namespace["_QQGatewayRuntime"],
        )

    async def _get_qq_gateway_runtime() -> Any:
        def _get_current_runtime() -> Any:
            return namespace["_QQ_RUNTIME"]

        def _set_runtime(runtime: Any) -> None:
            namespace["_QQ_RUNTIME"] = runtime

        return await namespace["_qq_get_cached_runtime_bridge"](
            get_current_runtime=_get_current_runtime,
            get_lock=namespace["_get_qq_runtime_lock"],
            prepare_runtime_environment=namespace["_prepare_runtime_environment"],
            settings_from_env=namespace["RuntimeSettings"].from_env,
            initialize_runtime=namespace["_initialize_qq_gateway_runtime"],
            set_runtime=_set_runtime,
        )

    namespace["_initialize_telegram_gateway_runtime"] = _initialize_telegram_gateway_runtime
    namespace["_get_telegram_gateway_runtime"] = _get_telegram_gateway_runtime
    namespace["_initialize_feishu_gateway_runtime"] = _initialize_feishu_gateway_runtime
    namespace["_get_feishu_webhook_security_settings"] = _get_feishu_webhook_security_settings
    namespace["_is_feishu_webhook_signature_valid"] = _is_feishu_webhook_signature_valid
    namespace["_decrypt_feishu_webhook_payload"] = _decrypt_feishu_webhook_payload
    namespace["_get_feishu_gateway_runtime"] = _get_feishu_gateway_runtime
    namespace["_initialize_qq_gateway_runtime"] = _initialize_qq_gateway_runtime
    namespace["_get_qq_gateway_runtime"] = _get_qq_gateway_runtime
