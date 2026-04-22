from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_modal_telegram_gateway_helpers(namespace: Namespace) -> None:
    def _normalize_public_https_url(value: str | None) -> str | None:
        return namespace["_telegram_gateway_normalize_https_url"](value)

    def _resolve_telegram_webhook_url(
        *,
        explicit_url: str | None = None,
        public_base_url: str | None = None,
    ) -> str | None:
        return namespace["_telegram_gateway_resolve_webhook_url"](
            explicit_url=explicit_url,
            public_base_url=public_base_url,
            normalize_public_https_url_fn=namespace["_normalize_public_https_url"],
        )

    def _desired_telegram_webhook_url() -> str | None:
        return namespace["_telegram_gateway_desired_webhook_url"](
            resolve_telegram_webhook_url_fn=namespace["_resolve_telegram_webhook_url"],
        )

    async def _fetch_telegram_webhook_info(bot_token: str) -> dict[str, Any]:
        return await namespace["_telegram_gateway_fetch_webhook_info"](bot_token)

    async def _set_telegram_webhook(
        bot_token: str,
        webhook_url: str,
        *,
        webhook_secret: str | None = None,
        drop_pending_updates: bool = False,
    ) -> dict[str, Any]:
        return await namespace["_telegram_gateway_set_webhook"](
            bot_token,
            webhook_url,
            webhook_secret=webhook_secret,
            drop_pending_updates=drop_pending_updates,
        )

    async def _get_telegram_webhook_status(
        settings: Any,
        *,
        ensure_registered: bool = False,
        drop_pending_updates: bool = False,
    ) -> dict[str, Any]:
        return await namespace["_telegram_gateway_get_webhook_status"](
            settings,
            ensure_registered=ensure_registered,
            drop_pending_updates=drop_pending_updates,
            is_valid_telegram_bot_token_format=namespace["_is_valid_telegram_bot_token_format"],
            is_truthy=namespace["_is_truthy"],
            set_telegram_webhook_fn=namespace["_set_telegram_webhook"],
            fetch_telegram_webhook_info_fn=namespace["_fetch_telegram_webhook_info"],
        )

    namespace["_normalize_public_https_url"] = _normalize_public_https_url
    namespace["_resolve_telegram_webhook_url"] = _resolve_telegram_webhook_url
    namespace["_desired_telegram_webhook_url"] = _desired_telegram_webhook_url
    namespace["_fetch_telegram_webhook_info"] = _fetch_telegram_webhook_info
    namespace["_set_telegram_webhook"] = _set_telegram_webhook
    namespace["_get_telegram_webhook_status"] = _get_telegram_webhook_status
