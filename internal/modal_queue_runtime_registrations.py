from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_modal_queue_runtime_helpers(namespace: Namespace) -> None:
    def _get_cron_queue() -> Any:
        namespace["CRON_QUEUE"] = namespace["_infra_get_named_queue"](
            modal_module=namespace["modal"],
            current_queue=namespace["CRON_QUEUE"],
            queue_name=namespace["DEFAULT_CRON_QUEUE_NAME"],
            missing_message="Modal is required for the cron queue",
        )
        return namespace["CRON_QUEUE"]

    def _get_chat_queue() -> Any:
        namespace["CHAT_QUEUE"] = namespace["_infra_get_named_queue"](
            modal_module=namespace["modal"],
            current_queue=namespace["CHAT_QUEUE"],
            queue_name=namespace["DEFAULT_CHAT_QUEUE_NAME"],
            missing_message="Modal is required for the chat queue",
        )
        return namespace["CHAT_QUEUE"]

    async def _prewarm_chat_queue_async() -> None:
        await namespace["_infra_prewarm_queue_len"](
            queue=namespace["_get_chat_queue"](),
            is_prewarmed=namespace["_CHAT_QUEUE_PREWARMED"],
            set_prewarmed=lambda value: namespace.__setitem__("_CHAT_QUEUE_PREWARMED", value),
            logger=namespace["logger"],
            warning_prefix="Modal chat queue prewarm failed",
        )

    def _sync_modal_volume(*, reload: bool = False, commit: bool = False) -> None:
        namespace["_infra_sync_modal_volume"](
            volume=namespace["MODAL_VOLUME"],
            reload=reload,
            commit=commit,
            logger=namespace["logger"],
            warning_prefix="Modal volume sync failed",
        )

    async def _sync_modal_volume_async(*, reload: bool = False, commit: bool = False) -> None:
        await namespace["_infra_sync_modal_volume_async"](
            volume=namespace["MODAL_VOLUME"],
            reload=reload,
            commit=commit,
            logger=namespace["logger"],
            warning_prefix="Modal volume async sync failed",
        )

    def _should_reload_modal_volume_for_claims(kind: str) -> bool:
        env_name = f"HERMES_MODAL_{str(kind or '').strip().upper()}_CLAIMS_RELOAD"
        return namespace["_is_truthy"](namespace["os"].getenv(env_name), default=False)

    def _safe_cron_queue_depth() -> int | None:
        return namespace["_infra_safe_queue_depth"](
            get_queue=namespace["_get_cron_queue"],
            logger=namespace["logger"],
            warning_prefix="Unable to read Modal cron queue depth",
        )

    async def _safe_cron_queue_depth_async() -> int | None:
        return await namespace["_infra_safe_queue_depth_async"](
            queue=namespace["_get_cron_queue"](),
            logger=namespace["logger"],
            warning_prefix="Unable to read Modal cron queue depth",
        )

    def _safe_chat_queue_depth() -> int | None:
        return namespace["_infra_safe_queue_depth"](
            get_queue=namespace["_get_chat_queue"],
            logger=namespace["logger"],
            warning_prefix="Unable to read Modal chat queue depth",
        )

    async def _safe_chat_queue_depth_async() -> int | None:
        return await namespace["_infra_safe_queue_depth_async"](
            queue=namespace["_get_chat_queue"](),
            logger=namespace["logger"],
            warning_prefix="Unable to read Modal chat queue depth",
        )

    namespace["_get_cron_queue"] = _get_cron_queue
    namespace["_get_chat_queue"] = _get_chat_queue
    namespace["_prewarm_chat_queue_async"] = _prewarm_chat_queue_async
    namespace["_sync_modal_volume"] = _sync_modal_volume
    namespace["_sync_modal_volume_async"] = _sync_modal_volume_async
    namespace["_should_reload_modal_volume_for_claims"] = _should_reload_modal_volume_for_claims
    namespace["_safe_cron_queue_depth"] = _safe_cron_queue_depth
    namespace["_safe_cron_queue_depth_async"] = _safe_cron_queue_depth_async
    namespace["_safe_chat_queue_depth"] = _safe_chat_queue_depth
    namespace["_safe_chat_queue_depth_async"] = _safe_chat_queue_depth_async
