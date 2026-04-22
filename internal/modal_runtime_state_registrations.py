from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


Namespace = MutableMapping[str, Any]


def register_modal_runtime_debug_helpers(namespace: Namespace) -> None:
    def _build_runtime_bootstrap_debug_state(
        platforms: tuple[str, ...] = ("cli", "feishu", "api_server"),
    ) -> dict[str, Any]:
        return namespace["_runtime_build_bootstrap_debug_state"](
            platforms=platforms,
            prepare_runtime_environment=namespace["_prepare_runtime_environment"],
            hermes_home_dir=namespace["HERMES_HOME_DIR"],
        )

    def _build_modal_official_parity_state() -> dict[str, Any]:
        return namespace["_runtime_build_modal_official_parity_state"](
            settings_from_env=namespace["RuntimeSettings"].from_env,
            get_camofox_url=namespace["_get_camofox_url"],
            is_local_camofox_url=namespace["_is_local_camofox_url"],
            is_camofox_healthcheck_ready=namespace["_is_camofox_healthcheck_ready"],
            modal_public_webhook_platforms=namespace["_MODAL_PUBLIC_WEBHOOK_PLATFORMS"],
        )

    namespace["_build_runtime_bootstrap_debug_state"] = _build_runtime_bootstrap_debug_state
    namespace["_build_modal_official_parity_state"] = _build_modal_official_parity_state
