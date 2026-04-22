from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from internal.session_debug import (
    debug_gateway_session_state as _session_debug_debug_gateway_session_state,
    debug_session_route_state as _session_debug_debug_session_route_state,
)


Namespace = MutableMapping[str, Any]


def register_session_debug_helpers(namespace: Namespace) -> None:
    def _debug_session_route_state(session_key: str) -> dict[str, Any]:
        return _session_debug_debug_session_route_state(
            session_key,
            prepare_runtime_environment=namespace["_prepare_runtime_environment"],
            load_session_state=namespace["_load_session_state"],
        )

    def _debug_gateway_session_state(session_key: str) -> dict[str, Any]:
        return _session_debug_debug_gateway_session_state(
            session_key,
            prepare_runtime_environment=namespace["_prepare_runtime_environment"],
            logger=namespace["logger"],
        )

    namespace["_debug_session_route_state"] = _debug_session_route_state
    namespace["_debug_gateway_session_state"] = _debug_gateway_session_state
