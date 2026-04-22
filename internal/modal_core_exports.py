from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict
from typing import Any

# Global variables to store dependencies
_deps = {}


def _initialize_globals(deps: dict[str, Any]) -> None:
    """Initialize global dependencies."""
    global _deps
    _deps = deps


def run_agent_task(
    task_input: str,
    session_key: str = "",
    model_name: str = "",
    max_tokens: int = 0,
) -> dict[str, Any]:
    deps = _deps
    return deps["run_agent_task_impl"](
        task_input,
        session_key=session_key or None,
        model_name=model_name or None,
        max_tokens=max_tokens or None,
    )


def probe_provider_request_metadata(
    provider: str = "",
    model: str = "",
    prompt: str = "",
) -> dict[str, Any]:
    deps = _deps
    return deps["probe_provider_request_metadata_impl"](
        provider=provider or None,
        model=model or None,
        prompt=prompt or None,
    )


def benchmark_modal_wait_background_hold(
    duration_seconds: int = 60,
) -> dict[str, Any]:
    deps = _deps
    return deps["benchmark_modal_wait_background_hold_impl"](
        duration_seconds=duration_seconds,
    )


def register_modal_core_exports(
    *,
    app: Any,
    modal_module: Any,
    image: Any,
    volume: Any,
    secrets: list[Any],
    deps: dict[str, Any]
) -> dict[str, Any]:
    """Register Modal core export functions with the app."""
    # Initialize global dependencies
    _initialize_globals(deps)

    # Apply Modal decorators with serialized=True
    run_agent_task_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=3600,
        memory=4096,
        cpu=2,
        serialized=True,
    )(run_agent_task)

    probe_provider_request_metadata_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=300,
        memory=2048,
        cpu=1,
        serialized=True,
    )(probe_provider_request_metadata)

    benchmark_modal_wait_background_hold_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=600,
        memory=2048,
        cpu=1,
        serialized=True,
    )(benchmark_modal_wait_background_hold)

    return {
        "run_agent_task": run_agent_task_fn,
        "probe_provider_request_metadata": probe_provider_request_metadata_fn,
        "benchmark_modal_wait_background_hold": benchmark_modal_wait_background_hold_fn,
    }
