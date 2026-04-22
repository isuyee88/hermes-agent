from __future__ import annotations
import asyncio
import math
import time
from typing import Any

# Global variables to store configuration and dependencies
_app = None
_image = None
_volume = None
_secrets = None
_deps = {}

# Configuration values
_default_feishu_ack_reaction_worker_memory_mb = 1024
_default_feishu_ack_reaction_worker_cpu = 1
_default_feishu_ack_reaction_scaledown_window_seconds = 60
_feishu_ack_reaction_memory_snapshot_enabled = False

_default_feishu_ingress_worker_memory_mb = 2048
_default_feishu_ingress_worker_cpu = 1
_default_feishu_ingress_scaledown_window_seconds = 60
_feishu_ingress_memory_snapshot_enabled = False

_default_feishu_chat_worker_memory_mb = 4096
_default_feishu_chat_worker_cpu = 2
_default_feishu_chat_scaledown_window_seconds = 120
_feishu_chat_memory_snapshot_enabled = False

_default_feishu_background_exec_worker_memory_mb = 8192
_default_feishu_background_exec_worker_cpu = 4
_default_feishu_background_exec_scaledown_window_seconds = 300
_feishu_background_exec_memory_snapshot_enabled = False

_default_chat_queue_worker_memory_mb = 4096
_default_chat_queue_worker_cpu = 2
_default_chat_queue_scaledown_window_seconds = 120
_chat_queue_memory_snapshot_enabled = False

_default_feishu_inline_worker_hard_budget_ms = 30000


def _initialize_globals(deps: dict[str, Any]) -> None:
    """Initialize global configuration values from deps."""
    global _deps
    global _default_feishu_ack_reaction_worker_memory_mb, _default_feishu_ack_reaction_worker_cpu
    global _default_feishu_ack_reaction_scaledown_window_seconds, _feishu_ack_reaction_memory_snapshot_enabled
    global _default_feishu_ingress_worker_memory_mb, _default_feishu_ingress_worker_cpu
    global _default_feishu_ingress_scaledown_window_seconds, _feishu_ingress_memory_snapshot_enabled
    global _default_feishu_chat_worker_memory_mb, _default_feishu_chat_worker_cpu
    global _default_feishu_chat_scaledown_window_seconds, _feishu_chat_memory_snapshot_enabled
    global _default_feishu_background_exec_worker_memory_mb, _default_feishu_background_exec_worker_cpu
    global _default_feishu_background_exec_scaledown_window_seconds, _feishu_background_exec_memory_snapshot_enabled
    global _default_chat_queue_worker_memory_mb, _default_chat_queue_worker_cpu
    global _default_chat_queue_scaledown_window_seconds, _chat_queue_memory_snapshot_enabled
    global _default_feishu_inline_worker_hard_budget_ms
    
    _deps = deps
    
    _default_feishu_ack_reaction_worker_memory_mb = deps.get("DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB", 1024)
    _default_feishu_ack_reaction_worker_cpu = deps.get("DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU", 1)
    _default_feishu_ack_reaction_scaledown_window_seconds = deps.get("DEFAULT_FEISHU_ACK_REACTION_SCALEDOWN_WINDOW_SECONDS", 60)
    _feishu_ack_reaction_memory_snapshot_enabled = deps.get("FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED", False)
    
    _default_feishu_ingress_worker_memory_mb = deps.get("DEFAULT_FEISHU_INGRESS_WORKER_MEMORY_MB", 2048)
    _default_feishu_ingress_worker_cpu = deps.get("DEFAULT_FEISHU_INGRESS_WORKER_CPU", 1)
    _default_feishu_ingress_scaledown_window_seconds = deps.get("DEFAULT_FEISHU_INGRESS_SCALEDOWN_WINDOW_SECONDS", 60)
    _feishu_ingress_memory_snapshot_enabled = deps.get("FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED", False)
    
    _default_feishu_chat_worker_memory_mb = deps.get("DEFAULT_FEISHU_CHAT_WORKER_MEMORY_MB", 4096)
    _default_feishu_chat_worker_cpu = deps.get("DEFAULT_FEISHU_CHAT_WORKER_CPU", 2)
    _default_feishu_chat_scaledown_window_seconds = deps.get("DEFAULT_FEISHU_CHAT_SCALEDOWN_WINDOW_SECONDS", 120)
    _feishu_chat_memory_snapshot_enabled = deps.get("FEISHU_CHAT_MEMORY_SNAPSHOT_ENABLED", False)
    
    _default_feishu_background_exec_worker_memory_mb = deps.get("DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB", 8192)
    _default_feishu_background_exec_worker_cpu = deps.get("DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU", 4)
    _default_feishu_background_exec_scaledown_window_seconds = deps.get("DEFAULT_FEISHU_BACKGROUND_EXEC_SCALEDOWN_WINDOW_SECONDS", 300)
    _feishu_background_exec_memory_snapshot_enabled = deps.get("FEISHU_BACKGROUND_EXEC_MEMORY_SNAPSHOT_ENABLED", False)
    
    _default_chat_queue_worker_memory_mb = deps.get("DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB", 4096)
    _default_chat_queue_worker_cpu = deps.get("DEFAULT_CHAT_QUEUE_WORKER_CPU", 2)
    _default_chat_queue_scaledown_window_seconds = deps.get("DEFAULT_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS", 120)
    _chat_queue_memory_snapshot_enabled = deps.get("CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED", False)
    
    _default_feishu_inline_worker_hard_budget_ms = deps.get("DEFAULT_FEISHU_INLINE_WORKER_HARD_BUDGET_MS", 30000)


# Function implementations at module level
def process_feishu_ack_reaction(
    payload: dict[str, Any],
    request_started_at_ms: int | None = None
) -> dict[str, Any]:
    deps = _deps
    snapshot_context = {}
    if deps.get("FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED"):
        snapshot_context = deps["prepare_worker_snapshot_context"](
            worker_name="Feishu ack reaction worker"
        ) or {}
    worker_context = deps["bootstrap_feishu_ack_reaction_worker_context"](
        snapshot_context=snapshot_context
    )
    return deps["add_feishu_ack_reaction_from_payload"](
        payload,
        request_started_at_ms=request_started_at_ms,
        worker_context=worker_context
    )


def process_feishu_event(
    payload: dict[str, Any],
    warmup_only: bool = False,
    warmup_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    deps = _deps
    snapshot_context = {}
    if deps.get("FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED"):
        snapshot_context = deps["prepare_worker_snapshot_context"](
            worker_name="Feishu ingress worker"
        ) or {}
    worker_context = deps["bootstrap_feishu_ingress_worker_context"](
        snapshot_context=snapshot_context
    )
    ingress_meta = payload.get("_hermes_ingress") if isinstance(payload, dict) else {}
    ingress_meta = dict(ingress_meta) if isinstance(ingress_meta, dict) else {}
    handoff_requested_at_ms = int(ingress_meta.get("handoff_requested_at_ms") or 0)
    ingress_started_at_ms = int(time.time() * 1000)
    ingress_execution_delay_ms = (
        max(0, ingress_started_at_ms - handoff_requested_at_ms) if handoff_requested_at_ms else 0
    )
    
    if warmup_only:
        normalized_warmup_context = {
            key: value
            for key, value in dict(warmup_context or {}).items()
            if key in {"event_type", "event_id", "chat_id", "actor_id", "started_at_ms", "partition"}
        }
        partition = str(
            normalized_warmup_context.get("partition")
            or deps["extract_feishu_warmup_context"](payload).get("partition")
            or ""
        )
        return {"status": "warmup", "partition": partition, "delay_ms": ingress_execution_delay_ms}
    
    return deps["spawn_chat_queue_worker_sync"](
        payload=payload,
        worker=process_feishu_message_inline,
        inline_budget_ms=_default_feishu_inline_worker_hard_budget_ms,
        logger=deps["logger"],
    )


def process_feishu_message_inline(
    payload: dict[str, Any],
    warmup_only: bool = False,
    warmup_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deps = _deps
    if warmup_only:
        return {"status": "warmup"}
    snapshot_context = {}
    if deps.get("FEISHU_CHAT_MEMORY_SNAPSHOT_ENABLED"):
        snapshot_context = deps["prepare_worker_snapshot_context"](
            worker_name="Feishu inline worker"
        ) or {}
    worker_context = deps["bootstrap_feishu_chat_worker_context"](
        snapshot_context=snapshot_context
    )
    return deps["process_feishu_message_inline_impl"](
        payload=payload,
        worker_context=worker_context,
    )


def process_feishu_background_exec(
    payload: dict[str, Any],
    warmup_only: bool = False,
    warmup_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deps = _deps
    if warmup_only:
        return {"status": "warmup"}
    snapshot_context = {}
    if deps.get("FEISHU_BACKGROUND_EXEC_MEMORY_SNAPSHOT_ENABLED"):
        snapshot_context = deps["prepare_worker_snapshot_context"](
            worker_name="Feishu background exec worker"
        ) or {}
    worker_context = deps["bootstrap_feishu_chat_worker_context"](
        snapshot_context=snapshot_context
    )
    return deps["process_feishu_background_exec_impl"](
        payload=payload,
        worker_context=worker_context,
    )


def process_chat_queue(
    batch: list[dict[str, Any]],
    warmup_only: bool = False,
    warmup_context: dict[str, Any] | None = None,
    warmup_wait_seconds: float = 0.0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deps = _deps
    if warmup_only:
        if warmup_wait_seconds > 0:
            time.sleep(warmup_wait_seconds)
        return {"status": "warmup", "metadata": warmup_metadata}
    snapshot_context = {}
    if deps.get("CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED"):
        snapshot_context = deps["prepare_worker_snapshot_context"](
            worker_name="Chat queue worker"
        ) or {}
    worker_context = deps["bootstrap_chat_queue_worker_context"](
        snapshot_context=snapshot_context
    )
    return deps["process_chat_queue_impl"](
        batch=batch,
        worker_context=worker_context,
    )


def register_modal_workers(
    *,
    app: Any,
    modal_module: Any,
    image: Any,
    volume: Any,
    secrets: list[Any],
    deps: dict[str, Any]
) -> dict[str, Any]:
    """Register Modal worker functions with the app."""
    global _app, _image, _volume, _secrets
    _app = app
    _image = image
    _volume = volume
    _secrets = secrets
    
    # Initialize global configuration
    _initialize_globals(deps)
    
    # Apply Modal decorators with serialized=True to allow non-global functions
    process_feishu_ack_reaction_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=120,
        memory=_default_feishu_ack_reaction_worker_memory_mb,
        cpu=_default_feishu_ack_reaction_worker_cpu,
        scaledown_window=_default_feishu_ack_reaction_scaledown_window_seconds,
        enable_memory_snapshot=_feishu_ack_reaction_memory_snapshot_enabled,
        serialized=True,
    )(process_feishu_ack_reaction)

    process_feishu_event_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=180,
        memory=_default_feishu_ingress_worker_memory_mb,
        cpu=_default_feishu_ingress_worker_cpu,
        scaledown_window=_default_feishu_ingress_scaledown_window_seconds,
        enable_memory_snapshot=_feishu_ingress_memory_snapshot_enabled,
        serialized=True,
    )(process_feishu_event)

    process_feishu_message_inline_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=600,
        memory=_default_feishu_chat_worker_memory_mb,
        cpu=_default_feishu_chat_worker_cpu,
        scaledown_window=_default_feishu_chat_scaledown_window_seconds,
        enable_memory_snapshot=_feishu_chat_memory_snapshot_enabled,
        serialized=True,
    )(process_feishu_message_inline)

    process_feishu_background_exec_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=1800,
        memory=_default_feishu_background_exec_worker_memory_mb,
        cpu=_default_feishu_background_exec_worker_cpu,
        scaledown_window=_default_feishu_background_exec_scaledown_window_seconds,
        enable_memory_snapshot=_feishu_background_exec_memory_snapshot_enabled,
        serialized=True,
    )(process_feishu_background_exec)

    process_chat_queue_fn = app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=300,
        memory=_default_chat_queue_worker_memory_mb,
        cpu=_default_chat_queue_worker_cpu,
        scaledown_window=_default_chat_queue_scaledown_window_seconds,
        enable_memory_snapshot=_chat_queue_memory_snapshot_enabled,
        serialized=True,
    )(process_chat_queue)

    return {
        "process_feishu_ack_reaction": process_feishu_ack_reaction_fn,
        "process_feishu_event": process_feishu_event_fn,
        "process_feishu_message_inline": process_feishu_message_inline_fn,
        "process_feishu_background_exec": process_feishu_background_exec_fn,
        "process_chat_queue": process_chat_queue_fn,
    }
