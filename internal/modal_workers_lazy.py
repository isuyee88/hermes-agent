from __future__ import annotations
import asyncio
import math
import time
from typing import Any

def register_modal_workers(
    *,
    app: Any,
    modal_module: Any,
    image: Any,
    volume: Any,
    secrets: list[Any],
    deps: dict[str, Any]
) -> dict[str, Any]:
    
    _common_kwargs = dict(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        serialized=True,
    )
    
    @app.function(
        **_common_kwargs,
        timeout=120,
        memory=deps["DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB"],
        cpu=deps["DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU"],
        scaledown_window=deps["DEFAULT_FEISHU_ACK_REACTION_SCALEDOWN_WINDOW_SECONDS"],
        enable_memory_snapshot=deps["FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED"],
    )
    def process_feishu_ack_reaction(
        payload: dict[str, Any],
        request_started_at_ms: int | None = None
    ) -> dict[str, Any]:
        from internal.modal_bridge_registrations import _prepare_worker_snapshot_context
        from internal.modal_bridge_registrations import _bootstrap_feishu_ack_reaction_worker_context
        from internal.modal_bridge_registrations import _add_feishu_ack_reaction_from_payload
        
        snapshot_context = {}
        if True:
            snapshot_context = _prepare_worker_snapshot_context(
                worker_name="Feishu ack reaction worker"
            ) or {}
        worker_context = _bootstrap_feishu_ack_reaction_worker_context(
            snapshot_context=snapshot_context
        )
        return _add_feishu_ack_reaction_from_payload(
            payload,
            request_started_at_ms=request_started_at_ms,
            worker_context=worker_context
        )
    
    @app.function(
        **_common_kwargs,
        timeout=180,
        memory=deps["DEFAULT_FEISHU_INGRESS_WORKER_MEMORY_MB"],
        cpu=deps["DEFAULT_FEISHU_INGRESS_WORKER_CPU"],
        scaledown_window=deps["DEFAULT_FEISHU_INGRESS_SCALEDOWN_WINDOW_SECONDS"],
        enable_memory_snapshot=deps["FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED"],
    )
    def process_feishu_event(
        payload: dict[str, Any],
        warmup_only: bool = False,
        warmup_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        from internal.modal_bridge_registrations import _prepare_worker_snapshot_context
        from internal.modal_bridge_registrations import _bootstrap_feishu_ingress_worker_context
        from internal.modal_bridge_registrations import _extract_feishu_warmup_context
        from internal.modal_bridge_registrations import _spawn_chat_queue_worker_sync
        import logging
        
        logger = logging.getLogger(__name__)
        
        snapshot_context = {}
        if True:
            snapshot_context = _prepare_worker_snapshot_context(
                worker_name="Feishu ingress worker"
            ) or {}
        worker_context = _bootstrap_feishu_ingress_worker_context(
            snapshot_context=snapshot_context
        )
        method_started_at = time.perf_counter()
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
                or _extract_feishu_warmup_context(payload).get("partition")
                or ""
            )
            return {"status": "warmup", "partition": partition, "delay_ms": ingress_execution_delay_ms}
        
        return _spawn_chat_queue_worker_sync(
            payload=payload,
            worker=globals().get("process_feishu_message_inline"),
            inline_budget_ms=30000,
            logger=logger,
        )
    
    @app.function(
        **_common_kwargs,
        timeout=600,
        memory=deps["DEFAULT_FEISHU_CHAT_WORKER_MEMORY_MB"],
        cpu=deps["DEFAULT_FEISHU_CHAT_WORKER_CPU"],
        scaledown_window=deps["DEFAULT_FEISHU_CHAT_SCALEDOWN_WINDOW_SECONDS"],
        enable_memory_snapshot=deps["FEISHU_CHAT_MEMORY_SNAPSHOT_ENABLED"],
    )
    def process_feishu_message_inline(
        payload: dict[str, Any],
        warmup_only: bool = False,
        warmup_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from internal.modal_bridge_registrations import _prepare_worker_snapshot_context
        from internal.modal_bridge_registrations import _bootstrap_feishu_chat_worker_context
        from internal.modal_bridge_registrations import _process_feishu_message_inline_impl
        
        if warmup_only:
            return {"status": "warmup"}
        snapshot_context = {}
        if True:
            snapshot_context = _prepare_worker_snapshot_context(
                worker_name="Feishu inline worker"
            ) or {}
        worker_context = _bootstrap_feishu_chat_worker_context(
            snapshot_context=snapshot_context
        )
        return _process_feishu_message_inline_impl(
            payload=payload,
            worker_context=worker_context,
        )
    
    @app.function(
        **_common_kwargs,
        timeout=1800,
        memory=deps["DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB"],
        cpu=deps["DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU"],
        scaledown_window=deps["DEFAULT_FEISHU_BACKGROUND_EXEC_SCALEDOWN_WINDOW_SECONDS"],
        enable_memory_snapshot=deps["FEISHU_BACKGROUND_EXEC_MEMORY_SNAPSHOT_ENABLED"],
    )
    def process_feishu_background_exec(
        payload: dict[str, Any],
        warmup_only: bool = False,
        warmup_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from internal.modal_bridge_registrations import _prepare_worker_snapshot_context
        from internal.modal_bridge_registrations import _bootstrap_feishu_chat_worker_context
        from internal.modal_bridge_registrations import _process_feishu_background_exec_impl
        
        if warmup_only:
            return {"status": "warmup"}
        snapshot_context = {}
        if True:
            snapshot_context = _prepare_worker_snapshot_context(
                worker_name="Feishu background exec worker"
            ) or {}
        worker_context = _bootstrap_feishu_chat_worker_context(
            snapshot_context=snapshot_context
        )
        return _process_feishu_background_exec_impl(
            payload=payload,
            worker_context=worker_context,
        )
    
    @app.function(
        **_common_kwargs,
        timeout=300,
        memory=deps["DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB"],
        cpu=deps["DEFAULT_CHAT_QUEUE_WORKER_CPU"],
        scaledown_window=deps["DEFAULT_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS"],
        enable_memory_snapshot=deps["CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED"],
    )
    def process_chat_queue(
        batch: list[dict[str, Any]],
        warmup_only: bool = False,
        warmup_context: dict[str, Any] | None = None,
        warmup_wait_seconds: float = 0.0,
        warmup_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from internal.modal_bridge_registrations import _prepare_worker_snapshot_context
        from internal.modal_bridge_registrations import _bootstrap_chat_queue_worker_context
        from internal.modal_bridge_registrations import _process_chat_queue_impl
        
        if warmup_only:
            if warmup_wait_seconds > 0:
                time.sleep(warmup_wait_seconds)
            return {"status": "warmup", "metadata": warmup_metadata}
        snapshot_context = {}
        if True:
            snapshot_context = _prepare_worker_snapshot_context(
                worker_name="Chat queue worker"
            ) or {}
        worker_context = _bootstrap_chat_queue_worker_context(
            snapshot_context=snapshot_context
        )
        return _process_chat_queue_impl(
            batch=batch,
            worker_context=worker_context,
        )
    
    return {
        "process_feishu_ack_reaction": process_feishu_ack_reaction,
        "process_feishu_event": process_feishu_event,
        "process_feishu_message_inline": process_feishu_message_inline,
        "process_feishu_background_exec": process_feishu_background_exec,
        "process_chat_queue": process_chat_queue,
    }