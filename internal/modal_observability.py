from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


def build_feishu_snapshot_profile_state(
    *,
    app_name: str,
    web_app_memory_snapshot_enabled: bool,
    feishu_ingress_memory_snapshot_enabled: bool,
    feishu_ack_reaction_memory_snapshot_enabled: bool,
    chat_queue_memory_snapshot_enabled: bool,
) -> dict[str, Any]:
    snapshot_flags = {
        "web_app_enabled": bool(web_app_memory_snapshot_enabled),
        "feishu_ingress_enabled": bool(feishu_ingress_memory_snapshot_enabled),
        "feishu_ack_reaction_enabled": bool(feishu_ack_reaction_memory_snapshot_enabled),
        "chat_queue_enabled": bool(chat_queue_memory_snapshot_enabled),
    }
    enabled_names = [name for name, enabled in snapshot_flags.items() if enabled]
    snapshot_profile = str(os.getenv("HERMES_MODAL_SNAPSHOT_PROFILE_LABEL") or "").strip().lower()
    if not snapshot_profile:
        snapshot_profile = (
            "none"
            if not enabled_names
            else "web{web}_ingress{ingress}_ack{ack}_chat{chat}".format(
                web=int(snapshot_flags["web_app_enabled"]),
                ingress=int(snapshot_flags["feishu_ingress_enabled"]),
                ack=int(snapshot_flags["feishu_ack_reaction_enabled"]),
                chat=int(snapshot_flags["chat_queue_enabled"]),
            )
        )
    experiment_label = str(os.getenv("HERMES_FEISHU_PERF_EXPERIMENT_LABEL") or "").strip().lower()
    if not experiment_label:
        experiment_label = snapshot_profile
    return {
        "app_name": app_name,
        "experiment_label": experiment_label,
        "snapshot_profile": snapshot_profile,
        "snapshot_flags": snapshot_flags,
    }


def append_feishu_trace(
    stage: str,
    payload: dict[str, Any],
    *,
    extract_feishu_trace_context: Callable[[dict[str, Any]], dict[str, Any]],
    build_feishu_snapshot_profile_state: Callable[[], dict[str, Any]],
    feishu_trace_path: Path,
    feishu_event_lock: Any,
    sync_modal_volume: Callable[..., None],
    sync_modal_volume_async: Callable[..., Any],
    extra: dict[str, Any] | None = None,
) -> None:
    trace = {
        "ts": int(time.time()),
        "stage": stage,
        **extract_feishu_trace_context(payload),
        **build_feishu_snapshot_profile_state(),
    }
    if extra:
        trace.update(extra)
    line = json.dumps(trace, ensure_ascii=False)
    with feishu_event_lock:
        feishu_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with feishu_trace_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")
        if stage in {
            "webhook.response_sent",
            "webhook.ack_reaction",
            "dispatch.done",
            "dispatch.error",
            "inline_message.done",
            "background_exec.done",
            "internal.agent_plan.done",
            "internal.agent_exec.done",
            "internal.agent_exec.error",
            "worker.done",
            "worker.error",
        }:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                sync_modal_volume(commit=True)
            else:
                try:
                    asyncio.create_task(sync_modal_volume_async(commit=True))
                except Exception:
                    sync_modal_volume(commit=True)


def build_chat_worker_observability(
    worker_context: dict[str, Any] | None = None,
    *,
    batch_size: int | None = None,
) -> dict[str, Any]:
    context = worker_context if isinstance(worker_context, dict) else {}
    payload = {
        "worker_boot_id": context.get("worker_boot_id"),
        "worker_started_at": context.get("worker_started_at"),
        "enter_elapsed_ms": context.get("enter_elapsed_ms"),
        "runtime_prepare_elapsed_ms": context.get("runtime_prepare_elapsed_ms"),
        "container_reused": bool(context.get("container_reused", False)),
    }
    if batch_size is not None:
        payload["batch_size"] = batch_size
    for optional_key in (
        "worker_partition_key",
        "snapshot_prepare_elapsed_ms",
        "snapshot_enabled",
        "snapshot_restored",
        "execution_mode",
        "handoff_reason",
        "provider_wait_elapsed_ms",
        "provider_attempt_count",
        "provider_retry_count",
        "provider_fallback_used",
        "provider_error_class",
        "background_send_elapsed_ms",
        "routing_state_refreshed_at",
        "model_registry_entry_count",
        "model_registry_source",
        "warmup_status",
        "warmup_event_id",
        "warmup_started_at",
        "warmup_ready_at",
        "warmup_age_ms",
        "warmup_same_container",
        "warmup_message_gap_ms",
        "warmup_actor_id",
        "warmup_chat_id",
    ):
        optional_value = context.get(optional_key)
        if optional_value is not None:
            payload[optional_key] = optional_value
    return payload
