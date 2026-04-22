from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from .webhook_fast_paths import handle_event_fast_paths


@dataclass(slots=True)
class FeishuWebhookRouteDeps:
    try_handle_verification_fast: Callable[[Any, Any], Awaitable[Any | None]]
    parse_webhook_request: Callable[[Any], Awaitable[tuple[Any, dict[str, Any]]]]
    capture_phase_elapsed: Callable[[dict[str, int], str, float], None]
    append_trace: Callable[..., None]
    extract_event_metadata: Callable[[dict[str, Any]], tuple[str, str]]
    mark_event_seen: Callable[[str], bool]
    build_ack_response: Callable[..., Any]
    extract_trace_token: Callable[[dict[str, Any]], str]
    is_session_warmup_event: Callable[[str], bool]
    extract_warmup_context: Callable[[dict[str, Any]], dict[str, Any]]
    spawn_ingress_warmup_async: Callable[..., Awaitable[dict[str, Any]]]
    extract_message_read_event_info: Callable[[Mapping[str, Any]], dict[str, Any]]
    with_internal_meta: Callable[..., dict[str, Any]]
    ack_reaction_mode: str
    supported_ack_reaction_modes: set[str]
    ack_inline_budget_ms: int
    add_ack_reaction_inline_async: Callable[..., Awaitable[Any]]
    spawn_ack_reaction_async: Callable[..., Awaitable[dict[str, Any]]]
    extract_inline_fast_command: Callable[[dict[str, Any]], str | None]
    dispatch_payload: Callable[..., Awaitable[dict[str, Any]]]
    send_local_registry_menu_card: Callable[[Mapping[str, Any]], Awaitable[bool]]
    extract_card_action_name: Callable[[dict[str, Any]], str]
    close_card_from_payload: Callable[[Mapping[str, Any]], Awaitable[bool]]
    enqueue_card_action_for_background: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    build_card_action_ack_payload: Callable[..., dict[str, Any]]
    should_inline_control_event: Callable[[str], bool]
    extract_queue_context: Callable[[dict[str, Any]], dict[str, Any]]
    resolve_message_ingress_strategy: Callable[[dict[str, Any], dict[str, Any]], str]
    spawn_event_handoff_async: Callable[..., Awaitable[dict[str, Any]]]
    spawn_message_inline_async: Callable[..., Awaitable[dict[str, Any]]]
    enqueue_chat_event_async: Callable[..., Awaitable[dict[str, Any]]]
    chat_queue_batch_size: int
    spawn_chat_queue_worker_optimistic_async: Callable[..., Awaitable[dict[str, Any]]]
    http_exception_cls: type[Exception]
    json_response_cls: Any
    logger: Any


async def handle_feishu_webhook(request: Any, *, settings: Any, deps: FeishuWebhookRouteDeps) -> Any:
    request_started_at = time.perf_counter()
    request_started_at_ms = int(time.time() * 1000)
    phase_timings: dict[str, int] = {}
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        raise deps.http_exception_cls(status_code=503, detail="Feishu app credentials are not configured")

    try:
        fast_response = await deps.try_handle_verification_fast(request, settings)
        if fast_response is not None:
            return fast_response

        parse_started_at = time.perf_counter()
        _adapter, payload = await deps.parse_webhook_request(request)
        deps.capture_phase_elapsed(phase_timings, "parse_verify_elapsed_ms", parse_started_at)
        if payload.get("type") == "url_verification":
            return deps.json_response_cls({"challenge": payload.get("challenge", "")})

        deps.append_trace("webhook.accepted", payload)
        event_id, event_type = deps.extract_event_metadata(payload)
        dedupe_started_at = time.perf_counter()
        if event_id and not deps.mark_event_seen(event_id):
            deps.capture_phase_elapsed(phase_timings, "dedupe_elapsed_ms", dedupe_started_at)
            deps.logger.warning(
                "[Feishu] duplicate webhook event ignored event_type=%s event_id=%s",
                event_type or "unknown",
                event_id or "none",
            )
            return deps.build_ack_response(
                payload,
                {"code": 0, "msg": "duplicate"},
                request_started_at=request_started_at,
                ack_kind="duplicate",
                reason="event_deduped",
                phase_timings=phase_timings,
            )

        deps.capture_phase_elapsed(phase_timings, "dedupe_elapsed_ms", dedupe_started_at)
        trace_token = deps.extract_trace_token(payload)
        deps.logger.warning(
            "[Feishu] webhook accepted event_type=%s event_id=%s trace_token=%s",
            event_type or "unknown",
            event_id or "none",
            trace_token or "none",
        )

        response = await handle_event_fast_paths(
            payload,
            event_id=event_id,
            event_type=event_type,
            request_started_at=request_started_at,
            request_started_at_ms=request_started_at_ms,
            phase_timings=phase_timings,
            deps=deps,
        )
        if response is not None:
            return response

        return await _handle_event_queue_path(
            payload,
            event_id=event_id,
            event_type=event_type,
            request_started_at=request_started_at,
            phase_timings=phase_timings,
            deps=deps,
        )
    except Exception as exc:
        deps.logger.exception("Feishu webhook dispatch failed")
        raise deps.http_exception_cls(status_code=500, detail=f"Feishu dispatch failed: {exc}") from exc




async def _handle_event_queue_path(
    payload: dict[str, Any],
    *,
    event_id: str,
    event_type: str,
    request_started_at: float,
    phase_timings: dict[str, int],
    deps: FeishuWebhookRouteDeps,
) -> Any:
    context_started_at = time.perf_counter()
    context = deps.extract_queue_context(payload)
    deps.capture_phase_elapsed(phase_timings, "context_extract_elapsed_ms", context_started_at)

    ingress_strategy = ""
    if event_type == "im.message.receive_v1":
        ingress_strategy = deps.resolve_message_ingress_strategy(payload, context)
        handoff_result: dict[str, Any] | None = None
        if ingress_strategy == "spawn_process_feishu_event":
            handoff_result = await deps.spawn_event_handoff_async(payload=payload, context=context)
        elif ingress_strategy == "spawn_process_feishu_message_inline":
            handoff_result = await deps.spawn_message_inline_async(payload=payload, context=context)

        if handoff_result:
            phase_timings["handoff_wait_elapsed_ms"] = int(handoff_result.get("handoff_wait_elapsed_ms") or 0)
            phase_timings["handoff_schedule_wait_elapsed_ms"] = int(
                handoff_result.get("handoff_schedule_wait_elapsed_ms") or 0
            )
        if handoff_result and handoff_result.get("status") == "scheduled":
            deps.append_trace(
                "webhook.handoff_spawned",
                payload,
                partition=context["partition"],
                lane=context.get("lane") or "",
                reason=handoff_result.get("reason") or "",
                ingress_strategy=ingress_strategy,
            )
            return deps.build_ack_response(
                payload,
                {"code": 0, "msg": "accepted"},
                request_started_at=request_started_at,
                ack_kind="queued_message",
                partition=context["partition"],
                lane=context.get("lane"),
                reason=str(handoff_result.get("reason") or "process_feishu_event_spawned"),
                ingress_strategy=ingress_strategy,
                phase_timings=phase_timings,
            )

    enqueue_result = await deps.enqueue_chat_event_async(
        platform="feishu",
        partition=context["partition"],
        payload=payload,
        metadata={**context, "ingress_strategy": ingress_strategy or "inline_enqueue_spawn"},
        include_queue_depth=False,
    )
    deps.append_trace(
        "queue.enqueue",
        payload,
        partition=context["partition"],
        lane=context.get("lane") or "",
        ingress_strategy=(ingress_strategy or "inline_enqueue_spawn") if event_type == "im.message.receive_v1" else "",
        queue_depth=enqueue_result.get("queue_depth"),
    )
    deps.logger.warning(
        "[Feishu] queue enqueue event_type=%s event_id=%s trace_token=%s partition=%s lane=%s queue_depth=%s",
        event_type or "unknown",
        event_id or "none",
        context.get("trace_token") or "none",
        context["partition"],
        context.get("lane") or "none",
        enqueue_result.get("queue_depth"),
    )

    worker_spawn_started_at = time.perf_counter()
    spawn_result = await deps.spawn_chat_queue_worker_optimistic_async(
        platform="feishu",
        partition=context["partition"],
        max_items=deps.chat_queue_batch_size,
    )
    worker_spawn_elapsed_ms = int((time.perf_counter() - worker_spawn_started_at) * 1000)
    schedule_claim_elapsed_ms = int(spawn_result.get("schedule_claim_elapsed_ms") or 0)
    spawn_rpc_elapsed_ms = int(spawn_result.get("spawn_rpc_elapsed_ms") or 0)
    phase_timings["worker_spawn_elapsed_ms"] = worker_spawn_elapsed_ms
    phase_timings["handoff_schedule_wait_elapsed_ms"] = worker_spawn_elapsed_ms
    phase_timings["handoff_wait_elapsed_ms"] = worker_spawn_elapsed_ms
    phase_timings["schedule_claim_elapsed_ms"] = schedule_claim_elapsed_ms
    phase_timings["spawn_rpc_elapsed_ms"] = spawn_rpc_elapsed_ms
    if spawn_result.get("status") == "scheduled":
        deps.logger.warning(
            "[Feishu] webhook spawned chat worker event_type=%s event_id=%s trace_token=%s partition=%s lane=%s spawn_elapsed_ms=%s schedule_claim_elapsed_ms=%s spawn_rpc_elapsed_ms=%s",
            event_type or "unknown",
            event_id or "none",
            context.get("trace_token") or "none",
            context["partition"],
            context.get("lane") or "none",
            worker_spawn_elapsed_ms,
            schedule_claim_elapsed_ms,
            spawn_rpc_elapsed_ms,
        )
    else:
        deps.logger.warning(
            "[Feishu] webhook skipped chat worker spawn event_type=%s event_id=%s trace_token=%s partition=%s lane=%s reason=%s spawn_elapsed_ms=%s schedule_claim_elapsed_ms=%s spawn_rpc_elapsed_ms=%s",
            event_type or "unknown",
            event_id or "none",
            context.get("trace_token") or "none",
            context["partition"],
            context.get("lane") or "none",
            spawn_result.get("reason") or "already_scheduled",
            worker_spawn_elapsed_ms,
            schedule_claim_elapsed_ms,
            spawn_rpc_elapsed_ms,
        )
    return deps.build_ack_response(
        payload,
        {"code": 0, "msg": "accepted"},
        request_started_at=request_started_at,
        ack_kind="queued_message",
        partition=context["partition"],
        lane=context.get("lane"),
        queue_depth=enqueue_result.get("queue_depth"),
        reason=str(spawn_result.get("reason") or spawn_result.get("status") or "chat_worker_spawned"),
        ingress_strategy=(ingress_strategy or "inline_enqueue_spawn") if event_type == "im.message.receive_v1" else "",
        phase_timings=phase_timings,
    )
