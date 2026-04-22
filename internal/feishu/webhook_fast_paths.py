from __future__ import annotations

import asyncio
import time
from typing import Any

from .trace import build_message_read_correlation_trace_extras, correlate_message_read_event


_CARD_ACTION_ACK_CONTENT = "\u5df2\u6536\u5230\uff0c\u6b63\u5728\u5904\u7406"
_CARD_CLOSE_ACK_CONTENT = "\u5df2\u5173\u95ed"


async def handle_event_fast_paths(
    payload: dict[str, Any],
    *,
    event_id: str,
    event_type: str,
    request_started_at: float,
    request_started_at_ms: int,
    phase_timings: dict[str, int],
    deps: Any,
) -> Any | None:
    if deps.is_session_warmup_event(event_type):
        context_started_at = time.perf_counter()
        warmup_context = deps.extract_warmup_context(payload)
        deps.capture_phase_elapsed(phase_timings, "context_extract_elapsed_ms", context_started_at)
        warmup_result = await deps.spawn_ingress_warmup_async(payload=payload, warmup_context=warmup_context)
        phase_timings["handoff_wait_elapsed_ms"] = int(warmup_result.get("warmup_handoff_wait_elapsed_ms") or 0)
        phase_timings["handoff_schedule_wait_elapsed_ms"] = int(
            warmup_result.get("warmup_handoff_schedule_wait_elapsed_ms") or 0
        )
        deps.append_trace(
            "webhook.session_warmup",
            payload,
            partition=warmup_context["partition"],
            warmup_status=warmup_result.get("status"),
            spawned=warmup_result.get("spawned"),
        )
        deps.logger.warning(
            "[Feishu] webhook session warmup event_type=%s event_id=%s partition=%s spawned=%s actor_id=%s",
            event_type or "unknown",
            event_id or "none",
            warmup_context["partition"],
            warmup_result.get("spawned"),
            warmup_context.get("actor_id") or "unknown",
        )
        return deps.build_ack_response(
            payload,
            {"code": 0, "msg": "accepted"},
            request_started_at=request_started_at,
            ack_kind="warmup",
            partition=warmup_context["partition"],
            lane=warmup_context.get("lane"),
            reason=str(warmup_result.get("status") or ""),
            phase_timings=phase_timings,
        )

    if event_type == "im.message.message_read_v1":
        read_started_at = time.perf_counter()
        read_event = deps.extract_message_read_event_info(payload)
        deps.capture_phase_elapsed(phase_timings, "read_event_extract_elapsed_ms", read_started_at)
        deps.append_trace(
            "webhook.message_read",
            payload,
            reader_open_id=read_event.get("reader_open_id") or "",
            reader_user_id=read_event.get("reader_user_id") or "",
            reader_union_id=read_event.get("reader_union_id") or "",
            tenant_key=read_event.get("tenant_key") or "",
            read_time=read_event.get("read_time") or 0,
            message_id_list=read_event.get("message_id_list") or [],
            message_count=read_event.get("message_count") or 0,
        )
        correlation = correlate_message_read_event(read_event)
        for match in correlation.get("matches") or []:
            deps.append_trace(
                "webhook.message_read.correlated",
                payload,
                **build_message_read_correlation_trace_extras(read_event, match),
            )
        unmatched_message_ids = correlation.get("unmatched_message_ids") or []
        if unmatched_message_ids:
            deps.append_trace(
                "webhook.message_read.unmatched",
                payload,
                reader_open_id=read_event.get("reader_open_id") or "",
                read_time=read_event.get("read_time") or 0,
                unmatched_message_ids=unmatched_message_ids,
                unmatched_count=len(unmatched_message_ids),
            )
        deps.logger.warning(
            "[Feishu] webhook message_read event_type=%s event_id=%s reader_open_id=%s message_count=%s matched=%s unmatched=%s read_time=%s",
            event_type or "unknown",
            event_id or "none",
            read_event.get("reader_open_id") or "none",
            read_event.get("message_count") or 0,
            correlation.get("matched_count") or 0,
            correlation.get("unmatched_count") or 0,
            read_event.get("read_time") or 0,
        )
        return deps.build_ack_response(
            payload,
            {"code": 0, "msg": "accepted"},
            request_started_at=request_started_at,
            ack_kind="message_read",
            reason="read_receipt_fast_path",
            phase_timings=phase_timings,
        )

    payload_for_routing, inline_fast_command = await prepare_message_receive_fast_path(
        payload,
        event_id=event_id,
        event_type=event_type,
        request_started_at_ms=request_started_at_ms,
        phase_timings=phase_timings,
        deps=deps,
    )
    if inline_fast_command:
        deps.append_trace("webhook.inline_fast_command", payload_for_routing, command=inline_fast_command)
        deps.logger.warning(
            "[Feishu] webhook inline fast command event_type=%s event_id=%s command=%s",
            event_type or "unknown",
            event_id or "none",
            inline_fast_command,
        )
        await deps.dispatch_payload(payload_for_routing, await_background_tasks=True)
        return deps.build_ack_response(
            payload_for_routing,
            {"code": 0, "msg": "accepted"},
            request_started_at=request_started_at,
            ack_kind="inline_fast_command",
            reason=inline_fast_command,
            phase_timings=phase_timings,
        )

    if event_type == "application.bot.menu_v6":
        try:
            if await deps.send_local_registry_menu_card(payload_for_routing):
                deps.append_trace("webhook.local_registry_menu", payload_for_routing)
                deps.logger.warning(
                    "[Feishu] webhook local registry menu event_type=%s event_id=%s",
                    event_type or "unknown",
                    event_id or "none",
                )
                return deps.build_ack_response(
                    payload_for_routing,
                    {"code": 0, "msg": "accepted"},
                    request_started_at=request_started_at,
                    ack_kind="local_registry_menu",
                    phase_timings=phase_timings,
                )
        except Exception as exc:
            deps.logger.warning(
                "[Feishu] local registry menu render failed event_id=%s error=%s",
                event_id or "none",
                exc,
                exc_info=True,
            )

    if event_type == "card.action.trigger":
        hermes_action = deps.extract_card_action_name(payload_for_routing)
        if hermes_action == "registry_close_card":
            try:
                if await deps.close_card_from_payload(payload_for_routing):
                    deps.append_trace("webhook.local_registry_close", payload_for_routing)
                    deps.logger.warning(
                        "[Feishu] webhook local registry close event_type=%s event_id=%s",
                        event_type or "unknown",
                        event_id or "none",
                    )
                    return deps.build_ack_response(
                        payload_for_routing,
                        deps.build_card_action_ack_payload(_CARD_CLOSE_ACK_CONTENT),
                        request_started_at=request_started_at,
                        ack_kind="local_registry_close",
                        phase_timings=phase_timings,
                    )
            except Exception as exc:
                deps.logger.warning(
                    "[Feishu] local registry close failed event_id=%s error=%s",
                    event_id or "none",
                    exc,
                    exc_info=True,
                )
        elif hermes_action:
            try:
                enqueue_result = await deps.enqueue_card_action_for_background(payload_for_routing)
                deps.logger.warning(
                    "[Feishu] webhook queued card action event_type=%s event_id=%s partition=%s queue_depth=%s action=%s",
                    event_type or "unknown",
                    event_id or "none",
                    enqueue_result.get("partition") or "none",
                    enqueue_result.get("queue_depth"),
                    hermes_action,
                )
                return deps.build_ack_response(
                    payload_for_routing,
                    deps.build_card_action_ack_payload(_CARD_ACTION_ACK_CONTENT),
                    request_started_at=request_started_at,
                    ack_kind="card_action_queued",
                    partition=enqueue_result.get("partition"),
                    lane=enqueue_result.get("lane"),
                    queue_depth=enqueue_result.get("queue_depth"),
                    reason=hermes_action,
                    phase_timings=phase_timings,
                )
            except Exception as exc:
                deps.logger.warning(
                    "[Feishu] card action queue handoff failed event_id=%s action=%s error=%s",
                    event_id or "none",
                    hermes_action,
                    exc,
                    exc_info=True,
                )

    if deps.should_inline_control_event(event_type):
        deps.append_trace("webhook.inline_control", payload_for_routing)
        deps.logger.warning(
            "[Feishu] webhook inline control event_type=%s event_id=%s",
            event_type or "unknown",
            event_id or "none",
        )
        await deps.dispatch_payload(
            payload_for_routing,
            await_background_tasks=event_type != "card.action.trigger",
        )
        ack_payload = (
            deps.build_card_action_ack_payload(_CARD_ACTION_ACK_CONTENT)
            if event_type == "card.action.trigger"
            else {"code": 0, "msg": "accepted"}
        )
        return deps.build_ack_response(
            payload_for_routing,
            ack_payload,
            request_started_at=request_started_at,
            ack_kind="inline_control",
            phase_timings=phase_timings,
        )

    if payload_for_routing is not payload:
        payload.clear()
        payload.update(payload_for_routing)
    return None


async def prepare_message_receive_fast_path(
    payload: dict[str, Any],
    *,
    event_id: str,
    event_type: str,
    request_started_at_ms: int,
    phase_timings: dict[str, int],
    deps: Any,
) -> tuple[dict[str, Any], str | None]:
    if event_type != "im.message.receive_v1":
        return payload, None

    ack_reaction_mode = (
        deps.ack_reaction_mode if deps.ack_reaction_mode in deps.supported_ack_reaction_modes else "inline"
    )
    ack_reaction_message_id = str((((payload.get("event") or {}).get("message") or {}).get("message_id") or "")).strip()
    payload_for_routing = payload
    if ack_reaction_mode != "off" and ack_reaction_message_id:
        payload_for_routing = deps.with_internal_meta(
            payload,
            ack_reaction_requested_at_ms=request_started_at_ms,
            webhook_message_id=ack_reaction_message_id,
        )
        if ack_reaction_mode == "inline":
            ack_reaction_started_at = time.perf_counter()
            ack_reaction_task = asyncio.create_task(
                deps.add_ack_reaction_inline_async(
                    payload=payload_for_routing,
                    request_started_at_ms=request_started_at_ms,
                )
            )
            ack_reaction_detached = False
            try:
                await asyncio.wait_for(
                    asyncio.shield(ack_reaction_task),
                    timeout=max(deps.ack_inline_budget_ms, 1) / 1000.0,
                )
            except asyncio.TimeoutError:
                ack_reaction_detached = True
                deps.append_trace(
                    "webhook.ack_reaction_detached",
                    payload_for_routing,
                    inline_budget_ms=deps.ack_inline_budget_ms,
                    reason="inline_budget_exceeded",
                    message_id=ack_reaction_message_id,
                )
                deps.logger.warning(
                    "[Feishu] webhook ack reaction detached event_type=%s event_id=%s message_id=%s inline_budget_ms=%s",
                    event_type or "unknown",
                    event_id or "none",
                    ack_reaction_message_id or "none",
                    deps.ack_inline_budget_ms,
                )
            deps.capture_phase_elapsed(phase_timings, "ack_reaction_inline_elapsed_ms", ack_reaction_started_at)
            phase_timings["ack_reaction_detached"] = 1 if ack_reaction_detached else 0
            phase_timings["ack_reaction_schedule_elapsed_ms"] = 0
        else:
            ack_reaction_result = await deps.spawn_ack_reaction_async(
                payload=payload_for_routing,
                request_started_at_ms=request_started_at_ms,
            )
            phase_timings["ack_reaction_schedule_elapsed_ms"] = int(
                ack_reaction_result.get("schedule_elapsed_ms") or 0
            )
            if ack_reaction_result.get("status") == "scheduled":
                deps.append_trace(
                    "webhook.ack_reaction_scheduled",
                    payload_for_routing,
                    reason=str(ack_reaction_result.get("reason") or "process_feishu_ack_reaction_spawned"),
                    message_id=str(ack_reaction_result.get("message_id") or ""),
                )
                deps.logger.warning(
                    "[Feishu] webhook ack reaction scheduled event_type=%s event_id=%s message_id=%s schedule_elapsed_ms=%s",
                    event_type or "unknown",
                    event_id or "none",
                    str(ack_reaction_result.get("message_id") or "") or "none",
                    int(ack_reaction_result.get("schedule_elapsed_ms") or 0),
                )
            elif ack_reaction_result.get("status") not in {"skipped", "scheduled"}:
                deps.logger.warning(
                    "[Feishu] webhook ack reaction schedule skipped event_type=%s event_id=%s reason=%s schedule_elapsed_ms=%s",
                    event_type or "unknown",
                    event_id or "none",
                    str(ack_reaction_result.get("reason") or ack_reaction_result.get("status") or "unknown"),
                    int(ack_reaction_result.get("schedule_elapsed_ms") or 0),
                )

    inline_started_at = time.perf_counter()
    inline_fast_command = deps.extract_inline_fast_command(payload_for_routing)
    deps.capture_phase_elapsed(phase_timings, "inline_fast_extract_elapsed_ms", inline_started_at)
    return payload_for_routing, inline_fast_command
