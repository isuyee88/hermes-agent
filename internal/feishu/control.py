from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping


BuildResult = Callable[..., dict[str, Any]]


async def run_internal_control(
    payload: Mapping[str, Any],
    *,
    get_runtime: Callable[[], Awaitable[Any]],
    build_source: Callable[[Mapping[str, Any]], Any],
    apply_pending_reconciles: Callable[..., dict[str, Any]],
    append_trace: Callable[..., None],
    logger: Any,
    build_session_state: Callable[..., dict[str, Any]],
    build_result: BuildResult,
    build_local_menu_card: Callable[[str], dict[str, Any] | None],
    skill_combo_definitions: list[dict[str, Any]],
    build_send_plan: Callable[..., list[dict[str, Any]]],
) -> dict[str, Any]:
    runtime = await get_runtime()
    runner = runtime.runner
    adapter = runtime.adapter
    action = str(payload.get("action") or "").strip().lower()
    source = build_source(payload)
    reconcile_summary = apply_pending_reconciles(runner=runner, source=source, payload=payload)
    if int(reconcile_summary.get("received") or 0) > 0:
        append_trace(
            "internal.control.reconcile",
            dict(payload or {}),
            correlation_id=str(payload.get("correlation_id") or "").strip(),
            reconcile_summary=reconcile_summary,
        )
        logger.warning(
            "[Feishu] internal control reconcile action=%s session_key=%s received=%s applied=%s skipped=%s",
            action or "none",
            reconcile_summary.get("session_key") or "none",
            reconcile_summary.get("received"),
            reconcile_summary.get("applied"),
            reconcile_summary.get("skipped"),
        )
    session_state_before = build_session_state(runner=runner, source=source)

    if action == "render_card":
        event_key = str(payload.get("event_key") or "").strip()
        card = build_local_menu_card(event_key)
        return build_result(
            status="ok" if card else "error",
            route_hint="fast_control",
            execution_mode="control_complete",
            session_state_before=session_state_before,
            session_state_after=session_state_before,
            action=action,
            event_key=event_key,
            card=card,
            error="" if card else f"unsupported event_key: {event_key}",
        )

    if action == "activate_skill_combo":
        combo_id = str(payload.get("combo_id") or "").strip()
        combo = next((item for item in skill_combo_definitions if str(item.get("id") or "").strip() == combo_id), None)
        if combo is None:
            return build_result(
                status="error",
                route_hint="fast_control",
                execution_mode="control_complete",
                session_state_before=session_state_before,
                session_state_after=session_state_before,
                action=action,
                error=f"unknown combo_id: {combo_id}",
            )
        from agent.skill_commands import build_session_start_skills_message
        from gateway.platforms.base import MessageEvent, MessageType

        combo_label = str(combo.get("label") or combo_id or "skill-combo").strip()
        suggested_personality = str(combo.get("suggested_personality") or "").strip().lower()
        user_instruction = (
            f"请切换到“{combo_label}”工作模式。"
            "先用中文在 3 行内确认已加载的技能、适用场景，以及接下来准备如何协作。"
        )
        if suggested_personality:
            user_instruction += f" 如需匹配风格，建议同步执行 `/personality {suggested_personality}`。"
        skill_message, _loaded_skills, missing_skills = build_session_start_skills_message(
            [str(item) for item in (combo.get("skills") or [])],
            user_instruction=user_instruction,
        )
        text = skill_message.strip() if skill_message else user_instruction
        if missing_skills:
            text += f"\n\n[Missing skills: {', '.join(missing_skills)}]"
        adapter.captured_operations.clear()
        response_text = await runner._handle_message(
            MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=None,
                message_id=str(payload.get("message_id") or "").strip() or None,
            )
        )
        session_state_after = build_session_state(runner=runner, source=source)
        send_plan = build_send_plan(adapter=adapter, response_text=response_text)
        return build_result(
            status="ok",
            route_hint="fast_control",
            execution_mode="control_complete",
            session_state_before=session_state_before,
            session_state_after=session_state_after,
            action=action,
            final_response=str(response_text or ""),
            send_plan=send_plan,
            reconcile_required=False,
        )

    if action in {"dispatch_command", "get_session_state"}:
        if action == "get_session_state":
            return build_result(
                status="ok",
                route_hint="fast_control",
                execution_mode="control_complete",
                session_state_before=session_state_before,
                session_state_after=session_state_before,
                action=action,
            )
        from gateway.platforms.base import MessageEvent, MessageType

        command_text = str(payload.get("command_text") or "").strip()
        if not command_text:
            return build_result(
                status="error",
                route_hint="fast_control",
                execution_mode="control_complete",
                session_state_before=session_state_before,
                session_state_after=session_state_before,
                action=action,
                error="missing command_text",
            )
        adapter.captured_operations.clear()
        response_text = await runner._handle_message(
            MessageEvent(
                text=command_text,
                message_type=MessageType.COMMAND,
                source=source,
                raw_message=None,
                message_id=str(payload.get("message_id") or "").strip() or None,
            )
        )
        session_state_after = build_session_state(runner=runner, source=source)
        send_plan = build_send_plan(adapter=adapter, response_text=response_text)
        return build_result(
            status="ok",
            route_hint="fast_control",
            execution_mode="control_complete",
            session_state_before=session_state_before,
            session_state_after=session_state_after,
            action=action,
            final_response=str(response_text or ""),
            send_plan=send_plan,
            reconcile_required=False,
        )

    return build_result(
        status="error",
        route_hint="fast_control",
        execution_mode="control_complete",
        session_state_before=session_state_before,
        session_state_after=session_state_before,
        action=action,
        error=f"unsupported action: {action}",
    )
