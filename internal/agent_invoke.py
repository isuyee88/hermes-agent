from __future__ import annotations

import time
import uuid
from typing import Any, Callable


def run_agent_task(
    task_input: str,
    *,
    session_key: str | None = None,
    model_name: str | None = None,
    max_tokens: int | None = None,
    prepare_runtime_environment: Callable[[], None],
    settings_from_env: Callable[[], Any],
    load_session_state: Callable[[str], dict[str, Any]],
    resolve_primary_route: Callable[[Any, str | None], dict[str, Any]],
    is_route_lease_active: Callable[[Any, dict[str, Any] | None], bool],
    hydrate_route_from_lease: Callable[[Any, dict[str, Any] | None], dict[str, Any] | None],
    resolve_runtime_model_name: Callable[[str, str | None], str],
    determine_route_refresh_reason: Callable[[dict[str, Any]], str | None],
    select_retry_route_for_result: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None],
    build_route_lease: Callable[..., dict[str, Any]],
    expire_route_lease: Callable[..., dict[str, Any]],
    refresh_route_lease: Callable[..., dict[str, Any]],
    save_session_state: Callable[..., None],
    extract_tool_names: Callable[[list[Any]], list[str]],
    increment_route_metric: Callable[[dict[str, Any], str], dict[str, Any]],
    logger: Any,
) -> dict[str, Any]:
    prepare_runtime_environment()
    settings = settings_from_env()
    effective_session_key = session_key or f"task:{uuid.uuid4()}"
    session_state = load_session_state(effective_session_key)
    route_lease = session_state.get("route_lease") if isinstance(session_state.get("route_lease"), dict) else None
    route_debug = dict(session_state.get("route_debug") or {})
    route_metrics = dict(session_state.get("route_metrics") or {})
    explicit_model_requested = bool(str(model_name or "").strip())

    from run_agent import AIAgent

    def _execute_once(route: dict[str, Any], *, route_selection: str) -> tuple[Any, dict[str, Any], str, int]:
        resolved_model = resolve_runtime_model_name(route["model"], route.get("provider"))
        agent = AIAgent(
            model=resolved_model,
            provider=route.get("provider"),
            base_url=route.get("base_url"),
            api_key=route.get("api_key"),
            max_iterations=settings.max_turns,
            enabled_toolsets=settings.enabled_toolsets or None,
            disabled_toolsets=settings.disabled_toolsets or None,
            quiet_mode=True,
            max_tokens=max_tokens or settings.max_tokens,
            platform="modal",
            persist_session=False,
            session_id=session_state["session_id"],
            trace_session_key=effective_session_key,
            trace_metadata={
                "channel_type": "invoke",
                "route_selection": route_selection,
            },
        )
        started_at = time.time()
        result = agent.run_conversation(
            task_input,
            conversation_history=session_state["messages"],
            persist_user_message=task_input,
        )
        elapsed_ms = int((time.time() - started_at) * 1000)
        return agent, result, resolved_model, elapsed_ms

    if explicit_model_requested:
        primary_route = resolve_primary_route(settings, model_name)
        route_selection = "explicit_override"
    elif is_route_lease_active(settings, route_lease):
        primary_route = hydrate_route_from_lease(settings, route_lease) or resolve_primary_route(settings, model_name)
        route_selection = "sticky_hit"
    else:
        primary_route = resolve_primary_route(settings, model_name)
        route_selection = "fresh_select"

    agent, result, resolved_model, elapsed_ms = _execute_once(primary_route, route_selection=route_selection)
    retried_after_refresh = False
    refreshed_route = None
    refresh_reason = determine_route_refresh_reason(result)

    refreshed_route = select_retry_route_for_result(primary_route, result)
    if refreshed_route is not None:
        retried_after_refresh = True
        route_selection = "refreshed_after_failure"
        agent, result, resolved_model, elapsed_ms = _execute_once(refreshed_route, route_selection=route_selection)

    active_route = refreshed_route or primary_route
    session_id = agent.session_id or session_state["session_id"]
    messages = result.get("messages") or []
    provider_usage = dict(result.get("provider_usage") or {})
    response_model = str(provider_usage.get("response_model") or "").strip()
    active_route_for_lease = dict(active_route)
    if response_model:
        active_route_for_lease["model"] = response_model
        resolved_model = response_model
    route_debug["last_error"] = str(result.get("error") or "") if result.get("error") else ""
    route_debug["last_route_selection"] = route_selection
    route_debug["last_failure_reason"] = refresh_reason if retried_after_refresh or not result.get("completed", True) else None
    route_debug["last_provider"] = result.get("provider") or active_route_for_lease.get("provider")
    route_debug["last_model"] = resolved_model
    route_debug["updated_at"] = int(time.time())
    route_metrics = increment_route_metric(route_metrics, route_selection)

    if result.get("error"):
        route_lease = expire_route_lease(
            route_lease if route_selection == "sticky_hit" else build_route_lease(active_route_for_lease, selection_reason=route_selection),
            error_text=str(result.get("error") or ""),
            failure_reason=refresh_reason,
        )
    else:
        route_lease = refresh_route_lease(
            route_lease if route_selection == "sticky_hit" else None,
            active_route_for_lease,
            selection_reason=(
                route_selection
                if route_selection != "sticky_hit"
                else (route_lease or {}).get("selection_reason")
            ),
        )

    save_session_state(
        effective_session_key,
        session_id,
        messages,
        route_lease=route_lease,
        route_debug=route_debug,
        route_metrics=route_metrics,
    )

    tool_names = extract_tool_names(messages)
    final_response = result.get("final_response")
    payload = {
        "status": "success" if result.get("completed", True) and not result.get("interrupted") else "partial",
        "session_key": effective_session_key,
        "session_id": session_id,
        "model": resolved_model,
        "provider": result.get("provider") or active_route_for_lease.get("provider") or settings.provider,
        "base_url": result.get("base_url") or active_route_for_lease.get("base_url") or settings.base_url,
        "input": task_input,
        "output": final_response,
        "completed": result.get("completed", True),
        "interrupted": result.get("interrupted", False),
        "api_calls": result.get("api_calls", 0),
        "tool_summary": tool_names,
        "elapsed_ms": elapsed_ms,
        "route_selection": route_selection,
        "route_lease_expires_at": (route_lease or {}).get("lease_expires_at"),
        "token_usage": {
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
        },
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "provider_usage": result.get("provider_usage") or {},
        "provider_usage_totals": result.get("provider_usage_totals") or {},
        "last_reasoning": result.get("last_reasoning"),
    }
    if retried_after_refresh:
        payload["retried_after_model_refresh"] = True
        payload["refreshed_model"] = active_route.get("model")
        payload["refreshed_provider"] = active_route.get("provider")
    if result.get("error"):
        payload["status"] = "error"
        payload["error"] = result["error"]
    logger.info(
        "[ModalInvoke] session_key=%s session_id=%s provider=%s model=%s route_selection=%s cache_key=%s retried_after_refresh=%s",
        effective_session_key,
        session_id,
        payload.get("provider"),
        payload.get("model"),
        route_selection,
        session_id,
        retried_after_refresh,
    )
    return payload
