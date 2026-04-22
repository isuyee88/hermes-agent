from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Mapping


_GATEWAY_ERROR_CLASS_ALIASES = {
    "auth_or_secret_missing": "provider_permission_denied",
    "provider_auth_error": "provider_permission_denied",
    "provider_model_invalid": "provider_model_not_found",
    "rate_exhausted": "rate_limited",
    "provider_timeout": "timeout",
    "config_hard_fail": "payload_incompatible",
    "provider_http_error": "upstream_5xx",
    "cf_execution_error": "upstream_5xx",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower() if value is not None else ""
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return None


def _coerce_optional_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    normalized = str(value).strip() if value is not None else ""
    if not normalized:
        return None
    try:
        if "." in normalized:
            return float(normalized)
        return int(normalized)
    except (TypeError, ValueError):
        return None


def _normalize_gateway_error_class(raw_value: Any, *, error_text: str = "") -> str:
    normalized = str(raw_value or "").strip().lower()
    if not normalized and error_text:
        error_lower = error_text.strip().lower()
        if "catalog" in error_lower and "stale" in error_lower:
            return "catalog_stale"
        if "timeout" in error_lower or "408" in error_lower:
            return "timeout"
        if "429" in error_lower or "rate" in error_lower:
            return "rate_limited"
        if "permission" in error_lower or "unauthorized" in error_lower or "forbidden" in error_lower:
            return "provider_permission_denied"
        if "404" in error_lower or "not found" in error_lower:
            return "provider_model_not_found"
        if "misrouted_request_class" in error_lower:
            return "misrouted_request_class"
        if "400" in error_lower or "payload" in error_lower:
            return "payload_incompatible"
        if any(code in error_lower for code in ("500", "502", "503", "504", "524", "5xx")):
            return "upstream_5xx"
        return ""
    if not normalized:
        return ""
    if normalized == "catalog_stale":
        return normalized
    if normalized in {"provider_permission_denied", "provider_model_not_found", "rate_limited", "upstream_5xx", "timeout", "payload_incompatible", "misrouted_request_class"}:
        return normalized
    if "catalog" in normalized and "stale" in normalized:
        return "catalog_stale"
    return _GATEWAY_ERROR_CLASS_ALIASES.get(normalized, normalized)


def _build_runtime_observability_fields(
    payload: Mapping[str, Any],
    *,
    agent_result: Mapping[str, Any] | None = None,
    provider_usage: Mapping[str, Any] | None = None,
    gateway_error_class: str = "",
    fallback_reason: str = "",
) -> dict[str, Any]:
    payload_dict = dict(payload or {})
    ingress_meta = _as_dict(payload_dict.get("_hermes_ingress"))
    gateway_meta = _as_dict(payload_dict.get("_hermes_gateway_meta"))
    provider_plan = _as_dict(payload_dict.get("provider_plan"))
    usage = _as_dict(provider_usage)
    result = _as_dict(agent_result)

    cache_eligible = _coerce_optional_bool(
        payload_dict.get("cache_eligible")
        if payload_dict.get("cache_eligible") is not None
        else gateway_meta.get("cache_eligible")
    )
    if cache_eligible is None:
        cache_mode = str(provider_plan.get("cache_mode") or "").strip().lower()
        if cache_mode:
            cache_eligible = cache_mode != "skip"

    cache_status = str(
        payload_dict.get("cache_status")
        or usage.get("cache_status")
        or gateway_meta.get("cache_status")
        or ""
    ).strip()
    if not cache_status and cache_eligible is not None:
        cache_discount = _coerce_optional_number(
            usage.get("cache_discount")
            if usage.get("cache_discount") is not None
            else _as_dict(result.get("provider_usage_totals")).get("cache_discount_usd")
        )
        if isinstance(cache_discount, (int, float)) and float(cache_discount) > 0:
            cache_status = "hit"
        elif cache_eligible:
            cache_status = "eligible_no_signal"
        else:
            cache_status = "bypass"

    capability_match = _coerce_optional_bool(
        payload_dict.get("capability_match")
        if payload_dict.get("capability_match") is not None
        else gateway_meta.get("capability_match")
    )
    if capability_match is None:
        route_hint = str(payload_dict.get("route_hint") or "").strip()
        task_kind = str(payload_dict.get("task_kind") or payload_dict.get("message_type") or "").strip().lower()
        request_class = str(payload_dict.get("request_class") or "").strip().lower()
        requires_browser = bool(payload_dict.get("requires_browser"))
        requires_media = bool(payload_dict.get("requires_media_hydration"))
        if route_hint == "fast_control":
            capability_match = task_kind == "command"
        elif route_hint == "cf_browser_first":
            capability_match = requires_browser
        elif route_hint == "modal_heavy_exec" and request_class in {"text_plain", "text_coding", ""}:
            capability_match = not requires_browser and not requires_media

    preferred_model_selected = _coerce_optional_bool(
        payload_dict.get("preferred_model_selected")
        if payload_dict.get("preferred_model_selected") is not None
        else gateway_meta.get("preferred_model_selected")
    )
    if preferred_model_selected is None:
        provider_fallback_used = _coerce_optional_bool(result.get("provider_fallback_used"))
        preferred_model_selected = False if provider_fallback_used is True else None
        planned_model = str(provider_plan.get("model") or "").strip().lower()
        response_model = str(usage.get("response_model") or "").strip().lower()
        if response_model and planned_model:
            preferred_model_selected = response_model == planned_model
        elif preferred_model_selected is None and provider_fallback_used is False:
            preferred_model_selected = True

    normalized_gateway_error = _normalize_gateway_error_class(
        payload_dict.get("gateway_error_class") or gateway_error_class or fallback_reason
    )
    misroute_detected = _coerce_optional_bool(
        payload_dict.get("misroute_detected")
        if payload_dict.get("misroute_detected") is not None
        else gateway_meta.get("misroute_detected")
    )
    if misroute_detected is None and normalized_gateway_error:
        misroute_detected = normalized_gateway_error == "misrouted_request_class"

    return {
        "route_version": str(
            payload_dict.get("route_version")
            or ingress_meta.get("route_version")
            or gateway_meta.get("route_version")
            or "modal_internal_v1"
        ).strip(),
        "provider_alias": str(
            payload_dict.get("provider_alias")
            or usage.get("provider")
            or provider_plan.get("provider")
            or gateway_meta.get("provider_alias")
            or ""
        ).strip(),
        "fallback_reason": str(fallback_reason or payload_dict.get("fallback_reason") or "").strip(),
        "gateway_error_class": normalized_gateway_error,
        "misroute_detected": bool(misroute_detected) if misroute_detected is not None else False,
        "cache_eligible": cache_eligible,
        "cache_status": cache_status,
        "ai_call_count": _coerce_optional_number(
            payload_dict.get("ai_call_count")
            if payload_dict.get("ai_call_count") is not None
            else result.get("api_calls")
        ),
        "capability_match": capability_match,
        "preferred_model_selected": preferred_model_selected,
        "model_catalog_version": str(
            payload_dict.get("model_catalog_version")
            or gateway_meta.get("model_catalog_version")
            or ""
        ).strip(),
        "feedback_score_before": _coerce_optional_number(
            payload_dict.get("feedback_score_before")
            if payload_dict.get("feedback_score_before") is not None
            else gateway_meta.get("feedback_score_before")
        ),
        "feedback_score_after": _coerce_optional_number(
            payload_dict.get("feedback_score_after")
            if payload_dict.get("feedback_score_after") is not None
            else gateway_meta.get("feedback_score_after")
        ),
    }


async def run_internal_agent_exec(
    payload: Mapping[str, Any],
    *,
    infer_route_hint: Callable[[Mapping[str, Any]], str],
    get_runtime: Callable[[], Awaitable[Any]],
    build_source: Callable[[Mapping[str, Any]], Any],
    apply_pending_reconciles: Callable[..., dict[str, Any]],
    append_trace: Callable[..., None],
    logger: Any,
    hydrate_event_media: Callable[[Mapping[str, Any], Any], Awaitable[Any]],
    build_event: Callable[[Mapping[str, Any]], Any],
    build_session_state: Callable[..., dict[str, Any]],
    build_send_plan: Callable[..., list[dict[str, Any]]],
    build_result: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    payload_dict = dict(payload or {})
    correlation_id = str(payload_dict.get("correlation_id") or "").strip()
    event_id = str(payload_dict.get("event_id") or "").strip()
    chat_id = str(payload_dict.get("chat_id") or "").strip()
    message_id = str(payload_dict.get("message_id") or "").strip()
    session_key = str(payload_dict.get("session_key") or "").strip()
    route_hint = infer_route_hint(payload_dict)
    route_decision_reason = str(payload_dict.get("route_decision_reason") or "").strip()
    fallback_reason = str(payload_dict.get("fallback_reason") or "").strip()
    browser_fallback_allowed = route_hint == "cf_browser_first"
    gateway_request_meta = payload_dict.get("_hermes_gateway_request")
    gateway_request_meta = dict(gateway_request_meta) if isinstance(gateway_request_meta, Mapping) else {}
    started_at = time.perf_counter()
    runtime_bootstrap_started_at = time.perf_counter()
    append_trace(
        "internal.agent_exec.start",
        payload_dict,
        correlation_id=correlation_id,
        route_decision_reason=route_decision_reason,
        fallback_reason=fallback_reason,
        gateway_hop=str(gateway_request_meta.get("gateway_hop") or ""),
        gateway_script=str(gateway_request_meta.get("gateway_script") or ""),
    )
    logger.warning(
        "[Feishu] internal agent exec start correlation_id=%s event_id=%s chat_id=%s message_id=%s",
        correlation_id or "none",
        event_id or "none",
        chat_id or "none",
        message_id or "none",
    )
    runtime = await get_runtime()
    runtime_bootstrap_elapsed_ms = int((time.perf_counter() - runtime_bootstrap_started_at) * 1000)
    runner = runtime.runner
    adapter = runtime.adapter
    adapter.captured_operations.clear()
    agent_observer_key = session_key or correlation_id or event_id or message_id
    agent_observation: dict[str, Any] = {}
    runner_observers = getattr(runner, "_agent_run_observers", None)
    if not isinstance(runner_observers, dict):
        runner_observers = {}
        setattr(runner, "_agent_run_observers", runner_observers)

    def _capture_agent_observation(value: Mapping[str, Any] | None) -> None:
        if isinstance(value, Mapping):
            agent_observation.update(dict(value))

    if agent_observer_key:
        runner_observers[agent_observer_key] = _capture_agent_observation
    source = build_source(payload_dict)
    pending_reconcile_started_at = time.perf_counter()
    reconcile_summary = apply_pending_reconciles(runner=runner, source=source, payload=payload_dict)
    pending_reconcile_apply_elapsed_ms = int((time.perf_counter() - pending_reconcile_started_at) * 1000)
    if int(reconcile_summary.get("received") or 0) > 0:
        append_trace(
            "internal.agent_exec.reconcile",
            payload_dict,
            correlation_id=correlation_id,
            route_decision_reason=route_decision_reason,
            fallback_reason=fallback_reason,
            reconcile_summary=reconcile_summary,
        )
        logger.warning(
            "[Feishu] internal agent exec reconcile correlation_id=%s session_key=%s received=%s applied=%s skipped=%s",
            correlation_id or "none",
            reconcile_summary.get("session_key") or "none",
            reconcile_summary.get("received"),
            reconcile_summary.get("applied"),
            reconcile_summary.get("skipped"),
        )
    event_hydration_started_at = time.perf_counter()
    event = await hydrate_event_media(payload_dict, build_event(payload_dict))
    event_hydration_elapsed_ms = int((time.perf_counter() - event_hydration_started_at) * 1000)
    session_snapshot_elapsed_ms = 0
    session_state_before_started_at = time.perf_counter()
    session_state_before = build_session_state(runner=runner, source=event.source)
    session_snapshot_elapsed_ms += int((time.perf_counter() - session_state_before_started_at) * 1000)
    try:
        response_text = await runner._handle_message(event)
        session_state_after_started_at = time.perf_counter()
        session_state_after = build_session_state(runner=runner, source=event.source)
        session_snapshot_elapsed_ms += int((time.perf_counter() - session_state_after_started_at) * 1000)
        send_plan_started_at = time.perf_counter()
        send_plan = build_send_plan(adapter=adapter, response_text=response_text)
        send_plan_build_elapsed_ms = int((time.perf_counter() - send_plan_started_at) * 1000)
        agent_exec_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        agent_phase_timings = dict(agent_observation.get("phase_timings") or {})
        agent_result = dict(agent_observation.get("agent_result") or {})
        sync_phase_timings = dict(agent_result.get("sync_phase_timings") or {})
        gateway_run_agent_total_elapsed_ms = (
            int(agent_phase_timings.get("run_agent_elapsed_ms"))
            if isinstance(agent_phase_timings.get("run_agent_elapsed_ms"), (int, float))
            else None
        )
        gateway_pre_agent_elapsed_ms = (
            int(agent_phase_timings.get("pre_agent_elapsed_ms"))
            if isinstance(agent_phase_timings.get("pre_agent_elapsed_ms"), (int, float))
            else None
        )
        provider_wait_elapsed_ms = (
            int(agent_result.get("provider_wait_elapsed_ms"))
            if isinstance(agent_result.get("provider_wait_elapsed_ms"), (int, float))
            else None
        )
        tool_exec_elapsed_ms = (
            int(agent_result.get("tool_exec_elapsed_ms"))
            if isinstance(agent_result.get("tool_exec_elapsed_ms"), (int, float))
            else None
        )
        response_finalize_elapsed_ms = (
            int(sync_phase_timings.get("result_finalize_elapsed_ms"))
            if isinstance(sync_phase_timings.get("result_finalize_elapsed_ms"), (int, float))
            else None
        )
        provider_wait_measurement_mode = str(agent_result.get("provider_wait_measurement_mode") or "").strip()
        if not provider_wait_measurement_mode:
            provider_wait_measurement_mode = (
                "provider_wait_observed"
                if isinstance(provider_wait_elapsed_ms, int)
                else ("legacy_proxy" if gateway_run_agent_total_elapsed_ms is not None else "insufficient_data")
            )
        hermes_overhead_elapsed_ms = (
            max(gateway_run_agent_total_elapsed_ms - provider_wait_elapsed_ms - tool_exec_elapsed_ms, 0)
            if isinstance(gateway_run_agent_total_elapsed_ms, int)
            and isinstance(provider_wait_elapsed_ms, int)
            and isinstance(tool_exec_elapsed_ms, int)
            else None
        )
        agent_model_elapsed_ms = gateway_run_agent_total_elapsed_ms if isinstance(gateway_run_agent_total_elapsed_ms, int) else None
        agent_non_model_elapsed_ms = (
            max(agent_exec_elapsed_ms - agent_model_elapsed_ms, 0) if isinstance(agent_model_elapsed_ms, int) else None
        )
        provider_usage = dict(agent_result.get("provider_usage") or {})
        provider_usage_totals = dict(agent_result.get("provider_usage_totals") or {})
        runtime_observability = _build_runtime_observability_fields(
            payload_dict,
            agent_result=agent_result,
            provider_usage=provider_usage,
            gateway_error_class="",
            fallback_reason=fallback_reason,
        )
        append_trace(
            "internal.agent_exec.done",
            payload_dict,
            correlation_id=correlation_id,
            session_key=session_key,
            chat_id=chat_id,
            message_id=message_id,
            route_decision_reason=route_decision_reason,
            agent_exec_elapsed_ms=agent_exec_elapsed_ms,
            agent_model_elapsed_ms=agent_model_elapsed_ms,
            agent_non_model_elapsed_ms=agent_non_model_elapsed_ms,
            runtime_bootstrap_elapsed_ms=runtime_bootstrap_elapsed_ms,
            pending_reconcile_apply_elapsed_ms=pending_reconcile_apply_elapsed_ms,
            event_hydration_elapsed_ms=event_hydration_elapsed_ms,
            gateway_pre_agent_elapsed_ms=gateway_pre_agent_elapsed_ms,
            gateway_run_agent_total_elapsed_ms=gateway_run_agent_total_elapsed_ms,
            provider_wait_elapsed_ms=provider_wait_elapsed_ms,
            tool_exec_elapsed_ms=tool_exec_elapsed_ms,
            response_finalize_elapsed_ms=response_finalize_elapsed_ms,
            send_plan_build_elapsed_ms=send_plan_build_elapsed_ms,
            session_snapshot_elapsed_ms=session_snapshot_elapsed_ms,
            hermes_overhead_elapsed_ms=hermes_overhead_elapsed_ms,
            provider_wait_measurement_mode=provider_wait_measurement_mode,
            provider_attempt_count=agent_result.get("provider_attempt_count"),
            provider_retry_count=agent_result.get("provider_retry_count"),
            provider_fallback_used=agent_result.get("provider_fallback_used"),
            send_plan_operation_count=len(send_plan),
            response_length=len(str(response_text or "")),
            route_hint=route_hint,
            execution_mode="modal_heavy_exec",
            provider_usage=provider_usage,
            provider_usage_totals=provider_usage_totals,
            gateway_hop=str(gateway_request_meta.get("gateway_hop") or ""),
            gateway_script=str(gateway_request_meta.get("gateway_script") or ""),
            **runtime_observability,
        )
        logger.warning(
            "[Feishu] internal agent exec done correlation_id=%s event_id=%s chat_id=%s message_id=%s agent_exec_elapsed_ms=%s agent_model_elapsed_ms=%s send_plan_operation_count=%s response_length=%s route_hint=%s",
            correlation_id or "none",
            event_id or "none",
            chat_id or "none",
            message_id or "none",
            agent_exec_elapsed_ms,
            agent_model_elapsed_ms,
            len(send_plan),
            len(str(response_text or "")),
            route_hint,
        )
        return build_result(
            status="ok",
            route_hint=route_hint,
            execution_mode="modal_heavy_exec",
            session_state_before=session_state_before,
            session_state_after=session_state_after,
            final_response=str(response_text or ""),
            send_plan=send_plan,
            reconcile_required=bool(send_plan or str(response_text or "").strip()),
            browser_fallback_allowed=browser_fallback_allowed,
            correlation_id=correlation_id,
            reconcile_summary=reconcile_summary,
            route_decision_reason=route_decision_reason,
            fallback_reason=fallback_reason,
            runtime_bootstrap_elapsed_ms=runtime_bootstrap_elapsed_ms,
            pending_reconcile_apply_elapsed_ms=pending_reconcile_apply_elapsed_ms,
            event_hydration_elapsed_ms=event_hydration_elapsed_ms,
            gateway_pre_agent_elapsed_ms=gateway_pre_agent_elapsed_ms,
            gateway_run_agent_total_elapsed_ms=gateway_run_agent_total_elapsed_ms,
            provider_wait_elapsed_ms=provider_wait_elapsed_ms,
            tool_exec_elapsed_ms=tool_exec_elapsed_ms,
            response_finalize_elapsed_ms=response_finalize_elapsed_ms,
            send_plan_build_elapsed_ms=send_plan_build_elapsed_ms,
            session_snapshot_elapsed_ms=session_snapshot_elapsed_ms,
            hermes_overhead_elapsed_ms=hermes_overhead_elapsed_ms,
            provider_wait_measurement_mode=provider_wait_measurement_mode,
        )
    except Exception as exc:
        agent_exec_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        normalized_gateway_error = _normalize_gateway_error_class(fallback_reason, error_text=str(exc))
        runtime_observability = _build_runtime_observability_fields(
            payload_dict,
            agent_result=None,
            provider_usage=None,
            gateway_error_class=normalized_gateway_error,
            fallback_reason=fallback_reason,
        )
        append_trace(
            "internal.agent_exec.error",
            payload_dict,
            correlation_id=correlation_id,
            agent_exec_elapsed_ms=agent_exec_elapsed_ms,
            error=str(exc),
            route_hint=route_hint,
            route_decision_reason=route_decision_reason,
            runtime_bootstrap_elapsed_ms=runtime_bootstrap_elapsed_ms,
            pending_reconcile_apply_elapsed_ms=pending_reconcile_apply_elapsed_ms,
            event_hydration_elapsed_ms=event_hydration_elapsed_ms,
            session_snapshot_elapsed_ms=session_snapshot_elapsed_ms,
            gateway_hop=str(gateway_request_meta.get("gateway_hop") or ""),
            gateway_script=str(gateway_request_meta.get("gateway_script") or ""),
            **runtime_observability,
        )
        logger.exception(
            "[Feishu] internal agent exec failed correlation_id=%s event_id=%s chat_id=%s message_id=%s agent_exec_elapsed_ms=%s route_hint=%s",
            correlation_id or "none",
            event_id or "none",
            chat_id or "none",
            message_id or "none",
            agent_exec_elapsed_ms,
            route_hint,
        )
        raise
    finally:
        if agent_observer_key and isinstance(runner_observers, dict):
            runner_observers.pop(agent_observer_key, None)


async def run_internal_agent_plan(
    payload: Mapping[str, Any],
    *,
    infer_route_hint: Callable[[Mapping[str, Any]], str],
    get_runtime: Callable[[], Awaitable[Any]],
    build_source: Callable[[Mapping[str, Any]], Any],
    build_session_state: Callable[..., dict[str, Any]],
    is_external_exec_candidate: Callable[[Mapping[str, Any], str], bool],
    build_external_conversation_history: Callable[..., list[dict[str, str]]],
    build_internal_plan: Callable[..., dict[str, Any]],
    append_trace: Callable[..., None],
    logger: Any,
) -> dict[str, Any]:
    payload_dict = dict(payload or {})
    correlation_id = str(payload_dict.get("correlation_id") or "").strip()
    route_hint = infer_route_hint(payload_dict)
    gateway_request_meta = payload_dict.get("_hermes_gateway_request")
    gateway_request_meta = dict(gateway_request_meta) if isinstance(gateway_request_meta, Mapping) else {}
    runtime = await get_runtime()
    runner = runtime.runner
    source = build_source(payload_dict)
    session_state_before = build_session_state(runner=runner, source=source)
    llm_request: dict[str, Any] = {}
    if is_external_exec_candidate(payload_dict, route_hint):
        llm_request = {
            "model": str(session_state_before.get("current_model") or ""),
            "messages": build_external_conversation_history(
                runner=runner,
                source=source,
                latest_user_text=str(payload_dict.get("text") or ""),
            ),
        }
    plan = build_internal_plan(payload_dict, session_state_before=session_state_before, llm_request=llm_request)
    append_trace(
        "internal.agent_plan.done",
        payload_dict,
        correlation_id=correlation_id,
        route_hint=route_hint,
        execution_mode=str(plan.get("execution_mode") or ""),
        external_exec_candidate=bool(plan.get("external_exec_candidate")),
        gateway_hop=str(gateway_request_meta.get("gateway_hop") or ""),
        gateway_script=str(gateway_request_meta.get("gateway_script") or ""),
    )
    logger.warning(
        "[Feishu] internal agent plan correlation_id=%s route_hint=%s execution_mode=%s external_exec_candidate=%s",
        correlation_id or "none",
        route_hint,
        str(plan.get("execution_mode") or "none"),
        bool(plan.get("external_exec_candidate")),
    )
    return plan
