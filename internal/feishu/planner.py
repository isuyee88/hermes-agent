from __future__ import annotations

import os
from typing import Any, Callable, Mapping


NormalizeRouteHint = Callable[[Any], str]
MessageRequestsBrowser = Callable[[str], bool]
IsFreeishModel = Callable[[Any], bool]


def infer_route_hint(
    payload: Mapping[str, Any],
    *,
    normalize_route_hint: NormalizeRouteHint,
    message_requests_browser: MessageRequestsBrowser,
) -> str:
    explicit = normalize_route_hint(payload.get("route_hint"))
    if explicit != "modal_heavy_exec":
        return explicit
    task_kind = str(payload.get("task_kind") or payload.get("message_type") or "").strip().lower()
    if task_kind == "command":
        return "fast_control"
    attachment_refs = payload.get("attachment_refs")
    if isinstance(attachment_refs, list) and attachment_refs:
        return "modal_heavy_exec"
    if message_requests_browser(str(payload.get("text") or "")):
        return "cf_browser_first"
    return "modal_heavy_exec"


def build_internal_action_plan(send_plan: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    action_plan: list[dict[str, Any]] = []
    for item in list(send_plan or []):
        if isinstance(item, dict):
            action_plan.append(dict(item))
    return action_plan


def is_external_exec_candidate(
    payload: Mapping[str, Any],
    route_hint: str,
    *,
    normalize_route_hint: NormalizeRouteHint,
    message_requests_browser: MessageRequestsBrowser,
) -> bool:
    if normalize_route_hint(route_hint) != "modal_heavy_exec":
        return False
    task_kind = str(payload.get("task_kind") or payload.get("message_type") or "").strip().lower()
    if task_kind not in {"text", ""}:
        return False
    attachment_refs = payload.get("attachment_refs")
    if isinstance(attachment_refs, list) and attachment_refs:
        return False
    text = str(payload.get("text") or "").strip()
    if not text or text.startswith("/"):
        return False
    if message_requests_browser(text):
        return False
    return True


def build_provider_plan(
    session_state: Mapping[str, Any] | None,
    *,
    is_freeish_model_name: IsFreeishModel,
) -> dict[str, Any]:
    state = dict(session_state or {})
    route_debug = dict(state.get("route_debug") or {})
    base_url = str(route_debug.get("base_url") or "").strip()
    current_model = str(state.get("current_model") or "").strip()
    current_provider = str(state.get("current_provider") or "").strip()
    last_model = str(route_debug.get("last_model") or "").strip()
    last_provider = str(route_debug.get("last_provider") or "").strip()
    effective_model = current_model
    effective_provider = current_provider
    default_external_exec_model = str(
        os.getenv(
            "HERMES_FEISHU_EXTERNAL_EXEC_MODEL",
            "mistralai/mistral-small-3.1-24b-instruct",
        )
        or "mistralai/mistral-small-3.1-24b-instruct"
    ).strip()
    if not effective_model or is_freeish_model_name(effective_model):
        effective_model = last_model if last_model and not is_freeish_model_name(last_model) else default_external_exec_model
    if not effective_provider:
        effective_provider = last_provider or "openrouter"
    request_timeout_ms = max(1000, int(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_TIMEOUT_MS", "18000") or "18000"))
    max_attempts = max(1, min(5, int(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_MAX_ATTEMPTS", "2") or "2")))
    retry_delay_ms = max(0, min(5000, int(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_RETRY_DELAY_MS", "250") or "250")))
    backoff = str(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_BACKOFF", "linear") or "linear").strip().lower()
    if backoff not in {"constant", "linear", "exponential"}:
        backoff = "linear"
    fallback_model = str(
        os.getenv(
            "HERMES_FEISHU_EXTERNAL_EXEC_FALLBACK_MODEL",
            "deepseek/deepseek-chat-v3-0324" if effective_model != "deepseek/deepseek-chat-v3-0324" else "",
        )
        or ""
    ).strip()
    fallback_provider = str(os.getenv("HERMES_FEISHU_EXTERNAL_EXEC_FALLBACK_PROVIDER") or effective_provider or "").strip().lower()
    cache_mode = str(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_CACHE_MODE", "ttl") or "ttl").strip().lower()
    if cache_mode not in {"skip", "ttl"}:
        cache_mode = "skip"
    cache_ttl_seconds = max(60, min(31 * 24 * 3600, int(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_CACHE_TTL_SECONDS", "300") or "300")))
    cache_scope = str(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_CACHE_SCOPE", "chat") or "chat").strip().lower()
    if cache_scope not in {"user", "chat", "global"}:
        cache_scope = "user"
    current_personality = str(state.get("current_personality") or "").strip().lower()
    plan: dict[str, Any] = {
        "mode": "cloudflare_workflow_candidate",
        "wait_strategy": "cloudflare_wait",
        "task_profile": "plain_text_llm",
        "reason": "plain_text_without_attachments_or_browser",
        "model": effective_model,
        "provider": effective_provider,
        "base_url": base_url,
        "request_timeout_ms": request_timeout_ms,
        "max_attempts": max_attempts,
        "retry_delay_ms": retry_delay_ms,
        "backoff": backoff,
        "cache_mode": cache_mode,
        "cache_scope": cache_scope,
        "byok_alias": str(os.getenv("HERMES_CF_AI_GATEWAY_BYOK_ALIAS", "default") or "default").strip(),
    }
    if cache_mode == "ttl":
        plan["cache_ttl_seconds"] = cache_ttl_seconds
    if current_model:
        plan["session_model"] = current_model
    if last_model:
        plan["last_model"] = last_model
    if fallback_model and fallback_model != effective_model:
        plan["fallback_model"] = fallback_model
        plan["fallback_provider"] = fallback_provider or effective_provider
    if current_personality:
        plan["personality"] = current_personality
    return plan


def build_internal_result(
    *,
    status: str,
    route_hint: str,
    execution_mode: str,
    session_state_before: Mapping[str, Any] | None,
    session_state_after: Mapping[str, Any] | None,
    normalize_route_hint: NormalizeRouteHint,
    send_plan: list[dict[str, Any]] | None = None,
    final_response: str | None = None,
    provider_usage: Mapping[str, Any] | None = None,
    reconcile_required: bool = False,
    browser_fallback_allowed: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    normalized_send_plan = list(send_plan or [])
    return {
        "status": status,
        "route_hint": normalize_route_hint(route_hint),
        "execution_mode": str(execution_mode or "modal_heavy_exec").strip() or "modal_heavy_exec",
        "final_response": str(final_response or ""),
        "send_plan": normalized_send_plan,
        "action_plan": build_internal_action_plan(normalized_send_plan),
        "session_state_before": dict(session_state_before or {}),
        "session_state_after": dict(session_state_after or {}),
        "provider_usage": dict(provider_usage or {}),
        "reconcile_required": bool(reconcile_required),
        "browser_fallback_allowed": bool(browser_fallback_allowed),
        **extra,
    }


def build_internal_plan(
    payload: Mapping[str, Any],
    *,
    session_state_before: Mapping[str, Any] | None = None,
    llm_request: Mapping[str, Any] | None = None,
    normalize_route_hint: NormalizeRouteHint,
    message_requests_browser: MessageRequestsBrowser,
    is_freeish_model_name: IsFreeishModel,
) -> dict[str, Any]:
    route_hint = infer_route_hint(
        payload,
        normalize_route_hint=normalize_route_hint,
        message_requests_browser=message_requests_browser,
    )
    external_exec_candidate = is_external_exec_candidate(
        payload,
        route_hint,
        normalize_route_hint=normalize_route_hint,
        message_requests_browser=message_requests_browser,
    )
    execution_mode = "deferred_reconcile" if external_exec_candidate else route_hint
    if execution_mode not in {"control_complete", "native_io_complete", "cf_browser_first", "modal_heavy_exec", "deferred_reconcile"}:
        execution_mode = "modal_heavy_exec"
    normalized_session_state_before = dict(session_state_before or {})
    provider_plan: dict[str, Any] = {}
    normalized_llm_request = dict(llm_request or {})
    if external_exec_candidate:
        provider_plan = build_provider_plan(
            normalized_session_state_before,
            is_freeish_model_name=is_freeish_model_name,
        )
        llm_model = str(normalized_llm_request.get("model") or "").strip()
        provider_model = str(provider_plan.get("model") or "").strip()
        if llm_model and provider_model and is_freeish_model_name(llm_model) and not is_freeish_model_name(provider_model):
            normalized_llm_request["model"] = provider_model
    return build_internal_result(
        status="ok",
        route_hint=route_hint,
        execution_mode=execution_mode,
        session_state_before=normalized_session_state_before,
        session_state_after=normalized_session_state_before,
        normalize_route_hint=normalize_route_hint,
        send_plan=[],
        final_response="",
        provider_usage={},
        reconcile_required=bool(external_exec_candidate),
        browser_fallback_allowed=route_hint == "cf_browser_first",
        provider_plan=provider_plan,
        external_exec_candidate=external_exec_candidate,
        llm_request=normalized_llm_request,
    )
