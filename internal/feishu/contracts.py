from __future__ import annotations

from typing import Any, Mapping


CONTRACT_VERSION = "feishu_internal.v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def build_legacy_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(payload or {})
    if str(raw.get("contract_version") or "").strip() != CONTRACT_VERSION:
        return raw

    gateway_meta = _as_dict(raw.get("gateway_meta"))
    legacy_payload = _as_dict(gateway_meta.get("legacy_payload"))
    if legacy_payload:
        return legacy_payload

    ingress = _as_dict(raw.get("ingress"))
    route = _as_dict(raw.get("route"))
    session_hints = _as_dict(raw.get("session_hints"))
    merged = {
        **ingress,
        **route,
        **session_hints,
    }
    if raw.get("site_prefetch") is not None:
        merged["site_prefetch"] = raw.get("site_prefetch")
    for key in ("site_category", "site_intent", "target_domain", "target_url"):
        if gateway_meta.get(key) is not None:
            merged[key] = gateway_meta.get(key)
    for key in (
        "route_version",
        "provider_alias",
        "fallback_reason",
        "gateway_error_class",
        "misroute_detected",
        "cache_eligible",
        "cache_status",
        "ai_call_count",
        "capability_match",
        "preferred_model_selected",
        "model_catalog_version",
        "feedback_score_before",
        "feedback_score_after",
    ):
        if gateway_meta.get(key) is not None and merged.get(key) is None:
            merged[key] = gateway_meta.get(key)
    return merged


def build_response_envelope(response: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(response or {})
    return {
        "contract_version": CONTRACT_VERSION,
        "result": {
            "status": payload.get("status"),
            "route_hint": payload.get("route_hint"),
            "execution_mode": payload.get("execution_mode"),
            "final_response": payload.get("final_response"),
            "error": payload.get("error"),
            "action": payload.get("action"),
        },
        "send_plan": {
            "send_plan": payload.get("send_plan") or [],
            "action_plan": payload.get("action_plan") or [],
            "card": payload.get("card"),
        },
        "session_patch": {
            "session_state_before": payload.get("session_state_before") or {},
            "session_state_after": payload.get("session_state_after") or {},
            "event_key": payload.get("event_key"),
        },
        "reconcile": {
            "reconcile_required": bool(payload.get("reconcile_required")),
            "reconcile_summary": payload.get("reconcile_summary"),
            "browser_fallback_allowed": bool(payload.get("browser_fallback_allowed")),
        },
        "provider_metrics": {
            "provider_usage": payload.get("provider_usage") or {},
            "provider_plan": payload.get("provider_plan") or {},
            "provider_wait_measurement_mode": payload.get("provider_wait_measurement_mode"),
            "route_decision_reason": payload.get("route_decision_reason"),
            "fallback_reason": payload.get("fallback_reason"),
            "external_exec_candidate": payload.get("external_exec_candidate"),
            "llm_request": payload.get("llm_request") or {},
        },
        "legacy_response": payload,
    }
