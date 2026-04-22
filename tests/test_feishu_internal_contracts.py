from internal.feishu.contracts import CONTRACT_VERSION, build_legacy_request_payload, build_response_envelope
from internal.feishu.planner import build_internal_plan


def _normalize_route_hint(value):
    normalized = str(value or "").strip()
    return normalized or "modal_heavy_exec"


def _message_requests_browser(text):
    return "http" in str(text or "") and "打开" in str(text or "")


def _is_freeish_model_name(value):
    normalized = str(value or "").strip().lower()
    return normalized == "openrouter/free" or normalized.endswith(":free")


def test_build_legacy_request_payload_extracts_envelope_legacy_payload():
    payload = {
        "contract_version": CONTRACT_VERSION,
        "ingress": {"correlation_id": "corr-1"},
        "gateway_meta": {
            "legacy_payload": {
                "session_key": "agent:main",
                "text": "hello",
                "route_hint": "modal_heavy_exec",
            }
        },
    }

    result = build_legacy_request_payload(payload)

    assert result["session_key"] == "agent:main"
    assert result["text"] == "hello"
    assert result["route_hint"] == "modal_heavy_exec"


def test_build_response_envelope_preserves_legacy_response():
    response = {
        "status": "ok",
        "route_hint": "fast_control",
        "execution_mode": "control_complete",
        "send_plan": [{"kind": "text", "content": "hello"}],
        "session_state_before": {"current_model": "a"},
        "session_state_after": {"current_model": "b"},
        "provider_plan": {"provider": "openrouter"},
    }

    envelope = build_response_envelope(response)

    assert envelope["contract_version"] == CONTRACT_VERSION
    assert envelope["result"]["route_hint"] == "fast_control"
    assert envelope["send_plan"]["send_plan"][0]["content"] == "hello"
    assert envelope["legacy_response"]["execution_mode"] == "control_complete"


def test_build_internal_plan_keeps_browser_requests_in_modal_path():
    result = build_internal_plan(
        {
            "task_kind": "text",
            "message_type": "text",
            "text": "请打开 https://example.com 并截图给我",
            "attachment_refs": [],
        },
        normalize_route_hint=_normalize_route_hint,
        message_requests_browser=_message_requests_browser,
        is_freeish_model_name=_is_freeish_model_name,
    )

    assert result["route_hint"] == "cf_browser_first"
    assert result["execution_mode"] == "cf_browser_first"
    assert result["external_exec_candidate"] is False
