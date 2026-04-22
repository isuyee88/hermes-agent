import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "feishu_triparty_tail.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("feishu_triparty_tail_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_normalize_modal_row_builds_stable_uid():
    module = _load_module()

    row = {
        "ts": 1776303807,
        "stage": "internal.agent_exec.done",
        "event_id": "evt_1",
        "message_id": "om_1",
        "correlation_id": "feishu:chat:evt_1",
        "chat_id": "oc_1",
        "route_hint": "modal_heavy_exec",
        "execution_mode": "modal_heavy_exec",
        "request_class": "text_plain",
        "content_modalities": ["text"],
        "route_family": "gateway_text",
        "gateway_route_name": "affiliate-general",
        "gateway_eligible": True,
        "dfmea_failure_mode": "nominal_path",
        "dfmea_control_phase": "gateway_execution",
        "dfmea_control_action": "continue_nominal_path",
        "agent_exec_elapsed_ms": 24682,
    }

    result = module._normalize_modal_row(row)

    assert result is not None
    assert result["uid"] == "modal:evt_1:internal.agent_exec.done:1776303807000:om_1"
    assert result["source"] == "modal_trace"
    assert result["event"] == "internal.agent_exec.done"
    assert result["details"]["request_class"] == "text_plain"
    assert result["details"]["gateway_route_name"] == "affiliate-general"
    assert result["details"]["dfmea_failure_mode"] == "nominal_path"
    assert result["details"]["agent_exec_elapsed_ms"] == 24682


def test_event_matches_filters_respects_event_and_chat_ids():
    module = _load_module()
    event = {
        "event_id": "evt_1",
        "correlation_id": "feishu:chat:evt_1",
        "chat_id": "oc_1",
    }

    assert module._event_matches_filters(
        event,
        event_ids={"evt_1"},
        correlation_ids=set(),
        chat_ids={"oc_1"},
    )
    assert not module._event_matches_filters(
        event,
        event_ids={"evt_2"},
        correlation_ids=set(),
        chat_ids=set(),
    )
