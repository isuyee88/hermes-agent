from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from internal.feishu.webhook_fast_paths import handle_event_fast_paths


@pytest.mark.asyncio
async def test_message_read_fast_path_appends_correlation_traces(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_MODAL_DATA_DIR", str(tmp_path))
    trace_path = tmp_path / "feishu_trace.jsonl"
    now = int(time.time())
    trace_path.write_text(
        (
            '{"ts": %d, "stage": "gateway.send.done", "event_type": "im.message.receive_v1", '
            '"event_id": "evt_in", "message_id": "om_in_1", "send_message_id": "om_out_1", '
            '"send_success": true, "session_key": "agent:main:feishu:group:oc_demo:ou_demo", '
            '"correlation_id": "feishu:oc_demo:evt_in", "chat_id": "oc_demo"}\n'
        )
        % now,
        encoding="utf-8",
    )

    appended: list[tuple[str, dict]] = []

    def _append_trace(stage: str, _payload: dict, **extra: object) -> None:
        appended.append((stage, dict(extra)))

    def _build_ack_response(_payload: dict, body: dict, **_: object) -> dict:
        return body

    deps = SimpleNamespace(
        is_session_warmup_event=lambda _event_type: False,
        extract_message_read_event_info=lambda payload: {
            "event_id": "evt_read",
            "reader_open_id": "ou_reader",
            "read_time": 1_712_970_029_000,
            "message_id_list": ["om_out_1"],
            "message_count": 1,
        },
        capture_phase_elapsed=lambda *_args, **_kwargs: None,
        append_trace=_append_trace,
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        build_ack_response=_build_ack_response,
    )
    payload = {
        "header": {"event_type": "im.message.message_read_v1", "event_id": "evt_read"},
        "event": {"message_id_list": ["om_out_1"]},
    }

    response = await handle_event_fast_paths(
        payload,
        event_id="evt_read",
        event_type="im.message.message_read_v1",
        request_started_at=time.perf_counter(),
        request_started_at_ms=int(time.time() * 1000),
        phase_timings={},
        deps=deps,
    )

    assert response == {"code": 0, "msg": "accepted"}
    stages = [stage for stage, _extra in appended]
    assert "webhook.message_read" in stages
    assert "webhook.message_read.correlated" in stages
    correlated = next(extra for stage, extra in appended if stage == "webhook.message_read.correlated")
    assert correlated["matched_kind"] == "reply"
    assert correlated["reply_send_message_id"] == "om_out_1"
    assert correlated["session_key"] == "agent:main:feishu:group:oc_demo:ou_demo"
