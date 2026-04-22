from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "feishu_chain_status.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("feishu_chain_status_test_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyze_chain_status_reports_successful_delivery(tmp_path: Path):
    module = _load_module()

    sessions_path = tmp_path / "sessions.json"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:feishu:group:oc_demo:on_demo": {
                    "session_key": "agent:main:feishu:group:oc_demo:on_demo",
                    "session_id": "20260421_050249_d828c9f2",
                    "updated_at": "2026-04-21T05:30:29.714454",
                    "display_name": "CEO 工作汇报",
                    "platform": "feishu",
                    "chat_type": "group",
                    "route_lease": {
                        "provider": "openrouter",
                        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
                    },
                    "origin": {
                        "chat_id": "oc_demo",
                        "user_id": "ou_demo",
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    transcript_path = tmp_path / "20260421_050249_d828c9f2.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "hello"}, ensure_ascii=False),
                json.dumps({"role": "assistant", "content": "正常回复"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    agent_log = tmp_path / "agent.log"
    agent_log.write_text(
        "\n".join(
            [
                "2026-04-21 05:30:29,710 INFO gateway.run: response ready: platform=feishu chat=oc_demo time=21.7s api_calls=1 response=223 chars",
                "2026-04-21 05:30:30,910 INFO gateway.platforms.base: [Feishu] Background task send complete session=agent:main:feishu:group:oc_demo:on_demo message_id=om_src success=True send_message_id=om_reply error=None",
            ]
        ),
        encoding="utf-8",
    )

    status = module.analyze_chain_status(
        sessions_json=sessions_path,
        agent_log=agent_log,
        errors_log=None,
        transcript_path=transcript_path,
    )

    assert status.status == "ok"
    assert status.blocker == "none"
    assert status.delivery is not None
    assert status.delivery.success is True
    assert status.delivery.send_message_id == "om_reply"
    assert status.response_ready is not None
    assert status.response_ready.time_seconds == 21.7


def test_analyze_chain_status_reports_provider_not_configured(tmp_path: Path):
    module = _load_module()

    sessions_path = tmp_path / "sessions.json"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:feishu:group:oc_demo:on_demo": {
                    "session_key": "agent:main:feishu:group:oc_demo:on_demo",
                    "session_id": "20260421_050249_d828c9f2",
                    "updated_at": "2026-04-21T05:02:57.603816",
                    "display_name": "CEO 工作汇报",
                    "platform": "feishu",
                    "chat_type": "group",
                    "origin": {
                        "chat_id": "oc_demo",
                        "user_id": "ou_demo",
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    transcript_path = tmp_path / "20260421_050249_d828c9f2.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "hello"}, ensure_ascii=False),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "⚠️ Provider authentication failed: No inference provider configured.",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    status = module.analyze_chain_status(
        sessions_json=sessions_path,
        agent_log=None,
        errors_log=None,
        transcript_path=transcript_path,
    )

    assert status.status == "blocked"
    assert status.blocker == "provider_not_configured"
    assert status.transcript is not None
    assert status.transcript.provider_auth_failed is True
