from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_ACCOUNT_ID = "d1215a30b84b673ef0367010b0e78c10"
DEFAULT_WORKFLOW_NAME = "hermes-feishu-agent-workflow"
DEFAULT_WORKER_NAME = "hermes-feishu-gateway"


def classify_instance_error(instance_id: str, error_message: str) -> str:
    normalized_id = trim(instance_id).lower()
    normalized_error = trim(error_message).lower()
    if "bot/user can not be out of the chat" in normalized_error:
        return "bot_not_in_target_chat"
    if "open_id cross app" in normalized_error:
        return "cross_app_target_mismatch"
    if "modal_internal_failed:401" in normalized_error:
        return "modal_auth_mismatch"
    if "modal_internal_failed:404" in normalized_error:
        return "modal_internal_route_missing"
    if "modal_internal_failed:500" in normalized_error:
        return "modal_runtime_error"
    if "test_" in normalized_id or "selftest" in normalized_id:
        return "synthetic_test_instance"
    return "unknown"


def trim(value: Any) -> str:
    return str(value or "").strip()


def json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> tuple[int, Any]:
    payload = None
    final_headers = dict(headers or {})
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        final_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=payload, headers=final_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {"raw": raw}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return exc.code, parsed


@dataclass
class DiagnoseConfig:
    cf_api_token: str
    cf_account_id: str
    workflow_name: str
    worker_name: str
    modal_probe_token: str
    instance_id: str
    chat_filter: str
    limit: int


def parse_args() -> DiagnoseConfig:
    parser = argparse.ArgumentParser(
        description="Diagnose the live Feishu -> Cloudflare Workflow -> Modal chain with real workflow instances.",
    )
    parser.add_argument("--cf-account-id", default=os.getenv("CLOUDFLARE_ACCOUNT_ID", DEFAULT_ACCOUNT_ID))
    parser.add_argument("--workflow-name", default=DEFAULT_WORKFLOW_NAME)
    parser.add_argument("--worker-name", default=DEFAULT_WORKER_NAME)
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--chat-filter", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--modal-probe-token",
        default=os.getenv("MODAL_INTERNAL_BEARER_TOKEN_PROBE", ""),
        help="Bearer token for replaying a workflow instance into the live Modal web_handler.",
    )
    args = parser.parse_args()

    cf_api_token = trim(os.getenv("CLOUDFLARE_API_TOKEN"))
    if not cf_api_token:
        parser.error("CLOUDFLARE_API_TOKEN is required")

    return DiagnoseConfig(
        cf_api_token=cf_api_token,
        cf_account_id=trim(args.cf_account_id) or DEFAULT_ACCOUNT_ID,
        workflow_name=trim(args.workflow_name) or DEFAULT_WORKFLOW_NAME,
        worker_name=trim(args.worker_name) or DEFAULT_WORKER_NAME,
        modal_probe_token=trim(args.modal_probe_token),
        instance_id=trim(args.instance_id),
        chat_filter=trim(args.chat_filter),
        limit=max(1, args.limit),
    )


def cf_headers(config: DiagnoseConfig) -> dict[str, str]:
    return {"Authorization": f"Bearer {config.cf_api_token}"}


def get_worker_settings(config: DiagnoseConfig) -> dict[str, Any]:
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{config.cf_account_id}"
        f"/workers/scripts/{config.worker_name}/settings"
    )
    status, payload = json_request(url, headers=cf_headers(config))
    if status != 200:
        raise RuntimeError(f"worker_settings_failed:{status}:{payload}")
    return payload.get("result", {}) if isinstance(payload, dict) else {}


def get_workflow_instances(config: DiagnoseConfig) -> list[dict[str, Any]]:
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{config.cf_account_id}"
        f"/workflows/{config.workflow_name}/instances"
    )
    status, payload = json_request(url, headers=cf_headers(config))
    if status != 200:
        raise RuntimeError(f"workflow_instances_failed:{status}:{payload}")
    result = payload.get("result", []) if isinstance(payload, dict) else []
    if not isinstance(result, list):
        return []
    return result[: config.limit]


def get_workflow_instance_detail(config: DiagnoseConfig, instance_id: str) -> dict[str, Any]:
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{config.cf_account_id}"
        f"/workflows/{config.workflow_name}/instances/{urllib.parse.quote(instance_id, safe='')}"
    )
    status, payload = json_request(url, headers=cf_headers(config))
    if status != 200:
        raise RuntimeError(f"workflow_instance_detail_failed:{status}:{payload}")
    return payload.get("result", {}) if isinstance(payload, dict) else {}


def select_instance(config: DiagnoseConfig, instances: list[dict[str, Any]]) -> dict[str, Any] | None:
    if config.instance_id:
        for item in instances:
            if trim(item.get("id")) == config.instance_id:
                return item
        return {"id": config.instance_id}
    if config.chat_filter:
        for item in instances:
            if config.chat_filter in trim(item.get("id")):
                return item
    return instances[0] if instances else None


def build_modal_probe_body(detail: dict[str, Any]) -> dict[str, Any]:
    params = detail.get("params", {}) if isinstance(detail.get("params"), dict) else {}
    raw_payload = params.get("raw_payload", {}) if isinstance(params.get("raw_payload"), dict) else {}
    path = "/internal/feishu/agent-exec"
    legacy_payload = {
        **params,
        "pending_reconcile": [],
        "raw_message": raw_payload,
    }
    return {
        "__path": path,
        "payload": legacy_payload,
    }


def replay_instance_to_modal(config: DiagnoseConfig, settings: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    if not config.modal_probe_token:
        return {"status": "skipped", "reason": "missing_modal_probe_token"}

    base_url = ""
    for item in settings.get("bindings", []) if isinstance(settings.get("bindings"), list) else []:
        if trim(item.get("name")) == "MODAL_INTERNAL_BASE_URL":
            base_url = trim(item.get("text"))
            break
    if not base_url:
        return {"status": "skipped", "reason": "missing_worker_modal_base_url"}

    status, payload = json_request(
        trim(base_url),
        method="POST",
        headers={
            "Authorization": f"Bearer {config.modal_probe_token}",
            "Content-Type": "application/json",
        },
        body=build_modal_probe_body(detail),
        timeout=120,
    )
    return {
        "status_code": status,
        "payload": payload,
    }


def probe_modal_internal_path(
    config: DiagnoseConfig,
    settings: dict[str, Any],
    *,
    path: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    if not config.modal_probe_token:
        return {"status": "skipped", "reason": "missing_modal_probe_token"}

    base_url = ""
    for item in settings.get("bindings", []) if isinstance(settings.get("bindings"), list) else []:
        if trim(item.get("name")) == "MODAL_INTERNAL_BASE_URL":
            base_url = trim(item.get("text"))
            break
    if not base_url:
        return {"status": "skipped", "reason": "missing_worker_modal_base_url"}

    status, payload = json_request(
        base_url,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.modal_probe_token}",
            "Content-Type": "application/json",
        },
        body={"__path": path, "payload": body},
        timeout=120,
    )
    return {"status_code": status, "payload": payload}


def build_session_control_probe_body() -> dict[str, Any]:
    return {
        "action": "get_session_state",
        "correlation_id": "feishu:probe:session-control",
        "session_key": "agent:main:feishu:dm:oc_probe",
        "event_id": "evt_probe_session_control",
        "event_type": "application.bot.menu_v6",
        "chat_id": "oc_probe",
        "chat_type": "dm",
        "chat_name": "oc_probe",
        "user_id": "ou_probe",
        "user_name": "probe",
        "message_id": "om_probe_session_control",
        "lane": "control",
        "task_kind": "text",
        "request_class": "session_status",
        "route_hint": "fast_control",
        "route_family": "modal_control",
        "gateway_route_name": "",
        "gateway_eligible": False,
        "modality_profile": "text",
        "requires_tools": False,
        "requires_browser": False,
        "requires_media_hydration": False,
        "requires_modal_runtime": True,
        "reason_code": "planner_forced_modal",
        "content_modalities": ["text"],
        "toolset": [],
    }


def _build_agent_plan_probe_body_legacy() -> dict[str, Any]:
    return {
        "ingress": {
            "correlation_id": "feishu:probe:agent-plan",
            "session_key": "agent:main:feishu:group:oc_probe:ou_probe",
            "event_id": "evt_probe_agent_plan",
            "event_type": "im.message.receive_v1",
            "chat_id": "oc_probe",
            "chat_type": "group",
            "chat_name": "oc_probe",
            "user_id": "ou_probe",
            "user_name": "probe",
            "message_id": "om_probe_agent_plan",
            "message_type": "text",
            "text": "介绍一下你自己",
            "lane": "agent",
            "task_kind": "text",
        },
        "route": {
            "request_class": "text_plain",
            "route_hint": "modal_heavy_exec",
            "route_family": "gateway_text",
            "gateway_route_name": "affiliate-general",
            "gateway_eligible": True,
            "modality_profile": "text",
            "requires_tools": False,
            "requires_browser": False,
            "requires_media_hydration": False,
            "requires_modal_runtime": False,
            "reason_code": "plain_text_without_attachments_or_browser",
            "content_modalities": ["text"],
            "toolset": [],
        },
        "gateway_meta": {
            "endpoint": "/internal/feishu/agent-plan",
            "gateway_hop": "cloudflare-worker",
            "gateway_script": DEFAULT_WORKER_NAME,
            "legacy_payload": {
                "session_key": "agent:main:feishu:group:oc_probe:ou_probe",
                "text": "介绍一下你自己",
            },
        },
    }


def build_agent_plan_probe_body() -> dict[str, Any]:
    return {
        "correlation_id": "feishu:probe:agent-plan",
        "session_key": "agent:main:feishu:dm:oc_probe",
        "event_id": "evt_probe_agent_plan",
        "event_type": "im.message.receive_v1",
        "chat_id": "oc_probe",
        "chat_type": "dm",
        "chat_name": "oc_probe",
        "user_id": "ou_probe",
        "user_name": "probe",
        "message_id": "om_probe_agent_plan",
        "message_type": "text",
        "text": "请介绍一下你自己",
        "lane": "agent",
        "task_kind": "text",
        "request_class": "text_plain",
        "route_hint": "modal_heavy_exec",
        "route_family": "gateway_text",
        "gateway_route_name": "affiliate-general",
        "gateway_eligible": True,
        "modality_profile": "text",
        "requires_tools": False,
        "requires_browser": False,
        "requires_media_hydration": False,
        "requires_modal_runtime": False,
        "reason_code": "plain_text_without_attachments_or_browser",
        "content_modalities": ["text"],
        "toolset": [],
    }


def summarize_steps(detail: dict[str, Any]) -> list[dict[str, Any]]:
    steps = detail.get("steps", []) if isinstance(detail.get("steps"), list) else []
    out: list[dict[str, Any]] = []
    for step in steps:
        attempts = step.get("attempts", []) if isinstance(step.get("attempts"), list) else []
        latest_error = ""
        if attempts:
            latest = attempts[-1]
            error = latest.get("error")
            if isinstance(error, dict):
                latest_error = trim(error.get("message"))
        out.append(
            {
                "name": trim(step.get("name")),
                "success": bool(step.get("success")),
                "latest_error": latest_error,
            }
        )
    return out


def summarize_recent_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in instances[:10]:
        instance_id = trim(item.get("id"))
        out.append(
            {
                "id": instance_id,
                "status": trim(item.get("status")),
                "created_on": trim(item.get("created_on")),
                "is_synthetic": "test_" in instance_id.lower() or "selftest" in instance_id.lower(),
            }
        )
    return out


def main() -> int:
    config = parse_args()
    try:
        settings = get_worker_settings(config)
        instances = get_workflow_instances(config)
        selected = select_instance(config, instances)
        if not selected:
            print(json.dumps({"status": "error", "reason": "no_workflow_instances"}, ensure_ascii=False, indent=2))
            return 1

        detail = get_workflow_instance_detail(config, trim(selected.get("id")))
        replay = replay_instance_to_modal(config, settings, detail)
        session_control_probe = probe_modal_internal_path(
            config,
            settings,
            path="/internal/feishu/session-control",
            body=build_session_control_probe_body(),
        )
        agent_plan_probe = probe_modal_internal_path(
            config,
            settings,
            path="/internal/feishu/agent-plan",
            body=build_agent_plan_probe_body(),
        )
        output = {
            "worker_name": config.worker_name,
            "workflow_name": config.workflow_name,
            "selected_instance_id": trim(detail.get("id")) or trim(selected.get("id")),
            "selected_instance_status": trim(detail.get("status")) or trim(selected.get("status")),
            "selected_instance_error": trim((detail.get("error") or {}).get("message")) if isinstance(detail.get("error"), dict) else "",
            "selected_instance_error_class": classify_instance_error(
                trim(detail.get("id")) or trim(selected.get("id")),
                trim((detail.get("error") or {}).get("message")) if isinstance(detail.get("error"), dict) else "",
            ),
            "selected_instance_chat_id": trim(
                ((detail.get("params") or {}) if isinstance(detail.get("params"), dict) else {}).get("chat_id")
            ),
            "selected_instance_is_synthetic": (
                "test_" in (trim(detail.get("id")) or trim(selected.get("id"))).lower()
                or "selftest" in (trim(detail.get("id")) or trim(selected.get("id"))).lower()
            ),
            "recent_instances": summarize_recent_instances(instances),
            "worker_settings": {
                "bindings": [
                    item
                    for item in (settings.get("bindings", []) if isinstance(settings.get("bindings"), list) else [])
                    if trim(item.get("name")) in {"MODAL_INTERNAL_BASE_URL", "FEISHU_API_BASE"}
                ]
            },
            "workflow_steps": summarize_steps(detail),
            "modal_replay": replay,
            "session_control_probe": session_control_probe,
            "agent_plan_probe": agent_plan_probe,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
