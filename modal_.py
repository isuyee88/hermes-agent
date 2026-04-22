from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:
    modal = None


APP_NAME = "hermes-agent"
MODAL_VOLUME_MOUNT_PATH = Path("/data")
DATA_ROOT = Path(
    os.getenv("HERMES_MODAL_DATA_DIR")
    or (MODAL_VOLUME_MOUNT_PATH / "hermes-modal-stub" if modal is not None else Path(tempfile.gettempdir()) / "hermes-modal-stub")
)
FEISHU_TRACE_PATH = DATA_ROOT / "feishu_trace.jsonl"
SESSION_STATE_PATH = DATA_ROOT / "session_state.json"
SESSION_STATE: dict[str, dict[str, str]] = {}
MODEL_PRESETS: list[tuple[str, str]] = [
    ("openrouter", "openai/gpt-5-mini"),
    ("openrouter", "anthropic/claude-3.7-sonnet"),
    ("openrouter", "google/gemini-2.5-flash"),
    ("nvidia", "moonshotai/kimi-k2.5"),
    ("nvidia", "meta/llama-3.3-70b-instruct"),
    ("nvidia", "nvidia/nemotron-nano-12b-v2"),
]
PERSONALITY_ORDER: tuple[str, ...] = (
    "none",
    "ceo",
    "cto",
    "staff",
    "sev",
    "grow",
    "content",
    "seo",
    "ads",
    "bd",
    "ops",
    "finance",
    "board",
)
PERSONALITY_LABELS = {
    "none": "Neutral",
    "ceo": "CEO",
    "cto": "CTO",
    "staff": "Chief of Staff",
    "sev": "Incident",
    "grow": "Growth",
    "content": "Content",
    "seo": "SEO",
    "ads": "Ads",
    "bd": "BD",
    "ops": "Ops",
    "finance": "Finance",
    "board": "Board",
}

if modal is not None:
    modal_volume = modal.Volume.from_name("hermes-agent-data", create_if_missing=True)
    modal_debug_store = modal.Dict.from_name("hermes-agent-debug-store", create_if_missing=True)
else:
    modal_volume = None
    modal_debug_store = None


def _ensure_runtime_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _reload_modal_volume() -> None:
    if modal_volume is None:
        return
    try:
        modal_volume.reload()
    except Exception:
        pass


def _commit_modal_volume() -> None:
    if modal_volume is None:
        return
    try:
        modal_volume.commit()
    except Exception:
        pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def _read_json_content(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _load_personality_names() -> list[str]:
    try:
        from hermes_cli.config import DEFAULT_CONFIG

        personalities = ((DEFAULT_CONFIG.get("agent") or {}).get("personalities") or {})
        configured = [str(name).strip().lower() for name in personalities.keys() if str(name).strip()]
    except Exception:
        configured = []
    if not configured:
        configured = [name for name in PERSONALITY_ORDER if name != "none"]
    ordered: list[str] = []
    seen: set[str] = set()
    for name in PERSONALITY_ORDER:
        if name == "none":
            continue
        if name in configured and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in sorted(configured):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _available_personality_names() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in ["none", *_load_personality_names(), *PERSONALITY_ORDER]:
        normalized = str(name or "").strip().lower()
        if not normalized or normalized in seen or normalized not in PERSONALITY_LABELS:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    if not ordered:
        return ["none", *[name for name in PERSONALITY_ORDER if name != "none"]]
    if ordered[0] != "none":
        ordered.insert(0, "none")
    return ordered


def _append_feishu_trace(stage: str, payload: dict[str, Any], **extra: Any) -> None:
    _reload_modal_volume()
    _ensure_runtime_dirs()
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timestamp_ms": _now_ms(),
        "stage": stage,
    }
    row.update(payload)
    row.update({key: value for key, value in extra.items() if value is not None})
    with FEISHU_TRACE_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    if modal_debug_store is not None:
        try:
            modal_debug_store.put(f"trace:{row['timestamp_ms']}:{uuid.uuid4().hex}", row)
        except Exception:
            pass
    _commit_modal_volume()


def _read_feishu_trace(limit: int = 100) -> list[dict[str, Any]]:
    _reload_modal_volume()
    _ensure_runtime_dirs()
    if not FEISHU_TRACE_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in FEISHU_TRACE_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-max(int(limit or 0), 0) :]:
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    if modal_debug_store is not None:
        try:
            store_rows = [
                value
                for key, value in modal_debug_store.items()
                if str(key).startswith("trace:") and isinstance(value, dict)
            ]
            if store_rows:
                rows = sorted(
                    [*rows, *store_rows],
                    key=lambda item: int(item.get("timestamp_ms") or 0),
                )[-max(int(limit or 0), 1) :]
        except Exception:
            pass
    return rows


def _load_session_state_map() -> dict[str, dict[str, str]]:
    _reload_modal_volume()
    _ensure_runtime_dirs()
    if modal_debug_store is not None:
        try:
            payload: dict[str, dict[str, str]] = {}
            for key, value in modal_debug_store.items():
                if not str(key).startswith("session:") or not isinstance(value, dict):
                    continue
                payload[str(key).split("session:", 1)[1]] = {
                    "current_model": str(value.get("current_model") or "openrouter/free"),
                    "current_provider": str(value.get("current_provider") or "openrouter"),
                    "current_personality": str(value.get("current_personality") or "none"),
                }
            if payload:
                return payload
        except Exception:
            pass
    if not SESSION_STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(SESSION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, dict[str, str]] = {}
    for session_key, raw_state in payload.items():
        if not isinstance(raw_state, dict):
            continue
        normalized = {
            "current_model": str(raw_state.get("current_model") or "openrouter/free"),
            "current_provider": str(raw_state.get("current_provider") or "openrouter"),
            "current_personality": str(raw_state.get("current_personality") or "none"),
        }
        result[str(session_key)] = normalized
    return result


def _persist_session_state_map() -> None:
    _reload_modal_volume()
    _ensure_runtime_dirs()
    SESSION_STATE_PATH.write_text(json.dumps(SESSION_STATE, ensure_ascii=False, indent=2), encoding="utf-8")
    if modal_debug_store is not None:
        for session_key, state in SESSION_STATE.items():
            try:
                modal_debug_store.put(f"session:{session_key}", state)
            except Exception:
                continue
    _commit_modal_volume()


def _extract_request_context(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", {}) if isinstance(data.get("payload"), dict) else {}
    ingress = payload.get("_hermes_ingress", {}) if isinstance(payload.get("_hermes_ingress"), dict) else {}
    event = payload.get("event", {}) if isinstance(payload.get("event"), dict) else {}
    message = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
    content = _read_json_content(message.get("content", {}))
    message_text = str(
        content.get("text")
        or payload.get("text")
        or payload.get("command_text")
        or ""
    ).strip()
    session_key = str(payload.get("session_key") or ingress.get("session_key") or "session:default").strip()
    event_id = str(payload.get("event_id") or ingress.get("event_id") or f"evt_{uuid.uuid4().hex}").strip()
    correlation_id = str(
        payload.get("correlation_id")
        or ingress.get("correlation_id")
        or f"feishu:{event_id}"
    ).strip()
    return {
        "path": str(data.get("__path") or "/").strip(),
        "payload": payload,
        "ingress": ingress,
        "event": event,
        "message_text": message_text,
        "session_key": session_key,
        "event_id": event_id,
        "correlation_id": correlation_id,
        "chat_id": str(payload.get("chat_id") or ingress.get("chat_id") or "").strip(),
        "user_id": str(payload.get("user_id") or ingress.get("user_id") or "").strip(),
        "message_id": str(payload.get("message_id") or ingress.get("message_id") or "").strip(),
        "chat_type": str(payload.get("chat_type") or ingress.get("chat_type") or "dm").strip() or "dm",
    }


def _get_session_state(session_key: str) -> dict[str, str]:
    if session_key not in SESSION_STATE:
        SESSION_STATE.update(_load_session_state_map())
    state = SESSION_STATE.setdefault(
        session_key,
        {
            "current_model": "openrouter/free",
            "current_provider": "openrouter",
            "current_personality": "none",
        },
    )
    return state


def _save_session_state(session_key: str) -> dict[str, str]:
    state = _get_session_state(session_key)
    SESSION_STATE[session_key] = {
        "current_model": str(state.get("current_model") or "openrouter/free"),
        "current_provider": str(state.get("current_provider") or "openrouter"),
        "current_personality": str(state.get("current_personality") or "none"),
    }
    _persist_session_state_map()
    return SESSION_STATE[session_key]


def _build_route_status_lines(state: dict[str, str]) -> list[str]:
    return [
        f"Current model: {state.get('current_model') or 'openrouter/free'}",
        f"Current provider: {state.get('current_provider') or 'openrouter'}",
        f"Current personality: {state.get('current_personality') or 'none'}",
        "Transport: Cloudflare AI Gateway",
        "Route mode: session scoped override/stub",
    ]


def _build_text_send_plan(text: str) -> list[dict[str, Any]]:
    return [{"kind": "text", "content": text}]


def _build_session_state_after(state: dict[str, str]) -> dict[str, Any]:
    return {
        "current_model": state.get("current_model") or "openrouter/free",
        "current_provider": state.get("current_provider") or "openrouter",
        "current_personality": state.get("current_personality") or "none",
        "route_status_lines": _build_route_status_lines(state),
    }


def _build_command_result(text: str, state: dict[str, str], *, action: str = "dispatch_command") -> dict[str, Any]:
    return {
        "status": "ok",
        "action": action,
        "route_hint": "fast_control",
        "execution_mode": "control_complete",
        "final_response": text,
        "send_plan": _build_text_send_plan(text),
        "action_plan": _build_text_send_plan(text),
        "session_state_after": _build_session_state_after(state),
        "reconcile_required": False,
    }


def _build_help_text() -> str:
    return "\n".join(
        [
            "Hermes control commands:",
            "/help",
            "/status",
            "/provider",
            "/model",
            "/model <model-id> --provider <slug>",
            "/personality",
            "/personality <name>",
        ]
    )


def _handle_command(command_text: str, session_key: str) -> dict[str, Any]:
    state = _get_session_state(session_key)
    text = str(command_text or "").strip()
    if not text:
        return _build_command_result("Empty command.", state)
    parts = text.split()
    command = parts[0].lower()
    args = parts[1:]

    if command == "/help":
        return _build_command_result(_build_help_text(), state)

    if command == "/status":
        return _build_command_result("\n".join(_build_route_status_lines(state)), state)

    if command == "/provider":
        content = "\n".join(
            [
                "Available providers:",
                "- openrouter",
                "- nvidia",
                "",
                f"Current provider: {state.get('current_provider') or 'openrouter'}",
                "Use `/model <model-id> --provider <slug>` to lock one for this session.",
            ]
        )
        return _build_command_result(content, state)

    if command == "/personality":
        available = _available_personality_names()
        if not args:
            content = "Available personalities:\n" + "\n".join(
                f"- {name} ({PERSONALITY_LABELS.get(name, name.upper())})" for name in available
            )
            return _build_command_result(content, state)
        requested = str(args[0]).strip().lower()
        valid = {"default", "neutral"} | set(available)
        if requested in {"default", "neutral"}:
            requested = "none"
        if requested not in valid:
            return _build_command_result(
                f"Unknown personality: {requested}\nAvailable: " + ", ".join(available),
                state,
            )
        state["current_personality"] = requested
        state = _save_session_state(session_key)
        label = PERSONALITY_LABELS.get(requested, requested.upper()) if requested != "none" else "Neutral"
        return _build_command_result(f"Personality set to `{label}` for this session.", state)

    if command == "/model":
        if not args:
            models = "\n".join(f"- {provider}: {model}" for provider, model in MODEL_PRESETS)
            content = "\n".join(
                [
                    * _build_route_status_lines(state),
                    "",
                    "Suggested models:",
                    models,
                    "",
                    "Use `/model <model-id> --provider <slug>` to switch.",
                ]
            )
            return _build_command_result(content, state)
        provider = state.get("current_provider") or "openrouter"
        model_tokens: list[str] = []
        index = 0
        while index < len(args):
            item = args[index]
            if item == "--provider" and index + 1 < len(args):
                provider = str(args[index + 1]).strip().lower() or provider
                index += 2
                continue
            model_tokens.append(item)
            index += 1
        model_id = " ".join(model_tokens).strip()
        if not model_id:
            return _build_command_result("Missing model id. Use `/model <model-id> --provider <slug>`.", state)
        state["current_model"] = model_id
        state["current_provider"] = provider
        state = _save_session_state(session_key)
        return _build_command_result(
            f"Model switched to `{model_id}`\nProvider: {provider}\nScope: session only",
            state,
        )

    return _build_command_result(f"Unsupported command: {command}", state)


def _button(label: str, hermes_action: str, *, extra: dict[str, Any] | None = None, btn_type: str = "default") -> dict[str, Any]:
    value = {"hermes_action": hermes_action}
    value.update(extra or {})
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": btn_type,
        "value": value,
    }


def _chunk_actions(actions: list[dict[str, Any]], size: int = 3) -> list[list[dict[str, Any]]]:
    normalized = max(1, int(size or 3))
    return [actions[i : i + normalized] for i in range(0, len(actions), normalized)]


def _build_personality_card(session_key: str) -> dict[str, Any]:
    state = _get_session_state(session_key)
    current = state.get("current_personality") or "none"
    available = _available_personality_names()
    actions = [
        _button(
            PERSONALITY_LABELS.get(name, name.upper()),
            "personality_set",
            extra={"personality": name},
            btn_type="primary" if name == current else "default",
        )
        for name in available
    ]
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": f"**Hermes Personality Picker**\nCurrent: `{PERSONALITY_LABELS.get(current, current)}`",
        }
    ]
    for chunk in _chunk_actions(actions, 3):
        elements.append({"tag": "action", "actions": chunk})
    elements.append({"tag": "action", "actions": [_button("Close", "registry_close_card")]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "Hermes Personality Picker"}, "template": "wathet"},
        "elements": elements,
    }


def _build_command_center_card() -> dict[str, Any]:
    actions = [
        _button("Help", "command_run", extra={"command_text": "/help"}, btn_type="primary"),
        _button("Status", "command_run", extra={"command_text": "/status"}, btn_type="primary"),
        _button("Provider", "command_run", extra={"command_text": "/provider"}),
        _button("Model", "command_run", extra={"command_text": "/model"}),
        _button("Personality", "command_run", extra={"command_text": "/personality"}),
    ]
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                "**Hermes Command Center**\n"
                "Run common Feishu-safe commands directly from the card."
            ),
        }
    ]
    for chunk in _chunk_actions(actions, 3):
        elements.append({"tag": "action", "actions": chunk})
    elements.append({"tag": "action", "actions": [_button("Close", "registry_close_card")]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "Hermes Command Center"}, "template": "orange"},
        "elements": elements,
    }


def _build_model_hub_card() -> dict[str, Any]:
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                "**Hermes Model Hub**\n"
                "Switch the active session model or open control cards."
            ),
        }
    ]
    model_actions = [
        _button(
            f"{provider}:{model.split('/')[-1][:18]}",
            "registry_switch_model",
            extra={"provider": provider, "model": model},
            btn_type="primary" if provider == "openrouter" else "default",
        )
        for provider, model in MODEL_PRESETS
    ]
    for chunk in _chunk_actions(model_actions, 2):
        elements.append({"tag": "action", "actions": chunk})
    elements.append(
        {
            "tag": "action",
            "actions": [
                _button("Status", "command_run", extra={"command_text": "/status"}),
                _button("Providers", "command_run", extra={"command_text": "/provider"}),
                _button("Model Text", "command_run", extra={"command_text": "/model"}),
            ],
        }
    )
    elements.append(
        {
            "tag": "action",
            "actions": [
                _button("Personality", "open_menu_card", extra={"event_key": "personality_picker"}),
                _button("Commands", "open_menu_card", extra={"event_key": "command_center"}),
                _button("Close", "registry_close_card"),
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "Hermes Model Hub"}, "template": "blue"},
        "elements": elements,
    }


def _build_skill_combo_card() -> dict[str, Any]:
    combos = [
        ("CTO Delivery", ["ship", "gov"], "cto"),
        ("Growth Sprint", ["affiliate-os", "browser-ops"], "grow"),
    ]
    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": "**Hermes Skill Combos**\nLoad a preset workflow starter."}
    ]
    for label, skills, personality in combos:
        elements.append({"tag": "markdown", "content": f"**{label}**\nSkills: {', '.join(skills)}"})
        elements.append(
            {
                "tag": "action",
                "actions": [
                    _button(
                        f"Apply {label}",
                        "skill_combo_apply",
                        extra={
                            "combo_id": label.lower().replace(" ", "_"),
                            "combo_label": label,
                            "skills": skills,
                            "suggested_personality": personality,
                        },
                        btn_type="primary",
                    )
                ],
            }
        )
    elements.append({"tag": "action", "actions": [_button("Close", "registry_close_card")]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "Hermes Skill Combos"}, "template": "turquoise"},
        "elements": elements,
    }


def _render_card(event_key: str, session_key: str) -> dict[str, Any] | None:
    normalized = str(event_key or "").strip()
    if normalized == "model_picker":
        return _build_model_hub_card()
    if normalized == "personality_picker":
        return _build_personality_card(session_key)
    if normalized == "command_center":
        return _build_command_center_card()
    if normalized == "skill_combo_picker":
        return _build_skill_combo_card()
    return None


def _build_interactive_send_plan(card: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"kind": "interactive", "card": card}]


def _handle_session_control(context: dict[str, Any]) -> dict[str, Any]:
    payload = context["payload"]
    session_key = context["session_key"]
    state = _get_session_state(session_key)
    action = str(payload.get("action") or "").strip().lower()

    if action == "get_session_state":
        return {
            "status": "ok",
            "action": action,
            "route_hint": "fast_control",
            "execution_mode": "control_complete",
            "session_state_after": _build_session_state_after(state),
            "reconcile_required": False,
        }

    if action == "render_card":
        card = _render_card(str(payload.get("event_key") or ""), session_key)
        return {
            "status": "ok" if card else "error",
            "action": action,
            "route_hint": "fast_control",
            "execution_mode": "control_complete",
            "card": card,
            "error": "" if card else f"unsupported event_key: {payload.get('event_key')}",
            "session_state_after": _build_session_state_after(state),
            "reconcile_required": False,
        }

    if action == "dispatch_command":
        result = _handle_command(str(payload.get("command_text") or ""), session_key)
        result["action"] = action
        return result

    if action == "activate_skill_combo":
        combo_label = str(payload.get("combo_label") or payload.get("combo_id") or "skill combo").strip()
        suggested = str(payload.get("suggested_personality") or "").strip().lower()
        if suggested:
            state["current_personality"] = suggested
            state = _save_session_state(session_key)
        return _build_command_result(
            f"Activated `{combo_label}`.\nSuggested personality: {suggested or 'none'}",
            state,
            action=action,
        )

    return {
        "status": "error",
        "action": action,
        "route_hint": "fast_control",
        "execution_mode": "control_complete",
        "error": f"unsupported action: {action}",
        "session_state_after": _build_session_state_after(state),
        "reconcile_required": False,
    }


if modal is not None:
    app = modal.App(APP_NAME)

    _PIP_DEPS = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "httpx",
        "openai",
        "anthropic",
        "tiktoken",
        "pyyaml",
    ]

    image = modal.Image.debian_slim().pip_install(*_PIP_DEPS)

    @app.function(
        image=image,
        timeout=300,
        memory=256,
        cpu=1,
        scaledown_window=30,
        volumes={str(MODAL_VOLUME_MOUNT_PATH): modal_volume},
    )
    @modal.fastapi_endpoint(method="POST")
    def web_handler(data: dict) -> dict:
        context = _extract_request_context(data if isinstance(data, dict) else {})
        _append_feishu_trace(
            "webhook.accepted",
            {
                "event_id": context["event_id"],
                "correlation_id": context["correlation_id"],
                "session_key": context["session_key"],
                "message_id": context["message_id"],
                "chat_id": context["chat_id"],
                "user_id": context["user_id"],
                "path": context["path"],
                "text": context["message_text"][:200],
            },
        )

        if context["path"] == "/internal/feishu/debug-runtime":
            return {"status": "ok", "configured": True, "runner_type": "stub-runner", "adapter_type": "feishu-adapter"}

        if context["path"] == "/internal/feishu/debug-bootstrap":
            return {"status": "ok", "bootstrap": True}

        if context["path"] == "/internal/feishu/debug-menu":
            return {"status": "ok", "configured": True}

        if context["path"] == "/internal/feishu/agent-plan":
            state = _get_session_state(context["session_key"])
            return {
                "status": "ok",
                "route_hint": "modal_heavy_exec",
                "execution_mode": "modal_heavy_exec",
                "external_exec_candidate": False,
                "request_class": "text_plain",
                "route_family": "modal_runtime",
                "gateway_route_name": "",
                "gateway_eligible": False,
                "cache_eligible": False,
                "ai_call_count": 0,
                "capability_match": True,
                "preferred_model_selected": True,
                "session_state_after": _build_session_state_after(state),
            }

        if context["path"] == "/internal/feishu/session-control":
            result = _handle_session_control(context)
            _append_feishu_trace(
                "internal.session_control.done",
                {
                    "event_id": context["event_id"],
                    "correlation_id": context["correlation_id"],
                    "session_key": context["session_key"],
                },
                action=result.get("action"),
                status=result.get("status"),
            )
            return result

        if context["path"] == "/internal/feishu/agent-exec":
            if context["message_text"].startswith("/"):
                result = _handle_command(context["message_text"], context["session_key"])
            else:
                state = _get_session_state(context["session_key"])
                prefix = ""
                if state.get("current_personality") and state.get("current_personality") != "none":
                    prefix = f"[{state['current_personality']}] "
                result = {
                    "status": "completed",
                    "route_hint": "modal_heavy_exec",
                    "execution_mode": "inline",
                    "final_response": f"{prefix}收到消息：{context['message_text'][:120] or '收到消息'}",
                    "send_plan": _build_text_send_plan(f"{prefix}我收到了你的消息：{context['message_text'][:120] or '收到消息'}"),
                    "action_plan": _build_text_send_plan(f"{prefix}我收到了你的消息：{context['message_text'][:120] or '收到消息'}"),
                    "session_state_after": _build_session_state_after(state),
                    "cache_eligible": False,
                    "ai_call_count": 0,
                    "capability_match": True,
                    "preferred_model_selected": True,
                    "reconcile_required": False,
                }
            result.setdefault("worker_context", {})
            result["worker_context"].update(
                {
                    "worker_boot_id": f"feishu-chat-{uuid.uuid4().hex}",
                    "worker_started_at": _now_ms(),
                    "initialized": True,
                }
            )
            result["timestamp"] = time.time()
            _append_feishu_trace(
                "internal.agent_exec.done",
                {
                    "event_id": context["event_id"],
                    "correlation_id": context["correlation_id"],
                    "session_key": context["session_key"],
                    "message_id": context["message_id"],
                },
                execution_mode=result.get("execution_mode"),
                request_class=result.get("request_class", "text_plain"),
                route_hint=result.get("route_hint"),
                ai_call_count=result.get("ai_call_count"),
                capability_match=result.get("capability_match"),
                preferred_model_selected=result.get("preferred_model_selected"),
            )
            return result

        if context["path"] == "/internal/feishu/message-inline":
            state = _get_session_state(context["session_key"])
            return {
                "status": "completed",
                "route_hint": "cf_ai_gateway",
                "execution_mode": "inline",
                "final_response": f"收到消息：{context['message_text'][:120] or '收到消息'}",
                "send_plan": _build_text_send_plan(f"我收到了你的消息：{context['message_text'][:120] or '收到消息'}"),
                "action_plan": _build_text_send_plan(f"我收到了你的消息：{context['message_text'][:120] or '收到消息'}"),
                "session_state_after": _build_session_state_after(state),
                "timestamp": time.time(),
            }

        if context["path"] == "/internal/feishu/ack-reaction":
            return {"status": "ok", "message": "ack reaction processed", "worker_context": True}

        if context["path"] == "/internal/feishu/event":
            return {
                "status": "spawned",
                "payload_summary": {k: str(v)[:50] for k, v in context["payload"].items() if k != "__path"},
                "budget_ms": 30000,
                "timestamp": time.time(),
            }

        if context["path"] == "/internal/feishu/background-exec":
            state = _get_session_state(context["session_key"])
            return {
                "status": "completed",
                "route_hint": "modal_heavy_exec",
                "execution_mode": "background",
                "final_response": "后台任务执行完成",
                "send_plan": _build_text_send_plan("后台任务已执行"),
                "action_plan": _build_text_send_plan("后台任务已执行"),
                "session_state_after": _build_session_state_after(state),
            }

        if context["path"] == "/internal/feishu/chat-queue":
            batch = data.get("batch", []) if isinstance(data.get("batch"), list) else []
            return {"status": "completed", "processed_count": len(batch)}

        if context["path"] == "/internal/feishu/probe-metadata":
            return {
                "status": "ok",
                "provider": data.get("provider", ""),
                "model": data.get("model", ""),
            }

        return {"status": "ok", "path": context["path"], "message": "Hermes Agent Web Handler"}

    @app.function(
        image=image,
        timeout=30,
        memory=256,
        cpu=1,
        scaledown_window=30,
        volumes={str(MODAL_VOLUME_MOUNT_PATH): modal_volume},
    )
    def debug_feishu_trace(limit: int = 100) -> dict[str, Any]:
        rows = _read_feishu_trace(limit=max(int(limit or 0), 1))
        return {"rows": rows, "count": len(rows), "path": str(FEISHU_TRACE_PATH)}

    @app.function(
        image=image,
        timeout=30,
        memory=256,
        cpu=1,
        scaledown_window=30,
        volumes={str(MODAL_VOLUME_MOUNT_PATH): modal_volume},
    )
    def debug_feishu_perf_summary(
        limit: int = 1000,
        since_seconds: int = 0,
        event_type: str = "",
        experiment_label: str = "",
        app_name_filter: str = "",
        snapshot_profile: str = "",
        include_duplicates: bool = False,
    ) -> dict[str, Any]:
        rows = _read_feishu_trace(limit=max(int(limit or 0), 1))
        if int(since_seconds or 0) > 0:
            cutoff_ms = _now_ms() - (int(since_seconds) * 1000)
            rows = [row for row in rows if int(row.get("timestamp_ms") or 0) >= cutoff_ms]
        sessions = {str(row.get("event_id") or "").strip() for row in rows if str(row.get("event_id") or "").strip()}
        return {
            "status": "ok",
            "trace_row_count": len(rows),
            "event_count": len(sessions),
            "stages": sorted({str(row.get("stage") or "").strip() for row in rows if str(row.get("stage") or "").strip()}),
            "filters": {
                "since_seconds": int(since_seconds or 0),
                "event_type": str(event_type or ""),
                "experiment_label": str(experiment_label or ""),
                "app_name_filter": str(app_name_filter or ""),
                "snapshot_profile": str(snapshot_profile or ""),
                "include_duplicates": bool(include_duplicates),
            },
        }

    @app.function(image=image, timeout=30, memory=128, cpu=1, scaledown_window=30)
    @modal.fastapi_endpoint(method="GET")
    def web_health() -> dict:
        return {"status": "ok", "service": APP_NAME, "timestamp": time.time()}
