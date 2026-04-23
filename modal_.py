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
RECENT_MODEL_HISTORY_KEY = "recent_models_json"
MODEL_REGISTRY_INTRO_KEYS: tuple[str, ...] = ("introduction", "description", "summary")
PERSONALITY_SYSTEM_PROMPTS = {
    "none": "Keep the tone neutral, helpful, and direct.",
    "ceo": "Answer like a concise CEO: prioritize decisions, tradeoffs, and outcomes.",
    "cto": "Answer like a pragmatic CTO: emphasize architecture, risks, and implementation detail.",
    "staff": "Answer like a chief of staff: structure the response clearly and keep stakeholders aligned.",
    "sev": "Answer like an incident commander: identify the issue, impact, next steps, and mitigation.",
    "grow": "Answer like a growth lead: focus on experiments, funnels, and measurable impact.",
    "content": "Answer like a content strategist: provide crisp messaging and audience-aware framing.",
    "seo": "Answer like an SEO lead: consider search intent, information architecture, and discoverability.",
    "ads": "Answer like a paid ads operator: focus on targeting, creative angles, and efficiency.",
    "bd": "Answer like a business development lead: focus on partnerships, positioning, and leverage.",
    "ops": "Answer like an operations lead: optimize for process clarity, handoffs, and reliability.",
    "finance": "Answer like a finance lead: highlight cost, ROI, and planning implications.",
    "board": "Answer like a board-ready advisor: summarize crisply, note risk, and stay outcome-oriented.",
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
                    RECENT_MODEL_HISTORY_KEY: str(value.get(RECENT_MODEL_HISTORY_KEY) or "{}"),
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
            RECENT_MODEL_HISTORY_KEY: str(raw_state.get(RECENT_MODEL_HISTORY_KEY) or "{}"),
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
            RECENT_MODEL_HISTORY_KEY: "{}",
        },
    )
    return state


def _save_session_state(session_key: str) -> dict[str, str]:
    state = _get_session_state(session_key)
    SESSION_STATE[session_key] = {
        "current_model": str(state.get("current_model") or "openrouter/free"),
        "current_provider": str(state.get("current_provider") or "openrouter"),
        "current_personality": str(state.get("current_personality") or "none"),
        RECENT_MODEL_HISTORY_KEY: str(state.get(RECENT_MODEL_HISTORY_KEY) or "{}"),
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


def _load_registry_payload(force_refresh: bool = False) -> dict[str, Any]:
    try:
        from tools.feishu_api import load_feishu_model_registry

        payload = load_feishu_model_registry(force_refresh=force_refresh)
    except Exception:
        return {"status": "fallback", "entries": []}
    return payload if isinstance(payload, dict) else {"status": "fallback", "entries": []}


def _iter_registry_entries(force_refresh: bool = False) -> list[dict[str, Any]]:
    payload = _load_registry_payload(force_refresh=force_refresh)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    normalized_entries: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "").strip().lower()
        model_id = str(item.get("model") or "").strip()
        if not provider or not model_id:
            continue
        normalized = dict(item)
        normalized["provider"] = provider
        normalized["model"] = model_id
        normalized_entries.append(normalized)
    normalized_entries.sort(
        key=lambda item: (
            bool(item.get("hidden")),
            0 if str(item.get("selection_hint") or "").strip().lower() == "recommended" else 1,
            int(item.get("rank") or 9999),
            str(item.get("provider") or ""),
            str(item.get("model") or ""),
        )
    )
    return normalized_entries


def _get_registry_entries_for_provider(provider_slug: str) -> list[dict[str, Any]]:
    normalized_provider = str(provider_slug or "").strip().lower()
    entries = [
        item
        for item in _iter_registry_entries(force_refresh=False)
        if str(item.get("provider") or "").strip().lower() == normalized_provider and not bool(item.get("hidden"))
    ]
    if entries:
        return entries
    return [
        {
            "provider": provider,
            "model": model,
            "display_name": model,
            "selection_hint": "recommended" if index == 0 else "",
            "rank": index + 1,
            "generated_command": f"/model {model} --provider {provider}",
        }
        for index, (provider, model) in enumerate(MODEL_PRESETS)
        if str(provider or "").strip().lower() == normalized_provider and str(model or "").strip()
    ]


def _lookup_registry_entry(provider_slug: str, model_id: str) -> dict[str, Any] | None:
    provider = str(provider_slug or "").strip().lower()
    model = str(model_id or "").strip()
    if not provider or not model:
        return None
    for item in _get_registry_entries_for_provider(provider):
        if str(item.get("model") or "").strip() == model:
            return item
    return None


def _model_intro_text(entry: dict[str, Any] | None) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in MODEL_REGISTRY_INTRO_KEYS:
        value = str(entry.get(key) or "").strip()
        if value:
            return value
    return ""


def _models_for_provider(provider_slug: str) -> list[str]:
    return [str(item.get("model") or "").strip() for item in _get_registry_entries_for_provider(provider_slug)]


def _load_recent_model_history(state: dict[str, str]) -> dict[str, list[str]]:
    raw = str(state.get(RECENT_MODEL_HISTORY_KEY) or "{}")
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, list[str]] = {}
    for provider, models in parsed.items():
        provider_slug = str(provider or "").strip().lower()
        if not provider_slug or not isinstance(models, list):
            continue
        result[provider_slug] = [str(model).strip() for model in models if str(model).strip()]
    return result


def _save_recent_model_history(state: dict[str, str], history: dict[str, list[str]]) -> None:
    compact: dict[str, list[str]] = {}
    for provider, models in history.items():
        provider_slug = str(provider or "").strip().lower()
        if not provider_slug:
            continue
        deduped: list[str] = []
        seen: set[str] = set()
        for model in models:
            normalized_model = str(model or "").strip()
            if not normalized_model or normalized_model in seen:
                continue
            deduped.append(normalized_model)
            seen.add(normalized_model)
        if deduped:
            compact[provider_slug] = deduped[:8]
    state[RECENT_MODEL_HISTORY_KEY] = json.dumps(compact, ensure_ascii=False)


def _remember_recent_model(state: dict[str, str], provider_slug: str, model_id: str) -> None:
    provider = str(provider_slug or "").strip().lower()
    model = str(model_id or "").strip()
    if not provider or not model:
        return
    history = _load_recent_model_history(state)
    existing = [item for item in history.get(provider, []) if item != model]
    history[provider] = [model, *existing][:8]
    _save_recent_model_history(state, history)


def _performance_priority(model_id: str) -> int:
    normalized = str(model_id or "").strip().lower()
    if not normalized:
        return 99
    if any(token in normalized for token in ("mini", "flash", "nano", "k2.5")):
        return 0
    if any(token in normalized for token in ("70b", "sonnet", "opus")):
        return 2
    return 1


def _personality_system_prompt(name: str) -> str:
    normalized = str(name or "none").strip().lower() or "none"
    return PERSONALITY_SYSTEM_PROMPTS.get(normalized, PERSONALITY_SYSTEM_PROMPTS["none"])


def _flatten_chat_completion_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "").strip()
    return str(value).strip()


def _build_session_system_prompt(state: dict[str, str]) -> str:
    personality = str(state.get("current_personality") or "none").strip().lower() or "none"
    return "\n".join(
        [
            "You are Hermes, a helpful assistant replying inside Feishu.",
            "Answer the user's request directly. Do not merely acknowledge, restate, or paraphrase the incoming message.",
            "When the user asks for analysis, provide actual reasoning and a concrete answer.",
            "Keep the reply concise but useful.",
            _personality_system_prompt(personality),
        ]
    )


def _generate_session_reply(message_text: str, session_key: str) -> dict[str, Any]:
    state = _get_session_state(session_key)
    provider = str(state.get("current_provider") or "openrouter").strip().lower() or "openrouter"
    model_id = str(state.get("current_model") or "openrouter/free").strip() or "openrouter/free"

    from hermes_cli.runtime_provider import resolve_runtime_provider
    import httpx

    runtime = resolve_runtime_provider(requested=provider)
    base_url = str(runtime.get("base_url") or "").rstrip("/")
    api_key = str(runtime.get("api_key") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError(f"Missing runtime credentials for provider '{provider}'")

    endpoint = base_url
    endpoint_lower = endpoint.lower()
    if not endpoint_lower.endswith("/chat/completions") and not endpoint_lower.endswith("/v1/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _build_session_system_prompt(state)},
            {"role": "user", "content": str(message_text or "").strip()},
        ],
        "temperature": 0.6,
    }
    response = httpx.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45.0,
    )
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Model returned no completion choices")
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    content = _flatten_chat_completion_content(message.get("content") if isinstance(message, dict) else "")
    if not content:
        raise RuntimeError("Model returned an empty reply")
    return {
        "text": content,
        "provider": provider,
        "model": model_id,
        "base_url": base_url,
        "ai_call_count": 1,
    }


def _build_generated_reply_result(message_text: str, session_key: str, *, route_hint: str) -> dict[str, Any]:
    state = _get_session_state(session_key)
    generated = _generate_session_reply(message_text, session_key)
    final_text = str(generated.get("text") or "").strip()
    return {
        "status": "completed",
        "route_hint": route_hint,
        "execution_mode": "inline",
        "final_response": final_text,
        "send_plan": _build_text_send_plan(final_text),
        "action_plan": _build_text_send_plan(final_text),
        "session_state_after": _build_session_state_after(state),
        "cache_eligible": False,
        "ai_call_count": int(generated.get("ai_call_count") or 1),
        "capability_match": True,
        "preferred_model_selected": True,
        "reconcile_required": False,
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
            registry_entries = [item for item in _iter_registry_entries(force_refresh=False) if not bool(item.get("hidden"))]
            grouped_models: dict[str, list[str]] = {}
            for item in registry_entries:
                provider_slug = str(item.get("provider") or "").strip().lower()
                model_id = str(item.get("model") or "").strip()
                if provider_slug and model_id:
                    grouped_models.setdefault(provider_slug, []).append(model_id)
            if not grouped_models:
                for provider, model in MODEL_PRESETS:
                    provider_slug = str(provider or "").strip().lower()
                    model_id = str(model or "").strip()
                    if provider_slug and model_id:
                        grouped_models.setdefault(provider_slug, []).append(model_id)
            models = "\n".join(
                f"{provider}:\n" + "\n".join(f"- {model_id}" for model_id in model_ids)
                for provider, model_ids in grouped_models.items()
            )
            content = "\n".join(
                [
                    *_build_route_status_lines(state),
                    "",
                    "Available models from Feishu registry:",
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
        _remember_recent_model(state, provider, model_id)
        state = _save_session_state(session_key)
        model_entry = _lookup_registry_entry(provider, model_id)
        intro = _model_intro_text(model_entry)
        switch_text = f"Model switched to `{model_id}`\nProvider: {provider}\nScope: session only"
        if intro:
            switch_text += f"\nIntroduction: {intro}"
        return _build_command_result(
            switch_text,
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
    registry_entries = [
        item
        for item in _iter_registry_entries(force_refresh=False)
        if not bool(item.get("hidden")) and bool(item.get("is_available", True))
    ]
    if not registry_entries:
        registry_entries = [
            {
                "provider": provider,
                "model": model,
                "selection_hint": "recommended" if index == 0 else "",
                "rank": index + 1,
            }
            for index, (provider, model) in enumerate(MODEL_PRESETS)
        ]
    grouped_entries: dict[str, list[dict[str, Any]]] = {}
    for item in registry_entries:
        provider_slug = str(item.get("provider") or "").strip().lower()
        model_id = str(item.get("model") or "").strip()
        if provider_slug and model_id:
            grouped_entries.setdefault(provider_slug, []).append(item)
    for items in grouped_entries.values():
        items.sort(key=lambda item: (int(item.get("rank") or 9999), str(item.get("model") or "")))
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                "**Hermes Model Hub**\n"
                "Choose a model ID from the Feishu Bitable registry and switch directly."
            ),
        }
    ]
    for provider_slug, items in grouped_entries.items():
        elements.append({"tag": "markdown", "content": f"**{provider_slug}**"})
        model_actions = [
            _button(
                str(item.get("model") or "").strip(),
                "registry_switch_model",
                extra={
                    "provider": provider_slug,
                    "model": str(item.get("model") or "").strip(),
                },
                btn_type="primary" if str(item.get("selection_hint") or "").strip().lower() == "recommended" else "default",
            )
            for item in items[:20]
            if str(item.get("model") or "").strip()
        ]
        for chunk in _chunk_actions(model_actions, 2):
            elements.append({"tag": "action", "actions": chunk})
    elements.append({"tag": "action", "actions": [_button("Close", "registry_close_card")]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "Hermes Model Hub"}, "template": "blue"},
        "elements": elements,
    }


def _build_provider_model_card(provider_slug: str, view_name: str = "featured", session_key: str = "session:default") -> dict[str, Any]:
    normalized_provider = str(provider_slug or "").strip().lower() or "openrouter"
    normalized_view = str(view_name or "").strip().lower() or "featured"
    provider_title = "OpenRouter" if normalized_provider == "openrouter" else "NVIDIA"
    view_title = {
        "featured": "Featured",
        "recent": "Recent",
        "performance": "Performance",
    }.get(normalized_view, "Featured")

    provider_entries = list(_get_registry_entries_for_provider(normalized_provider))
    session_state = _get_session_state(session_key)
    recent_history = _load_recent_model_history(session_state).get(normalized_provider, [])
    if normalized_view == "recent":
        provider_entries.sort(
            key=lambda item: (
                0 if str(item.get("model") or "").strip() in recent_history else 1,
                0 if bool(item.get("recent_used")) else 1,
                -int(item.get("recent_used_at") or 0),
                int(item.get("rank") or 9999),
                str(item.get("model") or ""),
            )
        )
    elif normalized_view == "performance":
        provider_entries.sort(
            key=lambda item: (
                int(item.get("latency_ms") or 10**9),
                _performance_priority(str(item.get("model") or "")),
                int(item.get("rank") or 9999),
                str(item.get("model") or ""),
            )
        )
    else:
        provider_entries.sort(
            key=lambda item: (
                0 if str(item.get("selection_hint") or "").strip().lower() == "recommended" else 1,
                int(item.get("rank") or 9999),
                str(item.get("model") or ""),
            )
        )
    note_by_view = {
        "featured": "Featured list from the Feishu registry.",
        "recent": "Recent list keeps your session history first.",
        "performance": "Performance list prefers lower-latency candidates.",
    }
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": f"**{provider_title} {view_title}**\nChoose a model from the Feishu Bitable registry.\n{note_by_view.get(normalized_view, '')}",
        }
    ]

    model_actions = [
        _button(
            str(item.get("model") or "").strip(),
            "registry_switch_model",
            extra={
                "provider": normalized_provider,
                "model": str(item.get("model") or "").strip(),
            },
            btn_type="primary" if str(item.get("selection_hint") or "").strip().lower() == "recommended" else "default",
        )
        for item in provider_entries[:16]
    ]
    for chunk in _chunk_actions(model_actions, 2):
        elements.append({"tag": "action", "actions": chunk})

    elements.append(
        {
            "tag": "action",
            "actions": [
                _button("Model Hub", "open_menu_card", extra={"event_key": "model_picker"}),
                _button("Close", "registry_close_card"),
            ],
        }
    )

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"{provider_title} {view_title}"}, "template": "blue"},
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
    provider_view_map = {
        "provider_openrouter": ("openrouter", "featured"),
        "provider_openrouter_featured": ("openrouter", "featured"),
        "provider_openrouter_recent": ("openrouter", "recent"),
        "provider_openrouter_performance": ("openrouter", "performance"),
        "provider_nvidia": ("nvidia", "featured"),
        "provider_nvidia_featured": ("nvidia", "featured"),
        "provider_nvidia_recent": ("nvidia", "recent"),
        "provider_nvidia_performance": ("nvidia", "performance"),
    }
    provider_view = provider_view_map.get(normalized)
    if provider_view is not None:
        provider_slug, view_name = provider_view
        return _build_provider_model_card(provider_slug, view_name, session_key)
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
        event_key = str(payload.get("event_key") or "").strip()
        card = _render_card(event_key, session_key)
        card_title = ""
        if isinstance(card, dict):
            header = card.get("header")
            if isinstance(header, dict):
                title = header.get("title")
                if isinstance(title, dict):
                    card_title = str(title.get("content") or "").strip()
        return {
            "status": "ok" if card else "error",
            "action": action,
            "event_key": event_key,
            "card_title": card_title,
            "route_hint": "fast_control",
            "execution_mode": "control_complete",
            "card": card,
            "error": "" if card else f"unsupported event_key: {event_key}",
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
                event_key=result.get("event_key") or context["payload"].get("event_key"),
                card_present=bool(result.get("card")),
                card_title=result.get("card_title"),
                error=result.get("error"),
            )
            return result

        if context["path"] == "/internal/feishu/agent-exec" and not context["message_text"].startswith("/"):
            try:
                result = _build_generated_reply_result(
                    context["message_text"],
                    context["session_key"],
                    route_hint="modal_heavy_exec",
                )
            except Exception as exc:
                state = _get_session_state(context["session_key"])
                error_text = f"Unable to generate a model reply for this session: {exc}"
                result = {
                    "status": "failed",
                    "route_hint": "modal_heavy_exec",
                    "execution_mode": "inline",
                    "final_response": error_text,
                    "send_plan": _build_text_send_plan(error_text),
                    "action_plan": _build_text_send_plan(error_text),
                    "session_state_after": _build_session_state_after(state),
                    "cache_eligible": False,
                    "ai_call_count": 0,
                    "capability_match": False,
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

        if context["path"] == "/internal/feishu/agent-exec":
            if context["message_text"].startswith("/"):
                result = _handle_command(context["message_text"], context["session_key"])
            else:
                result = _build_generated_reply_result(
                    context["message_text"],
                    context["session_key"],
                    route_hint="modal_heavy_exec",
                )
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
            try:
                result = _build_generated_reply_result(
                    context["message_text"],
                    context["session_key"],
                    route_hint="cf_ai_gateway",
                )
            except Exception as exc:
                state = _get_session_state(context["session_key"])
                error_text = f"Unable to generate an inline reply for this session: {exc}"
                result = {
                    "status": "failed",
                    "route_hint": "cf_ai_gateway",
                    "execution_mode": "inline",
                    "final_response": error_text,
                    "send_plan": _build_text_send_plan(error_text),
                    "action_plan": _build_text_send_plan(error_text),
                    "session_state_after": _build_session_state_after(state),
                    "ai_call_count": 0,
                }
            result["timestamp"] = time.time()
            return result

        if context["path"] == "/internal/feishu/message-inline":
            return _build_generated_reply_result(
                context["message_text"],
                context["session_key"],
                route_hint="cf_ai_gateway",
            )

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
