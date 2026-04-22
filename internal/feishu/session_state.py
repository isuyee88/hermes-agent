from __future__ import annotations

from typing import Any, Mapping


def build_external_conversation_history(
    *,
    runner: Any,
    source: Any,
    latest_user_text: str,
    max_messages: int = 12,
    max_chars: int = 12000,
) -> list[dict[str, str]]:
    history_messages: list[dict[str, str]] = []
    try:
        session_entry = runner.session_store.get_or_create_session(source)
        transcript = runner.session_store.load_transcript(session_entry.session_id)
    except Exception:
        transcript = []
    filtered: list[dict[str, str]] = []
    total_chars = 0
    for message in list(transcript or []):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content if item is not None)
        text = str(content or "").strip()
        if not text:
            continue
        filtered.append({"role": role, "content": text})
    for message in filtered[-max_messages:]:
        total_chars += len(message["content"])
        if total_chars > max_chars:
            break
        history_messages.append(message)
    latest_text = str(latest_user_text or "").strip()
    if latest_text:
        history_messages.append({"role": "user", "content": latest_text})
    return history_messages


def build_internal_session_state(*, runner: Any, source: Any) -> dict[str, Any]:
    current_model, current_provider, current_base_url, current_api_key, _ = runner._load_model_runtime_config()
    session_key = runner._session_key_for_source(source)
    route_state = runner._get_active_route_state(
        session_key,
        current_model=current_model,
        current_provider=current_provider,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
    )
    current_personality = ""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        current_personality = str(((config.get("agent") or {}).get("personality") or "")).strip().lower()
    except Exception:
        current_personality = ""
    return {
        "session_key": session_key,
        "current_model": str(route_state.get("current_model") or ""),
        "current_provider": str(route_state.get("current_provider") or ""),
        "route_status_lines": list(runner._render_route_status_lines(route_state)),
        "current_personality": current_personality,
        "route_debug": dict(route_state.get("route_debug") or {}),
    }
