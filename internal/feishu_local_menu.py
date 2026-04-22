from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Mapping


def build_card_action_ack_payload(content: str = "Action received.") -> dict[str, Any]:
    return {
        "toast": {
            "type": "info",
            "content": content,
        }
    }


def chunk_card_actions(actions: list[dict[str, Any]], *, size: int = 2) -> list[list[dict[str, Any]]]:
    normalized_size = max(1, int(size or 2))
    return [actions[index:index + normalized_size] for index in range(0, len(actions), normalized_size)]


def shorten_local_model_label(model_id: str, *, max_len: int = 42) -> str:
    text = str(model_id or "").strip()
    if len(text) <= max_len:
        return text
    if "/" in text:
        provider, remainder = text.split("/", 1)
        head = max(10, min(18, max_len // 2))
        tail = max(8, min(12, max_len - len(provider) - head - 4))
        return f"{provider}/{remainder[:head]}...{remainder[-tail:]}"
    return text[: max_len - 3] + "..."


def local_registry_card_button(
    *,
    label: str,
    action: str,
    extra: dict[str, Any] | None = None,
    btn_type: str = "default",
) -> dict[str, Any]:
    value = {"hermes_action": action}
    value.update(extra or {})
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": btn_type,
        "value": value,
    }


def resolve_feishu_menu_target(payload: Mapping[str, Any]) -> tuple[str, str] | None:
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return None
    context = event.get("context") or {}
    chat = event.get("chat") or {}
    operator = event.get("operator") or {}
    operator_id = operator.get("operator_id") or {}
    open_chat_id = str((context if isinstance(context, dict) else {}).get("open_chat_id") or "").strip()
    chat_id = str((chat if isinstance(chat, dict) else {}).get("chat_id") or open_chat_id or "").strip()
    open_id = str((operator_id if isinstance(operator_id, dict) else {}).get("open_id") or "").strip()
    if chat_id.startswith("oc_"):
        return chat_id, "chat_id"
    if open_id:
        return open_id, "open_id"
    if chat_id:
        return chat_id, "chat_id"
    return None


def build_local_registry_entries(provider_slug: str, view_name: str, *, limit: int = 20) -> list[dict[str, Any]]:
    from tools.feishu_api import load_feishu_model_registry

    registry_payload = load_feishu_model_registry(force_refresh=False)
    entries = [
        entry
        for entry in list(registry_payload.get("entries") or [])
        if isinstance(entry, dict)
        and str(entry.get("provider") or "").strip().lower() == provider_slug
        and not bool(entry.get("hidden"))
        and bool(entry.get("is_available", True))
        and str(entry.get("model") or "").strip()
    ]

    def _sort_featured(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            0 if str(item.get("selection_hint") or "").strip().lower() == "recommended" else 1,
            0 if bool(item.get("recent_used")) else 1,
            int(item.get("rank") or 9999),
            str(item.get("model") or ""),
        )

    def _sort_recent(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            -int(item.get("recent_used_at") or 0),
            -int(item.get("recent_used_count") or 0),
            int(item.get("rank") or 9999),
            str(item.get("model") or ""),
        )

    def _sort_performance(item: dict[str, Any]) -> tuple[Any, ...]:
        latency = item.get("latency_ms")
        return (
            1 if latency in (None, "", 0) else 0,
            int(latency or 10**9),
            -int(item.get("context_window") or 0),
            int(item.get("rank") or 9999),
            str(item.get("model") or ""),
        )

    if view_name == "recent":
        recent_entries = [item for item in entries if bool(item.get("recent_used"))]
        ordered = sorted(recent_entries or entries, key=_sort_recent)
    elif view_name == "performance":
        ordered = sorted(entries, key=_sort_performance)
    else:
        ordered = sorted(entries, key=_sort_featured)
    return ordered[: max(1, min(limit, 20))]


def load_feishu_personality_card_entries(
    *,
    personality_order: tuple[str, ...],
    personality_labels: dict[str, str],
) -> tuple[list[dict[str, str]], str]:
    current_personality = ""
    merged: dict[str, Any] = {}
    try:
        from hermes_cli.config import DEFAULT_CONFIG, load_config

        config = load_config() or {}
        default_personalities = ((DEFAULT_CONFIG.get("agent") or {}).get("personalities") or {})
        configured_personalities = ((config.get("agent") or {}).get("personalities") or {})
        if isinstance(default_personalities, dict):
            merged.update(default_personalities)
        if isinstance(configured_personalities, dict):
            merged.update(configured_personalities)
        current_personality = str(((config.get("agent") or {}).get("personality") or "")).strip().lower()
    except Exception:
        merged = {}

    ordered_names: list[str] = []
    seen: set[str] = set()
    for name in personality_order:
        if name in merged and name not in seen:
            ordered_names.append(name)
            seen.add(name)
    for name in sorted(str(key).strip().lower() for key in merged.keys() if str(key).strip()):
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)

    entries: list[dict[str, str]] = []
    for name in ordered_names:
        raw = merged.get(name)
        if isinstance(raw, dict):
            description = str(raw.get("description") or raw.get("tone") or "").strip()
        else:
            description = str(raw or "").strip().splitlines()[0][:48]
        entries.append(
            {
                "name": name,
                "label": personality_labels.get(name, name.upper()),
                "description": description[:36],
            }
        )
    return entries, current_personality


def load_feishu_command_card_sections(
    *,
    default_runs: dict[str, str],
    category_order: tuple[str, ...],
) -> list[tuple[str, list[dict[str, Any]]]]:
    from hermes_cli.commands import COMMAND_REGISTRY

    sections: dict[str, list[dict[str, Any]]] = {}
    for command in COMMAND_REGISTRY:
        category = str(command.category or "Other")
        usage = f"/{command.name}"
        if command.args_hint:
            usage = f"{usage} {command.args_hint}"
        sections.setdefault(category, []).append(
            {
                "name": str(command.name or "").strip(),
                "usage": usage,
                "description": str(command.description or "").strip(),
                "aliases": [str(alias).strip() for alias in tuple(command.aliases or ()) if str(alias).strip()],
                "cli_only": bool(command.cli_only and not command.gateway_config_gate),
                "default_command": default_runs.get(str(command.name or "").strip()),
            }
        )

    ordered_categories = [category for category in category_order if category in sections]
    ordered_categories.extend(category for category in sections.keys() if category not in ordered_categories)
    return [(category, sections.get(category, [])) for category in ordered_categories]


def split_card_reference_lines(entries: list[str], *, chunk_size: int = 2) -> list[str]:
    normalized_chunk_size = max(1, int(chunk_size or 2))
    return [" | ".join(entries[index:index + normalized_chunk_size]) for index in range(0, len(entries), normalized_chunk_size)]


def build_feishu_local_menu_card(
    event_key: str,
    *,
    local_model_menu_map: dict[str, tuple[str | None, str | None]],
    build_local_registry_provider_card: Callable[..., dict[str, Any]],
    build_feishu_model_hub_card: Callable[[], dict[str, Any]],
    build_feishu_personality_card: Callable[[], dict[str, Any]],
    build_feishu_skill_combo_card: Callable[[], dict[str, Any]],
    build_feishu_command_center_card: Callable[[], dict[str, Any]],
) -> dict[str, Any] | None:
    normalized_event_key = str(event_key or "").strip()
    provider_slug, view_name = local_model_menu_map.get(normalized_event_key, (None, None))
    if provider_slug and view_name:
        return build_local_registry_provider_card(provider_slug=provider_slug, view_name=view_name)
    if normalized_event_key == "model_picker":
        return build_feishu_model_hub_card()
    if normalized_event_key == "personality_picker":
        return build_feishu_personality_card()
    if normalized_event_key == "skill_combo_picker":
        return build_feishu_skill_combo_card()
    if normalized_event_key == "command_center":
        return build_feishu_command_center_card()
    return None


async def send_feishu_local_registry_menu_card(
    payload: Mapping[str, Any],
    *,
    build_feishu_local_menu_card_fn: Callable[[str], dict[str, Any] | None],
) -> bool:
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return False
    event_key = str(event.get("event_key") or "").strip()
    card = build_feishu_local_menu_card_fn(event_key)
    if not card:
        return False
    target = resolve_feishu_menu_target(payload)
    if not target:
        return False

    from tools.feishu_api import build_feishu_client

    receive_id, receive_id_type = target
    await asyncio.to_thread(
        build_feishu_client().send_message,
        receive_id=receive_id,
        receive_id_type=receive_id_type,
        msg_type="interactive",
        content=json.dumps(card, ensure_ascii=False),
    )
    return True


async def close_feishu_card_from_payload(payload: Mapping[str, Any]) -> bool:
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return False
    context = event.get("context") or {}
    if not isinstance(context, dict):
        return False
    message_id = str(context.get("open_message_id") or context.get("message_id") or "").strip()
    if not message_id:
        return False

    from tools.feishu_api import build_feishu_client

    await asyncio.to_thread(
        build_feishu_client().request_json,
        "DELETE",
        f"/open-apis/im/v1/messages/{message_id}",
    )
    return True


def extract_feishu_card_action_name(payload: Mapping[str, Any]) -> str:
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return ""
    action = event.get("action") or {}
    if not isinstance(action, dict):
        return ""
    action_value = action.get("value") or {}
    if not isinstance(action_value, dict):
        return ""
    return str(action_value.get("hermes_action") or "").strip()


async def enqueue_feishu_card_action_for_background(
    payload: dict[str, Any],
    *,
    extract_queue_context: Callable[[dict[str, Any]], dict[str, Any]],
    enqueue_chat_event_async: Callable[..., Any],
    schedule_chat_queue_worker_background: Callable[..., bool],
    append_trace: Callable[..., None],
    default_chat_queue_batch_size: int,
) -> dict[str, Any]:
    context = extract_queue_context(payload)
    enqueue_result = await enqueue_chat_event_async(
        platform="feishu",
        partition=context["partition"],
        payload=payload,
        metadata=context,
        include_queue_depth=False,
    )
    spawn_scheduled = schedule_chat_queue_worker_background(
        payload=payload,
        platform="feishu",
        partition=context["partition"],
        max_items=default_chat_queue_batch_size,
    )
    append_trace(
        "queue.enqueue",
        payload,
        partition=context["partition"],
        queue_depth=enqueue_result.get("queue_depth"),
        spawn_scheduled=spawn_scheduled,
    )
    return {**enqueue_result, "spawn_scheduled": spawn_scheduled}
