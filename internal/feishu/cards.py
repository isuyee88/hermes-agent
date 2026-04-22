from __future__ import annotations

from typing import Any, Callable


ButtonFactory = Callable[..., dict[str, Any]]
ChunkActions = Callable[..., list[list[dict[str, Any]]]]
SplitReferenceLines = Callable[..., list[str]]
BuildEntries = Callable[..., list[dict[str, Any]]]
ShortenLabel = Callable[..., str]


def build_feishu_menu_manifest() -> dict[str, Any]:
    items = [
        {
            "label": "Model Hub",
            "event_key": "model_picker",
            "description": "Browse the Hermes model hub and switch models.",
            "recommended": True,
        },
        {
            "label": "Model Status",
            "event_key": "model_status",
            "description": "Inspect the active model, provider, and route mode.",
            "recommended": True,
        },
        {
            "label": "Provider Status",
            "event_key": "provider_status",
            "description": "Review OpenRouter and NVIDIA provider health and routing.",
            "recommended": True,
        },
        {
            "label": "Use OpenRouter",
            "event_key": "provider_openrouter",
            "description": "Switch the default provider to OpenRouter.",
            "recommended": False,
        },
        {
            "label": "Use NVIDIA",
            "event_key": "provider_nvidia",
            "description": "Switch the default provider to NVIDIA.",
            "recommended": False,
        },
        {
            "label": "OpenRouter Featured",
            "event_key": "provider_openrouter_featured",
            "description": "Open the featured OpenRouter model list.",
            "recommended": False,
        },
        {
            "label": "OpenRouter Recent",
            "event_key": "provider_openrouter_recent",
            "description": "Open the recent OpenRouter model list.",
            "recommended": False,
        },
        {
            "label": "OpenRouter Performance",
            "event_key": "provider_openrouter_performance",
            "description": "Open the performance-oriented OpenRouter model list.",
            "recommended": False,
        },
        {
            "label": "Personality Picker",
            "event_key": "personality_picker",
            "description": "Choose the Hermes personality preset for the session.",
            "recommended": True,
        },
        {
            "label": "Skill Combos",
            "event_key": "skill_combo_picker",
            "description": "Load curated skill combinations for common Feishu workflows.",
            "recommended": True,
        },
        {
            "label": "Command Center",
            "event_key": "command_center",
            "description": "Review the core control commands available in Feishu.",
            "recommended": True,
        },
    ]
    return {
        "version": 1,
        "title": "Hermes Feishu Menu",
        "description": "Use model_picker, provider menus, personality presets, or command center actions.",
        "items": items,
    }


def build_local_registry_provider_card(
    *,
    provider_slug: str,
    view_name: str,
    build_entries: BuildEntries,
    shorten_label: ShortenLabel,
    button_factory: ButtonFactory,
    chunk_actions: ChunkActions,
) -> dict[str, Any]:
    entries = build_entries(provider_slug, view_name, limit=20)
    provider_title = provider_slug.title()
    view_title = {
        "featured": "Featured",
        "recent": "Recent",
        "performance": "Performance",
    }.get(view_name, "Featured")
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": f"**{provider_title} {view_title}**\nChoose a model from the local registry.",
        }
    ]

    actions = [
        button_factory(
            label=shorten_label(str(item.get("model") or "").strip()),
            action="registry_switch_model",
            extra={
                "provider": provider_slug,
                "model": str(item.get("model") or "").strip(),
            },
        )
        for item in entries
    ]
    for chunk in chunk_actions(actions, size=2):
        elements.append({"tag": "action", "actions": chunk})

    elements.append(
        {
            "tag": "action",
            "actions": [
                button_factory(
                    label="Close",
                    action="registry_close_card",
                    btn_type="danger",
                )
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"{provider_title} {view_title}"},
            "template": "blue",
        },
        "elements": elements,
    }


def build_feishu_personality_card(
    *,
    entries: list[dict[str, str]],
    current_personality: str,
    labels: dict[str, str],
    button_factory: ButtonFactory,
    chunk_actions: ChunkActions,
) -> dict[str, Any]:
    current_label = labels.get(current_personality, current_personality.upper()) if current_personality else "None"
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                f"**Hermes Personality Picker**\nCurrent personality: `{current_label}`\n"
                "Choose a preset below or clear the personality for a neutral session."
            ),
        }
    ]

    buttons: list[dict[str, Any]] = []
    for entry in entries:
        buttons.append(
            button_factory(
                label=entry["label"],
                action="personality_set",
                extra={"personality": entry["name"]},
                btn_type="primary" if entry["name"] == current_personality else "default",
            )
        )
    for chunk in chunk_actions(buttons, size=3):
        elements.append({"tag": "action", "actions": chunk})

    elements.append(
        {
            "tag": "action",
            "actions": [
                button_factory(label="Clear", action="personality_set", extra={"personality": "none"}, btn_type="danger"),
                button_factory(label="Close", action="registry_close_card", btn_type="default"),
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Hermes Personality Picker"},
            "template": "wathet",
        },
        "elements": elements,
    }


def build_feishu_command_center_card(
    *,
    sections: list[tuple[str, list[dict[str, Any]]]],
    category_labels: dict[str, str],
    button_factory: ButtonFactory,
    chunk_actions: ChunkActions,
    split_reference_lines: SplitReferenceLines,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                "**Hermes Command Center**\n"
                "Blue buttons run immediately in this Feishu session.\n"
                "Commands with args or local-only behavior remain as card references."
            ),
        }
    ]

    for category, entries in sections:
        if not entries:
            continue
        category_label = category_labels.get(category, category)
        elements.append({"tag": "markdown", "content": f"**{category_label}**"})

        runnable_actions = [
            button_factory(
                label=f"/{entry['name']}",
                action="command_run",
                extra={"command_text": entry["default_command"]},
                btn_type="primary" if entry["name"] in {"help", "status", "model", "provider"} else "default",
            )
            for entry in entries
            if str(entry.get("default_command") or "").strip()
        ]
        for chunk in chunk_actions(runnable_actions, size=3):
            elements.append({"tag": "action", "actions": chunk})

        manual_refs: list[str] = []
        cli_only_refs: list[str] = []
        for entry in entries:
            usage = str(entry.get("usage") or "").strip()
            aliases = [alias for alias in entry.get("aliases") or [] if alias]
            alias_text = f" (aliases: {' '.join('/' + alias for alias in aliases)})" if aliases else ""
            rendered = f"`{usage}`{alias_text}"
            if bool(entry.get("cli_only")):
                cli_only_refs.append(rendered)
            elif not str(entry.get("default_command") or "").strip():
                manual_refs.append(rendered)

        for line in split_reference_lines(manual_refs, chunk_size=2):
            elements.append({"tag": "markdown", "content": f"Input manually: {line}"})
        for line in split_reference_lines(cli_only_refs, chunk_size=2):
            elements.append({"tag": "markdown", "content": f"CLI only: {line}"})

    elements.append(
        {
            "tag": "markdown",
            "content": "Bottom note: use personality and skill cards for presets; use this card for slash command control.",
        }
    )
    elements.append(
        {
            "tag": "action",
            "actions": [
                button_factory(label="Close", action="registry_close_card", btn_type="default"),
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Hermes Command Center"},
            "template": "orange",
        },
        "elements": elements,
    }


def build_feishu_model_hub_card(*, button_factory: ButtonFactory) -> dict[str, Any]:
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                "**Hermes Model Hub**\n"
                "Use the shortcuts below to open local registry views, inspect route status, or jump into personality/skills."
            ),
        },
        {
            "tag": "action",
            "actions": [
                button_factory(
                    label="OR Featured",
                    action="open_menu_card",
                    extra={"event_key": "provider_openrouter_featured"},
                    btn_type="primary",
                ),
                button_factory(
                    label="OR Recent",
                    action="open_menu_card",
                    extra={"event_key": "provider_openrouter_recent"},
                ),
                button_factory(
                    label="OR Perf",
                    action="open_menu_card",
                    extra={"event_key": "provider_openrouter_performance"},
                ),
            ],
        },
        {
            "tag": "action",
            "actions": [
                button_factory(
                    label="NV Featured",
                    action="open_menu_card",
                    extra={"event_key": "provider_nvidia_featured"},
                    btn_type="primary",
                ),
                button_factory(
                    label="NV Recent",
                    action="open_menu_card",
                    extra={"event_key": "provider_nvidia_recent"},
                ),
                button_factory(
                    label="NV Perf",
                    action="open_menu_card",
                    extra={"event_key": "provider_nvidia_performance"},
                ),
            ],
        },
        {
            "tag": "action",
            "actions": [
                button_factory(label="Route Status", action="command_run", extra={"command_text": "/status"}),
                button_factory(label="Providers", action="command_run", extra={"command_text": "/provider"}),
                button_factory(label="Route", action="command_run", extra={"command_text": "/status"}),
            ],
        },
        {
            "tag": "action",
            "actions": [
                button_factory(
                    label="Personality",
                    action="open_menu_card",
                    extra={"event_key": "personality_picker"},
                ),
                button_factory(
                    label="Skill Combos",
                    action="open_menu_card",
                    extra={"event_key": "skill_combo_picker"},
                ),
                button_factory(
                    label="Commands",
                    action="open_menu_card",
                    extra={"event_key": "command_center"},
                ),
            ],
        },
        {
            "tag": "action",
            "actions": [
                button_factory(label="Close", action="registry_close_card", btn_type="default"),
            ],
        },
    ]
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Hermes Model Hub"},
            "template": "blue",
        },
        "elements": elements,
    }


def build_feishu_skill_combo_card(
    *,
    combos: list[dict[str, Any]],
    button_factory: ButtonFactory,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": "**Hermes Skill Combos**\nLoad a curated skill bundle for a common Feishu workflow.",
        }
    ]

    for combo in combos:
        combo_label = str(combo.get("label") or "").strip()
        combo_summary = str(combo.get("summary") or "").strip()
        suggested_personality = str(combo.get("suggested_personality") or "").strip()
        elements.append(
            {
                "tag": "markdown",
                "content": f"**{combo_label}**\n{combo_summary}",
            }
        )
        elements.append(
            {
                "tag": "action",
                "actions": [
                    button_factory(
                        label=f"Apply {combo_label}",
                        action="skill_combo_apply",
                        extra={
                            "combo_id": str(combo.get("id") or "").strip(),
                            "combo_label": combo_label,
                            "skills": list(combo.get("skills") or []),
                            "suggested_personality": suggested_personality,
                        },
                        btn_type="primary",
                    )
                ],
            }
        )

    elements.append(
        {
            "tag": "action",
            "actions": [
                button_factory(label="Close", action="registry_close_card", btn_type="default"),
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Hermes Skill Combos"},
            "template": "turquoise",
        },
        "elements": elements,
    }
