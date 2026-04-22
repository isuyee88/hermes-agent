from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, Callable


def build_runtime_bootstrap_debug_state(
    *,
    platforms: tuple[str, ...],
    prepare_runtime_environment: Callable[[], None],
    hermes_home_dir: Path,
) -> dict[str, Any]:
    prepare_runtime_environment()
    payload: dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "hermes_home": os.getenv("HERMES_HOME", str(hermes_home_dir)),
        "runtime_config_path": str(Path(os.getenv("HERMES_HOME", str(hermes_home_dir))) / "config.yaml"),
        "project_plugins_enabled_env": os.getenv("HERMES_ENABLE_PROJECT_PLUGINS", ""),
        "platforms": list(platforms),
    }

    config: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        loaded_config = load_config() or {}
        if isinstance(loaded_config, dict):
            config = loaded_config
        payload["config_loaded"] = True
    except Exception as exc:
        payload["config_loaded"] = False
        payload["config_error"] = str(exc)

    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    top_level_personalities = config.get("personalities") if isinstance(config.get("personalities"), dict) else {}
    nested_personalities = agent_cfg.get("personalities") if isinstance(agent_cfg.get("personalities"), dict) else {}
    payload["agent"] = {
        "default_model": str(model_cfg.get("default") or "").strip(),
        "default_personality": str(agent_cfg.get("personality") or "").strip(),
        "available_personalities": sorted(
            {
                str(name).strip()
                for name in (*top_level_personalities.keys(), *nested_personalities.keys())
                if str(name).strip()
            }
        ),
    }

    plugin_toolsets: list[tuple[str, str, str]] = []
    known_plugin_toolsets: set[str] = set()
    try:
        from hermes_cli.plugins import (
            _find_project_plugins_dir,
            discover_plugins,
            get_plugin_manager,
            get_plugin_toolsets,
        )

        project_plugins_dir = _find_project_plugins_dir()
        payload["project_plugins"] = {
            "search_dir": str(project_plugins_dir),
            "search_dir_exists": project_plugins_dir.is_dir(),
            "config_enabled": bool((config.get("plugins") or {}).get("enable_project", False)),
        }

        discover_plugins()
        plugin_toolsets = list(get_plugin_toolsets())
        known_plugin_toolsets = {str(name).strip() for name, _, _ in plugin_toolsets}
        manager = get_plugin_manager()
        payload["plugins"] = {
            "loaded": [
                {
                    "name": manifest_name,
                    "source": loaded.manifest.source,
                    "enabled": bool(loaded.enabled),
                    "path": loaded.manifest.path,
                    "tools_registered": list(loaded.tools_registered),
                    "hooks_registered": list(loaded.hooks_registered),
                    "error": loaded.error,
                }
                for manifest_name, loaded in sorted(manager._plugins.items())
            ],
            "toolsets": [
                {"key": key, "label": label, "description": description}
                for key, label, description in plugin_toolsets
            ],
        }
    except Exception as exc:
        payload["project_plugins"] = {
            "search_dir": "",
            "search_dir_exists": False,
            "config_enabled": bool((config.get("plugins") or {}).get("enable_project", False)),
            "error": str(exc),
        }
        payload["plugins"] = {"loaded": [], "toolsets": [], "error": str(exc)}

    try:
        from agent.skill_commands import get_configured_startup_skills

        payload["startup_skills"] = {
            "global": get_configured_startup_skills(config=config),
            "platforms": {
                platform: get_configured_startup_skills(config=config, platform=platform)
                for platform in platforms
            },
        }
    except Exception as exc:
        payload["startup_skills"] = {"global": [], "platforms": {}, "error": str(exc)}

    try:
        from hermes_cli.tools_config import _get_platform_tools
        from toolsets import get_toolset_names, resolve_toolset

        known_static_toolsets = {str(name).strip() for name in get_toolset_names()}
        configured_platform_toolsets = config.get("platform_toolsets") if isinstance(config.get("platform_toolsets"), dict) else {}
        platform_profiles: dict[str, Any] = {}
        for platform in platforms:
            configured_toolsets = sorted(
                str(name).strip()
                for name in (configured_platform_toolsets.get(platform) or [])
                if str(name).strip()
            )
            enabled_toolsets = sorted(str(name).strip() for name in _get_platform_tools(config, platform))
            resolved_tools = sorted(
                {
                    tool_name
                    for toolset_name in enabled_toolsets
                    for tool_name in resolve_toolset(toolset_name)
                }
            )
            unresolved_entries = [
                toolset_name
                for toolset_name in enabled_toolsets
                if toolset_name not in known_static_toolsets and toolset_name not in known_plugin_toolsets
            ]
            platform_profiles[platform] = {
                "configured_toolsets": configured_toolsets,
                "configured_toolset_count": len(configured_toolsets),
                "effective_toolsets": enabled_toolsets,
                "effective_toolset_count": len(enabled_toolsets),
                "resolved_tool_count": len(resolved_tools),
                "resolved_tool_sample": resolved_tools[:25],
                "unresolved_entries": unresolved_entries,
            }
        payload["platform_toolsets"] = platform_profiles
    except Exception as exc:
        payload["platform_toolsets"] = {"error": str(exc)}
    return payload


def build_modal_official_parity_state(
    *,
    settings_from_env: Callable[[], Any],
    get_camofox_url: Callable[[], str],
    is_local_camofox_url: Callable[[str], bool],
    is_camofox_healthcheck_ready: Callable[[str], bool],
    modal_public_webhook_platforms: tuple[str, ...],
) -> dict[str, Any]:
    settings = settings_from_env()
    browser_cli_available = shutil.which("agent-browser") is not None
    browser_node_available = shutil.which("node") is not None and shutil.which("npm") is not None
    camofox_url = get_camofox_url()
    camofox_local = is_local_camofox_url(camofox_url)
    camofox_ready = is_camofox_healthcheck_ready(camofox_url) if camofox_local else False
    honcho_available = importlib.util.find_spec("plugins.memory.honcho") is not None
    homeassistant_available = importlib.util.find_spec("tools.homeassistant_tool") is not None
    qq_sdk_available = importlib.util.find_spec("botpy") is not None

    features = {
        "core_agent_loop": {"status": "supported", "reason": "Modal entrypoint delegates task execution to Hermes AIAgent."},
        "session_learning_loop": {"status": "supported", "reason": "Session search, skill management, and background review loop ship in the Modal image."},
        "mcp_tools": {"status": "supported", "reason": "Modal image installs MCP support and runtime config can inject MCP servers."},
        "browser_tools": {
            "status": "supported" if (browser_cli_available or browser_node_available or camofox_ready) else "partial",
            "reason": (
                "Browser automation can run in Modal via agent-browser or a locally bootstrapped Camofox server."
                if (browser_cli_available or browser_node_available or camofox_ready)
                else "Node is present but no browser launcher was detected yet."
            ),
            "browser_cli_available": browser_cli_available,
            "node_runtime_available": browser_node_available,
            "camofox_url": camofox_url or None,
            "camofox_local": camofox_local,
            "camofox_ready": camofox_ready,
        },
        "send_message_tool": {"status": "supported", "reason": "Messaging toolset is enabled by default on the Modal profile."},
        "homeassistant_tools": {
            "status": "supported" if homeassistant_available else "partial",
            "reason": (
                "Home Assistant tool module is installed; feature still requires HASS_URL/HASS_TOKEN."
                if homeassistant_available
                else "Home Assistant integration dependencies are not present in the current image."
            ),
        },
        "honcho_memory_provider": {
            "status": "supported" if honcho_available else "partial",
            "reason": "Honcho plugin is present; still needs Honcho config/API credentials." if honcho_available else "Honcho plugin is not installed in the current image.",
        },
        "voice_mode_cli": {"status": "unsupported", "reason": "Interactive push-to-talk voice mode depends on a local audio device, which Modal web workers do not provide."},
        "rl_training_stack": {"status": "unsupported", "reason": "The production Modal profile does not install the official RL/Tinker extras or expose dedicated GPU training surfaces."},
    }
    platforms = {
        "telegram": {"status": "supported", "reason": "Webhook route is exposed on the Modal web app."},
        "feishu": {"status": "supported", "reason": "Webhook route is exposed and native Feishu adapter is wired through the chat queue."},
        "qq": {
            "status": "supported" if qq_sdk_available else "partial",
            "reason": "Webhook route is exposed and QQ SDK is installed." if qq_sdk_available else "Webhook route exists, but QQ SDK was not detected in the current image.",
        },
        "discord": {"status": "unsupported", "reason": "Official Discord adapter expects a long-lived gateway connection; the current Modal deployment exposes only webhook-style ingress."},
        "slack": {"status": "unsupported", "reason": "Slack Events / Socket Mode ingress is not exposed by the current Modal web surface."},
        "whatsapp": {"status": "unsupported", "reason": "WhatsApp bridge support requires long-lived bridge/session state not wired into this Modal profile."},
        "signal": {"status": "unsupported", "reason": "Signal transport depends on an external bridge/polling flow not wired into this Modal profile."},
        "email": {"status": "unsupported", "reason": "Inbound email polling is not exposed by the current Modal deployment shape."},
        "matrix": {"status": "unsupported", "reason": "Matrix sync requires a long-lived client loop that is not wired into the webhook-first Modal app."},
        "dingtalk": {"status": "unsupported", "reason": "DingTalk streaming ingress is not exposed by the current Modal deployment."},
    }
    return {
        "profile": "modal-webhook",
        "official_goal": "Align supportable official Hermes features on Modal; explicitly carve out Modal-incompatible surfaces.",
        "public_webhook_platforms": list(modal_public_webhook_platforms),
        "default_disabled_toolsets": list(settings.disabled_toolsets),
        "enabled_toolsets_override": list(settings.enabled_toolsets),
        "features": features,
        "platforms": platforms,
    }
