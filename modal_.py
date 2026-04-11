"""
Production Modal entrypoint for Hermes Agent.

This deployment keeps the public surface small while delegating real work to
Hermes itself:

- `run_agent_task`: authenticated single-turn/task execution through AIAgent
- `run_batch_tasks`: authenticated batch execution over a JSON task list
- `health_check`: runtime/import/config probe
- `web_app`: ASGI app exposing `/healthz`, `/invoke`, `/telegram/webhook`,
  `/feishu/webhook`, and `/qq/webhook`

Telegram and Feishu webhook requests are bridged into Hermes's official
gateway stack so authorization, sessions, memory, tools, slash commands, and
platform behavior come from the upstream codebase instead of a parallel
hand-written bot loop.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

try:
    from fastapi import FastAPI, Header, HTTPException, Request, Response
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover - local unit tests can still import helpers without FastAPI
    FastAPI = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]

try:
    import modal
except Exception:  # pragma: no cover - local unit tests can import helpers without a working Modal install
    modal = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

APP_NAME = os.getenv("HERMES_MODAL_APP_NAME", "hermes-agent")
DATA_ROOT = Path(os.getenv("HERMES_MODAL_DATA_DIR", "/data/hermes"))
SESSIONS_DIR = DATA_ROOT / "sessions"
UPDATES_PATH = DATA_ROOT / "telegram_updates.json"
FEISHU_EVENTS_PATH = DATA_ROOT / "feishu_events.json"
FEISHU_TRACE_PATH = DATA_ROOT / "feishu_trace.jsonl"
HERMES_HOME_DIR = Path(os.getenv("HERMES_HOME", "/data/hermes-home"))
DEFAULT_CONFIG_SOURCE = Path(__file__).with_name("config.modal.yaml")
DEFAULT_SUPERMEMORY_CONFIG_SOURCE = Path(__file__).with_name("supermemory.modal.json")
DEFAULT_DISABLED_TOOLSETS = ["browser", "messaging", "rl", "voice"]
DEFAULT_UPDATE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_SECRET_NAME = os.getenv("HERMES_MODAL_SECRET_NAME", "custom-secret")
DEFAULT_VOLUME_NAME = os.getenv("HERMES_MODAL_VOLUME_NAME", "hermes-agent-data")
DEFAULT_CHAT_QUEUE_NAME = os.getenv("HERMES_MODAL_CHAT_QUEUE_NAME", f"{APP_NAME}-chat-queue")
DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS = int(os.getenv("HERMES_MODAL_CHAT_QUEUE_CLAIM_TTL_SECONDS", "1800"))
DEFAULT_CHAT_QUEUE_BATCH_SIZE = int(os.getenv("HERMES_MODAL_CHAT_QUEUE_BATCH_SIZE", "8"))
DEFAULT_CRON_QUEUE_NAME = os.getenv("HERMES_MODAL_CRON_QUEUE_NAME", f"{APP_NAME}-cron-queue")
DEFAULT_CRON_QUEUE_CLAIM_TTL_SECONDS = int(os.getenv("HERMES_MODAL_CRON_QUEUE_CLAIM_TTL_SECONDS", "1800"))
DEFAULT_CRON_QUEUE_BATCH_SIZE = int(os.getenv("HERMES_MODAL_CRON_QUEUE_BATCH_SIZE", "8"))
DEFAULT_CRON_QUEUE_WORKERS = int(os.getenv("HERMES_MODAL_CRON_QUEUE_WORKERS", "2"))
DEFAULT_SESSION_ROUTE_TTL_SECONDS = int(os.getenv("HERMES_SESSION_ROUTE_TTL_SECONDS", "7200"))
_ANTHROPIC_DATED_MODEL_RE = re.compile(r"^(anthropic\/.+)-(\d{8})$")
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"^\d{6,}:[A-Za-z0-9_-]{20,}$")
ROUTING_STATE_PATH = DATA_ROOT / "free_model_routing.json"
CHAT_QUEUE_CLAIMS_PATH = DATA_ROOT / "chat_queue_claims.json"
CRON_QUEUE_CLAIMS_PATH = DATA_ROOT / "cron_queue_claims.json"
TELEGRAM_WEBHOOK_SYNC_STATE_PATH = DATA_ROOT / "telegram_webhook_sync.json"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_POPULAR_MODELS_URL = "https://build.nvidia.com/models?orderBy=weightPopular%3ADESC"
ROUTING_REFRESH_TTL_SECONDS = 24 * 60 * 60
DEFAULT_TELEGRAM_WEBHOOK_SYNC_BACKOFF_SECONDS = int(
    os.getenv("TELEGRAM_WEBHOOK_SYNC_BACKOFF_SECONDS", "300")
)
INVALID_MODEL_ERROR_MARKERS = (
    "not a valid model id",
    "invalid model",
    "model_not_found",
    "no such model",
    "unknown model",
)
TRANSIENT_ROUTE_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "readtimeout",
    "connecttimeout",
    "connection reset",
    "connection closed",
    "connection lost",
    "network error",
    "upstream connect error",
    "interrupted during api call",
)
FREE_MODEL_ALIASES = {"openrouter/free", "free"}
DEFAULT_NVIDIA_FREE_MODELS = [
    "qwen/qwq-32b",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
]
NVIDIA_NON_CHAT_MODEL_MARKERS = (
    "embed",
    "embedding",
    "retriever",
    "rerank",
    "asr",
    "tts",
    "speech",
    "audio",
    "guardrail",
    "safety",
)
NVIDIA_CHAT_FAMILY_MARKERS = (
    "instruct",
    "chat",
    "gpt",
    "qwen",
    "deepseek",
    "llama",
    "glm",
    "kimi",
    "nemotron",
    "minimax",
    "step",
    "mistral",
    "command",
    "granite",
)

_UPDATE_LOCK = threading.Lock()
_FEISHU_EVENT_LOCK = threading.Lock()
_CHAT_QUEUE_LOCK = threading.Lock()
_CRON_QUEUE_LOCK = threading.Lock()
_TELEGRAM_RUNTIME_LOCK: asyncio.Lock | None = None
_TELEGRAM_RUNTIME: "_TelegramGatewayRuntime | None" = None
_FEISHU_RUNTIME_LOCK: asyncio.Lock | None = None
_FEISHU_RUNTIME: "_FeishuGatewayRuntime | None" = None
_QQ_RUNTIME_LOCK: asyncio.Lock | None = None
_QQ_RUNTIME: "_QQGatewayRuntime | None" = None
CHAT_QUEUE = None
CRON_QUEUE = None
MODAL_VOLUME = None


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _is_valid_telegram_bot_token_format(value: str | None) -> bool:
    raw = str(value or "").strip()
    return bool(_TELEGRAM_BOT_TOKEN_RE.fullmatch(raw))


def _ensure_runtime_dirs() -> None:
    for path in (DATA_ROOT, SESSIONS_DIR, HERMES_HOME_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _is_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_feishu_menu_manifest() -> dict[str, Any]:
    items = [
        {
            "label": "模型切换",
            "event_key": "model_picker",
            "description": "打开 Hermes 的飞书交互式模型选择卡。",
            "recommended": True,
        },
        {
            "label": "当前模型",
            "event_key": "model_status",
            "description": "显示当前会话的模型、provider、route mode。",
            "recommended": True,
        },
        {
            "label": "Provider 状态",
            "event_key": "provider_status",
            "description": "显示 OpenRouter 和 NVIDIA 的认证与当前状态。",
            "recommended": True,
        },
        {
            "label": "切到 OpenRouter",
            "event_key": "provider_openrouter",
            "description": "直接打开 OpenRouter 的模型列表。",
            "recommended": False,
        },
        {
            "label": "切到 NVIDIA",
            "event_key": "provider_nvidia",
            "description": "直接打开 NVIDIA 的模型列表。",
            "recommended": False,
        },
    ]
    return {
        "platform": "feishu",
        "version": 1,
        "menu_items": items,
        "recommended_order": [item["event_key"] for item in items],
        "supported_event_keys": {item["event_key"]: item["label"] for item in items},
        "notes": [
            "在飞书应用后台的机器人菜单中，把每个菜单项的事件键设置为这里的 event_key。",
            "菜单项发布后，Hermes 会把点击事件转换成 /model 或 /provider 命令，沿用现有会话粘性与手工切换逻辑。",
            "模型切换推荐优先使用 model_picker，不建议只保留 provider_openrouter/provider_nvidia 两个快捷项。",
        ],
    }


def _sync_runtime_config() -> str | None:
    source = Path(os.getenv("HERMES_MODAL_CONFIG_SOURCE", str(DEFAULT_CONFIG_SOURCE)))
    if not source.exists():
        return None

    target = HERMES_HOME_DIR / "config.yaml"
    force_sync = _is_truthy(os.getenv("HERMES_MODAL_SYNC_CONFIG"), default=True)
    source_text = source.read_text(encoding="utf-8")

    resolved_text = source_text
    try:
        import yaml
        from hermes_cli.config import _expand_env_vars

        config_payload = yaml.safe_load(source_text) or {}
        if isinstance(config_payload, dict):
            config_payload = _expand_env_vars(config_payload)
            model_config = config_payload.get("model")
            if isinstance(model_config, dict):
                default_model = str(model_config.get("default") or "").strip()
                if not default_model or default_model.startswith("${"):
                    model_config["default"] = os.getenv("DEFAULT_MODEL", "openrouter/free")
            _materialize_dynamic_free_model_config(config_payload)
            resolved_text = yaml.safe_dump(
                config_payload,
                allow_unicode=True,
                sort_keys=False,
            )
    except Exception as exc:
        logger.warning("Falling back to raw Modal config sync without env expansion: %s", exc)

    if target.exists() and not force_sync:
        return str(target)

    if not target.exists() or target.read_text(encoding="utf-8") != resolved_text:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(resolved_text, encoding="utf-8")
    return str(target)


def _sync_runtime_support_file(
    *,
    env_var_name: str,
    default_source: Path,
    target_name: str,
) -> str | None:
    source = Path(os.getenv(env_var_name, str(default_source)))
    if not source.exists():
        return None

    target = HERMES_HOME_DIR / target_name
    force_sync = _is_truthy(os.getenv("HERMES_MODAL_SYNC_CONFIG"), default=True)
    source_text = source.read_text(encoding="utf-8")

    if target.exists() and not force_sync:
        return str(target)

    if not target.exists() or target.read_text(encoding="utf-8") != source_text:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source_text, encoding="utf-8")
    return str(target)


def _sync_supermemory_config() -> str | None:
    return _sync_runtime_support_file(
        env_var_name="HERMES_MODAL_SUPERMEMORY_CONFIG_SOURCE",
        default_source=DEFAULT_SUPERMEMORY_CONFIG_SOURCE,
        target_name="supermemory.json",
    )


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Ignoring unreadable JSON file at %s", path)
        return default


def _safe_json_loads(payload: str, default: Any) -> Any:
    try:
        return json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return default


def _extract_feishu_trace_context(payload: dict[str, Any]) -> dict[str, Any]:
    header = payload.get("header") or {}
    event = payload.get("event") or {}
    message = event.get("message") or {}
    sender = event.get("sender") or {}
    sender_id = sender.get("sender_id") or {}
    operator = event.get("operator") or {}
    operator_id = operator.get("operator_id") or {}
    context = event.get("context") or {}
    chat = event.get("chat") or {}
    return {
        "type": str(payload.get("type") or "").strip(),
        "event_type": str(header.get("event_type") or payload.get("event_type") or "").strip(),
        "event_id": str(header.get("event_id") or payload.get("event_id") or "").strip(),
        "message_id": str(message.get("message_id") or "").strip(),
        "chat_type": str(message.get("chat_type") or chat.get("chat_type") or event.get("chat_type") or "").strip(),
        "chat_id": str(message.get("chat_id") or chat.get("chat_id") or context.get("open_chat_id") or event.get("chat_id") or "").strip(),
        "sender_open_id": str(sender_id.get("open_id") or operator_id.get("open_id") or operator.get("open_id") or event.get("open_id") or "").strip(),
        "sender_user_id": str(sender_id.get("user_id") or operator_id.get("user_id") or operator.get("user_id") or event.get("user_id") or "").strip(),
    }


def _append_feishu_trace(stage: str, payload: dict[str, Any], **extra: Any) -> None:
    trace = {
        "ts": int(time.time()),
        "stage": stage,
        **_extract_feishu_trace_context(payload),
    }
    if extra:
        trace.update(extra)
    line = json.dumps(trace, ensure_ascii=False)
    with _FEISHU_EVENT_LOCK:
        FEISHU_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEISHU_TRACE_PATH.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


def _read_feishu_trace(limit: int = 100) -> list[dict[str, Any]]:
    if limit <= 0 or not FEISHU_TRACE_PATH.exists():
        return []
    try:
        lines = FEISHU_TRACE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for raw in lines[-limit:]:
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _score_free_model(model_id: str) -> tuple[int, int, int, str]:
    normalized = model_id.lower()
    family_rank = 0
    if "qwen3-coder" in normalized:
        family_rank = 100
    elif "qwq" in normalized:
        family_rank = 95
    elif "deepseek" in normalized and "r1" in normalized:
        family_rank = 92
    elif "glm-4.7" in normalized:
        family_rank = 90
    elif "llama-4" in normalized:
        family_rank = 88
    elif "llama-3.3" in normalized:
        family_rank = 86
    elif "llama-3.1-70b" in normalized:
        family_rank = 84
    elif "70b" in normalized:
        family_rank = 80
    elif "32b" in normalized:
        family_rank = 70
    elif "27b" in normalized:
        family_rank = 68
    elif "14b" in normalized:
        family_rank = 60
    elif "8b" in normalized:
        family_rank = 50

    parameter_rank = 0
    for size, rank in (
        ("405b", 405),
        ("480b", 400),
        ("236b", 236),
        ("120b", 120),
        ("90b", 90),
        ("70b", 70),
        ("49b", 49),
        ("32b", 32),
        ("27b", 27),
        ("14b", 14),
        ("8b", 8),
    ):
        if size in normalized:
            parameter_rank = rank
            break

    explicit_free_bonus = 1 if normalized.endswith(":free") else 0
    return (family_rank, parameter_rank, explicit_free_bonus, model_id)


def _extract_openrouter_free_model_candidates(payload: dict[str, Any]) -> list[str]:
    candidates: list[str] = ["openrouter/free"]
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        pricing = item.get("pricing") or {}
        prompt_price = str(pricing.get("prompt") or "").strip()
        completion_price = str(pricing.get("completion") or "").strip()
        if model_id.endswith(":free") or (
            prompt_price in {"0", "0.0", "0.000000"} and completion_price in {"0", "0.0", "0.000000"}
        ):
            candidates.append(model_id)
    deduped = _dedupe_keep_order(candidates)
    ranked = [deduped[0]] + sorted(deduped[1:], key=_score_free_model, reverse=True)
    return ranked


def _fetch_openrouter_free_model_candidates() -> list[str]:
    import httpx

    headers = {}
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with httpx.Client(timeout=20) as client:
        response = client.get(f"{DEFAULT_OPENROUTER_BASE_URL}/models", headers=headers)
        response.raise_for_status()
        payload = response.json()
    return _extract_openrouter_free_model_candidates(payload if isinstance(payload, dict) else {})


def _expand_nvidia_model_variants(model_id: str) -> list[str]:
    normalized = str(model_id or "").strip().lower()
    if not normalized or "/" not in normalized:
        return []
    variants = [normalized]
    dotted = re.sub(r"(\d)_(\d)", r"\1.\2", normalized)
    if dotted != normalized:
        variants.append(dotted)
    return _dedupe_keep_order(variants)


def _is_probable_nvidia_chat_model(model_id: str) -> bool:
    normalized = str(model_id or "").strip().lower()
    if not normalized or "/" not in normalized:
        return False
    if any(marker in normalized for marker in NVIDIA_NON_CHAT_MODEL_MARKERS):
        return False
    return any(marker in normalized for marker in NVIDIA_CHAT_FAMILY_MARKERS)


def _extract_nvidia_popular_model_candidates(html: str) -> list[str]:
    pairs = re.findall(r'href="/([a-z0-9_.-]+)/([a-z0-9_.-]+)"', html, flags=re.I)
    candidates: list[str] = []
    for publisher, slug in pairs:
        publisher = str(publisher or "").strip().lower()
        slug = str(slug or "").strip().lower()
        if not publisher or not slug:
            continue
        if publisher in {"models", "explore", "blueprints", "settings", "docs"}:
            continue
        if slug in {"models", "explore", "blueprints", "settings", "docs"}:
            continue
        candidate = f"{publisher}/{slug}"
        if not _is_probable_nvidia_chat_model(candidate):
            continue
        candidates.extend(_expand_nvidia_model_variants(candidate))
    return _dedupe_keep_order(candidates)


def _fetch_nvidia_popular_model_candidates() -> list[str]:
    import httpx

    url = os.getenv("NVIDIA_POPULAR_MODELS_URL") or DEFAULT_NVIDIA_POPULAR_MODELS_URL
    with httpx.Client(timeout=20) as client:
        response = client.get(url)
        response.raise_for_status()
        html = response.text
    candidates = _extract_nvidia_popular_model_candidates(html)
    try:
        limit = int(os.getenv("NVIDIA_POPULAR_MODELS_LIMIT", "48") or "48")
    except ValueError:
        limit = 48
    if limit > 0:
        candidates = candidates[:limit]
    return candidates


def _load_nvidia_free_model_candidates() -> list[str]:
    configured = _dedupe_keep_order(_split_csv(os.getenv("NVIDIA_FREE_MODELS")))
    if configured:
        return configured
    candidates = list(DEFAULT_NVIDIA_FREE_MODELS)
    enable_popular = _is_truthy(os.getenv("NVIDIA_POPULAR_MODELS_ENABLED"), default=True)
    if not enable_popular:
        return candidates
    try:
        popular = _fetch_nvidia_popular_model_candidates()
    except Exception as exc:
        logger.warning("Failed refreshing NVIDIA popular models: %s", exc)
        return candidates
    return _dedupe_keep_order(popular + candidates)


def _load_routing_state() -> dict[str, Any]:
    payload = _load_json_file(ROUTING_STATE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_routing_state(payload: dict[str, Any]) -> None:
    _atomic_json_write(ROUTING_STATE_PATH, payload)


def _refresh_free_model_routes(force: bool = False) -> dict[str, Any]:
    _ensure_runtime_dirs()
    existing = _load_routing_state()
    now = int(time.time())
    if not force:
        refreshed_at = int(existing.get("refreshed_at") or 0)
        if refreshed_at and now - refreshed_at < ROUTING_REFRESH_TTL_SECONDS:
            return existing

    providers: dict[str, Any] = {}

    try:
        providers["openrouter"] = {
            "base_url": os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
            "candidates": _fetch_openrouter_free_model_candidates(),
        }
    except Exception as exc:
        logger.warning("Failed refreshing OpenRouter free models: %s", exc)
        providers["openrouter"] = {
            "base_url": os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
            "candidates": ["openrouter/free"],
        }

    nvidia_candidates = _load_nvidia_free_model_candidates()
    if nvidia_candidates:
        providers["nvidia"] = {
            "base_url": os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL,
            "candidates": nvidia_candidates,
        }

    payload = {
        "refreshed_at": now,
        "providers": providers,
    }
    _save_routing_state(payload)
    return payload


def _is_dynamic_free_route_alias(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in FREE_MODEL_ALIASES


def _route_to_runtime_model_config(route: dict[str, Any]) -> dict[str, Any]:
    provider_name = str(route.get("provider") or "").strip().lower()
    payload: dict[str, Any] = {
        "default": str(route.get("model") or "").strip(),
        "provider": "openrouter" if provider_name == "openrouter" else "custom",
        "base_url": str(route.get("base_url") or "").strip(),
    }
    api_key = str(route.get("api_key") or "").strip()
    if provider_name != "openrouter" and api_key:
        payload["api_key"] = api_key
    return {key: value for key, value in payload.items() if value}


def _route_to_fallback_provider(route: dict[str, Any]) -> dict[str, Any]:
    provider_name = str(route.get("provider") or "").strip().lower()
    payload: dict[str, Any] = {
        "provider": "openrouter" if provider_name == "openrouter" else "custom",
        "model": str(route.get("model") or "").strip(),
        "base_url": str(route.get("base_url") or "").strip(),
    }
    api_key = str(route.get("api_key") or "").strip()
    if provider_name != "openrouter" and api_key:
        payload["api_key"] = api_key
    return {key: value for key, value in payload.items() if value}


def _select_dynamic_primary_route(*, force_refresh: bool = False) -> dict[str, Any] | None:
    preferred_provider = str(
        os.getenv("HERMES_FREE_MODEL_PRIMARY_PROVIDER")
        or ("openrouter" if os.getenv("OPENROUTER_API_KEY", "").strip() else "nvidia")
    ).strip().lower() or None
    state = _refresh_free_model_routes(force=force_refresh)
    routes = _candidate_routes_from_state(state, preferred_provider=preferred_provider)
    return routes[0] if routes else None


def _select_dynamic_fallback_routes(primary_route: dict[str, Any] | None) -> list[dict[str, Any]]:
    preferred_provider = None
    if primary_route:
        primary_provider = str(primary_route.get("provider") or "").strip().lower()
        if primary_provider == "openrouter" and os.getenv("NVIDIA_API_KEY", "").strip():
            preferred_provider = "nvidia"
        elif primary_provider == "nvidia" and os.getenv("OPENROUTER_API_KEY", "").strip():
            preferred_provider = "openrouter"

    state = _refresh_free_model_routes(force=False)
    routes = _candidate_routes_from_state(
        state,
        preferred_provider=preferred_provider,
        failed_route=primary_route,
    )
    fallbacks: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for route in routes:
        key = (
            str(route.get("provider") or "").strip().lower(),
            str(route.get("model") or "").strip(),
            str(route.get("base_url") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        fallbacks.append(route)
        if len(fallbacks) >= 4:
            break
    return fallbacks


def _materialize_dynamic_free_model_config(config_payload: dict[str, Any]) -> None:
    model_config = config_payload.get("model")
    if not isinstance(model_config, dict):
        return

    default_model = str(model_config.get("default") or "").strip()
    if not _is_dynamic_free_route_alias(default_model):
        return

    primary_route = _select_dynamic_primary_route(force_refresh=False)
    if primary_route is None:
        return

    model_config.clear()
    model_config.update(_route_to_runtime_model_config(primary_route))

    fallback_routes = _select_dynamic_fallback_routes(primary_route)
    if fallback_routes:
        config_payload["fallback_providers"] = [
            _route_to_fallback_provider(route)
            for route in fallback_routes
        ]
    elif "fallback_providers" in config_payload:
        config_payload.pop("fallback_providers", None)


def _is_invalid_model_error_text(error_text: str | None) -> bool:
    normalized = str(error_text or "").strip().lower()
    return any(marker in normalized for marker in INVALID_MODEL_ERROR_MARKERS)


def _is_transient_route_error_text(error_text: str | None) -> bool:
    normalized = str(error_text or "").strip().lower()
    return any(marker in normalized for marker in TRANSIENT_ROUTE_ERROR_MARKERS)


def _route_from_settings(settings: "RuntimeSettings", model_name: Optional[str]) -> dict[str, Any]:
    return {
        "provider": settings.provider or ("openrouter" if os.getenv("OPENROUTER_API_KEY") else ""),
        "base_url": settings.base_url or (
            (os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL)
            if os.getenv("OPENROUTER_API_KEY")
            else None
        ),
        "api_key": settings.api_key,
        "model": model_name or settings.model,
    }


def _resolve_primary_route(settings: "RuntimeSettings", model_name: Optional[str]) -> dict[str, Any]:
    primary_route = _route_from_settings(settings, model_name)
    if model_name and not _is_dynamic_free_route_alias(model_name):
        return primary_route
    if not _is_dynamic_free_route_alias(primary_route.get("model")):
        return primary_route
    selected = _select_dynamic_primary_route(force_refresh=False)
    return selected or primary_route


def _candidate_routes_from_state(
    state: dict[str, Any],
    *,
    preferred_provider: str | None = None,
    failed_route: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    providers = state.get("providers") or {}
    provider_order = list(providers.keys())
    if preferred_provider and preferred_provider in provider_order:
        provider_order = [preferred_provider] + [name for name in provider_order if name != preferred_provider]

    routes: list[dict[str, Any]] = []
    for provider_name in provider_order:
        entry = providers.get(provider_name) or {}
        candidates = entry.get("candidates") or []
        base_url = str(entry.get("base_url") or "").strip()
        if provider_name == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        elif provider_name == "nvidia":
            api_key = (os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY") or "").strip()
        else:
            api_key = ""
        if not api_key or not base_url:
            continue
        for candidate in candidates:
            route = {
                "provider": provider_name,
                "base_url": base_url,
                "api_key": api_key,
                "model": str(candidate).strip(),
            }
            if not route["model"]:
                continue
            if failed_route and route["provider"] == failed_route.get("provider") and route["model"] == failed_route.get("model"):
                continue
            routes.append(route)
    return routes


def _select_retry_route_for_result(
    primary_route: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    refresh_reason = _determine_route_refresh_reason(result)
    if not refresh_reason:
        return None

    preferred_provider = None
    primary_provider = str(primary_route.get("provider") or "").strip().lower()
    if primary_provider == "openrouter" and os.getenv("NVIDIA_API_KEY", "").strip():
        preferred_provider = "nvidia"
    elif primary_provider == "nvidia" and os.getenv("OPENROUTER_API_KEY", "").strip():
        preferred_provider = "openrouter"

    return _refresh_and_select_valid_route(
        preferred_provider=preferred_provider,
        failed_route=primary_route,
    )


def _probe_model_route(route: dict[str, Any]) -> bool:
    import httpx

    headers = {"Authorization": f"Bearer {route['api_key']}"}
    with httpx.Client(timeout=20) as client:
        response = client.post(
            f"{str(route['base_url']).rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": route["model"],
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 4,
                "temperature": 0,
            },
        )
    return response.status_code == 200


def _refresh_and_select_valid_route(
    *,
    preferred_provider: str | None = None,
    failed_route: Optional[dict[str, Any]] = None,
) -> dict[str, Any] | None:
    state = _refresh_free_model_routes(force=True)
    for route in _candidate_routes_from_state(
        state,
        preferred_provider=preferred_provider,
        failed_route=failed_route,
    ):
        try:
            if _probe_model_route(route):
                return route
        except Exception as exc:
            logger.warning("Probe failed for %s/%s: %s", route["provider"], route["model"], exc)
    return None


def _session_file(session_key: str) -> Path:
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return SESSIONS_DIR / f"{digest}.json"


def _load_session_state(session_key: str) -> dict[str, Any]:
    state = _load_json_file(
        _session_file(session_key),
        {
            "session_key": session_key,
            "session_id": str(uuid.uuid4()),
            "messages": [],
            "route_lease": None,
            "route_debug": {},
            "route_metrics": {},
            "updated_at": 0,
        },
    )
    state["session_key"] = session_key
    state.setdefault("session_id", str(uuid.uuid4()))
    state.setdefault("messages", [])
    state.setdefault("route_lease", None)
    state.setdefault("route_debug", {})
    state.setdefault("route_metrics", {})
    state.setdefault("updated_at", 0)
    return state


def _save_session_state(
    session_key: str,
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    route_lease: dict[str, Any] | None = None,
    route_debug: dict[str, Any] | None = None,
    route_metrics: dict[str, Any] | None = None,
) -> None:
    _atomic_json_write(
        _session_file(session_key),
        {
            "session_key": session_key,
            "session_id": session_id,
            "messages": messages,
            "route_lease": route_lease,
            "route_debug": route_debug or {},
            "route_metrics": route_metrics or {},
            "updated_at": int(time.time()),
        },
    )


def _route_api_key_source(provider_name: str) -> str:
    normalized = str(provider_name or "").strip().lower()
    if normalized == "openrouter":
        return "OPENROUTER_API_KEY"
    if normalized == "nvidia":
        return "NVIDIA_API_KEY"
    return normalized or "unknown"


def _build_route_lease(
    route: dict[str, Any],
    *,
    selection_reason: str,
    selected_at: int | None = None,
    last_success_at: int | None = None,
    fail_count: int = 0,
    lease_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    now = int(time.time())
    selected_ts = int(selected_at or now)
    success_ts = int(last_success_at or selected_ts)
    ttl = max(int(lease_ttl_seconds or DEFAULT_SESSION_ROUTE_TTL_SECONDS), 1)
    provider_name = str(route.get("provider") or "").strip().lower()
    return {
        "provider": provider_name,
        "model": str(route.get("model") or "").strip(),
        "base_url": str(route.get("base_url") or "").strip(),
        "api_key_source": _route_api_key_source(provider_name),
        "selected_at": selected_ts,
        "last_success_at": success_ts,
        "fail_count": max(int(fail_count), 0),
        "lease_expires_at": success_ts + ttl,
        "selection_reason": str(selection_reason or "fresh_select").strip() or "fresh_select",
    }


def _refresh_route_lease(
    existing_lease: dict[str, Any] | None,
    route: dict[str, Any],
    *,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    selected_at = int((existing_lease or {}).get("selected_at") or time.time())
    chosen_reason = str(
        selection_reason
        or (existing_lease or {}).get("selection_reason")
        or "fresh_select"
    ).strip() or "fresh_select"
    return _build_route_lease(
        route,
        selection_reason=chosen_reason,
        selected_at=selected_at,
        last_success_at=int(time.time()),
        fail_count=0,
    )


def _expire_route_lease(
    existing_lease: dict[str, Any] | None,
    *,
    error_text: str = "",
    failure_reason: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(existing_lease, dict):
        return None
    lease = dict(existing_lease)
    lease["fail_count"] = int(lease.get("fail_count") or 0) + 1
    lease["lease_expires_at"] = int(time.time()) - 1
    if error_text:
        lease["last_error"] = str(error_text)
    if failure_reason:
        lease["last_failure_reason"] = str(failure_reason)
    return lease


def _hydrate_route_from_lease(settings: "RuntimeSettings", lease: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(lease, dict):
        return None
    provider_name = str(lease.get("provider") or "").strip().lower()
    model_name = str(lease.get("model") or "").strip()
    base_url = str(lease.get("base_url") or "").strip()
    if not provider_name or not model_name or not base_url:
        return None

    api_key = ""
    if provider_name == "openrouter":
        api_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
    elif provider_name == "nvidia":
        api_key = str(os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY") or "").strip()
    elif settings.provider == provider_name and settings.api_key:
        api_key = str(settings.api_key)
    if not api_key:
        return None
    return {
        "provider": provider_name,
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
    }


def _is_route_lease_active(settings: "RuntimeSettings", lease: dict[str, Any] | None) -> bool:
    route = _hydrate_route_from_lease(settings, lease)
    if route is None:
        return False
    expires_at = int((lease or {}).get("lease_expires_at") or 0)
    return expires_at > int(time.time())


def _increment_route_metric(route_metrics: dict[str, Any], key: str) -> dict[str, Any]:
    metrics = dict(route_metrics or {})
    metrics[key] = int(metrics.get(key) or 0) + 1
    metrics["updated_at"] = int(time.time())
    return metrics


def _determine_route_refresh_reason(result: dict[str, Any]) -> str | None:
    error_text = str(result.get("error") or "").strip()
    final_response = str(result.get("final_response") or "").strip()
    if _is_invalid_model_error_text(error_text):
        return "invalid_model"
    if _is_transient_route_error_text(error_text):
        return "transient_error"
    if result.get("interrupted"):
        return "interrupted"
    if not final_response and not result.get("completed", True):
        return "incomplete_result"
    return None


def _iter_session_route_payloads(limit: int = 20) -> list[dict[str, Any]]:
    if limit <= 0 or not SESSIONS_DIR.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for path in SESSIONS_DIR.glob("*.json"):
        payload = _load_json_file(path, {})
        if isinstance(payload, dict):
            payloads.append(payload)
    payloads.sort(key=lambda item: int(item.get("updated_at") or 0), reverse=True)
    return payloads[:limit]


def _build_recent_session_route_summaries(limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = int(time.time())
    for payload in _iter_session_route_payloads(limit=limit):
        lease = payload.get("route_lease") or {}
        route_debug = payload.get("route_debug") or {}
        metrics = payload.get("route_metrics") or {}
        rows.append(
            {
                "session_key": payload.get("session_key"),
                "session_id": payload.get("session_id"),
                "updated_at": payload.get("updated_at"),
                "route_lease": lease if isinstance(lease, dict) else None,
                "route_selection": route_debug.get("last_route_selection"),
                "last_failure_reason": route_debug.get("last_failure_reason"),
                "last_error": route_debug.get("last_error"),
                "lease_ttl_remaining_seconds": max(int((lease or {}).get("lease_expires_at") or 0) - now, 0),
                "metrics": metrics if isinstance(metrics, dict) else {},
            }
        )
    return rows


def _aggregate_session_route_metrics(limit: int = 200) -> dict[str, Any]:
    totals = {
        "sticky_hit": 0,
        "fresh_select": 0,
        "explicit_override": 0,
        "refreshed_after_failure": 0,
    }
    sessions_with_lease = 0
    for payload in _iter_session_route_payloads(limit=limit):
        lease = payload.get("route_lease")
        if isinstance(lease, dict) and lease:
            sessions_with_lease += 1
        metrics = payload.get("route_metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for key in totals:
            totals[key] += int(metrics.get(key) or 0)
    totals["sessions_with_route_lease"] = sessions_with_lease
    totals["sampled_sessions"] = len(_iter_session_route_payloads(limit=limit))
    return totals


def _prune_seen_updates(seen_updates: dict[str, int], ttl_seconds: int) -> dict[str, int]:
    now = int(time.time())
    cutoff = now - ttl_seconds
    return {key: ts for key, ts in seen_updates.items() if ts >= cutoff}


def _mark_update_seen(update_id: str, ttl_seconds: int = DEFAULT_UPDATE_TTL_SECONDS) -> bool:
    with _UPDATE_LOCK:
        seen_updates = _load_json_file(UPDATES_PATH, {})
        if not isinstance(seen_updates, dict):
            seen_updates = {}
        seen_updates = _prune_seen_updates(seen_updates, ttl_seconds)
        if update_id in seen_updates:
            return False
        seen_updates[update_id] = int(time.time())
        _atomic_json_write(UPDATES_PATH, seen_updates)
        return True


def _mark_feishu_event_seen(event_id: str, ttl_seconds: int = DEFAULT_UPDATE_TTL_SECONDS) -> bool:
    normalized = str(event_id or "").strip()
    if not normalized:
        return True
    with _FEISHU_EVENT_LOCK:
        seen_events = _load_json_file(FEISHU_EVENTS_PATH, {})
        if not isinstance(seen_events, dict):
            seen_events = {}
        seen_events = _prune_seen_updates(seen_events, ttl_seconds)
        if normalized in seen_events:
            return False
        seen_events[normalized] = int(time.time())
        _atomic_json_write(FEISHU_EVENTS_PATH, seen_events)
        return True


def _extract_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    tool_names: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            name = None
            if isinstance(tool_call, dict):
                function_data = tool_call.get("function") or {}
                if isinstance(function_data, dict):
                    name = function_data.get("name")
            else:
                function_data = getattr(tool_call, "function", None)
                name = getattr(function_data, "name", None)
            if isinstance(name, str) and name:
                tool_names.append(name)
    return tool_names


def _validate_bearer_token(header_value: str | None, expected_token: str | None) -> bool:
    if not expected_token:
        return True
    if not header_value:
        return False
    prefix = "Bearer "
    if not header_value.startswith(prefix):
        return False
    provided = header_value[len(prefix) :].strip()
    return bool(provided) and provided == expected_token


def _validate_telegram_secret(header_value: str | None, expected_secret: str | None) -> bool:
    if not expected_secret:
        return True
    return bool(header_value) and header_value == expected_secret


def _pick_runtime_api_config() -> tuple[Optional[str], Optional[str], Optional[str]]:
    provider = os.getenv("HERMES_PROVIDER") or None
    base_url = (
        os.getenv("HERMES_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or None
    )
    api_key = (
        os.getenv("HERMES_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or None
    )
    if not provider and not base_url and os.getenv("OPENROUTER_API_KEY"):
        provider = "openrouter"
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    return provider, base_url, api_key


def _get_memory_provider_status() -> dict[str, Any]:
    provider_name = ""
    config_error = None
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        provider_name = str(((config.get("memory") or {}).get("provider")) or "").strip()
    except Exception as exc:
        config_error = str(exc)

    status = {
        "provider": provider_name,
        "configured": bool(provider_name),
    }
    if config_error:
        status["config_error"] = config_error

    if provider_name == "supermemory":
        config_path = Path(os.getenv("HERMES_HOME", str(HERMES_HOME_DIR))) / "supermemory.json"
        status["api_key_configured"] = bool(os.getenv("SUPERMEMORY_API_KEY", "").strip())
        status["sdk_available"] = importlib.util.find_spec("supermemory") is not None
        status["container_tag_override"] = os.getenv("SUPERMEMORY_CONTAINER_TAG", "").strip()
        status["config_path"] = str(config_path)
        status["config_present"] = config_path.exists()

    return status


def _resolve_runtime_model_name(model_name: str, provider: Optional[str]) -> str:
    if provider == "openrouter":
        match = _ANTHROPIC_DATED_MODEL_RE.fullmatch(model_name.strip())
        if match:
            return match.group(1)
    return model_name


def _prepare_runtime_environment() -> None:
    os.environ.setdefault("HERMES_HOME", str(HERMES_HOME_DIR))
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ["TERMINAL_ENV"] = os.getenv("HERMES_MODAL_TERMINAL_ENV", "local")
    os.environ.setdefault("HERMES_ENABLE_PROJECT_PLUGINS", "true")
    # Modal webhook workers are short-lived per invocation; disable delayed
    # Feishu text batching to avoid pending flush tasks being cancelled when
    # the worker event loop exits.
    os.environ.setdefault("HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS", "0")
    os.environ.setdefault("HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES", "1")
    os.environ.setdefault("HERMES_FEISHU_RESOLVE_SENDER_NAMES", "false")
    os.environ.setdefault("HERMES_FEISHU_MENU_OPEN_BY_OPEN_ID", "true")
    # Feishu webhook traffic is the most sensitive to runaway multi-tool loops.
    # Keep the default surface focused on fast chat/research unless the user
    # explicitly re-enables heavier toolsets via secrets/env overrides.
    os.environ.setdefault(
        "HERMES_FEISHU_DISABLED_TOOLSETS",
        "skills,browser,terminal,code_execution,delegation,tts,messaging,rl",
    )
    if "NGC_API_KEY" in os.environ and "NVIDIA_API_KEY" not in os.environ:
        os.environ["NVIDIA_API_KEY"] = os.environ["NGC_API_KEY"]
    os.environ.setdefault("HERMES_STREAM_READ_TIMEOUT", "25")
    if "HERMES_MAX_TURNS" in os.environ and "HERMES_MAX_ITERATIONS" not in os.environ:
        os.environ["HERMES_MAX_ITERATIONS"] = os.environ["HERMES_MAX_TURNS"]
    _ensure_runtime_dirs()
    _sync_runtime_config()
    _sync_supermemory_config()


@dataclass(slots=True)
class RuntimeSettings:
    model: str
    max_turns: int
    max_tokens: Optional[int]
    provider: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    bearer_token: Optional[str]
    telegram_bot_token: Optional[str]
    telegram_webhook_secret: Optional[str]
    telegram_webhook_url: Optional[str]
    telegram_send_ack: bool
    feishu_app_id: Optional[str]
    feishu_app_secret: Optional[str]
    feishu_domain: Optional[str]
    feishu_connection_mode: Optional[str]
    feishu_verification_token: Optional[str]
    feishu_encrypt_key: Optional[str]
    feishu_bitable_app_token: Optional[str]
    feishu_bitable_table_id: Optional[str]
    feishu_model_registry_mirror_enabled: bool
    feishu_tool_capabilities: list[str]
    feishu_default_workspace: Optional[str]
    qq_app_id: Optional[str]
    qq_app_secret: Optional[str]
    nvidia_api_key: Optional[str]
    nvidia_base_url: Optional[str]
    enabled_toolsets: list[str]
    disabled_toolsets: list[str]

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        provider, base_url, api_key = _pick_runtime_api_config()
        return cls(
            model=os.getenv("DEFAULT_MODEL", "openrouter/free"),
            max_turns=int(os.getenv("HERMES_MAX_TURNS", "16")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "0")) or None,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            bearer_token=os.getenv("WEBHOOK_SECRET") or os.getenv("HERMES_WEBHOOK_BEARER_TOKEN"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET"),
            telegram_webhook_url=_desired_telegram_webhook_url(),
            telegram_send_ack=os.getenv("TELEGRAM_SEND_ACK", "false").lower() in {"1", "true", "yes"},
            feishu_app_id=os.getenv("FEISHU_APP_ID"),
            feishu_app_secret=os.getenv("FEISHU_APP_SECRET"),
            feishu_domain=os.getenv("FEISHU_DOMAIN") or "feishu",
            feishu_connection_mode=os.getenv("FEISHU_CONNECTION_MODE") or "webhook",
            feishu_verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN"),
            feishu_encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY"),
            feishu_bitable_app_token=os.getenv("FEISHU_BITABLE_APP_TOKEN"),
            feishu_bitable_table_id=os.getenv("FEISHU_BITABLE_TABLE_ID"),
            feishu_model_registry_mirror_enabled=_is_truthy(os.getenv("FEISHU_MODEL_REGISTRY_MIRROR_ENABLED"), default=False),
            feishu_tool_capabilities=_split_csv(os.getenv("HERMES_FEISHU_TOOL_CAPABILITIES")),
            feishu_default_workspace=os.getenv("HERMES_FEISHU_DEFAULT_WORKSPACE"),
            qq_app_id=os.getenv("QQ_APP_ID"),
            qq_app_secret=os.getenv("QQ_APP_SECRET"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY"),
            nvidia_base_url=os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL,
            enabled_toolsets=_split_csv(os.getenv("HERMES_ENABLED_TOOLSETS")),
            disabled_toolsets=_split_csv(os.getenv("HERMES_DISABLED_TOOLSETS")) or list(DEFAULT_DISABLED_TOOLSETS),
        )


def _serialize_settings_for_log(settings: RuntimeSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["api_key"] = _mask_secret(settings.api_key)
    payload["bearer_token"] = _mask_secret(settings.bearer_token)
    payload["telegram_bot_token"] = _mask_secret(settings.telegram_bot_token)
    payload["telegram_webhook_secret"] = _mask_secret(settings.telegram_webhook_secret)
    payload["feishu_app_id"] = _mask_secret(settings.feishu_app_id)
    payload["feishu_app_secret"] = _mask_secret(settings.feishu_app_secret)
    payload["feishu_verification_token"] = _mask_secret(settings.feishu_verification_token)
    payload["feishu_encrypt_key"] = _mask_secret(settings.feishu_encrypt_key)
    payload["feishu_bitable_app_token"] = _mask_secret(settings.feishu_bitable_app_token)
    payload["feishu_bitable_table_id"] = _mask_secret(settings.feishu_bitable_table_id)
    payload["qq_app_id"] = _mask_secret(settings.qq_app_id)
    payload["qq_app_secret"] = _mask_secret(settings.qq_app_secret)
    payload["nvidia_api_key"] = _mask_secret(settings.nvidia_api_key)
    return payload


@dataclass(slots=True)
class _TelegramGatewayRuntime:
    runner: Any
    adapter: Any


@dataclass(slots=True)
class _FeishuGatewayRuntime:
    runner: Any
    adapter: Any


@dataclass(slots=True)
class _QQGatewayRuntime:
    runner: Any
    adapter: Any


@dataclass(slots=True)
class _FeishuWebhookRequestAdapter:
    body: bytes
    headers: dict[str, Any]
    remote: str

    @property
    def content_length(self) -> int:
        return len(self.body)

    async def read(self) -> bytes:
        return self.body


def _normalize_public_https_url(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    normalized = raw.rstrip("/")
    return normalized or None


def _resolve_telegram_webhook_url(
    *,
    explicit_url: str | None = None,
    public_base_url: str | None = None,
) -> str | None:
    direct = _normalize_public_https_url(explicit_url)
    if direct:
        return direct

    base_url = _normalize_public_https_url(public_base_url)
    if not base_url:
        return None
    return f"{base_url}/telegram/webhook"


def _desired_telegram_webhook_url() -> str | None:
    return _resolve_telegram_webhook_url(
        explicit_url=os.getenv("TELEGRAM_WEBHOOK_URL"),
        public_base_url=os.getenv("HERMES_PUBLIC_BASE_URL") or os.getenv("PUBLIC_BASE_URL"),
    )


async def _fetch_telegram_webhook_info(bot_token: str) -> dict[str, Any]:
    import httpx

    url = f"https://api.telegram.org/bot{bot_token}/getWebhookInfo"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(f"Telegram getWebhookInfo failed: {payload}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"Telegram getWebhookInfo returned invalid result: {payload}")
    return result


async def _set_telegram_webhook(
    bot_token: str,
    webhook_url: str,
    *,
    webhook_secret: str | None = None,
    drop_pending_updates: bool = False,
) -> dict[str, Any]:
    import httpx

    payload: dict[str, Any] = {
        "url": webhook_url,
        "drop_pending_updates": bool(drop_pending_updates),
        "allowed_updates": [
            "message",
            "edited_message",
            "callback_query",
            "channel_post",
            "edited_channel_post",
        ],
    }
    if webhook_secret:
        payload["secret_token"] = webhook_secret

    url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError(f"Telegram setWebhook failed: {result}")
    return result


async def _get_telegram_webhook_status(
    settings: RuntimeSettings,
    *,
    ensure_registered: bool = False,
    drop_pending_updates: bool = False,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "configured": bool(settings.telegram_bot_token),
        "token_format_valid": _is_valid_telegram_bot_token_format(settings.telegram_bot_token),
        "expected_url": settings.telegram_webhook_url,
        "registered_url": "",
        "matches_expected": False,
        "pending_update_count": None,
        "last_error_message": "",
        "ip_address": "",
        "has_custom_certificate": False,
        "auto_sync_enabled": _is_truthy(os.getenv("TELEGRAM_WEBHOOK_AUTO_SYNC"), default=True),
    }
    if not settings.telegram_bot_token:
        status["reason"] = "telegram_bot_token_missing"
        return status

    if not status["token_format_valid"]:
        status["reason"] = "telegram_bot_token_invalid_format"
        return status

    if not settings.telegram_webhook_url:
        status["reason"] = "telegram_webhook_url_missing"
        return status

    if ensure_registered:
        await _set_telegram_webhook(
            settings.telegram_bot_token,
            settings.telegram_webhook_url,
            webhook_secret=settings.telegram_webhook_secret,
            drop_pending_updates=drop_pending_updates,
        )

    info = await _fetch_telegram_webhook_info(settings.telegram_bot_token)
    registered_url = str(info.get("url") or "").strip()
    status.update(
        {
            "registered_url": registered_url,
            "matches_expected": registered_url == settings.telegram_webhook_url,
            "pending_update_count": info.get("pending_update_count"),
            "last_error_message": str(info.get("last_error_message") or "").strip(),
            "ip_address": str(info.get("ip_address") or "").strip(),
            "has_custom_certificate": bool(info.get("has_custom_certificate")),
        }
    )
    return status


def _load_telegram_webhook_sync_state() -> dict[str, Any]:
    payload = _load_json_file(TELEGRAM_WEBHOOK_SYNC_STATE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_telegram_webhook_sync_state(payload: dict[str, Any]) -> None:
    _atomic_json_write(TELEGRAM_WEBHOOK_SYNC_STATE_PATH, payload)


def _telegram_webhook_retry_after_seconds(exc: Exception) -> int:
    retry_after = DEFAULT_TELEGRAM_WEBHOOK_SYNC_BACKOFF_SECONDS
    response = getattr(exc, "response", None)
    if response is None:
        return retry_after

    header_retry = response.headers.get("retry-after") if getattr(response, "headers", None) else None
    if header_retry:
        try:
            return max(int(header_retry), 1)
        except (TypeError, ValueError):
            pass

    try:
        payload = response.json()
    except Exception:
        payload = {}

    parameters = payload.get("parameters") if isinstance(payload, dict) else {}
    try:
        parsed_retry = int((parameters or {}).get("retry_after") or 0)
        if parsed_retry > 0:
            return parsed_retry
    except (TypeError, ValueError):
        pass
    return retry_after


async def _maybe_sync_telegram_webhook(
    settings: RuntimeSettings,
    *,
    drop_pending_updates: bool = False,
) -> dict[str, Any]:
    if not _is_truthy(os.getenv("TELEGRAM_WEBHOOK_AUTO_SYNC"), default=True):
        return await _get_telegram_webhook_status(settings, ensure_registered=False)

    status = await _get_telegram_webhook_status(settings, ensure_registered=False)
    if status.get("matches_expected"):
        _save_telegram_webhook_sync_state(
            {
                "last_attempt_at": int(time.time()),
                "next_retry_at": 0,
                "registered_url": status.get("registered_url") or "",
                "last_error": "",
            }
        )
        return status

    sync_state = _load_telegram_webhook_sync_state()
    now = int(time.time())
    next_retry_at = int(sync_state.get("next_retry_at") or 0)
    if next_retry_at and next_retry_at > now:
        status["sync_deferred"] = True
        status["retry_after_seconds"] = next_retry_at - now
        status["last_sync_error"] = str(sync_state.get("last_error") or "").strip()
        return status

    try:
        await _set_telegram_webhook(
            settings.telegram_bot_token,
            settings.telegram_webhook_url,
            webhook_secret=settings.telegram_webhook_secret,
            drop_pending_updates=drop_pending_updates,
        )
    except Exception as exc:
        retry_after_seconds = _telegram_webhook_retry_after_seconds(exc)
        sync_payload = {
            "last_attempt_at": now,
            "next_retry_at": now + retry_after_seconds,
            "registered_url": status.get("registered_url") or "",
            "last_error": str(exc),
        }
        _save_telegram_webhook_sync_state(sync_payload)
        status["last_sync_error"] = str(exc)
        if "429" in str(exc):
            status["sync_rate_limited"] = True
            status["retry_after_seconds"] = retry_after_seconds
            return status
        raise

    refreshed = await _get_telegram_webhook_status(settings, ensure_registered=False)
    _save_telegram_webhook_sync_state(
        {
            "last_attempt_at": now,
            "next_retry_at": 0,
            "registered_url": refreshed.get("registered_url") or "",
            "last_error": "",
        }
    )
    return refreshed


def _get_telegram_runtime_lock() -> asyncio.Lock:
    global _TELEGRAM_RUNTIME_LOCK
    if _TELEGRAM_RUNTIME_LOCK is None:
        _TELEGRAM_RUNTIME_LOCK = asyncio.Lock()
    return _TELEGRAM_RUNTIME_LOCK


def _get_feishu_runtime_lock() -> asyncio.Lock:
    global _FEISHU_RUNTIME_LOCK
    if _FEISHU_RUNTIME_LOCK is None:
        _FEISHU_RUNTIME_LOCK = asyncio.Lock()
    return _FEISHU_RUNTIME_LOCK


def _get_qq_runtime_lock() -> asyncio.Lock:
    global _QQ_RUNTIME_LOCK
    if _QQ_RUNTIME_LOCK is None:
        _QQ_RUNTIME_LOCK = asyncio.Lock()
    return _QQ_RUNTIME_LOCK


async def _initialize_telegram_gateway_runtime(settings: RuntimeSettings) -> _TelegramGatewayRuntime:
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.telegram import (
        Application,
        HTTPXRequest,
        TelegramAdapter,
        TelegramFallbackTransport,
        discover_fallback_ips,
    )
    from gateway.run import GatewayRunner

    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    runner = GatewayRunner()
    telegram_config = runner.config.platforms.get(Platform.TELEGRAM) or PlatformConfig()
    telegram_config.enabled = True
    telegram_config.token = settings.telegram_bot_token

    adapter = TelegramAdapter(telegram_config)
    builder = Application.builder().token(telegram_config.token)
    fallback_ips = adapter._fallback_ips()
    if not fallback_ips:
        try:
            fallback_ips = await discover_fallback_ips()
        except Exception as exc:
            logger.warning("Telegram fallback IP discovery failed: %s", exc)
            fallback_ips = []
    if fallback_ips:
        transport = TelegramFallbackTransport(fallback_ips)
        request = HTTPXRequest(httpx_kwargs={"transport": transport})
        get_updates_request = HTTPXRequest(httpx_kwargs={"transport": transport})
        builder = builder.request(request).get_updates_request(get_updates_request)

    adapter._app = builder.build()
    adapter._bot = adapter._app.bot
    await adapter._app.initialize()
    adapter._mark_connected()
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)

    runner.adapters[Platform.TELEGRAM] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)

    return _TelegramGatewayRuntime(runner=runner, adapter=adapter)


async def _get_telegram_gateway_runtime() -> _TelegramGatewayRuntime:
    global _TELEGRAM_RUNTIME
    if _TELEGRAM_RUNTIME is not None:
        return _TELEGRAM_RUNTIME

    async with _get_telegram_runtime_lock():
        if _TELEGRAM_RUNTIME is not None:
            return _TELEGRAM_RUNTIME
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        _TELEGRAM_RUNTIME = await _initialize_telegram_gateway_runtime(settings)
        return _TELEGRAM_RUNTIME


async def _initialize_feishu_gateway_runtime(settings: RuntimeSettings) -> _FeishuGatewayRuntime:
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.feishu import (
        FEISHU_DOMAIN,
        LARK_DOMAIN,
        FeishuAdapter,
        check_feishu_requirements,
    )
    from gateway.run import GatewayRunner

    if not settings.feishu_app_id or not settings.feishu_app_secret:
        raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are not configured")
    if not check_feishu_requirements():
        raise RuntimeError("Feishu dependencies are not installed")

    # Modal webhook workers are short-lived. Delayed batch flush tasks can be
    # cancelled when the request loop exits, which leads to dropped outbound
    # replies and executor-shutdown errors. Keep webhook processing synchronous
    # by default unless operators explicitly override these values.
    if not os.getenv("HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS", "").strip():
        os.environ["HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS"] = "0"
    if not os.getenv("HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS", "").strip():
        os.environ["HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS"] = "0"

    approved_feishu_users: list[str] = []
    try:
        from gateway.pairing import PairingStore

        store = PairingStore()
        approved_feishu_users = [
            str(item.get("user_id") or "").strip()
            for item in store.list_approved("feishu")
            if str(item.get("user_id") or "").strip()
        ]
    except Exception:
        logger.warning("[Feishu] Failed to read approved pairing users for allowlist merge", exc_info=True)

    env_allowed_users = _split_csv(os.getenv("FEISHU_ALLOWED_USERS"))
    merged_allowed_users = _dedupe_keep_order(env_allowed_users + approved_feishu_users)
    group_policy = str(os.getenv("FEISHU_GROUP_POLICY", "allowlist") or "allowlist").strip().lower() or "allowlist"
    group_require_mention = _is_truthy(os.getenv("FEISHU_GROUP_REQUIRE_MENTION"), default=False)

    logger.info(
        "[Feishu] Runtime policy=%s require_mention=%s env_allowlist=%d paired_allowlist=%d merged_allowlist=%d",
        group_policy,
        group_require_mention,
        len(env_allowed_users),
        len(approved_feishu_users),
        len(merged_allowed_users),
    )

    runner = GatewayRunner()
    feishu_config = runner.config.platforms.get(Platform.FEISHU) or PlatformConfig()
    feishu_config.enabled = True
    feishu_config.extra.update(
        {
            "app_id": settings.feishu_app_id,
            "app_secret": settings.feishu_app_secret,
            "domain": settings.feishu_domain or "feishu",
            "connection_mode": "webhook",
            "webhook_path": "/feishu/webhook",
            "group_policy": group_policy,
            "allowed_group_users": merged_allowed_users,
            "group_require_mention": group_require_mention,
        }
    )

    adapter = FeishuAdapter(feishu_config)
    adapter._loop = asyncio.get_running_loop()
    domain = FEISHU_DOMAIN if adapter._domain_name != "lark" else LARK_DOMAIN
    adapter._client = adapter._build_lark_client(domain)
    adapter._event_handler = adapter._build_event_handler()
    if adapter._event_handler is None:
        raise RuntimeError("failed to build Feishu event handler")
    await adapter._hydrate_bot_identity()
    adapter._mark_connected()
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)
    adapter.set_menu_action_handler(runner._handle_feishu_menu_action)

    runner.adapters[Platform.FEISHU] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)
    return _FeishuGatewayRuntime(runner=runner, adapter=adapter)


async def _get_feishu_gateway_runtime() -> _FeishuGatewayRuntime:
    global _FEISHU_RUNTIME
    if _FEISHU_RUNTIME is not None:
        return _FEISHU_RUNTIME

    async with _get_feishu_runtime_lock():
        if _FEISHU_RUNTIME is not None:
            return _FEISHU_RUNTIME
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        _FEISHU_RUNTIME = await _initialize_feishu_gateway_runtime(settings)
        return _FEISHU_RUNTIME


async def _initialize_qq_gateway_runtime(settings: RuntimeSettings) -> _QQGatewayRuntime:
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.qq import QQAdapter
    from gateway.run import GatewayRunner

    if not settings.qq_app_id or not settings.qq_app_secret:
        raise RuntimeError("QQ_APP_ID and QQ_APP_SECRET are not configured")

    runner = GatewayRunner()
    qq_config = runner.config.platforms.get(Platform.QQ) or PlatformConfig()
    qq_config.enabled = True
    qq_config.extra.update(
        {
            "app_id": settings.qq_app_id,
            "app_secret": settings.qq_app_secret,
            "connection_mode": "webhook",
            "verify_appid_header": True,
            "webhook_path": "/qq/webhook",
        }
    )

    adapter = QQAdapter(qq_config)
    adapter._mark_connected()
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)

    runner.adapters[Platform.QQ] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)
    return _QQGatewayRuntime(runner=runner, adapter=adapter)


async def _get_qq_gateway_runtime() -> _QQGatewayRuntime:
    global _QQ_RUNTIME
    if _QQ_RUNTIME is not None:
        return _QQ_RUNTIME

    async with _get_qq_runtime_lock():
        if _QQ_RUNTIME is not None:
            return _QQ_RUNTIME
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        _QQ_RUNTIME = await _initialize_qq_gateway_runtime(settings)
        return _QQ_RUNTIME


async def _dispatch_telegram_update(update_payload: dict[str, Any]) -> dict[str, Any]:
    runtime = await _get_telegram_gateway_runtime()
    adapter = runtime.adapter

    from telegram import Update
    from gateway.platforms.base import MessageType
    from gateway.session import build_session_key

    async def _process_event_sync(event: Any) -> None:
        session_key = build_session_key(
            event.source,
            group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        )

        # Normal gateway adapters schedule background work so they can support
        # interrupts while living inside a long-running process. In a webhook
        # request we need the full turn to stay inside the request lifecycle,
        # otherwise the container can return 200 OK before the background task
        # gets a chance to send the reply.
        if session_key in adapter._active_sessions:
            await adapter.handle_message(event)
            return

        adapter._active_sessions[session_key] = asyncio.Event()
        await adapter._process_message_background(event, session_key)

    update = Update.de_json(update_payload, adapter._bot)
    if update is None:
        return {"status": "ignored", "reason": "invalid_update"}

    if update.callback_query:
        await adapter._handle_callback_query(update, None)
        return {"status": "accepted", "kind": "callback_query"}

    message = (
        update.message
        or update.edited_message
        or update.channel_post
        or update.edited_channel_post
    )
    if not message:
        return {"status": "ignored", "reason": "unsupported_update"}

    if message.text:
        if message.text.lstrip().startswith("/"):
            await adapter._handle_command(update, None)
            return {"status": "accepted", "kind": "command"}

        # TelegramAdapter batches text via asyncio.create_task() so long user
        # messages split by the client can be recombined. That works in the
        # always-on gateway process, but a serverless webhook request may return
        # before the deferred flush task runs. In Modal webhook mode we dispatch
        # text synchronously so the request lifecycle covers the full agent turn.
        if not adapter._should_process_message(message):
            return {"status": "ignored", "reason": "message_filtered"}
        event = adapter._build_message_event(message, MessageType.TEXT)
        event.text = adapter._clean_bot_trigger_text(event.text)
        await _process_event_sync(event)
        return {"status": "accepted", "kind": "text"}

    if getattr(message, "location", None) or getattr(message, "venue", None):
        await adapter._handle_location_message(update, None)
        return {"status": "accepted", "kind": "location"}

    if any(
        getattr(message, attr, None)
        for attr in ("sticker", "photo", "video", "audio", "voice", "document")
    ):
        await adapter._handle_media_message(update, None)
        return {"status": "accepted", "kind": "media"}

    return {"status": "ignored", "reason": "no_supported_content"}


async def _dispatch_qq_update(
    payload: dict[str, Any],
    *,
    headers: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    runtime = await _get_qq_gateway_runtime()
    adapter = runtime.adapter
    return await adapter.handle_webhook_payload(payload, headers=headers or {})


def _to_fastapi_response_from_aiohttp(aiohttp_response: Any) -> Response:
    if Response is None:
        raise RuntimeError("FastAPI Response support is unavailable")

    status_code = int(getattr(aiohttp_response, "status", 200) or 200)
    raw_headers = dict(getattr(aiohttp_response, "headers", {}) or {})
    response_headers = {
        key: value
        for key, value in raw_headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "content-encoding", "connection"}
    }
    body = getattr(aiohttp_response, "body", None)
    if body is None:
        text = getattr(aiohttp_response, "text", "")
        body = text.encode("utf-8") if isinstance(text, str) else (text or b"")
    return Response(content=body, status_code=status_code, headers=response_headers)


async def _parse_feishu_webhook_request(request: Request) -> tuple[Any, dict[str, Any]]:
    runtime = await _get_feishu_gateway_runtime()
    adapter = runtime.adapter
    client = getattr(request, "client", None)
    remote_ip = getattr(client, "host", None) or "unknown"
    headers = dict(request.headers)

    content_type = str(headers.get("content-type", "") or "").split(";", 1)[0].strip().lower()
    if content_type and content_type != "application/json":
        adapter._record_webhook_anomaly(remote_ip, "415")
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 1024 * 1024:
                adapter._record_webhook_anomaly(remote_ip, "413")
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError:
            pass

    body = await request.body()
    if len(body) > 1024 * 1024:
        adapter._record_webhook_anomaly(remote_ip, "413")
        raise HTTPException(status_code=413, detail="Request body too large")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        adapter._record_webhook_anomaly(remote_ip, "400")
        raise HTTPException(status_code=400, detail="invalid json")

    if adapter._encrypt_key and not adapter._is_webhook_signature_valid(headers, body):
        adapter._record_webhook_anomaly(remote_ip, "401-sig")
        raise HTTPException(status_code=401, detail="Invalid signature")

    if payload.get("encrypt"):
        try:
            payload = adapter._decrypt_webhook_payload(str(payload.get("encrypt") or ""))
        except Exception:
            adapter._record_webhook_anomaly(remote_ip, "400-encrypted")
            logger.exception("Feishu encrypted webhook decrypt failed")
            raise HTTPException(status_code=400, detail="failed to decrypt webhook payload")

    if adapter._verification_token:
        header = payload.get("header") or {}
        incoming_token = str(header.get("token") or payload.get("token") or "")
        if not incoming_token or not hmac.compare_digest(incoming_token, adapter._verification_token):
            adapter._record_webhook_anomaly(remote_ip, "401-token")
            raise HTTPException(status_code=401, detail="Invalid verification token")

    adapter._clear_webhook_anomaly(remote_ip)
    return adapter, payload


def _extract_feishu_event_metadata(payload: dict[str, Any]) -> tuple[str, str]:
    header = payload.get("header") or {}
    event_id = str(header.get("event_id") or payload.get("event_id") or "").strip()
    event_type = str(header.get("event_type") or "").strip()
    return event_id, event_type


async def _await_feishu_pending_batches(adapter: Any) -> None:
    timeout_raw = os.getenv("HERMES_FEISHU_WEBHOOK_DRAIN_TIMEOUT_SECONDS", "6").strip()
    try:
        timeout_seconds = max(0.0, float(timeout_raw))
    except ValueError:
        timeout_seconds = 6.0
    if timeout_seconds <= 0:
        return

    pending: list[asyncio.Task] = []
    for attr in ("_pending_text_batch_tasks", "_pending_media_batch_tasks"):
        task_map = getattr(adapter, attr, None)
        if not isinstance(task_map, dict):
            continue
        for task in task_map.values():
            if isinstance(task, asyncio.Task) and not task.done():
                pending.append(task)
    if not pending:
        return

    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "[Feishu] pending batch drain timed out after %.1fs (pending=%d)",
            timeout_seconds,
            len(pending),
        )


async def _await_feishu_background_tasks(adapter: Any) -> None:
    timeout_raw = os.getenv("HERMES_FEISHU_WEBHOOK_BACKGROUND_TIMEOUT_SECONDS", "180").strip()
    try:
        timeout_seconds = max(0.0, float(timeout_raw))
    except ValueError:
        timeout_seconds = 180.0
    if timeout_seconds <= 0:
        return

    background = getattr(adapter, "_background_tasks", None)
    if not isinstance(background, set):
        return
    pending = [task for task in list(background) if isinstance(task, asyncio.Task) and not task.done()]
    if not pending:
        return

    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "[Feishu] background processing timeout after %.1fs (pending=%d)",
            timeout_seconds,
            len(pending),
        )


async def _dispatch_feishu_payload(
    payload: dict[str, Any],
    *,
    await_background_tasks: bool = True,
) -> None:
    runtime = await _get_feishu_gateway_runtime()
    adapter = runtime.adapter
    event_id, event_type = _extract_feishu_event_metadata(payload)
    data = adapter._namespace_from_mapping(payload)

    _append_feishu_trace("dispatch.start", payload)
    logger.warning(
        "[Feishu] dispatch start event_type=%s event_id=%s",
        event_type or "unknown",
        event_id or "none",
    )

    try:
        if event_type == "im.message.receive_v1":
            await adapter._handle_message_event_data(data)
            if await_background_tasks:
                await _await_feishu_pending_batches(adapter)
                await _await_feishu_background_tasks(adapter)
        elif event_type == "im.message.message_read_v1":
            adapter._on_message_read_event(data)
        elif event_type == "im.chat.member.bot.added_v1":
            adapter._on_bot_added_to_chat(data)
        elif event_type == "im.chat.member.bot.deleted_v1":
            adapter._on_bot_removed_from_chat(data)
        elif event_type in ("im.message.reaction.created_v1", "im.message.reaction.deleted_v1"):
            await adapter._handle_reaction_event(event_type, data)
            if await_background_tasks:
                await _await_feishu_background_tasks(adapter)
        elif event_type == "card.action.trigger":
            await adapter._handle_card_action_event(data)
            if await_background_tasks:
                await _await_feishu_background_tasks(adapter)
        elif event_type == "application.bot.menu_v6":
            await adapter._handle_bot_menu_event(data)
            if await_background_tasks:
                await _await_feishu_background_tasks(adapter)
        else:
            logger.warning("[Feishu] Ignoring unsupported event type in dispatcher: %s", event_type or "unknown")
    except Exception as exc:
        _append_feishu_trace("dispatch.error", payload, error=str(exc))
        raise
    else:
        _append_feishu_trace("dispatch.done", payload)
        logger.warning(
            "[Feishu] dispatch done event_type=%s event_id=%s",
            event_type or "unknown",
            event_id or "none",
        )


async def _dispatch_feishu_update(request: Request) -> Response:
    _adapter, payload = await _parse_feishu_webhook_request(request)
    if payload.get("type") == "url_verification":
        return JSONResponse({"challenge": payload.get("challenge", "")})
    await _dispatch_feishu_payload(payload)
    return JSONResponse({"code": 0, "msg": "ok"})


def _process_chat_queue_item(payload: Any) -> dict[str, Any]:
    _prepare_runtime_environment()
    if _is_truthy(os.getenv("HERMES_MODAL_CHAT_WORKER_RELOAD"), default=False):
        _sync_modal_volume(reload=True)

    envelope = payload if isinstance(payload, dict) else {}
    platform = str(envelope.get("platform") or "").strip().lower()
    partition = str(envelope.get("partition") or "").strip()
    raw_payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    metadata = envelope.get("metadata") if isinstance(envelope.get("metadata"), dict) else {}
    enqueued_at_ms = int(envelope.get("enqueued_at_ms") or 0)
    queue_latency_ms = max(0, int(time.time() * 1000) - enqueued_at_ms) if enqueued_at_ms else None

    if platform not in {"feishu", "telegram"}:
        return {"status": "skipped", "reason": "unsupported_platform", "platform": platform, "partition": partition}
    if not raw_payload:
        return {"status": "skipped", "reason": "missing_payload", "platform": platform, "partition": partition}
    if platform == "feishu":
        _append_feishu_trace(
            "worker.start",
            raw_payload,
            partition=partition,
            queue_latency_ms=queue_latency_ms,
        )
        logger.warning(
            "[Feishu] worker start event_type=%s event_id=%s partition=%s queue_latency_ms=%s",
            metadata.get("event_type") or "unknown",
            metadata.get("event_id") or "none",
            partition or "none",
            queue_latency_ms if queue_latency_ms is not None else "na",
        )
    else:
        logger.info(
            "Telegram worker start update_id=%s partition=%s queue_latency_ms=%s",
            metadata.get("event_id") or "none",
            partition or "none",
            queue_latency_ms if queue_latency_ms is not None else "na",
        )

    started_at = time.time()
    try:
        if platform == "feishu":
            asyncio.run(_dispatch_feishu_payload(raw_payload, await_background_tasks=True))
        else:
            asyncio.run(_dispatch_telegram_update(raw_payload))
    except Exception as exc:
        if platform == "feishu":
            _append_feishu_trace(
                "worker.error",
                raw_payload,
                partition=partition,
                queue_latency_ms=queue_latency_ms,
                error=str(exc),
            )
            logger.exception(
                "[Feishu] worker error event_type=%s event_id=%s partition=%s",
                metadata.get("event_type") or "unknown",
                metadata.get("event_id") or "none",
                partition or "none",
            )
        else:
            logger.exception(
                "Telegram worker error update_id=%s partition=%s",
                metadata.get("event_id") or "none",
                partition or "none",
            )
        return {
            "status": "error",
            "platform": platform,
            "partition": partition,
            "event_id": metadata.get("event_id"),
            "message_id": metadata.get("message_id"),
            "queue_latency_ms": queue_latency_ms,
            "error": str(exc),
        }

    elapsed_ms = int((time.time() - started_at) * 1000)
    if platform == "feishu":
        _append_feishu_trace(
            "worker.done",
            raw_payload,
            partition=partition,
            queue_latency_ms=queue_latency_ms,
            worker_elapsed_ms=elapsed_ms,
        )
        logger.warning(
            "[Feishu] worker done event_type=%s event_id=%s partition=%s worker_elapsed_ms=%s",
            metadata.get("event_type") or "unknown",
            metadata.get("event_id") or "none",
            partition or "none",
            elapsed_ms,
        )
    else:
        logger.info(
            "Telegram worker done update_id=%s partition=%s worker_elapsed_ms=%s",
            metadata.get("event_id") or "none",
            partition or "none",
            elapsed_ms,
        )
    return {
        "status": "ok",
        "platform": platform,
        "partition": partition,
        "event_id": metadata.get("event_id"),
        "message_id": metadata.get("message_id"),
        "queue_latency_ms": queue_latency_ms,
        "worker_elapsed_ms": elapsed_ms,
    }


def _process_chat_queue_impl(*, platform: str, partition: str, max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE) -> dict[str, Any]:
    _prepare_runtime_environment()

    normalized_platform = str(platform or "").strip().lower()
    normalized_partition = str(partition or "").strip()
    if not normalized_platform or not normalized_partition:
        return {"status": "skipped", "reason": "missing_partition", "platform": normalized_platform, "partition": normalized_partition}

    claimed, claim_token = _claim_chat_partition(normalized_partition, platform=normalized_platform)
    if not claimed:
        return {
            "status": "skipped",
            "reason": "already_claimed",
            "platform": normalized_platform,
            "partition": normalized_partition,
        }

    queue = _get_chat_queue()
    processed: list[dict[str, Any]] = []
    try:
        while True:
            items = queue.get_many(max(max_items, 1), block=False, partition=normalized_partition)
            if not items:
                break
            for item in items:
                processed.append(_process_chat_queue_item(item))
    finally:
        _sync_modal_volume(commit=True)
        _release_chat_partition_claim(normalized_partition, claim_token=claim_token)

    return {
        "status": "ok",
        "platform": normalized_platform,
        "partition": normalized_partition,
        "processed_count": len(processed),
        "results": processed,
        "queue_depth": _safe_chat_queue_depth(),
    }


def _run_agent_task_impl(
    task_input: str,
    *,
    session_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    _prepare_runtime_environment()
    settings = RuntimeSettings.from_env()
    session_key = session_key or f"task:{uuid.uuid4()}"
    session_state = _load_session_state(session_key)
    route_lease = session_state.get("route_lease") if isinstance(session_state.get("route_lease"), dict) else None
    route_debug = dict(session_state.get("route_debug") or {})
    route_metrics = dict(session_state.get("route_metrics") or {})
    explicit_model_requested = bool(str(model_name or "").strip())

    from run_agent import AIAgent

    def _execute_once(route: dict[str, Any], *, route_selection: str) -> tuple[Any, dict[str, Any], str, int]:
        resolved_model = _resolve_runtime_model_name(route["model"], route.get("provider"))
        agent = AIAgent(
            model=resolved_model,
            provider=route.get("provider"),
            base_url=route.get("base_url"),
            api_key=route.get("api_key"),
            max_iterations=settings.max_turns,
            enabled_toolsets=settings.enabled_toolsets or None,
            disabled_toolsets=settings.disabled_toolsets or None,
            quiet_mode=True,
            max_tokens=max_tokens or settings.max_tokens,
            platform="modal",
            persist_session=False,
            session_id=session_state["session_id"],
            trace_session_key=session_key,
            trace_metadata={
                "channel_type": "invoke",
                "route_selection": route_selection,
            },
        )
        started_at = time.time()
        result = agent.run_conversation(
            task_input,
            conversation_history=session_state["messages"],
            persist_user_message=task_input,
        )
        elapsed_ms = int((time.time() - started_at) * 1000)
        return agent, result, resolved_model, elapsed_ms

    if explicit_model_requested:
        primary_route = _resolve_primary_route(settings, model_name)
        route_selection = "explicit_override"
    elif _is_route_lease_active(settings, route_lease):
        primary_route = _hydrate_route_from_lease(settings, route_lease) or _resolve_primary_route(settings, model_name)
        route_selection = "sticky_hit"
    else:
        primary_route = _resolve_primary_route(settings, model_name)
        route_selection = "fresh_select"

    agent, result, resolved_model, elapsed_ms = _execute_once(primary_route, route_selection=route_selection)
    retried_after_refresh = False
    refreshed_route = None
    refresh_reason = _determine_route_refresh_reason(result)

    refreshed_route = _select_retry_route_for_result(primary_route, result)
    if refreshed_route is not None:
        retried_after_refresh = True
        route_selection = "refreshed_after_failure"
        agent, result, resolved_model, elapsed_ms = _execute_once(refreshed_route, route_selection=route_selection)

    active_route = refreshed_route or primary_route
    session_id = agent.session_id or session_state["session_id"]
    messages = result.get("messages") or []
    provider_usage = dict(result.get("provider_usage") or {})
    response_model = str(provider_usage.get("response_model") or "").strip()
    active_route_for_lease = dict(active_route)
    if response_model:
        active_route_for_lease["model"] = response_model
        resolved_model = response_model
    if result.get("error"):
        route_debug["last_error"] = str(result.get("error") or "")
    else:
        route_debug["last_error"] = ""
    route_debug["last_route_selection"] = route_selection
    route_debug["last_failure_reason"] = refresh_reason if retried_after_refresh or not result.get("completed", True) else None
    route_debug["last_provider"] = result.get("provider") or active_route_for_lease.get("provider")
    route_debug["last_model"] = resolved_model
    route_debug["updated_at"] = int(time.time())
    route_metrics = _increment_route_metric(route_metrics, route_selection)

    if result.get("error"):
        route_lease = _expire_route_lease(
            route_lease if route_selection == "sticky_hit" else _build_route_lease(
                active_route_for_lease,
                selection_reason=route_selection,
            ),
            error_text=str(result.get("error") or ""),
            failure_reason=refresh_reason,
        )
    else:
        route_lease = _refresh_route_lease(
            route_lease if route_selection == "sticky_hit" else None,
            active_route_for_lease,
            selection_reason=(
                route_selection
                if route_selection != "sticky_hit"
                else (route_lease or {}).get("selection_reason")
            ),
        )

    _save_session_state(
        session_key,
        session_id,
        messages,
        route_lease=route_lease,
        route_debug=route_debug,
        route_metrics=route_metrics,
    )

    tool_names = _extract_tool_names(messages)
    final_response = result.get("final_response")

    payload = {
        "status": "success" if result.get("completed", True) and not result.get("interrupted") else "partial",
        "session_key": session_key,
        "session_id": session_id,
        "model": resolved_model,
        "provider": result.get("provider") or active_route_for_lease.get("provider") or settings.provider,
        "base_url": result.get("base_url") or active_route_for_lease.get("base_url") or settings.base_url,
        "input": task_input,
        "output": final_response,
        "completed": result.get("completed", True),
        "interrupted": result.get("interrupted", False),
        "api_calls": result.get("api_calls", 0),
        "tool_summary": tool_names,
        "elapsed_ms": elapsed_ms,
        "route_selection": route_selection,
        "route_lease_expires_at": (route_lease or {}).get("lease_expires_at"),
        "token_usage": {
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
        },
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "provider_usage": result.get("provider_usage") or {},
        "provider_usage_totals": result.get("provider_usage_totals") or {},
        "last_reasoning": result.get("last_reasoning"),
    }
    if retried_after_refresh:
        payload["retried_after_model_refresh"] = True
        payload["refreshed_model"] = active_route.get("model")
        payload["refreshed_provider"] = active_route.get("provider")
    if result.get("error"):
        payload["status"] = "error"
        payload["error"] = result["error"]
    logger.info(
        "[ModalInvoke] session_key=%s session_id=%s provider=%s model=%s route_selection=%s cache_key=%s retried_after_refresh=%s",
        session_key,
        session_id,
        payload.get("provider"),
        payload.get("model"),
        route_selection,
        session_id,
        retried_after_refresh,
    )
    return payload


def _validate_tavily_integration_impl() -> dict[str, Any]:
    _prepare_runtime_environment()

    from tools import web_tools

    backend = web_tools._get_backend()
    backend_available = web_tools._is_backend_available(backend)

    search_payload = _safe_json_loads(
        web_tools.web_search_tool("Tavily AI official website", limit=3),
        {"success": False, "error": "invalid_json"},
    )
    extract_payload = _safe_json_loads(
        asyncio.run(
            web_tools.web_extract_tool(
                ["https://tavily.com/"],
                use_llm_processing=False,
            )
        ),
        {"success": False, "error": "invalid_json"},
    )
    crawl_payload = _safe_json_loads(
        asyncio.run(
            web_tools.web_crawl_tool(
                "https://tavily.com/",
                "Find basic site information",
                use_llm_processing=False,
            )
        ),
        {"success": False, "error": "invalid_json"},
    )

    search_results = ((search_payload.get("data") or {}).get("web") or []) if isinstance(search_payload, dict) else []
    extract_results = (extract_payload.get("results") or []) if isinstance(extract_payload, dict) else []
    crawl_results = (crawl_payload.get("results") or []) if isinstance(crawl_payload, dict) else []

    return {
        "status": "ok",
        "integration": "tavily",
        "backend": backend,
        "backend_available": backend_available,
        "env_configured": bool(os.getenv("TAVILY_API_KEY", "").strip()),
        "search": {
            "success": bool(search_payload.get("success")) if isinstance(search_payload, dict) else False,
            "result_count": len(search_results),
            "top_result": search_results[0] if search_results else None,
            "error": search_payload.get("error") if isinstance(search_payload, dict) else "invalid_response",
        },
        "extract": {
            "success": bool(extract_results),
            "result_count": len(extract_results),
            "top_result": extract_results[0] if extract_results else None,
            "error": extract_payload.get("error") if isinstance(extract_payload, dict) else "invalid_response",
        },
        "crawl": {
            "success": bool(crawl_results),
            "result_count": len(crawl_results),
            "top_result": crawl_results[0] if crawl_results else None,
            "error": crawl_payload.get("error") if isinstance(crawl_payload, dict) else "invalid_response",
        },
    }


def _encrypt_feishu_payload(encrypt_key: str, payload: dict[str, Any]) -> str:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    aes_key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    iv = aes_key[:16]
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(encrypted).decode("utf-8")


def _decrypt_feishu_payload(encrypt_key: str, encrypted_payload: str) -> dict[str, Any]:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    encrypted_bytes = base64.b64decode(encrypted_payload)
    aes_key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    iv = aes_key[:16]
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(encrypted_bytes) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()
    payload = json.loads(plaintext.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("decrypted Feishu payload must be a JSON object")
    return payload


def _is_feishu_signature_valid(headers: dict[str, Any], body_bytes: bytes, encrypt_key: str) -> bool:
    timestamp = str(headers.get("x-lark-request-timestamp", "") or "")
    nonce = str(headers.get("x-lark-request-nonce", "") or "")
    signature = str(headers.get("x-lark-signature", "") or "")
    if not timestamp or not nonce or not signature:
        return False
    body_str = body_bytes.decode("utf-8", errors="replace")
    computed = hashlib.sha256(f"{timestamp}{nonce}{encrypt_key}{body_str}".encode("utf-8")).hexdigest()
    return hmac.compare_digest(computed, signature)


async def _try_handle_feishu_verification_fast(request: Request, settings: RuntimeSettings) -> Response | None:
    if Response is None or JSONResponse is None:
        return None

    body = await request.body()
    if not body:
        return None

    payload = _safe_json_loads(body.decode("utf-8", errors="replace"), None)
    if not isinstance(payload, dict):
        return None

    if payload.get("type") == "url_verification":
        return JSONResponse({"challenge": payload.get("challenge", "")})

    encrypted_payload = str(payload.get("encrypt") or "").strip()
    if not encrypted_payload or not settings.feishu_encrypt_key:
        return None
    if not _is_feishu_signature_valid(dict(request.headers), body, settings.feishu_encrypt_key):
        return None

    try:
        inner_payload = _decrypt_feishu_payload(settings.feishu_encrypt_key, encrypted_payload)
    except Exception:
        logger.debug("Fast Feishu verification decrypt failed", exc_info=True)
        return None

    if inner_payload.get("type") != "url_verification":
        return None
    expected_token = str(settings.feishu_verification_token or "").strip()
    provided_token = str((inner_payload.get("header") or {}).get("token") or inner_payload.get("token") or "").strip()
    if expected_token and (not provided_token or not hmac.compare_digest(provided_token, expected_token)):
        return None
    return JSONResponse({"challenge": inner_payload.get("challenge", "")})


def _validate_feishu_webhook_impl() -> dict[str, Any]:
    import httpx

    _prepare_runtime_environment()
    settings = RuntimeSettings.from_env()
    public_base = _normalize_public_https_url(os.getenv("HERMES_PUBLIC_BASE_URL") or os.getenv("PUBLIC_BASE_URL"))
    webhook_url = (
        f"{public_base}/feishu/webhook"
        if public_base
        else "https://isuyee88--hermes-agent-web-app.modal.run/feishu/webhook"
    )
    verification_token = str(settings.feishu_verification_token or "").strip()
    encrypt_key = str(settings.feishu_encrypt_key or "").strip()

    if not settings.feishu_app_id or not settings.feishu_app_secret:
        return {"status": "error", "message": "Feishu app credentials are not configured"}
    if not verification_token:
        return {"status": "error", "message": "FEISHU_VERIFICATION_TOKEN is not configured"}
    if not encrypt_key:
        return {"status": "error", "message": "FEISHU_ENCRYPT_KEY is not configured"}

    inner_payload = {
        "type": "url_verification",
        "challenge": "feishu-encrypted-selftest-ok",
        "token": verification_token,
    }
    outer_payload = {
        "encrypt": _encrypt_feishu_payload(encrypt_key, inner_payload),
    }
    body = json.dumps(outer_payload, ensure_ascii=False)
    timestamp = str(int(time.time()))
    nonce = "hermes-feishu-selftest"
    signature = hashlib.sha256(f"{timestamp}{nonce}{encrypt_key}{body}".encode("utf-8")).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "x-lark-request-timestamp": timestamp,
        "x-lark-request-nonce": nonce,
        "x-lark-signature": signature,
    }

    with httpx.Client(timeout=20) as client:
        response = client.post(webhook_url, content=body.encode("utf-8"), headers=headers)

    return {
        "status": "ok" if response.status_code == 200 else "error",
        "webhook_url": webhook_url,
        "status_code": response.status_code,
        "response": _safe_json_loads(response.text, response.text),
        "verification_token_configured": True,
        "encrypt_key_configured": True,
    }


def _approve_pairing_impl(platform: str, code: str) -> dict[str, Any]:
    _prepare_runtime_environment()

    from gateway.pairing import PairingStore

    normalized_platform = str(platform or "").strip().lower()
    normalized_code = str(code or "").strip().upper()
    if not normalized_platform or not normalized_code:
        return {"status": "error", "message": "platform and code are required"}

    store = PairingStore()
    pending_before = store.list_pending(normalized_platform)
    result = store.approve_code(normalized_platform, normalized_code)
    approved_after = store.list_approved(normalized_platform)
    pending_after = store.list_pending(normalized_platform)

    if not result:
        return {
            "status": "not_found",
            "platform": normalized_platform,
            "code": normalized_code,
            "pending_before": pending_before,
            "pending_after": pending_after,
            "approved_after": approved_after,
        }

    return {
        "status": "approved",
        "platform": normalized_platform,
        "code": normalized_code,
        "approved_user": result,
        "pending_before": pending_before,
        "pending_after": pending_after,
        "approved_after": approved_after,
    }


def _get_cron_queue():
    if modal is None:
        raise RuntimeError("Modal is required for the cron queue")
    global CRON_QUEUE
    if CRON_QUEUE is None:
        CRON_QUEUE = modal.Queue.from_name(DEFAULT_CRON_QUEUE_NAME, create_if_missing=True)
    return CRON_QUEUE


def _get_chat_queue():
    if modal is None:
        raise RuntimeError("Modal is required for the chat queue")
    global CHAT_QUEUE
    if CHAT_QUEUE is None:
        CHAT_QUEUE = modal.Queue.from_name(DEFAULT_CHAT_QUEUE_NAME, create_if_missing=True)
    return CHAT_QUEUE


def _sync_modal_volume(*, reload: bool = False, commit: bool = False) -> None:
    if MODAL_VOLUME is None:
        return
    try:
        if reload:
            MODAL_VOLUME.reload()
        if commit:
            MODAL_VOLUME.commit()
    except Exception as exc:
        logger.warning("Modal volume sync failed (reload=%s commit=%s): %s", reload, commit, exc)


def _should_reload_modal_volume_for_claims(kind: str) -> bool:
    env_name = f"HERMES_MODAL_{str(kind or '').strip().upper()}_CLAIMS_RELOAD"
    return _is_truthy(os.getenv(env_name), default=False)


def _safe_cron_queue_depth() -> int | None:
    try:
        return int(_get_cron_queue().len())
    except Exception as exc:
        logger.warning("Unable to read Modal cron queue depth: %s", exc)
        return None


def _safe_chat_queue_depth() -> int | None:
    try:
        return int(_get_chat_queue().len())
    except Exception as exc:
        logger.warning("Unable to read Modal chat queue depth: %s", exc)
        return None


async def _safe_chat_queue_depth_async() -> int | None:
    queue = _get_chat_queue()
    try:
        if hasattr(queue, "len") and hasattr(queue.len, "aio"):
            return int(await queue.len.aio())  # type: ignore[union-attr]
        return int(queue.len())
    except Exception as exc:
        logger.warning("Unable to read Modal chat queue depth: %s", exc)
        return None


def _load_chat_queue_claims() -> dict[str, Any]:
    payload = _load_json_file(CHAT_QUEUE_CLAIMS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_chat_queue_claims(payload: dict[str, Any]) -> None:
    _atomic_json_write(CHAT_QUEUE_CLAIMS_PATH, payload)


def _prune_chat_queue_claims(
    claims: dict[str, Any],
    *,
    ttl_seconds: int = DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
) -> dict[str, Any]:
    now = int(time.time())
    pruned: dict[str, Any] = {}
    for partition_key, claim in claims.items():
        if not isinstance(claim, dict):
            continue
        claimed_at = int(claim.get("claimed_at") or 0)
        if claimed_at and now - claimed_at < ttl_seconds:
            pruned[partition_key] = claim
    return pruned


def _claim_chat_partition(
    partition_key: str,
    *,
    platform: str,
    claim_token: str | None = None,
    ttl_seconds: int = DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
) -> tuple[bool, str]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return False, ""

    now = int(time.time())
    token = str(claim_token or f"{platform}:{normalized}:{now}:{uuid.uuid4().hex[:8]}").strip()
    with _CHAT_QUEUE_LOCK:
        if _should_reload_modal_volume_for_claims("chat"):
            _sync_modal_volume(reload=True)
        claims = _prune_chat_queue_claims(_load_chat_queue_claims(), ttl_seconds=ttl_seconds)
        existing = claims.get(normalized) or {}
        if existing:
            return False, str(existing.get("claim_token") or "")
        claims[normalized] = {
            "claim_token": token,
            "claimed_at": now,
            "platform": platform,
        }
        _save_chat_queue_claims(claims)
        _sync_modal_volume(commit=True)
    return True, token


def _release_chat_partition_claim(partition_key: str, *, claim_token: str | None = None) -> None:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return

    with _CHAT_QUEUE_LOCK:
        if _should_reload_modal_volume_for_claims("chat"):
            _sync_modal_volume(reload=True)
        claims = _prune_chat_queue_claims(_load_chat_queue_claims())
        existing = claims.get(normalized)
        if not existing:
            return
        if claim_token and existing.get("claim_token") != claim_token:
            return
        claims.pop(normalized, None)
        _save_chat_queue_claims(claims)
        _sync_modal_volume(commit=True)


def _extract_feishu_queue_context(payload: dict[str, Any]) -> dict[str, str]:
    trace = _extract_feishu_trace_context(payload)
    event = payload.get("event") or {}
    message = event.get("message") or {}
    sender = event.get("sender") or {}
    sender_id = sender.get("sender_id") or {}
    operator = event.get("operator") or {}
    operator_id = operator.get("operator_id") or {}
    context = event.get("context") or {}
    chat = event.get("chat") or {}
    chat_id = str(
        trace.get("chat_id")
        or message.get("chat_id")
        or chat.get("chat_id")
        or context.get("open_chat_id")
        or event.get("chat_id")
        or ""
    ).strip()
    actor_id = str(
        trace.get("sender_open_id")
        or trace.get("sender_user_id")
        or sender_id.get("open_id")
        or sender_id.get("user_id")
        or operator_id.get("open_id")
        or operator_id.get("user_id")
        or operator.get("open_id")
        or operator.get("user_id")
        or "unknown"
    ).strip()
    partition = f"feishu:{chat_id or actor_id or trace.get('event_id') or 'unknown'}"
    return {
        "platform": "feishu",
        "partition": partition,
        "chat_id": chat_id,
        "message_id": str(trace.get("message_id") or "").strip(),
        "event_id": str(trace.get("event_id") or "").strip(),
        "event_type": str(trace.get("event_type") or "").strip(),
        "actor_id": actor_id,
    }


def _extract_telegram_queue_context(update: dict[str, Any]) -> dict[str, str]:
    update_id = str(update.get("update_id") or "").strip()
    message = (
        update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
        or {}
    )
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = str(chat.get("id") or "").strip()
    user_id = str(sender.get("id") or "").strip()
    partition = f"telegram:{chat_id or user_id or update_id or 'unknown'}"
    return {
        "platform": "telegram",
        "partition": partition,
        "chat_id": chat_id,
        "message_id": str(message.get("message_id") or "").strip(),
        "event_id": update_id,
        "event_type": "telegram.update",
        "actor_id": user_id,
    }


def _enqueue_chat_event(*, platform: str, partition: str, payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    queue = _get_chat_queue()
    enqueued_at = int(time.time() * 1000)
    envelope = {
        "platform": platform,
        "partition": partition,
        "payload": payload,
        "metadata": metadata,
        "enqueued_at_ms": enqueued_at,
    }
    queue.put(envelope, partition=partition, partition_ttl=max(DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS, 3600))
    return {
        "status": "enqueued",
        "platform": platform,
        "partition": partition,
        "enqueued_at_ms": enqueued_at,
        "queue_depth": _safe_chat_queue_depth(),
        **metadata,
    }


async def _enqueue_chat_event_async(
    *,
    platform: str,
    partition: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        queue = _get_chat_queue()
    except Exception:
        return _enqueue_chat_event(
            platform=platform,
            partition=partition,
            payload=payload,
            metadata=metadata,
        )
    enqueued_at = int(time.time() * 1000)
    envelope = {
        "platform": platform,
        "partition": partition,
        "payload": payload,
        "metadata": metadata,
        "enqueued_at_ms": enqueued_at,
    }
    partition_ttl = max(DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS, 3600)
    if hasattr(queue, "put") and hasattr(queue.put, "aio"):
        await queue.put.aio(envelope, partition=partition, partition_ttl=partition_ttl)  # type: ignore[union-attr]
    else:
        queue.put(envelope, partition=partition, partition_ttl=partition_ttl)
    return {
        "status": "enqueued",
        "platform": platform,
        "partition": partition,
        "enqueued_at_ms": enqueued_at,
        "queue_depth": await _safe_chat_queue_depth_async(),
        **metadata,
    }


def _build_model_routing_debug_state(*, force_refresh: bool = False, allow_network: bool = False) -> dict[str, Any]:
    _prepare_runtime_environment()
    state = _load_routing_state()
    if force_refresh or (allow_network and not state):
        state = _refresh_free_model_routes(force=force_refresh)

    preferred_provider = str(
        os.getenv("HERMES_FREE_MODEL_PRIMARY_PROVIDER")
        or ("openrouter" if os.getenv("OPENROUTER_API_KEY", "").strip() else "nvidia")
    ).strip().lower() or None
    candidate_routes = _candidate_routes_from_state(state, preferred_provider=preferred_provider)
    primary_route = candidate_routes[0] if candidate_routes else None
    fallback_routes = []
    seen_fallbacks: set[tuple[str, str, str]] = set()
    for route in candidate_routes[1:]:
        key = (
            str(route.get("provider") or "").strip().lower(),
            str(route.get("model") or "").strip(),
            str(route.get("base_url") or "").strip(),
        )
        if key in seen_fallbacks:
            continue
        seen_fallbacks.add(key)
        fallback_routes.append(route)
        if len(fallback_routes) >= 4:
            break
    recent_sessions = _build_recent_session_route_summaries(limit=20)
    return {
        "configured_default_model": os.getenv("DEFAULT_MODEL", "openrouter/free"),
        "free_model_primary_provider": str(os.getenv("HERMES_FREE_MODEL_PRIMARY_PROVIDER") or "").strip().lower() or None,
        "cheap_routing_enabled": False,
        "active_primary_route": {
            "provider": primary_route.get("provider"),
            "model": primary_route.get("model"),
            "base_url": primary_route.get("base_url"),
        } if primary_route else None,
        "fallback_candidates": [
            {
                "provider": route.get("provider"),
                "model": route.get("model"),
                "base_url": route.get("base_url"),
            }
            for route in fallback_routes
        ],
        "session_route_metrics": _aggregate_session_route_metrics(limit=200),
        "recent_session_routes": recent_sessions,
        "routing_state": state,
    }


def _build_feishu_capabilities_debug_state(*, probe: bool = False) -> dict[str, Any]:
    _prepare_runtime_environment()
    try:
        from tools.feishu_api import get_feishu_capability_snapshot

        return get_feishu_capability_snapshot(probe=probe)
    except Exception as exc:
        return {
            "configured": False,
            "error": str(exc),
        }


def _build_feishu_model_registry_debug_state(*, force_refresh: bool = False) -> dict[str, Any]:
    _prepare_runtime_environment()
    try:
        from tools.feishu_api import load_feishu_model_registry

        payload = load_feishu_model_registry(force_refresh=force_refresh)
        return {
            "status": "ok",
            **payload,
            "entry_count": len(payload.get("entries") or []),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def _debug_session_route_state(session_key: str) -> dict[str, Any]:
    _prepare_runtime_environment()
    normalized = str(session_key or "").strip()
    if not normalized:
        return {
            "status": "error",
            "message": "session_key is required",
        }
    payload = _load_session_state(normalized)
    lease = payload.get("route_lease")
    now = int(time.time())
    return {
        "status": "ok",
        "session_key": normalized,
        "session_id": payload.get("session_id"),
        "updated_at": payload.get("updated_at"),
        "route_lease": lease,
        "route_debug": payload.get("route_debug") or {},
        "route_metrics": payload.get("route_metrics") or {},
        "lease_active": bool(isinstance(lease, dict) and int((lease or {}).get("lease_expires_at") or 0) > now),
        "lease_ttl_remaining_seconds": max(int((lease or {}).get("lease_expires_at") or 0) - now, 0),
    }


def _debug_gateway_session_state(session_key: str) -> dict[str, Any]:
    _prepare_runtime_environment()

    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home

    normalized = str(session_key or "").strip()
    runner = GatewayRunner()
    try:
        runner.session_store._ensure_loaded()
    except Exception:
        logger.debug("Failed to eagerly load gateway session store for debug", exc_info=True)
    entry = runner.session_store._entries.get(normalized)
    sessions_dir = Path(runner.session_store.sessions_dir)
    sessions_file = sessions_dir / "sessions.json"
    return {
        "status": "ok",
        "session_key": normalized,
        "env_hermes_home": os.getenv("HERMES_HOME", ""),
        "resolved_hermes_home": str(get_hermes_home()),
        "gateway_sessions_dir": str(sessions_dir),
        "sessions_file_exists": sessions_file.exists(),
        "session_count": len(runner.session_store._entries),
        "entry": entry.to_dict() if entry else None,
        "known_session_keys_sample": sorted(list(runner.session_store._entries.keys()))[:20],
    }


def _load_cron_queue_claims() -> dict[str, Any]:
    payload = _load_json_file(CRON_QUEUE_CLAIMS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_cron_queue_claims(payload: dict[str, Any]) -> None:
    _atomic_json_write(CRON_QUEUE_CLAIMS_PATH, payload)


def _prune_cron_queue_claims(
    claims: dict[str, Any],
    *,
    ttl_seconds: int = DEFAULT_CRON_QUEUE_CLAIM_TTL_SECONDS,
) -> dict[str, Any]:
    now = int(time.time())
    pruned: dict[str, Any] = {}
    for job_id, claim in claims.items():
        if not isinstance(claim, dict):
            continue
        claimed_at = int(claim.get("claimed_at") or 0)
        if claimed_at and now - claimed_at < ttl_seconds:
            pruned[job_id] = claim
    return pruned


def _cleanup_orphan_cron_queue_claims(live_job_ids: set[str]) -> dict[str, Any]:
    with _CRON_QUEUE_LOCK:
        if _should_reload_modal_volume_for_claims("cron"):
            _sync_modal_volume(reload=True)
        claims = _prune_cron_queue_claims(_load_cron_queue_claims())
        filtered = {job_id: claim for job_id, claim in claims.items() if job_id in live_job_ids}
        if filtered != claims:
            _save_cron_queue_claims(filtered)
            _sync_modal_volume(commit=True)
        return filtered


def _make_cron_claim_token(job: dict[str, Any]) -> str:
    return f"{job.get('id', '')}:{job.get('next_run_at', '')}"


def _claim_due_cron_job(
    job: dict[str, Any],
    *,
    ttl_seconds: int = DEFAULT_CRON_QUEUE_CLAIM_TTL_SECONDS,
) -> tuple[bool, str]:
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        return False, ""

    claim_token = _make_cron_claim_token(job)
    now = int(time.time())

    with _CRON_QUEUE_LOCK:
        if _should_reload_modal_volume_for_claims("cron"):
            _sync_modal_volume(reload=True)
        claims = _prune_cron_queue_claims(_load_cron_queue_claims(), ttl_seconds=ttl_seconds)
        existing = claims.get(job_id) or {}
        if existing.get("claim_token") == claim_token:
            return False, claim_token

        claims[job_id] = {
            "claim_token": claim_token,
            "claimed_at": now,
            "next_run_at": job.get("next_run_at"),
            "job_name": job.get("name"),
        }
        _save_cron_queue_claims(claims)
        _sync_modal_volume(commit=True)
    return True, claim_token


def _release_cron_job_claim(job_id: str, *, claim_token: str | None = None) -> None:
    normalized = str(job_id or "").strip()
    if not normalized:
        return

    with _CRON_QUEUE_LOCK:
        if _should_reload_modal_volume_for_claims("cron"):
            _sync_modal_volume(reload=True)
        claims = _prune_cron_queue_claims(_load_cron_queue_claims())
        existing = claims.get(normalized)
        if not existing:
            return
        if claim_token and existing.get("claim_token") != claim_token:
            return
        claims.pop(normalized, None)
        _save_cron_queue_claims(claims)
        _sync_modal_volume(commit=True)


def _cron_status_impl(limit: int = 10) -> dict[str, Any]:
    _prepare_runtime_environment()
    if _should_reload_modal_volume_for_claims("cron"):
        _sync_modal_volume(reload=True)

    from cron.jobs import get_due_jobs, list_jobs

    jobs = list_jobs(include_disabled=True)
    live_job_ids = {str(job.get("id") or "").strip() for job in jobs if job.get("id")}
    due_jobs = get_due_jobs()
    queue_depth = _safe_cron_queue_depth()
    claims = _cleanup_orphan_cron_queue_claims(live_job_ids)
    summarized_jobs = []
    for job in jobs[: max(limit, 0)]:
        summarized_jobs.append(
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "state": job.get("state"),
                "enabled": job.get("enabled", True),
                "deliver": job.get("deliver"),
                "schedule": job.get("schedule_display"),
                "next_run_at": job.get("next_run_at"),
                "last_run_at": job.get("last_run_at"),
                "last_status": job.get("last_status"),
                "last_delivery_error": job.get("last_delivery_error"),
            }
        )

    return {
        "status": "ok",
        "queue_name": DEFAULT_CRON_QUEUE_NAME,
        "queue_depth": queue_depth,
        "claim_count": len(claims),
        "due_count": len(due_jobs),
        "due_jobs": [
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "next_run_at": job.get("next_run_at"),
                "schedule": job.get("schedule_display"),
            }
            for job in due_jobs[: max(limit, 0)]
        ],
        "jobs": summarized_jobs,
    }


def _enqueue_due_cron_jobs_impl(limit: int = DEFAULT_CRON_QUEUE_BATCH_SIZE) -> dict[str, Any]:
    _prepare_runtime_environment()
    if _should_reload_modal_volume_for_claims("cron"):
        _sync_modal_volume(reload=True)

    from cron.jobs import get_due_jobs

    due_jobs = get_due_jobs()
    queue = _get_cron_queue()
    enqueued: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for job in due_jobs[: max(limit, 0)]:
        claimed, claim_token = _claim_due_cron_job(job)
        if not claimed:
            skipped.append({"id": job.get("id"), "name": job.get("name"), "reason": "already_claimed"})
            continue

        payload = {
            "job_id": job.get("id"),
            "job_name": job.get("name"),
            "scheduled_for": job.get("next_run_at"),
            "claim_token": claim_token,
            "enqueued_at": int(time.time()),
        }
        queue.put(payload, partition_ttl=max(DEFAULT_CRON_QUEUE_CLAIM_TTL_SECONDS, 3600))
        enqueued.append(payload)

    return {
        "status": "ok",
        "due_count": len(due_jobs),
        "enqueued_count": len(enqueued),
        "skipped_count": len(skipped),
        "enqueued": enqueued,
        "skipped": skipped,
        "queue_depth": _safe_cron_queue_depth(),
    }


def _should_process_queued_cron_job(job: dict[str, Any], payload: dict[str, Any]) -> tuple[bool, str | None]:
    if not job:
        return False, "job_not_found"
    if not job.get("enabled", True):
        return False, "job_disabled"

    scheduled_for = str(payload.get("scheduled_for") or "").strip()
    current_next = str(job.get("next_run_at") or "").strip()
    if scheduled_for and current_next and scheduled_for != current_next:
        return False, "schedule_changed"

    if current_next:
        try:
            next_run_dt = datetime.fromisoformat(current_next)
            now = datetime.now(next_run_dt.tzinfo)
            if next_run_dt > now:
                return False, "not_due"
        except Exception:
            pass

    return True, None


def _process_cron_queue_item(payload: Any) -> dict[str, Any]:
    _prepare_runtime_environment()
    if _should_reload_modal_volume_for_claims("cron"):
        _sync_modal_volume(reload=True)

    from cron.jobs import get_job, mark_job_run, save_job_output
    from cron.scheduler import SILENT_MARKER, _deliver_result, run_job

    if not isinstance(payload, dict):
        payload = {"job_id": str(payload or "")}

    job_id = str(payload.get("job_id") or "").strip()
    claim_token = str(payload.get("claim_token") or "").strip() or None
    if not job_id:
        return {"status": "skipped", "reason": "missing_job_id"}

    job = get_job(job_id)
    should_run, skip_reason = _should_process_queued_cron_job(job, payload)
    if not should_run:
        _release_cron_job_claim(job_id, claim_token=claim_token)
        return {"status": "skipped", "job_id": job_id, "reason": skip_reason}

    success, output, final_response, error = run_job(job)
    output_file = str(save_job_output(job_id, output))

    delivery_error = None
    deliver_content = final_response if success else f"⚠️ Cron job '{job.get('name', job_id)}' failed:\n{error}"
    should_deliver = bool(deliver_content)
    if should_deliver and success and SILENT_MARKER in deliver_content.strip().upper():
        should_deliver = False

    if should_deliver:
        try:
            delivery_error = _deliver_result(job, deliver_content)
        except Exception as exc:
            delivery_error = str(exc)
            logger.error("Cron delivery failed for job %s: %s", job_id, exc)

    mark_job_run(job_id, success, error, delivery_error=delivery_error)
    _sync_modal_volume(commit=True)
    _release_cron_job_claim(job_id, claim_token=claim_token)

    return {
        "status": "ok" if success else "error",
        "job_id": job_id,
        "job_name": job.get("name"),
        "output_file": output_file,
        "delivery_error": delivery_error,
        "error": error,
    }


def _process_cron_queue_impl(max_jobs: int = 1) -> dict[str, Any]:
    _prepare_runtime_environment()

    queue = _get_cron_queue()
    items = queue.get_many(max(max_jobs, 0), block=False) if max_jobs > 0 else []
    results = [_process_cron_queue_item(item) for item in items]
    return {
        "status": "ok",
        "processed_count": len(results),
        "results": results,
        "queue_depth": _safe_cron_queue_depth(),
    }


def _cron_scheduler_tick_impl(
    *,
    enqueue_limit: int = DEFAULT_CRON_QUEUE_BATCH_SIZE,
    worker_count: int = DEFAULT_CRON_QUEUE_WORKERS,
) -> dict[str, Any]:
    enqueue_result = _enqueue_due_cron_jobs_impl(limit=enqueue_limit)
    queue_depth = enqueue_result.get("queue_depth")
    if queue_depth is None:
        queue_depth = _safe_cron_queue_depth() or 0

    spawned_workers = 0
    if modal is not None and worker_count > 0 and queue_depth:
        spawned_workers = min(int(queue_depth), max(worker_count, 0))
        for _ in range(spawned_workers):
            process_cron_queue.spawn(max_jobs=1)  # type: ignore[name-defined]

    return {
        "status": "ok",
        "enqueue": enqueue_result,
        "spawned_workers": spawned_workers,
        "queue_depth": _safe_cron_queue_depth(),
    }


async def _send_telegram_message(bot_token: str, chat_id: str | int, text: str) -> None:
    import httpx

    response = None
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
    response.raise_for_status()


def create_web_app():
    if FastAPI is None or Header is None or HTTPException is None or Request is None:
        raise RuntimeError("FastAPI is required to build the Modal ASGI app")

    app = FastAPI(title="Hermes Agent Modal Gateway", version="1.0.0")

    @app.on_event("startup")
    async def _startup_sync_telegram_webhook() -> None:
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        if not settings.telegram_bot_token or not settings.telegram_webhook_url:
            return
        try:
            status = await _maybe_sync_telegram_webhook(settings)
            logger.info(
                "Telegram webhook startup sync: expected=%s registered=%s matched=%s pending=%s",
                status.get("expected_url"),
                status.get("registered_url"),
                status.get("matches_expected"),
                status.get("pending_update_count"),
            )
        except Exception as exc:
            logger.warning("Telegram webhook startup sync failed: %s", exc)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        gateway_import_ok = True
        gateway_import_error = None
        telegram_webhook = None
        memory_provider = _get_memory_provider_status()
        try:
            import gateway.run  # noqa: F401
            import run_agent  # noqa: F401
        except Exception as exc:
            gateway_import_ok = False
            gateway_import_error = str(exc)
        if settings.telegram_bot_token:
            try:
                telegram_webhook = await _get_telegram_webhook_status(settings)
            except Exception as exc:
                telegram_webhook = {
                    "configured": True,
                    "expected_url": settings.telegram_webhook_url,
                    "error": str(exc),
                }
        return {
            "status": "ok",
            "service": APP_NAME,
            "telegram_configured": bool(settings.telegram_bot_token),
            "feishu_configured": bool(settings.feishu_app_id and settings.feishu_app_secret),
            "qq_configured": bool(settings.qq_app_id and settings.qq_app_secret),
            "telegram_webhook": telegram_webhook,
            "chat_queue": {
                "queue_name": DEFAULT_CHAT_QUEUE_NAME,
                "queue_depth": _safe_chat_queue_depth(),
            },
            "cron": _cron_status_impl(limit=5),
            "memory_provider": memory_provider,
            "model_routing": _build_model_routing_debug_state(force_refresh=False, allow_network=False),
            "runtime_config": _sync_runtime_config(),
            "gateway_import_ok": gateway_import_ok,
            "gateway_import_error": gateway_import_error,
            "settings": _serialize_settings_for_log(settings),
        }

    @app.post("/invoke")
    async def invoke(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        settings = RuntimeSettings.from_env()
        if not _validate_bearer_token(authorization, settings.bearer_token):
            raise HTTPException(status_code=401, detail="Unauthorized")

        payload = await request.json()
        task_input = str(payload.get("input") or "").strip()
        if not task_input:
            raise HTTPException(status_code=400, detail="Missing input")

        return _run_agent_task_impl(
            task_input,
            session_key=payload.get("session_key"),
            model_name=payload.get("model_name"),
            max_tokens=payload.get("max_tokens"),
        )

    @app.post("/telegram/webhook")
    async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        settings = RuntimeSettings.from_env()
        if not settings.telegram_bot_token:
            raise HTTPException(status_code=503, detail="Telegram bot token is not configured")
        if not _validate_telegram_secret(
            x_telegram_bot_api_secret_token,
            settings.telegram_webhook_secret,
        ):
            raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")

        update = await request.json()
        update_id = update.get("update_id")
        message = (
            update.get("message")
            or update.get("edited_message")
            or update.get("channel_post")
            or update.get("edited_channel_post")
            or {}
        )
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        logger.info(
            "Telegram webhook inbound: update_id=%s chat_id=%s chat_type=%s user_id=%s username=%s text=%r",
            update_id,
            chat.get("id"),
            chat.get("type"),
            sender.get("id"),
            sender.get("username"),
            (message.get("text") or "")[:200],
        )
        if update_id is not None and not _mark_update_seen(str(update_id)):
            return {"status": "duplicate", "update_id": update_id}
        if settings.telegram_send_ack:
            chat_id = chat.get("id")
            if chat_id:
                await _send_telegram_message(settings.telegram_bot_token, chat_id, "Thinking...")
        context = _extract_telegram_queue_context(update)
        enqueue_result = await _enqueue_chat_event_async(
            platform="telegram",
            partition=context["partition"],
            payload=update,
            metadata=context,
        )
        logger.info(
            "Telegram webhook queued: update_id=%s partition=%s queue_depth=%s",
            update_id,
            context["partition"],
            enqueue_result.get("queue_depth"),
        )

        worker = globals().get("process_chat_queue")
        if worker is not None and hasattr(worker, "spawn"):
            spawn_handle = getattr(worker, "spawn")
            spawn_kwargs = {"platform": "telegram", "partition": context["partition"], "max_items": DEFAULT_CHAT_QUEUE_BATCH_SIZE}
            if hasattr(spawn_handle, "aio"):
                await spawn_handle.aio(**spawn_kwargs)  # type: ignore[union-attr]
            else:
                spawn_handle(**spawn_kwargs)  # type: ignore[operator]
        else:
            _process_chat_queue_impl(platform="telegram", partition=context["partition"], max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE)

        return {"status": "accepted", "update_id": update_id}

    @app.post("/feishu/webhook")
    async def feishu_webhook(request: Request) -> Response:
        settings = RuntimeSettings.from_env()
        if not settings.feishu_app_id or not settings.feishu_app_secret:
            raise HTTPException(status_code=503, detail="Feishu app credentials are not configured")

        try:
            fast_response = await _try_handle_feishu_verification_fast(request, settings)
            if fast_response is not None:
                return fast_response
            _adapter, payload = await _parse_feishu_webhook_request(request)
            if payload.get("type") == "url_verification":
                return JSONResponse({"challenge": payload.get("challenge", "")})

            _append_feishu_trace("webhook.accepted", payload)
            event_id, event_type = _extract_feishu_event_metadata(payload)
            if event_id and not _mark_feishu_event_seen(event_id):
                logger.warning(
                    "[Feishu] duplicate webhook event ignored event_type=%s event_id=%s",
                    event_type or "unknown",
                    event_id or "none",
                )
                return JSONResponse({"code": 0, "msg": "duplicate"})
            logger.warning(
                "[Feishu] webhook accepted event_type=%s event_id=%s",
                event_type or "unknown",
                event_id or "none",
            )
            context = _extract_feishu_queue_context(payload)
            enqueue_result = await _enqueue_chat_event_async(
                platform="feishu",
                partition=context["partition"],
                payload=payload,
                metadata=context,
            )
            _append_feishu_trace(
                "queue.enqueue",
                payload,
                partition=context["partition"],
                queue_depth=enqueue_result.get("queue_depth"),
            )
            logger.warning(
                "[Feishu] queue enqueue event_type=%s event_id=%s partition=%s queue_depth=%s",
                event_type or "unknown",
                event_id or "none",
                context["partition"],
                enqueue_result.get("queue_depth"),
            )

            worker = globals().get("process_chat_queue")
            if worker is not None and hasattr(worker, "spawn"):
                spawn_handle = getattr(worker, "spawn")
                spawn_kwargs = {"platform": "feishu", "partition": context["partition"], "max_items": DEFAULT_CHAT_QUEUE_BATCH_SIZE}
                if hasattr(spawn_handle, "aio"):
                    await spawn_handle.aio(**spawn_kwargs)  # type: ignore[union-attr]
                else:
                    spawn_handle(**spawn_kwargs)  # type: ignore[operator]
                _append_feishu_trace("webhook.spawned", payload, partition=context["partition"])
                logger.warning(
                    "[Feishu] webhook spawned chat worker event_type=%s event_id=%s partition=%s",
                    event_type or "unknown",
                    event_id or "none",
                    context["partition"],
                )
                return JSONResponse({"code": 0, "msg": "accepted"})

            _process_chat_queue_impl(platform="feishu", partition=context["partition"], max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE)
            return JSONResponse({"code": 0, "msg": "accepted"})
        except Exception as exc:
            logger.exception("Feishu webhook dispatch failed")
            raise HTTPException(status_code=500, detail=f"Feishu dispatch failed: {exc}") from exc

    @app.post("/qq/webhook")
    async def qq_webhook(
        request: Request,
        x_bot_appid: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        settings = RuntimeSettings.from_env()
        if not settings.qq_app_id or not settings.qq_app_secret:
            raise HTTPException(status_code=503, detail="QQ bot credentials are not configured")

        payload = await request.json()
        try:
            return await _dispatch_qq_update(
                payload,
                headers={"X-Bot-Appid": x_bot_appid or ""},
            )
        except Exception as exc:
            try:
                from gateway.platforms.qq import QQWebhookError

                if isinstance(exc, QQWebhookError):
                    raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
            except ImportError:
                pass
            logger.exception("QQ webhook dispatch failed")
            raise HTTPException(status_code=500, detail=f"QQ dispatch failed: {exc}") from exc

    return app


if modal is not None:
    app = modal.App(APP_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("nodejs", "npm")
        .pip_install_from_pyproject(
            "pyproject.toml",
            optional_dependencies=["modal", "messaging", "cron", "mcp", "pty", "feishu"],
        )
        .pip_install(["fastapi[standard]", "supermemory>=3.33.0,<4", "uv>=0.7.0,<1"])
        .env(
            {
                "HERMES_HOME": "/data/hermes-home",
                "HERMES_BUNDLED_SKILLS": "/root/skills",
            }
        )
        .add_local_python_source(
            "acp_adapter",
            "agent",
            "cron",
            "environments",
            "gateway",
            "hermes_cli",
            "plugins",
            "tools",
            copy=True,
        )
        .add_local_dir("acp_registry", remote_path="/root/acp_registry", copy=True)
        .add_local_dir("skills", remote_path="/root/skills", copy=True)
        .add_local_dir("optional-skills", remote_path="/root/optional-skills", copy=True)
        .add_local_file("run_agent.py", remote_path="/root/run_agent.py", copy=True)
        .add_local_file("batch_runner.py", remote_path="/root/batch_runner.py", copy=True)
        .add_local_file("model_tools.py", remote_path="/root/model_tools.py", copy=True)
        .add_local_file("toolsets.py", remote_path="/root/toolsets.py", copy=True)
        .add_local_file(
            "toolset_distributions.py",
            remote_path="/root/toolset_distributions.py",
            copy=True,
        )
        .add_local_file(
            "trajectory_compressor.py",
            remote_path="/root/trajectory_compressor.py",
            copy=True,
        )
        .add_local_file("cli.py", remote_path="/root/cli.py", copy=True)
        .add_local_file("rl_cli.py", remote_path="/root/rl_cli.py", copy=True)
        .add_local_file(
            "hermes_constants.py",
            remote_path="/root/hermes_constants.py",
            copy=True,
        )
        .add_local_file(
            "hermes_logging.py",
            remote_path="/root/hermes_logging.py",
            copy=True,
        )
        .add_local_file("hermes_state.py", remote_path="/root/hermes_state.py", copy=True)
        .add_local_file("hermes_time.py", remote_path="/root/hermes_time.py", copy=True)
        .add_local_file("utils.py", remote_path="/root/utils.py", copy=True)
        .add_local_file("README.md", remote_path="/root/README.md", copy=True)
        .add_local_file("MANIFEST.in", remote_path="/root/MANIFEST.in", copy=True)
        .add_local_file(
            "config.modal.yaml",
            remote_path="/root/config.modal.yaml",
            copy=True,
        )
        .add_local_file(
            ".env.modal.example",
            remote_path="/root/.env.modal.example",
            copy=True,
        )
        .add_local_file(
            "supermemory.modal.json",
            remote_path="/root/supermemory.modal.json",
            copy=True,
        )
    )
    if Path(".hermes/plugins").is_dir():
        image = image.add_local_dir(
            ".hermes/plugins",
            remote_path="/root/.hermes/plugins",
            copy=True,
        )
    volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)
    MODAL_VOLUME = volume
    secrets = [modal.Secret.from_name(DEFAULT_SECRET_NAME)]

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=3600,
        memory=4096,
        cpu=2,
    )
    def run_agent_task(
        task_input: str,
        session_key: str = "",
        model_name: str = "",
        max_tokens: int = 0,
    ) -> dict[str, Any]:
        return _run_agent_task_impl(
            task_input,
            session_key=session_key or None,
            model_name=model_name or None,
            max_tokens=max_tokens or None,
        )

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=7200,
        memory=8192,
        cpu=4,
    )
    def run_batch_tasks(
        task_file: str = "/data/tasks.json",
        model_name: str = "",
    ) -> dict[str, Any]:
        tasks_path = Path(task_file)
        if not tasks_path.exists():
            return {"status": "error", "message": f"Task file not found: {task_file}"}

        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        results: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            result = _run_agent_task_impl(
                str(task.get("input") or ""),
                session_key=task.get("session_key"),
                model_name=model_name or task.get("model_name"),
                max_tokens=task.get("max_tokens"),
            )
            results.append({"index": index, **result})
        return {"status": "completed", "results": results}

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
    )
    def health_check() -> dict[str, Any]:
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        gateway_import_ok = True
        gateway_import_error = None
        telegram_webhook = None
        memory_provider = _get_memory_provider_status()
        try:
            import gateway.run  # noqa: F401
            import run_agent  # noqa: F401
        except Exception as exc:
            gateway_import_ok = False
            gateway_import_error = str(exc)
        if settings.telegram_bot_token:
            try:
                telegram_webhook = asyncio.run(_get_telegram_webhook_status(settings))
            except Exception as exc:
                telegram_webhook = {
                    "configured": True,
                    "expected_url": settings.telegram_webhook_url,
                    "error": str(exc),
                }
        return {
            "status": "healthy",
            "service": APP_NAME,
            "volume_root": str(DATA_ROOT),
            "telegram_configured": bool(settings.telegram_bot_token),
            "feishu_configured": bool(settings.feishu_app_id and settings.feishu_app_secret),
            "qq_configured": bool(settings.qq_app_id and settings.qq_app_secret),
            "telegram_webhook": telegram_webhook,
            "chat_queue": {
                "queue_name": DEFAULT_CHAT_QUEUE_NAME,
                "queue_depth": _safe_chat_queue_depth(),
            },
            "cron": _cron_status_impl(limit=5),
            "memory_provider": memory_provider,
            "model_routing": _build_model_routing_debug_state(force_refresh=False, allow_network=False),
            "runtime_config": _sync_runtime_config(),
            "gateway_import_ok": gateway_import_ok,
            "gateway_import_error": gateway_import_error,
            "settings": _serialize_settings_for_log(settings),
        }

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
    )
    def debug_telegram_auth_state(user_id: str = "6379576758") -> dict[str, Any]:
        _prepare_runtime_environment()

        from gateway.config import Platform
        from gateway.pairing import PAIRING_DIR, PairingStore
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource
        from hermes_constants import get_hermes_home

        store = PairingStore()
        runner = GatewayRunner()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=user_id,
            chat_type="dm",
            user_id=user_id,
            user_name="debug-user",
        )
        approved_path = PAIRING_DIR / "telegram-approved.json"
        pending_path = PAIRING_DIR / "telegram-pending.json"
        legacy_pairing_dir = get_hermes_home() / "pairing"

        return {
            "hermes_home": str(get_hermes_home()),
            "pairing_dir": str(PAIRING_DIR),
            "legacy_pairing_dir": str(legacy_pairing_dir),
            "legacy_pairing_dir_exists": legacy_pairing_dir.exists(),
            "approved_path": str(approved_path),
            "approved_path_exists": approved_path.exists(),
            "approved_raw": _load_json_file(approved_path, {}),
            "pending_path": str(pending_path),
            "pending_path_exists": pending_path.exists(),
            "pending_raw": _load_json_file(pending_path, {}),
            "is_approved": store.is_approved("telegram", user_id),
            "list_approved": store.list_approved("telegram"),
            "runner_is_authorized": runner._is_user_authorized(source),
            "telegram_allowed_users": os.getenv("TELEGRAM_ALLOWED_USERS", ""),
            "telegram_allow_all_users": os.getenv("TELEGRAM_ALLOW_ALL_USERS", ""),
            "gateway_allowed_users": os.getenv("GATEWAY_ALLOWED_USERS", ""),
            "gateway_allow_all_users": os.getenv("GATEWAY_ALLOW_ALL_USERS", ""),
        }

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=600,
        memory=4096,
        cpu=2,
    )
    def process_feishu_event(payload: dict[str, Any]) -> dict[str, Any]:
        _prepare_runtime_environment()
        context = _extract_feishu_queue_context(payload)
        result = _enqueue_chat_event(
            platform="feishu",
            partition=context["partition"],
            payload=payload,
            metadata=context,
        )
        _append_feishu_trace("queue.enqueue", payload, partition=context["partition"], queue_depth=result.get("queue_depth"))
        process_chat_queue.spawn(platform="feishu", partition=context["partition"], max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE)  # type: ignore[name-defined]
        return result

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=600,
        memory=4096,
        cpu=2,
    )
    def process_chat_queue(platform: str, partition: str, max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE) -> dict[str, Any]:
        return _process_chat_queue_impl(platform=platform, partition=partition, max_items=max_items)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def debug_feishu_runtime() -> dict[str, Any]:
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()

        from gateway.platforms.feishu import check_feishu_requirements

        payload: dict[str, Any] = {
            "settings": _serialize_settings_for_log(settings),
            "configured": bool(settings.feishu_app_id and settings.feishu_app_secret),
            "requirements_ok": bool(check_feishu_requirements()),
            "webhook_path": "/feishu/webhook",
            "public_webhook_url": f"{(_normalize_public_https_url(os.getenv('HERMES_PUBLIC_BASE_URL') or os.getenv('PUBLIC_BASE_URL')) or 'https://isuyee88--hermes-agent-web-app.modal.run')}/feishu/webhook",
            "verification_token_configured": bool(settings.feishu_verification_token),
            "encrypt_key_configured": bool(settings.feishu_encrypt_key),
            "allowed_users": _split_csv(os.getenv("FEISHU_ALLOWED_USERS")),
            "group_policy": os.getenv("FEISHU_GROUP_POLICY", "allowlist"),
            "group_require_mention": _is_truthy(os.getenv("FEISHU_GROUP_REQUIRE_MENTION"), default=False),
            "feishu_disabled_toolsets": _split_csv(os.getenv("HERMES_FEISHU_DISABLED_TOOLSETS")),
            "feishu_resolve_sender_names": _is_truthy(os.getenv("HERMES_FEISHU_RESOLVE_SENDER_NAMES"), default=False),
            "feishu_menu_open_by_open_id": _is_truthy(os.getenv("HERMES_FEISHU_MENU_OPEN_BY_OPEN_ID"), default=True),
            "home_channel": os.getenv("FEISHU_HOME_CHANNEL", ""),
            "menu_manifest": _build_feishu_menu_manifest(),
            "chat_queue": {
                "queue_name": DEFAULT_CHAT_QUEUE_NAME,
                "queue_depth": _safe_chat_queue_depth(),
            },
            "capabilities": _build_feishu_capabilities_debug_state(probe=False),
        }

        if not payload["configured"] or not payload["requirements_ok"]:
            return payload

        try:
            runtime = asyncio.run(_get_feishu_gateway_runtime())
            adapter = runtime.adapter
            payload["runtime"] = {
                "connected": True,
                "domain": getattr(adapter, "_domain_name", ""),
                "connection_mode": getattr(adapter, "_connection_mode", ""),
                "webhook_path": getattr(adapter, "_webhook_path", ""),
                "group_policy": getattr(adapter, "_group_policy", ""),
                "default_group_policy": getattr(adapter, "_default_group_policy", ""),
                "group_require_mention": bool(getattr(adapter, "_group_require_mention", True)),
                "allowed_group_users": sorted(getattr(adapter, "_allowed_group_users", set())),
                "bot_open_id": getattr(adapter, "_bot_open_id", ""),
                "bot_user_id": getattr(adapter, "_bot_user_id", ""),
                "bot_name": getattr(adapter, "_bot_name", ""),
            }
        except Exception as exc:
            payload["runtime"] = {
                "connected": False,
                "error": str(exc),
            }

        return payload

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
    )
    def debug_feishu_menu_config() -> dict[str, Any]:
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        payload = _build_feishu_menu_manifest()
        payload["configured"] = bool(settings.feishu_app_id and settings.feishu_app_secret)
        payload["public_webhook_url"] = (
            f"{(_normalize_public_https_url(os.getenv('HERMES_PUBLIC_BASE_URL') or os.getenv('PUBLIC_BASE_URL')) or 'https://isuyee88--hermes-agent-web-app.modal.run')}/feishu/webhook"
        )
        payload["home_channel"] = os.getenv("FEISHU_HOME_CHANNEL", "")
        return payload

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def debug_feishu_capabilities(probe: bool = False) -> dict[str, Any]:
        return _build_feishu_capabilities_debug_state(probe=probe)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def debug_feishu_model_registry(force_refresh: bool = False) -> dict[str, Any]:
        return _build_feishu_model_registry_debug_state(force_refresh=force_refresh)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
    )
    def debug_model_routing_state(force_refresh: bool = False) -> dict[str, Any]:
        return _build_model_routing_debug_state(force_refresh=force_refresh, allow_network=True)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
    )
    def debug_feishu_trace(limit: int = 100) -> dict[str, Any]:
        _prepare_runtime_environment()
        rows = _read_feishu_trace(limit=limit)
        return {
            "status": "ok",
            "trace_file": str(FEISHU_TRACE_PATH),
            "count": len(rows),
            "rows": rows,
        }

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
    )
    def debug_session_route(session_key: str) -> dict[str, Any]:
        return _debug_session_route_state(session_key)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
    )
    def debug_gateway_session(session_key: str) -> dict[str, Any]:
        return _debug_gateway_session_state(session_key)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def validate_feishu_webhook() -> dict[str, Any]:
        return _validate_feishu_webhook_impl()

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def sync_telegram_webhook(
        webhook_url: str = "",
        public_base_url: str = "",
        drop_pending_updates: bool = False,
    ) -> dict[str, Any]:
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        if not settings.telegram_bot_token:
            return {"status": "error", "message": "TELEGRAM_BOT_TOKEN is not configured"}
        if not _is_valid_telegram_bot_token_format(settings.telegram_bot_token):
            return {
                "status": "error",
                "message": "TELEGRAM_BOT_TOKEN has invalid format. Expected '<bot_id>:<secret>'.",
            }

        desired_url = _resolve_telegram_webhook_url(
            explicit_url=webhook_url or settings.telegram_webhook_url,
            public_base_url=public_base_url,
        ) or settings.telegram_webhook_url
        if not desired_url:
            return {
                "status": "error",
                "message": "Set TELEGRAM_WEBHOOK_URL or HERMES_PUBLIC_BASE_URL before syncing Telegram webhook",
            }

        settings = RuntimeSettings(
            **{
                **asdict(settings),
                "telegram_webhook_url": desired_url,
            }
        )
        status = asyncio.run(
            _get_telegram_webhook_status(
                settings,
                ensure_registered=True,
                drop_pending_updates=drop_pending_updates,
            )
        )
        return {"status": "ok", "telegram_webhook": status}

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def approve_pairing_code(platform: str, code: str) -> dict[str, Any]:
        return _approve_pairing_impl(platform, code)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=180,
    )
    def validate_tavily_integration() -> dict[str, Any]:
        return _validate_tavily_integration_impl()

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=30,
    )
    def cron_status(limit: int = 10) -> dict[str, Any]:
        return _cron_status_impl(limit=limit)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=900,
        memory=4096,
        cpu=2,
    )
    def process_cron_queue(max_jobs: int = 1) -> dict[str, Any]:
        return _process_cron_queue_impl(max_jobs=max_jobs)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def cron_scheduler_tick(
        enqueue_limit: int = DEFAULT_CRON_QUEUE_BATCH_SIZE,
        worker_count: int = DEFAULT_CRON_QUEUE_WORKERS,
    ) -> dict[str, Any]:
        return _cron_scheduler_tick_impl(
            enqueue_limit=enqueue_limit,
            worker_count=worker_count,
        )

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
        schedule=modal.Period(minutes=1),
    )
    def cron_scheduler_heartbeat() -> dict[str, Any]:
        return _cron_scheduler_tick_impl(
            enqueue_limit=DEFAULT_CRON_QUEUE_BATCH_SIZE,
            worker_count=DEFAULT_CRON_QUEUE_WORKERS,
        )

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=300,
    )
    @modal.asgi_app()
    def web_app():
        return create_web_app()

    @app.local_entrypoint()
    def main() -> None:
        print(f"Deploying/serving Modal app: {APP_NAME}")
        print(f"Secrets source: {DEFAULT_SECRET_NAME}")
        print(f"Volume source: {DEFAULT_VOLUME_NAME}")
        print("Exposed routes: /healthz, /invoke, /telegram/webhook, /feishu/webhook, /qq/webhook")
