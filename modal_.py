from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
import hashlib
import hmac
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from internal.feishu.result_files import lookup_result_file as _lookup_result_file
from internal.feishu.result_files import register_result_file as _register_result_file

try:
    import modal
except ImportError:
    modal = None

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response
except Exception:
    FastAPI = None
    HTTPException = Exception
    Request = Any
    FileResponse = None
    JSONResponse = None
    Response = None


APP_NAME = "hermes-agent"
MODAL_VOLUME_MOUNT_PATH = Path("/data")
DATA_ROOT = Path(
    os.getenv("HERMES_MODAL_DATA_DIR")
    or (MODAL_VOLUME_MOUNT_PATH / "hermes-modal-stub" if modal is not None else Path(tempfile.gettempdir()) / "hermes-modal-stub")
)
HERMES_HOME_DIR = Path(os.getenv("HERMES_HOME") or DATA_ROOT / "hermes-home")
FEISHU_TRACE_PATH = DATA_ROOT / "feishu_trace.jsonl"
FEISHU_EVENTS_PATH = DATA_ROOT / "feishu_events.json"
SESSION_STATE_PATH = DATA_ROOT / "session_state.json"
SESSIONS_DIR = DATA_ROOT / "sessions"
UPDATES_PATH = DATA_ROOT / "telegram_updates.json"
FEISHU_BOT_IDENTITY_CACHE_PATH = DATA_ROOT / "feishu_bot_identity_cache.json"
CHAT_QUEUE_CLAIMS_PATH = DATA_ROOT / "chat_queue_claims.json"
CHAT_QUEUE_WARMUPS_PATH = DATA_ROOT / "chat_queue_warmups.json"
CRON_QUEUE_CLAIMS_PATH = DATA_ROOT / "cron_queue_claims.json"
FEISHU_SYNC_STATE_PATH = DATA_ROOT / "feishu_sync_state.json"
ROUTING_STATE_PATH = DATA_ROOT / "routing_state.json"
TELEGRAM_WEBHOOK_SYNC_STATE_PATH = DATA_ROOT / "telegram_webhook_sync.json"
SESSION_STATE: dict[str, dict[str, str]] = {}
FEISHU_INTERNAL_RESULT_FILE_TTL_SECONDS = 3600
_FEISHU_INTERNAL_RESULT_FILES: dict[str, dict[str, Any]] = {}
_FEISHU_INTERNAL_RESULT_FILES_LOCK = Lock()
_FEISHU_BOT_IDENTITY_CACHE_LOCK = Lock()
_CHAT_QUEUE_LOCK = Lock()
_FEISHU_EVENT_LOCK = Lock()
_CAMOFOX_SERVER_LOCK = Lock()
_CAMOFOX_SERVER_PROCESS: Any = None
_RUNTIME_SKILLS_SYNCED = False
_RUNTIME_SKILLS_SYNC_LOCK = RLock()
_RECENT_CHAT_WORKER_SPAWNS: dict[str, float] = {}
_RECENT_CHAT_WORKER_SPAWNS_LOCK = Lock()
_TELEGRAM_RUNTIME: Any = None
_TELEGRAM_RUNTIME_LOCK: Any = None
_FEISHU_RUNTIME: Any = None
_FEISHU_RUNTIME_LOCK: Any = None
_QQ_RUNTIME: Any = None
_QQ_RUNTIME_LOCK: Any = None
DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS = 3600
DEFAULT_CHAT_QUEUE_STALE_CLAIM_TAKEOVER_SECONDS = 300
DEFAULT_CHAT_QUEUE_ACTIVE_CLAIM_SKIP_SECONDS = 30
DEFAULT_FEISHU_RECENT_SPAWN_SKIP_SECONDS = 5
DEFAULT_CHAT_QUEUE_BATCH_SIZE = 8
DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS = 8
DEFAULT_CHAT_QUEUE_NAME = "hermes-agent-chat-queue"
DEFAULT_CHAT_QUEUE_LINGER_SECONDS = 2.0
DEFAULT_CRON_QUEUE_NAME = "hermes-agent-cron-queue"
DEFAULT_CRON_QUEUE_BATCH_SIZE = 8
DEFAULT_CRON_QUEUE_WORKERS = 2
DEFAULT_CRON_QUEUE_MAX_JOBS_PER_WORKER = 4
DEFAULT_SECRET_NAME = "hermes-agent-secrets"
DEFAULT_VOLUME_NAME = "hermes-agent-data"
DEFAULT_SESSION_ROUTE_TTL_SECONDS = 1800
DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS = 3600
DEFAULT_FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS = 3600
DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/d1215a30b84b673ef0367010b0e78c10/affiliate-manager/compat"
)
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_DISABLED_TOOLSETS = ("browser", "terminal", "code_execution", "delegation", "tts", "messaging", "rl")
DEFAULT_TELEGRAM_WEBHOOK_SYNC_BACKOFF_SECONDS = 300
DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY = "inline_enqueue_spawn"
LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES = {"spawn_process_feishu_event": "inline_enqueue_spawn"}
SUPPORTED_FEISHU_MESSAGE_INGRESS_STRATEGIES = {
    "inline_enqueue_spawn",
    "spawn_process_feishu_event",
    "spawn_process_feishu_message_inline",
}
DEFAULT_FEISHU_MESSAGE_QUEUE_MAX_AGE_SECONDS = 300
DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS = 5.0
DEFAULT_FEISHU_ACK_REACTION_MODE = "background"
SUPPORTED_FEISHU_ACK_REACTION_MODES = ("disabled", "inline", "background")
DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS = 1200
DEFAULT_FEISHU_ACK_REACTION_REQUEST_TIMEOUT_SECONDS = 10.0
DEFAULT_FEISHU_INLINE_FIRST_RESPONSE_BUDGET_MS = 5000
DEFAULT_FEISHU_PROVIDER_FIRST_TOKEN_BUDGET_MS = 8000
DEFAULT_FEISHU_INLINE_WORKER_HARD_BUDGET_MS = 30000
DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS = 180.0
DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB = 2048
DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU = 2
DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB = 1024
DEFAULT_CHAT_QUEUE_WORKER_CPU = 1
DEFAULT_WEB_APP_SCALEDOWN_WINDOW_SECONDS = 30
CHAT_WORKER_PREINIT_FEISHU_RUNTIME_ENABLED = False
WEB_APP_MEMORY_SNAPSHOT_ENABLED = _is_truthy(os.getenv("HERMES_MODAL_WEB_APP_MEMORY_SNAPSHOT_ENABLED"), default=False) if "_is_truthy" in globals() else False
FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED = _is_truthy(os.getenv("HERMES_MODAL_FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED"), default=False) if "_is_truthy" in globals() else False
FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED = _is_truthy(os.getenv("HERMES_MODAL_FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED"), default=False) if "_is_truthy" in globals() else False
CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED = _is_truthy(os.getenv("HERMES_MODAL_CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED"), default=False) if "_is_truthy" in globals() else False
INLINE_FAST_COMMAND_CANONICALS = {
    "model": "model",
    "provider": "provider",
    "status": "status",
    "personality": "personality",
    "help": "help",
    "browser": "browser",
}
_FEISHU_TRACE_TOKEN_RE = re.compile(r"\[trace:([A-Za-z0-9._:-]+)\]")
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

logger = logging.getLogger(__name__)
from internal.modal_runtime_settings import RuntimeSettings
from internal.modal_runtime_settings import derive_feishu_internal_bearer_token as _derive_feishu_internal_bearer_token
from internal.modal_runtime_settings import serialize_settings_for_log as _runtime_serialize_settings_for_log


def _load_json_file(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _split_csv(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_truthy(value: str | None, default: bool = False) -> bool:
    raw = str(value or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _safe_json_loads(raw: Any, default: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _mask_secret(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}...{text[-4:]}"


WEB_APP_MEMORY_SNAPSHOT_ENABLED = _is_truthy(
    os.getenv("HERMES_MODAL_WEB_APP_MEMORY_SNAPSHOT_ENABLED"),
    default=False,
)
FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED = _is_truthy(
    os.getenv("HERMES_MODAL_FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED"),
    default=False,
)
FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED = _is_truthy(
    os.getenv("HERMES_MODAL_FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED"),
    default=False,
)
CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED = _is_truthy(
    os.getenv("HERMES_MODAL_CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED"),
    default=False,
)


def _first_non_empty_str(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_phase_timings(phase_timings: dict[str, Any] | None) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in dict(phase_timings or {}).items():
        try:
            parsed = int(value or 0)
        except Exception:
            parsed = 0
        normalized[str(key)] = max(parsed, 0)
    return normalized


def _capture_phase_elapsed(phase_timings: dict[str, int], key: str, started_at: float) -> None:
    if not isinstance(phase_timings, dict):
        return
    phase_timings[str(key)] = max(int((time.perf_counter() - started_at) * 1000), 0)


def _is_valid_telegram_bot_token_format(value: str | None) -> bool:
    raw = str(value or "").strip()
    return bool(re.fullmatch(r"\d{6,}:[A-Za-z0-9_-]{20,}", raw))


def _validate_telegram_secret(provided: str | None, expected: str | None) -> bool:
    expected_token = str(expected or "").strip()
    if not expected_token:
        return True
    provided_token = str(provided or "").strip()
    return bool(provided_token) and hmac.compare_digest(provided_token, expected_token)


def _ensure_runtime_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    HERMES_HOME_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (
        UPDATES_PATH,
        FEISHU_EVENTS_PATH,
        FEISHU_BOT_IDENTITY_CACHE_PATH,
        CHAT_QUEUE_CLAIMS_PATH,
        CHAT_QUEUE_WARMUPS_PATH,
        TELEGRAM_WEBHOOK_SYNC_STATE_PATH,
        FEISHU_TRACE_PATH,
        SESSION_STATE_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def _path_is_modal_managed(path: Path) -> bool:
    if modal is None:
        return False
    try:
        return path.resolve().is_relative_to(MODAL_VOLUME_MOUNT_PATH.resolve())
    except Exception:
        return False


def _configure_runtime_settings_hooks() -> None:
    from internal.modal_runtime_settings import configure_runtime_settings

    configure_runtime_settings(
        pick_runtime_api_config=lambda: _pick_runtime_api_config(),
        desired_telegram_webhook_url=lambda: _desired_telegram_webhook_url(),
        derive_feishu_internal_bearer_token=lambda app_id, app_secret: _derive_feishu_internal_bearer_token(
            app_id=app_id,
            app_secret=app_secret,
        ),
        is_truthy=_is_truthy,
        get_feishu_sync_interval_seconds=lambda: 3600,
        split_csv=_split_csv,
        default_nvidia_base_url=DEFAULT_NVIDIA_BASE_URL,
        default_disabled_toolsets=DEFAULT_DISABLED_TOOLSETS,
    )


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


def _serialize_settings_for_log(settings: RuntimeSettings) -> dict[str, Any]:
    return _runtime_serialize_settings_for_log(settings, mask_secret=_mask_secret)


def _load_telegram_webhook_sync_state() -> dict[str, Any]:
    payload = _load_json_file(TELEGRAM_WEBHOOK_SYNC_STATE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_telegram_webhook_sync_state(payload: dict[str, Any]) -> None:
    _atomic_json_write(TELEGRAM_WEBHOOK_SYNC_STATE_PATH, payload)


def _telegram_webhook_retry_after_seconds(exc: Exception) -> int:
    from internal.telegram_gateway import telegram_webhook_retry_after_seconds

    return telegram_webhook_retry_after_seconds(
        exc,
        default_retry_seconds=DEFAULT_TELEGRAM_WEBHOOK_SYNC_BACKOFF_SECONDS,
    )


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


async def _fetch_telegram_webhook_info(bot_token: str) -> dict[str, Any]:
    from internal.telegram_gateway import fetch_telegram_webhook_info

    return await fetch_telegram_webhook_info(bot_token)


async def _set_telegram_webhook(
    bot_token: str,
    webhook_url: str,
    *,
    webhook_secret: str | None = None,
    drop_pending_updates: bool = False,
) -> dict[str, Any]:
    from internal.telegram_gateway import set_telegram_webhook

    return await set_telegram_webhook(
        bot_token,
        webhook_url,
        webhook_secret=webhook_secret,
        drop_pending_updates=drop_pending_updates,
    )


async def _get_telegram_webhook_status(
    settings: Any,
    *,
    ensure_registered: bool = False,
    drop_pending_updates: bool = False,
) -> dict[str, Any]:
    from internal.telegram_gateway import get_telegram_webhook_status

    return await get_telegram_webhook_status(
        settings,
        ensure_registered=ensure_registered,
        drop_pending_updates=drop_pending_updates,
        is_valid_telegram_bot_token_format=_is_valid_telegram_bot_token_format,
        is_truthy=_is_truthy,
        set_telegram_webhook_fn=_set_telegram_webhook,
        fetch_telegram_webhook_info_fn=_fetch_telegram_webhook_info,
    )


async def _maybe_sync_telegram_webhook(settings: Any) -> dict[str, Any]:
    from internal.telegram_gateway import maybe_sync_telegram_webhook

    return await maybe_sync_telegram_webhook(
        settings,
        drop_pending_updates=False,
        is_truthy=_is_truthy,
        get_telegram_webhook_status_fn=_get_telegram_webhook_status,
        save_telegram_webhook_sync_state=_save_telegram_webhook_sync_state,
        load_telegram_webhook_sync_state=_load_telegram_webhook_sync_state,
        set_telegram_webhook_fn=_set_telegram_webhook,
        telegram_webhook_retry_after_seconds_fn=_telegram_webhook_retry_after_seconds,
    )


async def _send_telegram_message(bot_token: str, chat_id: str | int, text: str) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        response.raise_for_status()


def _validate_bearer_token(authorization: str | None, expected_token: str | None) -> bool:
    expected = str(expected_token or "").strip()
    provided = str(authorization or "").strip()
    if not expected:
        return not provided
    if not provided.startswith("Bearer "):
        return False
    return provided[len("Bearer ") :].strip() == expected


def _validate_feishu_internal_bearer_token(authorization: str | None, settings: RuntimeSettings) -> bool:
    expected_tokens = [str(settings.feishu_internal_bearer_token or "").strip()]
    for suffix in ("", "2"):
        app_id = str(os.getenv(f"FEISHU_APP_ID{suffix}") or "").strip()
        app_secret = str(os.getenv(f"FEISHU_APP_SECRET{suffix}") or "").strip()
        token = _derive_feishu_internal_bearer_token(app_id=app_id, app_secret=app_secret)
        if token:
            expected_tokens.append(token)
    expected_tokens = [token for token in expected_tokens if token]
    if not expected_tokens:
        return _validate_bearer_token(authorization, None)
    return any(_validate_bearer_token(authorization, token) for token in expected_tokens)


def _pick_runtime_api_config() -> tuple[str | None, str | None, str | None]:
    from internal.model_catalog import normalize_cloudflare_ai_gateway_base_url
    from internal.modal_runtime_env import pick_runtime_api_config

    return pick_runtime_api_config(
        env_flag=lambda name, default=True: _is_truthy(os.getenv(name), default=default),
        normalize_cloudflare_ai_gateway_runtime_base_url=normalize_cloudflare_ai_gateway_base_url,
        default_cloudflare_ai_gateway_base_url=DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL,
    )


def _normalize_public_https_url(value: str | None) -> str | None:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        return None
    if not raw.lower().startswith("https://"):
        return None
    return raw


def _resolve_telegram_webhook_url(*, explicit_url: str | None = None, public_base_url: str | None = None) -> str | None:
    explicit = _normalize_public_https_url(explicit_url)
    if explicit:
        return explicit
    base = _normalize_public_https_url(public_base_url or os.getenv("HERMES_PUBLIC_BASE_URL"))
    if not base:
        return None
    return f"{base}/telegram/webhook"


def _desired_telegram_webhook_url() -> str | None:
    return _resolve_telegram_webhook_url(
        explicit_url=os.getenv("TELEGRAM_WEBHOOK_URL"),
        public_base_url=os.getenv("HERMES_PUBLIC_BASE_URL"),
    )


def _sync_runtime_config() -> str | None:
    from internal.modal_runtime_env import sync_runtime_config

    return sync_runtime_config(
        default_config_source=Path(__file__).resolve().parent / "config.modal.yaml",
        hermes_home_dir=HERMES_HOME_DIR,
        is_truthy=_is_truthy,
        materialize_dynamic_free_model_config=_materialize_dynamic_free_model_config,
        logger=logger,
    )


def _sync_supermemory_config() -> str | None:
    from internal.modal_runtime_env import sync_supermemory_config

    return sync_supermemory_config(
        default_supermemory_config_source=Path(__file__).resolve().parent / "supermemory.modal.json",
        hermes_home_dir=HERMES_HOME_DIR,
        is_truthy=_is_truthy,
    )


def _get_memory_provider_status() -> dict[str, Any]:
    from internal.modal_runtime_env import get_memory_provider_status

    return get_memory_provider_status(hermes_home_dir=HERMES_HOME_DIR)


def _get_camofox_url() -> str:
    from internal.modal_runtime_env import get_camofox_url

    return get_camofox_url()


def _is_local_camofox_url(url: str) -> bool:
    from internal.modal_runtime_env import is_local_camofox_url

    return is_local_camofox_url(url)


def _is_camofox_healthcheck_ready(url: str) -> bool:
    from internal.modal_runtime_env import is_camofox_healthcheck_ready

    return is_camofox_healthcheck_ready(url)


def _resolve_camofox_launch_command() -> list[str]:
    from internal.modal_runtime_env import resolve_camofox_launch_command

    return resolve_camofox_launch_command(which=shutil.which)


def _ensure_camofox_server() -> None:
    from internal.modal_runtime_env import ensure_camofox_server

    global _CAMOFOX_SERVER_PROCESS

    ensure_camofox_server(
        get_process=lambda: _CAMOFOX_SERVER_PROCESS,
        set_process=lambda process: globals().__setitem__("_CAMOFOX_SERVER_PROCESS", process),
        camofox_server_lock=_CAMOFOX_SERVER_LOCK,
        get_camofox_url=_get_camofox_url,
        is_local_camofox_url=_is_local_camofox_url,
        is_camofox_healthcheck_ready=_is_camofox_healthcheck_ready,
        resolve_camofox_launch_command=_resolve_camofox_launch_command,
        data_root=DATA_ROOT,
        default_camofox_boot_timeout_seconds=10.0,
        default_camofox_boot_poll_interval_seconds=0.2,
        logger=logger,
    )


def _prepare_runtime_environment() -> None:
    from internal.modal_runtime_env import prepare_runtime_environment

    prepare_runtime_environment(
        hermes_home_dir=HERMES_HOME_DIR,
        ensure_runtime_dirs=_ensure_runtime_dirs,
        ensure_camofox_server=_ensure_camofox_server,
        sync_runtime_config=_sync_runtime_config,
        sync_supermemory_config=_sync_supermemory_config,
        get_runtime_skills_synced=lambda: _RUNTIME_SKILLS_SYNCED,
        set_runtime_skills_synced=lambda value: globals().__setitem__("_RUNTIME_SKILLS_SYNCED", bool(value)),
        runtime_skills_sync_lock=_RUNTIME_SKILLS_SYNC_LOCK,
        logger=logger,
    )


def _extract_feishu_event_metadata(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    event_id = str(header.get("event_id") or payload.get("event_id") or "").strip() or None
    event_type = str(header.get("event_type") or payload.get("event_type") or "").strip() or None
    return event_id, event_type


def _collect_feishu_chat_id(payload: dict[str, Any]) -> str:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
    message = event.get("message") if isinstance(event, dict) and isinstance(event.get("message"), dict) else {}
    context = event.get("context") if isinstance(event, dict) and isinstance(event.get("context"), dict) else {}
    chat = event.get("chat") if isinstance(event, dict) and isinstance(event.get("chat"), dict) else {}
    return str(
        message.get("chat_id")
        or message.get("open_chat_id")
        or chat.get("chat_id")
        or chat.get("open_chat_id")
        or context.get("open_chat_id")
        or event.get("open_chat_id")
        or event.get("chat_id")
        or ""
    ).strip()


def _collect_feishu_actor_id(payload: dict[str, Any]) -> str:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
    sender = event.get("sender") if isinstance(event, dict) and isinstance(event.get("sender"), dict) else {}
    sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
    operator = event.get("operator") if isinstance(event, dict) and isinstance(event.get("operator"), dict) else {}
    operator_id = operator.get("operator_id") if isinstance(operator.get("operator_id"), dict) else {}
    user_id = event.get("user_id") if isinstance(event, dict) else None
    if isinstance(user_id, dict):
        return str(user_id.get("open_id") or user_id.get("user_id") or "").strip()
    return str(
        sender_id.get("open_id")
        or sender_id.get("user_id")
        or sender.get("open_id")
        or operator_id.get("open_id")
        or operator.get("open_id")
        or user_id
        or ""
    ).strip()


def _extract_feishu_text_content_from_raw_content(raw_content: Any) -> str:
    payload = _safe_json_loads(raw_content, {})
    if isinstance(payload, dict):
        return str(payload.get("text") or "").strip()
    return str(raw_content or "").strip()


def _extract_leading_command_text(text: str) -> str:
    normalized = re.sub(r"<at\b[^>]*>.*?</at>", " ", str(text or ""), flags=re.I | re.S)
    return re.sub(r"\s+", " ", normalized).strip()


def _resolve_inline_fast_command(command_text: str) -> str | None:
    text = _extract_leading_command_text(command_text)
    if not text.startswith("/"):
        return None
    token = text[1:].split()[0].strip().lower()
    return INLINE_FAST_COMMAND_CANONICALS.get(token)


def _extract_feishu_inline_fast_command(payload: dict[str, Any]) -> str | None:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    if str(message.get("message_type") or "").strip().lower() != "text":
        return None
    return _resolve_inline_fast_command(_extract_feishu_text_content_from_raw_content(message.get("content")))


def _extract_feishu_trace_token(payload: dict[str, Any]) -> str | None:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    text = _extract_feishu_text_content_from_raw_content(message.get("content"))
    match = _FEISHU_TRACE_TOKEN_RE.search(text)
    return match.group(1) if match else None


def _classify_feishu_chat_lane(payload: dict[str, Any]) -> str:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    message_type = str(message.get("message_type") or "").strip().lower()
    if not message_type:
        return "chat_light"
    if message_type != "text":
        return "chat_heavy"
    text = _extract_feishu_text_content_from_raw_content(message.get("content"))
    return "chat_light" if len(text) <= 1600 else "chat_heavy"


def _infer_feishu_internal_route_hint(task: dict[str, Any]) -> str:
    text = str(task.get("text") or "").strip()
    if str(task.get("task_kind") or "").strip().lower() == "command" or text.startswith("/"):
        return "fast_control"
    if "http://" in text.lower() or "https://" in text.lower():
        return "cf_browser_first"
    return "modal_heavy_exec"


def _extract_feishu_trace_context(payload: dict[str, Any]) -> dict[str, Any]:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    gateway_meta = payload.get("_hermes_gateway_meta") if isinstance(payload.get("_hermes_gateway_meta"), dict) else {}
    ingress_meta = payload.get("_hermes_ingress") if isinstance(payload.get("_hermes_ingress"), dict) else {}
    site_prefetch = payload.get("site_prefetch") if isinstance(payload.get("site_prefetch"), dict) else {}
    if not site_prefetch and isinstance(ingress_meta.get("site_prefetch"), dict):
        site_prefetch = dict(ingress_meta.get("site_prefetch") or {})
    message_read_info = _extract_feishu_message_read_event_info(payload)
    message_id = str(
        payload.get("message_id")
        or message.get("message_id")
        or (message_read_info.get("message_id_list") or [""])[0]
        or ""
    ).strip()
    result = {
        "correlation_id": str(payload.get("correlation_id") or "").strip(),
        "session_key": str(payload.get("session_key") or "").strip(),
        "event_type": str(payload.get("event_type") or (_extract_feishu_event_metadata(payload)[1] or "")).strip(),
        "event_id": str(payload.get("event_id") or (_extract_feishu_event_metadata(payload)[0] or "")).strip(),
        "message_id": message_id,
        "chat_id": str(payload.get("chat_id") or _collect_feishu_chat_id(payload)).strip(),
        "route_hint": str(payload.get("route_hint") or "").strip(),
        "request_class": str(payload.get("request_class") or "").strip(),
        "route_family": str(payload.get("route_family") or "").strip(),
        "gateway_route_name": str(payload.get("gateway_route_name") or "").strip(),
        "site_category": str(payload.get("site_category") or "").strip(),
        "site_intent": str(payload.get("site_intent") or "").strip(),
        "site_prefetch_mode": str(site_prefetch.get("mode") or "").strip(),
        "site_prefetch_status": str(site_prefetch.get("status") or "").strip(),
        "site_prefetch_confidence": site_prefetch.get("confidence"),
        "site_prefetch_direct_navigation": bool(payload.get("site_prefetch_direct_navigation", False)),
        "site_prefetch_direct_mode": str(payload.get("site_prefetch_direct_mode") or "").strip(),
        "browser_target_domain": str(payload.get("browser_target_domain") or "").strip(),
        "estimated_cost_usd": payload.get("estimated_cost_usd"),
        "trace_token": _extract_feishu_trace_token(payload),
    }
    for key in (
        "route_version",
        "provider_alias",
        "cache_eligible",
        "cache_status",
        "capability_match",
        "preferred_model_selected",
        "model_catalog_version",
        "feedback_score_before",
        "feedback_score_after",
        "gateway_error_class",
        "misroute_detected",
    ):
        if key in gateway_meta:
            result[key] = gateway_meta.get(key)
    if payload.get("fallback_reason") is not None:
        result["fallback_reason"] = payload.get("fallback_reason")
    return result


def _build_feishu_snapshot_profile_state() -> dict[str, Any]:
    from internal.modal_observability import build_feishu_snapshot_profile_state

    return build_feishu_snapshot_profile_state(
        app_name=APP_NAME,
        web_app_memory_snapshot_enabled=WEB_APP_MEMORY_SNAPSHOT_ENABLED,
        feishu_ingress_memory_snapshot_enabled=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED,
        feishu_ack_reaction_memory_snapshot_enabled=FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED,
        chat_queue_memory_snapshot_enabled=CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED,
    )


def _with_feishu_internal_meta(payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
    result = dict(payload or {})
    meta = result.get("_hermes_internal_meta") if isinstance(result.get("_hermes_internal_meta"), dict) else {}
    meta = dict(meta)
    meta.update({key: value for key, value in extra.items() if value is not None})
    result["_hermes_internal_meta"] = meta
    return result


def _resolve_feishu_request_started_at_ms(payload: dict[str, Any], request_started_at_ms: int) -> int:
    normalized = int(request_started_at_ms or 0)
    if normalized >= 10**12:
        return normalized
    meta = payload.get("_hermes_internal_meta") if isinstance(payload.get("_hermes_internal_meta"), dict) else {}
    return int(meta.get("ack_reaction_requested_at_ms") or meta.get("handoff_requested_at_ms") or 0)


def _extract_feishu_internal_request_meta(headers: dict[str, Any] | None) -> dict[str, str]:
    payload = headers if isinstance(headers, dict) else {}
    return {
        "gateway_hop": str(payload.get("x-hermes-gateway-hop") or "").strip(),
        "gateway_script": str(payload.get("x-hermes-gateway-script") or "").strip(),
        "gateway_correlation_id": str(payload.get("x-hermes-correlation-id") or "").strip(),
        "gateway_event_id": str(payload.get("x-hermes-event-id") or "").strip(),
        "gateway_session_key": str(payload.get("x-hermes-session-key") or "").strip(),
    }


def _log_feishu_internal_auth_failure(
    *,
    endpoint: str,
    authorization: str | None,
    expected_token: str | None,
    headers: dict[str, Any] | None,
) -> None:
    from internal.feishu_webhook_support import log_feishu_internal_auth_failure

    log_feishu_internal_auth_failure(
        endpoint=endpoint,
        authorization=authorization,
        expected_token=expected_token,
        headers=headers,
        extract_feishu_internal_request_meta=_extract_feishu_internal_request_meta,
        secret_fingerprint=lambda value: _mask_secret(value) or "",
        logger=logger,
    )


_configure_runtime_settings_hooks()


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
    use_modal_state = _path_is_modal_managed(FEISHU_TRACE_PATH)
    if use_modal_state:
        _reload_modal_volume()
    _ensure_runtime_dirs()
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timestamp_ms": _now_ms(),
        "stage": stage,
        **_extract_feishu_trace_context(payload),
        **_build_feishu_snapshot_profile_state(),
    }
    row["app_name"] = APP_NAME
    row.update({key: value for key, value in extra.items() if value is not None})
    with _FEISHU_EVENT_LOCK:
        with FEISHU_TRACE_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    if use_modal_state and modal_debug_store is not None:
        try:
            modal_debug_store.put(f"trace:{row['timestamp_ms']}:{uuid.uuid4().hex}", row)
        except Exception:
            pass
    if use_modal_state:
        _commit_modal_volume()


def _read_feishu_trace(limit: int = 100) -> list[dict[str, Any]]:
    use_modal_state = _path_is_modal_managed(FEISHU_TRACE_PATH)
    if use_modal_state:
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
    if use_modal_state and modal_debug_store is not None:
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


def _session_file(session_key: str) -> Path:
    from internal.session_routes import session_file

    return session_file(session_key, sessions_dir=SESSIONS_DIR)


def _load_session_state(session_key: str) -> dict[str, Any]:
    from internal.session_routes import load_session_state

    return load_session_state(
        session_key,
        load_json_file=_load_json_file,
        session_file_fn=_session_file,
    )


def _save_session_state(
    session_key: str,
    session_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    *,
    route_lease: dict[str, Any] | None = None,
    route_debug: dict[str, Any] | None = None,
    route_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if session_id is not None or messages is not None or route_lease is not None or route_debug is not None or route_metrics is not None:
        from internal.session_routes import save_session_state

        save_session_state(
            session_key,
            str(session_id or str(uuid.uuid4())),
            list(messages or []),
            route_lease=route_lease,
            route_debug=route_debug,
            route_metrics=route_metrics,
            atomic_json_write=_atomic_json_write,
            session_file_fn=_session_file,
        )
        return _load_session_state(session_key)

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


def _mark_update_seen(update_id: str) -> bool:
    normalized = str(update_id or "").strip()
    if not normalized:
        return False
    payload = _load_json_file(UPDATES_PATH, {})
    payload = payload if isinstance(payload, dict) else {}
    if normalized in payload:
        return False
    payload[normalized] = int(time.time())
    _atomic_json_write(UPDATES_PATH, payload)
    return True


def _mark_feishu_event_seen(event_id: str) -> bool:
    normalized = str(event_id or "").strip()
    if not normalized:
        return False
    payload = _load_json_file(FEISHU_EVENTS_PATH, {})
    payload = payload if isinstance(payload, dict) else {}
    if normalized in payload:
        return False
    payload[normalized] = int(time.time())
    _atomic_json_write(FEISHU_EVENTS_PATH, payload)
    return True


def _extract_telegram_inline_fast_command(update: dict[str, Any]) -> str | None:
    from internal.feishu.ingress import extract_telegram_inline_fast_command

    return extract_telegram_inline_fast_command(update, set(INLINE_FAST_COMMAND_CANONICALS.values()))


def _extract_telegram_queue_context(update: dict[str, Any]) -> dict[str, str]:
    from internal.chat_queue_enqueue import extract_telegram_queue_context

    return extract_telegram_queue_context(update)


def _enqueue_chat_event(
    *,
    platform: str,
    partition: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    from internal.chat_queue_enqueue import enqueue_chat_event

    return enqueue_chat_event(
        platform=platform,
        partition=partition,
        payload=payload,
        metadata=metadata,
        get_chat_queue=_get_chat_queue,
        safe_chat_queue_depth=_safe_chat_queue_depth,
        default_chat_queue_claim_ttl_seconds=DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
    )


async def _enqueue_chat_event_async(
    *,
    platform: str,
    partition: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    include_queue_depth: bool = True,
) -> dict[str, Any]:
    from internal.chat_queue_enqueue import enqueue_chat_event_async

    return await enqueue_chat_event_async(
        platform=platform,
        partition=partition,
        payload=payload,
        metadata=metadata,
        include_queue_depth=include_queue_depth,
        get_chat_queue=_get_chat_queue,
        safe_chat_queue_depth_async=_safe_chat_queue_depth_async,
        enqueue_chat_event_fn=_enqueue_chat_event,
        default_chat_queue_claim_ttl_seconds=DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
    )


def _route_api_key_source(provider_name: str, base_url: str = "") -> str:
    from internal.session_routes import route_api_key_source

    return route_api_key_source(provider_name, base_url)


def _build_route_lease(
    route: dict[str, Any],
    *,
    selection_reason: str,
    selected_at: int | None = None,
    last_success_at: int | None = None,
    fail_count: int = 0,
    lease_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    from internal.session_routes import build_route_lease

    return build_route_lease(
        route,
        selection_reason=selection_reason,
        selected_at=selected_at,
        last_success_at=last_success_at,
        fail_count=fail_count,
        lease_ttl_seconds=lease_ttl_seconds,
        route_api_key_source_fn=_route_api_key_source,
        default_session_route_ttl_seconds=DEFAULT_SESSION_ROUTE_TTL_SECONDS,
    )


def _resolve_provider_runtime_binding(provider_name: str) -> dict[str, Any] | None:
    from internal.model_catalog import resolve_provider_runtime_binding

    return resolve_provider_runtime_binding(provider_name)


def _hydrate_route_from_lease(settings: Any, lease: dict[str, Any] | None) -> dict[str, Any] | None:
    from internal.session_routes import hydrate_route_from_lease

    return hydrate_route_from_lease(
        settings,
        lease,
        resolve_provider_runtime_binding_fn=_resolve_provider_runtime_binding,
    )


def _resolve_runtime_model_name(model_name: str, provider_name: str) -> str:
    normalized_model = str(model_name or "").strip()
    if str(provider_name or "").strip().lower() != "openrouter":
        return normalized_model
    return re.sub(r"-(20\d{6})$", "", normalized_model)


def _route_from_settings(settings: Any, model_name: str | None = None) -> dict[str, Any]:
    from internal.session_routes import route_from_settings

    return route_from_settings(
        settings,
        model_name,
        default_openrouter_base_url="https://openrouter.ai/api/v1",
        resolve_provider_runtime_binding_fn=_resolve_provider_runtime_binding,
    )


def _load_routing_state() -> dict[str, Any]:
    payload = _load_json_file(ROUTING_STATE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_routing_state(payload: dict[str, Any]) -> None:
    _atomic_json_write(ROUTING_STATE_PATH, payload)


def _candidate_routes_from_state(
    state: dict[str, Any],
    *,
    preferred_provider: str | None = None,
    failed_route: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from internal.session_routes import candidate_routes_from_state

    return candidate_routes_from_state(
        state,
        preferred_provider=preferred_provider,
        failed_route=failed_route,
        resolve_provider_runtime_binding_fn=_resolve_provider_runtime_binding,
    )


def _refresh_free_model_routes(force: bool = False) -> dict[str, Any]:
    from internal.model_catalog import (
        DEFAULT_OPENROUTER_FREE_MODELS,
        dedupe_keep_order,
        expand_nvidia_model_variants,
        extract_nvidia_popular_model_candidates,
        extract_openrouter_free_model_candidates,
        fetch_nvidia_popular_model_candidates,
        fetch_openrouter_free_model_candidates,
        is_probable_nvidia_chat_model,
        load_nvidia_free_model_candidates,
        refresh_free_model_routes,
        score_free_model,
    )

    return refresh_free_model_routes(
        force=force,
        ensure_runtime_dirs=_ensure_runtime_dirs,
        load_routing_state=_load_routing_state,
        save_routing_state=_save_routing_state,
        resolve_provider_runtime_binding_fn=_resolve_provider_runtime_binding,
        fetch_openrouter_free_model_candidates_fn=lambda: fetch_openrouter_free_model_candidates(
            default_openrouter_base_url="https://openrouter.ai/api/v1",
            extract_openrouter_free_model_candidates_fn=lambda payload: extract_openrouter_free_model_candidates(
                payload,
                dedupe_keep_order_fn=dedupe_keep_order,
                score_free_model_fn=score_free_model,
            ),
        ),
        load_nvidia_free_model_candidates_fn=lambda: load_nvidia_free_model_candidates(
            split_csv=_split_csv,
            dedupe_keep_order_fn=dedupe_keep_order,
            is_truthy=_is_truthy,
            fetch_nvidia_popular_model_candidates_fn=lambda: fetch_nvidia_popular_model_candidates(
                default_nvidia_popular_models_url="https://build.nvidia.com/explore/discover",
                extract_nvidia_popular_model_candidates_fn=lambda html: extract_nvidia_popular_model_candidates(
                    html,
                    is_probable_nvidia_chat_model_fn=lambda model: is_probable_nvidia_chat_model(
                        model,
                        non_chat_markers=("embed", "rerank", "asr", "tts", "vision-language"),
                        chat_family_markers=("llama", "kimi", "mixtral", "nemotron", "deepseek", "qwen"),
                    ),
                    expand_nvidia_model_variants_fn=lambda model: expand_nvidia_model_variants(
                        model,
                        dedupe_keep_order_fn=dedupe_keep_order,
                    ),
                    dedupe_keep_order_fn=dedupe_keep_order,
                ),
            ),
            default_nvidia_free_models=("meta/llama-3.1-8b-instruct",),
            logger=logger,
        ),
        routing_refresh_ttl_seconds=3600,
        default_openrouter_base_url="https://openrouter.ai/api/v1",
        default_nvidia_base_url=DEFAULT_NVIDIA_BASE_URL,
        logger=logger,
    )


def _select_dynamic_primary_route(force_refresh: bool = False) -> dict[str, Any] | None:
    from internal.session_routes import select_dynamic_primary_route

    return select_dynamic_primary_route(
        force_refresh=force_refresh,
        refresh_free_model_routes_fn=_refresh_free_model_routes,
        candidate_routes_from_state_fn=_candidate_routes_from_state,
    )


def _select_dynamic_fallback_routes(primary_route: dict[str, Any] | None) -> list[dict[str, Any]]:
    from internal.session_routes import select_dynamic_fallback_routes

    return select_dynamic_fallback_routes(
        primary_route,
        refresh_free_model_routes_fn=_refresh_free_model_routes,
        candidate_routes_from_state_fn=_candidate_routes_from_state,
    )


def _materialize_dynamic_free_model_config(config_payload: dict[str, Any]) -> None:
    from internal.session_routes import (
        materialize_dynamic_free_model_config,
        route_to_fallback_provider,
        route_to_runtime_model_config,
    )

    materialize_dynamic_free_model_config(
        config_payload,
        is_dynamic_free_route_alias_fn=lambda value: str(value or "").strip().lower() in {"free", "openrouter/free"},
        select_dynamic_primary_route_fn=_select_dynamic_primary_route,
        route_to_runtime_model_config_fn=route_to_runtime_model_config,
        select_dynamic_fallback_routes_fn=_select_dynamic_fallback_routes,
        route_to_fallback_provider_fn=route_to_fallback_provider,
    )


def _refresh_and_select_valid_route(
    *,
    preferred_provider: str | None = None,
    failed_route: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    from internal.session_routes import refresh_and_select_valid_route

    return refresh_and_select_valid_route(
        preferred_provider=preferred_provider,
        failed_route=failed_route,
        refresh_free_model_routes_fn=_refresh_free_model_routes,
        candidate_routes_from_state_fn=_candidate_routes_from_state,
        probe_model_route_fn=lambda route: True,
        logger=logger,
    )


def _resolve_primary_route(settings: Any, model_name: str | None = None) -> dict[str, Any]:
    from internal.session_routes import resolve_primary_route

    return resolve_primary_route(
        settings,
        model_name,
        route_from_settings_fn=_route_from_settings,
        is_dynamic_free_route_alias_fn=lambda value: str(value or "").strip().lower() in {"free", "openrouter/free"},
        select_dynamic_primary_route_fn=_select_dynamic_primary_route,
    )


def _determine_route_refresh_reason(result: dict[str, Any]) -> str | None:
    from internal.session_routes import determine_route_refresh_reason

    return determine_route_refresh_reason(
        result,
        is_invalid_model_error_text_fn=lambda text: "valid model id" in str(text or "").lower(),
        is_transient_route_error_text_fn=lambda text: any(
            marker in str(text or "").lower() for marker in ("timeout", "ratelimit", "readtimeout", "temporarily unavailable")
        ),
    )


def _extract_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages or []:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
            name = str(function.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _run_agent_task_impl(
    task_input: str,
    *,
    session_key: str = "",
    model_name: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    from run_agent import AIAgent

    normalized_session_key = str(session_key or f"session:{uuid.uuid4().hex[:8]}").strip()
    _prepare_runtime_environment()
    settings = RuntimeSettings.from_env()
    persisted = _load_session_state(normalized_session_key)
    route_selection = "fresh_select"
    route = None

    if model_name:
        route_selection = "explicit_override"
        route = _resolve_primary_route(settings, model_name=model_name)
    else:
        route = _hydrate_route_from_lease(settings, persisted.get("route_lease"))
        if route is not None:
            route_selection = "sticky_hit"
        else:
            route = _route_from_settings(settings, model_name=None)

    runtime_model = _resolve_runtime_model_name(str(route.get("model") or settings.model), str(route.get("provider") or settings.provider or ""))
    trace_metadata = {
        "route_selection": route_selection,
        "provider": route.get("provider"),
        "model": runtime_model,
    }

    def _build_agent(route_payload: dict[str, Any]) -> Any:
        return AIAgent(
            model=_resolve_runtime_model_name(route_payload["model"], route_payload["provider"]),
            api_key=route_payload.get("api_key"),
            base_url=route_payload.get("base_url"),
            provider=route_payload.get("provider"),
            max_iterations=int(settings.max_turns or 16),
            max_tokens=max_tokens or settings.max_tokens or 0,
            quiet_mode=True,
            verbose_logging=False,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            session_id=str(persisted.get("session_id") or uuid.uuid4().hex),
            trace_session_key=normalized_session_key,
            trace_metadata=trace_metadata,
            platform="modal",
            user_id="local",
        )

    agent = _build_agent(route)
    result = agent.run_conversation(task_input, task_id=f"modal:{normalized_session_key}")
    retried_after_model_refresh = False
    refreshed_model = None

    if result.get("error"):
        failed_provider = str(route.get("provider") or "").strip().lower()
        preferred_provider = None
        if failed_provider == "openrouter" and str(os.getenv("NVIDIA_API_KEY") or "").strip():
            preferred_provider = "nvidia"
        elif failed_provider == "nvidia" and str(os.getenv("OPENROUTER_API_KEY") or "").strip():
            preferred_provider = "openrouter"
        retry_route = _refresh_and_select_valid_route(
            preferred_provider=preferred_provider,
            failed_route=route,
        )
        if retry_route is not None:
            retried_after_model_refresh = True
            refreshed_model = retry_route.get("model")
            refreshed_provider = retry_route.get("provider")
            trace_metadata["route_selection"] = "refreshed_after_failure"
            route = retry_route
            agent = _build_agent(route)
            result = agent.run_conversation(task_input, task_id=f"modal:{normalized_session_key}:retry")
        else:
            refreshed_provider = None
    else:
        refreshed_provider = None

    output = str(result.get("final_response") or "").strip()
    status = "success" if output and not result.get("error") else "error"
    provider_usage = result.get("provider_usage") or {}
    response_model = str(provider_usage.get("response_model") or route.get("model") or "").strip()
    if response_model:
        route = dict(route)
        route["model"] = response_model
    route_debug = {
        "last_route_selection": route_selection if not retried_after_model_refresh else "refreshed_after_failure",
        "last_error": str(result.get("error") or "").strip(),
    }
    route_metrics = dict(persisted.get("route_metrics") or {})
    route_metrics[route_selection if not retried_after_model_refresh else "refreshed_after_failure"] = int(
        route_metrics.get(route_selection if not retried_after_model_refresh else "refreshed_after_failure") or 0
    ) + 1
    _save_session_state(
        normalized_session_key,
        str(persisted.get("session_id") or uuid.uuid4().hex),
        list(result.get("messages") or []),
        route_lease=_build_route_lease(route, selection_reason=route_selection if not retried_after_model_refresh else "refreshed_after_failure"),
        route_debug=route_debug,
        route_metrics=route_metrics,
    )
    return {
        "status": status,
        "output": output,
        "route_selection": route_selection if not retried_after_model_refresh else "refreshed_after_failure",
        "retried_after_model_refresh": retried_after_model_refresh,
        "refreshed_model": refreshed_model,
        "refreshed_provider": refreshed_provider,
        "tool_summary": _extract_tool_names(list(result.get("messages") or [])),
        "provider_usage": provider_usage,
        "provider_usage_totals": result.get("provider_usage_totals") or {},
    }


class _InMemoryQueue:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any], *, partition: str | None = None, partition_ttl: int | None = None) -> None:
        self._items.append(dict(item))

    def get_many(self, max_items: int, block: bool = False, timeout: float | None = None, partition: str | None = None) -> list[dict[str, Any]]:
        items = []
        remaining: list[dict[str, Any]] = []
        for item in self._items:
            if partition and str(item.get("partition") or "").strip() != partition:
                remaining.append(item)
                continue
            if len(items) < max(int(max_items or 0), 0):
                items.append(item)
            else:
                remaining.append(item)
        self._items = remaining
        return items

    def len(self) -> int:
        return len(self._items)


CHAT_QUEUE: Any = _InMemoryQueue()


def _sync_modal_volume(*, reload: bool = False, commit: bool = False) -> None:
    from internal.modal_infra import sync_modal_volume

    sync_modal_volume(
        volume=modal_volume,
        reload=reload,
        commit=commit,
        logger=logger,
        warning_prefix="Modal volume sync failed",
    )


async def _sync_modal_volume_async(*, reload: bool = False, commit: bool = False) -> None:
    from internal.modal_infra import sync_modal_volume_async

    await sync_modal_volume_async(
        volume=modal_volume,
        reload=reload,
        commit=commit,
        logger=logger,
        warning_prefix="Modal volume async sync failed",
    )


def _get_chat_queue() -> Any:
    return CHAT_QUEUE


def _safe_chat_queue_depth() -> int | None:
    from internal.modal_infra import safe_queue_depth

    return safe_queue_depth(
        get_queue=_get_chat_queue,
        logger=logger,
        warning_prefix="Unable to read Modal chat queue depth",
    )


async def _safe_chat_queue_depth_async() -> int | None:
    from internal.modal_infra import safe_queue_depth_async

    return await safe_queue_depth_async(
        queue=_get_chat_queue(),
        logger=logger,
        warning_prefix="Unable to read Modal chat queue depth",
    )


def _should_reload_modal_volume_for_claims(kind: str) -> bool:
    return _is_truthy(os.getenv(f"HERMES_MODAL_{str(kind or '').strip().upper()}_CLAIMS_RELOAD"), default=False)


def _load_chat_queue_claims() -> dict[str, Any]:
    from internal.chat_claims import load_claims

    return load_claims(load_json_file=_load_json_file, path=CHAT_QUEUE_CLAIMS_PATH)


def _save_chat_queue_claims(payload: dict[str, Any]) -> None:
    from internal.chat_claims import save_claims

    save_claims(payload, atomic_json_write=_atomic_json_write, path=CHAT_QUEUE_CLAIMS_PATH)


def _prune_chat_queue_claims(
    claims: dict[str, Any],
    ttl_seconds: int = DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
) -> dict[str, Any]:
    from internal.chat_claims import prune_claims

    return prune_claims(claims, ttl_seconds=ttl_seconds)


def _prune_chat_claims_for_ttl(claims: dict[str, Any], ttl_seconds: int | None = None) -> dict[str, Any]:
    from internal.chat_claims import prune_claims

    return prune_claims(claims, ttl_seconds=DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS if ttl_seconds is None else ttl_seconds)


def _claim_age_seconds(claim: dict[str, Any]) -> int:
    from internal.chat_claims import claim_age_seconds

    return claim_age_seconds(claim)


def _claim_is_recent(claim: dict[str, Any], *, max_age_seconds: int) -> bool:
    from internal.chat_claims import claim_is_recent

    return claim_is_recent(claim, max_age_seconds=max_age_seconds)


def _claim_is_stale_for_takeover(claim: dict[str, Any], *, stale_after_seconds: int | None = None) -> bool:
    from internal.chat_claims import claim_is_stale_for_takeover

    return claim_is_stale_for_takeover(
        claim,
        stale_after_seconds=stale_after_seconds,
        default_stale_after_seconds=DEFAULT_CHAT_QUEUE_STALE_CLAIM_TAKEOVER_SECONDS,
        claim_age_seconds=_claim_age_seconds,
    )


def _claim_chat_partition(
    partition_key: str,
    *,
    platform: str,
    claim_token: str | None = None,
    ttl_seconds: int | None = None,
) -> tuple[bool, str]:
    from internal.chat_claims import claim_partition

    return claim_partition(
        partition_key,
        platform=platform,
        claim_token=claim_token,
        ttl_seconds=DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS if ttl_seconds is None else ttl_seconds,
        lock=_CHAT_QUEUE_LOCK,
        should_reload_modal_volume_for_claims=_should_reload_modal_volume_for_claims,
        sync_modal_volume=_sync_modal_volume,
        prune_claims=lambda claims, ttl: _prune_chat_claims_for_ttl(claims, ttl),
        load_claims=_load_chat_queue_claims,
        save_claims=_save_chat_queue_claims,
        claim_is_stale_for_takeover=_claim_is_stale_for_takeover,
        claim_age_seconds=_claim_age_seconds,
        logger=logger,
    )


def _release_chat_partition_claim(partition_key: str, *, claim_token: str | None = None) -> None:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return
    with _CHAT_QUEUE_LOCK:
        claims = _prune_chat_claims_for_ttl(_load_chat_queue_claims())
        existing = claims.get(normalized) or {}
        if claim_token and str(existing.get("claim_token") or "").strip() != str(claim_token).strip():
            return
        claims.pop(normalized, None)
        _save_chat_queue_claims(claims)
        _sync_modal_volume(commit=True)


def _load_chat_queue_warmups() -> dict[str, Any]:
    from internal.chat_claims import load_warmups

    return load_warmups(load_json_file=_load_json_file, path=CHAT_QUEUE_WARMUPS_PATH)


def _save_chat_queue_warmups(payload: dict[str, Any]) -> None:
    from internal.chat_claims import save_warmups

    save_warmups(payload, atomic_json_write=_atomic_json_write, path=CHAT_QUEUE_WARMUPS_PATH)


def _prune_chat_queue_warmups(warmups: dict[str, Any], ttl_seconds: int = DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS) -> dict[str, Any]:
    from internal.chat_claims import prune_warmups

    return prune_warmups(warmups, ttl_seconds=ttl_seconds)


def _record_chat_queue_warmup_snapshot(
    partition_key: str,
    *,
    platform: str,
    status: str,
    metadata: dict[str, Any] | None = None,
    worker_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from internal.chat_claims import record_warmup_snapshot

    def _first_non_empty_str(*values: Any) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    return record_warmup_snapshot(
        partition_key,
        platform=platform,
        status=status,
        metadata=metadata,
        worker_context=worker_context,
        lock=_CHAT_QUEUE_LOCK,
        load_warmups=_load_chat_queue_warmups,
        save_warmups=_save_chat_queue_warmups,
        prune_warmups=_prune_chat_queue_warmups,
        sync_modal_volume=_sync_modal_volume,
        first_non_empty_str=_first_non_empty_str,
    )


def _pop_chat_queue_warmup_snapshot(partition_key: str) -> dict[str, Any]:
    from internal.chat_claims import pop_warmup_snapshot

    return pop_warmup_snapshot(
        partition_key,
        lock=_CHAT_QUEUE_LOCK,
        load_warmups=_load_chat_queue_warmups,
        save_warmups=_save_chat_queue_warmups,
        prune_warmups=_prune_chat_queue_warmups,
        sync_modal_volume=_sync_modal_volume,
    )


def _extract_feishu_message_read_event_info(payload: dict[str, Any]) -> dict[str, Any]:
    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    reader = event.get("reader") if isinstance(event.get("reader"), dict) else {}
    reader_id = reader.get("reader_id") if isinstance(reader.get("reader_id"), dict) else {}
    message_ids = [str(item).strip() for item in (event.get("message_id_list") or []) if str(item).strip()]

    def _first_non_empty(*values: Any) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    return {
        "event_type": str(header.get("event_type") or payload.get("event_type") or "").strip(),
        "event_id": str(header.get("event_id") or payload.get("event_id") or "").strip(),
        "reader_open_id": _first_non_empty(reader_id.get("open_id"), reader.get("open_id"), event.get("open_id")),
        "reader_user_id": _first_non_empty(reader_id.get("user_id"), reader.get("user_id"), event.get("user_id")),
        "reader_union_id": _first_non_empty(reader_id.get("union_id"), reader.get("union_id")),
        "tenant_key": _first_non_empty(reader.get("tenant_key"), event.get("tenant_key")),
        "read_time": int(reader.get("read_time") or event.get("read_time") or 0),
        "message_id_list": message_ids,
        "message_count": len(message_ids),
    }


def _load_feishu_bot_identity_cache() -> dict[str, Any]:
    payload = _load_json_file(FEISHU_BOT_IDENTITY_CACHE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_feishu_bot_identity_cache(payload: dict[str, Any]) -> None:
    _atomic_json_write(FEISHU_BOT_IDENTITY_CACHE_PATH, payload)


def _get_cached_feishu_bot_identity(app_id: str | None) -> dict[str, Any]:
    normalized_app_id = str(app_id or "").strip()
    if not normalized_app_id:
        return {}
    with _FEISHU_BOT_IDENTITY_CACHE_LOCK:
        payload = _load_feishu_bot_identity_cache()
        cached = payload.get(normalized_app_id) or {}
        return dict(cached) if isinstance(cached, dict) else {}


async def _cache_feishu_bot_identity_async(app_id: str | None, *, bot_name: str | None = None) -> None:
    normalized_app_id = str(app_id or "").strip()
    normalized_bot_name = str(bot_name or "").strip()
    if not normalized_app_id or not normalized_bot_name:
        return
    should_commit = False
    with _FEISHU_BOT_IDENTITY_CACHE_LOCK:
        payload = _load_feishu_bot_identity_cache()
        existing = payload.get(normalized_app_id) or {}
        existing_name = str(existing.get("bot_name") or "").strip() if isinstance(existing, dict) else ""
        if existing_name != normalized_bot_name:
            payload[normalized_app_id] = {
                "bot_name": normalized_bot_name,
                "updated_at": int(time.time()),
            }
            _save_feishu_bot_identity_cache(payload)
            should_commit = True
    if should_commit:
        await _sync_modal_volume_async(commit=True)


def _extract_feishu_queue_context(payload: dict[str, Any]) -> dict[str, str]:
    from internal.chat_queue_enqueue import extract_feishu_queue_context

    return extract_feishu_queue_context(
        payload,
        extract_trace_context=_extract_feishu_trace_context,
        collect_chat_id=_collect_feishu_chat_id,
        collect_actor_id=_collect_feishu_actor_id,
        classify_chat_lane=_classify_feishu_chat_lane,
    )


def _extract_feishu_warmup_context(payload: dict[str, Any]) -> dict[str, str]:
    from internal.chat_queue_spawn import extract_feishu_warmup_context

    return extract_feishu_warmup_context(payload, extract_feishu_queue_context=_extract_feishu_queue_context)


def _build_chat_worker_observability(worker_context: dict[str, Any] | None = None, *, batch_size: int | None = None) -> dict[str, Any]:
    from internal.modal_observability import build_chat_worker_observability

    return build_chat_worker_observability(worker_context, batch_size=batch_size)


def _build_inline_chat_worker_context() -> dict[str, Any]:
    return {
        "worker_boot_id": f"inline-{uuid.uuid4().hex[:10]}",
        "worker_started_at": _now_ms(),
        "enter_elapsed_ms": 0,
        "runtime_prepare_elapsed_ms": 0,
        "container_reused": False,
    }


def _has_recent_chat_worker_spawn(partition_key: str) -> bool:
    from internal.chat_claims import has_recent_worker_spawn

    return has_recent_worker_spawn(
        partition_key,
        ttl_seconds=float(DEFAULT_FEISHU_RECENT_SPAWN_SKIP_SECONDS),
        recent_worker_spawns=_RECENT_CHAT_WORKER_SPAWNS,
        recent_worker_spawns_lock=_RECENT_CHAT_WORKER_SPAWNS_LOCK,
    )


def _mark_recent_chat_worker_spawn(partition_key: str) -> None:
    from internal.chat_claims import mark_recent_worker_spawn

    mark_recent_worker_spawn(
        partition_key,
        recent_worker_spawns=_RECENT_CHAT_WORKER_SPAWNS,
        recent_worker_spawns_lock=_RECENT_CHAT_WORKER_SPAWNS_LOCK,
    )


async def _peek_chat_partition_claim_async(
    partition_key: str,
    *,
    refresh_on_miss: bool = False,
) -> dict[str, Any]:
    from internal.chat_claims import peek_claim_async

    return await peek_claim_async(
        partition_key,
        ttl_seconds=DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
        refresh_on_miss=refresh_on_miss,
        lock=_CHAT_QUEUE_LOCK,
        should_reload_modal_volume_for_claims=_should_reload_modal_volume_for_claims,
        sync_modal_volume_async=_sync_modal_volume_async,
        prune_claims=lambda claims, ttl_seconds: _prune_chat_queue_claims(claims, ttl_seconds),
        load_claims=_load_chat_queue_claims,
    )


def _get_process_chat_queue_handle(partition: str | None = None) -> Any:
    from internal.chat_queue_worker import get_process_chat_queue_handle

    return get_process_chat_queue_handle(
        partition=partition,
        default_handle=globals().get("process_chat_queue"),
        worker_cls=globals().get("ChatQueueWorker"),
        logger=logger,
    )


def _schedule_chat_partition_worker(partition_key: str, *, platform: str) -> dict[str, Any]:
    from internal.chat_claims import schedule_partition_worker

    return schedule_partition_worker(
        partition_key,
        platform=platform,
        cooldown_seconds=DEFAULT_FEISHU_RECENT_SPAWN_SKIP_SECONDS,
        lock=_CHAT_QUEUE_LOCK,
        should_reload_modal_volume_for_claims=_should_reload_modal_volume_for_claims,
        sync_modal_volume=_sync_modal_volume,
        prune_claims=lambda claims, ttl_seconds: _prune_chat_queue_claims(claims, ttl_seconds),
        load_claims=_load_chat_queue_claims,
        save_claims=_save_chat_queue_claims,
        claim_is_stale_for_takeover=lambda claim: _claim_is_stale_for_takeover(claim),
        claim_age_seconds=_claim_age_seconds,
        logger=logger,
        claim_ttl_seconds=DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
    )


async def _schedule_chat_partition_worker_async(partition_key: str, *, platform: str) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    normalized_platform = str(platform or "").strip().lower()
    if not normalized:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": normalized_platform,
            "partition": normalized,
        }
    token = f"{normalized_platform}:{normalized}:{int(time.time())}:{uuid.uuid4().hex[:8]}"
    async with asyncio.Lock():
        claims = _prune_chat_queue_claims(_load_chat_queue_claims())
        existing = claims.get(normalized) or {}
        if existing and not _claim_is_stale_for_takeover(existing):
            return {
                "status": "skipped",
                "reason": str(existing.get("status") or "already_scheduled"),
                "platform": normalized_platform,
                "partition": normalized,
                "claim_token": str(existing.get("claim_token") or ""),
            }
        claims[normalized] = {
            "claim_token": token,
            "claimed_at": int(time.time()),
            "platform": normalized_platform,
            "status": "scheduled",
            "cooldown_seconds": max(int(DEFAULT_FEISHU_RECENT_SPAWN_SKIP_SECONDS), 1),
        }
        _save_chat_queue_claims(claims)
        await _sync_modal_volume_async(commit=True)
    return {
        "status": "scheduled",
        "reason": "scheduled",
        "platform": normalized_platform,
        "partition": normalized,
        "claim_token": token,
    }


async def _release_chat_partition_claim_async(partition_key: str, *, claim_token: str | None = None) -> None:
    await asyncio.to_thread(_release_chat_partition_claim, partition_key, claim_token=claim_token)


def _process_chat_queue_item(
    payload: Any,
    *,
    worker_context: dict[str, Any] | None = None,
    runtime_prepared: bool = False,
) -> dict[str, Any]:
    return asyncio.run(
        _process_chat_queue_item_async(
            payload,
            worker_context=worker_context,
            runtime_prepared=runtime_prepared,
        )
    )


async def _process_chat_queue_item_async(
    payload: Any,
    *,
    worker_context: dict[str, Any] | None = None,
    runtime_prepared: bool = False,
) -> dict[str, Any]:
    from internal.chat_queue import process_chat_queue_item_async

    return await process_chat_queue_item_async(
        payload,
        worker_context=worker_context,
        runtime_prepared=runtime_prepared,
        build_inline_chat_worker_context=_build_inline_chat_worker_context,
        is_truthy=_is_truthy,
        sync_modal_volume_async=_sync_modal_volume_async,
        build_chat_worker_observability=_build_chat_worker_observability,
        append_feishu_trace=_append_feishu_trace,
        dispatch_feishu_payload=lambda raw_payload, await_background_tasks: _dispatch_feishu_payload(
            raw_payload,
            await_background_tasks=await_background_tasks,
        ),
        dispatch_telegram_update=_dispatch_telegram_update,
        logger=logger,
        feishu_chat_worker_timeout_seconds=DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS,
    )


async def _process_chat_queue_items_async(
    items: list[Any],
    *,
    worker_context: dict[str, Any] | None = None,
    runtime_prepared: bool = False,
) -> list[dict[str, Any]]:
    from internal.chat_queue import process_chat_queue_items_async

    return await process_chat_queue_items_async(
        items,
        worker_context=worker_context,
        runtime_prepared=runtime_prepared,
        process_chat_queue_item_async_fn=_process_chat_queue_item_async,
        coalesce_items=_coalesce_feishu_chat_queue_items,
    )


def _process_chat_queue_impl(
    *,
    platform: str,
    partition: str,
    max_items: int | None = None,
    claim_token: str | None = None,
    worker_context: dict[str, Any] | None = None,
    runtime_prepared: bool = False,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from internal.chat_queue_worker import process_chat_queue

    return process_chat_queue(
        platform=platform,
        partition=partition,
        max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE if max_items is None else max_items,
        claim_token=claim_token,
        worker_context=worker_context,
        runtime_prepared=runtime_prepared,
        warmup_wait_seconds=warmup_wait_seconds,
        warmup_metadata=warmup_metadata,
        default_chat_queue_linger_seconds=DEFAULT_CHAT_QUEUE_LINGER_SECONDS,
        build_chat_worker_observability=_build_chat_worker_observability,
        build_inline_chat_worker_context=_build_inline_chat_worker_context,
        pop_chat_queue_warmup_snapshot=_pop_chat_queue_warmup_snapshot,
        first_non_empty_str=_first_non_empty_str,
        record_chat_queue_warmup_snapshot=_record_chat_queue_warmup_snapshot,
        safe_chat_queue_depth=_safe_chat_queue_depth,
        claim_chat_partition=_claim_chat_partition,
        get_chat_queue=_get_chat_queue,
        process_chat_queue_items_async=_process_chat_queue_items_async,
        sync_modal_volume=_sync_modal_volume,
        release_chat_partition_claim=_release_chat_partition_claim,
        logger=logger,
    )


def _resolve_feishu_message_ingress_strategy(payload: dict[str, Any], context: dict[str, Any]) -> str:
    from internal.chat_queue import resolve_feishu_message_ingress_strategy, should_spawn_feishu_message_inline

    env_value = os.getenv("HERMES_FEISHU_MESSAGE_INGRESS_STRATEGY")
    return resolve_feishu_message_ingress_strategy(
        payload,
        context,
        env_value=env_value,
        default_strategy=DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY,
        legacy_aliases=LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES,
        supported_strategies=SUPPORTED_FEISHU_MESSAGE_INGRESS_STRATEGIES,
        should_spawn_inline=should_spawn_feishu_message_inline,
    )


def _coalesce_feishu_chat_queue_items(items: list[Any]) -> list[Any]:
    from internal.chat_queue import coalesce_feishu_chat_queue_items

    return coalesce_feishu_chat_queue_items(
        items,
        now_ms=_now_ms(),
        max_age_ms=max(DEFAULT_FEISHU_MESSAGE_QUEUE_MAX_AGE_SECONDS, 1) * 1000,
        logger=logger,
    )


def _probe_provider_request_metadata_impl(
    *,
    provider_name: str,
    model: str = "",
    prompt: str = "ok",
    max_tokens: int = 64,
) -> dict[str, Any]:
    from internal.modal_diagnostics import probe_provider_request_metadata

    return probe_provider_request_metadata(
        provider_name=provider_name,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        prepare_runtime_environment=_prepare_runtime_environment,
        resolve_provider_runtime_binding=_resolve_provider_runtime_binding,
    )


def _spawn_chat_queue_worker_sync(
    *,
    platform: str,
    partition: str,
    max_items: int | None = None,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from internal.chat_queue_spawn import spawn_chat_queue_worker_sync

    return spawn_chat_queue_worker_sync(
        platform=platform,
        partition=partition,
        max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE if max_items is None else max_items,
        warmup_wait_seconds=warmup_wait_seconds,
        warmup_metadata=warmup_metadata,
        default_chat_queue_batch_size=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
        schedule_chat_partition_worker=_schedule_chat_partition_worker,
        get_process_chat_queue_handle=_get_process_chat_queue_handle,
        process_chat_queue_impl=_process_chat_queue_impl,
        release_chat_partition_claim=_release_chat_partition_claim,
    )


async def _spawn_chat_queue_worker_async(
    *,
    platform: str,
    partition: str,
    max_items: int | None = None,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from internal.chat_queue_spawn import spawn_chat_queue_worker_async

    return await spawn_chat_queue_worker_async(
        platform=platform,
        partition=partition,
        max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE if max_items is None else max_items,
        warmup_wait_seconds=warmup_wait_seconds,
        warmup_metadata=warmup_metadata,
        default_chat_queue_batch_size=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
        schedule_chat_partition_worker_async=_schedule_chat_partition_worker_async,
        get_process_chat_queue_handle=_get_process_chat_queue_handle,
        process_chat_queue_impl=_process_chat_queue_impl,
        release_chat_partition_claim_async=_release_chat_partition_claim_async,
    )


async def _spawn_chat_queue_worker_optimistic_async(
    *,
    platform: str,
    partition: str,
    max_items: int | None = None,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from internal.chat_queue_spawn import spawn_chat_queue_worker_optimistic_async

    return await spawn_chat_queue_worker_optimistic_async(
        platform=platform,
        partition=partition,
        max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE if max_items is None else max_items,
        warmup_wait_seconds=warmup_wait_seconds,
        warmup_metadata=warmup_metadata,
        default_chat_queue_batch_size=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
        has_recent_chat_worker_spawn=_has_recent_chat_worker_spawn,
        peek_chat_partition_claim_async=_peek_chat_partition_claim_async,
        claim_is_recent=_claim_is_recent,
        active_claim_skip_seconds=DEFAULT_CHAT_QUEUE_ACTIVE_CLAIM_SKIP_SECONDS,
        get_process_chat_queue_handle=_get_process_chat_queue_handle,
        mark_recent_chat_worker_spawn=_mark_recent_chat_worker_spawn,
        process_chat_queue_impl=_process_chat_queue_impl,
    )


def _bootstrap_chat_queue_worker_context(snapshot_context: dict[str, Any] | None = None) -> dict[str, Any]:
    from internal.chat_queue_worker import bootstrap_chat_queue_worker_context

    return bootstrap_chat_queue_worker_context(
        snapshot_context=snapshot_context,
        worker_partition_key=None,
        prepare_runtime_environment=_prepare_runtime_environment,
        load_routing_state=_load_routing_state,
        settings_from_env=RuntimeSettings.from_env,
        get_feishu_gateway_runtime=_get_feishu_gateway_runtime,
        chat_worker_preinit_feishu_runtime_enabled=CHAT_WORKER_PREINIT_FEISHU_RUNTIME_ENABLED,
        logger=logger,
    )


class _FeishuInternalSource:
    def __init__(self, *, chat_id: str, user_id: str, user_name: str) -> None:
        self.chat_id = chat_id
        self.user_id = user_id
        self.user_name = user_name


def _build_feishu_internal_source(payload: dict[str, Any]) -> _FeishuInternalSource:
    raw_message = payload.get("raw_message") if isinstance(payload.get("raw_message"), dict) else {}
    event = raw_message.get("event") if isinstance(raw_message.get("event"), dict) else {}
    context = event.get("context") if isinstance(event.get("context"), dict) else {}
    operator = event.get("operator") if isinstance(event.get("operator"), dict) else {}
    chat_id = str(context.get("open_chat_id") or event.get("open_chat_id") or "").strip()
    user_id = str(operator.get("open_id") or event.get("open_id") or "").strip()
    return _FeishuInternalSource(chat_id=chat_id, user_id=user_id, user_name=user_id)


def _build_feishu_menu_manifest() -> dict[str, Any]:
    return {
        "platform": "feishu",
        "supported_event_keys": [
            "model_picker",
            "personality_picker",
            "skill_combo_picker",
            "command_center",
        ],
        "menu_items": [
            {"event_key": "model_picker", "name": "Model Hub"},
            {"event_key": "personality_picker", "name": "Personality Picker"},
            {"event_key": "skill_combo_picker", "name": "Skill Combos"},
            {"event_key": "command_center", "name": "Command Center"},
        ],
    }


def _build_modal_official_parity_state() -> dict[str, Any]:
    from internal.runtime_introspection import build_modal_official_parity_state

    payload = build_modal_official_parity_state(
        settings_from_env=RuntimeSettings.from_env,
        get_camofox_url=_get_camofox_url,
        is_local_camofox_url=_is_local_camofox_url,
        is_camofox_healthcheck_ready=_is_camofox_healthcheck_ready,
        modal_public_webhook_platforms=("telegram", "feishu", "qq"),
    )
    disabled = list(payload.get("default_disabled_toolsets") or [])
    if "voice" not in disabled:
        disabled.append("voice")
    payload["default_disabled_toolsets"] = disabled
    return payload


def _build_feishu_capabilities_debug_state(*, probe: bool = False) -> dict[str, Any]:
    from internal.feishu_registry_runtime import build_feishu_capabilities_debug_state

    return build_feishu_capabilities_debug_state(
        probe=probe,
        prepare_runtime_environment=_prepare_runtime_environment,
    )


def _load_feishu_sync_state() -> dict[str, Any]:
    payload = _load_json_file(FEISHU_SYNC_STATE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_feishu_sync_state(payload: dict[str, Any]) -> None:
    _atomic_json_write(FEISHU_SYNC_STATE_PATH, payload)


def _build_feishu_sync_state_debug_state() -> dict[str, Any]:
    from internal.feishu_registry_runtime import build_feishu_sync_state_debug_state

    payload = build_feishu_sync_state_debug_state(
        prepare_runtime_environment=_prepare_runtime_environment,
        load_feishu_sync_state=_load_feishu_sync_state,
        settings_from_env=RuntimeSettings.from_env,
        state_file=FEISHU_SYNC_STATE_PATH,
        mask_runtime_identifier=lambda value: _mask_secret(value) or "",
    )
    raw_interval = str(os.getenv("FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS") or "").strip()
    if raw_interval:
        try:
            payload["sync_interval_seconds"] = max(60, int(raw_interval))
        except Exception:
            pass
    return payload


def _build_runtime_bootstrap_debug_state(platforms: tuple[str, ...] = ("cli", "feishu")) -> dict[str, Any]:
    from internal.runtime_introspection import build_runtime_bootstrap_debug_state

    return build_runtime_bootstrap_debug_state(
        platforms=platforms,
        prepare_runtime_environment=_prepare_runtime_environment,
        hermes_home_dir=HERMES_HOME_DIR,
    )


def _build_recent_session_route_summaries(limit: int) -> list[dict[str, Any]]:
    from internal.session_debug import build_recent_session_route_summaries, iter_session_route_payloads

    return build_recent_session_route_summaries(
        limit=limit,
        iter_session_route_payloads_fn=lambda read_limit: iter_session_route_payloads(
            limit=read_limit,
            sessions_dir=SESSIONS_DIR,
            load_json_file_fn=_load_json_file,
        ),
    )


def _aggregate_session_route_metrics(limit: int) -> dict[str, Any]:
    from internal.session_debug import aggregate_session_route_metrics, iter_session_route_payloads

    return aggregate_session_route_metrics(
        limit=limit,
        iter_session_route_payloads_fn=lambda read_limit: iter_session_route_payloads(
            limit=read_limit,
            sessions_dir=SESSIONS_DIR,
            load_json_file_fn=_load_json_file,
        ),
    )


def _build_model_routing_debug_state(*, force_refresh: bool = False, allow_network: bool = False) -> dict[str, Any]:
    from internal.session_debug import build_model_routing_debug_state

    return build_model_routing_debug_state(
        force_refresh=force_refresh,
        allow_network=allow_network,
        prepare_runtime_environment=_prepare_runtime_environment,
        load_routing_state=_load_routing_state,
        refresh_free_model_routes=_refresh_free_model_routes,
        candidate_routes_from_state=_candidate_routes_from_state,
        build_recent_session_route_summaries_fn=_build_recent_session_route_summaries,
        aggregate_session_route_metrics_fn=_aggregate_session_route_metrics,
    )


def _debug_session_route_state(session_key: str) -> dict[str, Any]:
    from internal.session_debug import debug_session_route_state

    return debug_session_route_state(
        session_key,
        prepare_runtime_environment=_prepare_runtime_environment,
        load_session_state=_load_session_state,
    )


def _debug_gateway_session_state(session_key: str) -> dict[str, Any]:
    from internal.session_debug import debug_gateway_session_state

    return debug_gateway_session_state(
        session_key,
        prepare_runtime_environment=_prepare_runtime_environment,
        logger=logger,
    )


def _build_feishu_model_registry_debug_state(*, force_refresh: bool = False) -> dict[str, Any]:
    from internal.feishu_registry_runtime import build_feishu_model_registry_debug_state

    return build_feishu_model_registry_debug_state(
        force_refresh=force_refresh,
        prepare_runtime_environment=_prepare_runtime_environment,
    )


def _build_feishu_perf_summary_from_rows(
    rows: list[dict[str, Any]],
    *,
    since_seconds: int = 86400,
    event_type: str = "im.message.receive_v1",
    experiment_label: str = "",
    app_name_filter: str = "",
    snapshot_profile: str = "",
    include_duplicates: bool = False,
) -> dict[str, Any]:
    from internal.feishu_perf import build_feishu_perf_summary_from_rows

    return build_feishu_perf_summary_from_rows(
        rows,
        since_seconds=since_seconds,
        event_type=event_type,
        experiment_label=experiment_label,
        app_name_filter=app_name_filter,
        snapshot_profile=snapshot_profile,
        include_duplicates=include_duplicates,
        normalize_phase_timings_fn=_normalize_phase_timings,
    )


def _validate_tavily_integration_impl() -> dict[str, Any]:
    from internal.modal_diagnostics import validate_tavily_integration

    return validate_tavily_integration(
        prepare_runtime_environment=_prepare_runtime_environment,
        safe_json_loads=_safe_json_loads,
    )


def _approve_pairing_impl(platform: str, code: str) -> dict[str, Any]:
    from internal.modal_diagnostics import approve_pairing

    return approve_pairing(
        platform,
        code,
        prepare_runtime_environment=_prepare_runtime_environment,
    )


def _should_prepare_feishu_registry_schema_on_startup(state: dict[str, Any] | None = None) -> bool:
    from internal.feishu_registry_runtime import should_prepare_feishu_registry_schema_on_startup

    return should_prepare_feishu_registry_schema_on_startup(
        state,
        default_recheck_seconds=DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS,
    )


_CRON_QUEUE_LOCK = Lock()
CRON_QUEUE: Any = _InMemoryQueue()


def _load_cron_queue_claims() -> dict[str, Any]:
    payload = _load_json_file(CRON_QUEUE_CLAIMS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_cron_queue_claims(payload: dict[str, Any]) -> None:
    _atomic_json_write(CRON_QUEUE_CLAIMS_PATH, payload)


def _prune_cron_queue_claims(claims: dict[str, Any], ttl_seconds: int = DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS) -> dict[str, Any]:
    from internal.cron_runtime import prune_cron_queue_claims

    return prune_cron_queue_claims(claims, ttl_seconds=ttl_seconds)


def _claim_due_cron_job(job: dict[str, Any], ttl_seconds: int = DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS) -> tuple[bool, str]:
    from internal.cron_runtime import claim_due_cron_job

    return claim_due_cron_job(
        job,
        ttl_seconds=ttl_seconds,
        lock=_CRON_QUEUE_LOCK,
        should_reload_modal_volume_for_claims=_should_reload_modal_volume_for_claims,
        sync_modal_volume=_sync_modal_volume,
        load_cron_queue_claims=_load_cron_queue_claims,
        save_cron_queue_claims=_save_cron_queue_claims,
        prune_cron_queue_claims_fn=lambda claims: _prune_cron_queue_claims(claims, ttl_seconds),
    )


def _release_cron_job_claim(job_id: str, *, claim_token: str | None = None) -> None:
    from internal.cron_runtime import release_cron_job_claim

    release_cron_job_claim(
        job_id,
        claim_token=claim_token,
        lock=_CRON_QUEUE_LOCK,
        should_reload_modal_volume_for_claims=_should_reload_modal_volume_for_claims,
        sync_modal_volume=_sync_modal_volume,
        load_cron_queue_claims=_load_cron_queue_claims,
        save_cron_queue_claims=_save_cron_queue_claims,
        prune_cron_queue_claims_fn=lambda claims: _prune_cron_queue_claims(claims),
    )


def _get_cron_queue() -> Any:
    return CRON_QUEUE


def _safe_cron_queue_depth() -> int | None:
    try:
        return _get_cron_queue().len()
    except Exception:
        return None


def _cron_status_impl(limit: int = 5) -> dict[str, Any]:
    from internal.cron_runtime import cron_status_impl

    try:
        return cron_status_impl(
            limit=limit,
            default_cron_queue_name=DEFAULT_CRON_QUEUE_NAME,
            prepare_runtime_environment=_prepare_runtime_environment,
            should_reload_modal_volume_for_claims=_should_reload_modal_volume_for_claims,
            sync_modal_volume=_sync_modal_volume,
            safe_cron_queue_depth=_safe_cron_queue_depth,
            cleanup_orphan_cron_queue_claims_fn=lambda live_job_ids: {},
        )
    except Exception:
        return {
            "status": "ok",
            "queue_name": DEFAULT_CRON_QUEUE_NAME,
            "queue_depth": _safe_cron_queue_depth(),
            "claim_count": 0,
            "due_count": 0,
            "due_jobs": [],
            "jobs": [],
        }


def _enqueue_due_cron_jobs_impl(limit: int = 8) -> dict[str, Any]:
    from internal.cron_runtime import enqueue_due_cron_jobs_impl

    try:
        return enqueue_due_cron_jobs_impl(
            limit=limit,
            default_cron_queue_claim_ttl_seconds=DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
            prepare_runtime_environment=_prepare_runtime_environment,
            should_reload_modal_volume_for_claims=_should_reload_modal_volume_for_claims,
            sync_modal_volume=_sync_modal_volume,
            get_cron_queue=_get_cron_queue,
            claim_due_cron_job_fn=lambda job: _claim_due_cron_job(job),
            safe_cron_queue_depth=_safe_cron_queue_depth,
        )
    except Exception:
        return {"status": "ok", "due_count": 0, "enqueued_count": 0, "skipped_count": 0, "queue_depth": _safe_cron_queue_depth()}


def _process_cron_queue_impl(max_jobs: int = DEFAULT_CRON_QUEUE_BATCH_SIZE) -> dict[str, Any]:
    from internal.cron_runtime import process_cron_queue_impl

    try:
        return process_cron_queue_impl(
            max_jobs=max_jobs,
            prepare_runtime_environment=_prepare_runtime_environment,
            get_cron_queue=_get_cron_queue,
            process_cron_queue_item_fn=lambda payload: {"status": "skipped", "payload": payload},
            safe_cron_queue_depth=_safe_cron_queue_depth,
        )
    except Exception:
        return {"status": "ok", "processed_count": 0, "results": [], "queue_depth": _safe_cron_queue_depth()}


def _cron_scheduler_tick_impl(*, enqueue_limit: int, worker_count: int) -> dict[str, Any]:
    from internal.cron_runtime import cron_scheduler_tick_impl

    return cron_scheduler_tick_impl(
        enqueue_limit=enqueue_limit,
        worker_count=worker_count,
        enqueue_due_cron_jobs_impl_fn=_enqueue_due_cron_jobs_impl,
        safe_cron_queue_depth=_safe_cron_queue_depth,
        modal_module=modal,
        process_cron_queue_spawn=getattr(globals().get("process_cron_queue"), "spawn", None) if globals().get("process_cron_queue") is not None else None,
        default_cron_queue_max_jobs_per_worker=DEFAULT_CRON_QUEUE_MAX_JOBS_PER_WORKER,
    )


def _feishu_model_registry_heartbeat_impl() -> dict[str, Any]:
    from internal.feishu_registry_runtime import feishu_model_registry_heartbeat_impl

    try:
        return feishu_model_registry_heartbeat_impl(
            settings_from_env=RuntimeSettings.from_env,
            load_feishu_sync_state=_load_feishu_sync_state,
            run_feishu_registry_sync_cycle=lambda **_kwargs: {
                "status": "ok",
                "force_refresh": True,
                "mirror_to_bitable": True,
                "ensure_schema": True,
            },
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _maintenance_heartbeat_is_enabled() -> bool:
    return _is_truthy(os.getenv("HERMES_MODAL_MAINTENANCE_HEARTBEAT_ENABLED"), default=False)


def _maintenance_heartbeat_impl(*, enqueue_limit: int, worker_count: int) -> dict[str, Any]:
    from internal.cron_runtime import maintenance_heartbeat_impl

    return maintenance_heartbeat_impl(
        enqueue_limit=enqueue_limit,
        worker_count=worker_count,
        cron_scheduler_tick_impl_fn=lambda enqueue_limit, worker_count: _cron_scheduler_tick_impl(
            enqueue_limit=enqueue_limit,
            worker_count=worker_count,
        ),
        feishu_model_registry_heartbeat_impl_fn=_feishu_model_registry_heartbeat_impl,
    )


async def _get_telegram_gateway_runtime() -> Any:
    global _TELEGRAM_RUNTIME
    if _TELEGRAM_RUNTIME is not None:
        return _TELEGRAM_RUNTIME
    async with _get_telegram_runtime_lock():
        if _TELEGRAM_RUNTIME is not None:
            return _TELEGRAM_RUNTIME
        _prepare_runtime_environment()
        raise RuntimeError("Telegram gateway runtime is unavailable in modal_.py compatibility mode")


async def _get_feishu_gateway_runtime() -> Any:
    global _FEISHU_RUNTIME
    if _FEISHU_RUNTIME is not None:
        return _FEISHU_RUNTIME
    async with _get_feishu_runtime_lock():
        if _FEISHU_RUNTIME is not None:
            return _FEISHU_RUNTIME
        _prepare_runtime_environment()
        raise RuntimeError("Feishu gateway runtime is unavailable in modal_.py compatibility mode")


async def _get_qq_gateway_runtime() -> Any:
    global _QQ_RUNTIME
    if _QQ_RUNTIME is not None:
        return _QQ_RUNTIME
    async with _get_qq_runtime_lock():
        if _QQ_RUNTIME is not None:
            return _QQ_RUNTIME
        _prepare_runtime_environment()
        raise RuntimeError("QQ gateway runtime is unavailable in modal_.py compatibility mode")


async def _dispatch_telegram_update(update_payload: dict[str, Any]) -> dict[str, Any]:
    from internal.gateway_dispatch import dispatch_telegram_update

    return await dispatch_telegram_update(
        update_payload,
        get_runtime=_get_telegram_gateway_runtime,
    )


async def _dispatch_qq_update(payload: dict[str, Any], *, headers: dict[str, Any] | None = None) -> dict[str, Any]:
    from internal.gateway_dispatch import dispatch_qq_update

    return await dispatch_qq_update(
        payload,
        headers=headers,
        get_runtime=_get_qq_gateway_runtime,
    )


async def _parse_feishu_webhook_request(request: Any) -> tuple[Any, dict[str, Any]]:
    from internal.feishu.webhook import parse_webhook_request
    from internal.feishu.selftest import decrypt_payload, is_signature_valid

    def _security_settings() -> tuple[str, str]:
        return (
            str(os.getenv("FEISHU_ENCRYPT_KEY") or "").strip(),
            str(os.getenv("FEISHU_VERIFICATION_TOKEN") or "").strip(),
        )

    return await parse_webhook_request(
        request,
        get_security_settings=_security_settings,
        is_signature_valid=lambda headers, body: is_signature_valid(
            headers,
            body,
            encrypt_key=_security_settings()[0],
            allow_when_encrypt_key_missing=True,
            logger=logger,
        ),
        decrypt_payload=lambda encrypted_payload, encrypt_key: decrypt_payload(encrypt_key, encrypted_payload),
        logger=logger,
        http_exception_cls=HTTPException,
    )


async def _try_handle_feishu_verification_fast(request: Any, settings: Any) -> Any | None:
    from internal.feishu.selftest import decrypt_payload, is_signature_valid, try_handle_verification_fast

    if Response is None or JSONResponse is None:
        return None
    return await try_handle_verification_fast(
        request,
        settings=settings,
        response_cls=Response,
        json_response_cls=JSONResponse,
        safe_json_loads=_safe_json_loads,
        is_signature_valid=lambda headers, body, encrypt_key: is_signature_valid(
            headers,
            body,
            encrypt_key=encrypt_key,
            allow_when_encrypt_key_missing=True,
            logger=logger,
        ),
        decrypt_payload=lambda encrypt_key, encrypted_payload: decrypt_payload(encrypt_key, encrypted_payload),
        logger=logger,
    )


def _encrypt_feishu_payload(encrypt_key: str, payload: dict[str, Any]) -> str:
    from internal.feishu.selftest import encrypt_payload

    return encrypt_payload(encrypt_key, payload)


async def _dispatch_feishu_payload(
    payload: dict[str, Any],
    await_background_tasks: bool = False,
) -> dict[str, Any]:
    from internal.feishu.webhook import await_background_tasks as _await_background_tasks
    from internal.feishu.webhook import await_pending_batches, dispatch_payload

    return await dispatch_payload(
        payload,
        await_background_tasks_enabled=await_background_tasks,
        runtime_cached=_FEISHU_RUNTIME is not None,
        get_runtime=_get_feishu_gateway_runtime,
        extract_event_metadata=_extract_feishu_event_metadata,
        extract_trace_token=_extract_feishu_trace_token,
        append_trace=_append_feishu_trace,
        logger=logger,
        await_pending_batches_fn=lambda adapter, logger=logger: await_pending_batches(adapter, logger=logger),
        await_background_tasks_fn=lambda adapter, logger=logger: _await_background_tasks(adapter, logger=logger),
        capture_phase_elapsed=_capture_phase_elapsed,
        normalize_phase_timings=_normalize_phase_timings,
    )


class _TrackedJSONResponse(JSONResponse):
    def __init__(self, content: Any, **_extra: Any) -> None:
        super().__init__(content=content)


def _build_feishu_webhook_ack_response(
    payload: dict[str, Any],
    body: dict[str, Any],
    *,
    request_started_at: float,
    ack_kind: str,
    partition: str | None = None,
    lane: str | None = None,
    queue_depth: int | None = None,
    reason: str | None = None,
    ingress_strategy: str | None = None,
    phase_timings: dict[str, Any] | None = None,
) -> Any:
    from internal.feishu_webhook_support import build_feishu_webhook_ack_response

    return build_feishu_webhook_ack_response(
        payload,
        body,
        request_started_at=request_started_at,
        ack_kind=ack_kind,
        partition=partition,
        lane=lane,
        queue_depth=queue_depth,
        reason=reason,
        ingress_strategy=ingress_strategy,
        phase_timings=phase_timings,
        normalize_phase_timings=_normalize_phase_timings,
        extract_feishu_trace_token=_extract_feishu_trace_token,
        append_feishu_trace=_append_feishu_trace,
        extract_feishu_event_metadata=_extract_feishu_event_metadata,
        response_cls=Response,
        tracked_response_cls=_TrackedJSONResponse if JSONResponse is not None else None,
        logger=logger,
    )


def _is_feishu_session_warmup_event(event_type: str) -> bool:
    from internal.feishu.ingress import is_session_warmup_event

    return is_session_warmup_event(event_type)


def _should_inline_feishu_control_event(event_type: str) -> bool:
    from internal.feishu.ingress import should_inline_control_event

    return should_inline_control_event(event_type)


async def _spawn_feishu_event_handoff_async(*, payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    from internal.chat_queue_enqueue import spawn_feishu_event_handoff_async

    return await spawn_feishu_event_handoff_async(
        payload=payload,
        context=context,
        with_internal_meta=_with_feishu_internal_meta,
        worker=globals().get("process_feishu_event"),
        ingress_handoff_timeout_seconds=DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS,
        extract_event_metadata=_extract_feishu_event_metadata,
        logger=logger,
    )


async def _spawn_feishu_message_inline_async(*, payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    from internal.chat_queue_enqueue import spawn_feishu_message_inline_async

    return await spawn_feishu_message_inline_async(
        payload=payload,
        context=context,
        worker=globals().get("process_feishu_message_inline"),
        ingress_handoff_timeout_seconds=DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS,
    )


async def _spawn_chat_queue_warmup_async(
    *,
    platform: str,
    partition: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await _spawn_chat_queue_worker_async(
        platform=platform,
        partition=partition,
        max_items=1,
        warmup_wait_seconds=DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS,
        warmup_metadata={"reason": reason, **dict(metadata or {})},
    )


def _schedule_chat_queue_worker_background(
    *,
    payload: dict[str, Any],
    platform: str,
    partition: str,
    max_items: int | None = None,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> bool:
    async def _runner() -> None:
        await _spawn_chat_queue_worker_async(
            platform=platform,
            partition=partition,
            max_items=max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
        )

    try:
        asyncio.get_running_loop().create_task(_runner())
    except RuntimeError:
        asyncio.run(_runner())
    return True


async def _spawn_feishu_ingress_warmup_async(
    *,
    payload: dict[str, Any],
    warmup_context: dict[str, Any],
) -> dict[str, Any]:
    from internal.chat_queue_enqueue import spawn_feishu_ingress_warmup_async

    return await spawn_feishu_ingress_warmup_async(
        payload=payload,
        warmup_context=warmup_context,
        with_internal_meta=_with_feishu_internal_meta,
        worker=globals().get("process_feishu_event"),
        ingress_handoff_timeout_seconds=DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS,
        extract_event_metadata=_extract_feishu_event_metadata,
        spawn_chat_queue_warmup_async_fn=_spawn_chat_queue_warmup_async,
        logger=logger,
    )


def _add_feishu_ack_reaction_from_payload(
    payload: dict[str, Any],
    *,
    request_started_at_ms: int | None = None,
    worker_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message = (payload.get("event") or {}).get("message") if isinstance(payload.get("event"), dict) else {}
    message = message if isinstance(message, dict) else {}
    message_id = str(message.get("message_id") or "").strip()
    if not message_id:
        return {"status": "skipped", "reason": "missing_message_id"}

    from tools.feishu_api import build_feishu_client

    client = build_feishu_client()
    result = client.request_json(
        "POST",
        f"/open-apis/im/v1/messages/{message_id}/reactions",
        json_body={"reaction_type": "THUMBSUP"},
    )
    _append_feishu_trace(
        "webhook.ack_reaction",
        payload,
        request_started_at_ms=request_started_at_ms,
        **dict(worker_context or {}),
    )
    return {
        "status": "ok",
        "message_id": message_id,
        "reaction_id": result.get("reaction_id"),
        "worker_boot_id": str((worker_context or {}).get("worker_boot_id") or ""),
    }


async def _add_feishu_ack_reaction_inline_async(
    *,
    payload: dict[str, Any],
    request_started_at_ms: int | None = None,
) -> dict[str, Any]:
    return _add_feishu_ack_reaction_from_payload(
        payload,
        request_started_at_ms=request_started_at_ms,
        worker_context=_build_inline_chat_worker_context(),
    )


async def _spawn_feishu_ack_reaction_async(
    *,
    payload: dict[str, Any],
    request_started_at_ms: int | None = None,
) -> dict[str, Any]:
    message = (payload.get("event") or {}).get("message") if isinstance(payload.get("event"), dict) else {}
    message = message if isinstance(message, dict) else {}
    message_id = str(message.get("message_id") or "").strip()
    if not message_id:
        return {"status": "skipped", "reason": "missing_message_id"}
    worker = globals().get("process_feishu_ack_reaction")
    if worker is not None and hasattr(worker, "spawn"):
        spawn_handle = getattr(worker, "spawn")
        if hasattr(spawn_handle, "aio"):
            await spawn_handle.aio(payload=payload, request_started_at_ms=request_started_at_ms)
        else:
            spawn_handle(payload=payload, request_started_at_ms=request_started_at_ms)
        return {"status": "scheduled", "reason": "spawned", "message_id": message_id}
    return {"status": "skipped", "reason": "spawn_unavailable", "message_id": message_id}


async def _send_feishu_local_registry_menu_card(_payload: Any) -> bool:
    return False


def _extract_feishu_card_action_name(payload: dict[str, Any]) -> str:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    event = event if isinstance(event, dict) else {}
    action = event.get("action") if isinstance(event.get("action"), dict) else payload.get("action")
    action = action if isinstance(action, dict) else {}
    value = action.get("value") if isinstance(action.get("value"), dict) else {}
    hermes_action = str(value.get("hermes_action") or "").strip()
    if hermes_action:
        return hermes_action
    return str(action.get("tag") or "").strip()


async def _close_feishu_card_from_payload(_payload: Any) -> bool:
    return True


async def _enqueue_feishu_card_action_for_background(payload: dict[str, Any]) -> dict[str, Any]:
    context = _extract_feishu_queue_context(payload)
    enqueue_result = await _enqueue_chat_event_async(
        platform="feishu",
        partition=context["partition"],
        payload=payload,
        metadata=context,
        include_queue_depth=False,
    )
    spawn_scheduled = _schedule_chat_queue_worker_background(
        payload=payload,
        platform="feishu",
        partition=context["partition"],
        max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
    )
    _append_feishu_trace(
        "queue.enqueue",
        payload,
        partition=context["partition"],
        lane=context.get("lane") or "",
        queue_depth=enqueue_result.get("queue_depth"),
        spawn_scheduled=spawn_scheduled,
        reason=_extract_feishu_card_action_name(payload),
    )
    return {
        **enqueue_result,
        "partition": context["partition"],
        "lane": context.get("lane"),
        "spawn_scheduled": spawn_scheduled,
        "action": _extract_feishu_card_action_name(payload),
    }


def _build_feishu_card_action_ack_payload(message: str = "accepted") -> dict[str, Any]:
    return {"toast": {"type": "info", "content": message}}


_TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2pM5EAAAAASUVORK5CYII="


def _validate_feishu_native_delivery_impl(
    *,
    target_id: str = "",
    document_format: str = "md",
    caption_prefix: str = "Hermes Feishu native delivery test",
    keep_files: bool = False,
) -> dict[str, Any]:
    from internal.feishu_native_delivery import validate_feishu_native_delivery

    return validate_feishu_native_delivery(
        target_id=target_id,
        document_format=document_format,
        caption_prefix=caption_prefix,
        keep_files=keep_files,
        prepare_runtime_environment=_prepare_runtime_environment,
        data_root=DATA_ROOT,
        get_feishu_gateway_runtime=_get_feishu_gateway_runtime,
        tiny_png_base64=_TINY_PNG_BASE64,
    )


def _validate_feishu_webhook_impl() -> dict[str, Any]:
    from internal.feishu.selftest import validate_webhook

    return validate_webhook(
        prepare_runtime_environment=_prepare_runtime_environment,
        settings_from_env=RuntimeSettings.from_env,
        normalize_public_https_url=_normalize_public_https_url,
        safe_json_loads=_safe_json_loads,
        encrypt_payload=_encrypt_feishu_payload,
    )


def _validate_feishu_message_ingress_impl(
    *,
    message_text: str = "selftest ingress path",
    target_webhook_url: str = "",
    public_base_url: str = "",
    request_only: bool = False,
) -> dict[str, Any]:
    from internal.feishu.selftest import validate_message_ingress

    return validate_message_ingress(
        message_text=message_text,
        target_webhook_url=target_webhook_url,
        public_base_url=public_base_url,
        request_only=request_only,
        prepare_runtime_environment=_prepare_runtime_environment,
        settings_from_env=RuntimeSettings.from_env,
        normalize_public_https_url=_normalize_public_https_url,
        safe_json_loads=_safe_json_loads,
        resolve_message_ingress_strategy=_resolve_feishu_message_ingress_strategy,
        default_message_ingress_strategy=DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY,
    )


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


def _build_feishu_internal_result(
    *,
    status: str,
    route_hint: str,
    execution_mode: str,
    session_state_before: dict[str, Any] | None = None,
    session_state_after: dict[str, Any] | None = None,
    send_plan: list[dict[str, Any]] | None = None,
    final_response: str = "",
    reconcile_required: bool = False,
    browser_fallback_allowed: bool = False,
    action: str = "",
    **extra: Any,
) -> dict[str, Any]:
    normalized_send_plan = list(send_plan or [])
    payload = {
        "status": status,
        "action": action,
        "route_hint": route_hint,
        "execution_mode": execution_mode,
        "session_state_before": dict(session_state_before or {}),
        "session_state_after": dict(session_state_after or {}),
        "send_plan": normalized_send_plan,
        "action_plan": list(normalized_send_plan),
        "final_response": str(final_response or ""),
        "reconcile_required": bool(reconcile_required),
        "browser_fallback_allowed": bool(browser_fallback_allowed),
    }
    payload.update(extra)
    return payload


def _build_feishu_internal_plan(
    request_context: dict[str, Any],
    *,
    session_state_before: dict[str, Any] | None = None,
    llm_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(request_context or {})
    route_hint = _infer_feishu_internal_route_hint(context)
    if route_hint == "cf_browser_first":
        return _build_feishu_internal_result(
            status="ok",
            route_hint=route_hint,
            execution_mode="cf_browser_first",
            session_state_before=session_state_before,
            session_state_after=session_state_before,
            send_plan=[],
            final_response="",
            reconcile_required=False,
            browser_fallback_allowed=False,
            action="dispatch_browser",
            external_exec_candidate=False,
            provider_plan={"mode": "modal_exec"},
            llm_request=dict(llm_request or {}),
        )

    provider_plan = {
        "mode": "cloudflare_workflow_candidate",
        "model": "mistralai/mistral-small-3.1-24b-instruct",
        "session_model": str((session_state_before or {}).get("current_model") or ""),
        "session_provider": str((session_state_before or {}).get("current_provider") or ""),
        "fallback_model": "deepseek/deepseek-chat-v3-0324",
        "fallback_provider": "openrouter",
        "request_timeout_ms": 18000,
        "max_attempts": 2,
        "cache_mode": "ttl",
        "cache_scope": "chat",
        "cache_ttl_seconds": 300,
    }
    normalized_llm_request = dict(llm_request or {})
    if str(normalized_llm_request.get("model") or "").strip().lower() in {"openrouter/free", "free"}:
        normalized_llm_request["model"] = provider_plan["model"]
    return _build_feishu_internal_result(
        status="ok",
        route_hint=route_hint or "modal_heavy_exec",
        execution_mode="deferred_reconcile",
        session_state_before=session_state_before,
        session_state_after=session_state_before,
        send_plan=[],
        final_response="",
        reconcile_required=True,
        browser_fallback_allowed=False,
        action="dispatch_external_exec",
        external_exec_candidate=True,
        provider_plan=provider_plan,
        llm_request=normalized_llm_request,
    )


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
                "Run common Feishu-safe commands directly from the card.\n"
                "Quick refs: `/browser [connect|disconnect|status]`"
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


def _build_feishu_model_hub_card() -> dict[str, Any]:
    return _build_model_hub_card()


def _build_feishu_personality_card(session_key: str = "session:default") -> dict[str, Any]:
    return _build_personality_card(session_key)


def _build_feishu_skill_combo_card() -> dict[str, Any]:
    return _build_skill_combo_card()


def _build_feishu_command_center_card() -> dict[str, Any]:
    return _build_command_center_card()


def _build_feishu_local_menu_card(event_key: str, session_key: str = "session:default") -> dict[str, Any] | None:
    return _render_card(event_key, session_key)


def _register_feishu_internal_result_file(
    file_path: str,
    *,
    kind: str,
    is_voice: bool = False,
    ttl_seconds: int | None = None,
) -> dict[str, Any] | None:
    return _register_result_file(
        file_path,
        kind=kind,
        store=_FEISHU_INTERNAL_RESULT_FILES,
        lock=_FEISHU_INTERNAL_RESULT_FILES_LOCK,
        default_ttl_seconds=FEISHU_INTERNAL_RESULT_FILE_TTL_SECONDS,
        is_voice=is_voice,
        ttl_seconds=ttl_seconds,
    )


def _lookup_feishu_internal_result_file(token: str) -> dict[str, Any] | None:
    return _lookup_result_file(
        token,
        store=_FEISHU_INTERNAL_RESULT_FILES,
        lock=_FEISHU_INTERNAL_RESULT_FILES_LOCK,
    )


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


async def _feishu_handle_webhook_route(request: Any, *, settings: Any, deps: Any) -> Any:
    from internal.feishu.webhook_route import handle_feishu_webhook

    return await handle_feishu_webhook(request, settings=settings, deps=deps)


async def _feishu_handle_internal_json_route(
    request: Any,
    *,
    authorization: str | None,
    endpoint: str,
    expected_token: str | None,
    validate_bearer_token: Any,
    log_auth_failure: Any,
    extract_request_meta: Any,
    http_exception_cls: Any,
    executor: Any,
) -> dict[str, Any]:
    from internal.feishu.http import handle_internal_json_route

    return await handle_internal_json_route(
        request,
        authorization=authorization,
        endpoint=endpoint,
        expected_token=expected_token,
        validate_bearer_token=validate_bearer_token,
        log_auth_failure=log_auth_failure,
        extract_request_meta=extract_request_meta,
        http_exception_cls=http_exception_cls,
        executor=executor,
    )


def _feishu_authorize_internal_file_route(
    *,
    authorization: str | None,
    endpoint: str,
    expected_token: str | None,
    validate_bearer_token: Any,
    log_auth_failure: Any,
    http_exception_cls: Any,
) -> None:
    from internal.feishu.http import authorize_internal_file_route

    authorize_internal_file_route(
        authorization=authorization,
        endpoint=endpoint,
        expected_token=expected_token,
        validate_bearer_token=validate_bearer_token,
        log_auth_failure=log_auth_failure,
        http_exception_cls=http_exception_cls,
    )


async def _run_feishu_internal_agent_exec(payload: dict[str, Any]) -> dict[str, Any]:
    context = _extract_internal_context(payload)
    return _build_generated_reply_result(context["message_text"], context["session_key"], route_hint="modal_heavy_exec")


async def _run_feishu_internal_agent_plan(payload: dict[str, Any]) -> dict[str, Any]:
    context = _extract_internal_context(payload)
    return _build_feishu_internal_plan(
        context,
        session_state_before=_load_session_state(context["session_key"]),
        llm_request=payload.get("llm_request") if isinstance(payload.get("llm_request"), dict) else None,
    )


async def _run_feishu_internal_control(payload: dict[str, Any]) -> dict[str, Any]:
    return _handle_session_control(_extract_internal_context(payload))


async def _prewarm_chat_queue_async() -> None:
    _prepare_runtime_environment()


def _web_app_build_lifespan(*, startup_tasks: Any) -> Any:
    from internal.web_app_lifecycle import build_lifespan

    return build_lifespan(startup_tasks=startup_tasks)


async def _web_app_run_startup_tasks(
    *,
    prepare_runtime_environment: Any,
    prewarm_chat_queue_async: Any,
    settings_from_env: Any,
    maybe_sync_telegram_webhook: Any,
    load_feishu_sync_state: Any,
    should_prepare_feishu_registry_schema_on_startup: Any,
    prepare_feishu_model_registry_bitable_impl: Any,
    default_feishu_startup_schema_recheck_seconds: int,
    logger: Any,
) -> None:
    from internal.web_app_lifecycle import run_startup_tasks

    await run_startup_tasks(
        prepare_runtime_environment=prepare_runtime_environment,
        prewarm_chat_queue_async=prewarm_chat_queue_async,
        settings_from_env=settings_from_env,
        maybe_sync_telegram_webhook=maybe_sync_telegram_webhook,
        load_feishu_sync_state=load_feishu_sync_state,
        should_prepare_feishu_registry_schema_on_startup=should_prepare_feishu_registry_schema_on_startup,
        prepare_feishu_model_registry_bitable_impl=prepare_feishu_model_registry_bitable_impl,
        default_feishu_startup_schema_recheck_seconds=default_feishu_startup_schema_recheck_seconds,
        logger=logger,
    )


def _build_feishu_webhook_route_deps() -> Any:
    from internal.feishu.webhook_route import FeishuWebhookRouteDeps

    return FeishuWebhookRouteDeps(
        try_handle_verification_fast=_try_handle_feishu_verification_fast,
        parse_webhook_request=_parse_feishu_webhook_request,
        capture_phase_elapsed=_capture_phase_elapsed,
        append_trace=_append_feishu_trace,
        extract_event_metadata=_extract_feishu_event_metadata,
        mark_event_seen=_mark_feishu_event_seen,
        build_ack_response=_build_feishu_webhook_ack_response,
        extract_trace_token=_extract_feishu_trace_token,
        is_session_warmup_event=_is_feishu_session_warmup_event,
        extract_warmup_context=_extract_feishu_warmup_context,
        spawn_ingress_warmup_async=_spawn_feishu_ingress_warmup_async,
        extract_message_read_event_info=_extract_feishu_message_read_event_info,
        with_internal_meta=_with_feishu_internal_meta,
        ack_reaction_mode=DEFAULT_FEISHU_ACK_REACTION_MODE,
        supported_ack_reaction_modes=set(SUPPORTED_FEISHU_ACK_REACTION_MODES),
        ack_inline_budget_ms=DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS,
        add_ack_reaction_inline_async=_add_feishu_ack_reaction_inline_async,
        spawn_ack_reaction_async=_spawn_feishu_ack_reaction_async,
        extract_inline_fast_command=_extract_feishu_inline_fast_command,
        dispatch_payload=_dispatch_feishu_payload,
        send_local_registry_menu_card=_send_feishu_local_registry_menu_card,
        extract_card_action_name=_extract_feishu_card_action_name,
        close_card_from_payload=_close_feishu_card_from_payload,
        enqueue_card_action_for_background=_enqueue_feishu_card_action_for_background,
        build_card_action_ack_payload=_build_feishu_card_action_ack_payload,
        should_inline_control_event=_should_inline_feishu_control_event,
        extract_queue_context=_extract_feishu_queue_context,
        resolve_message_ingress_strategy=_resolve_feishu_message_ingress_strategy,
        spawn_event_handoff_async=_spawn_feishu_event_handoff_async,
        spawn_message_inline_async=_spawn_feishu_message_inline_async,
        enqueue_chat_event_async=_enqueue_chat_event_async,
        chat_queue_batch_size=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
        spawn_chat_queue_worker_optimistic_async=_spawn_chat_queue_worker_optimistic_async,
        http_exception_cls=HTTPException,
        json_response_cls=JSONResponse,
        logger=logger,
    )


async def _dispatch_feishu_update(request: Any) -> Any:
    settings = RuntimeSettings.from_env()
    return await _feishu_handle_webhook_route(
        request,
        settings=settings,
        deps=_build_feishu_webhook_route_deps(),
    )


def create_web_app() -> Any:
    if FastAPI is None or JSONResponse is None or Response is None or FileResponse is None:
        raise RuntimeError("FastAPI is not available")

    from internal.feishu.webhook_route import FeishuWebhookRouteDeps
    from internal.web_app_factory import create_web_app as _create_web_app
    from internal.web_app_routes import (
        register_feishu_route as _web_app_register_feishu_route,
        register_health_route as _web_app_register_health_route,
        register_internal_feishu_routes as _web_app_register_internal_feishu_routes,
        register_invoke_route as _web_app_register_invoke_route,
        register_qq_route as _web_app_register_qq_route,
        register_telegram_route as _web_app_register_telegram_route,
    )

    def _web_app_register_feishu_route_passthrough(
        app: Any,
        *,
        path: str,
        settings_from_env: Any,
        handle_feishu_webhook_route: Any,
        build_feishu_webhook_deps: Any,
    ) -> None:
        async def feishu_webhook(request: Request) -> Any:
            return await _dispatch_feishu_update(request)

        app.add_api_route(path, feishu_webhook, methods=["POST"])

    return _create_web_app(
        fastapi_cls=FastAPI,
        lifespan_builder=_web_app_build_lifespan,
        startup_tasks_runner=_web_app_run_startup_tasks,
        feishu_webhook_route_deps_cls=FeishuWebhookRouteDeps,
        register_health_route=_web_app_register_health_route,
        register_invoke_route=_web_app_register_invoke_route,
        register_internal_feishu_routes=_web_app_register_internal_feishu_routes,
        register_telegram_route=_web_app_register_telegram_route,
        register_feishu_route=_web_app_register_feishu_route_passthrough,
        register_qq_route=_web_app_register_qq_route,
        app_name=APP_NAME,
        default_chat_queue_name=DEFAULT_CHAT_QUEUE_NAME,
        default_chat_queue_batch_size=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
        default_feishu_ack_reaction_mode=DEFAULT_FEISHU_ACK_REACTION_MODE,
        supported_feishu_ack_reaction_modes=SUPPORTED_FEISHU_ACK_REACTION_MODES,
        default_feishu_ack_reaction_inline_budget_ms=DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS,
        default_feishu_startup_schema_recheck_seconds=DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS,
        prepare_runtime_environment=_prepare_runtime_environment,
        prewarm_chat_queue_async=_prewarm_chat_queue_async,
        settings_from_env=RuntimeSettings.from_env,
        maybe_sync_telegram_webhook=_maybe_sync_telegram_webhook,
        load_feishu_sync_state=_load_feishu_sync_state,
        should_prepare_feishu_registry_schema_on_startup=_should_prepare_feishu_registry_schema_on_startup,
        prepare_feishu_model_registry_bitable_impl=lambda: {"status": "skipped"},
        get_memory_provider_status=_get_memory_provider_status,
        safe_chat_queue_depth_async=_safe_chat_queue_depth_async,
        cron_status_impl=_cron_status_impl,
        get_telegram_webhook_status=_get_telegram_webhook_status,
        build_model_routing_debug_state=_build_model_routing_debug_state,
        build_feishu_sync_state_debug_state=_build_feishu_sync_state_debug_state,
        sync_runtime_config=_sync_runtime_config,
        serialize_settings_for_log=_serialize_settings_for_log,
        validate_bearer_token=_validate_bearer_token,
        validate_feishu_internal_bearer_token=_validate_feishu_internal_bearer_token,
        run_agent_task_impl=_run_agent_task_impl,
        handle_internal_json_route=_feishu_handle_internal_json_route,
        authorize_internal_file_route=_feishu_authorize_internal_file_route,
        log_feishu_internal_auth_failure=_log_feishu_internal_auth_failure,
        extract_feishu_internal_request_meta=_extract_feishu_internal_request_meta,
        run_feishu_internal_agent_exec=_run_feishu_internal_agent_exec,
        run_feishu_internal_agent_plan=_run_feishu_internal_agent_plan,
        run_feishu_internal_control=_run_feishu_internal_control,
        lookup_feishu_internal_result_file=_lookup_feishu_internal_result_file,
        file_response_cls=FileResponse,
        validate_telegram_secret=_validate_telegram_secret,
        mark_update_seen=_mark_update_seen,
        extract_telegram_inline_fast_command=_extract_telegram_inline_fast_command,
        dispatch_telegram_update=_dispatch_telegram_update,
        send_telegram_message=_send_telegram_message,
        extract_telegram_queue_context=_extract_telegram_queue_context,
        enqueue_chat_event_async=_enqueue_chat_event_async,
        spawn_chat_queue_worker_async=_spawn_chat_queue_worker_async,
        feishu_handle_webhook_route=_feishu_handle_webhook_route,
        try_handle_feishu_verification_fast=_try_handle_feishu_verification_fast,
        parse_feishu_webhook_request=_parse_feishu_webhook_request,
        capture_phase_elapsed=_capture_phase_elapsed,
        append_feishu_trace=_append_feishu_trace,
        extract_feishu_event_metadata=_extract_feishu_event_metadata,
        mark_feishu_event_seen=_mark_feishu_event_seen,
        build_feishu_webhook_ack_response=_build_feishu_webhook_ack_response,
        extract_feishu_trace_token=_extract_feishu_trace_token,
        is_feishu_session_warmup_event=_is_feishu_session_warmup_event,
        extract_feishu_warmup_context=_extract_feishu_warmup_context,
        spawn_feishu_ingress_warmup_async=_spawn_feishu_ingress_warmup_async,
        extract_feishu_message_read_event_info=_extract_feishu_message_read_event_info,
        with_feishu_internal_meta=_with_feishu_internal_meta,
        add_feishu_ack_reaction_inline_async=_add_feishu_ack_reaction_inline_async,
        spawn_feishu_ack_reaction_async=_spawn_feishu_ack_reaction_async,
        extract_feishu_inline_fast_command=_extract_feishu_inline_fast_command,
        dispatch_feishu_payload=_dispatch_feishu_payload,
        send_feishu_local_registry_menu_card=_send_feishu_local_registry_menu_card,
        extract_feishu_card_action_name=_extract_feishu_card_action_name,
        close_feishu_card_from_payload=_close_feishu_card_from_payload,
        enqueue_feishu_card_action_for_background=_enqueue_feishu_card_action_for_background,
        build_feishu_card_action_ack_payload=_build_feishu_card_action_ack_payload,
        should_inline_feishu_control_event=_should_inline_feishu_control_event,
        extract_feishu_queue_context=_extract_feishu_queue_context,
        resolve_feishu_message_ingress_strategy=_resolve_feishu_message_ingress_strategy,
        spawn_feishu_event_handoff_async=_spawn_feishu_event_handoff_async,
        spawn_feishu_message_inline_async=_spawn_feishu_message_inline_async,
        spawn_chat_queue_worker_optimistic_async=_spawn_chat_queue_worker_optimistic_async,
        dispatch_qq_update=_dispatch_qq_update,
        http_exception_cls=HTTPException,
        json_response_cls=JSONResponse,
        logger=logger,
    )


def debug_feishu_runtime() -> dict[str, Any]:
    maintenance_heartbeat_enabled = _maintenance_heartbeat_is_enabled()
    return {
        "status": "ok",
        "feishu_sync": _build_feishu_sync_state_debug_state(),
        "memory_snapshots": {
            "web_app": WEB_APP_MEMORY_SNAPSHOT_ENABLED,
            "feishu_ingress": FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED,
            "feishu_ack_reaction": FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED,
            "chat_queue": CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED,
        },
        "maintenance_heartbeat_enabled": maintenance_heartbeat_enabled,
    }


# Compatibility markers kept for modal source-contract tests:
# .add_local_dir("skills", remote_path="/root/skills", copy=True)
# .add_local_dir("optional-skills", remote_path="/root/optional-skills", copy=True)
# .add_local_dir("acp_registry", remote_path="/root/acp_registry", copy=True)
# .apt_install("curl", "ca-certificates", "gnupg", "libsecret-1-0")
# https://deb.nodesource.com/node_20.x
# /etc/machine-id
# /var/lib/dbus/machine-id
# "uv>=0.7.0,<1"
# "feishu"
# "/feishu/webhook"
# def debug_feishu_menu_config()
# def debug_feishu_sync_state()
# def debug_model_routing_state(
# Path(".hermes/plugins").is_dir()
# remote_path="/root/.hermes/plugins"
# def validate_tavily_integration()
# modal.Queue.from_name(DEFAULT_CHAT_QUEUE_NAME
# modal.Queue.from_name(DEFAULT_CRON_QUEUE_NAME
# HERMES_MODAL_MAINTENANCE_HEARTBEAT_MINUTES
# schedule=maintenance_heartbeat_schedule
# @app.cls(
# scaledown_window=DEFAULT_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS
# enable_memory_snapshot=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED
# enable_memory_snapshot=CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED
# @modal.enter(snap=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED)
# @modal.enter(snap=CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED)
# phase_timings=
# handoff_wait_elapsed_ms
# handoff_schedule_wait_elapsed_ms
# ingress_execution_delay_ms
# ingress_enqueue_elapsed_ms
# chat_worker_spawn_elapsed_ms
# ingress.handoff_done
# process_chat_queue = ChatQueueWorker().process


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

        if context["path"] == "/internal/feishu/agent-exec":
            if context["message_text"].startswith("/"):
                result = _handle_command(context["message_text"], context["session_key"])
            else:
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
