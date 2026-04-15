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
import math
import mimetypes
import os
import queue as pyqueue
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

try:
    from fastapi import FastAPI, Header, HTTPException, Request, Response
    from fastapi.responses import FileResponse, JSONResponse
except Exception:  # pragma: no cover - local unit tests can still import helpers without FastAPI
    FastAPI = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    FileResponse = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]

try:
    import modal
except Exception:  # pragma: no cover - local unit tests can import helpers without a working Modal install
    modal = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_FEISHU_TRACE_TOKEN_RE = re.compile(r"\[trace:([A-Za-z0-9_-]{6,64})\]", re.IGNORECASE)
_RUNTIME_SKILLS_SYNC_LOCK = threading.Lock()
_RUNTIME_SKILLS_SYNCED = False

APP_NAME = os.getenv("HERMES_MODAL_APP_NAME", "hermes-agent")
DATA_ROOT = Path(os.getenv("HERMES_MODAL_DATA_DIR", "/data/hermes"))
SESSIONS_DIR = DATA_ROOT / "sessions"
UPDATES_PATH = DATA_ROOT / "telegram_updates.json"
FEISHU_EVENTS_PATH = DATA_ROOT / "feishu_events.json"
FEISHU_TRACE_PATH = DATA_ROOT / "feishu_trace.jsonl"
FEISHU_SYNC_STATE_PATH = DATA_ROOT / "feishu_sync_state.json"
FEISHU_BOT_IDENTITY_CACHE_PATH = DATA_ROOT / "feishu_bot_identity_cache.json"
HERMES_HOME_DIR = Path(os.getenv("HERMES_HOME", "/data/hermes-home"))
DEFAULT_CONFIG_SOURCE = Path(__file__).with_name("config.modal.yaml")
DEFAULT_SUPERMEMORY_CONFIG_SOURCE = Path(__file__).with_name("supermemory.modal.json")
DEFAULT_DISABLED_TOOLSETS = ["rl", "voice"]
DEFAULT_UPDATE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_SECRET_NAME = os.getenv("HERMES_MODAL_SECRET_NAME", "custom-secret")
DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/"
    "d1215a30b84b673ef0367010b0e78c10/affiliate-manager"
)
DEFAULT_VOLUME_NAME = os.getenv("HERMES_MODAL_VOLUME_NAME", "hermes-agent-data")
DEFAULT_CHAT_QUEUE_NAME = os.getenv("HERMES_MODAL_CHAT_QUEUE_NAME", f"{APP_NAME}-chat-queue")
DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS = int(os.getenv("HERMES_MODAL_CHAT_QUEUE_CLAIM_TTL_SECONDS", "1800"))
DEFAULT_CHAT_QUEUE_BATCH_SIZE = int(os.getenv("HERMES_MODAL_CHAT_QUEUE_BATCH_SIZE", "8"))
DEFAULT_CHAT_QUEUE_SPAWN_COOLDOWN_SECONDS = max(
    1,
    int(os.getenv("HERMES_MODAL_CHAT_QUEUE_SPAWN_COOLDOWN_SECONDS", "15")),
)
_FEISHU_LOCAL_MODEL_MENU_MAP: dict[str, tuple[str | None, str | None]] = {
    "provider_openrouter": ("openrouter", "featured"),
    "provider_openrouter_featured": ("openrouter", "featured"),
    "provider_openrouter_recent": ("openrouter", "recent"),
    "provider_openrouter_performance": ("openrouter", "performance"),
    "provider_nvidia": ("nvidia", "featured"),
    "provider_nvidia_featured": ("nvidia", "featured"),
    "provider_nvidia_recent": ("nvidia", "recent"),
    "provider_nvidia_performance": ("nvidia", "performance"),
}
_FEISHU_PERSONALITY_CARD_ORDER: tuple[str, ...] = (
    "ceo",
    "cto",
    "grow",
    "staff",
    "sev",
    "board",
    "content",
    "seo",
    "ads",
    "bd",
    "ops",
    "finance",
)
_FEISHU_PERSONALITY_CARD_LABELS: dict[str, str] = {
    "board": "BOARD",
    "ceo": "CEO",
    "grow": "GROW",
    "cto": "CTO",
    "staff": "STAFF",
    "sev": "SEV",
    "content": "CONTENT",
    "seo": "SEO",
    "ads": "ADS",
    "bd": "BD",
    "ops": "OPS",
    "finance": "FIN",
}
_FEISHU_COMMAND_CARD_CATEGORY_ORDER: tuple[str, ...] = (
    "Session",
    "Configuration",
    "Tools & Skills",
    "Info",
    "Exit",
)
_FEISHU_COMMAND_CARD_CATEGORY_LABELS: dict[str, str] = {
    "Session": "Session",
    "Configuration": "Config",
    "Tools & Skills": "Tools",
    "Info": "Info",
    "Exit": "Exit",
}
_FEISHU_COMMAND_CARD_DEFAULT_RUNS: dict[str, str] = {
    "new": "/new",
    "retry": "/retry",
    "undo": "/undo",
    "compress": "/compress",
    "stop": "/stop",
    "status": "/status",
    "profile": "/profile",
    "sethome": "/sethome",
    "model": "/model",
    "provider": "/provider",
    "personality": "/personality",
    "reasoning": "/reasoning",
    "yolo": "/yolo",
    "voice": "/voice status",
    "reload-mcp": "/reload-mcp",
    "commands": "/commands",
    "help": "/help",
    "usage": "/usage",
    "insights": "/insights",
    "update": "/update",
}
_FEISHU_SKILL_COMBO_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "ceo_stack",
        "label": "CEO 操盘",
        "summary": "ceo-os：目标、优先级、推进闭环",
        "skills": ["ceo-os"],
        "suggested_personality": "ceo",
    },
    {
        "id": "cto_ship",
        "label": "CTO 交付",
        "summary": "ship + gov：实现、性能、回归验证",
        "skills": ["ship", "gov"],
        "suggested_personality": "cto",
    },
    {
        "id": "incident_sev",
        "label": "SEV 止血",
        "summary": "gov + retro：止血、恢复、复盘",
        "skills": ["gov", "retro"],
        "suggested_personality": "sev",
    },
    {
        "id": "growth_launch",
        "label": "增长实验",
        "summary": "growth-os + affiliate-os：转化、验证、包装",
        "skills": ["growth-os", "affiliate-os"],
        "suggested_personality": "grow",
    },
    {
        "id": "ops_exec",
        "label": "运营执行",
        "summary": "ops-os + automation-os + browser-ops：SOP、自动化、实操",
        "skills": ["ops-os", "automation-os", "browser-ops"],
        "suggested_personality": "ops",
    },
    {
        "id": "board_review",
        "label": "董事评审",
        "summary": "board-review + retro：方向、投入、停损",
        "skills": ["board-review", "retro"],
        "suggested_personality": "board",
    },
)
DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB = max(
    1024,
    int(os.getenv("HERMES_MODAL_CHAT_QUEUE_WORKER_MEMORY_MB", "3072")),
)
DEFAULT_CHAT_QUEUE_WORKER_CPU = max(
    0.5,
    float(os.getenv("HERMES_MODAL_CHAT_QUEUE_WORKER_CPU", "1")),
)
DEFAULT_FEISHU_INGRESS_WORKER_MEMORY_MB = max(
    256,
    int(os.getenv("HERMES_MODAL_FEISHU_INGRESS_WORKER_MEMORY_MB", "512")),
)
DEFAULT_FEISHU_INGRESS_WORKER_CPU = max(
    0.25,
    float(os.getenv("HERMES_MODAL_FEISHU_INGRESS_WORKER_CPU", "0.25")),
)
DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB = max(
    128,
    int(os.getenv("HERMES_MODAL_FEISHU_ACK_REACTION_WORKER_MEMORY_MB", "256")),
)
DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU = max(
    0.1,
    float(os.getenv("HERMES_MODAL_FEISHU_ACK_REACTION_WORKER_CPU", "0.25")),
)
DEFAULT_FEISHU_INGRESS_SCALEDOWN_WINDOW_SECONDS = max(
    60,
    int(os.getenv("HERMES_MODAL_FEISHU_INGRESS_SCALEDOWN_WINDOW_SECONDS", "180")),
)
DEFAULT_FEISHU_ACK_REACTION_SCALEDOWN_WINDOW_SECONDS = max(
    60,
    int(os.getenv("HERMES_MODAL_FEISHU_ACK_REACTION_SCALEDOWN_WINDOW_SECONDS", "300")),
)
DEFAULT_FEISHU_ACK_REACTION_MODE = (
    str(os.getenv("HERMES_FEISHU_ACK_REACTION_MODE") or "inline").strip().lower() or "inline"
)
SUPPORTED_FEISHU_ACK_REACTION_MODES = ("inline", "spawn", "off")
DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS = max(
    50,
    int(os.getenv("HERMES_FEISHU_ACK_REACTION_INLINE_BUDGET_MS", "600")),
)
DEFAULT_FEISHU_ACK_REACTION_REQUEST_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("HERMES_FEISHU_ACK_REACTION_REQUEST_TIMEOUT_SECONDS", "3.0")),
)
DEFAULT_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS = max(
    60,
    int(os.getenv("HERMES_MODAL_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS", "420")),
)
DEFAULT_CHAT_QUEUE_WARMUP_TTL_SECONDS = max(
    15,
    int(os.getenv("HERMES_MODAL_CHAT_QUEUE_WARMUP_TTL_SECONDS", "45")),
)
DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS = max(
    3,
    int(os.getenv("HERMES_MODAL_CHAT_QUEUE_WARMUP_WAIT_SECONDS", "20")),
)
DEFAULT_CHAT_QUEUE_LINGER_SECONDS = max(
    0.0,
    float(os.getenv("HERMES_MODAL_CHAT_QUEUE_LINGER_SECONDS", "6")),
)
DEFAULT_CHAT_QUEUE_ACTIVE_CLAIM_SKIP_SECONDS = max(
    0,
    int(os.getenv("HERMES_MODAL_CHAT_ACTIVE_CLAIM_SKIP_SECONDS", "30")),
)
DEFAULT_CHAT_QUEUE_STALE_CLAIM_TAKEOVER_SECONDS = max(
    DEFAULT_CHAT_QUEUE_ACTIVE_CLAIM_SKIP_SECONDS + 30,
    int(os.getenv("HERMES_MODAL_CHAT_STALE_CLAIM_TAKEOVER_SECONDS", "180")),
)
DEFAULT_FEISHU_RECENT_SPAWN_SKIP_SECONDS = max(
    0.0,
    float(os.getenv("HERMES_MODAL_FEISHU_RECENT_SPAWN_SKIP_SECONDS", "12")),
)
DEFAULT_FEISHU_MESSAGE_QUEUE_MAX_AGE_SECONDS = max(
    30,
    int(os.getenv("HERMES_MODAL_FEISHU_MESSAGE_QUEUE_MAX_AGE_SECONDS", "180")),
)
DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS = max(
    0.1,
    float(os.getenv("HERMES_MODAL_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS", "0.8")),
)
DEFAULT_WEB_APP_SCALEDOWN_WINDOW_SECONDS = max(
    0,
    int(os.getenv("HERMES_MODAL_WEB_APP_SCALEDOWN_WINDOW_SECONDS", "60")),
)
WEB_APP_MEMORY_SNAPSHOT_ENABLED = str(
    os.getenv("HERMES_MODAL_WEB_APP_MEMORY_SNAPSHOT_ENABLED", "false")
).strip().lower() not in {"0", "false", "no", "off", ""}
FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED = str(
    os.getenv("HERMES_MODAL_FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED", "false")
).strip().lower() not in {"0", "false", "no", "off", ""}
FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED = str(
    os.getenv("HERMES_MODAL_FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED", "false")
).strip().lower() not in {"0", "false", "no", "off", ""}
CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED = str(
    os.getenv("HERMES_MODAL_CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED", "false")
).strip().lower() not in {"0", "false", "no", "off", ""}
CHAT_WORKER_PREINIT_FEISHU_RUNTIME_ENABLED = str(
    os.getenv("HERMES_MODAL_CHAT_WORKER_PREINIT_FEISHU_RUNTIME", "true")
).strip().lower() not in {"0", "false", "no", "off", ""}
DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS = max(
    15.0,
    float(os.getenv("HERMES_MODAL_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS", "120")),
)
DEFAULT_FEISHU_INLINE_FIRST_RESPONSE_BUDGET_MS = max(
    250,
    int(os.getenv("HERMES_FEISHU_INLINE_FIRST_RESPONSE_BUDGET_MS", "2500")),
)
DEFAULT_FEISHU_PROVIDER_FIRST_TOKEN_BUDGET_MS = max(
    250,
    int(os.getenv("HERMES_FEISHU_PROVIDER_FIRST_TOKEN_BUDGET_MS", "1800")),
)
DEFAULT_FEISHU_INLINE_WORKER_HARD_BUDGET_MS = max(
    DEFAULT_FEISHU_INLINE_FIRST_RESPONSE_BUDGET_MS,
    int(os.getenv("HERMES_FEISHU_INLINE_WORKER_HARD_BUDGET_MS", "4000")),
)
DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB = max(
    512,
    int(os.getenv("HERMES_MODAL_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB", "1536")),
)
DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU = max(
    0.25,
    float(os.getenv("HERMES_MODAL_FEISHU_BACKGROUND_EXEC_WORKER_CPU", "0.5")),
)
DEFAULT_FEISHU_BACKGROUND_EXEC_SCALEDOWN_WINDOW_SECONDS = max(
    60,
    int(os.getenv("HERMES_MODAL_FEISHU_BACKGROUND_EXEC_SCALEDOWN_WINDOW_SECONDS", "180")),
)
DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY = (
    str(os.getenv("HERMES_FEISHU_MESSAGE_INGRESS_STRATEGY") or "inline_enqueue_spawn").strip().lower()
    or "inline_enqueue_spawn"
)
SUPPORTED_FEISHU_MESSAGE_INGRESS_STRATEGIES = (
    "inline_enqueue_spawn",
    "spawn_process_feishu_event",
    "spawn_process_feishu_message_inline",
)
LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES = {
    "spawn_process_feishu_event": "inline_enqueue_spawn",
}
DEFAULT_CRON_QUEUE_NAME = os.getenv("HERMES_MODAL_CRON_QUEUE_NAME", f"{APP_NAME}-cron-queue")
DEFAULT_CRON_QUEUE_CLAIM_TTL_SECONDS = int(os.getenv("HERMES_MODAL_CRON_QUEUE_CLAIM_TTL_SECONDS", "1800"))
DEFAULT_CRON_QUEUE_BATCH_SIZE = int(os.getenv("HERMES_MODAL_CRON_QUEUE_BATCH_SIZE", "8"))
DEFAULT_CRON_QUEUE_WORKERS = int(os.getenv("HERMES_MODAL_CRON_QUEUE_WORKERS", "2"))
DEFAULT_CRON_QUEUE_MAX_JOBS_PER_WORKER = max(
    1,
    int(os.getenv("HERMES_MODAL_CRON_QUEUE_MAX_JOBS_PER_WORKER", "4")),
)
DEFAULT_SESSION_ROUTE_TTL_SECONDS = int(os.getenv("HERMES_SESSION_ROUTE_TTL_SECONDS", "7200"))
_ANTHROPIC_DATED_MODEL_RE = re.compile(r"^(anthropic\/.+)-(\d{8})$")
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"^\d{6,}:[A-Za-z0-9_-]{20,}$")
ROUTING_STATE_PATH = DATA_ROOT / "free_model_routing.json"
CHAT_QUEUE_CLAIMS_PATH = DATA_ROOT / "chat_queue_claims.json"
CHAT_QUEUE_WARMUPS_PATH = DATA_ROOT / "chat_queue_warmups.json"
CRON_QUEUE_CLAIMS_PATH = DATA_ROOT / "cron_queue_claims.json"
TELEGRAM_WEBHOOK_SYNC_STATE_PATH = DATA_ROOT / "telegram_webhook_sync.json"
DEFAULT_FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS = int(
    os.getenv("FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS", "1800")
)
DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS = int(
    os.getenv("FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS", "21600")
)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_POPULAR_MODELS_URL = "https://build.nvidia.com/models?orderBy=weightPopular%3ADESC"
ROUTING_REFRESH_TTL_SECONDS = 24 * 60 * 60
DEFAULT_TELEGRAM_WEBHOOK_SYNC_BACKOFF_SECONDS = int(
    os.getenv("TELEGRAM_WEBHOOK_SYNC_BACKOFF_SECONDS", "300")
)
DEFAULT_MAINTENANCE_HEARTBEAT_MINUTES = max(
    1,
    int(os.getenv("HERMES_MODAL_MAINTENANCE_HEARTBEAT_MINUTES", "5")),
)
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9WlH0xQAAAAASUVORK5CYII="
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
INLINE_FAST_COMMAND_CANONICALS = {
    "help",
    "commands",
    "status",
    "model",
    "provider",
}
_FEISHU_ACK_REACTION_EMOJI = "OK"

_UPDATE_LOCK = threading.Lock()
_FEISHU_EVENT_LOCK = threading.Lock()
_CHAT_QUEUE_LOCK = threading.Lock()
_CRON_QUEUE_LOCK = threading.Lock()
_TELEGRAM_RUNTIME_LOCK: asyncio.Lock | None = None
_TELEGRAM_RUNTIME: "_TelegramGatewayRuntime | None" = None
_FEISHU_RUNTIME_LOCK: asyncio.Lock | None = None
_FEISHU_RUNTIME: "_FeishuGatewayRuntime | None" = None
_FEISHU_INTERNAL_RUNTIME_LOCK: asyncio.Lock | None = None
_FEISHU_INTERNAL_RUNTIME: "_FeishuGatewayRuntime | None" = None
_FEISHU_BOT_IDENTITY_CACHE_LOCK = threading.Lock()
_FEISHU_INTERNAL_RESULT_FILE_LOCK = threading.Lock()
_FEISHU_INTERNAL_RESULT_FILES: dict[str, dict[str, Any]] = {}
_RECENT_CHAT_WORKER_SPAWNS_LOCK = threading.Lock()
_RECENT_CHAT_WORKER_SPAWNS: dict[str, float] = {}
_QQ_RUNTIME_LOCK: asyncio.Lock | None = None
_QQ_RUNTIME: "_QQGatewayRuntime | None" = None
_CHAT_QUEUE_PREWARMED = False
CHAT_QUEUE = None
CRON_QUEUE = None
MODAL_VOLUME = None
DEFAULT_FEISHU_INTERNAL_RESULT_FILE_TTL_SECONDS = max(
    60,
    int(os.getenv("HERMES_FEISHU_INTERNAL_RESULT_FILE_TTL_SECONDS", "1800")),
)

_MODAL_PUBLIC_WEBHOOK_PLATFORMS = ("telegram", "feishu", "qq")


if JSONResponse is not None:
    class _FeishuTrackedJSONResponse(JSONResponse):
        def __init__(
            self,
            content: Any,
            *,
            payload: dict[str, Any],
            request_started_at: float,
            ack_kind: str,
            partition: str | None = None,
            lane: str | None = None,
            queue_depth: int | None = None,
            reason: str | None = None,
            ingress_strategy: str | None = None,
            phase_timings: dict[str, Any] | None = None,
            status_code: int = 200,
            headers: Mapping[str, str] | None = None,
        ) -> None:
            super().__init__(content=content, status_code=status_code, headers=dict(headers or {}))
            self._feishu_payload = payload
            self._feishu_request_started_at = request_started_at
            self._feishu_ack_kind = ack_kind
            self._feishu_partition = partition or ""
            self._feishu_lane = lane or ""
            self._feishu_queue_depth = queue_depth
            self._feishu_reason = reason or ""
            self._feishu_ingress_strategy = ingress_strategy or ""
            self._feishu_phase_timings = _normalize_phase_timings(phase_timings)
            self._feishu_response_logged = False

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                if self._feishu_response_logged:
                    return
                self._feishu_response_logged = True
                response_elapsed_ms = int((time.perf_counter() - self._feishu_request_started_at) * 1000)
                _append_feishu_trace(
                    "webhook.response_sent",
                    self._feishu_payload,
                    ack_kind=self._feishu_ack_kind,
                    response_elapsed_ms=response_elapsed_ms,
                    partition=self._feishu_partition,
                    lane=self._feishu_lane,
                    queue_depth=self._feishu_queue_depth,
                    reason=self._feishu_reason,
                    ingress_strategy=self._feishu_ingress_strategy,
                    phase_timings=self._feishu_phase_timings,
                )
                event_id, event_type = _extract_feishu_event_metadata(self._feishu_payload)
                logger.warning(
                    "[Feishu] webhook response sent event_type=%s event_id=%s ack_kind=%s response_elapsed_ms=%s partition=%s lane=%s queue_depth=%s reason=%s ingress_strategy=%s",
                    event_type or "unknown",
                    event_id or "none",
                    self._feishu_ack_kind,
                    response_elapsed_ms,
                    self._feishu_partition or "none",
                    self._feishu_lane or "none",
                    self._feishu_queue_depth,
                    self._feishu_reason,
                    self._feishu_ingress_strategy,
                )
else:  # pragma: no cover - used only when FastAPI isn't importable
    _FeishuTrackedJSONResponse = None  # type: ignore[assignment]


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
        {
            "label": "OpenRouter 精选",
            "event_key": "provider_openrouter_featured",
            "description": "OpenRouter featured model picker",
            "recommended": False,
        },
        {
            "label": "OpenRouter 最近",
            "event_key": "provider_openrouter_recent",
            "description": "OpenRouter recent model picker",
            "recommended": False,
        },
        {
            "label": "OpenRouter 性能",
            "event_key": "provider_openrouter_performance",
            "description": "OpenRouter performance model picker",
            "recommended": False,
        },
        {
            "label": "NVIDIA 精选",
            "event_key": "provider_nvidia_featured",
            "description": "NVIDIA featured model picker",
            "recommended": False,
        },
        {
            "label": "NVIDIA 最近",
            "event_key": "provider_nvidia_recent",
            "description": "NVIDIA recent model picker",
            "recommended": False,
        },
        {
            "label": "NVIDIA 性能",
            "event_key": "provider_nvidia_performance",
            "description": "NVIDIA performance model picker",
            "recommended": False,
        },
        {
            "label": "人格特质",
            "event_key": "personality_picker",
            "description": "打开人格特质切换卡片",
            "recommended": True,
        },
        {
            "label": "技能组合",
            "event_key": "skill_combo_picker",
            "description": "打开预制技能组合卡片",
            "recommended": True,
        },
        {
            "label": "Command Center",
            "event_key": "command_center",
            "description": "Open Hermes slash command center card",
            "recommended": True,
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
            original_model_default = ""
            original_model_config = config_payload.get("model")
            if isinstance(original_model_config, dict):
                original_model_default = str(original_model_config.get("default") or "").strip()
            config_payload = _expand_env_vars(config_payload)
            model_config = config_payload.get("model")
            if isinstance(model_config, dict):
                default_model = str(model_config.get("default") or "").strip()
                if not default_model or default_model.startswith("${"):
                    model_config["default"] = os.getenv("DEFAULT_MODEL", "openrouter/free")
            if _is_dynamic_free_route_alias(original_model_default):
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


def _first_non_empty_str(*values: Any) -> str:
    for value in values:
        candidate = str(value or "").strip()
        if candidate:
            return candidate
    return ""


def _collect_feishu_actor_id(event: dict[str, Any]) -> str:
    sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
    sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
    operator = event.get("operator") if isinstance(event.get("operator"), dict) else {}
    operator_id = operator.get("operator_id") if isinstance(operator.get("operator_id"), dict) else {}
    user = event.get("user") if isinstance(event.get("user"), dict) else {}
    user_id = event.get("user_id") if isinstance(event.get("user_id"), dict) else {}
    for container in (sender_id, sender, operator_id, operator, user_id, user, event):
        if not isinstance(container, dict):
            continue
        actor_id = _first_non_empty_str(
            container.get("open_id"),
            container.get("user_id"),
            container.get("employee_id"),
            container.get("union_id"),
        )
        if actor_id:
            return actor_id
    return ""


def _collect_feishu_chat_id(event: dict[str, Any]) -> str:
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    context = event.get("context") if isinstance(event.get("context"), dict) else {}
    chat = event.get("chat") if isinstance(event.get("chat"), dict) else {}
    return _first_non_empty_str(
        message.get("chat_id"),
        chat.get("chat_id"),
        chat.get("open_chat_id"),
        context.get("open_chat_id"),
        context.get("chat_id"),
        event.get("chat_id"),
        event.get("open_chat_id"),
    )


def _extract_feishu_trace_context(payload: dict[str, Any]) -> dict[str, Any]:
    header = payload.get("header") or {}
    event = payload.get("event") or {}
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    chat = event.get("chat") if isinstance(event.get("chat"), dict) else {}
    sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
    sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
    operator = event.get("operator") if isinstance(event.get("operator"), dict) else {}
    operator_id = operator.get("operator_id") if isinstance(operator.get("operator_id"), dict) else {}
    user = event.get("user") if isinstance(event.get("user"), dict) else {}
    user_id = event.get("user_id") if isinstance(event.get("user_id"), dict) else {}
    return {
        "type": str(payload.get("type") or "").strip(),
        "event_type": str(header.get("event_type") or payload.get("event_type") or "").strip(),
        "event_id": str(header.get("event_id") or payload.get("event_id") or "").strip(),
        "message_id": str(message.get("message_id") or "").strip(),
        "chat_type": str(message.get("chat_type") or chat.get("chat_type") or event.get("chat_type") or "").strip(),
        "chat_id": _collect_feishu_chat_id(event),
        "sender_open_id": _first_non_empty_str(
            sender_id.get("open_id"),
            operator_id.get("open_id"),
            operator.get("open_id"),
            user_id.get("open_id"),
            user.get("open_id"),
            event.get("open_id"),
        ),
        "sender_user_id": _first_non_empty_str(
            sender_id.get("user_id"),
            operator_id.get("user_id"),
            operator.get("user_id"),
            user_id.get("user_id"),
            user.get("user_id"),
            event.get("user_id") if isinstance(event.get("user_id"), str) else "",
        ),
        "trace_token": _extract_feishu_trace_token(payload),
    }


def _build_feishu_snapshot_profile_state() -> dict[str, Any]:
    snapshot_flags = {
        "web_app_enabled": bool(WEB_APP_MEMORY_SNAPSHOT_ENABLED),
        "feishu_ingress_enabled": bool(FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED),
        "feishu_ack_reaction_enabled": bool(FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED),
        "chat_queue_enabled": bool(CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED),
    }
    enabled_names = [name for name, enabled in snapshot_flags.items() if enabled]
    snapshot_profile = str(os.getenv("HERMES_MODAL_SNAPSHOT_PROFILE_LABEL") or "").strip().lower()
    if not snapshot_profile:
        snapshot_profile = (
            "none"
            if not enabled_names
            else "web{web}_ingress{ingress}_ack{ack}_chat{chat}".format(
                web=int(snapshot_flags["web_app_enabled"]),
                ingress=int(snapshot_flags["feishu_ingress_enabled"]),
                ack=int(snapshot_flags["feishu_ack_reaction_enabled"]),
                chat=int(snapshot_flags["chat_queue_enabled"]),
            )
        )
    experiment_label = str(os.getenv("HERMES_FEISHU_PERF_EXPERIMENT_LABEL") or "").strip().lower()
    if not experiment_label:
        experiment_label = snapshot_profile
    return {
        "app_name": APP_NAME,
        "experiment_label": experiment_label,
        "snapshot_profile": snapshot_profile,
        "snapshot_flags": snapshot_flags,
    }


def _append_feishu_trace(stage: str, payload: dict[str, Any], **extra: Any) -> None:
    trace = {
        "ts": int(time.time()),
        "stage": stage,
        **_extract_feishu_trace_context(payload),
        **_build_feishu_snapshot_profile_state(),
    }
    if extra:
        trace.update(extra)
    line = json.dumps(trace, ensure_ascii=False)
    with _FEISHU_EVENT_LOCK:
        FEISHU_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEISHU_TRACE_PATH.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")
        # Modal Volumes require an explicit commit before other containers can
        # observe newly appended trace rows. Restrict commits to key terminal
        # stages so recent perf summaries stay fresh without paying a commit
        # penalty on every intermediate log line.
        if stage in {
            "webhook.response_sent",
            "webhook.ack_reaction",
            "dispatch.done",
            "dispatch.error",
            "inline_message.done",
            "background_exec.done",
            "worker.done",
            "worker.error",
        }:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                _sync_modal_volume(commit=True)
            else:
                try:
                    asyncio.create_task(_sync_modal_volume_async(commit=True))
                except Exception:
                    _sync_modal_volume(commit=True)


def _capture_phase_elapsed(phase_timings: dict[str, int], key: str, started_at: float) -> int:
    elapsed_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    phase_timings[key] = elapsed_ms
    return elapsed_ms


def _normalize_phase_timings(phase_timings: dict[str, Any] | None) -> dict[str, int]:
    normalized: dict[str, int] = {}
    if not isinstance(phase_timings, dict):
        return normalized
    for key, value in phase_timings.items():
        if value is None:
            continue
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _with_feishu_internal_meta(payload: dict[str, Any], **meta: Any) -> dict[str, Any]:
    cloned = dict(payload or {})
    existing = cloned.get("_hermes_ingress")
    normalized = dict(existing) if isinstance(existing, dict) else {}
    for key, value in meta.items():
        if value is not None:
            normalized[str(key)] = value
    cloned["_hermes_ingress"] = normalized
    return cloned


def _normalize_epoch_ms(value: Any) -> int | None:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    # Ignore monotonic/perf-counter style numbers and obviously invalid wall-clock values.
    if normalized < 1_600_000_000_000:
        return None
    return normalized


def _resolve_feishu_request_started_at_ms(
    payload: Mapping[str, Any] | None,
    request_started_at_ms: int | None = None,
) -> int | None:
    normalized = _normalize_epoch_ms(request_started_at_ms)
    if normalized is not None:
        return normalized
    ingress_meta = payload.get("_hermes_ingress") if isinstance(payload, Mapping) else None
    if isinstance(ingress_meta, Mapping):
        normalized = _normalize_epoch_ms(ingress_meta.get("ack_reaction_requested_at_ms"))
        if normalized is not None:
            return normalized
    return None


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
) -> JSONResponse:
    ack_elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
    normalized_phase_timings = _normalize_phase_timings(phase_timings)
    trace_token = _extract_feishu_trace_token(payload)
    _append_feishu_trace(
        "webhook.ack",
        payload,
        ack_kind=ack_kind,
        ack_elapsed_ms=ack_elapsed_ms,
        partition=partition or "",
        lane=lane or "",
        queue_depth=queue_depth,
        reason=reason or "",
        ingress_strategy=ingress_strategy or "",
        phase_timings=normalized_phase_timings,
    )
    event_id, event_type = _extract_feishu_event_metadata(payload)
    logger.warning(
        "[Feishu] webhook ack event_type=%s event_id=%s trace_token=%s ack_kind=%s ack_elapsed_ms=%s partition=%s lane=%s queue_depth=%s reason=%s ingress_strategy=%s phase_timings=%s",
        event_type or "unknown",
        event_id or "none",
        trace_token or "none",
        ack_kind,
        ack_elapsed_ms,
        partition or "none",
        lane or "none",
        queue_depth,
        reason or "",
        ingress_strategy or "",
        json.dumps(normalized_phase_timings, ensure_ascii=False, sort_keys=True),
    )
    response_cls = _FeishuTrackedJSONResponse or JSONResponse
    return response_cls(
        body,
        payload=payload,
        request_started_at=request_started_at,
        ack_kind=ack_kind,
        partition=partition,
        lane=lane,
        queue_depth=queue_depth,
        reason=reason,
        ingress_strategy=ingress_strategy,
        phase_timings=normalized_phase_timings,
    )


def _extract_feishu_message_read_event_info(payload: Mapping[str, Any]) -> dict[str, Any]:
    header = payload.get("header") if isinstance(payload, Mapping) else {}
    event = payload.get("event") if isinstance(payload, Mapping) else {}
    header = header if isinstance(header, Mapping) else {}
    event = event if isinstance(event, Mapping) else {}
    reader = event.get("reader") if isinstance(event.get("reader"), Mapping) else {}
    reader = reader if isinstance(reader, Mapping) else {}
    reader_id = reader.get("reader_id") if isinstance(reader.get("reader_id"), Mapping) else {}
    reader_id = reader_id if isinstance(reader_id, Mapping) else {}

    message_ids: list[str] = []
    raw_message_ids = event.get("message_id_list")
    if isinstance(raw_message_ids, list):
        for item in raw_message_ids:
            value = str(item or "").strip()
            if value:
                message_ids.append(value)

    return {
        "event_type": str(header.get("event_type") or payload.get("event_type") or "").strip(),
        "event_id": str(header.get("event_id") or payload.get("event_id") or "").strip(),
        "reader_open_id": _first_non_empty_str(
            reader_id.get("open_id"),
            reader.get("open_id"),
            event.get("open_id"),
        ),
        "reader_user_id": _first_non_empty_str(
            reader_id.get("user_id"),
            reader.get("user_id"),
            event.get("user_id") if isinstance(event.get("user_id"), str) else "",
        ),
        "reader_union_id": _first_non_empty_str(
            reader_id.get("union_id"),
            reader.get("union_id"),
        ),
        "tenant_key": _first_non_empty_str(
            reader.get("tenant_key"),
            event.get("tenant_key"),
        ),
        "read_time": int(reader.get("read_time") or event.get("read_time") or 0),
        "message_id_list": message_ids,
        "message_count": len(message_ids),
    }


def _read_feishu_trace(limit: int = 100) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    _sync_modal_volume(reload=True)
    if not FEISHU_TRACE_PATH.exists():
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


def _summarize_numeric_series(values: list[int | float]) -> dict[str, Any]:
    series = [float(value) for value in values if value is not None]
    if not series:
        return {"count": 0}
    series.sort()

    def _percentile(p: float) -> float:
        if len(series) == 1:
            return series[0]
        idx = (len(series) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(series) - 1)
        frac = idx - lo
        return series[lo] * (1 - frac) + series[hi] * frac

    return {
        "count": len(series),
        "avg": round(sum(series) / len(series), 1),
        "p50": round(_percentile(0.5), 1),
        "p90": round(_percentile(0.9), 1),
        "max": round(max(series), 1),
    }


def _pick_feishu_stage_row(rows: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    stage_rows = [row for row in rows if str(row.get("stage") or "") == stage]
    if not stage_rows:
        return None
    if stage in {"webhook.ack", "webhook.response_sent"}:
        for row in stage_rows:
            if str(row.get("ack_kind") or "").strip().lower() != "duplicate":
                return row
    return stage_rows[-1]


def _count_by_normalized_value(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        normalized = str(value or "").strip() or "unspecified"
        counts[normalized] = counts.get(normalized, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


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
    now_ts = int(time.time())
    min_ts = now_ts - max(int(since_seconds or 0), 0) if since_seconds else None
    normalized_event_type = str(event_type or "").strip()
    normalized_label = str(experiment_label or "").strip().lower()
    normalized_app_name = str(app_name_filter or "").strip().lower()
    normalized_snapshot_profile = str(snapshot_profile or "").strip().lower()

    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_ts = int(row.get("ts") or 0)
        if min_ts is not None and row_ts and row_ts < min_ts:
            continue
        if normalized_event_type and str(row.get("event_type") or "").strip() != normalized_event_type:
            continue
        if normalized_label and str(row.get("experiment_label") or "").strip().lower() != normalized_label:
            continue
        if normalized_app_name and str(row.get("app_name") or "").strip().lower() != normalized_app_name:
            continue
        if normalized_snapshot_profile and str(row.get("snapshot_profile") or "").strip().lower() != normalized_snapshot_profile:
            continue
        filtered_rows.append(row)

    events: dict[str, list[dict[str, Any]]] = {}
    for row in filtered_rows:
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        events.setdefault(event_id, []).append(row)

    metric_series: dict[str, list[float]] = {}
    event_summaries: list[dict[str, Any]] = []
    duplicate_only_events = 0

    for event_id, event_rows in events.items():
        ack_row = _pick_feishu_stage_row(event_rows, "webhook.ack")
        if ack_row is None:
            continue
        ack_kind = str(ack_row.get("ack_kind") or "").strip().lower()
        if ack_kind == "duplicate" and not include_duplicates:
            duplicate_only_events += 1
            continue

        response_row = _pick_feishu_stage_row(event_rows, "webhook.response_sent")
        dispatch_done_row = _pick_feishu_stage_row(event_rows, "dispatch.done")
        inline_done_row = _pick_feishu_stage_row(event_rows, "inline_message.done")
        background_exec_done_row = _pick_feishu_stage_row(event_rows, "background_exec.done")
        gateway_handler_done_row = _pick_feishu_stage_row(event_rows, "gateway.handler.done")
        gateway_send_done_row = _pick_feishu_stage_row(event_rows, "gateway.send.done")
        worker_start_row = _pick_feishu_stage_row(event_rows, "worker.start")
        worker_done_row = _pick_feishu_stage_row(event_rows, "worker.done")
        ack_reaction_row = _pick_feishu_stage_row(event_rows, "webhook.ack_reaction")

        ack_phase_timings = _normalize_phase_timings(ack_row.get("phase_timings"))
        dispatch_phase_timings = _normalize_phase_timings(
            dispatch_done_row.get("phase_timings") if isinstance(dispatch_done_row, dict) else {}
        )

        event_summary = {
            "event_id": event_id,
            "event_type": str(ack_row.get("event_type") or "").strip(),
            "app_name": str(ack_row.get("app_name") or "").strip(),
            "experiment_label": str(ack_row.get("experiment_label") or "").strip(),
            "snapshot_profile": str(ack_row.get("snapshot_profile") or "").strip(),
            "ack_kind": ack_kind or "unspecified",
            "execution_mode": str(
                (inline_done_row or {}).get("execution_mode")
                or (background_exec_done_row or {}).get("execution_mode")
                or ""
            ).strip()
            or "unspecified",
            "handoff_reason": str(
                (inline_done_row or {}).get("handoff_reason")
                or (background_exec_done_row or {}).get("handoff_reason")
                or ""
            ).strip()
            or "unspecified",
            "ingress_strategy": str(
                ack_row.get("ingress_strategy")
                or (worker_start_row or {}).get("ingress_strategy")
                or "unspecified"
            ).strip()
            or "unspecified",
            "ack_elapsed_ms": ack_row.get("ack_elapsed_ms"),
            "response_elapsed_ms": (response_row or {}).get("response_elapsed_ms"),
            "queue_latency_ms": (worker_start_row or {}).get("queue_latency_ms"),
            "worker_elapsed_ms": (worker_done_row or {}).get("worker_elapsed_ms"),
            "inline_elapsed_ms": (inline_done_row or {}).get("inline_elapsed_ms"),
            "background_exec_elapsed_ms": (background_exec_done_row or {}).get("worker_elapsed_ms"),
            "gateway_handler_elapsed_ms": (gateway_handler_done_row or {}).get("handler_elapsed_ms"),
            "gateway_send_elapsed_ms": (gateway_send_done_row or {}).get("send_elapsed_ms"),
            "gateway_send_success": (gateway_send_done_row or {}).get("send_success"),
            "ack_reaction_total_elapsed_ms": (ack_reaction_row or {}).get("total_elapsed_ms"),
            "ack_reaction_from_request_elapsed_ms": (ack_reaction_row or {}).get("from_request_elapsed_ms"),
            "message_handle_elapsed_ms": dispatch_phase_timings.get("message_handle_elapsed_ms"),
            "background_tasks_elapsed_ms": dispatch_phase_timings.get("background_tasks_elapsed_ms"),
            "pending_batches_elapsed_ms": dispatch_phase_timings.get("pending_batches_elapsed_ms"),
            "background_send_elapsed_ms": (background_exec_done_row or {}).get("background_send_elapsed_ms"),
            "ack_reaction_inline_elapsed_ms": ack_phase_timings.get("ack_reaction_inline_elapsed_ms"),
            "ack_reaction_schedule_elapsed_ms": ack_phase_timings.get("ack_reaction_schedule_elapsed_ms"),
            "dedupe_elapsed_ms": ack_phase_timings.get("dedupe_elapsed_ms"),
            "context_extract_elapsed_ms": ack_phase_timings.get("context_extract_elapsed_ms"),
            "handoff_wait_elapsed_ms": ack_phase_timings.get("handoff_wait_elapsed_ms"),
            "handoff_schedule_wait_elapsed_ms": ack_phase_timings.get("handoff_schedule_wait_elapsed_ms"),
            "inline_fast_extract_elapsed_ms": ack_phase_timings.get("inline_fast_extract_elapsed_ms"),
            "parse_verify_elapsed_ms": ack_phase_timings.get("parse_verify_elapsed_ms"),
            "read_event_extract_elapsed_ms": ack_phase_timings.get("read_event_extract_elapsed_ms"),
            "worker_spawn_elapsed_ms": ack_phase_timings.get("worker_spawn_elapsed_ms"),
            "schedule_claim_elapsed_ms": ack_phase_timings.get("schedule_claim_elapsed_ms"),
            "spawn_rpc_elapsed_ms": ack_phase_timings.get("spawn_rpc_elapsed_ms"),
        }
        event_summaries.append(event_summary)
        for key, value in event_summary.items():
            if isinstance(value, (int, float)):
                metric_series.setdefault(key, []).append(float(value))

    metrics_summary = {
        metric_name: _summarize_numeric_series(values)
        for metric_name, values in sorted(metric_series.items())
        if values
    }
    event_summaries.sort(key=lambda item: str(item.get("event_id") or ""))

    return {
        "status": "ok",
        "window": {
            "since_seconds": int(since_seconds or 0),
            "event_type": normalized_event_type,
            "experiment_label": normalized_label,
            "app_name": normalized_app_name,
            "snapshot_profile": normalized_snapshot_profile,
            "include_duplicates": bool(include_duplicates),
        },
        "trace_row_count": len(filtered_rows),
        "event_count": len(event_summaries),
        "duplicate_only_event_count": duplicate_only_events,
        "by_ingress_strategy": _count_by_normalized_value([str(item.get("ingress_strategy") or "") for item in event_summaries]),
        "by_execution_mode": _count_by_normalized_value([str(item.get("execution_mode") or "") for item in event_summaries]),
        "by_handoff_reason": _count_by_normalized_value([str(item.get("handoff_reason") or "") for item in event_summaries]),
        "by_experiment_label": _count_by_normalized_value([str(item.get("experiment_label") or "") for item in event_summaries]),
        "by_snapshot_profile": _count_by_normalized_value([str(item.get("snapshot_profile") or "") for item in event_summaries]),
        "by_app_name": _count_by_normalized_value([str(item.get("app_name") or "") for item in event_summaries]),
        "ack_kind_counts": _count_by_normalized_value([str(item.get("ack_kind") or "") for item in event_summaries]),
        "metrics": metrics_summary,
        "events": event_summaries,
    }


def _build_feishu_ingress_strategy_debug_state(limit: int = 500) -> dict[str, Any]:
    rows = _read_feishu_trace(limit=max(limit, 1))
    configured_strategy = str(DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY or "").strip().lower()
    effective_strategy = _resolve_feishu_message_ingress_strategy({}, {})
    ack_rows = [row for row in rows if str(row.get("stage") or "") == "webhook.ack"]
    by_strategy: dict[str, dict[str, list[float]]] = {}
    for row in ack_rows:
        strategy = str(row.get("ingress_strategy") or "").strip() or "unspecified"
        bucket = by_strategy.setdefault(strategy, {"ack_elapsed_ms": []})
        ack_elapsed = row.get("ack_elapsed_ms")
        if isinstance(ack_elapsed, (int, float)):
            bucket["ack_elapsed_ms"].append(float(ack_elapsed))

    handoff_rows = [row for row in rows if str(row.get("stage") or "") == "webhook.handoff_spawned"]
    for row in handoff_rows:
        strategy = str(row.get("ingress_strategy") or "").strip() or "unspecified"
        by_strategy.setdefault(strategy, {"ack_elapsed_ms": []})

    queue_rows = [row for row in rows if str(row.get("stage") or "") == "worker.start"]
    for row in queue_rows:
        strategy = str(row.get("ingress_strategy") or "").strip() or "unspecified"
        bucket = by_strategy.setdefault(strategy, {"ack_elapsed_ms": [], "queue_latency_ms": []})
        queue_latency = row.get("queue_latency_ms")
        if isinstance(queue_latency, (int, float)):
            bucket.setdefault("queue_latency_ms", []).append(float(queue_latency))

    summary = {}
    for strategy, metrics in by_strategy.items():
        summary[strategy] = {
            key: _summarize_numeric_series([int(value) for value in values])
            for key, values in metrics.items()
            if values
        }
    return {
        "status": "ok",
        "supported_strategies": list(SUPPORTED_FEISHU_MESSAGE_INGRESS_STRATEGIES),
        "configured_strategy": configured_strategy,
        "effective_strategy": effective_strategy,
        "legacy_aliases": dict(LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES),
        "experiment_candidates": _split_csv(os.getenv("HERMES_FEISHU_MESSAGE_INGRESS_EXPERIMENT")),
        "summary": summary,
        "trace_count": len(rows),
    }


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


def _resolve_provider_runtime_binding(provider_name: str) -> dict[str, Any] | None:
    normalized = str(provider_name or "").strip().lower()
    if not normalized:
        return None
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=normalized) or {}
    except Exception:
        return None

    base_url = str(runtime.get("base_url") or "").strip()
    api_key = str(runtime.get("api_key") or "").strip()
    resolved_provider = str(runtime.get("provider") or normalized).strip().lower() or normalized
    if not base_url or not api_key:
        return None
    return {
        "provider": resolved_provider,
        "base_url": base_url,
        "api_key": api_key,
        "api_mode": str(runtime.get("api_mode") or "").strip(),
    }


def _refresh_free_model_routes(force: bool = False) -> dict[str, Any]:
    _ensure_runtime_dirs()
    existing = _load_routing_state()
    now = int(time.time())
    if not force:
        refreshed_at = int(existing.get("refreshed_at") or 0)
        if refreshed_at and now - refreshed_at < ROUTING_REFRESH_TTL_SECONDS:
            return existing

    providers: dict[str, Any] = {}
    openrouter_runtime = _resolve_provider_runtime_binding("openrouter")
    nvidia_runtime = _resolve_provider_runtime_binding("nvidia")

    try:
        providers["openrouter"] = {
            "base_url": str(
                (openrouter_runtime or {}).get("base_url")
                or os.getenv("OPENROUTER_BASE_URL")
                or DEFAULT_OPENROUTER_BASE_URL
            ).strip(),
            "candidates": _fetch_openrouter_free_model_candidates(),
        }
    except Exception as exc:
        logger.warning("Failed refreshing OpenRouter free models: %s", exc)
        providers["openrouter"] = {
            "base_url": str(
                (openrouter_runtime or {}).get("base_url")
                or os.getenv("OPENROUTER_BASE_URL")
                or DEFAULT_OPENROUTER_BASE_URL
            ).strip(),
            "candidates": ["openrouter/free"],
        }

    nvidia_candidates = _load_nvidia_free_model_candidates()
    if nvidia_candidates:
        providers["nvidia"] = {
            "base_url": str(
                (nvidia_runtime or {}).get("base_url")
                or os.getenv("NVIDIA_BASE_URL")
                or DEFAULT_NVIDIA_BASE_URL
            ).strip(),
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
        runtime_binding = _resolve_provider_runtime_binding(provider_name)
        resolved_provider = str((runtime_binding or {}).get("provider") or provider_name).strip().lower() or provider_name
        base_url = str((runtime_binding or {}).get("base_url") or entry.get("base_url") or "").strip()
        api_key = str((runtime_binding or {}).get("api_key") or "").strip()
        if not api_key:
            if provider_name == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            elif provider_name == "nvidia":
                api_key = (os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY") or "").strip()
        if not api_key or not base_url:
            continue
        for candidate in candidates:
            route = {
                "provider": resolved_provider,
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


def _route_api_key_source(provider_name: str, base_url: str = "") -> str:
    normalized = str(provider_name or "").strip().lower()
    normalized_base_url = str(base_url or "").strip().lower()
    if "gateway.ai.cloudflare.com" in normalized_base_url and normalized in {"openrouter", "nvidia"}:
        return "CLOUDFLARE_API_TOKEN"
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
        "api_key_source": _route_api_key_source(provider_name, route.get("base_url") or ""),
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

    runtime_binding = None
    if provider_name in {"openrouter", "nvidia"}:
        runtime_binding = _resolve_provider_runtime_binding(provider_name)

    api_key = str((runtime_binding or {}).get("api_key") or "").strip()
    base_url = str((runtime_binding or {}).get("base_url") or base_url).strip()
    provider_name = str((runtime_binding or {}).get("provider") or provider_name).strip().lower() or provider_name
    if not api_key and provider_name == "openrouter":
        api_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
    elif not api_key and provider_name == "nvidia":
        api_key = str(os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY") or "").strip()
    elif not api_key and settings.provider == provider_name and settings.api_key:
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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _normalize_cloudflare_ai_gateway_runtime_base_url(raw_url: str | None) -> str | None:
    base_url = str(raw_url or "").strip().rstrip("/")
    if not base_url:
        return None
    lower = base_url.lower()
    if lower.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
        lower = base_url.lower()
    if lower.endswith("/v1/chat/completions"):
        base_url = base_url[: -len("/v1/chat/completions")]
        lower = base_url.lower()
    if not lower.endswith("/compat"):
        base_url = f"{base_url}/compat"
    return base_url.rstrip("/")


def _pick_runtime_api_config() -> tuple[Optional[str], Optional[str], Optional[str]]:
    provider = os.getenv("HERMES_PROVIDER") or None
    cloudflare_base_url = None
    if _env_flag("HERMES_INFERENCE_USE_CLOUDFLARE_AI_GATEWAY", default=False):
        cloudflare_base_url = _normalize_cloudflare_ai_gateway_runtime_base_url(
            os.getenv("CLOUDFLARE_AI_GATEWAY_BASE_URL", DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL)
        )
    using_cloudflare_gateway = bool(cloudflare_base_url) and not os.getenv("HERMES_BASE_URL")
    base_url = (
        os.getenv("HERMES_BASE_URL")
        or cloudflare_base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or None
    )
    api_key_candidates = [os.getenv("HERMES_API_KEY")]
    if using_cloudflare_gateway:
        api_key_candidates.extend(
            [
                os.getenv("CLOUDFLARE_API_TOKEN"),
                os.getenv("CLOUDFLARE_AI_GATEWAY_API_KEY"),
            ]
        )
    api_key_candidates.extend(
        [
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
        ]
    )
    api_key = next((candidate for candidate in api_key_candidates if candidate), None)
    if not provider and not base_url and os.getenv("OPENROUTER_API_KEY"):
        provider = "openrouter"
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    elif not provider and cloudflare_base_url:
        provider = "openrouter"
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
        hermes_home = Path(os.getenv("HERMES_HOME", str(HERMES_HOME_DIR)))
        config_path = hermes_home / "config.yaml"
        provider_config_path = hermes_home / "supermemory.json"
        status["api_key_configured"] = bool(os.getenv("SUPERMEMORY_API_KEY", "").strip())
        status["sdk_available"] = importlib.util.find_spec("supermemory") is not None
        status["container_tag_override"] = os.getenv("SUPERMEMORY_CONTAINER_TAG", "").strip()
        status["config_path"] = str(config_path)
        status["config_present"] = config_path.exists()
        status["provider_config_path"] = str(provider_config_path)
        status["provider_config_present"] = provider_config_path.exists()

    return status


def _build_runtime_bootstrap_debug_state(
    platforms: tuple[str, ...] = ("cli", "feishu", "api_server"),
) -> dict[str, Any]:
    _prepare_runtime_environment()

    payload: dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "hermes_home": os.getenv("HERMES_HOME", str(HERMES_HOME_DIR)),
        "runtime_config_path": str(Path(os.getenv("HERMES_HOME", str(HERMES_HOME_DIR))) / "config.yaml"),
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
    top_level_personalities = (
        config.get("personalities") if isinstance(config.get("personalities"), dict) else {}
    )
    nested_personalities = (
        agent_cfg.get("personalities") if isinstance(agent_cfg.get("personalities"), dict) else {}
    )
    payload["agent"] = {
        "default_model": str(model_cfg.get("default") or "").strip(),
        "default_personality": str(agent_cfg.get("personality") or "").strip(),
        "available_personalities": sorted(
            {str(name).strip() for name in (*top_level_personalities.keys(), *nested_personalities.keys()) if str(name).strip()}
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
                {
                    "key": key,
                    "label": label,
                    "description": description,
                }
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
        configured_platform_toolsets = (
            config.get("platform_toolsets") if isinstance(config.get("platform_toolsets"), dict) else {}
        )
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


def _resolve_runtime_model_name(model_name: str, provider: Optional[str]) -> str:
    if provider == "openrouter":
        match = _ANTHROPIC_DATED_MODEL_RE.fullmatch(model_name.strip())
        if match:
            return match.group(1)
    return model_name


def _prepare_runtime_environment() -> None:
    global _RUNTIME_SKILLS_SYNCED
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
    # Do not disable "skills": Feishu workbench guidance is delivered through
    # bundled skills, and removing the skill toolset prevents that prompt from
    # being injected even when the native feishu_* tools are available.
    os.environ.setdefault(
        "HERMES_FEISHU_DISABLED_TOOLSETS",
        "browser,terminal,code_execution,delegation,tts,messaging,rl",
    )
    if "NGC_API_KEY" in os.environ and "NVIDIA_API_KEY" not in os.environ:
        os.environ["NVIDIA_API_KEY"] = os.environ["NGC_API_KEY"]
    os.environ.setdefault("HERMES_STREAM_READ_TIMEOUT", "25")
    if "HERMES_MAX_TURNS" in os.environ and "HERMES_MAX_ITERATIONS" not in os.environ:
        os.environ["HERMES_MAX_ITERATIONS"] = os.environ["HERMES_MAX_TURNS"]
    _ensure_runtime_dirs()
    _sync_runtime_config()
    _sync_supermemory_config()
    if not _RUNTIME_SKILLS_SYNCED:
        with _RUNTIME_SKILLS_SYNC_LOCK:
            if not _RUNTIME_SKILLS_SYNCED:
                try:
                    from tools.skills_sync import sync_skills

                    sync_skills(quiet=True)
                    _RUNTIME_SKILLS_SYNCED = True
                except Exception:
                    logger.warning("Failed to sync bundled skills into HERMES_HOME", exc_info=True)


def _maintenance_heartbeat_is_enabled() -> bool:
    return _is_truthy(
        os.getenv("HERMES_MODAL_MAINTENANCE_HEARTBEAT_ENABLED"),
        default=False,
    )


def _build_chat_worker_observability(
    worker_context: dict[str, Any] | None = None,
    *,
    batch_size: int | None = None,
) -> dict[str, Any]:
    context = worker_context if isinstance(worker_context, dict) else {}
    payload = {
        "worker_boot_id": context.get("worker_boot_id"),
        "worker_started_at": context.get("worker_started_at"),
        "enter_elapsed_ms": context.get("enter_elapsed_ms"),
        "runtime_prepare_elapsed_ms": context.get("runtime_prepare_elapsed_ms"),
        "container_reused": bool(context.get("container_reused", False)),
    }
    if batch_size is not None:
        payload["batch_size"] = batch_size
    for optional_key in (
        "worker_partition_key",
        "snapshot_prepare_elapsed_ms",
        "snapshot_enabled",
        "snapshot_restored",
        "execution_mode",
        "handoff_reason",
        "provider_wait_elapsed_ms",
        "provider_attempt_count",
        "provider_retry_count",
        "provider_fallback_used",
        "provider_error_class",
        "background_send_elapsed_ms",
        "routing_state_refreshed_at",
        "model_registry_entry_count",
        "model_registry_source",
        "warmup_status",
        "warmup_event_id",
        "warmup_started_at",
        "warmup_ready_at",
        "warmup_age_ms",
        "warmup_same_container",
        "warmup_message_gap_ms",
        "warmup_actor_id",
        "warmup_chat_id",
    ):
        optional_value = context.get(optional_key)
        if optional_value is not None:
            payload[optional_key] = optional_value
    return payload


def _build_inline_chat_worker_context() -> dict[str, Any]:
    worker_started_at = int(time.time() * 1000)
    runtime_prepare_started = time.time()
    _prepare_runtime_environment()
    runtime_prepare_elapsed_ms = int((time.time() - runtime_prepare_started) * 1000)
    return {
        "worker_boot_id": f"inline-{uuid.uuid4().hex}",
        "worker_started_at": worker_started_at,
        "enter_elapsed_ms": runtime_prepare_elapsed_ms,
        "runtime_prepare_elapsed_ms": runtime_prepare_elapsed_ms,
        "container_reused": False,
    }


def _prepare_worker_snapshot_context(*, worker_name: str) -> dict[str, Any]:
    started_at = time.time()
    _prepare_runtime_environment()
    # Preload hot Python modules, but avoid freezing per-request or volume-backed state
    # into the snapshot so restores stay correct.
    try:
        import gateway.config as _gateway_config  # noqa: F401
        import gateway.run as _gateway_run  # noqa: F401
        import gateway.platforms.feishu as _feishu_platform  # noqa: F401
        import tools.feishu_api as _feishu_api  # noqa: F401
    except Exception as exc:
        logger.warning("%s snapshot preload failed: %s", worker_name, exc)
    return {
        "snapshot_prepare_elapsed_ms": int((time.time() - started_at) * 1000),
        "snapshot_enabled": True,
    }


def _bootstrap_feishu_ingress_worker_context(snapshot_context: dict[str, Any] | None = None) -> dict[str, Any]:
    worker_started_at = int(time.time() * 1000)
    enter_started = time.time()
    runtime_prepare_started = time.time()
    _prepare_runtime_environment()
    runtime_prepare_elapsed_ms = int((time.time() - runtime_prepare_started) * 1000)

    queue_prewarm_started = time.time()
    try:
        asyncio.run(_prewarm_chat_queue_async())
        chat_queue_prewarm_ok = True
    except Exception as exc:
        logger.warning("Feishu ingress worker queue prewarm failed: %s", exc)
        chat_queue_prewarm_ok = False
    chat_queue_prewarm_elapsed_ms = int((time.time() - queue_prewarm_started) * 1000)

    context = {
        "worker_boot_id": f"feishu-ingress-{uuid.uuid4().hex}",
        "worker_started_at": worker_started_at,
        "runtime_prepare_elapsed_ms": runtime_prepare_elapsed_ms,
        "enter_elapsed_ms": int((time.time() - enter_started) * 1000),
        "chat_queue_prewarm_ok": chat_queue_prewarm_ok,
        "chat_queue_prewarm_elapsed_ms": chat_queue_prewarm_elapsed_ms,
        "container_reused": False,
    }
    if isinstance(snapshot_context, dict):
        context.update({key: value for key, value in snapshot_context.items() if value is not None})
        context["snapshot_restored"] = True
    return context


def _bootstrap_feishu_ack_reaction_worker_context(snapshot_context: dict[str, Any] | None = None) -> dict[str, Any]:
    worker_started_at = int(time.time() * 1000)
    enter_started = time.time()
    runtime_prepare_started = time.time()
    _prepare_runtime_environment()
    runtime_prepare_elapsed_ms = int((time.time() - runtime_prepare_started) * 1000)

    context = {
        "worker_boot_id": f"feishu-ack-{uuid.uuid4().hex}",
        "worker_started_at": worker_started_at,
        "runtime_prepare_elapsed_ms": runtime_prepare_elapsed_ms,
        "enter_elapsed_ms": int((time.time() - enter_started) * 1000),
        "container_reused": False,
    }
    if isinstance(snapshot_context, dict):
        context.update({key: value for key, value in snapshot_context.items() if value is not None})
        context["snapshot_restored"] = True
    return context


def _bootstrap_chat_queue_worker_context(
    snapshot_context: dict[str, Any] | None = None,
    *,
    worker_partition_key: str | None = None,
) -> dict[str, Any]:
    worker_started_at = int(time.time() * 1000)
    enter_started = time.time()
    runtime_prepare_started = time.time()
    _prepare_runtime_environment()
    runtime_prepare_elapsed_ms = int((time.time() - runtime_prepare_started) * 1000)

    context: dict[str, Any] = {
        "worker_boot_id": uuid.uuid4().hex,
        "worker_started_at": worker_started_at,
        "runtime_prepare_elapsed_ms": runtime_prepare_elapsed_ms,
        "container_reused": False,
    }
    normalized_partition_key = str(worker_partition_key or "").strip()
    if normalized_partition_key:
        context["worker_partition_key"] = normalized_partition_key
    if isinstance(snapshot_context, dict):
        context.update({key: value for key, value in snapshot_context.items() if value is not None})
        context["snapshot_restored"] = True

    routing_state = _load_routing_state()
    routing_state_refreshed_at = int(routing_state.get("refreshed_at") or 0) if isinstance(routing_state, dict) else 0
    if routing_state_refreshed_at:
        context["routing_state_refreshed_at"] = routing_state_refreshed_at

    try:
        from tools.feishu_api import load_feishu_model_registry

        registry_payload = load_feishu_model_registry(force_refresh=False)
        context["model_registry_entry_count"] = len(registry_payload.get("entries") or [])
        model_registry_source = str(registry_payload.get("source") or "").strip()
        if model_registry_source:
            context["model_registry_source"] = model_registry_source
    except Exception as exc:
        logger.warning("Chat queue worker model registry preload failed: %s", exc)

    if CHAT_WORKER_PREINIT_FEISHU_RUNTIME_ENABLED:
        settings = RuntimeSettings.from_env()
        if settings.feishu_app_id and settings.feishu_app_secret:
            preinit_started = time.time()
            try:
                asyncio.run(_get_feishu_gateway_runtime())
                context["feishu_runtime_preinit_ok"] = True
            except Exception as exc:
                context["feishu_runtime_preinit_ok"] = False
                context["feishu_runtime_preinit_error"] = str(exc)
                logger.warning("Chat queue worker Feishu runtime preinit failed: %s", exc)
            context["feishu_runtime_preinit_elapsed_ms"] = int((time.time() - preinit_started) * 1000)

    context["enter_elapsed_ms"] = int((time.time() - enter_started) * 1000)
    return context


def _get_process_chat_queue_handle(partition: str | None = None) -> Any:
    default_handle = globals().get("process_chat_queue")
    normalized_partition = str(partition or "").strip()
    if not normalized_partition:
        return default_handle

    worker_cls = globals().get("ChatQueueWorker")
    if worker_cls is None:
        return default_handle

    # In production Modal returns a Function handle here; tests often monkeypatch a
    # simple stub object. Only bypass the parameterized pool for explicit test stubs
    # and other non-Modal handles.
    if default_handle is not None and type(default_handle).__name__ != "Function":
        return default_handle

    try:
        return worker_cls(partition_key=normalized_partition).process
    except Exception as exc:
        logger.warning(
            "[ChatQueue] parameterized worker handle fallback partition=%s error=%s",
            normalized_partition,
            exc,
        )
        return default_handle


@dataclass(slots=True)
class RuntimeSettings:
    model: str
    max_turns: int
    max_tokens: Optional[int]
    provider: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    bearer_token: Optional[str]
    feishu_internal_bearer_token: Optional[str]
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
    feishu_bitable_wiki_token: Optional[str] = None
    feishu_model_registry_sync_interval_seconds: int = DEFAULT_FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        provider, base_url, api_key = _pick_runtime_api_config()
        feishu_app_id = os.getenv("FEISHU_APP_ID")
        feishu_app_secret = os.getenv("FEISHU_APP_SECRET")
        feishu_internal_bearer_token = (
            os.getenv("HERMES_FEISHU_INTERNAL_BEARER_TOKEN")
            or os.getenv("WEBHOOK_SECRET")
            or os.getenv("HERMES_WEBHOOK_BEARER_TOKEN")
            or _derive_feishu_internal_bearer_token(
                app_id=feishu_app_id,
                app_secret=feishu_app_secret,
            )
        )
        return cls(
            model=os.getenv("DEFAULT_MODEL", "openrouter/free"),
            max_turns=int(os.getenv("HERMES_MAX_TURNS", "16")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "0")) or None,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            bearer_token=os.getenv("WEBHOOK_SECRET") or os.getenv("HERMES_WEBHOOK_BEARER_TOKEN"),
            feishu_internal_bearer_token=feishu_internal_bearer_token,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET"),
            telegram_webhook_url=_desired_telegram_webhook_url(),
            telegram_send_ack=os.getenv("TELEGRAM_SEND_ACK", "false").lower() in {"1", "true", "yes"},
            feishu_app_id=feishu_app_id,
            feishu_app_secret=feishu_app_secret,
            feishu_domain=os.getenv("FEISHU_DOMAIN") or "feishu",
            feishu_connection_mode=os.getenv("FEISHU_CONNECTION_MODE") or "webhook",
            feishu_verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN"),
            feishu_encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY"),
            feishu_bitable_app_token=os.getenv("FEISHU_BITABLE_APP_TOKEN"),
            feishu_bitable_wiki_token=os.getenv("FEISHU_BITABLE_WIKI_TOKEN"),
            feishu_bitable_table_id=os.getenv("FEISHU_BITABLE_TABLE_ID"),
            feishu_model_registry_mirror_enabled=_is_truthy(os.getenv("FEISHU_MODEL_REGISTRY_MIRROR_ENABLED"), default=False),
            feishu_model_registry_sync_interval_seconds=_get_feishu_sync_interval_seconds(),
            feishu_tool_capabilities=_split_csv(os.getenv("HERMES_FEISHU_TOOL_CAPABILITIES")),
            feishu_default_workspace=os.getenv("HERMES_FEISHU_DEFAULT_WORKSPACE"),
            qq_app_id=os.getenv("QQ_APP_ID"),
            qq_app_secret=os.getenv("QQ_APP_SECRET"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY"),
            nvidia_base_url=os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL,
            enabled_toolsets=_split_csv(os.getenv("HERMES_ENABLED_TOOLSETS")),
            disabled_toolsets=_split_csv(os.getenv("HERMES_DISABLED_TOOLSETS")) or list(DEFAULT_DISABLED_TOOLSETS),
        )


def _derive_feishu_internal_bearer_token(
    *,
    app_id: str | None,
    app_secret: str | None,
) -> str | None:
    normalized_app_id = str(app_id or "").strip()
    normalized_app_secret = str(app_secret or "").strip()
    if not normalized_app_id or not normalized_app_secret:
        return None
    digest = hashlib.sha256(
        f"hermes-feishu-internal:{normalized_app_id}:{normalized_app_secret}".encode("utf-8")
    ).hexdigest()
    return f"fi_{digest}"


def _serialize_settings_for_log(settings: RuntimeSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["api_key"] = _mask_secret(settings.api_key)
    payload["bearer_token"] = _mask_secret(settings.bearer_token)
    payload["feishu_internal_bearer_token"] = _mask_secret(settings.feishu_internal_bearer_token)
    payload["telegram_bot_token"] = _mask_secret(settings.telegram_bot_token)
    payload["telegram_webhook_secret"] = _mask_secret(settings.telegram_webhook_secret)
    payload["feishu_app_id"] = _mask_secret(settings.feishu_app_id)
    payload["feishu_app_secret"] = _mask_secret(settings.feishu_app_secret)
    payload["feishu_verification_token"] = _mask_secret(settings.feishu_verification_token)
    payload["feishu_encrypt_key"] = _mask_secret(settings.feishu_encrypt_key)
    payload["feishu_bitable_app_token"] = _mask_secret(settings.feishu_bitable_app_token)
    payload["feishu_bitable_wiki_token"] = _mask_secret(settings.feishu_bitable_wiki_token)
    payload["feishu_bitable_table_id"] = _mask_secret(settings.feishu_bitable_table_id)
    payload["qq_app_id"] = _mask_secret(settings.qq_app_id)
    payload["qq_app_secret"] = _mask_secret(settings.qq_app_secret)
    payload["nvidia_api_key"] = _mask_secret(settings.nvidia_api_key)
    return payload


def _build_modal_official_parity_state() -> dict[str, Any]:
    """Summarize which official Hermes features are available on this Modal profile.

    The goal is explicit parity accounting: features are marked supported when the
    current Modal deployment exposes and installs them, and marked unsupported
    only when the current serverless/webhook shape is a genuine mismatch for the
    official feature.
    """

    settings = RuntimeSettings.from_env()
    browser_cli_available = shutil.which("agent-browser") is not None
    browser_node_available = shutil.which("node") is not None and shutil.which("npm") is not None
    honcho_available = importlib.util.find_spec("plugins.memory.honcho") is not None
    homeassistant_available = importlib.util.find_spec("tools.homeassistant_tool") is not None
    qq_sdk_available = importlib.util.find_spec("botpy") is not None

    features = {
        "core_agent_loop": {
            "status": "supported",
            "reason": "Modal entrypoint delegates task execution to Hermes AIAgent.",
        },
        "session_learning_loop": {
            "status": "supported",
            "reason": "Session search, skill management, and background review loop ship in the Modal image.",
        },
        "mcp_tools": {
            "status": "supported",
            "reason": "Modal image installs MCP support and runtime config can inject MCP servers.",
        },
        "browser_tools": {
            "status": "supported" if (browser_cli_available or browser_node_available) else "partial",
            "reason": (
                "Browser automation can run in Modal via headless browser tooling."
                if (browser_cli_available or browser_node_available)
                else "Node is present but no browser launcher was detected yet."
            ),
            "browser_cli_available": browser_cli_available,
            "node_runtime_available": browser_node_available,
        },
        "send_message_tool": {
            "status": "supported",
            "reason": "Messaging toolset is enabled by default on the Modal profile.",
        },
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
            "reason": (
                "Honcho plugin is present; still needs Honcho config/API credentials."
                if honcho_available
                else "Honcho plugin is not installed in the current image."
            ),
        },
        "voice_mode_cli": {
            "status": "unsupported",
            "reason": "Interactive push-to-talk voice mode depends on a local audio device, which Modal web workers do not provide.",
        },
        "rl_training_stack": {
            "status": "unsupported",
            "reason": "The production Modal profile does not install the official RL/Tinker extras or expose dedicated GPU training surfaces.",
        },
    }

    platforms = {
        "telegram": {
            "status": "supported",
            "reason": "Webhook route is exposed on the Modal web app.",
        },
        "feishu": {
            "status": "supported",
            "reason": "Webhook route is exposed and native Feishu adapter is wired through the chat queue.",
        },
        "qq": {
            "status": "supported" if qq_sdk_available else "partial",
            "reason": (
                "Webhook route is exposed and QQ SDK is installed."
                if qq_sdk_available
                else "Webhook route exists, but QQ SDK was not detected in the current image."
            ),
        },
        "discord": {
            "status": "unsupported",
            "reason": "Official Discord adapter expects a long-lived gateway connection; the current Modal deployment exposes only webhook-style ingress.",
        },
        "slack": {
            "status": "unsupported",
            "reason": "Slack Events / Socket Mode ingress is not exposed by the current Modal web surface.",
        },
        "whatsapp": {
            "status": "unsupported",
            "reason": "WhatsApp bridge support requires long-lived bridge/session state not wired into this Modal profile.",
        },
        "signal": {
            "status": "unsupported",
            "reason": "Signal transport depends on an external bridge/polling flow not wired into this Modal profile.",
        },
        "email": {
            "status": "unsupported",
            "reason": "Inbound email polling is not exposed by the current Modal deployment shape.",
        },
        "matrix": {
            "status": "unsupported",
            "reason": "Matrix sync requires a long-lived client loop that is not wired into the webhook-first Modal app.",
        },
        "dingtalk": {
            "status": "unsupported",
            "reason": "DingTalk streaming ingress is not exposed by the current Modal deployment.",
        },
    }

    return {
        "profile": "modal-webhook",
        "official_goal": "Align supportable official Hermes features on Modal; explicitly carve out Modal-incompatible surfaces.",
        "public_webhook_platforms": list(_MODAL_PUBLIC_WEBHOOK_PLATFORMS),
        "default_disabled_toolsets": list(settings.disabled_toolsets),
        "enabled_toolsets_override": list(settings.enabled_toolsets),
        "features": features,
        "platforms": platforms,
    }


@dataclass(slots=True)
class _TelegramGatewayRuntime:
    runner: Any
    adapter: Any


@dataclass(slots=True)
class _FeishuGatewayRuntime:
    runner: Any
    adapter: Any
    init_timings: dict[str, int] | None = None


@dataclass(slots=True)
class _QQGatewayRuntime:
    runner: Any
    adapter: Any


def _make_feishu_internal_capture_adapter(config: Any) -> Any:
    from gateway.config import Platform
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    class _FeishuInternalCaptureAdapter(BasePlatformAdapter):
        def __init__(self, adapter_config: Any):
            super().__init__(adapter_config, Platform.FEISHU)
            self.captured_operations: list[dict[str, Any]] = []
            self._mark_connected()

        async def connect(self) -> bool:
            self._mark_connected()
            return True

        async def disconnect(self) -> None:
            self._mark_disconnected()

        async def send(
            self,
            chat_id: str,
            content: str,
            reply_to: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "text",
                    "chat_id": chat_id,
                    "content": str(content or ""),
                    "reply_to": reply_to,
                    "metadata": dict(metadata or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def edit_message(self, chat_id: str, message_id: str, content: str) -> Any:
            self.captured_operations.append(
                {
                    "kind": "edit",
                    "chat_id": chat_id,
                    "content": str(content or ""),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_image(
            self,
            chat_id: str,
            image_url: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "image_url",
                    "chat_id": chat_id,
                    "image_url": str(image_url or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(metadata or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_voice(
            self,
            chat_id: str,
            audio_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "audio_file",
                    "chat_id": chat_id,
                    "file_path": str(audio_path or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_video(
            self,
            chat_id: str,
            video_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "video_file",
                    "chat_id": chat_id,
                    "file_path": str(video_path or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_document(
            self,
            chat_id: str,
            file_path: str,
            caption: Optional[str] = None,
            file_name: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "document_file",
                    "chat_id": chat_id,
                    "file_path": str(file_path or ""),
                    "file_name": str(file_name or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_image_file(
            self,
            chat_id: str,
            image_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "image_file",
                    "chat_id": chat_id,
                    "file_path": str(image_path or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
            normalized_chat_id = str(chat_id or "").strip()
            return {
                "chat_id": normalized_chat_id,
                "name": normalized_chat_id or "Feishu Chat",
                "type": "dm",
            }

    return _FeishuInternalCaptureAdapter(config)


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


def _load_feishu_sync_state() -> dict[str, Any]:
    payload = _load_json_file(FEISHU_SYNC_STATE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_feishu_sync_state(payload: dict[str, Any]) -> None:
    _atomic_json_write(FEISHU_SYNC_STATE_PATH, payload)


def _get_feishu_sync_interval_seconds() -> int:
    raw = str(os.getenv("FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS", "") or "").strip()
    if not raw:
        return DEFAULT_FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS
    try:
        return max(60, int(raw))
    except Exception:
        return DEFAULT_FEISHU_MODEL_REGISTRY_SYNC_INTERVAL_SECONDS


def _mask_runtime_identifier(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= 10:
        return _mask_secret(raw)
    return f"{raw[:4]}...{raw[-4:]}"


def _compact_feishu_schema_status(result: dict[str, Any]) -> dict[str, Any]:
    existing_field_names = result.get("existing_field_names") if isinstance(result.get("existing_field_names"), list) else []
    existing_view_names = result.get("existing_view_names") if isinstance(result.get("existing_view_names"), list) else []
    return {
        "status": result.get("status"),
        "app_token": str(result.get("app_token") or ""),
        "app_token_masked": _mask_runtime_identifier(result.get("app_token")),
        "table_id": str(result.get("table_id") or ""),
        "table_name": result.get("table_name"),
        "created_table": bool(result.get("created_table")),
        "created_field_count": len(result.get("created_fields") or []),
        "created_view_count": len(result.get("created_views") or []),
        "field_count": len(existing_field_names),
        "view_count": len(existing_view_names),
        "missing_required_fields": list(result.get("missing_required_fields") or []),
        "missing_optional_fields": list(result.get("missing_optional_fields") or []),
        "missing_views": list(result.get("missing_views") or []),
    }


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


def _get_feishu_internal_runtime_lock() -> asyncio.Lock:
    global _FEISHU_INTERNAL_RUNTIME_LOCK
    if _FEISHU_INTERNAL_RUNTIME_LOCK is None:
        _FEISHU_INTERNAL_RUNTIME_LOCK = asyncio.Lock()
    return _FEISHU_INTERNAL_RUNTIME_LOCK


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
    total_started_at = time.perf_counter()
    timings: dict[str, int] = {}

    import_started_at = time.perf_counter()
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.feishu import (
        FEISHU_DOMAIN,
        LARK_DOMAIN,
        FeishuAdapter,
        check_feishu_requirements,
    )
    from gateway.run import GatewayRunner
    timings["imports_ms"] = int((time.perf_counter() - import_started_at) * 1000)

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

    allowlist_started_at = time.perf_counter()
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
    timings["allowlist_ms"] = int((time.perf_counter() - allowlist_started_at) * 1000)

    logger.info(
        "[Feishu] Runtime policy=%s require_mention=%s env_allowlist=%d paired_allowlist=%d merged_allowlist=%d",
        group_policy,
        group_require_mention,
        len(env_allowed_users),
        len(approved_feishu_users),
        len(merged_allowed_users),
    )

    runner_started_at = time.perf_counter()
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
    timings["runner_config_ms"] = int((time.perf_counter() - runner_started_at) * 1000)

    adapter_started_at = time.perf_counter()
    adapter = FeishuAdapter(feishu_config)
    adapter._loop = asyncio.get_running_loop()
    domain = FEISHU_DOMAIN if adapter._domain_name != "lark" else LARK_DOMAIN
    adapter._client = adapter._build_lark_client(domain)
    adapter._event_handler = adapter._build_event_handler()
    if adapter._event_handler is None:
        raise RuntimeError("failed to build Feishu event handler")
    timings["adapter_setup_ms"] = int((time.perf_counter() - adapter_started_at) * 1000)

    cached_identity = _get_cached_feishu_bot_identity(settings.feishu_app_id)
    cached_bot_name = str(cached_identity.get("bot_name") or "").strip()
    if cached_bot_name:
        adapter._bot_name = cached_bot_name
        timings["hydrate_bot_identity_ms"] = 0
        timings["bot_identity_cache_hit"] = 1
    else:
        hydrate_started_at = time.perf_counter()
        await adapter._hydrate_bot_identity()
        timings["hydrate_bot_identity_ms"] = int((time.perf_counter() - hydrate_started_at) * 1000)
        timings["bot_identity_cache_hit"] = 0
        if str(getattr(adapter, "_bot_name", "") or "").strip():
            await _cache_feishu_bot_identity_async(
                settings.feishu_app_id,
                bot_name=str(getattr(adapter, "_bot_name", "") or "").strip(),
            )

    finalize_started_at = time.perf_counter()
    adapter._mark_connected()
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)
    adapter.set_menu_action_handler(runner._handle_feishu_menu_action)

    runner.adapters[Platform.FEISHU] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)
    timings["finalize_ms"] = int((time.perf_counter() - finalize_started_at) * 1000)
    timings["total_ms"] = int((time.perf_counter() - total_started_at) * 1000)
    logger.warning(
        "[Feishu] runtime init timings imports_ms=%s allowlist_ms=%s runner_config_ms=%s adapter_setup_ms=%s hydrate_bot_identity_ms=%s finalize_ms=%s total_ms=%s bot_identity_cache_hit=%s",
        timings.get("imports_ms", 0),
        timings.get("allowlist_ms", 0),
        timings.get("runner_config_ms", 0),
        timings.get("adapter_setup_ms", 0),
        timings.get("hydrate_bot_identity_ms", 0),
        timings.get("finalize_ms", 0),
        timings.get("total_ms", 0),
        bool(timings.get("bot_identity_cache_hit")),
    )
    return _FeishuGatewayRuntime(runner=runner, adapter=adapter, init_timings=timings)


def _get_feishu_webhook_security_settings() -> tuple[str, str]:
    encrypt_key = str(os.getenv("FEISHU_ENCRYPT_KEY", "") or "").strip()
    verification_token = str(os.getenv("FEISHU_VERIFICATION_TOKEN", "") or "").strip()
    return encrypt_key, verification_token


def _is_feishu_webhook_signature_valid(headers: Mapping[str, Any], body_bytes: bytes, *, encrypt_key: str) -> bool:
    if not encrypt_key:
        return True
    timestamp = str(headers.get("x-lark-request-timestamp", "") or "")
    nonce = str(headers.get("x-lark-request-nonce", "") or "")
    signature = str(headers.get("x-lark-signature", "") or "")
    if not timestamp or not nonce or not signature:
        return False
    try:
        body_str = body_bytes.decode("utf-8", errors="replace")
        content = f"{timestamp}{nonce}{encrypt_key}{body_str}"
        computed = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return hmac.compare_digest(computed, signature)
    except Exception:
        logger.debug("[Feishu] Signature verification raised an exception", exc_info=True)
        return False


def _decrypt_feishu_webhook_payload(encrypted_payload: str, *, encrypt_key: str) -> dict[str, Any]:
    if not encrypted_payload:
        raise ValueError("encrypted webhook payload is empty")
    if not encrypt_key:
        raise ValueError("encrypt_key is required to decrypt webhook payloads")

    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError as exc:
        raise RuntimeError("cryptography is required for Feishu webhook decryption") from exc

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
        raise ValueError("decrypted webhook payload must be a JSON object")
    return payload


def _build_feishu_card_action_ack_payload(content: str = "已收到，正在处理") -> dict[str, Any]:
    return {
        "toast": {
            "type": "info",
            "content": content,
        }
    }


def _chunk_card_actions(actions: list[dict[str, Any]], *, size: int = 2) -> list[list[dict[str, Any]]]:
    size = max(1, int(size or 2))
    return [actions[index:index + size] for index in range(0, len(actions), size)]


def _shorten_local_model_label(model_id: str, *, max_len: int = 42) -> str:
    text = str(model_id or "").strip()
    if len(text) <= max_len:
        return text
    if "/" in text:
        provider, remainder = text.split("/", 1)
        head = max(10, min(18, max_len // 2))
        tail = max(8, min(12, max_len - len(provider) - head - 4))
        return f"{provider}/{remainder[:head]}...{remainder[-tail:]}"
    return text[: max_len - 3] + "..."


def _local_registry_card_button(*, label: str, action: str, extra: dict[str, Any] | None = None, btn_type: str = "default") -> dict[str, Any]:
    value = {"hermes_action": action}
    value.update(extra or {})
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": btn_type,
        "value": value,
    }


def _resolve_feishu_menu_target(payload: Mapping[str, Any]) -> tuple[str, str] | None:
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


def _build_local_registry_entries(provider_slug: str, view_name: str, *, limit: int = 20) -> list[dict[str, Any]]:
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


def _build_local_registry_provider_card(*, provider_slug: str, view_name: str) -> dict[str, Any]:
    entries = _build_local_registry_entries(provider_slug, view_name, limit=20)
    provider_title = provider_slug.title()
    view_title = {"featured": "精选", "recent": "最近", "performance": "性能"}.get(view_name, "精选")
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": f"**{provider_title} · {view_title}**\n本地 registry 直出\n点击模型立即切换",
        }
    ]

    actions = [
        _local_registry_card_button(
            label=_shorten_local_model_label(str(item.get("model") or "").strip()),
            action="registry_switch_model",
            extra={
                "provider": provider_slug,
                "model": str(item.get("model") or "").strip(),
            },
        )
        for item in entries
    ]
    for chunk in _chunk_card_actions(actions, size=2):
        elements.append({"tag": "action", "actions": chunk})

    elements.append(
        {
            "tag": "action",
            "actions": [
                _local_registry_card_button(
                    label="关闭",
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


def _load_feishu_personality_card_entries() -> tuple[list[dict[str, str]], str]:
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
    for name in _FEISHU_PERSONALITY_CARD_ORDER:
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
                "label": _FEISHU_PERSONALITY_CARD_LABELS.get(name, name.upper()),
                "description": description[:36],
            }
        )
    return entries, current_personality


def _build_feishu_personality_card() -> dict[str, Any]:
    entries, current_personality = _load_feishu_personality_card_entries()
    current_label = _FEISHU_PERSONALITY_CARD_LABELS.get(current_personality, current_personality.upper()) if current_personality else "未设置"
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                f"**人格特质**\n当前：`{current_label}`\n"
                "点击按钮等同执行 `/personality <name>`，下一条消息开始生效。"
            ),
        }
    ]

    buttons: list[dict[str, Any]] = []
    for entry in entries:
        label = entry["label"]
        if entry["description"]:
            label = f"{label}"
        buttons.append(
            _local_registry_card_button(
                label=label,
                action="personality_set",
                extra={"personality": entry["name"]},
                btn_type="primary" if entry["name"] == current_personality else "default",
            )
        )
    for chunk in _chunk_card_actions(buttons, size=3):
        elements.append({"tag": "action", "actions": chunk})

    elements.append(
        {
            "tag": "markdown",
            "content": "底部说明：人格改变风格与决策侧重点，不改变工具边界。",
        }
    )
    elements.append(
        {
            "tag": "action",
            "actions": [
                _local_registry_card_button(label="清除人格", action="personality_set", extra={"personality": "none"}, btn_type="danger"),
                _local_registry_card_button(label="关闭", action="registry_close_card", btn_type="default"),
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "人格特质"},
            "template": "wathet",
        },
        "elements": elements,
    }


def _load_feishu_command_card_sections() -> list[tuple[str, list[dict[str, Any]]]]:
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
                "default_command": _FEISHU_COMMAND_CARD_DEFAULT_RUNS.get(str(command.name or "").strip()),
            }
        )

    ordered_categories = [
        category for category in _FEISHU_COMMAND_CARD_CATEGORY_ORDER if category in sections
    ]
    ordered_categories.extend(
        category for category in sections.keys() if category not in ordered_categories
    )
    return [(category, sections.get(category, [])) for category in ordered_categories]


def _split_card_reference_lines(entries: list[str], *, chunk_size: int = 2) -> list[str]:
    chunk_size = max(1, int(chunk_size or 2))
    return [
        " | ".join(entries[index:index + chunk_size])
        for index in range(0, len(entries), chunk_size)
    ]


def _build_feishu_command_center_card() -> dict[str, Any]:
    sections = _load_feishu_command_card_sections()
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
        category_label = _FEISHU_COMMAND_CARD_CATEGORY_LABELS.get(category, category)
        elements.append({"tag": "markdown", "content": f"**{category_label}**"})

        runnable_actions = [
            _local_registry_card_button(
                label=f"/{entry['name']}",
                action="command_run",
                extra={"command_text": entry["default_command"]},
                btn_type="primary" if entry["name"] in {"help", "status", "model", "provider"} else "default",
            )
            for entry in entries
            if str(entry.get("default_command") or "").strip()
        ]
        for chunk in _chunk_card_actions(runnable_actions, size=3):
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

        for line in _split_card_reference_lines(manual_refs, chunk_size=2):
            elements.append({"tag": "markdown", "content": f"Input manually: {line}"})
        for line in _split_card_reference_lines(cli_only_refs, chunk_size=2):
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
                _local_registry_card_button(label="Close", action="registry_close_card", btn_type="default"),
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


def _build_feishu_model_hub_card() -> dict[str, Any]:
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
                _local_registry_card_button(
                    label="OR Featured",
                    action="open_menu_card",
                    extra={"event_key": "provider_openrouter_featured"},
                    btn_type="primary",
                ),
                _local_registry_card_button(
                    label="OR Recent",
                    action="open_menu_card",
                    extra={"event_key": "provider_openrouter_recent"},
                ),
                _local_registry_card_button(
                    label="OR Perf",
                    action="open_menu_card",
                    extra={"event_key": "provider_openrouter_performance"},
                ),
            ],
        },
        {
            "tag": "action",
            "actions": [
                _local_registry_card_button(
                    label="NV Featured",
                    action="open_menu_card",
                    extra={"event_key": "provider_nvidia_featured"},
                    btn_type="primary",
                ),
                _local_registry_card_button(
                    label="NV Recent",
                    action="open_menu_card",
                    extra={"event_key": "provider_nvidia_recent"},
                ),
                _local_registry_card_button(
                    label="NV Perf",
                    action="open_menu_card",
                    extra={"event_key": "provider_nvidia_performance"},
                ),
            ],
        },
        {
            "tag": "action",
            "actions": [
                _local_registry_card_button(label="Route Status", action="command_run", extra={"command_text": "/status"}),
                _local_registry_card_button(label="Providers", action="command_run", extra={"command_text": "/provider"}),
                _local_registry_card_button(label="Route", action="command_run", extra={"command_text": "/status"}),
            ],
        },
        {
            "tag": "action",
            "actions": [
                _local_registry_card_button(
                    label="Personality",
                    action="open_menu_card",
                    extra={"event_key": "personality_picker"},
                ),
                _local_registry_card_button(
                    label="Skill Combos",
                    action="open_menu_card",
                    extra={"event_key": "skill_combo_picker"},
                ),
                _local_registry_card_button(
                    label="Commands",
                    action="open_menu_card",
                    extra={"event_key": "command_center"},
                ),
            ],
        },
        {
            "tag": "action",
            "actions": [
                _local_registry_card_button(label="Close", action="registry_close_card", btn_type="default"),
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


def _build_feishu_skill_combo_card() -> dict[str, Any]:
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                "**技能组合**\n点击按钮会把对应技能提示加载进当前会话，"
                "Hermes 会用简短回复确认当前工作模式。人格请单独在人格卡片切换。"
            ),
        }
    ]

    for combo in _FEISHU_SKILL_COMBO_DEFINITIONS:
        combo_label = str(combo.get("label") or "").strip()
        combo_summary = str(combo.get("summary") or "").strip()
        suggested_personality = str(combo.get("suggested_personality") or "").strip()
        elements.append(
            {
                "tag": "markdown",
                "content": (
                    f"**{combo_label}**\n{combo_summary}\n"
                    f"建议人格：`{suggested_personality or 'none'}`"
                ),
            }
        )
        elements.append(
            {
                "tag": "action",
                "actions": [
                    _local_registry_card_button(
                        label=f"应用 {combo_label}",
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
            "tag": "markdown",
            "content": "底部说明：组合主要影响方法论与检查表，不会绕过平台控制逻辑。",
        }
    )
    elements.append(
        {
            "tag": "action",
            "actions": [
                _local_registry_card_button(label="关闭", action="registry_close_card", btn_type="default"),
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "技能组合"},
            "template": "turquoise",
        },
        "elements": elements,
    }


def _build_feishu_local_menu_card(event_key: str) -> dict[str, Any] | None:
    normalized_event_key = str(event_key or "").strip()
    provider_slug, view_name = _FEISHU_LOCAL_MODEL_MENU_MAP.get(normalized_event_key, (None, None))
    if provider_slug and view_name:
        return _build_local_registry_provider_card(provider_slug=provider_slug, view_name=view_name)
    if normalized_event_key == "model_picker":
        return _build_feishu_model_hub_card()
    if normalized_event_key == "personality_picker":
        return _build_feishu_personality_card()
    if normalized_event_key == "skill_combo_picker":
        return _build_feishu_skill_combo_card()
    if normalized_event_key == "command_center":
        return _build_feishu_command_center_card()
    return None


async def _send_feishu_local_registry_menu_card(payload: Mapping[str, Any]) -> bool:
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return False
    event_key = str(event.get("event_key") or "").strip()
    card = _build_feishu_local_menu_card(event_key)
    if not card:
        return False
    target = _resolve_feishu_menu_target(payload)
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


async def _close_feishu_card_from_payload(payload: Mapping[str, Any]) -> bool:
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


def _extract_feishu_card_action_name(payload: Mapping[str, Any]) -> str:
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
        queue_depth=enqueue_result.get("queue_depth"),
        spawn_scheduled=spawn_scheduled,
    )
    return {**enqueue_result, "spawn_scheduled": spawn_scheduled}


def _add_feishu_ack_reaction_from_payload(
    payload: dict[str, Any],
    *,
    request_started_at_ms: int | None = None,
    worker_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from tools.feishu_api import build_feishu_client

    context = _extract_feishu_queue_context(payload)
    message_id = str(context.get("message_id") or "").strip()
    event_id, event_type = _extract_feishu_event_metadata(payload)
    if not message_id:
        return {
            "status": "skipped",
            "reason": "missing_message_id",
            "event_id": event_id,
            "event_type": event_type,
        }

    started_at = time.perf_counter()
    client_started_at = time.perf_counter()
    try:
        client = build_feishu_client(timeout=DEFAULT_FEISHU_ACK_REACTION_REQUEST_TIMEOUT_SECONDS)
    except TypeError:
        client = build_feishu_client()
    client_acquire_elapsed_ms = int((time.perf_counter() - client_started_at) * 1000)
    reaction_started_at = time.perf_counter()
    success = False
    error_text = ""
    reaction_id = ""
    try:
        try:
            response_payload = client.request_json(
                "POST",
                f"/open-apis/im/v1/messages/{message_id}/reactions",
                json_body={"reaction_type": {"emoji_type": _FEISHU_ACK_REACTION_EMOJI}},
                retries=1,
            )
        except TypeError:
            response_payload = client.request_json(
                "POST",
                f"/open-apis/im/v1/messages/{message_id}/reactions",
                json_body={"reaction_type": {"emoji_type": _FEISHU_ACK_REACTION_EMOJI}},
            )
        reaction_id = str(
            response_payload.get("reaction_id")
            or (response_payload.get("reaction") or {}).get("reaction_id")
            or ""
        ).strip()
        success = True
    except Exception as exc:
        error_text = str(exc)
    reaction_elapsed_ms = int((time.perf_counter() - reaction_started_at) * 1000)
    total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    resolved_request_started_at_ms = _resolve_feishu_request_started_at_ms(payload, request_started_at_ms)
    from_request_elapsed_ms = (
        max(0, int(time.time() * 1000) - int(resolved_request_started_at_ms))
        if resolved_request_started_at_ms is not None
        else None
    )
    worker_boot_id = str((worker_context or {}).get("worker_boot_id") or "").strip()
    worker_reused = (worker_context or {}).get("container_reused")
    _append_feishu_trace(
        "webhook.ack_reaction",
        payload,
        success=success,
        message_id=message_id,
        reaction_id=reaction_id or "",
        client_acquire_elapsed_ms=client_acquire_elapsed_ms,
        reaction_elapsed_ms=reaction_elapsed_ms,
        total_elapsed_ms=total_elapsed_ms,
        from_request_elapsed_ms=from_request_elapsed_ms,
        worker_boot_id=worker_boot_id,
        worker_reused=worker_reused,
        error=error_text,
    )
    if success:
        logger.warning(
            "[Feishu] webhook ack reaction event_type=%s event_id=%s message_id=%s success=%s client_acquire_elapsed_ms=%s reaction_elapsed_ms=%s total_elapsed_ms=%s from_request_elapsed_ms=%s worker_boot_id=%s worker_reused=%s",
            event_type or "unknown",
            event_id or "none",
            message_id,
            success,
            client_acquire_elapsed_ms,
            reaction_elapsed_ms,
            total_elapsed_ms,
            from_request_elapsed_ms if from_request_elapsed_ms is not None else "na",
            worker_boot_id or "none",
            worker_reused,
        )
    else:
        logger.warning(
            "[Feishu] webhook ack reaction failed event_type=%s event_id=%s message_id=%s client_acquire_elapsed_ms=%s reaction_elapsed_ms=%s total_elapsed_ms=%s from_request_elapsed_ms=%s worker_boot_id=%s worker_reused=%s error=%s",
            event_type or "unknown",
            event_id or "none",
            message_id,
            client_acquire_elapsed_ms,
            reaction_elapsed_ms,
            total_elapsed_ms,
            from_request_elapsed_ms if from_request_elapsed_ms is not None else "na",
            worker_boot_id or "none",
            worker_reused,
            error_text or "unknown",
        )
    return {
        "status": "ok" if success else "error",
        "event_id": event_id,
        "event_type": event_type,
        "message_id": message_id,
        "reaction_id": reaction_id or "",
        "client_acquire_elapsed_ms": client_acquire_elapsed_ms,
        "reaction_elapsed_ms": reaction_elapsed_ms,
        "total_elapsed_ms": total_elapsed_ms,
        "from_request_elapsed_ms": from_request_elapsed_ms,
        "worker_boot_id": worker_boot_id,
        "worker_reused": worker_reused,
        "error": error_text,
    }


async def _add_feishu_ack_reaction_inline_async(
    *,
    payload: dict[str, Any],
    request_started_at_ms: int | None = None,
) -> dict[str, Any]:
    worker_context = {
        "worker_boot_id": f"feishu-webhook-inline-{uuid.uuid4().hex}",
        "container_reused": True,
    }
    return await asyncio.to_thread(
        _add_feishu_ack_reaction_from_payload,
        payload,
        request_started_at_ms=request_started_at_ms,
        worker_context=worker_context,
    )


async def _spawn_feishu_ack_reaction_async(
    *,
    payload: dict[str, Any],
    request_started_at: float | None = None,
    request_started_at_ms: int | None = None,
) -> dict[str, Any]:
    context = _extract_feishu_queue_context(payload)
    started_at = time.perf_counter()
    message_id = str(context.get("message_id") or "").strip()
    if not message_id:
        return {
            "status": "skipped",
            "reason": "missing_message_id",
            "schedule_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        }

    worker = globals().get("process_feishu_ack_reaction")
    if worker is None or not hasattr(worker, "spawn"):
        return {
            "status": "unavailable",
            "reason": "process_feishu_ack_reaction_spawn_unavailable",
            "schedule_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "message_id": message_id,
        }

    payload_for_spawn = _with_feishu_internal_meta(
        payload,
        ack_reaction_requested_at_ms=int(time.time() * 1000),
        webhook_message_id=message_id,
    )
    spawn_handle = getattr(worker, "spawn")
    timeout_seconds = max(float(DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS or 0.0), 0.1)
    if request_started_at_ms is None and request_started_at is not None:
        request_started_at_ms = None
    try:
        if hasattr(spawn_handle, "aio"):
            await asyncio.wait_for(
                spawn_handle.aio(
                    payload=payload_for_spawn,
                    request_started_at_ms=request_started_at_ms,
                ),  # type: ignore[union-attr]
                timeout=timeout_seconds,
            )
        else:
            await asyncio.wait_for(
                asyncio.to_thread(
                    spawn_handle,
                    payload=payload_for_spawn,
                    request_started_at_ms=request_started_at_ms,
                ),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "reason": "process_feishu_ack_reaction_spawn_timeout",
            "schedule_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "message_id": message_id,
        }
    return {
        "status": "scheduled",
        "reason": "process_feishu_ack_reaction_spawned",
        "schedule_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        "message_id": message_id,
    }


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


def _prune_feishu_internal_result_files(now_ts: float | None = None) -> None:
    now_value = float(now_ts or time.time())
    expired_tokens = [
        token
        for token, payload in _FEISHU_INTERNAL_RESULT_FILES.items()
        if float(payload.get("expires_at") or 0) <= now_value
    ]
    for token in expired_tokens:
        _FEISHU_INTERNAL_RESULT_FILES.pop(token, None)


def _register_feishu_internal_result_file(
    file_path: str,
    *,
    kind: str,
    is_voice: bool = False,
    ttl_seconds: int = DEFAULT_FEISHU_INTERNAL_RESULT_FILE_TTL_SECONDS,
) -> dict[str, Any] | None:
    normalized_path = str(file_path or "").strip()
    if not normalized_path:
        return None
    path = Path(normalized_path).expanduser()
    if not path.exists() or not path.is_file():
        return None
    token = uuid.uuid4().hex
    expires_at = int(time.time()) + max(60, int(ttl_seconds or DEFAULT_FEISHU_INTERNAL_RESULT_FILE_TTL_SECONDS))
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with _FEISHU_INTERNAL_RESULT_FILE_LOCK:
        _prune_feishu_internal_result_files()
        _FEISHU_INTERNAL_RESULT_FILES[token] = {
            "path": str(path),
            "kind": kind,
            "is_voice": bool(is_voice),
            "content_type": content_type,
            "filename": path.name,
            "expires_at": expires_at,
        }
    return {
        "token": token,
        "kind": kind,
        "is_voice": bool(is_voice),
        "filename": path.name,
        "content_type": content_type,
        "expires_at": expires_at,
    }


def _lookup_feishu_internal_result_file(token: str) -> dict[str, Any] | None:
    normalized_token = str(token or "").strip()
    if not normalized_token:
        return None
    with _FEISHU_INTERNAL_RESULT_FILE_LOCK:
        _prune_feishu_internal_result_files()
        payload = _FEISHU_INTERNAL_RESULT_FILES.get(normalized_token)
        if not payload:
            return None
        path = Path(str(payload.get("path") or ""))
        if not path.exists() or not path.is_file():
            _FEISHU_INTERNAL_RESULT_FILES.pop(normalized_token, None)
            return None
        return dict(payload)


def _build_feishu_internal_source(payload: Mapping[str, Any]) -> Any:
    from gateway.config import Platform
    from gateway.session import SessionSource

    raw_message = payload.get("raw_message") if isinstance(payload.get("raw_message"), Mapping) else {}
    raw_event = raw_message.get("event") if isinstance(raw_message, Mapping) and isinstance(raw_message.get("event"), Mapping) else {}
    chat_id = str(payload.get("chat_id") or "").strip()
    if not chat_id and isinstance(raw_event, dict):
        chat_id = _collect_feishu_chat_id(raw_event)
    user_id = str(payload.get("user_id") or "").strip()
    if not user_id and isinstance(raw_event, dict):
        user_id = _collect_feishu_actor_id(raw_event)
    chat_type = str(payload.get("chat_type") or "dm").strip().lower() or "dm"
    raw_user_name = ""
    if isinstance(raw_event, dict):
        sender = raw_event.get("sender") if isinstance(raw_event.get("sender"), dict) else {}
        operator = raw_event.get("operator") if isinstance(raw_event.get("operator"), dict) else {}
        user = raw_event.get("user") if isinstance(raw_event.get("user"), dict) else {}
        raw_user_name = _first_non_empty_str(
            payload.get("user_name"),
            sender.get("name"),
            operator.get("name"),
            user.get("name"),
            user_id,
            "feishu-user",
        )
    user_name = str(raw_user_name or payload.get("user_name") or user_id or "feishu-user").strip()
    chat_name = str(payload.get("chat_name") or chat_id or "Feishu Chat").strip()
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_name=chat_name,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
        thread_id=str(payload.get("thread_id") or "").strip() or None,
        user_id_alt=str(payload.get("user_id_alt") or "").strip() or None,
        chat_id_alt=str(payload.get("chat_id_alt") or "").strip() or None,
    )


def _normalize_feishu_internal_message_type(raw_value: Any) -> Any:
    from gateway.platforms.base import MessageType

    normalized = str(raw_value or "text").strip().lower()
    for candidate in MessageType:
        if candidate.value == normalized:
            return candidate
    return MessageType.TEXT


def _build_feishu_internal_event(payload: Mapping[str, Any]) -> Any:
    from gateway.platforms.base import MessageEvent

    return MessageEvent(
        text=str(payload.get("text") or "").strip(),
        message_type=_normalize_feishu_internal_message_type(payload.get("message_type")),
        source=_build_feishu_internal_source(payload),
        raw_message=dict(payload.get("raw_message") or {}),
        message_id=str(payload.get("message_id") or "").strip() or None,
        media_urls=[str(item) for item in (payload.get("media_urls") or []) if str(item or "").strip()],
        media_types=[str(item) for item in (payload.get("media_types") or []) if str(item or "").strip()],
        reply_to_message_id=str(payload.get("reply_to_message_id") or "").strip() or None,
        reply_to_text=str(payload.get("reply_to_text") or "").strip() or None,
        internal=bool(payload.get("internal", False)),
    )


def _normalize_feishu_pending_reconciles(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_items = payload.get("pending_reconcile")
    if not isinstance(raw_items, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        correlation_id = str(item.get("correlation_id") or "").strip()
        if not correlation_id or correlation_id in seen_ids:
            continue
        operation_kinds = [
            str(kind).strip()
            for kind in (item.get("operation_kinds") or [])
            if str(kind or "").strip()
        ]
        normalized.append(
            {
                "correlation_id": correlation_id,
                "assistant_text": str(item.get("assistant_text") or "").strip(),
                "final_response": str(item.get("final_response") or "").strip(),
                "delivered_at": str(item.get("delivered_at") or "").strip(),
                "operation_kinds": operation_kinds,
            }
        )
        seen_ids.add(correlation_id)
    return normalized


def _build_feishu_pending_reconcile_text(item: Mapping[str, Any]) -> str:
    assistant_text = str(item.get("assistant_text") or "").strip()
    if assistant_text:
        return assistant_text
    final_response = str(item.get("final_response") or "").strip()
    if final_response:
        return final_response
    placeholders = {
        "image": "[Sent image]",
        "image_url": "[Sent image]",
        "image_file": "[Sent image]",
        "file": "[Sent file]",
        "document_file": "[Sent file]",
        "audio": "[Sent audio]",
        "audio_file": "[Sent audio]",
        "media": "[Sent media]",
        "video_file": "[Sent media]",
        "interactive": "[Sent card]",
    }
    parts: list[str] = []
    seen_parts: set[str] = set()
    for kind in item.get("operation_kinds") or []:
        placeholder = placeholders.get(str(kind).strip().lower())
        if placeholder and placeholder not in seen_parts:
            seen_parts.add(placeholder)
            parts.append(placeholder)
    return "\n".join(parts).strip()


def _apply_feishu_pending_reconciles(
    *,
    runner: Any,
    source: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    session_store = getattr(runner, "session_store", None)
    pending_items = _normalize_feishu_pending_reconciles(payload)
    if session_store is None or not pending_items:
        return {
            "received": len(pending_items),
            "applied": 0,
            "skipped": 0,
            "session_key": str(payload.get("session_key") or "").strip(),
        }

    session_entry = session_store.get_or_create_session(source)
    history = session_store.load_transcript(session_entry.session_id)
    known_ids: set[str] = set()
    for message in history:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "session_meta":
            continue
        if str(message.get("kind") or "").strip().lower() != "pending_reconcile":
            continue
        correlation_id = str(message.get("correlation_id") or message.get("reconcile_id") or "").strip()
        if correlation_id:
            known_ids.add(correlation_id)

    timestamp = datetime.now().isoformat()
    applied = 0
    skipped = 0
    for item in pending_items:
        correlation_id = str(item.get("correlation_id") or "").strip()
        if not correlation_id or correlation_id in known_ids:
            skipped += 1
            continue
        session_store.append_to_transcript(
            session_entry.session_id,
            {
                "role": "session_meta",
                "kind": "pending_reconcile",
                "correlation_id": correlation_id,
                "delivered_at": str(item.get("delivered_at") or "").strip(),
                "operation_kinds": list(item.get("operation_kinds") or []),
                "timestamp": timestamp,
            },
        )
        content = _build_feishu_pending_reconcile_text(item)
        if content:
            session_store.append_to_transcript(
                session_entry.session_id,
                {
                    "role": "assistant",
                    "content": content,
                    "correlation_id": correlation_id,
                    "timestamp": timestamp,
                },
            )
        known_ids.add(correlation_id)
        applied += 1

    session_store.update_session(session_entry.session_key)
    return {
        "received": len(pending_items),
        "applied": applied,
        "skipped": skipped,
        "session_key": session_entry.session_key,
    }


def _download_feishu_internal_attachment(ref: Mapping[str, Any]) -> tuple[str, str] | None:
    from tools.feishu_api import (
        build_download_target_path,
        build_feishu_client,
        extract_filename_from_headers,
        validate_message_resource_type,
    )

    message_id = str(ref.get("message_id") or "").strip()
    file_key = str(ref.get("file_key") or ref.get("image_key") or "").strip()
    if not message_id or not file_key:
        return None
    resource_type = validate_message_resource_type(str(ref.get("resource_type") or "file"))
    default_name = (
        str(ref.get("file_name") or "").strip()
        or f"{resource_type}_{file_key}"
    )
    content, headers = build_feishu_client().request_bytes(
        "GET",
        f"/open-apis/im/v1/messages/{message_id}/resources/{file_key}",
        params={"type": resource_type},
    )
    target_path = build_download_target_path(
        file_name=extract_filename_from_headers(headers, default_name)
    )
    target_path.write_bytes(content)
    media_type = str(headers.get("content-type", "") or "").strip()
    return str(target_path), media_type


async def _hydrate_feishu_internal_event_media(payload: Mapping[str, Any], event: Any) -> Any:
    attachment_refs = payload.get("attachment_refs") or []
    if not isinstance(attachment_refs, list) or not attachment_refs:
        return event

    downloaded_media_urls: list[str] = list(getattr(event, "media_urls", []) or [])
    downloaded_media_types: list[str] = list(getattr(event, "media_types", []) or [])
    for raw_ref in attachment_refs:
        if not isinstance(raw_ref, Mapping):
            continue
        downloaded = await asyncio.to_thread(_download_feishu_internal_attachment, raw_ref)
        if downloaded is None:
            continue
        local_path, media_type = downloaded
        downloaded_media_urls.append(local_path)
        downloaded_media_types.append(media_type)

    if downloaded_media_urls:
        event.media_urls = downloaded_media_urls
        event.media_types = downloaded_media_types
    return event


def _build_feishu_internal_session_state(*, runner: Any, source: Any) -> dict[str, Any]:
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


def _serialize_feishu_internal_operation(operation: Mapping[str, Any], *, adapter: Any) -> dict[str, Any]:
    kind = str(operation.get("kind") or "").strip()
    payload = {
        "kind": kind,
        "chat_id": str(operation.get("chat_id") or "").strip(),
        "message_id": str(operation.get("message_id") or "").strip(),
        "reply_to": str(operation.get("reply_to") or "").strip(),
        "content": str(operation.get("content") or "").strip(),
        "caption": str(operation.get("caption") or "").strip(),
        "metadata": dict(operation.get("metadata") or {}),
    }
    image_url = str(operation.get("image_url") or "").strip()
    if image_url:
        payload["image_url"] = image_url
    file_path = str(operation.get("file_path") or "").strip()
    if file_path:
        kind_hint = kind
        is_voice = kind_hint == "audio_file"
        if kind_hint not in {"audio_file", "video_file", "image_file", "document_file"}:
            attachment_kind = adapter.classify_local_attachment(file_path)
            kind_hint = {
                "audio": "audio_file",
                "video": "video_file",
                "image": "image_file",
            }.get(attachment_kind, "document_file")
        ticket = _register_feishu_internal_result_file(
            file_path,
            kind=kind_hint,
            is_voice=is_voice,
        )
        if ticket:
            payload["file"] = ticket
            payload["file_name"] = str(operation.get("file_name") or ticket.get("filename") or "").strip()
    return payload


def _build_feishu_internal_send_plan(*, adapter: Any, response_text: str | None) -> list[dict[str, Any]]:
    operations = [
        _serialize_feishu_internal_operation(item, adapter=adapter)
        for item in list(getattr(adapter, "captured_operations", []) or [])
        if isinstance(item, dict)
    ]
    response = str(response_text or "").strip()
    if not response:
        return operations

    media_items, cleaned = adapter.extract_media(response)
    image_items, cleaned = adapter.extract_images(cleaned)
    local_files, cleaned = adapter.extract_local_files(cleaned)

    if cleaned.strip():
        operations.append({"kind": "text", "content": cleaned.strip()})

    for image_url, caption in image_items:
        operations.append(
            {
                "kind": "image_url",
                "image_url": image_url,
                "caption": str(caption or "").strip(),
            }
        )

    seen_local_paths: set[str] = set()
    for file_path, is_voice in media_items:
        normalized_path = str(file_path or "").strip()
        if not normalized_path:
            continue
        seen_local_paths.add(normalized_path)
        attachment_kind = adapter.classify_local_attachment(normalized_path)
        kind = {
            "audio": "audio_file",
            "video": "video_file",
            "image": "image_file",
        }.get(attachment_kind, "document_file")
        ticket = _register_feishu_internal_result_file(
            normalized_path,
            kind=kind,
            is_voice=is_voice,
        )
        if ticket:
            operations.append({"kind": kind, "file": ticket})

    for file_path in local_files:
        normalized_path = str(file_path or "").strip()
        if not normalized_path or normalized_path in seen_local_paths:
            continue
        attachment_kind = adapter.classify_local_attachment(normalized_path)
        kind = {
            "audio": "audio_file",
            "video": "video_file",
            "image": "image_file",
        }.get(attachment_kind, "document_file")
        ticket = _register_feishu_internal_result_file(normalized_path, kind=kind)
        if ticket:
            operations.append({"kind": kind, "file": ticket})

    return operations


def _normalize_feishu_internal_route_hint(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"fast_control", "native_io", "cf_browser_first", "modal_heavy_exec"}:
        return normalized
    return "modal_heavy_exec"


def _feishu_message_explicitly_requests_browser_tools(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    patterns = (
        "http://",
        "https://",
        "browser",
        "visit ",
        "open ",
        "navigate",
        "website",
        "浏览器",
        "打开",
        "访问",
        "网页",
        "网站",
        "截图",
        "注册",
        "登录",
        "cloudflare",
        "google",
    )
    return any(pattern in text for pattern in patterns)


def _infer_feishu_internal_route_hint(payload: Mapping[str, Any]) -> str:
    explicit = _normalize_feishu_internal_route_hint(payload.get("route_hint"))
    if explicit != "modal_heavy_exec":
        return explicit
    task_kind = str(payload.get("task_kind") or payload.get("message_type") or "").strip().lower()
    if task_kind == "command":
        return "fast_control"
    attachment_refs = payload.get("attachment_refs")
    if isinstance(attachment_refs, list) and attachment_refs:
        return "modal_heavy_exec"
    if _feishu_message_explicitly_requests_browser_tools(str(payload.get("text") or "")):
        return "cf_browser_first"
    return "modal_heavy_exec"


def _build_feishu_internal_action_plan(send_plan: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    action_plan: list[dict[str, Any]] = []
    for item in list(send_plan or []):
        if isinstance(item, dict):
            action_plan.append(dict(item))
    return action_plan


def _is_feishu_external_exec_candidate(payload: Mapping[str, Any], route_hint: str) -> bool:
    if _normalize_feishu_internal_route_hint(route_hint) != "modal_heavy_exec":
        return False
    task_kind = str(payload.get("task_kind") or payload.get("message_type") or "").strip().lower()
    if task_kind not in {"text", ""}:
        return False
    attachment_refs = payload.get("attachment_refs")
    if isinstance(attachment_refs, list) and attachment_refs:
        return False
    text = str(payload.get("text") or "").strip()
    if not text or text.startswith("/"):
        return False
    if _feishu_message_explicitly_requests_browser_tools(text):
        return False
    return True


def _build_feishu_external_conversation_history(
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


def _build_feishu_provider_plan(session_state: Mapping[str, Any] | None) -> dict[str, Any]:
    state = dict(session_state or {})
    route_debug = dict(state.get("route_debug") or {})
    base_url = str(route_debug.get("base_url") or "").strip()
    current_model = str(state.get("current_model") or "").strip()
    current_provider = str(state.get("current_provider") or "").strip()
    last_model = str(route_debug.get("last_model") or "").strip()
    last_provider = str(route_debug.get("last_provider") or "").strip()
    effective_model = current_model
    effective_provider = current_provider
    if not effective_model or effective_model.lower() in FREE_MODEL_ALIASES:
        effective_model = last_model or os.getenv("HERMES_FEISHU_EXTERNAL_EXEC_MODEL", "openai/gpt-oss-20b:free")
    if not effective_provider:
        effective_provider = last_provider or "openrouter"
    request_timeout_ms = max(1000, int(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_TIMEOUT_MS", "4500") or "4500"))
    max_attempts = max(1, min(5, int(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_MAX_ATTEMPTS", "1") or "1")))
    retry_delay_ms = max(0, min(5000, int(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_RETRY_DELAY_MS", "250") or "250")))
    backoff = str(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_BACKOFF", "linear") or "linear").strip().lower()
    if backoff not in {"constant", "linear", "exponential"}:
        backoff = "linear"
    fallback_model = str(os.getenv("HERMES_FEISHU_EXTERNAL_EXEC_FALLBACK_MODEL") or "").strip()
    fallback_provider = str(os.getenv("HERMES_FEISHU_EXTERNAL_EXEC_FALLBACK_PROVIDER") or effective_provider or "").strip().lower()
    cache_mode = str(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_CACHE_MODE", "skip") or "skip").strip().lower()
    if cache_mode not in {"skip", "ttl"}:
        cache_mode = "skip"
    cache_ttl_seconds = max(60, min(31 * 24 * 3600, int(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_CACHE_TTL_SECONDS", "300") or "300")))
    cache_scope = str(os.getenv("HERMES_CF_AI_GATEWAY_DIRECT_CACHE_SCOPE", "user") or "user").strip().lower()
    if cache_scope not in {"user", "chat", "global"}:
        cache_scope = "user"
    current_personality = str(state.get("current_personality") or "").strip().lower()
    plan: dict[str, Any] = {
        "mode": "cloudflare_workflow_candidate",
        "wait_strategy": "cloudflare_wait",
        "task_profile": "plain_text_llm",
        "reason": "plain_text_without_attachments_or_browser",
        "model": effective_model,
        "provider": effective_provider,
        "base_url": base_url,
        "request_timeout_ms": request_timeout_ms,
        "max_attempts": max_attempts,
        "retry_delay_ms": retry_delay_ms,
        "backoff": backoff,
        "cache_mode": cache_mode,
        "cache_scope": cache_scope,
        "byok_alias": str(os.getenv("HERMES_CF_AI_GATEWAY_BYOK_ALIAS", "default") or "default").strip(),
    }
    if cache_mode == "ttl":
        plan["cache_ttl_seconds"] = cache_ttl_seconds
    if current_model:
        plan["session_model"] = current_model
    if last_model:
        plan["last_model"] = last_model
    if fallback_model and fallback_model != effective_model:
        plan["fallback_model"] = fallback_model
        plan["fallback_provider"] = fallback_provider or effective_provider
    if current_personality:
        plan["personality"] = current_personality
    return plan


def _build_feishu_internal_plan(
    payload: Mapping[str, Any],
    *,
    session_state_before: Mapping[str, Any] | None = None,
    llm_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    route_hint = _infer_feishu_internal_route_hint(payload)
    external_exec_candidate = _is_feishu_external_exec_candidate(payload, route_hint)
    execution_mode = "deferred_reconcile" if external_exec_candidate else route_hint
    if execution_mode not in {"control_complete", "native_io_complete", "cf_browser_first", "modal_heavy_exec", "deferred_reconcile"}:
        execution_mode = "modal_heavy_exec"
    normalized_session_state_before = dict(session_state_before or {})
    provider_plan: dict[str, Any] = {}
    if external_exec_candidate:
        provider_plan = _build_feishu_provider_plan(normalized_session_state_before)
    return _build_feishu_internal_result(
        status="ok",
        route_hint=route_hint,
        execution_mode=execution_mode,
        session_state_before=normalized_session_state_before,
        session_state_after=normalized_session_state_before,
        send_plan=[],
        final_response="",
        provider_usage={},
        reconcile_required=bool(external_exec_candidate),
        browser_fallback_allowed=route_hint == "cf_browser_first",
        provider_plan=provider_plan,
        external_exec_candidate=external_exec_candidate,
        llm_request=dict(llm_request or {}),
    )


def _build_feishu_internal_result(
    *,
    status: str,
    route_hint: str,
    execution_mode: str,
    session_state_before: Mapping[str, Any] | None,
    session_state_after: Mapping[str, Any] | None,
    send_plan: list[dict[str, Any]] | None = None,
    final_response: str | None = None,
    provider_usage: Mapping[str, Any] | None = None,
    reconcile_required: bool = False,
    browser_fallback_allowed: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    normalized_send_plan = list(send_plan or [])
    return {
        "status": status,
        "route_hint": _normalize_feishu_internal_route_hint(route_hint),
        "execution_mode": str(execution_mode or "modal_heavy_exec").strip() or "modal_heavy_exec",
        "final_response": str(final_response or ""),
        "send_plan": normalized_send_plan,
        "action_plan": _build_feishu_internal_action_plan(normalized_send_plan),
        "session_state_before": dict(session_state_before or {}),
        "session_state_after": dict(session_state_after or {}),
        "provider_usage": dict(provider_usage or {}),
        "reconcile_required": bool(reconcile_required),
        "browser_fallback_allowed": bool(browser_fallback_allowed),
        **extra,
    }


async def _initialize_feishu_internal_gateway_runtime(settings: RuntimeSettings) -> _FeishuGatewayRuntime:
    from gateway.config import Platform, PlatformConfig
    from gateway.run import GatewayRunner

    runner = GatewayRunner()
    live_runtime = _FEISHU_RUNTIME
    if live_runtime is not None and getattr(live_runtime, "runner", None) is not None:
        live_runner = live_runtime.runner
        runner.session_store = live_runner.session_store
        runner._voice_mode = dict(getattr(live_runner, "_voice_mode", {}) or {})
        runner._session_model_overrides = dict(getattr(live_runner, "_session_model_overrides", {}) or {})
        runner._pending_model_notes = dict(getattr(live_runner, "_pending_model_notes", {}) or {})
        runner._ephemeral_system_prompt = str(getattr(live_runner, "_ephemeral_system_prompt", "") or "")

    feishu_config = runner.config.platforms.get(Platform.FEISHU) or PlatformConfig()
    feishu_config.enabled = True
    feishu_config.extra.update(
        {
            "app_id": settings.feishu_app_id,
            "app_secret": settings.feishu_app_secret,
            "domain": settings.feishu_domain or "feishu",
            "connection_mode": "internal_capture",
            "webhook_path": "/internal/feishu",
        }
    )
    adapter = _make_feishu_internal_capture_adapter(feishu_config)
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(runner.session_store)
    runner.adapters[Platform.FEISHU] = adapter
    runner.delivery_router.adapters = runner.adapters
    runner._sync_voice_mode_state_to_adapter(adapter)
    return _FeishuGatewayRuntime(
        runner=runner,
        adapter=adapter,
        init_timings={"mode": "internal_capture"},
    )


async def _get_feishu_internal_gateway_runtime() -> _FeishuGatewayRuntime:
    global _FEISHU_INTERNAL_RUNTIME
    if _FEISHU_INTERNAL_RUNTIME is not None:
        return _FEISHU_INTERNAL_RUNTIME

    async with _get_feishu_internal_runtime_lock():
        if _FEISHU_INTERNAL_RUNTIME is not None:
            return _FEISHU_INTERNAL_RUNTIME
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        _FEISHU_INTERNAL_RUNTIME = await _initialize_feishu_internal_gateway_runtime(settings)
        return _FEISHU_INTERNAL_RUNTIME


async def _run_feishu_internal_agent_exec(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_dict = dict(payload or {})
    correlation_id = str(payload_dict.get("correlation_id") or "").strip()
    event_id = str(payload_dict.get("event_id") or "").strip()
    chat_id = str(payload_dict.get("chat_id") or "").strip()
    message_id = str(payload_dict.get("message_id") or "").strip()
    route_hint = _infer_feishu_internal_route_hint(payload_dict)
    browser_fallback_allowed = route_hint == "cf_browser_first"
    started_at = time.perf_counter()
    _append_feishu_trace(
        "internal.agent_exec.start",
        payload_dict,
        correlation_id=correlation_id,
    )
    logger.warning(
        "[Feishu] internal agent exec start correlation_id=%s event_id=%s chat_id=%s message_id=%s",
        correlation_id or "none",
        event_id or "none",
        chat_id or "none",
        message_id or "none",
    )
    runtime = await _get_feishu_internal_gateway_runtime()
    runner = runtime.runner
    adapter = runtime.adapter
    adapter.captured_operations.clear()
    source = _build_feishu_internal_source(payload_dict)
    reconcile_summary = _apply_feishu_pending_reconciles(
        runner=runner,
        source=source,
        payload=payload_dict,
    )
    if int(reconcile_summary.get("received") or 0) > 0:
        _append_feishu_trace(
            "internal.agent_exec.reconcile",
            payload_dict,
            correlation_id=correlation_id,
            reconcile_summary=reconcile_summary,
        )
        logger.warning(
            "[Feishu] internal agent exec reconcile correlation_id=%s session_key=%s received=%s applied=%s skipped=%s",
            correlation_id or "none",
            reconcile_summary.get("session_key") or "none",
            reconcile_summary.get("received"),
            reconcile_summary.get("applied"),
            reconcile_summary.get("skipped"),
        )
    event = await _hydrate_feishu_internal_event_media(payload_dict, _build_feishu_internal_event(payload_dict))
    session_state_before = _build_feishu_internal_session_state(runner=runner, source=event.source)
    try:
        response_text = await runner._handle_message(event)
        session_state_after = _build_feishu_internal_session_state(runner=runner, source=event.source)
        send_plan = _build_feishu_internal_send_plan(adapter=adapter, response_text=response_text)
        agent_exec_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        _append_feishu_trace(
            "internal.agent_exec.done",
            payload_dict,
            correlation_id=correlation_id,
            agent_exec_elapsed_ms=agent_exec_elapsed_ms,
            send_plan_operation_count=len(send_plan),
            response_length=len(str(response_text or "")),
            route_hint=route_hint,
            execution_mode="modal_heavy_exec",
        )
        logger.warning(
            "[Feishu] internal agent exec done correlation_id=%s event_id=%s chat_id=%s message_id=%s agent_exec_elapsed_ms=%s send_plan_operation_count=%s response_length=%s route_hint=%s",
            correlation_id or "none",
            event_id or "none",
            chat_id or "none",
            message_id or "none",
            agent_exec_elapsed_ms,
            len(send_plan),
            len(str(response_text or "")),
            route_hint,
        )
        return _build_feishu_internal_result(
            status="ok",
            route_hint=route_hint,
            execution_mode="modal_heavy_exec",
            session_state_before=session_state_before,
            session_state_after=session_state_after,
            final_response=str(response_text or ""),
            send_plan=send_plan,
            reconcile_required=bool(send_plan or str(response_text or "").strip()),
            browser_fallback_allowed=browser_fallback_allowed,
            correlation_id=correlation_id,
            reconcile_summary=reconcile_summary,
        )
    except Exception as exc:
        agent_exec_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        _append_feishu_trace(
            "internal.agent_exec.error",
            payload_dict,
            correlation_id=correlation_id,
            agent_exec_elapsed_ms=agent_exec_elapsed_ms,
            error=str(exc),
            route_hint=route_hint,
        )
        logger.exception(
            "[Feishu] internal agent exec failed correlation_id=%s event_id=%s chat_id=%s message_id=%s agent_exec_elapsed_ms=%s route_hint=%s",
            correlation_id or "none",
            event_id or "none",
            chat_id or "none",
            message_id or "none",
            agent_exec_elapsed_ms,
            route_hint,
        )
        raise


async def _run_feishu_internal_agent_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_dict = dict(payload or {})
    correlation_id = str(payload_dict.get("correlation_id") or "").strip()
    route_hint = _infer_feishu_internal_route_hint(payload_dict)
    runtime = await _get_feishu_internal_gateway_runtime()
    runner = runtime.runner
    source = _build_feishu_internal_source(payload_dict)
    session_state_before = _build_feishu_internal_session_state(runner=runner, source=source)
    llm_request: dict[str, Any] = {}
    if _is_feishu_external_exec_candidate(payload_dict, route_hint):
        llm_request = {
            "model": str(session_state_before.get("current_model") or ""),
            "messages": _build_feishu_external_conversation_history(
                runner=runner,
                source=source,
                latest_user_text=str(payload_dict.get("text") or ""),
            ),
        }
    plan = _build_feishu_internal_plan(
        payload_dict,
        session_state_before=session_state_before,
        llm_request=llm_request,
    )
    _append_feishu_trace(
        "internal.agent_plan.done",
        payload_dict,
        correlation_id=correlation_id,
        route_hint=route_hint,
        execution_mode=str(plan.get("execution_mode") or ""),
        external_exec_candidate=bool(plan.get("external_exec_candidate")),
    )
    logger.warning(
        "[Feishu] internal agent plan correlation_id=%s route_hint=%s execution_mode=%s external_exec_candidate=%s",
        correlation_id or "none",
        route_hint,
        str(plan.get("execution_mode") or "none"),
        bool(plan.get("external_exec_candidate")),
    )
    return plan


async def _run_feishu_internal_control(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = await _get_feishu_internal_gateway_runtime()
    runner = runtime.runner
    adapter = runtime.adapter
    action = str(payload.get("action") or "").strip().lower()
    source = _build_feishu_internal_source(payload)
    reconcile_summary = _apply_feishu_pending_reconciles(
        runner=runner,
        source=source,
        payload=payload,
    )
    if int(reconcile_summary.get("received") or 0) > 0:
        _append_feishu_trace(
            "internal.control.reconcile",
            dict(payload or {}),
            correlation_id=str(payload.get("correlation_id") or "").strip(),
            reconcile_summary=reconcile_summary,
        )
        logger.warning(
            "[Feishu] internal control reconcile action=%s session_key=%s received=%s applied=%s skipped=%s",
            action or "none",
            reconcile_summary.get("session_key") or "none",
            reconcile_summary.get("received"),
            reconcile_summary.get("applied"),
            reconcile_summary.get("skipped"),
        )
    session_state_before = _build_feishu_internal_session_state(runner=runner, source=source)

    if action == "render_card":
        event_key = str(payload.get("event_key") or "").strip()
        card = _build_feishu_local_menu_card(event_key)
        return _build_feishu_internal_result(
            status="ok" if card else "error",
            route_hint="fast_control",
            execution_mode="control_complete",
            session_state_before=session_state_before,
            session_state_after=session_state_before,
            action=action,
            event_key=event_key,
            card=card,
            error="" if card else f"unsupported event_key: {event_key}",
        )

    if action == "activate_skill_combo":
        combo_id = str(payload.get("combo_id") or "").strip()
        combo = next(
            (item for item in _FEISHU_SKILL_COMBO_DEFINITIONS if str(item.get("id") or "").strip() == combo_id),
            None,
        )
        if combo is None:
            return _build_feishu_internal_result(
                status="error",
                route_hint="fast_control",
                execution_mode="control_complete",
                session_state_before=session_state_before,
                session_state_after=session_state_before,
                action=action,
                error=f"unknown combo_id: {combo_id}",
            )
        from agent.skill_commands import build_session_start_skills_message
        from gateway.platforms.base import MessageEvent, MessageType

        combo_label = str(combo.get("label") or combo_id or "skill-combo").strip()
        suggested_personality = str(combo.get("suggested_personality") or "").strip().lower()
        user_instruction = (
            f"请切换到「{combo_label}」工作模式。"
            "先用中文 3 行内确认已加载的技能、适用场景和下一步协作方式。"
        )
        if suggested_personality:
            user_instruction += f" 如需匹配风格，建议配合 `/personality {suggested_personality}`。"
        user_instruction = (
            f"请切换到“{combo_label}”工作模式。"
            "先用中文在 3 行内确认已加载的技能、适用场景，以及接下来准备如何协作。"
        )
        if suggested_personality:
            user_instruction += (
                f" 如需匹配风格，建议同步执行 `/personality {suggested_personality}`。"
            )
        skill_message, _loaded_skills, missing_skills = build_session_start_skills_message(
            [str(item) for item in (combo.get('skills') or [])],
            user_instruction=user_instruction,
        )
        text = skill_message.strip() if skill_message else user_instruction
        if missing_skills:
            text += f"\n\n[Missing skills: {', '.join(missing_skills)}]"
        adapter.captured_operations.clear()
        response_text = await runner._handle_message(
            MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=None,
                message_id=str(payload.get("message_id") or "").strip() or None,
            )
        )
        session_state_after = _build_feishu_internal_session_state(runner=runner, source=source)
        send_plan = _build_feishu_internal_send_plan(adapter=adapter, response_text=response_text)
        return _build_feishu_internal_result(
            status="ok",
            route_hint="fast_control",
            execution_mode="control_complete",
            session_state_before=session_state_before,
            session_state_after=session_state_after,
            action=action,
            final_response=str(response_text or ""),
            send_plan=send_plan,
            reconcile_required=False,
        )

    if action in {"dispatch_command", "get_session_state"}:
        if action == "get_session_state":
            return _build_feishu_internal_result(
                status="ok",
                route_hint="fast_control",
                execution_mode="control_complete",
                session_state_before=session_state_before,
                session_state_after=session_state_before,
                action=action,
            )
        from gateway.platforms.base import MessageEvent, MessageType

        command_text = str(payload.get("command_text") or "").strip()
        if not command_text:
            return _build_feishu_internal_result(
                status="error",
                route_hint="fast_control",
                execution_mode="control_complete",
                session_state_before=session_state_before,
                session_state_after=session_state_before,
                action=action,
                error="missing command_text",
            )
        adapter.captured_operations.clear()
        response_text = await runner._handle_message(
            MessageEvent(
                text=command_text,
                message_type=MessageType.COMMAND,
                source=source,
                raw_message=None,
                message_id=str(payload.get("message_id") or "").strip() or None,
            )
        )
        session_state_after = _build_feishu_internal_session_state(runner=runner, source=source)
        send_plan = _build_feishu_internal_send_plan(adapter=adapter, response_text=response_text)
        return _build_feishu_internal_result(
            status="ok",
            route_hint="fast_control",
            execution_mode="control_complete",
            session_state_before=session_state_before,
            session_state_after=session_state_after,
            action=action,
            final_response=str(response_text or ""),
            send_plan=send_plan,
            reconcile_required=False,
        )

    return _build_feishu_internal_result(
        status="error",
        route_hint="fast_control",
        execution_mode="control_complete",
        session_state_before=session_state_before,
        session_state_after=session_state_before,
        action=action,
        error=f"unsupported action: {action}",
    )


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
    headers = dict(request.headers)

    content_type = str(headers.get("content-type", "") or "").split(";", 1)[0].strip().lower()
    if content_type and content_type != "application/json":
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 1024 * 1024:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError:
            pass

    body = await request.body()
    if len(body) > 1024 * 1024:
        raise HTTPException(status_code=413, detail="Request body too large")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="invalid json")

    encrypt_key, verification_token = _get_feishu_webhook_security_settings()
    if encrypt_key and not _is_feishu_webhook_signature_valid(headers, body, encrypt_key=encrypt_key):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if payload.get("encrypt"):
        try:
            payload = _decrypt_feishu_webhook_payload(str(payload.get("encrypt") or ""), encrypt_key=encrypt_key)
        except Exception:
            logger.exception("Feishu encrypted webhook decrypt failed")
            raise HTTPException(status_code=400, detail="failed to decrypt webhook payload")

    if verification_token:
        header = payload.get("header") or {}
        incoming_token = str(header.get("token") or payload.get("token") or "")
        if not incoming_token or not hmac.compare_digest(incoming_token, verification_token):
            raise HTTPException(status_code=401, detail="Invalid verification token")

    return None, payload


def _extract_feishu_event_metadata(payload: dict[str, Any]) -> tuple[str, str]:
    header = payload.get("header") or {}
    event_id = str(header.get("event_id") or payload.get("event_id") or "").strip()
    event_type = str(header.get("event_type") or "").strip()
    return event_id, event_type


def _extract_leading_command_text(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    tokens = normalized.split()
    while tokens:
        first = tokens[0]
        if first.startswith("@") or first.startswith("[@"):
            tokens.pop(0)
            continue
        if first.startswith("<at"):
            while tokens:
                token = tokens.pop(0)
                if "</at>" in token:
                    break
            continue
        break
    if not tokens or not tokens[0].startswith("/"):
        return ""
    return " ".join(tokens)


def _resolve_inline_fast_command(command_text: str) -> str | None:
    candidate = _extract_leading_command_text(command_text)
    if not candidate:
        return None
    command_token = candidate.split(None, 1)[0].lstrip("/")
    if not command_token:
        return None
    command_token = command_token.split("@", 1)[0].strip().lower()
    if not command_token:
        return None
    return command_token if command_token in INLINE_FAST_COMMAND_CANONICALS else None


def _extract_feishu_text_content_from_raw_content(raw_content: str) -> str:
    raw = str(raw_content or "")
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw
    if isinstance(parsed, dict):
        text_value = parsed.get("text", "")
        return str(text_value or "")
    return raw


def _extract_feishu_trace_token(payload: dict[str, Any]) -> str:
    event = payload.get("event") or {}
    message = event.get("message") or {}
    raw_content = str(message.get("content") or "")
    if not raw_content:
        return ""
    try:
        text_content = _extract_feishu_text_content_from_raw_content(raw_content)
    except Exception:
        text_content = raw_content
    match = _FEISHU_TRACE_TOKEN_RE.search(str(text_content or ""))
    return str(match.group(1) or "").strip() if match else ""


def _extract_feishu_inline_fast_command(payload: dict[str, Any]) -> str | None:
    event = payload.get("event") or {}
    message = event.get("message") or {}
    message_type = str(message.get("message_type") or "").strip().lower()
    if message_type != "text":
        return None
    raw_content = str(message.get("content") or "")
    if not raw_content:
        return None
    try:
        text_content = _extract_feishu_text_content_from_raw_content(raw_content)
    except Exception:
        text_content = raw_content
    return _resolve_inline_fast_command(text_content)


def _classify_feishu_chat_lane(payload: dict[str, Any]) -> str:
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return "chat_light"
    message = event.get("message") or {}
    if not isinstance(message, dict):
        return "chat_light"
    message_type = str(message.get("message_type") or "").strip().lower()
    if message_type and message_type != "text":
        return "chat_heavy"
    raw_content = str(message.get("content") or "")
    text_length = len(raw_content)
    if text_length >= 4000:
        return "chat_heavy"
    if any(key in raw_content for key in ("image_key", "file_key", "audio_key", "media_key")):
        return "chat_heavy"
    mentions = event.get("mentions") or message.get("mentions") or []
    if isinstance(mentions, list) and len(mentions) >= 5:
        return "chat_heavy"
    return "chat_light"


def _extract_telegram_inline_fast_command(update: dict[str, Any]) -> str | None:
    message = (
        update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
        or {}
    )
    text = str(message.get("text") or "").strip()
    if not text:
        return None
    return _resolve_inline_fast_command(text)


def _should_inline_feishu_control_event(event_type: str) -> bool:
    normalized = str(event_type or "").strip().lower()
    return normalized in {"application.bot.menu_v6", "card.action.trigger"}


def _is_feishu_session_warmup_event(event_type: str) -> bool:
    normalized = str(event_type or "").strip().lower()
    return "bot_p2p_chat_entered" in normalized


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
) -> dict[str, Any]:
    runtime_started_at = time.perf_counter()
    runtime_cached_before = _FEISHU_RUNTIME is not None
    runtime = await _get_feishu_gateway_runtime()
    runtime_acquire_elapsed_ms = int((time.perf_counter() - runtime_started_at) * 1000)
    adapter = runtime.adapter
    event_id, event_type = _extract_feishu_event_metadata(payload)
    trace_token = _extract_feishu_trace_token(payload)
    data = adapter._namespace_from_mapping(payload)
    dispatch_phase_timings: dict[str, int] = {}

    _append_feishu_trace(
        "dispatch.start",
        payload,
        runtime_acquire_elapsed_ms=runtime_acquire_elapsed_ms,
        runtime_cached=runtime_cached_before,
    )
    logger.warning(
        "[Feishu] dispatch start event_type=%s event_id=%s trace_token=%s runtime_acquire_elapsed_ms=%s runtime_cached=%s",
        event_type or "unknown",
        event_id or "none",
        trace_token or "none",
        runtime_acquire_elapsed_ms,
        runtime_cached_before,
    )

    try:
        if event_type == "im.message.receive_v1":
            handle_started_at = time.perf_counter()
            await adapter._handle_message_event_data(data)
            _capture_phase_elapsed(dispatch_phase_timings, "message_handle_elapsed_ms", handle_started_at)
            if await_background_tasks:
                pending_started_at = time.perf_counter()
                await _await_feishu_pending_batches(adapter)
                _capture_phase_elapsed(dispatch_phase_timings, "pending_batches_elapsed_ms", pending_started_at)
                background_started_at = time.perf_counter()
                await _await_feishu_background_tasks(adapter)
                _capture_phase_elapsed(dispatch_phase_timings, "background_tasks_elapsed_ms", background_started_at)
        elif event_type == "im.message.message_read_v1":
            handle_started_at = time.perf_counter()
            adapter._on_message_read_event(data)
            _capture_phase_elapsed(dispatch_phase_timings, "message_read_handle_elapsed_ms", handle_started_at)
        elif event_type == "im.chat.member.bot.added_v1":
            handle_started_at = time.perf_counter()
            adapter._on_bot_added_to_chat(data)
            _capture_phase_elapsed(dispatch_phase_timings, "bot_added_handle_elapsed_ms", handle_started_at)
        elif event_type == "im.chat.member.bot.deleted_v1":
            handle_started_at = time.perf_counter()
            adapter._on_bot_removed_from_chat(data)
            _capture_phase_elapsed(dispatch_phase_timings, "bot_deleted_handle_elapsed_ms", handle_started_at)
        elif event_type in ("im.message.reaction.created_v1", "im.message.reaction.deleted_v1"):
            handle_started_at = time.perf_counter()
            await adapter._handle_reaction_event(event_type, data)
            _capture_phase_elapsed(dispatch_phase_timings, "reaction_handle_elapsed_ms", handle_started_at)
            if await_background_tasks:
                background_started_at = time.perf_counter()
                await _await_feishu_background_tasks(adapter)
                _capture_phase_elapsed(dispatch_phase_timings, "background_tasks_elapsed_ms", background_started_at)
        elif event_type == "card.action.trigger":
            if await_background_tasks:
                handle_started_at = time.perf_counter()
                await adapter._handle_card_action_event(data)
                _capture_phase_elapsed(dispatch_phase_timings, "card_action_handle_elapsed_ms", handle_started_at)
                background_started_at = time.perf_counter()
                await _await_feishu_background_tasks(adapter)
                _capture_phase_elapsed(dispatch_phase_timings, "background_tasks_elapsed_ms", background_started_at)
            else:
                background = getattr(adapter, "_background_tasks", None)
                task = asyncio.create_task(adapter._handle_card_action_event(data))
                if isinstance(background, set):
                    background.add(task)
                    def _cleanup_background_card_action(done_task: asyncio.Task[Any]) -> None:
                        background.discard(done_task)
                        if done_task.cancelled():
                            return
                        try:
                            exc = done_task.exception()
                        except Exception:
                            logger.exception("[Feishu] Failed to inspect card action background task")
                            return
                        if exc is not None:
                            logger.exception("[Feishu] Card action background task failed", exc_info=exc)

                    task.add_done_callback(_cleanup_background_card_action)
        elif event_type == "application.bot.menu_v6":
            handle_started_at = time.perf_counter()
            await adapter._handle_bot_menu_event(data)
            _capture_phase_elapsed(dispatch_phase_timings, "menu_handle_elapsed_ms", handle_started_at)
            if await_background_tasks:
                background_started_at = time.perf_counter()
                await _await_feishu_background_tasks(adapter)
                _capture_phase_elapsed(dispatch_phase_timings, "background_tasks_elapsed_ms", background_started_at)
        else:
            logger.warning("[Feishu] Ignoring unsupported event type in dispatcher: %s", event_type or "unknown")
    except Exception as exc:
        normalized_phase_timings = _normalize_phase_timings(dispatch_phase_timings)
        _append_feishu_trace("dispatch.error", payload, error=str(exc), phase_timings=normalized_phase_timings)
        raise
    else:
        normalized_phase_timings = _normalize_phase_timings(dispatch_phase_timings)
        _append_feishu_trace("dispatch.done", payload, phase_timings=normalized_phase_timings)
        logger.warning(
            "[Feishu] dispatch done event_type=%s event_id=%s phase_timings=%s",
            event_type or "unknown",
            event_id or "none",
            normalized_phase_timings,
        )
        return {
            "event_id": event_id,
            "event_type": event_type,
            "runtime_acquire_elapsed_ms": runtime_acquire_elapsed_ms,
            "phase_timings": normalized_phase_timings,
            "await_background_tasks": await_background_tasks,
        }


async def _dispatch_feishu_update(request: Request) -> Response:
    _adapter, payload = await _parse_feishu_webhook_request(request)
    if payload.get("type") == "url_verification":
        return JSONResponse({"challenge": payload.get("challenge", "")})
    await _dispatch_feishu_payload(payload)
    return JSONResponse({"code": 0, "msg": "ok"})


async def _process_chat_queue_item_async(
    payload: Any,
    *,
    worker_context: dict[str, Any] | None = None,
    runtime_prepared: bool = False,
) -> dict[str, Any]:
    context = dict(worker_context or {})
    if not runtime_prepared:
        inline_context = _build_inline_chat_worker_context()
        inline_context.update(context)
        context = inline_context

    if _is_truthy(os.getenv("HERMES_MODAL_CHAT_WORKER_RELOAD"), default=False):
        await _sync_modal_volume_async(reload=True)

    envelope = payload if isinstance(payload, dict) else {}
    platform = str(envelope.get("platform") or "").strip().lower()
    partition = str(envelope.get("partition") or "").strip()
    raw_payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    metadata = envelope.get("metadata") if isinstance(envelope.get("metadata"), dict) else {}
    enqueued_at_ms = int(envelope.get("enqueued_at_ms") or 0)
    queue_latency_ms = max(0, int(time.time() * 1000) - enqueued_at_ms) if enqueued_at_ms else None
    lane = str(metadata.get("lane") or "").strip() or None
    ingress_strategy = str(metadata.get("ingress_strategy") or "").strip() or None
    warmup_ready_at = int(context.get("warmup_ready_at") or 0)
    if warmup_ready_at and enqueued_at_ms:
        context["warmup_message_gap_ms"] = max(0, enqueued_at_ms - warmup_ready_at)

    if platform not in {"feishu", "telegram"}:
        return {
            "status": "skipped",
            "reason": "unsupported_platform",
            "platform": platform,
            "partition": partition,
            **_build_chat_worker_observability(context),
        }
    if not raw_payload:
        return {
            "status": "skipped",
            "reason": "missing_payload",
            "platform": platform,
            "partition": partition,
            **_build_chat_worker_observability(context),
        }
    if platform == "feishu":
        _append_feishu_trace(
            "worker.start",
            raw_payload,
            partition=partition,
            lane=lane,
            ingress_strategy=ingress_strategy or "",
            queue_latency_ms=queue_latency_ms,
            **_build_chat_worker_observability(context),
        )
        logger.warning(
            "[Feishu] worker start event_type=%s event_id=%s trace_token=%s partition=%s lane=%s queue_latency_ms=%s worker_boot_id=%s reused=%s warmup_status=%s warmup_age_ms=%s warmup_same_container=%s ingress_strategy=%s",
            metadata.get("event_type") or "unknown",
            metadata.get("event_id") or "none",
            metadata.get("trace_token") or "none",
            partition or "none",
            lane or "none",
            queue_latency_ms if queue_latency_ms is not None else "na",
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
            context.get("warmup_status") or "none",
            context.get("warmup_age_ms") if context.get("warmup_age_ms") is not None else "na",
            context.get("warmup_same_container") if context.get("warmup_same_container") is not None else "na",
            ingress_strategy or "",
        )
    else:
        logger.info(
            "Telegram worker start update_id=%s partition=%s queue_latency_ms=%s worker_boot_id=%s reused=%s",
            metadata.get("event_id") or "none",
            partition or "none",
            queue_latency_ms if queue_latency_ms is not None else "na",
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
        )

    started_at = time.time()
    try:
        if platform == "feishu":
            event_type = str(metadata.get("event_type") or "").strip().lower()
            dispatch_coro = _dispatch_feishu_payload(raw_payload, await_background_tasks=True)
            if event_type == "im.message.receive_v1":
                await asyncio.wait_for(dispatch_coro, timeout=DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS)
            else:
                await dispatch_coro
        else:
            await _dispatch_telegram_update(raw_payload)
    except asyncio.TimeoutError:
        timeout_error = (
            f"feishu dispatch exceeded {DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS:.1f}s "
            f"for {str(metadata.get('event_type') or 'unknown')}"
        )
        if platform == "feishu":
            _append_feishu_trace(
                "worker.timeout",
                raw_payload,
                partition=partition,
                lane=lane,
                queue_latency_ms=queue_latency_ms,
                error=timeout_error,
                **_build_chat_worker_observability(context),
            )
            logger.error(
                "[Feishu] worker timeout event_type=%s event_id=%s partition=%s timeout_seconds=%.1f",
                metadata.get("event_type") or "unknown",
                metadata.get("event_id") or "none",
                partition or "none",
                DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS,
            )
        return {
            "status": "error",
            "platform": platform,
            "partition": partition,
            "event_id": metadata.get("event_id"),
            "message_id": metadata.get("message_id"),
            "queue_latency_ms": queue_latency_ms,
            "error": timeout_error,
            **_build_chat_worker_observability(context),
        }
    except Exception as exc:
        if platform == "feishu":
            _append_feishu_trace(
                "worker.error",
                raw_payload,
                partition=partition,
                lane=lane,
                queue_latency_ms=queue_latency_ms,
                error=str(exc),
                **_build_chat_worker_observability(context),
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
            **_build_chat_worker_observability(context),
        }

    elapsed_ms = int((time.time() - started_at) * 1000)
    if platform == "feishu":
        _append_feishu_trace(
                "worker.done",
                raw_payload,
                partition=partition,
                lane=lane,
                queue_latency_ms=queue_latency_ms,
                worker_elapsed_ms=elapsed_ms,
                **_build_chat_worker_observability(context),
        )
        logger.warning(
            "[Feishu] worker done event_type=%s event_id=%s trace_token=%s partition=%s worker_elapsed_ms=%s worker_boot_id=%s reused=%s",
            metadata.get("event_type") or "unknown",
            metadata.get("event_id") or "none",
            metadata.get("trace_token") or "none",
            partition or "none",
            elapsed_ms,
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
        )
    else:
        logger.info(
            "Telegram worker done update_id=%s partition=%s worker_elapsed_ms=%s worker_boot_id=%s reused=%s",
            metadata.get("event_id") or "none",
            partition or "none",
            elapsed_ms,
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
        )
    return {
        "status": "ok",
        "platform": platform,
        "partition": partition,
        "event_id": metadata.get("event_id"),
        "message_id": metadata.get("message_id"),
        "queue_latency_ms": queue_latency_ms,
        "worker_elapsed_ms": elapsed_ms,
        **_build_chat_worker_observability(context),
    }


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


async def _process_chat_queue_items_async(
    items: list[Any],
    *,
    worker_context: dict[str, Any] | None = None,
    runtime_prepared: bool = False,
) -> list[dict[str, Any]]:
    items = _coalesce_feishu_chat_queue_items(items)
    return [
        await _process_chat_queue_item_async(
            item,
            worker_context=worker_context,
            runtime_prepared=runtime_prepared,
        )
        for item in items
    ]


def _coalesce_feishu_chat_queue_items(items: list[Any]) -> list[Any]:
    if not items:
        return items

    now_ms = int(time.time() * 1000)
    max_age_ms = max(int(DEFAULT_FEISHU_MESSAGE_QUEUE_MAX_AGE_SECONDS or 0), 1) * 1000
    newest_message_index_by_partition: dict[str, int] = {}
    filtered: list[Any] = []

    for item in items:
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        if str(item.get("platform") or "").strip().lower() != "feishu":
            filtered.append(item)
            continue

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            filtered.append(item)
            continue
        event_type = str(metadata.get("event_type") or "").strip()
        if event_type != "im.message.receive_v1":
            filtered.append(item)
            continue

        partition = str(item.get("partition") or metadata.get("partition") or "").strip()
        enqueued_at_ms = int(item.get("enqueued_at_ms") or 0)
        age_ms = max(0, now_ms - enqueued_at_ms) if enqueued_at_ms else 0
        if age_ms and age_ms > max_age_ms:
            logger.warning(
                "[ChatQueue] skipping stale feishu message event_id=%s partition=%s age_ms=%s max_age_ms=%s",
                metadata.get("event_id") or "none",
                partition or "none",
                age_ms,
                max_age_ms,
            )
            continue

        previous_index = newest_message_index_by_partition.get(partition)
        if previous_index is None:
            newest_message_index_by_partition[partition] = len(filtered)
            filtered.append(item)
            continue

        previous_item = filtered[previous_index]
        previous_enqueued_at_ms = 0
        if isinstance(previous_item, dict):
            previous_enqueued_at_ms = int(previous_item.get("enqueued_at_ms") or 0)
        if enqueued_at_ms >= previous_enqueued_at_ms:
            logger.warning(
                "[ChatQueue] superseding older feishu message older_event_id=%s newer_event_id=%s partition=%s",
                ((previous_item.get("metadata") or {}).get("event_id") if isinstance(previous_item, dict) else None) or "none",
                metadata.get("event_id") or "none",
                partition or "none",
            )
            filtered[previous_index] = item
        else:
            logger.warning(
                "[ChatQueue] dropping superseded feishu message older_event_id=%s newer_event_id=%s partition=%s",
                metadata.get("event_id") or "none",
                ((previous_item.get("metadata") or {}).get("event_id") if isinstance(previous_item, dict) else None) or "none",
                partition or "none",
            )

    return filtered


def _resolve_feishu_message_ingress_strategy(payload: dict[str, Any], context: dict[str, Any]) -> str:
    configured_from_env = os.getenv("HERMES_FEISHU_MESSAGE_INGRESS_STRATEGY")
    configured = configured_from_env if configured_from_env is not None else DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY
    effective = str(configured or "").strip().lower()
    configured_is_legacy_alias = effective in LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES
    if effective in LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES:
        effective = LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES[effective]
    if (
        (not str(configured_from_env or "").strip() or configured_is_legacy_alias)
        and _should_spawn_feishu_message_inline(payload, context)
    ):
        return "spawn_process_feishu_message_inline"
    if effective in SUPPORTED_FEISHU_MESSAGE_INGRESS_STRATEGIES:
        return effective
    return "inline_enqueue_spawn"


def _should_spawn_feishu_message_inline(payload: dict[str, Any], context: dict[str, Any]) -> bool:
    event_type = str((context or {}).get("event_type") or ((payload.get("header") or {}).get("event_type") or "")).strip()
    if event_type != "im.message.receive_v1":
        return False

    lane = str((context or {}).get("lane") or "").strip().lower()
    if lane and lane != "chat_light":
        return False

    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return False
    message = event.get("message") or {}
    if not isinstance(message, dict):
        return False

    message_type = str(message.get("message_type") or "").strip().lower()
    if message_type and message_type != "text":
        return False

    chat_type = str(message.get("chat_type") or event.get("chat_type") or "").strip().lower()
    if chat_type and chat_type != "p2p":
        return False

    raw_content = str(message.get("content") or "")
    if not raw_content:
        return False
    if len(raw_content) >= 1600:
        return False

    mentions = event.get("mentions") or message.get("mentions") or []
    if isinstance(mentions, list) and len(mentions) >= 3:
        return False

    return True


def _sample_light_p2p_feishu_message_payload() -> dict[str, Any]:
    return {
        "header": {
            "event_type": "im.message.receive_v1",
        },
        "event": {
            "message": {
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": "ping"}, ensure_ascii=False),
            }
        },
    }


def _process_chat_queue_impl(
    *,
    platform: str,
    partition: str,
    max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE,
    claim_token: str | None = None,
    worker_context: dict[str, Any] | None = None,
    runtime_prepared: bool = False,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_platform = str(platform or "").strip().lower()
    normalized_partition = str(partition or "").strip()
    if not normalized_platform or not normalized_partition:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": normalized_platform,
            "partition": normalized_partition,
            **_build_chat_worker_observability(worker_context),
        }

    context = dict(worker_context or {})
    if not runtime_prepared:
        inline_context = _build_inline_chat_worker_context()
        inline_context.update(context)
        context = inline_context
    normalized_warmup_wait_seconds = max(int(warmup_wait_seconds or 0), 0)
    normalized_linger_seconds = max(float(DEFAULT_CHAT_QUEUE_LINGER_SECONDS or 0.0), 0.0)
    warmup_payload = dict(warmup_metadata or {})
    is_waiting_warmup = normalized_warmup_wait_seconds > 0
    if is_waiting_warmup:
        started_at_ms = int(warmup_payload.get("started_at_ms") or int(time.time() * 1000))
        context.update(
            {
                "event_id": context.get("event_id") or warmup_payload.get("event_id"),
                "event_type": context.get("event_type") or warmup_payload.get("event_type"),
                "actor_id": context.get("actor_id") or warmup_payload.get("actor_id"),
                "chat_id": context.get("chat_id") or warmup_payload.get("chat_id"),
                "started_at_ms": started_at_ms,
                "warmup_status": "waiting",
                "warmup_event_id": _first_non_empty_str(warmup_payload.get("event_id")),
                "warmup_started_at": started_at_ms,
                "warmup_actor_id": _first_non_empty_str(warmup_payload.get("actor_id")),
                "warmup_chat_id": _first_non_empty_str(warmup_payload.get("chat_id")),
            }
        )

    warmup_snapshot = _pop_chat_queue_warmup_snapshot(normalized_partition)
    if warmup_snapshot:
        ready_at_ms = int(warmup_snapshot.get("ready_at_ms") or warmup_snapshot.get("updated_at_ms") or 0)
        now_ms = int(time.time() * 1000)
        context.update(
            {
                "warmup_status": str(warmup_snapshot.get("status") or "ready").strip().lower(),
                "warmup_event_id": _first_non_empty_str(warmup_snapshot.get("event_id")),
                "warmup_started_at": int(warmup_snapshot.get("started_at_ms") or 0) or None,
                "warmup_ready_at": ready_at_ms or None,
                "warmup_age_ms": max(0, now_ms - ready_at_ms) if ready_at_ms else None,
                "warmup_same_container": bool(
                    warmup_snapshot.get("worker_boot_id")
                    and warmup_snapshot.get("worker_boot_id") == context.get("worker_boot_id")
                ),
                "warmup_actor_id": _first_non_empty_str(warmup_snapshot.get("actor_id")),
                "warmup_chat_id": _first_non_empty_str(warmup_snapshot.get("chat_id")),
            }
        )

    if int(max_items or 0) <= 0:
        snapshot = _record_chat_queue_warmup_snapshot(
            normalized_partition,
            platform=normalized_platform,
            status="ready",
            metadata={
                "event_id": context.get("event_id"),
                "event_type": context.get("event_type"),
                "actor_id": context.get("actor_id"),
                "chat_id": context.get("chat_id"),
                "started_at_ms": context.get("started_at_ms"),
            },
            worker_context=context,
        )
        if snapshot:
            context.update(
                {
                    "warmup_status": str(snapshot.get("status") or "ready").strip().lower(),
                    "warmup_event_id": _first_non_empty_str(snapshot.get("event_id")),
                    "warmup_started_at": int(snapshot.get("started_at_ms") or 0) or None,
                    "warmup_ready_at": int(snapshot.get("ready_at_ms") or 0) or None,
                    "warmup_age_ms": 0,
                    "warmup_same_container": True,
                    "warmup_actor_id": _first_non_empty_str(snapshot.get("actor_id")),
                    "warmup_chat_id": _first_non_empty_str(snapshot.get("chat_id")),
                }
            )
        logger.warning(
            "[ChatQueue] warmup ready platform=%s partition=%s worker_boot_id=%s reused=%s enter_elapsed_ms=%s actor_id=%s chat_id=%s",
            normalized_platform or "unknown",
            normalized_partition or "none",
            context.get("worker_boot_id") or "inline",
            bool(context.get("container_reused", False)),
            context.get("enter_elapsed_ms"),
            context.get("warmup_actor_id") or "unknown",
            context.get("warmup_chat_id") or "unknown",
        )
        return {
            "status": "warmed",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "processed_count": 0,
            "results": [],
            "queue_depth": _safe_chat_queue_depth(),
            **_build_chat_worker_observability(context, batch_size=0),
        }

    claimed, claim_token = _claim_chat_partition(
        normalized_partition,
        platform=normalized_platform,
        claim_token=claim_token,
    )
    if not claimed:
        logger.warning(
            "[ChatQueue] worker skipped platform=%s partition=%s reason=already_claimed",
            normalized_platform or "unknown",
            normalized_partition or "none",
        )
        return {
            "status": "skipped",
            "reason": "already_claimed",
            "platform": normalized_platform,
            "partition": normalized_partition,
            **_build_chat_worker_observability(context),
        }

    logger.warning(
        "[ChatQueue] worker claimed platform=%s partition=%s worker_pool=%s worker_boot_id=%s reused=%s enter_elapsed_ms=%s feishu_runtime_preinit_ok=%s feishu_runtime_preinit_elapsed_ms=%s",
        normalized_platform or "unknown",
        normalized_partition or "none",
        context.get("worker_partition_key") or "shared",
        context.get("worker_boot_id") or "inline",
        bool(context.get("container_reused", False)),
        context.get("enter_elapsed_ms"),
        context.get("feishu_runtime_preinit_ok"),
        context.get("feishu_runtime_preinit_elapsed_ms"),
    )

    queue = _get_chat_queue()
    processed: list[dict[str, Any]] = []
    poll_timed_out = False
    try:
        first_poll = True
        while True:
            try:
                if first_poll:
                    try:
                        items = queue.get_many(
                            max(max_items, 1),
                            block=True,
                            timeout=float(normalized_warmup_wait_seconds or 2.0),
                            partition=normalized_partition,
                        )
                    except TypeError:
                        items = queue.get_many(
                            max(max_items, 1),
                            block=True,
                            partition=normalized_partition,
                        )
                    first_poll = False
                else:
                    if processed and normalized_linger_seconds > 0:
                        try:
                            items = queue.get_many(
                                max(max_items, 1),
                                block=True,
                                timeout=normalized_linger_seconds,
                                partition=normalized_partition,
                            )
                        except TypeError:
                            items = queue.get_many(
                                max(max_items, 1),
                                block=True,
                                partition=normalized_partition,
                            )
                    else:
                        items = queue.get_many(max(max_items, 1), block=False, partition=normalized_partition)
            except pyqueue.Empty:
                poll_timed_out = True
                if is_waiting_warmup and not processed:
                    context["warmup_status"] = "timeout"
                if processed and normalized_linger_seconds > 0:
                    logger.warning(
                        "[ChatQueue] worker linger timed out platform=%s partition=%s linger_seconds=%s processed_count=%s",
                        normalized_platform or "unknown",
                        normalized_partition or "none",
                        normalized_linger_seconds,
                        len(processed),
                    )
                else:
                    logger.warning(
                        "[ChatQueue] worker poll timed out platform=%s partition=%s warmup_wait_seconds=%s",
                        normalized_platform or "unknown",
                        normalized_partition or "none",
                        normalized_warmup_wait_seconds if is_waiting_warmup else 0,
                    )
                break
            if not items:
                if is_waiting_warmup and not processed:
                    context["warmup_status"] = "empty"
                logger.warning(
                    "[ChatQueue] worker found no items platform=%s partition=%s",
                    normalized_platform or "unknown",
                    normalized_partition or "none",
                )
                break
            if is_waiting_warmup and not processed:
                now_ms = int(time.time() * 1000)
                started_at_ms = int(context.get("warmup_started_at") or context.get("started_at_ms") or now_ms)
                context.update(
                    {
                        "warmup_status": "hit",
                        "warmup_ready_at": now_ms,
                        "warmup_age_ms": max(0, now_ms - started_at_ms),
                        "warmup_same_container": True,
                    }
                )
                logger.warning(
                    "[ChatQueue] warmup hit platform=%s partition=%s worker_boot_id=%s waited_ms=%s",
                    normalized_platform or "unknown",
                    normalized_partition or "none",
                    context.get("worker_boot_id") or "inline",
                    context.get("warmup_age_ms") if context.get("warmup_age_ms") is not None else "na",
                )
            processed.extend(
                asyncio.run(
                    _process_chat_queue_items_async(
                        items,
                        worker_context=context,
                        runtime_prepared=True,
                    )
                )
            )
    finally:
        _sync_modal_volume(commit=True)
        _release_chat_partition_claim(normalized_partition, claim_token=claim_token)

    logger.warning(
        "[ChatQueue] worker released platform=%s partition=%s processed_count=%s worker_boot_id=%s reused=%s",
        normalized_platform or "unknown",
        normalized_partition or "none",
        len(processed),
        context.get("worker_boot_id") or "inline",
        bool(context.get("container_reused", False)),
    )

    if is_waiting_warmup and not processed and poll_timed_out:
        return {
            "status": "warmup_timeout",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "processed_count": 0,
            "results": [],
            "queue_depth": _safe_chat_queue_depth(),
            **_build_chat_worker_observability(context, batch_size=0),
        }

    return {
        "status": "ok",
        "platform": normalized_platform,
        "partition": normalized_partition,
        "processed_count": len(processed),
        "results": processed,
        "queue_depth": _safe_chat_queue_depth(),
        **_build_chat_worker_observability(context, batch_size=len(processed)),
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


def _probe_provider_request_metadata_impl(
    *,
    provider_name: str,
    model_name: str = "",
    model: str = "",
    prompt: str = "只回复 ok",
    max_tokens: int = 64,
) -> dict[str, Any]:
    _prepare_runtime_environment()
    settings = RuntimeSettings.from_env()
    normalized_provider = str(provider_name or "").strip().lower()
    normalized_model = str(model_name or model or "").strip()
    if not normalized_provider:
        raise ValueError("provider_name is required")
    if not normalized_model:
        raise ValueError("model_name or model is required")

    runtime_binding = _resolve_provider_runtime_binding(normalized_provider)
    if runtime_binding is None:
        raise RuntimeError(f"Missing runtime binding for provider {normalized_provider}")

    api_key = str(runtime_binding.get("api_key") or "").strip()
    base_url = str(runtime_binding.get("base_url") or "").strip()
    resolved_provider = str(runtime_binding.get("provider") or normalized_provider).strip().lower() or normalized_provider

    from run_agent import AIAgent

    agent = AIAgent(
        model=normalized_model,
        api_key=api_key,
        base_url=base_url,
        provider=resolved_provider,
        max_iterations=4,
        max_tokens=max(int(max_tokens or 64), 16),
        quiet_mode=True,
        verbose_logging=False,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
        session_id=f"provider-probe:{resolved_provider}:{uuid.uuid4().hex[:10]}",
        trace_session_key=f"provider-probe:{resolved_provider}",
        trace_metadata={
            "channel_type": "provider_probe",
            "probe_provider": resolved_provider,
            "probe_model": normalized_model,
        },
        platform="probe",
        user_id="local",
    )
    started_at = time.time()
    result = agent.run_conversation(prompt, task_id=f"provider_probe:{resolved_provider}")
    elapsed_ms = int((time.time() - started_at) * 1000)
    return {
        "status": "ok" if not result.get("error") else "error",
        "provider": resolved_provider,
        "model": normalized_model,
        "base_url": base_url,
        "elapsed_ms": elapsed_ms,
        "completed": bool(result.get("completed", True)),
        "final_response": result.get("final_response"),
        "error": result.get("error"),
        "provider_usage": result.get("provider_usage") or {},
        "provider_usage_totals": result.get("provider_usage_totals") or {},
        "cloudflare_ai_gateway": "gateway.ai.cloudflare.com" in base_url.lower(),
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


def _validate_feishu_message_ingress_impl(
    *,
    message_text: str = "selftest ingress path",
    target_webhook_url: str = "",
    public_base_url: str = "",
    request_only: bool = False,
) -> dict[str, Any]:
    import httpx

    _prepare_runtime_environment()
    settings = RuntimeSettings.from_env()
    explicit_webhook_url = str(target_webhook_url or "").strip()
    explicit_public_base = _normalize_public_https_url(public_base_url)
    runtime_public_base = _normalize_public_https_url(
        os.getenv("HERMES_PUBLIC_BASE_URL") or os.getenv("PUBLIC_BASE_URL")
    )
    if explicit_webhook_url:
        webhook_url = explicit_webhook_url
    else:
        resolved_public_base = explicit_public_base or runtime_public_base
        webhook_url = (
            f"{resolved_public_base}/feishu/webhook"
            if resolved_public_base
            else "https://isuyee88--hermes-agent-web-app.modal.run/feishu/webhook"
        )
    verification_token = str(settings.feishu_verification_token or "").strip()
    encrypt_key = str(settings.feishu_encrypt_key or "").strip()
    event_id = f"evt_selftest_{uuid.uuid4().hex[:16]}"
    message_id = f"om_selftest_{uuid.uuid4().hex[:12]}"
    session_suffix = uuid.uuid4().hex[:12]
    payload = {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": event_id,
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": f"ou_selftest_ingress_{session_suffix}"},
                "sender_type": "user",
            },
            "message": {
                "message_id": message_id,
                "chat_id": f"oc_selftest_ingress_{session_suffix}",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": str(message_text or "selftest ingress path")}, ensure_ascii=False),
            },
        },
    }
    if verification_token:
        payload["header"]["token"] = verification_token

    body = json.dumps(payload, ensure_ascii=False)
    headers = {"Content-Type": "application/json"}
    if encrypt_key:
        timestamp = str(int(time.time()))
        nonce = "hermes-feishu-message-selftest"
        signature = hashlib.sha256(f"{timestamp}{nonce}{encrypt_key}{body}".encode("utf-8")).hexdigest()
        headers.update(
            {
                "x-lark-request-timestamp": timestamp,
                "x-lark-request-nonce": nonce,
                "x-lark-signature": signature,
            }
        )

    if request_only:
        return {
            "status": "prepared",
            "webhook_url": webhook_url,
            "headers": headers,
            "body": body,
            "event_id": event_id,
            "message_id": message_id,
            "verification_token_configured": bool(verification_token),
            "encrypt_key_configured": bool(encrypt_key),
            "message_ingress_strategy": {
                "configured": DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY,
                "effective": _resolve_feishu_message_ingress_strategy({}, {}),
            },
        }

    with httpx.Client(timeout=20) as client:
        response = client.post(webhook_url, content=body.encode("utf-8"), headers=headers)

    return {
        "status": "ok" if response.status_code == 200 else "error",
        "webhook_url": webhook_url,
        "status_code": response.status_code,
        "response": _safe_json_loads(response.text, response.text),
        "event_id": event_id,
        "message_id": message_id,
        "verification_token_configured": bool(verification_token),
        "encrypt_key_configured": bool(encrypt_key),
        "message_ingress_strategy": {
            "configured": DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY,
            "effective": _resolve_feishu_message_ingress_strategy({}, {}),
        },
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


async def _prewarm_chat_queue_async() -> None:
    global _CHAT_QUEUE_PREWARMED
    if _CHAT_QUEUE_PREWARMED:
        return
    queue = _get_chat_queue()
    try:
        if hasattr(queue, "len") and hasattr(queue.len, "aio"):
            await queue.len.aio()  # type: ignore[union-attr]
        else:
            await asyncio.to_thread(queue.len)
        _CHAT_QUEUE_PREWARMED = True
    except Exception as exc:
        logger.warning("Modal chat queue prewarm failed: %s", exc)


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


async def _sync_modal_volume_async(*, reload: bool = False, commit: bool = False) -> None:
    if MODAL_VOLUME is None:
        return
    try:
        if reload:
            reload_handle = getattr(MODAL_VOLUME, "reload", None)
            if reload_handle is not None and hasattr(reload_handle, "aio"):
                await reload_handle.aio()  # type: ignore[union-attr]
            else:
                await asyncio.to_thread(MODAL_VOLUME.reload)
        if commit:
            commit_handle = getattr(MODAL_VOLUME, "commit", None)
            if commit_handle is not None and hasattr(commit_handle, "aio"):
                await commit_handle.aio()  # type: ignore[union-attr]
            else:
                await asyncio.to_thread(MODAL_VOLUME.commit)
    except Exception as exc:
        logger.warning("Modal volume async sync failed (reload=%s commit=%s): %s", reload, commit, exc)


def _should_reload_modal_volume_for_claims(kind: str) -> bool:
    env_name = f"HERMES_MODAL_{str(kind or '').strip().upper()}_CLAIMS_RELOAD"
    return _is_truthy(os.getenv(env_name), default=False)


def _safe_cron_queue_depth() -> int | None:
    try:
        return int(_get_cron_queue().len())
    except Exception as exc:
        logger.warning("Unable to read Modal cron queue depth: %s", exc)
        return None


async def _safe_cron_queue_depth_async() -> int | None:
    queue = _get_cron_queue()
    try:
        if hasattr(queue, "len") and hasattr(queue.len, "aio"):
            return int(await queue.len.aio())  # type: ignore[union-attr]
        return int(await asyncio.to_thread(queue.len))
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
        return int(await asyncio.to_thread(queue.len))
    except Exception as exc:
        logger.warning("Unable to read Modal chat queue depth: %s", exc)
        return None


def _load_chat_queue_claims() -> dict[str, Any]:
    payload = _load_json_file(CHAT_QUEUE_CLAIMS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_chat_queue_claims(payload: dict[str, Any]) -> None:
    _atomic_json_write(CHAT_QUEUE_CLAIMS_PATH, payload)


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
        if not isinstance(existing, dict) or str(existing.get("bot_name") or "").strip() != normalized_bot_name:
            payload[normalized_app_id] = {
                "bot_name": normalized_bot_name,
                "updated_at": int(time.time()),
            }
            _save_feishu_bot_identity_cache(payload)
            should_commit = True

    if should_commit:
        await _sync_modal_volume_async(commit=True)


def _load_chat_queue_warmups() -> dict[str, Any]:
    payload = _load_json_file(CHAT_QUEUE_WARMUPS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_chat_queue_warmups(payload: dict[str, Any]) -> None:
    _atomic_json_write(CHAT_QUEUE_WARMUPS_PATH, payload)


def _prune_chat_queue_warmups(
    warmups: dict[str, Any],
    *,
    ttl_seconds: int = DEFAULT_CHAT_QUEUE_WARMUP_TTL_SECONDS,
) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    max_age_ms = max(int(ttl_seconds or 0), 1) * 1000
    pruned: dict[str, Any] = {}
    for partition_key, snapshot in warmups.items():
        if not isinstance(snapshot, dict):
            continue
        updated_at_ms = int(
            snapshot.get("ready_at_ms")
            or snapshot.get("updated_at_ms")
            or snapshot.get("started_at_ms")
            or 0
        )
        if updated_at_ms and now_ms - updated_at_ms < max_age_ms:
            pruned[partition_key] = snapshot
    return pruned


def _record_chat_queue_warmup_snapshot(
    partition_key: str,
    *,
    platform: str,
    status: str,
    metadata: dict[str, Any] | None = None,
    worker_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {}

    now_ms = int(time.time() * 1000)
    metadata_payload = dict(metadata or {})
    worker_payload = dict(worker_context or {})
    snapshot = {
        "platform": str(platform or "").strip().lower(),
        "partition": normalized,
        "status": str(status or "").strip().lower() or "ready",
        "event_id": _first_non_empty_str(metadata_payload.get("event_id")),
        "event_type": _first_non_empty_str(metadata_payload.get("event_type")),
        "actor_id": _first_non_empty_str(metadata_payload.get("actor_id")),
        "chat_id": _first_non_empty_str(metadata_payload.get("chat_id")),
        "started_at_ms": int(metadata_payload.get("started_at_ms") or now_ms),
        "ready_at_ms": now_ms,
        "updated_at_ms": now_ms,
        "worker_boot_id": _first_non_empty_str(worker_payload.get("worker_boot_id")),
        "worker_started_at": worker_payload.get("worker_started_at"),
        "enter_elapsed_ms": worker_payload.get("enter_elapsed_ms"),
        "runtime_prepare_elapsed_ms": worker_payload.get("runtime_prepare_elapsed_ms"),
    }
    with _CHAT_QUEUE_LOCK:
        warmups = _prune_chat_queue_warmups(_load_chat_queue_warmups())
        warmups[normalized] = snapshot
        _save_chat_queue_warmups(warmups)
        _sync_modal_volume(commit=True)
    return snapshot


def _pop_chat_queue_warmup_snapshot(partition_key: str) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {}

    with _CHAT_QUEUE_LOCK:
        warmups = _prune_chat_queue_warmups(_load_chat_queue_warmups())
        snapshot = warmups.pop(normalized, None)
        if not isinstance(snapshot, dict):
            return {}
        _save_chat_queue_warmups(warmups)
        _sync_modal_volume(commit=True)
    return dict(snapshot)


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
        status = str(claim.get("status") or "claimed").strip().lower()
        claim_ttl_seconds = (
            max(int(claim.get("cooldown_seconds") or 0), 1)
            if status == "scheduled"
            else ttl_seconds
        )
        if claimed_at and now - claimed_at < claim_ttl_seconds:
            pruned[partition_key] = claim
    return pruned


def _claim_is_recent(claim: Mapping[str, Any], *, max_age_seconds: int) -> bool:
    if max_age_seconds <= 0:
        return False
    claimed_at = int((claim or {}).get("claimed_at") or 0)
    if claimed_at <= 0:
        return False
    return int(time.time()) - claimed_at <= max_age_seconds


def _claim_age_seconds(claim: Mapping[str, Any]) -> int:
    claimed_at = int((claim or {}).get("claimed_at") or 0)
    if claimed_at <= 0:
        return 0
    return max(0, int(time.time()) - claimed_at)


def _claim_is_stale_for_takeover(
    claim: Mapping[str, Any],
    *,
    stale_after_seconds: int | None = None,
) -> bool:
    threshold = (
        DEFAULT_CHAT_QUEUE_STALE_CLAIM_TAKEOVER_SECONDS
        if stale_after_seconds is None
        else int(stale_after_seconds)
    )
    if threshold <= 0:
        return False
    return _claim_age_seconds(claim) >= threshold


def _peek_chat_partition_claim(
    partition_key: str,
    *,
    ttl_seconds: int = DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {}

    with _CHAT_QUEUE_LOCK:
        if _should_reload_modal_volume_for_claims("chat"):
            _sync_modal_volume(reload=True)
        claims = _prune_chat_queue_claims(_load_chat_queue_claims(), ttl_seconds=ttl_seconds)
        existing = claims.get(normalized) or {}
        return dict(existing) if isinstance(existing, dict) else {}


async def _peek_chat_partition_claim_async(
    partition_key: str,
    *,
    ttl_seconds: int = DEFAULT_CHAT_QUEUE_CLAIM_TTL_SECONDS,
    refresh_on_miss: bool = False,
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {}

    should_reload = _should_reload_modal_volume_for_claims("chat")
    if should_reload:
        await _sync_modal_volume_async(reload=True)

    with _CHAT_QUEUE_LOCK:
        claims = _prune_chat_queue_claims(_load_chat_queue_claims(), ttl_seconds=ttl_seconds)
        existing = claims.get(normalized) or {}
        claim = dict(existing) if isinstance(existing, dict) else {}

    if claim or not refresh_on_miss or should_reload:
        return claim

    # On shared-volume workers, a fresh reload on miss helps avoid spawning a
    # second container for a partition that is already actively claimed in
    # another container but hasn't been observed locally yet.
    await _sync_modal_volume_async(reload=True)
    with _CHAT_QUEUE_LOCK:
        claims = _prune_chat_queue_claims(_load_chat_queue_claims(), ttl_seconds=ttl_seconds)
        existing = claims.get(normalized) or {}
        return dict(existing) if isinstance(existing, dict) else {}


def _has_recent_chat_worker_spawn(partition_key: str, *, ttl_seconds: float = DEFAULT_FEISHU_RECENT_SPAWN_SKIP_SECONDS) -> bool:
    normalized = str(partition_key or "").strip()
    if not normalized or ttl_seconds <= 0:
        return False

    now = time.monotonic()
    with _RECENT_CHAT_WORKER_SPAWNS_LOCK:
        expired = [key for key, ts in _RECENT_CHAT_WORKER_SPAWNS.items() if now - ts > ttl_seconds]
        for key in expired:
            _RECENT_CHAT_WORKER_SPAWNS.pop(key, None)
        last_spawned_at = _RECENT_CHAT_WORKER_SPAWNS.get(normalized)
        return bool(last_spawned_at is not None and now - last_spawned_at <= ttl_seconds)


def _mark_recent_chat_worker_spawn(partition_key: str) -> None:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return
    with _RECENT_CHAT_WORKER_SPAWNS_LOCK:
        _RECENT_CHAT_WORKER_SPAWNS[normalized] = time.monotonic()


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
            existing_token = str(existing.get("claim_token") or "").strip()
            if existing_token and claim_token and existing_token == token:
                claims[normalized] = {
                    "claim_token": token,
                    "claimed_at": now,
                    "platform": platform,
                    "status": "claimed",
                }
                _save_chat_queue_claims(claims)
                _sync_modal_volume(commit=True)
                return True, token
            if not _claim_is_stale_for_takeover(existing):
                return False, existing_token
            logger.warning(
                "[ChatQueue] taking over stale claim partition=%s old_status=%s old_age_seconds=%s old_token=%s",
                normalized,
                str(existing.get("status") or "claimed").strip().lower() or "claimed",
                _claim_age_seconds(existing),
                existing_token or "none",
            )
        claims[normalized] = {
            "claim_token": token,
            "claimed_at": now,
            "platform": platform,
            "status": "claimed",
        }
        _save_chat_queue_claims(claims)
        _sync_modal_volume(commit=True)
    return True, token


def _schedule_chat_partition_worker(
    partition_key: str,
    *,
    platform: str,
    cooldown_seconds: int = DEFAULT_CHAT_QUEUE_SPAWN_COOLDOWN_SECONDS,
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": str(platform or "").strip().lower(),
            "partition": normalized,
        }

    normalized_platform = str(platform or "").strip().lower()
    now = int(time.time())
    token = f"{normalized_platform}:{normalized}:{now}:{uuid.uuid4().hex[:8]}"
    with _CHAT_QUEUE_LOCK:
        if _should_reload_modal_volume_for_claims("chat"):
            _sync_modal_volume(reload=True)
        claims = _prune_chat_queue_claims(_load_chat_queue_claims())
        existing = claims.get(normalized) or {}
        if existing:
            if not _claim_is_stale_for_takeover(existing):
                return {
                    "status": "skipped",
                    "reason": str(existing.get("status") or "already_scheduled"),
                    "platform": normalized_platform,
                    "partition": normalized,
                    "claim_token": str(existing.get("claim_token") or ""),
                }
            logger.warning(
                "[ChatQueue] replacing stale scheduled claim partition=%s old_status=%s old_age_seconds=%s old_token=%s",
                normalized,
                str(existing.get("status") or "claimed").strip().lower() or "claimed",
                _claim_age_seconds(existing),
                str(existing.get("claim_token") or "") or "none",
            )
        claims[normalized] = {
            "claim_token": token,
            "claimed_at": now,
            "platform": normalized_platform,
            "status": "scheduled",
            "cooldown_seconds": max(int(cooldown_seconds or 0), 1),
        }
        _save_chat_queue_claims(claims)
        _sync_modal_volume(commit=True)
    return {
        "status": "scheduled",
        "platform": normalized_platform,
        "partition": normalized,
        "claim_token": token,
    }


async def _schedule_chat_partition_worker_async(
    partition_key: str,
    *,
    platform: str,
    cooldown_seconds: int = DEFAULT_CHAT_QUEUE_SPAWN_COOLDOWN_SECONDS,
) -> dict[str, Any]:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": str(platform or "").strip().lower(),
            "partition": normalized,
        }

    normalized_platform = str(platform or "").strip().lower()
    now = int(time.time())
    token = f"{normalized_platform}:{normalized}:{now}:{uuid.uuid4().hex[:8]}"
    if _should_reload_modal_volume_for_claims("chat"):
        await _sync_modal_volume_async(reload=True)

    should_commit = False
    with _CHAT_QUEUE_LOCK:
        claims = _prune_chat_queue_claims(_load_chat_queue_claims())
        existing = claims.get(normalized) or {}
        if existing:
            if not _claim_is_stale_for_takeover(existing):
                return {
                    "status": "skipped",
                    "reason": str(existing.get("status") or "already_scheduled"),
                    "platform": normalized_platform,
                    "partition": normalized,
                    "claim_token": str(existing.get("claim_token") or ""),
                }
            logger.warning(
                "[ChatQueue] replacing stale scheduled claim partition=%s old_status=%s old_age_seconds=%s old_token=%s",
                normalized,
                str(existing.get("status") or "claimed").strip().lower() or "claimed",
                _claim_age_seconds(existing),
                str(existing.get("claim_token") or "") or "none",
            )
        claims[normalized] = {
            "claim_token": token,
            "claimed_at": now,
            "platform": normalized_platform,
            "status": "scheduled",
            "cooldown_seconds": max(int(cooldown_seconds or 0), 1),
        }
        _save_chat_queue_claims(claims)
        should_commit = True

    if should_commit:
        await _sync_modal_volume_async(commit=True)
    return {
        "status": "scheduled",
        "platform": normalized_platform,
        "partition": normalized,
        "claim_token": token,
    }


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


async def _release_chat_partition_claim_async(partition_key: str, *, claim_token: str | None = None) -> None:
    normalized = str(partition_key or "").strip()
    if not normalized:
        return

    if _should_reload_modal_volume_for_claims("chat"):
        await _sync_modal_volume_async(reload=True)

    should_commit = False
    with _CHAT_QUEUE_LOCK:
        claims = _prune_chat_queue_claims(_load_chat_queue_claims())
        existing = claims.get(normalized)
        if not existing:
            return
        if claim_token and existing.get("claim_token") != claim_token:
            return
        claims.pop(normalized, None)
        _save_chat_queue_claims(claims)
        should_commit = True

    if should_commit:
        await _sync_modal_volume_async(commit=True)


def _extract_feishu_queue_context(payload: dict[str, Any]) -> dict[str, str]:
    trace = _extract_feishu_trace_context(payload)
    event = payload.get("event") or {}
    chat_id = _first_non_empty_str(trace.get("chat_id"), _collect_feishu_chat_id(event))
    actor_id = _first_non_empty_str(
        trace.get("sender_open_id"),
        trace.get("sender_user_id"),
        _collect_feishu_actor_id(event),
        "unknown",
    )
    event_type = str(trace.get("event_type") or "").strip()
    lane = "inline_local"
    if event_type == "im.message.receive_v1":
        lane = _classify_feishu_chat_lane(payload)
    elif event_type:
        lane = "control"
    partition = f"feishu:{lane}:{chat_id or actor_id or trace.get('event_id') or 'unknown'}"
    return {
        "platform": "feishu",
        "partition": partition,
        "lane": lane,
        "chat_id": chat_id,
        "message_id": str(trace.get("message_id") or "").strip(),
        "event_id": str(trace.get("event_id") or "").strip(),
        "event_type": event_type,
        "actor_id": actor_id,
        "trace_token": str(trace.get("trace_token") or "").strip(),
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
    include_queue_depth: bool = True,
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
        "queue_depth": await _safe_chat_queue_depth_async() if include_queue_depth else None,
        **metadata,
    }


async def _spawn_feishu_event_handoff_async(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    handoff_requested_at_ms = int(time.time() * 1000)
    payload_for_spawn = _with_feishu_internal_meta(
        payload,
        handoff_requested_at_ms=handoff_requested_at_ms,
        webhook_event_id=context.get("event_id") or "",
        webhook_partition=context.get("partition") or "",
    )
    worker = globals().get("process_feishu_event")
    if worker is None or not hasattr(worker, "spawn"):
        return {
            "status": "unavailable",
            "reason": "process_feishu_event_spawn_unavailable",
            "handoff_wait_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "handoff_schedule_wait_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            **context,
        }

    spawn_handle = getattr(worker, "spawn")
    timeout_seconds = max(float(DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS or 0.0), 0.1)
    try:
        if hasattr(spawn_handle, "aio"):
            await asyncio.wait_for(
                spawn_handle.aio(payload=payload_for_spawn, warmup_only=False, warmup_context=None),  # type: ignore[union-attr]
                timeout=timeout_seconds,
            )
        else:
            await asyncio.wait_for(
                asyncio.to_thread(spawn_handle, payload=payload_for_spawn, warmup_only=False, warmup_context=None),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        event_id, event_type = _extract_feishu_event_metadata(payload)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.warning(
            "[Feishu] ingress handoff timed out event_type=%s event_id=%s timeout_ms=%s partition=%s",
            event_type or "unknown",
            event_id or "none",
            int(timeout_seconds * 1000),
            context.get("partition") or "none",
        )
        return {
            "status": "timeout",
            "reason": "process_feishu_event_spawn_timeout",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            **context,
        }
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "status": "scheduled",
        "reason": "process_feishu_event_spawned",
        "handoff_wait_elapsed_ms": elapsed_ms,
        "handoff_schedule_wait_elapsed_ms": elapsed_ms,
        **context,
    }


async def _spawn_feishu_message_inline_async(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    worker = globals().get("process_feishu_message_inline")
    if worker is None or not hasattr(worker, "spawn"):
        return {
            "status": "unavailable",
            "reason": "process_feishu_message_inline_spawn_unavailable",
            "handoff_wait_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "handoff_schedule_wait_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            **context,
        }

    spawn_handle = getattr(worker, "spawn")
    timeout_seconds = max(float(DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS or 0.0), 0.1)
    try:
        if hasattr(spawn_handle, "aio"):
            await asyncio.wait_for(spawn_handle.aio(payload=payload), timeout=timeout_seconds)  # type: ignore[union-attr]
        else:
            await asyncio.wait_for(asyncio.to_thread(spawn_handle, payload=payload), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "timeout",
            "reason": "process_feishu_message_inline_spawn_timeout",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            **context,
        }
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "status": "scheduled",
        "reason": "process_feishu_message_inline_spawned",
        "handoff_wait_elapsed_ms": elapsed_ms,
        "handoff_schedule_wait_elapsed_ms": elapsed_ms,
        **context,
    }


async def _spawn_feishu_background_exec_async(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    handoff_reason: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    payload_for_spawn = _with_feishu_internal_meta(
        payload,
        background_exec_requested_at_ms=int(time.time() * 1000),
        handoff_reason=str(handoff_reason or "provider_slow_path").strip() or "provider_slow_path",
        execution_mode="inline_to_background",
        webhook_event_id=context.get("event_id") or "",
        webhook_partition=context.get("partition") or "",
    )
    worker = globals().get("process_feishu_background_exec")
    if worker is None or not hasattr(worker, "spawn"):
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "unavailable",
            "reason": "process_feishu_background_exec_spawn_unavailable",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            "execution_mode": "inline_to_background",
            "handoff_reason": handoff_reason,
            **context,
        }

    spawn_handle = getattr(worker, "spawn")
    timeout_seconds = max(float(DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS or 0.0), 0.1)
    try:
        if hasattr(spawn_handle, "aio"):
            await asyncio.wait_for(
                spawn_handle.aio(payload=payload_for_spawn),  # type: ignore[union-attr]
                timeout=timeout_seconds,
            )
        else:
            await asyncio.wait_for(
                asyncio.to_thread(spawn_handle, payload=payload_for_spawn),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "timeout",
            "reason": "process_feishu_background_exec_spawn_timeout",
            "handoff_wait_elapsed_ms": elapsed_ms,
            "handoff_schedule_wait_elapsed_ms": elapsed_ms,
            "execution_mode": "inline_to_background",
            "handoff_reason": handoff_reason,
            **context,
        }
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "status": "scheduled",
        "reason": "process_feishu_background_exec_spawned",
        "handoff_wait_elapsed_ms": elapsed_ms,
        "handoff_schedule_wait_elapsed_ms": elapsed_ms,
        "execution_mode": "inline_to_background",
        "handoff_reason": handoff_reason,
        **context,
    }


async def _spawn_feishu_ingress_warmup_async(
    *,
    payload: dict[str, Any],
    warmup_context: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    warmup_requested_at_ms = int(time.time() * 1000)
    payload_for_spawn = _with_feishu_internal_meta(
        payload,
        handoff_requested_at_ms=warmup_requested_at_ms,
        webhook_event_id=warmup_context.get("event_id") or "",
        webhook_partition=warmup_context.get("partition") or "",
        warmup_only=True,
    )
    worker = globals().get("process_feishu_event")
    if worker is not None and hasattr(worker, "spawn"):
        spawn_handle = getattr(worker, "spawn")
        timeout_seconds = max(float(DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS or 0.0), 0.1)
        try:
            if hasattr(spawn_handle, "aio"):
                await asyncio.wait_for(
                    spawn_handle.aio(
                        payload=payload_for_spawn,
                        warmup_only=True,
                        warmup_context=warmup_context,
                    ),  # type: ignore[union-attr]
                    timeout=timeout_seconds,
                )
            else:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        spawn_handle,
                        payload=payload_for_spawn,
                        warmup_only=True,
                        warmup_context=warmup_context,
                    ),
                    timeout=timeout_seconds,
                )
        except asyncio.TimeoutError:
            event_id, event_type = _extract_feishu_event_metadata(payload)
            logger.warning(
                "[Feishu] ingress warmup timed out event_type=%s event_id=%s timeout_ms=%s partition=%s",
                event_type or "unknown",
                event_id or "none",
                int(timeout_seconds * 1000),
                warmup_context.get("partition") or "none",
            )
            fallback_result = await _spawn_chat_queue_warmup_async(
                platform="feishu",
                partition=str(warmup_context.get("partition") or "").strip(),
                reason="bot_p2p_chat_entered",
                metadata=warmup_context,
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            fallback_result["warmup_handoff_wait_elapsed_ms"] = elapsed_ms
            fallback_result["warmup_handoff_schedule_wait_elapsed_ms"] = elapsed_ms
            return fallback_result
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "status": "scheduled",
            "spawned": True,
            "mode": "feishu_ingress",
            "reason": "bot_p2p_chat_entered",
            "warmup_handoff_wait_elapsed_ms": elapsed_ms,
            "warmup_handoff_schedule_wait_elapsed_ms": elapsed_ms,
            "metadata": {
                key: value
                for key, value in dict(warmup_context or {}).items()
                if key in {"event_type", "event_id", "chat_id", "actor_id"}
            },
            "partition": warmup_context.get("partition") or "",
        }

    fallback_result = await _spawn_chat_queue_warmup_async(
        platform="feishu",
        partition=str(warmup_context.get("partition") or "").strip(),
        reason="bot_p2p_chat_entered",
        metadata=warmup_context,
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    fallback_result["warmup_handoff_wait_elapsed_ms"] = elapsed_ms
    fallback_result["warmup_handoff_schedule_wait_elapsed_ms"] = elapsed_ms
    return fallback_result


def _spawn_chat_queue_worker_sync(
    *,
    platform: str,
    partition: str,
    max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schedule_started_at = time.perf_counter()
    schedule_result = _schedule_chat_partition_worker(partition, platform=platform)
    schedule_claim_elapsed_ms = int((time.perf_counter() - schedule_started_at) * 1000)
    if schedule_result.get("status") != "scheduled":
        return {**schedule_result, "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms}

    claim_token = str(schedule_result.get("claim_token") or "").strip()
    spawn_kwargs = {
        "platform": platform,
        "partition": partition,
        "max_items": max_items,
        "claim_token": claim_token,
    }
    if int(warmup_wait_seconds or 0) > 0:
        spawn_kwargs["warmup_wait_seconds"] = int(warmup_wait_seconds)
        spawn_kwargs["warmup_metadata"] = dict(warmup_metadata or {})
    worker = _get_process_chat_queue_handle(partition)
    try:
        if worker is not None and hasattr(worker, "spawn"):
            spawn_started_at = time.perf_counter()
            getattr(worker, "spawn")(**spawn_kwargs)  # type: ignore[misc]
            spawn_rpc_elapsed_ms = int((time.perf_counter() - spawn_started_at) * 1000)
            return {
                **schedule_result,
                "spawned": True,
                "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms,
                "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
            }

        inline_started_at = time.perf_counter()
        _process_chat_queue_impl(
            platform=platform,
            partition=partition,
            max_items=max_items,
            claim_token=claim_token,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
        )
        spawn_rpc_elapsed_ms = int((time.perf_counter() - inline_started_at) * 1000)
        return {
            **schedule_result,
            "spawned": False,
            "mode": "inline",
            "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms,
            "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
        }
    except Exception:
        _release_chat_partition_claim(partition, claim_token=claim_token)
        raise


async def _spawn_chat_queue_worker_async(
    *,
    platform: str,
    partition: str,
    max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schedule_started_at = time.perf_counter()
    schedule_result = await _schedule_chat_partition_worker_async(partition, platform=platform)
    schedule_claim_elapsed_ms = int((time.perf_counter() - schedule_started_at) * 1000)
    if schedule_result.get("status") != "scheduled":
        return {**schedule_result, "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms}

    claim_token = str(schedule_result.get("claim_token") or "").strip()
    spawn_kwargs = {
        "platform": platform,
        "partition": partition,
        "max_items": max_items,
        "claim_token": claim_token,
    }
    if int(warmup_wait_seconds or 0) > 0:
        spawn_kwargs["warmup_wait_seconds"] = int(warmup_wait_seconds)
        spawn_kwargs["warmup_metadata"] = dict(warmup_metadata or {})
    worker = _get_process_chat_queue_handle(partition)
    try:
        if worker is not None and hasattr(worker, "spawn"):
            spawn_handle = getattr(worker, "spawn")
            spawn_started_at = time.perf_counter()
            if hasattr(spawn_handle, "aio"):
                await spawn_handle.aio(**spawn_kwargs)  # type: ignore[union-attr]
            else:
                spawn_handle(**spawn_kwargs)  # type: ignore[operator]
            spawn_rpc_elapsed_ms = int((time.perf_counter() - spawn_started_at) * 1000)
            return {
                **schedule_result,
                "spawned": True,
                "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms,
                "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
            }

        inline_started_at = time.perf_counter()
        _process_chat_queue_impl(
            platform=platform,
            partition=partition,
            max_items=max_items,
            claim_token=claim_token,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
        )
        spawn_rpc_elapsed_ms = int((time.perf_counter() - inline_started_at) * 1000)
        return {
            **schedule_result,
            "spawned": False,
            "mode": "inline",
            "schedule_claim_elapsed_ms": schedule_claim_elapsed_ms,
            "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
        }
    except Exception:
        await _release_chat_partition_claim_async(partition, claim_token=claim_token)
        raise


async def _spawn_chat_queue_worker_optimistic_async(
    *,
    platform: str,
    partition: str,
    max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_platform = str(platform or "").strip().lower()
    normalized_partition = str(partition or "").strip()
    if not normalized_partition:
        return {
            "status": "skipped",
            "reason": "missing_partition",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "schedule_claim_elapsed_ms": 0,
            "spawn_rpc_elapsed_ms": 0,
        }

    if _has_recent_chat_worker_spawn(normalized_partition):
        return {
            "status": "skipped",
            "reason": "recent_spawn_gate",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "spawned": False,
            "schedule_claim_elapsed_ms": 0,
            "spawn_rpc_elapsed_ms": 0,
        }

    existing_claim = await _peek_chat_partition_claim_async(normalized_partition, refresh_on_miss=True)
    if existing_claim and _claim_is_recent(existing_claim, max_age_seconds=DEFAULT_CHAT_QUEUE_ACTIVE_CLAIM_SKIP_SECONDS):
        return {
            "status": "skipped",
            "reason": f"active_{str(existing_claim.get('status') or 'claimed').strip().lower() or 'claimed'}",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "spawned": False,
            "claim_token": str(existing_claim.get("claim_token") or ""),
            "schedule_claim_elapsed_ms": 0,
            "spawn_rpc_elapsed_ms": 0,
        }

    spawn_kwargs = {
        "platform": normalized_platform,
        "partition": normalized_partition,
        "max_items": max_items,
        "claim_token": None,
    }
    if int(warmup_wait_seconds or 0) > 0:
        spawn_kwargs["warmup_wait_seconds"] = int(warmup_wait_seconds)
        spawn_kwargs["warmup_metadata"] = dict(warmup_metadata or {})
    worker = _get_process_chat_queue_handle(partition)
    if worker is not None and hasattr(worker, "spawn"):
        spawn_handle = getattr(worker, "spawn")
        spawn_started_at = time.perf_counter()
        if hasattr(spawn_handle, "aio"):
            await spawn_handle.aio(**spawn_kwargs)  # type: ignore[union-attr]
        else:
            spawn_handle(**spawn_kwargs)  # type: ignore[operator]
        spawn_rpc_elapsed_ms = int((time.perf_counter() - spawn_started_at) * 1000)
        _mark_recent_chat_worker_spawn(normalized_partition)
        return {
            "status": "scheduled",
            "reason": "optimistic_spawned",
            "platform": normalized_platform,
            "partition": normalized_partition,
            "spawned": True,
            "schedule_claim_elapsed_ms": 0,
            "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
        }

    inline_started_at = time.perf_counter()
    _process_chat_queue_impl(
        platform=normalized_platform,
        partition=normalized_partition,
        max_items=max_items,
        claim_token=None,
        warmup_wait_seconds=warmup_wait_seconds,
        warmup_metadata=warmup_metadata,
    )
    spawn_rpc_elapsed_ms = int((time.perf_counter() - inline_started_at) * 1000)
    return {
        "status": "scheduled",
        "reason": "optimistic_inline",
        "platform": normalized_platform,
        "partition": normalized_partition,
        "spawned": False,
        "mode": "inline",
        "schedule_claim_elapsed_ms": 0,
        "spawn_rpc_elapsed_ms": spawn_rpc_elapsed_ms,
    }


async def _spawn_chat_queue_worker_background_task(
    *,
    payload: dict[str, Any],
    platform: str,
    partition: str,
    max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> None:
    try:
        spawn_result = await _spawn_chat_queue_worker_async(
            platform=platform,
            partition=partition,
            max_items=max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
        )
        stage = "webhook.spawned" if spawn_result.get("status") == "scheduled" else "webhook.spawn_skipped"
        _append_feishu_trace(stage, payload, partition=partition, reason=spawn_result.get("reason") or "")
        event_id, event_type = _extract_feishu_event_metadata(payload)
        if spawn_result.get("status") == "scheduled":
            logger.warning(
                "[Feishu] webhook spawned chat worker event_type=%s event_id=%s partition=%s",
                event_type or "unknown",
                event_id or "none",
                partition,
            )
        else:
            logger.warning(
                "[Feishu] webhook skipped chat worker spawn event_type=%s event_id=%s partition=%s reason=%s",
                event_type or "unknown",
                event_id or "none",
                partition,
                spawn_result.get("reason") or "already_scheduled",
            )
    except Exception as exc:
        _append_feishu_trace("webhook.spawn_error", payload, partition=partition, error=str(exc))
        event_id, event_type = _extract_feishu_event_metadata(payload)
        logger.warning(
            "[Feishu] webhook spawn failed event_type=%s event_id=%s partition=%s error=%s",
            event_type or "unknown",
            event_id or "none",
            partition,
            exc,
            exc_info=True,
        )


def _schedule_chat_queue_worker_background(
    *,
    payload: dict[str, Any],
    platform: str,
    partition: str,
    max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE,
    warmup_wait_seconds: int = 0,
    warmup_metadata: dict[str, Any] | None = None,
) -> bool:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    loop.create_task(
        _spawn_chat_queue_worker_background_task(
            payload=payload,
            platform=platform,
            partition=partition,
            max_items=max_items,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
        )
    )
    return True


def _extract_feishu_warmup_context(payload: dict[str, Any]) -> dict[str, str]:
    context = _extract_feishu_queue_context(payload)
    context["lane"] = "chat_light"
    chat_or_actor = str(context.get("chat_id") or context.get("actor_id") or context.get("event_id") or "unknown").strip()
    context["partition"] = f"feishu:chat_light:{chat_or_actor or 'unknown'}"
    context["started_at_ms"] = str(int(time.time() * 1000))
    if not str(context.get("partition") or "").strip():
        actor_id = str(context.get("actor_id") or "").strip()
        event_id = str(context.get("event_id") or "").strip()
        context["partition"] = f"feishu:{actor_id or event_id or 'warmup'}"
    return context


def _warmup_chat_queue_worker_impl(
    *,
    platform: str,
    partition: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_context = _bootstrap_chat_queue_worker_context()
    payload: dict[str, Any] = {
        "status": "warming_inline",
        "platform": platform,
        "partition": partition,
        "reason": str(reason or "").strip() or "manual",
        "warmup_wait_seconds": DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS,
        **_build_chat_worker_observability(worker_context, batch_size=0),
    }
    if isinstance(metadata, dict) and metadata:
        payload["metadata"] = {
            key: value
            for key, value in metadata.items()
            if key in {"event_type", "event_id", "chat_id", "actor_id"}
        }
    return payload


async def _spawn_chat_queue_warmup_async(
    *,
    platform: str,
    partition: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_metadata = {
        key: value
        for key, value in dict(metadata or {}).items()
        if key in {"event_type", "event_id", "chat_id", "actor_id", "started_at_ms"}
    }
    scheduled = await _spawn_chat_queue_worker_async(
        platform=platform,
        partition=partition,
        max_items=1,
        warmup_wait_seconds=DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS,
        warmup_metadata=normalized_metadata,
    )
    if scheduled.get("status") != "error":
        scheduled["reason"] = reason
        scheduled["metadata"] = {
            key: value
            for key, value in normalized_metadata.items()
            if key in {"event_type", "event_id", "chat_id", "actor_id"}
        }
        return scheduled

    inline_result = _warmup_chat_queue_worker_impl(
        platform=platform,
        partition=partition,
        reason=reason,
        metadata=metadata,
    )
    inline_result["spawned"] = False
    inline_result["mode"] = "inline"
    return inline_result


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


def _run_feishu_registry_sync_cycle(
    *,
    force_refresh: bool = False,
    mirror_to_bitable: bool | None = None,
    ensure_schema: bool = True,
) -> dict[str, Any]:
    _prepare_runtime_environment()
    state = _load_feishu_sync_state()
    now = int(time.time())
    settings = RuntimeSettings.from_env()
    should_mirror = (
        mirror_to_bitable
        if mirror_to_bitable is not None
        else settings.feishu_model_registry_mirror_enabled
    )
    state.update(
        {
            "last_attempt_at": now,
            "sync_interval_seconds": settings.feishu_model_registry_sync_interval_seconds,
            "mirror_enabled": bool(should_mirror),
        }
    )
    try:
        from tools.feishu_api import (
            build_feishu_client,
            build_model_registry,
            ensure_model_registry_bitable_schema,
            mirror_model_registry_to_bitable,
            resolve_bitable_target,
        )

        if force_refresh:
            try:
                _refresh_free_model_routes(force=True)
            except Exception as exc:
                logger.warning("Failed refreshing provider routes before model registry sync: %s", exc)

        registry_payload = build_model_registry(force_refresh=force_refresh)
        result: dict[str, Any] = {
            "status": "ok",
            "entry_count": len(registry_payload.get("entries") or []),
            "registry": registry_payload,
            "mirrored": False,
        }
        state["last_registry_entry_count"] = result["entry_count"]

        if should_mirror:
            client = build_feishu_client()
            app_token, table_id = resolve_bitable_target({}, client)
            state["target"] = {
                "app_token": app_token,
                "app_token_masked": _mask_runtime_identifier(app_token),
                "table_id": table_id,
                "resolution_mode": "app_token" if settings.feishu_bitable_app_token else "wiki_token",
            }
            if ensure_schema:
                schema_result = ensure_model_registry_bitable_schema(
                    client,
                    app_token=app_token,
                    table_id=table_id or None,
                    table_name="Hermes Model Registry",
                    create_missing_table=True,
                    create_missing_fields=True,
                    create_missing_views=True,
                )
                result["bitable_schema"] = schema_result
                compact_schema = _compact_feishu_schema_status(schema_result)
                compact_schema["checked_at"] = now
                state["schema"] = compact_schema

            mirror_result = mirror_model_registry_to_bitable(
                client,
                registry_payload,
                app_token=app_token,
                table_id=table_id,
            )
            result["bitable_mirror"] = mirror_result
            result["mirrored"] = True
            state["last_success_at"] = now
            state["last_status"] = "ok"
            state["last_error"] = ""
            state["last_sync"] = {
                "status": "ok",
                "mirrored": True,
                "created": int(mirror_result.get("created") or 0),
                "updated": int(mirror_result.get("updated") or 0),
                "upserted": int(mirror_result.get("created") or 0) + int(mirror_result.get("updated") or 0),
                "hidden": int(mirror_result.get("hidden") or 0),
                "entry_count": result["entry_count"],
                "completed_at": now,
            }
        else:
            state["last_success_at"] = now
            state["last_status"] = "ok"
            state["last_error"] = ""
            state["last_sync"] = {
                "status": "ok",
                "mirrored": False,
                "entry_count": result["entry_count"],
                "completed_at": now,
            }

        state["next_due_at"] = now + settings.feishu_model_registry_sync_interval_seconds
        _save_feishu_sync_state(state)
        return result
    except Exception as exc:
        state["last_status"] = "error"
        state["last_error"] = str(exc)
        state["last_error_at"] = now
        state["next_due_at"] = now + min(settings.feishu_model_registry_sync_interval_seconds, 300)
        _save_feishu_sync_state(state)
        return {
            "status": "error",
            "error": str(exc),
        }


def _sync_feishu_model_registry_impl(*, force_refresh: bool = False, mirror_to_bitable: bool | None = None) -> dict[str, Any]:
    return _run_feishu_registry_sync_cycle(
        force_refresh=force_refresh,
        mirror_to_bitable=mirror_to_bitable,
        ensure_schema=True,
    )


def _prepare_feishu_model_registry_bitable_impl(
    *,
    app_token: str | None = None,
    table_id: str | None = None,
    table_name: str = "Hermes Model Registry",
    create_missing_table: bool = True,
    create_missing_fields: bool = True,
    create_missing_views: bool = True,
) -> dict[str, Any]:
    _prepare_runtime_environment()
    state = _load_feishu_sync_state()
    now = int(time.time())
    try:
        from tools.feishu_api import build_feishu_client, ensure_model_registry_bitable_schema, resolve_bitable_target

        client = build_feishu_client()
        resolved_app_token, resolved_table_id = resolve_bitable_target(
            {
                "app_token": app_token,
                "table_id": table_id,
            },
            client,
            require_table_id=False,
        )
        result = ensure_model_registry_bitable_schema(
            client,
            app_token=resolved_app_token,
            table_id=resolved_table_id or None,
            table_name=str(table_name or "Hermes Model Registry").strip() or "Hermes Model Registry",
            create_missing_table=create_missing_table,
            create_missing_fields=create_missing_fields,
            create_missing_views=create_missing_views,
        )
        state["target"] = {
            "app_token": resolved_app_token,
            "app_token_masked": _mask_runtime_identifier(resolved_app_token),
            "table_id": str(result.get("table_id") or resolved_table_id or ""),
            "resolution_mode": "app_token" if RuntimeSettings.from_env().feishu_bitable_app_token else "wiki_token",
        }
        compact_schema = _compact_feishu_schema_status(result)
        compact_schema["checked_at"] = now
        state["schema"] = compact_schema
        state["last_status"] = "ok"
        state["last_error"] = ""
        _save_feishu_sync_state(state)
        return result
    except Exception as exc:
        state["last_status"] = "error"
        state["last_error"] = str(exc)
        state["last_error_at"] = now
        _save_feishu_sync_state(state)
        return {
            "status": "error",
            "error": str(exc),
        }


def _build_feishu_sync_state_debug_state() -> dict[str, Any]:
    _prepare_runtime_environment()
    state = _load_feishu_sync_state()
    settings = RuntimeSettings.from_env()
    payload: dict[str, Any] = {
        "configured": bool(
            (settings.feishu_bitable_app_token or settings.feishu_bitable_wiki_token)
            and settings.feishu_bitable_table_id
        ),
        "mirror_enabled": settings.feishu_model_registry_mirror_enabled,
        "sync_interval_seconds": settings.feishu_model_registry_sync_interval_seconds,
        "state_file": str(FEISHU_SYNC_STATE_PATH),
        "last_attempt_at": state.get("last_attempt_at"),
        "last_success_at": state.get("last_success_at"),
        "last_status": state.get("last_status") or "",
        "last_error": state.get("last_error") or "",
        "next_due_at": state.get("next_due_at"),
        "last_registry_entry_count": state.get("last_registry_entry_count"),
        "target": state.get("target") or {},
        "schema": state.get("schema") or {},
        "last_sync": state.get("last_sync") or {},
    }
    try:
        from tools.feishu_api import build_feishu_client, resolve_bitable_target

        if payload["configured"]:
            client = build_feishu_client()
            app_token, table_id = resolve_bitable_target({}, client)
            payload["resolved_target"] = {
                "app_token": app_token,
                "app_token_masked": _mask_runtime_identifier(app_token),
                "table_id": table_id,
            }
    except Exception as exc:
        payload["resolved_target_error"] = str(exc)
    return payload


def _should_prepare_feishu_registry_schema_on_startup(state: dict[str, Any] | None = None) -> bool:
    payload = state if isinstance(state, dict) else {}
    schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
    checked_at = int(schema.get("checked_at") or 0)
    if not checked_at:
        return True
    if str(schema.get("status") or "").strip().lower() != "ok":
        return True
    if schema.get("missing_required_fields") or schema.get("missing_views"):
        return True
    return (int(time.time()) - checked_at) >= DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS


def _validate_feishu_native_delivery_impl(
    *,
    target_id: str = "",
    document_format: str = "md",
    caption_prefix: str = "Hermes Feishu native delivery test",
    keep_files: bool = False,
) -> dict[str, Any]:
    _prepare_runtime_environment()
    normalized_target_id = str(target_id or os.getenv("FEISHU_HOME_CHANNEL") or "").strip()
    if not normalized_target_id:
        return {
            "status": "error",
            "message": "Provide target_id or set FEISHU_HOME_CHANNEL before validating native Feishu delivery",
        }

    normalized_format = str(document_format or "md").strip().lower() or "md"
    if normalized_format not in {"md", "txt"}:
        return {
            "status": "error",
            "message": "document_format must be one of: md, txt",
        }

    async def _run() -> dict[str, Any]:
        runtime = await _get_feishu_gateway_runtime()
        adapter = runtime.adapter

        work_dir = DATA_ROOT / "feishu-native-delivery" / uuid.uuid4().hex
        work_dir.mkdir(parents=True, exist_ok=True)
        image_path = work_dir / "hermes-feishu-test.png"
        document_path = work_dir / f"hermes-feishu-test.{normalized_format}"

        image_path.write_bytes(base64.b64decode(_TINY_PNG_BASE64))
        if normalized_format == "txt":
            document_path.write_text(
                (
                    "Hermes Feishu native delivery test\n"
                    "=================================\n\n"
                    f"Target: {normalized_target_id}\n"
                    "This file was generated by validate_feishu_native_delivery.\n"
                ),
                encoding="utf-8",
            )
        else:
            document_path.write_text(
                (
                    "# Hermes Feishu native delivery test\n\n"
                    f"- Target: `{normalized_target_id}`\n"
                    "- Generated by `validate_feishu_native_delivery`\n"
                    "- Purpose: validate native image/document send from the deployed Modal runtime\n"
                ),
                encoding="utf-8",
            )

        image_result = await adapter.send_image_file(
            chat_id=normalized_target_id,
            image_path=str(image_path),
            caption=f"{caption_prefix} image",
        )
        document_result = await adapter.send_document(
            chat_id=normalized_target_id,
            file_path=str(document_path),
            caption=f"{caption_prefix} document",
        )
        payload = {
            "status": "ok" if image_result.success and document_result.success else "error",
            "target_id": normalized_target_id,
            "document_format": normalized_format,
            "image": {
                "success": bool(image_result.success),
                "message_id": image_result.message_id,
                "error": image_result.error,
                "path": str(image_path),
            },
            "document": {
                "success": bool(document_result.success),
                "message_id": document_result.message_id,
                "error": document_result.error,
                "path": str(document_path),
            },
        }
        if not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)
        return payload

    return asyncio.run(_run())


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
    jobs_per_worker = 0
    if modal is not None and worker_count > 0 and queue_depth:
        spawned_workers = min(int(queue_depth), max(worker_count, 0))
        jobs_per_worker = min(
            DEFAULT_CRON_QUEUE_MAX_JOBS_PER_WORKER,
            max(1, math.ceil(int(queue_depth) / max(spawned_workers, 1))),
        )
        for _ in range(spawned_workers):
            process_cron_queue.spawn(max_jobs=jobs_per_worker)  # type: ignore[name-defined]

    return {
        "status": "ok",
        "enqueue": enqueue_result,
        "spawned_workers": spawned_workers,
        "jobs_per_worker": jobs_per_worker,
        "queue_depth": _safe_cron_queue_depth(),
    }


def _feishu_model_registry_heartbeat_impl() -> dict[str, Any]:
    settings = RuntimeSettings.from_env()
    if not settings.feishu_model_registry_mirror_enabled:
        return {"status": "skipped", "reason": "mirror_disabled"}
    if not settings.feishu_bitable_table_id or not (
        settings.feishu_bitable_app_token or settings.feishu_bitable_wiki_token
    ):
        return {"status": "skipped", "reason": "bitable_not_configured"}
    sync_state = _load_feishu_sync_state()
    now = int(time.time())
    next_due_at = int(sync_state.get("next_due_at") or 0)
    if next_due_at and now < next_due_at:
        return {
            "status": "skipped",
            "reason": "not_due",
            "next_due_at": next_due_at,
            "seconds_until_due": next_due_at - now,
        }
    return _run_feishu_registry_sync_cycle(force_refresh=True, mirror_to_bitable=True, ensure_schema=True)


def _maintenance_heartbeat_impl(
    *,
    enqueue_limit: int = DEFAULT_CRON_QUEUE_BATCH_SIZE,
    worker_count: int = DEFAULT_CRON_QUEUE_WORKERS,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "cron": _cron_scheduler_tick_impl(
            enqueue_limit=enqueue_limit,
            worker_count=worker_count,
        ),
        "feishu_registry": _feishu_model_registry_heartbeat_impl(),
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

    async def _run_startup_tasks() -> None:
        _prepare_runtime_environment()
        await _prewarm_chat_queue_async()
        settings = RuntimeSettings.from_env()
        if not settings.telegram_bot_token or not settings.telegram_webhook_url:
            pass
        else:
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

        if not settings.feishu_model_registry_mirror_enabled:
            return
        if not settings.feishu_bitable_table_id or not (
            settings.feishu_bitable_app_token or settings.feishu_bitable_wiki_token
        ):
            return
        sync_state = _load_feishu_sync_state()
        if not _should_prepare_feishu_registry_schema_on_startup(sync_state):
            logger.info(
                "Feishu registry schema startup prepare skipped: checked_at=%s ttl=%ss",
                (sync_state.get("schema") or {}).get("checked_at"),
                DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS,
            )
            return
        try:
            result = await asyncio.to_thread(_prepare_feishu_model_registry_bitable_impl)
            logger.info(
                "Feishu registry schema startup prepare: status=%s table=%s created_table=%s missing_required=%s",
                result.get("status"),
                result.get("table_id"),
                result.get("created_table"),
                len(result.get("missing_required_fields") or []),
            )
        except Exception as exc:
            logger.warning("Feishu registry schema startup prepare failed: %s", exc)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        await _run_startup_tasks()
        yield

    app = FastAPI(title="Hermes Agent Modal Gateway", version="1.0.0", lifespan=_lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        _prepare_runtime_environment()
        runtime_config = _sync_runtime_config()
        settings = RuntimeSettings.from_env()
        gateway_import_ok = True
        gateway_import_error = None
        telegram_webhook = None
        memory_provider = _get_memory_provider_status()
        chat_queue_depth = await _safe_chat_queue_depth_async()
        cron_status = await asyncio.to_thread(_cron_status_impl, 5)
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
                "queue_depth": chat_queue_depth,
            },
            "cron": cron_status,
            "memory_provider": memory_provider,
            "model_routing": _build_model_routing_debug_state(force_refresh=False, allow_network=False),
            "feishu_sync": _build_feishu_sync_state_debug_state(),
            "runtime_config": runtime_config,
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

    @app.post("/internal/feishu/agent-exec")
    async def feishu_internal_agent_exec(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        settings = RuntimeSettings.from_env()
        if not _validate_bearer_token(authorization, settings.feishu_internal_bearer_token):
            raise HTTPException(status_code=401, detail="Unauthorized")
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object payload")
        return await _run_feishu_internal_agent_exec(payload)

    @app.post("/internal/feishu/agent-plan")
    async def feishu_internal_agent_plan(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        settings = RuntimeSettings.from_env()
        if not _validate_bearer_token(authorization, settings.feishu_internal_bearer_token):
            raise HTTPException(status_code=401, detail="Unauthorized")
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object payload")
        return await _run_feishu_internal_agent_plan(payload)

    @app.post("/internal/feishu/session-control")
    async def feishu_internal_session_control(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        settings = RuntimeSettings.from_env()
        if not _validate_bearer_token(authorization, settings.feishu_internal_bearer_token):
            raise HTTPException(status_code=401, detail="Unauthorized")
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object payload")
        return await _run_feishu_internal_control(payload)

    @app.get("/internal/feishu/result-file/{token}")
    async def feishu_internal_result_file(
        token: str,
        authorization: Optional[str] = Header(default=None),
    ) -> Any:
        settings = RuntimeSettings.from_env()
        if not _validate_bearer_token(authorization, settings.feishu_internal_bearer_token):
            raise HTTPException(status_code=401, detail="Unauthorized")
        payload = _lookup_feishu_internal_result_file(token)
        if payload is None:
            raise HTTPException(status_code=404, detail="Result file not found")
        return FileResponse(
            path=str(payload.get("path") or ""),
            filename=str(payload.get("filename") or "artifact.bin"),
            media_type=str(payload.get("content_type") or "application/octet-stream"),
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
        inline_fast_command = _extract_telegram_inline_fast_command(update)
        if inline_fast_command:
            logger.info(
                "Telegram webhook inline fast command: update_id=%s command=%s",
                update_id,
                inline_fast_command,
            )
            await _dispatch_telegram_update(update)
            return {
                "status": "accepted",
                "update_id": update_id,
                "mode": "inline_fast_command",
                "command": inline_fast_command,
            }
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

        await _spawn_chat_queue_worker_async(
            platform="telegram",
            partition=context["partition"],
            max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
        )

        return {"status": "accepted", "update_id": update_id}

    @app.post("/feishu/webhook")
    async def feishu_webhook(request: Request) -> Response:
        request_started_at = time.perf_counter()
        request_started_at_ms = int(time.time() * 1000)
        phase_timings: dict[str, int] = {}
        settings = RuntimeSettings.from_env()
        if not settings.feishu_app_id or not settings.feishu_app_secret:
            raise HTTPException(status_code=503, detail="Feishu app credentials are not configured")

        try:
            fast_response = await _try_handle_feishu_verification_fast(request, settings)
            if fast_response is not None:
                return fast_response
            parse_started_at = time.perf_counter()
            _adapter, payload = await _parse_feishu_webhook_request(request)
            _capture_phase_elapsed(phase_timings, "parse_verify_elapsed_ms", parse_started_at)
            if payload.get("type") == "url_verification":
                return JSONResponse({"challenge": payload.get("challenge", "")})

            _append_feishu_trace("webhook.accepted", payload)
            event_id, event_type = _extract_feishu_event_metadata(payload)
            dedupe_started_at = time.perf_counter()
            if event_id and not _mark_feishu_event_seen(event_id):
                _capture_phase_elapsed(phase_timings, "dedupe_elapsed_ms", dedupe_started_at)
                logger.warning(
                    "[Feishu] duplicate webhook event ignored event_type=%s event_id=%s",
                    event_type or "unknown",
                    event_id or "none",
                )
                return _build_feishu_webhook_ack_response(
                    payload,
                    {"code": 0, "msg": "duplicate"},
                    request_started_at=request_started_at,
                    ack_kind="duplicate",
                    reason="event_deduped",
                    phase_timings=phase_timings,
                )
            _capture_phase_elapsed(phase_timings, "dedupe_elapsed_ms", dedupe_started_at)
            trace_token = _extract_feishu_trace_token(payload)
            logger.warning(
                "[Feishu] webhook accepted event_type=%s event_id=%s trace_token=%s",
                event_type or "unknown",
                event_id or "none",
                trace_token or "none",
            )
            if _is_feishu_session_warmup_event(event_type):
                context_started_at = time.perf_counter()
                warmup_context = _extract_feishu_warmup_context(payload)
                _capture_phase_elapsed(phase_timings, "context_extract_elapsed_ms", context_started_at)
                warmup_result = await _spawn_feishu_ingress_warmup_async(
                    payload=payload,
                    warmup_context=warmup_context,
                )
                phase_timings["handoff_wait_elapsed_ms"] = int(warmup_result.get("warmup_handoff_wait_elapsed_ms") or 0)
                phase_timings["handoff_schedule_wait_elapsed_ms"] = int(
                    warmup_result.get("warmup_handoff_schedule_wait_elapsed_ms") or 0
                )
                _append_feishu_trace(
                    "webhook.session_warmup",
                    payload,
                    partition=warmup_context["partition"],
                    warmup_status=warmup_result.get("status"),
                    spawned=warmup_result.get("spawned"),
                )
                logger.warning(
                    "[Feishu] webhook session warmup event_type=%s event_id=%s partition=%s spawned=%s actor_id=%s",
                    event_type or "unknown",
                    event_id or "none",
                    warmup_context["partition"],
                    warmup_result.get("spawned"),
                    warmup_context.get("actor_id") or "unknown",
                )
                return _build_feishu_webhook_ack_response(
                    payload,
                    {"code": 0, "msg": "accepted"},
                    request_started_at=request_started_at,
                    ack_kind="warmup",
                    partition=warmup_context["partition"],
                    lane=warmup_context.get("lane"),
                    reason=str(warmup_result.get("status") or ""),
                    phase_timings=phase_timings,
                )
            if event_type == "im.message.message_read_v1":
                read_started_at = time.perf_counter()
                read_event = _extract_feishu_message_read_event_info(payload)
                _capture_phase_elapsed(phase_timings, "read_event_extract_elapsed_ms", read_started_at)
                _append_feishu_trace(
                    "webhook.message_read",
                    payload,
                    reader_open_id=read_event.get("reader_open_id") or "",
                    reader_user_id=read_event.get("reader_user_id") or "",
                    reader_union_id=read_event.get("reader_union_id") or "",
                    tenant_key=read_event.get("tenant_key") or "",
                    read_time=read_event.get("read_time") or 0,
                    message_id_list=read_event.get("message_id_list") or [],
                    message_count=read_event.get("message_count") or 0,
                )
                logger.warning(
                    "[Feishu] webhook message_read event_type=%s event_id=%s reader_open_id=%s message_count=%s read_time=%s",
                    event_type or "unknown",
                    event_id or "none",
                    read_event.get("reader_open_id") or "none",
                    read_event.get("message_count") or 0,
                    read_event.get("read_time") or 0,
                )
                return _build_feishu_webhook_ack_response(
                    payload,
                    {"code": 0, "msg": "accepted"},
                    request_started_at=request_started_at,
                    ack_kind="message_read",
                    reason="read_receipt_fast_path",
                    phase_timings=phase_timings,
                )
            inline_fast_command = None
            if event_type == "im.message.receive_v1":
                ack_reaction_mode = (
                    DEFAULT_FEISHU_ACK_REACTION_MODE
                    if DEFAULT_FEISHU_ACK_REACTION_MODE in SUPPORTED_FEISHU_ACK_REACTION_MODES
                    else "inline"
                )
                ack_reaction_message_id = str(
                    (((payload.get("event") or {}).get("message") or {}).get("message_id") or "")
                ).strip()
                if ack_reaction_mode != "off" and ack_reaction_message_id:
                    payload = _with_feishu_internal_meta(
                        payload,
                        ack_reaction_requested_at_ms=request_started_at_ms,
                        webhook_message_id=ack_reaction_message_id,
                    )
                    if ack_reaction_mode == "inline":
                        ack_reaction_started_at = time.perf_counter()
                        ack_reaction_task = asyncio.create_task(
                            _add_feishu_ack_reaction_inline_async(
                                payload=payload,
                                request_started_at_ms=request_started_at_ms,
                            )
                        )
                        ack_reaction_detached = False
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(ack_reaction_task),
                                timeout=max(DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS, 1) / 1000.0,
                            )
                        except asyncio.TimeoutError:
                            ack_reaction_detached = True
                            _append_feishu_trace(
                                "webhook.ack_reaction_detached",
                                payload,
                                inline_budget_ms=DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS,
                                reason="inline_budget_exceeded",
                                message_id=ack_reaction_message_id,
                            )
                            logger.warning(
                                "[Feishu] webhook ack reaction detached event_type=%s event_id=%s message_id=%s inline_budget_ms=%s",
                                event_type or "unknown",
                                event_id or "none",
                                ack_reaction_message_id or "none",
                                DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS,
                            )
                        _capture_phase_elapsed(
                            phase_timings,
                            "ack_reaction_inline_elapsed_ms",
                            ack_reaction_started_at,
                        )
                        phase_timings["ack_reaction_detached"] = 1 if ack_reaction_detached else 0
                        phase_timings["ack_reaction_schedule_elapsed_ms"] = 0
                    else:
                        ack_reaction_result = await _spawn_feishu_ack_reaction_async(
                            payload=payload,
                            request_started_at_ms=request_started_at_ms,
                        )
                        phase_timings["ack_reaction_schedule_elapsed_ms"] = int(
                            ack_reaction_result.get("schedule_elapsed_ms") or 0
                        )
                        if ack_reaction_result.get("status") == "scheduled":
                            _append_feishu_trace(
                                "webhook.ack_reaction_scheduled",
                                payload,
                                reason=str(ack_reaction_result.get("reason") or "process_feishu_ack_reaction_spawned"),
                                message_id=str(ack_reaction_result.get("message_id") or ""),
                            )
                            logger.warning(
                                "[Feishu] webhook ack reaction scheduled event_type=%s event_id=%s message_id=%s schedule_elapsed_ms=%s",
                                event_type or "unknown",
                                event_id or "none",
                                str(ack_reaction_result.get("message_id") or "") or "none",
                                int(ack_reaction_result.get("schedule_elapsed_ms") or 0),
                            )
                        elif ack_reaction_result.get("status") not in {"skipped", "scheduled"}:
                            logger.warning(
                                "[Feishu] webhook ack reaction schedule skipped event_type=%s event_id=%s reason=%s schedule_elapsed_ms=%s",
                                event_type or "unknown",
                                event_id or "none",
                                str(ack_reaction_result.get("reason") or ack_reaction_result.get("status") or "unknown"),
                                int(ack_reaction_result.get("schedule_elapsed_ms") or 0),
                            )
                inline_started_at = time.perf_counter()
                inline_fast_command = _extract_feishu_inline_fast_command(payload)
                _capture_phase_elapsed(phase_timings, "inline_fast_extract_elapsed_ms", inline_started_at)
            if inline_fast_command:
                _append_feishu_trace("webhook.inline_fast_command", payload, command=inline_fast_command)
                logger.warning(
                    "[Feishu] webhook inline fast command event_type=%s event_id=%s command=%s",
                    event_type or "unknown",
                    event_id or "none",
                    inline_fast_command,
                )
                await _dispatch_feishu_payload(payload, await_background_tasks=True)
                return _build_feishu_webhook_ack_response(
                    payload,
                    {"code": 0, "msg": "accepted"},
                    request_started_at=request_started_at,
                    ack_kind="inline_fast_command",
                    reason=inline_fast_command,
                    phase_timings=phase_timings,
                )
            if event_type == "application.bot.menu_v6":
                try:
                    if await _send_feishu_local_registry_menu_card(payload):
                        _append_feishu_trace("webhook.local_registry_menu", payload)
                        logger.warning(
                            "[Feishu] webhook local registry menu event_type=%s event_id=%s",
                            event_type or "unknown",
                            event_id or "none",
                        )
                        return _build_feishu_webhook_ack_response(
                            payload,
                            {"code": 0, "msg": "accepted"},
                            request_started_at=request_started_at,
                            ack_kind="local_registry_menu",
                            phase_timings=phase_timings,
                        )
                except Exception as exc:
                    logger.warning(
                        "[Feishu] local registry menu render failed event_id=%s error=%s",
                        event_id or "none",
                        exc,
                        exc_info=True,
                    )
            if event_type == "card.action.trigger":
                hermes_action = _extract_feishu_card_action_name(payload)
                if hermes_action == "registry_close_card":
                    try:
                        if await _close_feishu_card_from_payload(payload):
                            _append_feishu_trace("webhook.local_registry_close", payload)
                            logger.warning(
                                "[Feishu] webhook local registry close event_type=%s event_id=%s",
                                event_type or "unknown",
                                event_id or "none",
                            )
                            return _build_feishu_webhook_ack_response(
                                payload,
                                _build_feishu_card_action_ack_payload("已关闭"),
                                request_started_at=request_started_at,
                                ack_kind="local_registry_close",
                                phase_timings=phase_timings,
                            )
                    except Exception as exc:
                        logger.warning(
                            "[Feishu] local registry close failed event_id=%s error=%s",
                            event_id or "none",
                            exc,
                            exc_info=True,
                        )
                elif hermes_action:
                    try:
                        enqueue_result = await _enqueue_feishu_card_action_for_background(payload)
                        logger.warning(
                            "[Feishu] webhook queued card action event_type=%s event_id=%s partition=%s queue_depth=%s action=%s",
                            event_type or "unknown",
                            event_id or "none",
                            enqueue_result.get("partition") or "none",
                            enqueue_result.get("queue_depth"),
                            hermes_action,
                        )
                        return _build_feishu_webhook_ack_response(
                            payload,
                            _build_feishu_card_action_ack_payload(),
                            request_started_at=request_started_at,
                            ack_kind="card_action_queued",
                            partition=enqueue_result.get("partition"),
                            lane=enqueue_result.get("lane"),
                            queue_depth=enqueue_result.get("queue_depth"),
                            reason=hermes_action,
                            phase_timings=phase_timings,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[Feishu] card action queue handoff failed event_id=%s action=%s error=%s",
                            event_id or "none",
                            hermes_action,
                            exc,
                            exc_info=True,
                        )
            if _should_inline_feishu_control_event(event_type):
                _append_feishu_trace("webhook.inline_control", payload)
                logger.warning(
                    "[Feishu] webhook inline control event_type=%s event_id=%s",
                    event_type or "unknown",
                    event_id or "none",
                )
                await _dispatch_feishu_payload(
                    payload,
                    await_background_tasks=event_type != "card.action.trigger",
                )
                if event_type == "card.action.trigger":
                    return _build_feishu_webhook_ack_response(
                        payload,
                        _build_feishu_card_action_ack_payload(),
                        request_started_at=request_started_at,
                        ack_kind="inline_control",
                        phase_timings=phase_timings,
                    )
                return _build_feishu_webhook_ack_response(
                    payload,
                    {"code": 0, "msg": "accepted"},
                    request_started_at=request_started_at,
                    ack_kind="inline_control",
                    phase_timings=phase_timings,
                )
            context_started_at = time.perf_counter()
            context = _extract_feishu_queue_context(payload)
            _capture_phase_elapsed(phase_timings, "context_extract_elapsed_ms", context_started_at)
            ingress_strategy = ""
            if event_type == "im.message.receive_v1":
                ingress_strategy = _resolve_feishu_message_ingress_strategy(payload, context)
                handoff_result: dict[str, Any] | None = None
                if ingress_strategy == "spawn_process_feishu_event":
                    handoff_result = await _spawn_feishu_event_handoff_async(
                        payload=payload,
                        context=context,
                    )
                elif ingress_strategy == "spawn_process_feishu_message_inline":
                    handoff_result = await _spawn_feishu_message_inline_async(
                        payload=payload,
                        context=context,
                    )

                if handoff_result:
                    phase_timings["handoff_wait_elapsed_ms"] = int(handoff_result.get("handoff_wait_elapsed_ms") or 0)
                    phase_timings["handoff_schedule_wait_elapsed_ms"] = int(
                        handoff_result.get("handoff_schedule_wait_elapsed_ms") or 0
                    )
                if handoff_result and handoff_result.get("status") == "scheduled":
                    _append_feishu_trace(
                        "webhook.handoff_spawned",
                        payload,
                        partition=context["partition"],
                        lane=context.get("lane") or "",
                        reason=handoff_result.get("reason") or "",
                        ingress_strategy=ingress_strategy,
                    )
                    return _build_feishu_webhook_ack_response(
                        payload,
                        {"code": 0, "msg": "accepted"},
                        request_started_at=request_started_at,
                        ack_kind="queued_message",
                        partition=context["partition"],
                        lane=context.get("lane"),
                        reason=str(handoff_result.get("reason") or "process_feishu_event_spawned"),
                        ingress_strategy=ingress_strategy,
                        phase_timings=phase_timings,
                    )
            enqueue_result = await _enqueue_chat_event_async(
                platform="feishu",
                partition=context["partition"],
                payload=payload,
                metadata={**context, "ingress_strategy": ingress_strategy or "inline_enqueue_spawn"},
                include_queue_depth=False,
            )
            _append_feishu_trace(
                "queue.enqueue",
                payload,
                partition=context["partition"],
                lane=context.get("lane") or "",
                ingress_strategy=ingress_strategy or "inline_enqueue_spawn" if event_type == "im.message.receive_v1" else "",
                queue_depth=enqueue_result.get("queue_depth"),
            )
            logger.warning(
                "[Feishu] queue enqueue event_type=%s event_id=%s trace_token=%s partition=%s lane=%s queue_depth=%s",
                event_type or "unknown",
                event_id or "none",
                context.get("trace_token") or "none",
                context["partition"],
                context.get("lane") or "none",
                enqueue_result.get("queue_depth"),
            )
            worker_spawn_started_at = time.perf_counter()
            spawn_result = await _spawn_chat_queue_worker_optimistic_async(
                platform="feishu",
                partition=context["partition"],
                max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
            )
            worker_spawn_elapsed_ms = int((time.perf_counter() - worker_spawn_started_at) * 1000)
            schedule_claim_elapsed_ms = int(spawn_result.get("schedule_claim_elapsed_ms") or 0)
            spawn_rpc_elapsed_ms = int(spawn_result.get("spawn_rpc_elapsed_ms") or 0)
            phase_timings["worker_spawn_elapsed_ms"] = worker_spawn_elapsed_ms
            phase_timings["handoff_schedule_wait_elapsed_ms"] = worker_spawn_elapsed_ms
            phase_timings["handoff_wait_elapsed_ms"] = worker_spawn_elapsed_ms
            phase_timings["schedule_claim_elapsed_ms"] = schedule_claim_elapsed_ms
            phase_timings["spawn_rpc_elapsed_ms"] = spawn_rpc_elapsed_ms
            if spawn_result.get("status") == "scheduled":
                logger.warning(
                    "[Feishu] webhook spawned chat worker event_type=%s event_id=%s trace_token=%s partition=%s lane=%s spawn_elapsed_ms=%s schedule_claim_elapsed_ms=%s spawn_rpc_elapsed_ms=%s",
                    event_type or "unknown",
                    event_id or "none",
                    context.get("trace_token") or "none",
                    context["partition"],
                    context.get("lane") or "none",
                    worker_spawn_elapsed_ms,
                    schedule_claim_elapsed_ms,
                    spawn_rpc_elapsed_ms,
                )
            else:
                logger.warning(
                    "[Feishu] webhook skipped chat worker spawn event_type=%s event_id=%s trace_token=%s partition=%s lane=%s reason=%s spawn_elapsed_ms=%s schedule_claim_elapsed_ms=%s spawn_rpc_elapsed_ms=%s",
                    event_type or "unknown",
                    event_id or "none",
                    context.get("trace_token") or "none",
                    context["partition"],
                    context.get("lane") or "none",
                    spawn_result.get("reason") or "already_scheduled",
                    worker_spawn_elapsed_ms,
                    schedule_claim_elapsed_ms,
                    spawn_rpc_elapsed_ms,
                )
            return _build_feishu_webhook_ack_response(
                payload,
                {"code": 0, "msg": "accepted"},
                request_started_at=request_started_at,
                ack_kind="queued_message",
                partition=context["partition"],
                lane=context.get("lane"),
                queue_depth=enqueue_result.get("queue_depth"),
                reason=str(spawn_result.get("reason") or spawn_result.get("status") or "chat_worker_spawned"),
                ingress_strategy=ingress_strategy or "inline_enqueue_spawn" if event_type == "im.message.receive_v1" else "",
                phase_timings=phase_timings,
            )
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
        .apt_install("curl", "ca-certificates", "gnupg", "libsecret-1-0")
        .apt_install(
            "libnspr4",
            "libnss3",
            "libatk1.0-0",
            "libatk-bridge2.0-0",
            "libcups2",
            "libdrm2",
            "libdbus-1-3",
            "libxcb1",
            "libxkbcommon0",
            "libx11-6",
            "libx11-xcb1",
            "libxcomposite1",
            "libxdamage1",
            "libxext6",
            "libxfixes3",
            "libxrandr2",
            "libgbm1",
            "libasound2",
            "libatspi2.0-0",
            "libgtk-3-0",
            "fonts-liberation",
        )
        .run_commands(
            "install -d /etc/apt/keyrings",
            "curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg",
            'echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list',
            "apt-get update",
            "apt-get install -y nodejs",
            # Install Hermes's browser runtime up front so Modal stays closer
            # to the official browser-capable profile instead of relying on a
            # slow first-request npx bootstrap.
            "npm install -g agent-browser @askjo/camoufox-browser",
            "export PYTHONIOENCODING=UTF-8 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8; agent-browser install --with-deps >/tmp/agent-browser-install.log 2>&1 || agent-browser install >/tmp/agent-browser-install.log 2>&1 || true",
            "python -c \"from pathlib import Path; import uuid; mid = uuid.uuid4().hex; Path('/etc/machine-id').write_text(mid + '\\n', encoding='utf-8'); Path('/var/lib/dbus').mkdir(parents=True, exist_ok=True); Path('/var/lib/dbus/machine-id').write_text(mid + '\\n', encoding='utf-8')\"",
        )
        .pip_install_from_pyproject(
            "pyproject.toml",
            optional_dependencies=["modal", "messaging", "qq", "cron", "mcp", "pty", "feishu", "honcho", "homeassistant"],
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
    maintenance_heartbeat_enabled = _maintenance_heartbeat_is_enabled()
    maintenance_heartbeat_schedule = (
        modal.Period(minutes=DEFAULT_MAINTENANCE_HEARTBEAT_MINUTES)
        if maintenance_heartbeat_enabled
        else None
    )

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
        timeout=1800,
        memory=4096,
        cpu=2,
    )
    def probe_provider_request_metadata(
        provider_name: str,
        model_name: str = "",
        model: str = "",
        prompt: str = "只回复 ok",
        max_tokens: int = 64,
    ) -> dict[str, Any]:
        return _probe_provider_request_metadata_impl(
            provider_name=provider_name,
            model_name=model_name,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )

    def _run_modal_wait_pattern_probe(
        *,
        pattern: str,
        wait_seconds: float = 0.0,
        work_ms: int = 0,
        cpu: float,
        memory_mb: int,
        notes: str = "",
    ) -> dict[str, Any]:
        requested_wait_seconds = max(0.0, float(wait_seconds or 0.0))
        requested_work_ms = max(0, int(work_ms or 0))
        started_at = time.perf_counter()
        if requested_wait_seconds > 0:
            time.sleep(requested_wait_seconds)
        if requested_work_ms > 0:
            time.sleep(requested_work_ms / 1000.0)
        elapsed_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        return {
            "status": "ok",
            "pattern": pattern,
            "requested_wait_seconds": requested_wait_seconds,
            "requested_work_ms": requested_work_ms,
            "elapsed_ms": elapsed_ms,
            "resource_shape": {
                "cpu": cpu,
                "memory_mb": memory_mb,
            },
            "notes": notes,
            "completed_at_ms": int(time.time() * 1000),
        }

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=900,
        memory=DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB,
        cpu=DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU,
    )
    def benchmark_modal_wait_background_hold(
        wait_seconds: float = 15.0,
        work_ms: int = 250,
    ) -> dict[str, Any]:
        return _run_modal_wait_pattern_probe(
            pattern="background_hold",
            wait_seconds=wait_seconds,
            work_ms=work_ms,
            cpu=DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU,
            memory_mb=DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB,
            notes="Simulates the current Feishu background exec worker holding the request until a provider finishes.",
        )

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=900,
        memory=256,
        cpu=0.125,
    )
    def benchmark_modal_wait_light_hold(
        wait_seconds: float = 15.0,
        work_ms: int = 250,
    ) -> dict[str, Any]:
        return _run_modal_wait_pattern_probe(
            pattern="light_hold",
            wait_seconds=wait_seconds,
            work_ms=work_ms,
            cpu=0.125,
            memory_mb=256,
            notes="Simulates keeping the wait inside Modal but on a minimal worker shape.",
        )

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
        memory=256,
        cpu=0.125,
    )
    def benchmark_modal_wait_poll_tick(
        work_ms: int = 150,
    ) -> dict[str, Any]:
        return _run_modal_wait_pattern_probe(
            pattern="poll_tick",
            wait_seconds=0.0,
            work_ms=work_ms,
            cpu=0.125,
            memory_mb=256,
            notes="Simulates a single low-cost poll tick when waiting is handled outside Modal.",
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
            "official_parity": _build_modal_official_parity_state(),
            "feishu_sync": _build_feishu_sync_state_debug_state(),
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

    @app.cls(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=120,
        memory=DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB,
        cpu=DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU,
        scaledown_window=DEFAULT_FEISHU_ACK_REACTION_SCALEDOWN_WINDOW_SECONDS,
        enable_memory_snapshot=FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED,
    )
    class FeishuAckReactionWorker:
        @modal.enter(snap=FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED)
        def prepare_snapshot(self) -> None:
            if FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED:
                self._snapshot_context = _prepare_worker_snapshot_context(worker_name="Feishu ack reaction worker")

        @modal.enter()
        def enter(self) -> None:
            self._worker_context = _bootstrap_feishu_ack_reaction_worker_context(
                snapshot_context=dict(getattr(self, "_snapshot_context", {}) or {}),
            )
            self._processed_reactions = 0

        @modal.method()
        def add_reaction(
            self,
            payload: dict[str, Any],
            request_started_at_ms: int | None = None,
        ) -> dict[str, Any]:
            worker_context = dict(getattr(self, "_worker_context", {}) or {})
            worker_context["container_reused"] = bool(getattr(self, "_processed_reactions", 0))
            try:
                return _add_feishu_ack_reaction_from_payload(
                    payload,
                    request_started_at_ms=request_started_at_ms,
                    worker_context=worker_context,
                )
            finally:
                self._processed_reactions = int(getattr(self, "_processed_reactions", 0)) + 1

    process_feishu_ack_reaction = FeishuAckReactionWorker().add_reaction

    @app.cls(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=180,
        memory=DEFAULT_FEISHU_INGRESS_WORKER_MEMORY_MB,
        cpu=DEFAULT_FEISHU_INGRESS_WORKER_CPU,
        scaledown_window=DEFAULT_FEISHU_INGRESS_SCALEDOWN_WINDOW_SECONDS,
        enable_memory_snapshot=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED,
    )
    class FeishuIngressWorker:
        @modal.enter(snap=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED)
        def prepare_snapshot(self) -> None:
            if FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED:
                self._snapshot_context = _prepare_worker_snapshot_context(worker_name="Feishu ingress worker")

        @modal.enter()
        def enter(self) -> None:
            self._worker_context = _bootstrap_feishu_ingress_worker_context(
                snapshot_context=dict(getattr(self, "_snapshot_context", {}) or {}),
            )
            self._processed_handoffs = 0

        @modal.method()
        def handoff(
            self,
            payload: dict[str, Any],
            warmup_only: bool = False,
            warmup_context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            method_started_at = time.perf_counter()
            worker_context = dict(getattr(self, "_worker_context", {}) or {})
            worker_context["container_reused"] = bool(getattr(self, "_processed_handoffs", 0))
            ingress_meta = payload.get("_hermes_ingress") if isinstance(payload, dict) else {}
            ingress_meta = dict(ingress_meta) if isinstance(ingress_meta, dict) else {}
            handoff_requested_at_ms = int(ingress_meta.get("handoff_requested_at_ms") or 0)
            ingress_started_at_ms = int(time.time() * 1000)
            ingress_execution_delay_ms = (
                max(0, ingress_started_at_ms - handoff_requested_at_ms) if handoff_requested_at_ms else 0
            )

            try:
                if warmup_only:
                    normalized_warmup_context = {
                        key: value
                        for key, value in dict(warmup_context or {}).items()
                        if key in {"event_type", "event_id", "chat_id", "actor_id", "started_at_ms", "partition"}
                    }
                    partition = str(
                        normalized_warmup_context.get("partition")
                        or _extract_feishu_warmup_context(payload).get("partition")
                        or ""
                    ).strip()
                    warmup_result = _spawn_chat_queue_worker_sync(
                        platform="feishu",
                        partition=partition,
                        max_items=1,
                        warmup_wait_seconds=DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS,
                        warmup_metadata=normalized_warmup_context,
                    )
                    warmup_result["warmup_ingress_execution_delay_ms"] = ingress_execution_delay_ms
                    warmup_result["warmup_ingress_total_elapsed_ms"] = int((time.perf_counter() - method_started_at) * 1000)
                    return {
                        "status": str(warmup_result.get("status") or "scheduled"),
                        "spawned": bool(warmup_result.get("spawned", False)),
                        "mode": "feishu_ingress",
                        "partition": partition,
                        "reason": "bot_p2p_chat_entered",
                        "metadata": {
                            key: value
                            for key, value in normalized_warmup_context.items()
                            if key in {"event_type", "event_id", "chat_id", "actor_id"}
                        },
                        "warmup_ingress_execution_delay_ms": ingress_execution_delay_ms,
                        "warmup_ingress_total_elapsed_ms": int((time.perf_counter() - method_started_at) * 1000),
                        "ingress_worker_boot_id": worker_context.get("worker_boot_id"),
                        "ingress_container_reused": worker_context.get("container_reused"),
                    }

                context = _extract_feishu_queue_context(payload)
                context["ingress_strategy"] = "spawn_process_feishu_event"
                context["ingress_worker_boot_id"] = worker_context.get("worker_boot_id")
                context["ingress_container_reused"] = worker_context.get("container_reused")
                context["ingress_execution_delay_ms"] = ingress_execution_delay_ms
                enqueue_started_at = time.perf_counter()
                result = _enqueue_chat_event(
                    platform="feishu",
                    partition=context["partition"],
                    payload=payload,
                    metadata=context,
                )
                ingress_enqueue_elapsed_ms = int((time.perf_counter() - enqueue_started_at) * 1000)
                context["ingress_enqueue_elapsed_ms"] = ingress_enqueue_elapsed_ms
                _append_feishu_trace(
                    "ingress.queue_enqueue",
                    payload,
                    partition=context["partition"],
                    lane=context.get("lane") or "",
                    ingress_strategy=context.get("ingress_strategy") or "",
                    queue_depth=result.get("queue_depth"),
                    ingress_execution_delay_ms=ingress_execution_delay_ms,
                    ingress_enqueue_elapsed_ms=ingress_enqueue_elapsed_ms,
                    ingress_worker_boot_id=worker_context.get("worker_boot_id") or "",
                    ingress_container_reused=worker_context.get("container_reused"),
                )
                event_id, event_type = _extract_feishu_event_metadata(payload)
                logger.warning(
                    "[Feishu] queue enqueue event_type=%s event_id=%s partition=%s lane=%s queue_depth=%s ingress_worker_boot_id=%s ingress_container_reused=%s ingress_execution_delay_ms=%s ingress_enqueue_elapsed_ms=%s",
                    event_type or "unknown",
                    event_id or "none",
                    context["partition"],
                    context.get("lane") or "none",
                    result.get("queue_depth"),
                    worker_context.get("worker_boot_id") or "none",
                    worker_context.get("container_reused"),
                    ingress_execution_delay_ms,
                    ingress_enqueue_elapsed_ms,
                )
                chat_spawn_started_at = time.perf_counter()
                spawn_result = _spawn_chat_queue_worker_sync(
                    platform="feishu",
                    partition=context["partition"],
                    max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
                )
                chat_worker_spawn_elapsed_ms = int((time.perf_counter() - chat_spawn_started_at) * 1000)
                ingress_total_elapsed_ms = int((time.perf_counter() - method_started_at) * 1000)
                result["ingress_execution_delay_ms"] = ingress_execution_delay_ms
                result["ingress_enqueue_elapsed_ms"] = ingress_enqueue_elapsed_ms
                result["chat_worker_spawn_elapsed_ms"] = chat_worker_spawn_elapsed_ms
                result["ingress_total_elapsed_ms"] = ingress_total_elapsed_ms
                result["chat_worker_spawn_status"] = str(spawn_result.get("status") or "")
                _append_feishu_trace(
                    "ingress.handoff_done",
                    payload,
                    partition=context["partition"],
                    lane=context.get("lane") or "",
                    ingress_strategy=context.get("ingress_strategy") or "",
                    ingress_execution_delay_ms=ingress_execution_delay_ms,
                    ingress_enqueue_elapsed_ms=ingress_enqueue_elapsed_ms,
                    chat_worker_spawn_elapsed_ms=chat_worker_spawn_elapsed_ms,
                    ingress_total_elapsed_ms=ingress_total_elapsed_ms,
                    chat_worker_spawn_status=str(spawn_result.get("status") or ""),
                    ingress_worker_boot_id=worker_context.get("worker_boot_id") or "",
                    ingress_container_reused=worker_context.get("container_reused"),
                )
                logger.warning(
                    "[Feishu] ingress handoff done event_type=%s event_id=%s partition=%s ingress_execution_delay_ms=%s ingress_enqueue_elapsed_ms=%s chat_worker_spawn_elapsed_ms=%s ingress_total_elapsed_ms=%s chat_worker_spawn_status=%s ingress_worker_boot_id=%s ingress_container_reused=%s",
                    event_type or "unknown",
                    event_id or "none",
                    context["partition"],
                    ingress_execution_delay_ms,
                    ingress_enqueue_elapsed_ms,
                    chat_worker_spawn_elapsed_ms,
                    ingress_total_elapsed_ms,
                    str(spawn_result.get("status") or ""),
                    worker_context.get("worker_boot_id") or "none",
                    worker_context.get("container_reused"),
                )
                return result
            finally:
                self._processed_handoffs = int(getattr(self, "_processed_handoffs", 0)) + 1

    process_feishu_event = FeishuIngressWorker().handoff

    @app.cls(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=max(30, int(math.ceil(DEFAULT_FEISHU_INLINE_WORKER_HARD_BUDGET_MS / 1000)) + 5),
        memory=1024,
        cpu=0.5,
        scaledown_window=DEFAULT_FEISHU_INGRESS_SCALEDOWN_WINDOW_SECONDS,
        enable_memory_snapshot=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED,
    )
    class FeishuInlineMessageWorker:
        @modal.enter(snap=FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED)
        def prepare_snapshot(self) -> None:
            if FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED:
                self._snapshot_context = _prepare_worker_snapshot_context(worker_name="Feishu inline message worker")

        @modal.enter()
        def enter(self) -> None:
            self._worker_context = _bootstrap_chat_queue_worker_context(
                snapshot_context=dict(getattr(self, "_snapshot_context", {}) or {}),
            )
            self._processed_messages = 0

        @modal.method()
        def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
            context = _extract_feishu_queue_context(payload)
            context["ingress_strategy"] = "spawn_process_feishu_message_inline"
            worker_context = dict(getattr(self, "_worker_context", {}) or {})
            worker_context["container_reused"] = bool(getattr(self, "_processed_messages", 0))
            worker_context["execution_mode"] = "inline_to_background"
            worker_context["handoff_reason"] = "must_ai_default"
            started_at = time.perf_counter()
            _append_feishu_trace(
                "inline_message.start",
                payload,
                partition=context["partition"],
                lane=context.get("lane") or "",
                ingress_strategy=context.get("ingress_strategy") or "",
                worker_boot_id=worker_context.get("worker_boot_id") or "",
                container_reused=worker_context.get("container_reused"),
                execution_mode=worker_context.get("execution_mode"),
                handoff_reason=worker_context.get("handoff_reason"),
            )
            try:
                handoff_result = asyncio.run(
                    _spawn_feishu_background_exec_async(
                        payload=payload,
                        context=context,
                        handoff_reason=str(worker_context.get("handoff_reason") or "provider_slow_path"),
                    )
                )
                if str(handoff_result.get("status") or "") != "scheduled":
                    fallback_metadata = {
                        **context,
                        "ingress_strategy": "spawn_process_feishu_message_inline",
                        "execution_mode": "inline_to_queue_fallback",
                        "handoff_reason": str(handoff_result.get("reason") or worker_context.get("handoff_reason") or "background_exec_unavailable"),
                    }
                    _enqueue_chat_event(
                        platform="feishu",
                        partition=context["partition"],
                        payload=payload,
                        metadata=fallback_metadata,
                    )
                    spawn_result = _spawn_chat_queue_worker_sync(
                        platform="feishu",
                        partition=context["partition"],
                        max_items=DEFAULT_CHAT_QUEUE_BATCH_SIZE,
                    )
                    handoff_result = {
                        "status": "scheduled" if str(spawn_result.get("status") or "") == "scheduled" else "fallback",
                        "reason": "queue_fallback_spawned" if str(spawn_result.get("status") or "") == "scheduled" else str(spawn_result.get("reason") or "queue_fallback"),
                        "execution_mode": "inline_to_queue_fallback",
                        "handoff_reason": fallback_metadata["handoff_reason"],
                    }
            finally:
                self._processed_messages = int(getattr(self, "_processed_messages", 0)) + 1

            inline_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            status = str(handoff_result.get("status") or "error")
            reason = str(handoff_result.get("reason") or "")
            worker_context["execution_mode"] = str(
                handoff_result.get("execution_mode") or worker_context.get("execution_mode") or "inline_to_background"
            )
            worker_context["handoff_reason"] = str(
                handoff_result.get("handoff_reason") or worker_context.get("handoff_reason") or "provider_slow_path"
            )
            _append_feishu_trace(
                "inline_message.done",
                payload,
                partition=context["partition"],
                lane=context.get("lane") or "",
                ingress_strategy=context.get("ingress_strategy") or "",
                worker_boot_id=worker_context.get("worker_boot_id") or "",
                container_reused=worker_context.get("container_reused"),
                inline_elapsed_ms=inline_elapsed_ms,
                execution_mode=worker_context.get("execution_mode"),
                handoff_reason=worker_context.get("handoff_reason"),
                handoff_status=status,
                handoff_result_reason=reason,
            )
            logger.warning(
                "[Feishu] inline worker done event_id=%s partition=%s lane=%s worker_boot_id=%s reused=%s inline_elapsed_ms=%s execution_mode=%s handoff_reason=%s handoff_status=%s handoff_result_reason=%s",
                context.get("event_id") or "none",
                context["partition"],
                context.get("lane") or "none",
                worker_context.get("worker_boot_id") or "none",
                worker_context.get("container_reused"),
                inline_elapsed_ms,
                worker_context.get("execution_mode") or "unknown",
                worker_context.get("handoff_reason") or "unknown",
                status,
                reason or "none",
            )
            return {
                "status": status,
                "platform": "feishu",
                "partition": context["partition"],
                "lane": context.get("lane") or "",
                "event_id": context.get("event_id") or "",
                "worker_boot_id": worker_context.get("worker_boot_id") or "",
                "container_reused": worker_context.get("container_reused"),
                "inline_elapsed_ms": inline_elapsed_ms,
                "execution_mode": worker_context.get("execution_mode") or "inline_to_background",
                "handoff_reason": worker_context.get("handoff_reason") or "provider_slow_path",
                "handoff_result_reason": reason,
            }

    process_feishu_message_inline = FeishuInlineMessageWorker().handle

    @app.cls(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=max(15, int(math.ceil(DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS))),
        memory=DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB,
        cpu=DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU,
        scaledown_window=DEFAULT_FEISHU_BACKGROUND_EXEC_SCALEDOWN_WINDOW_SECONDS,
        enable_memory_snapshot=False,
    )
    class FeishuBackgroundExecWorker:
        @modal.enter()
        def enter(self) -> None:
            self._worker_context = _bootstrap_chat_queue_worker_context()
            self._processed_messages = 0

        @modal.method()
        def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
            context = _extract_feishu_queue_context(payload)
            ingress_meta = payload.get("_hermes_ingress") if isinstance(payload, dict) else {}
            ingress_meta = dict(ingress_meta) if isinstance(ingress_meta, dict) else {}
            worker_context = dict(getattr(self, "_worker_context", {}) or {})
            worker_context["container_reused"] = bool(getattr(self, "_processed_messages", 0))
            worker_context["execution_mode"] = str(ingress_meta.get("execution_mode") or "inline_to_background").strip()
            worker_context["handoff_reason"] = str(ingress_meta.get("handoff_reason") or "provider_slow_path").strip()
            started_at = time.perf_counter()
            _append_feishu_trace(
                "background_exec.start",
                payload,
                partition=context["partition"],
                lane=context.get("lane") or "",
                ingress_strategy=context.get("ingress_strategy") or "",
                worker_boot_id=worker_context.get("worker_boot_id") or "",
                container_reused=worker_context.get("container_reused"),
                execution_mode=worker_context.get("execution_mode"),
                handoff_reason=worker_context.get("handoff_reason"),
            )
            try:
                dispatch_result = asyncio.run(_dispatch_feishu_payload(payload, await_background_tasks=True))
            finally:
                self._processed_messages = int(getattr(self, "_processed_messages", 0)) + 1
            worker_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            phase_timings = _normalize_phase_timings((dispatch_result or {}).get("phase_timings"))
            background_send_elapsed_ms = int(phase_timings.get("background_tasks_elapsed_ms") or 0)
            _append_feishu_trace(
                "background_exec.done",
                payload,
                partition=context["partition"],
                lane=context.get("lane") or "",
                ingress_strategy=context.get("ingress_strategy") or "",
                worker_boot_id=worker_context.get("worker_boot_id") or "",
                container_reused=worker_context.get("container_reused"),
                execution_mode=worker_context.get("execution_mode"),
                handoff_reason=worker_context.get("handoff_reason"),
                worker_elapsed_ms=worker_elapsed_ms,
                background_send_elapsed_ms=background_send_elapsed_ms,
                phase_timings=phase_timings,
            )
            logger.warning(
                "[Feishu] background exec done event_id=%s partition=%s lane=%s worker_boot_id=%s reused=%s worker_elapsed_ms=%s execution_mode=%s handoff_reason=%s background_send_elapsed_ms=%s",
                context.get("event_id") or "none",
                context["partition"],
                context.get("lane") or "none",
                worker_context.get("worker_boot_id") or "none",
                worker_context.get("container_reused"),
                worker_elapsed_ms,
                worker_context.get("execution_mode") or "unknown",
                worker_context.get("handoff_reason") or "unknown",
                background_send_elapsed_ms,
            )
            return {
                "status": "ok",
                "platform": "feishu",
                "partition": context["partition"],
                "lane": context.get("lane") or "",
                "event_id": context.get("event_id") or "",
                "worker_boot_id": worker_context.get("worker_boot_id") or "",
                "container_reused": worker_context.get("container_reused"),
                "execution_mode": worker_context.get("execution_mode") or "inline_to_background",
                "handoff_reason": worker_context.get("handoff_reason") or "provider_slow_path",
                "worker_elapsed_ms": worker_elapsed_ms,
                "background_send_elapsed_ms": background_send_elapsed_ms,
                "phase_timings": phase_timings,
            }

    process_feishu_background_exec = FeishuBackgroundExecWorker().handle

    @app.cls(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=600,
        memory=DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB,
        cpu=DEFAULT_CHAT_QUEUE_WORKER_CPU,
        scaledown_window=DEFAULT_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS,
        enable_memory_snapshot=CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED,
    )
    class ChatQueueWorker:
        __annotations__ = {"partition_key": str}
        partition_key = modal.parameter(default="")

        @modal.enter(snap=CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED)
        def prepare_snapshot(self) -> None:
            if CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED:
                self._snapshot_context = _prepare_worker_snapshot_context(worker_name="Chat queue worker")

        @modal.enter()
        def enter(self) -> None:
            partition_key = str(getattr(self, "partition_key", "") or "").strip()
            self._worker_context = _bootstrap_chat_queue_worker_context(
                snapshot_context=dict(getattr(self, "_snapshot_context", {}) or {}),
                worker_partition_key=partition_key,
            )
            self._processed_batches = 0

        @modal.method()
        def process(
            self,
            platform: str,
            partition: str,
            max_items: int = DEFAULT_CHAT_QUEUE_BATCH_SIZE,
            claim_token: str | None = None,
            warmup_wait_seconds: int = 0,
            warmup_metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            worker_context = dict(getattr(self, "_worker_context", {}) or {})
            worker_context["container_reused"] = bool(getattr(self, "_processed_batches", 0))
            worker_context["worker_partition_key"] = str(
                worker_context.get("worker_partition_key") or getattr(self, "partition_key", "") or ""
            ).strip()
            try:
                return _process_chat_queue_impl(
                    platform=platform,
                    partition=partition,
                    max_items=max_items,
                    claim_token=claim_token,
                    worker_context=worker_context,
                    runtime_prepared=True,
                    warmup_wait_seconds=warmup_wait_seconds,
                    warmup_metadata=warmup_metadata,
                )
            finally:
                self._processed_batches = int(getattr(self, "_processed_batches", 0)) + 1

    process_chat_queue = ChatQueueWorker().process

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def debug_feishu_runtime() -> dict[str, Any]:
        _prepare_runtime_environment()
        settings = RuntimeSettings.from_env()
        configured_ingress_env = str(os.getenv("HERMES_FEISHU_MESSAGE_INGRESS_STRATEGY") or "").strip().lower()
        light_p2p_context = {"event_type": "im.message.receive_v1", "lane": "chat_light"}
        light_p2p_payload = _sample_light_p2p_feishu_message_payload()

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
            "message_ingress_strategy": {
                "configured": DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY,
                "configured_raw_env": configured_ingress_env,
                "effective": _resolve_feishu_message_ingress_strategy({}, {}),
                "effective_for_light_p2p": _resolve_feishu_message_ingress_strategy(
                    light_p2p_payload,
                    light_p2p_context,
                ),
                "legacy_aliases": dict(LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES),
                "supported": list(SUPPORTED_FEISHU_MESSAGE_INGRESS_STRATEGIES),
                "handoff_timeout_seconds": DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS,
            },
            "ack_reaction": {
                "mode": DEFAULT_FEISHU_ACK_REACTION_MODE,
                "supported_modes": list(SUPPORTED_FEISHU_ACK_REACTION_MODES),
                "inline_budget_ms": DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS,
                "request_timeout_seconds": DEFAULT_FEISHU_ACK_REACTION_REQUEST_TIMEOUT_SECONDS,
            },
            "execution_budgets": {
                "inline_first_response_budget_ms": DEFAULT_FEISHU_INLINE_FIRST_RESPONSE_BUDGET_MS,
                "provider_first_token_budget_ms": DEFAULT_FEISHU_PROVIDER_FIRST_TOKEN_BUDGET_MS,
                "inline_worker_hard_budget_ms": DEFAULT_FEISHU_INLINE_WORKER_HARD_BUDGET_MS,
            },
            "worker_shapes": {
                "feishu_inline_worker": {
                    "cpu": 0.5,
                    "memory_mb": 1024,
                },
                "feishu_background_exec_worker": {
                    "cpu": DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU,
                    "memory_mb": DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB,
                },
                "chat_queue_worker": {
                    "cpu": DEFAULT_CHAT_QUEUE_WORKER_CPU,
                    "memory_mb": DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB,
                },
            },
            "memory_snapshots": {
                "web_app_enabled": WEB_APP_MEMORY_SNAPSHOT_ENABLED,
                "feishu_ingress_enabled": FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED,
                "feishu_ack_reaction_enabled": FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED,
                "chat_queue_enabled": CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED,
            },
            "performance_experiment": _build_feishu_snapshot_profile_state(),
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
        timeout=60,
    )
    def debug_runtime_bootstrap() -> dict[str, Any]:
        return _build_runtime_bootstrap_debug_state()

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
    def debug_feishu_sync_state() -> dict[str, Any]:
        return _build_feishu_sync_state_debug_state()

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=180,
    )
    def sync_feishu_model_registry(
        force_refresh: bool = False,
        mirror_to_bitable: bool | None = None,
    ) -> dict[str, Any]:
        return _sync_feishu_model_registry_impl(
            force_refresh=force_refresh,
            mirror_to_bitable=mirror_to_bitable,
        )

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=180,
    )
    def validate_feishu_native_delivery(
        target_id: str = "",
        document_format: str = "md",
        caption_prefix: str = "Hermes Feishu native delivery test",
        keep_files: bool = False,
    ) -> dict[str, Any]:
        return _validate_feishu_native_delivery_impl(
            target_id=target_id,
            document_format=document_format,
            caption_prefix=caption_prefix,
            keep_files=keep_files,
        )

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=180,
    )
    def prepare_feishu_model_registry_bitable(
        app_token: str = "",
        table_id: str = "",
        table_name: str = "Hermes Model Registry",
        create_missing_table: bool = True,
        create_missing_fields: bool = True,
        create_missing_views: bool = True,
    ) -> dict[str, Any]:
        return _prepare_feishu_model_registry_bitable_impl(
            app_token=app_token or None,
            table_id=table_id or None,
            table_name=table_name,
            create_missing_table=create_missing_table,
            create_missing_fields=create_missing_fields,
            create_missing_views=create_missing_views,
        )

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
    def debug_modal_official_parity() -> dict[str, Any]:
        _prepare_runtime_environment()
        return _build_modal_official_parity_state()

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
    def debug_feishu_ingress_strategies(limit: int = 500) -> dict[str, Any]:
        _prepare_runtime_environment()
        return _build_feishu_ingress_strategy_debug_state(limit=limit)

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=60,
    )
    def debug_feishu_perf_summary(
        limit: int = 5000,
        since_seconds: int = 86400,
        event_type: str = "im.message.receive_v1",
        experiment_label: str = "",
        app_name_filter: str = "",
        snapshot_profile: str = "",
        include_duplicates: bool = False,
    ) -> dict[str, Any]:
        _prepare_runtime_environment()
        rows = _read_feishu_trace(limit=max(limit, 1))
        summary = _build_feishu_perf_summary_from_rows(
            rows,
            since_seconds=since_seconds,
            event_type=event_type,
            experiment_label=experiment_label,
            app_name_filter=app_name_filter,
            snapshot_profile=snapshot_profile,
            include_duplicates=include_duplicates,
        )
        summary["trace_file"] = str(FEISHU_TRACE_PATH)
        summary["read_limit"] = max(limit, 1)
        return summary

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
    def validate_feishu_message_ingress(
        message_text: str = "selftest ingress path",
        target_webhook_url: str = "",
        public_base_url: str = "",
        request_only: bool = False,
    ) -> dict[str, Any]:
        return _validate_feishu_message_ingress_impl(
            message_text=message_text,
            target_webhook_url=target_webhook_url,
            public_base_url=public_base_url,
            request_only=request_only,
        )

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
        timeout=30,
        schedule=maintenance_heartbeat_schedule,
    )
    def cron_scheduler_heartbeat() -> dict[str, Any]:
        return _maintenance_heartbeat_impl(
            enqueue_limit=DEFAULT_CRON_QUEUE_BATCH_SIZE,
            worker_count=DEFAULT_CRON_QUEUE_WORKERS,
        )

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=300,
    )
    def feishu_model_registry_heartbeat() -> dict[str, Any]:
        return _feishu_model_registry_heartbeat_impl()

    @app.function(
        image=image,
        volumes={"/data": volume},
        secrets=secrets,
        timeout=300,
        memory=2048,
        cpu=1,
        scaledown_window=DEFAULT_WEB_APP_SCALEDOWN_WINDOW_SECONDS,
        enable_memory_snapshot=WEB_APP_MEMORY_SNAPSHOT_ENABLED,
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
