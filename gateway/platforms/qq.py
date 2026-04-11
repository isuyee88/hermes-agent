"""QQ official bot webhook adapter.

Webhook-first integration for the QQ official bot platform. This adapter keeps
Hermes on the native gateway path:

- inbound QQ webhook payloads are normalized into ``MessageEvent``
- events are routed through ``GatewayRunner._handle_message``
- outbound replies use QQ official bot HTTP APIs via ``qq-botpy``
- callback URL validation follows the official Ed25519 signing flow

Current scope intentionally prioritizes the production path for Hermes on
Modal/self-hosted webhook deployments over long-lived QQ WebSocket sessions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    Ed25519PrivateKey = None  # type: ignore[assignment]
    CRYPTOGRAPHY_AVAILABLE = False

try:
    from botpy.api import BotAPI
    from botpy.http import BotHttp
    from botpy.message import C2CMessage, DirectMessage, GroupMessage, Message

    BOTPY_AVAILABLE = True
except ImportError:
    BotAPI = None  # type: ignore[assignment]
    BotHttp = None  # type: ignore[assignment]
    Message = None  # type: ignore[assignment]
    DirectMessage = None  # type: ignore[assignment]
    GroupMessage = None  # type: ignore[assignment]
    C2CMessage = None  # type: ignore[assignment]
    BOTPY_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_WEBHOOK_PATH = "/qq/webhook"
DEFAULT_MAX_BODY_BYTES = 1_048_576
DEFAULT_DEDUP_TTL_SECONDS = 300
DEFAULT_MAX_MESSAGE_LENGTH = 4000

SUPPORTED_QQ_EVENTS = frozenset(
    {
        "AT_MESSAGE_CREATE",
        "DIRECT_MESSAGE_CREATE",
        "GROUP_AT_MESSAGE_CREATE",
        "C2C_MESSAGE_CREATE",
    }
)

_MENTION_RE = re.compile(r"<@!?.+?>")


class QQWebhookError(Exception):
    """Structured webhook error with an HTTP status code."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def check_qq_requirements() -> bool:
    """Check whether QQ runtime dependencies are available."""
    return AIOHTTP_AVAILABLE and BOTPY_AVAILABLE and CRYPTOGRAPHY_AVAILABLE


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    value = str(value).strip()
    return [value] if value else []


def _normalize_payload_data(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _message_id_from_response(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        identifier = response.get("id") or response.get("message_id")
        return str(identifier) if identifier else None
    identifier = getattr(response, "id", None) or getattr(response, "message_id", None)
    return str(identifier) if identifier else None


def _timestamp_from_value(value: Any) -> datetime:
    if not value:
        return datetime.now()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    raw = str(value).strip()
    if not raw:
        return datetime.now()
    try:
        return datetime.fromtimestamp(float(raw))
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now()


class QQAdapter(BasePlatformAdapter):
    """QQ official bot adapter for webhook deployments."""

    MAX_MESSAGE_LENGTH = DEFAULT_MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQ)
        extra = config.extra or {}

        self._app_id = str(extra.get("app_id") or os.getenv("QQ_APP_ID", "")).strip()
        self._app_secret = str(extra.get("app_secret") or os.getenv("QQ_APP_SECRET", "")).strip()
        self._connection_mode = str(
            extra.get("connection_mode") or os.getenv("QQ_CONNECTION_MODE", "webhook")
        ).strip().lower() or "webhook"
        self._host = str(extra.get("webhook_host") or os.getenv("QQ_WEBHOOK_HOST", DEFAULT_HOST)).strip() or DEFAULT_HOST
        self._port = int(str(extra.get("webhook_port") or os.getenv("QQ_WEBHOOK_PORT", DEFAULT_PORT)).strip() or DEFAULT_PORT)
        self._webhook_path = str(
            extra.get("webhook_path") or os.getenv("QQ_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH)
        ).strip() or DEFAULT_WEBHOOK_PATH
        self._is_sandbox = str(extra.get("sandbox") or os.getenv("QQ_SANDBOX", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._verify_appid_header = str(
            extra.get("verify_appid_header", os.getenv("QQ_VERIFY_APPID_HEADER", "true"))
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._max_body_bytes = int(
            str(extra.get("max_body_bytes") or os.getenv("QQ_WEBHOOK_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)).strip()
            or DEFAULT_MAX_BODY_BYTES
        )
        self._dedup_ttl_seconds = int(
            str(extra.get("dedup_ttl_seconds") or os.getenv("QQ_DEDUP_TTL_SECONDS", DEFAULT_DEDUP_TTL_SECONDS)).strip()
            or DEFAULT_DEDUP_TTL_SECONDS
        )
        self._allowed_events = set(_coerce_list(extra.get("events") or os.getenv("QQ_EVENTS"))) or set(SUPPORTED_QQ_EVENTS)

        self._runner: Any = None
        self._site: Any = None
        self._http: Any = None
        self._api: Any = None
        self._seen_events: Dict[str, float] = {}
        self._reply_sequences: Dict[Tuple[str, str], int] = {}

    async def connect(self) -> bool:
        if not check_qq_requirements():
            message = (
                "QQ startup failed: aiohttp, cryptography, and qq-botpy are required. "
                "Run: pip install 'hermes-agent[messaging]'"
            )
            self._set_fatal_error("qq_missing_dependency", message, retryable=True)
            logger.warning("[%s] %s", self.name, message)
            return False
        if not self._app_id or not self._app_secret:
            message = "QQ startup failed: QQ_APP_ID and QQ_APP_SECRET are required"
            self._set_fatal_error("qq_missing_credentials", message, retryable=True)
            logger.warning("[%s] %s", self.name, message)
            return False
        if self._connection_mode != "webhook":
            message = "QQ adapter currently supports webhook mode only"
            self._set_fatal_error("qq_unsupported_mode", message, retryable=False)
            logger.warning("[%s] %s", self.name, message)
            return False
        if not AIOHTTP_AVAILABLE:
            message = "QQ webhook mode requires aiohttp"
            self._set_fatal_error("qq_missing_dependency", message, retryable=True)
            logger.warning("[%s] %s", self.name, message)
            return False

        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._webhook_path, self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._mark_connected()
        logger.info("[%s] Listening on %s:%s%s", self.name, self._host, self._port, self._webhook_path)
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        if self._http:
            await self._http.close()
            self._http = None
            self._api = None
        self._seen_events.clear()
        self._reply_sequences.clear()
        self._mark_disconnected()
        logger.info("[%s] Disconnected", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del metadata
        if not self._app_id or not self._app_secret:
            return SendResult(success=False, error="QQ credentials are not configured")

        api = self._get_api()
        route_kind, target_id = self._parse_chat_id(chat_id)
        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH) if content else [""]
        message_ids: list[str] = []

        try:
            for chunk in chunks:
                if route_kind == "group":
                    msg_seq = self._next_reply_sequence(chat_id, reply_to)
                    response = await api.post_group_message(
                        group_openid=target_id,
                        content=chunk,
                        msg_id=reply_to,
                        msg_seq=msg_seq,
                    )
                elif route_kind == "c2c":
                    msg_seq = self._next_reply_sequence(chat_id, reply_to)
                    response = await api.post_c2c_message(
                        openid=target_id,
                        content=chunk,
                        msg_id=reply_to,
                        msg_seq=msg_seq,
                    )
                elif route_kind == "guild":
                    response = await api.post_message(
                        channel_id=target_id,
                        content=chunk,
                        msg_id=reply_to,
                    )
                elif route_kind == "dm":
                    response = await api.post_dms(
                        guild_id=target_id,
                        content=chunk,
                        msg_id=reply_to,
                    )
                else:
                    return SendResult(success=False, error=f"Unsupported QQ chat target: {chat_id}")

                message_id = _message_id_from_response(response)
                if message_id:
                    message_ids.append(message_id)

            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids},
            )
        except Exception as exc:
            logger.warning("[%s] Failed sending message to %s: %s", self.name, chat_id, exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        del chat_id, metadata

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        route_kind, target_id = self._parse_chat_id(chat_id)
        label_map = {
            "guild": "channel",
            "dm": "dm",
            "group": "group",
            "c2c": "dm",
        }
        return {
            "name": f"QQ {route_kind} {target_id}",
            "type": label_map.get(route_kind, "dm"),
            "chat_id": chat_id,
        }

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        del request
        return web.json_response({"status": "ok", "platform": "qq"})

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        if request.content_length and request.content_length > self._max_body_bytes:
            raise web.HTTPRequestEntityTooLarge(max_size=self._max_body_bytes, actual_size=request.content_length)

        raw_body = await request.read()
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise web.HTTPBadRequest(text=f"Invalid JSON payload: {exc}") from exc

        try:
            result = await self.handle_webhook_payload(payload, headers=dict(request.headers))
        except QQWebhookError as exc:
            return web.json_response({"error": exc.message}, status=exc.status_code)
        return web.json_response(result)

    async def handle_webhook_payload(
        self,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        self._validate_appid_header(headers)

        op = int(payload.get("op", -1))
        if op == 13:
            return self._build_validation_response(payload)
        if op != 0:
            logger.debug("[%s] Ignoring QQ webhook with unsupported op=%s", self.name, op)
            return {"op": 12}

        event_id = str(payload.get("id") or "")
        if event_id and self._remember_event(event_id) is False:
            logger.info("[%s] Ignoring duplicate QQ event %s", self.name, event_id)
            return {"op": 12}

        event_type = str(payload.get("t") or "").strip()
        if event_type not in self._allowed_events:
            logger.info("[%s] Ignoring unsupported QQ event type %s", self.name, event_type)
            return {"op": 12}

        event = self._payload_to_message_event(payload, event_type)
        if event is None:
            logger.info("[%s] Ignoring empty QQ event type %s", self.name, event_type)
            return {"op": 12}

        await self.handle_message(event)
        logger.info("[%s] Accepted QQ event %s (%s)", self.name, event_id or "-", event_type)
        return {"op": 12}

    def _build_validation_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = _normalize_payload_data(payload.get("d"))
        plain_token = str(data.get("plain_token") or "").strip()
        event_ts = str(data.get("event_ts") or "").strip()
        if not plain_token or not event_ts:
            raise QQWebhookError(400, "QQ validation payload is missing plain_token or event_ts")

        seed = self._derive_seed_bytes(self._app_secret)
        private_key = Ed25519PrivateKey.from_private_bytes(seed)
        signature = private_key.sign(f"{event_ts}{plain_token}".encode("utf-8")).hex()
        return {"plain_token": plain_token, "signature": signature}

    def _validate_appid_header(self, headers: Dict[str, str]) -> None:
        if not self._verify_appid_header:
            return
        header_value = headers.get("x-bot-appid", "").strip()
        if not header_value:
            raise QQWebhookError(401, "Missing X-Bot-Appid header")
        if self._app_id and header_value != self._app_id:
            raise QQWebhookError(401, "Invalid X-Bot-Appid header")

    def _remember_event(self, event_id: str) -> bool:
        now = time.time()
        cutoff = now - self._dedup_ttl_seconds
        stale = [key for key, ts in self._seen_events.items() if ts < cutoff]
        for key in stale:
            self._seen_events.pop(key, None)
        if event_id in self._seen_events:
            return False
        self._seen_events[event_id] = now
        return True

    def _payload_to_message_event(self, payload: Dict[str, Any], event_type: str) -> Optional[MessageEvent]:
        api = self._get_api()
        event_id = str(payload.get("id") or "")
        data = _normalize_payload_data(payload.get("d"))

        if event_type == "AT_MESSAGE_CREATE":
            message = Message(api, event_id, data)
            source = self.build_source(
                chat_id=f"guild:{message.channel_id}",
                chat_name=f"QQ channel {message.channel_id}",
                chat_type="channel",
                user_id=getattr(message.author, "id", None),
                user_name=getattr(message.author, "username", None),
            )
            text = self._clean_content(message.content)
            reply_to = getattr(getattr(message, "message_reference", None), "message_id", None)
            return MessageEvent(
                text=text or "[QQ channel message]",
                message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
                source=source,
                raw_message=payload,
                message_id=str(message.id or event_id),
                reply_to_message_id=reply_to,
                timestamp=_timestamp_from_value(getattr(message, "timestamp", None)),
            )

        if event_type == "DIRECT_MESSAGE_CREATE":
            message = DirectMessage(api, event_id, data)
            source = self.build_source(
                chat_id=f"dm:{message.guild_id or message.channel_id}",
                chat_name="QQ direct message",
                chat_type="dm",
                user_id=getattr(message.author, "id", None),
                user_name=getattr(message.author, "username", None),
            )
            text = self._clean_content(message.content)
            reply_to = getattr(getattr(message, "message_reference", None), "message_id", None)
            return MessageEvent(
                text=text or "[QQ direct message]",
                message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
                source=source,
                raw_message=payload,
                message_id=str(message.id or event_id),
                reply_to_message_id=reply_to,
                timestamp=_timestamp_from_value(getattr(message, "timestamp", None)),
            )

        if event_type == "GROUP_AT_MESSAGE_CREATE":
            message = GroupMessage(api, event_id, data)
            source = self.build_source(
                chat_id=f"group:{message.group_openid}",
                chat_name=f"QQ group {message.group_openid}",
                chat_type="group",
                user_id=getattr(message.author, "member_openid", None),
                user_name="QQ group user",
            )
            text = self._clean_content(message.content)
            reply_to = getattr(getattr(message, "message_reference", None), "message_id", None)
            return MessageEvent(
                text=text or "[QQ group message]",
                message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
                source=source,
                raw_message=payload,
                message_id=str(message.id or event_id),
                reply_to_message_id=reply_to,
                timestamp=_timestamp_from_value(getattr(message, "timestamp", None)),
            )

        if event_type == "C2C_MESSAGE_CREATE":
            message = C2CMessage(api, event_id, data)
            source = self.build_source(
                chat_id=f"c2c:{message.author.user_openid}",
                chat_name="QQ c2c",
                chat_type="dm",
                user_id=getattr(message.author, "user_openid", None),
                user_name="QQ user",
            )
            text = self._clean_content(message.content)
            reply_to = getattr(getattr(message, "message_reference", None), "message_id", None)
            return MessageEvent(
                text=text or "[QQ c2c message]",
                message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
                source=source,
                raw_message=payload,
                message_id=str(message.id or event_id),
                reply_to_message_id=reply_to,
                timestamp=_timestamp_from_value(getattr(message, "timestamp", None)),
            )

        return None

    def _clean_content(self, content: Any) -> str:
        text = str(content or "")
        text = _MENTION_RE.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    def _get_api(self) -> Any:
        if self._api is not None:
            return self._api
        if not BOTPY_AVAILABLE:
            raise RuntimeError("qq-botpy is not installed")
        self._http = BotHttp(timeout=30, is_sandbox=self._is_sandbox, app_id=self._app_id, secret=self._app_secret)
        self._api = BotAPI(http=self._http)
        return self._api

    def _parse_chat_id(self, chat_id: str) -> Tuple[str, str]:
        if ":" not in str(chat_id):
            raise ValueError(f"QQ chat_id must include a route prefix, got: {chat_id}")
        route_kind, target_id = str(chat_id).split(":", 1)
        route_kind = route_kind.strip().lower()
        target_id = target_id.strip()
        if route_kind not in {"guild", "dm", "group", "c2c"} or not target_id:
            raise ValueError(f"Invalid QQ chat target: {chat_id}")
        return route_kind, target_id

    def _next_reply_sequence(self, chat_id: str, reply_to: Optional[str]) -> int:
        if not reply_to:
            return 1
        key = (str(chat_id), str(reply_to))
        next_value = self._reply_sequences.get(key, 0) + 1
        self._reply_sequences[key] = next_value
        return next_value

    @staticmethod
    def _derive_seed_bytes(secret: str) -> bytes:
        if not secret:
            raise QQWebhookError(500, "QQ app secret is not configured")
        seed = secret.encode("utf-8")
        while len(seed) < 32:
            seed += seed
        return seed[:32]
