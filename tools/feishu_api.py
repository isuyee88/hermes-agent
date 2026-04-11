from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import quote, urlparse

import httpx

logger = logging.getLogger(__name__)

_TOKEN_CACHE: Dict[str, Dict[str, Any]] = {}
_TOKEN_CACHE_LOCK = threading.Lock()
_DOC_URL_RE = re.compile(r"/docx/([A-Za-z0-9]+)")
_SHEET_URL_RE = re.compile(r"/sheets/([A-Za-z0-9]+)")
_TRUNCATE_RAW_CONTENT_AT = 12_000
_FEISHU_FILE_UPLOAD_TYPE = "stream"
_FEISHU_DOC_UPLOAD_TYPES = {
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "doc",
    ".xls": "xls",
    ".xlsx": "xls",
    ".ppt": "ppt",
    ".pptx": "ppt",
}
_FEISHU_OPUS_UPLOAD_EXTENSIONS = {".ogg", ".opus"}
_FEISHU_MEDIA_UPLOAD_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}
_DEFAULT_FEISHU_TOOL_CAPABILITIES = {
    "docs",
    "sheets",
    "bitable",
    "contacts",
    "messages",
    "files",
    "model_registry",
}
_SUPPORTED_MESSAGE_RESOURCE_TYPES = {"file", "image", "audio", "media"}
_MODEL_REGISTRY_FILE_NAME = "feishu_model_registry.json"


@dataclass(slots=True)
class FeishuOpenApiError(RuntimeError):
    message: str
    code: Any = None
    log_id: str = ""
    status_code: int | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.code not in (None, ""):
            parts.append(f"code={self.code}")
        if self.status_code:
            parts.append(f"http={self.status_code}")
        if self.log_id:
            parts.append(f"log_id={self.log_id}")
        return " ".join(parts)


def get_feishu_config() -> Tuple[str, str, str]:
    app_id = os.getenv("FEISHU_APP_ID", "").strip()
    app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    domain_name = os.getenv("FEISHU_DOMAIN", "feishu").strip().lower() or "feishu"
    return app_id, app_secret, domain_name


def get_feishu_base_url() -> str:
    _app_id, _app_secret, domain_name = get_feishu_config()
    return "https://open.larksuite.com" if domain_name == "lark" else "https://open.feishu.cn"


def get_modal_data_root() -> Path:
    return Path(os.getenv("HERMES_MODAL_DATA_DIR", "/data/hermes"))


def get_model_registry_path() -> Path:
    return get_modal_data_root() / _MODEL_REGISTRY_FILE_NAME


def get_routing_state_path() -> Path:
    return get_modal_data_root() / "free_model_routing.json"


def ensure_feishu_data_dir() -> None:
    get_modal_data_root().mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed reading JSON file %s", path, exc_info=True)
        return default


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def get_feishu_tool_capabilities() -> set[str]:
    raw = str(os.getenv("HERMES_FEISHU_TOOL_CAPABILITIES", "") or "").strip()
    if not raw:
        return set(_DEFAULT_FEISHU_TOOL_CAPABILITIES)
    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if "*" in values or "all" in values:
        return set(_DEFAULT_FEISHU_TOOL_CAPABILITIES)
    return values


def feishu_capability_enabled(capability: str | None) -> bool:
    if not capability:
        return True
    return capability.strip().lower() in get_feishu_tool_capabilities()


def check_feishu_available(capability: str | None = None) -> bool:
    app_id, app_secret, _domain_name = get_feishu_config()
    return bool(app_id and app_secret and feishu_capability_enabled(capability))


def make_capability_check(capability: str):
    return lambda: check_feishu_available(capability)


def extract_document_id(document_id_or_url: str) -> str:
    raw = str(document_id_or_url or "").strip()
    if not raw:
        raise ValueError("document_id_or_url is required")
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        match = _DOC_URL_RE.search(parsed.path)
        if match:
            return match.group(1)
    return raw


def extract_spreadsheet_token(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("spreadsheet_token_or_url is required")
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        match = _SHEET_URL_RE.search(parsed.path)
        if match:
            return match.group(1)
    return raw


def truncate_text(text: str, *, limit: int = _TRUNCATE_RAW_CONTENT_AT) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def normalize_document_summary(document_id: str, info: Dict[str, Any], raw_content: str | None = None) -> Dict[str, Any]:
    document = info.get("document") if isinstance(info.get("document"), dict) else info
    result: Dict[str, Any] = {
        "success": True,
        "document_id": document_id,
        "title": document.get("title", ""),
        "revision_id": document.get("revision_id"),
        "url": document.get("url"),
        "owner_id": document.get("owner_id"),
    }
    if raw_content is not None:
        truncated_content, was_truncated = truncate_text(raw_content)
        result["raw_content"] = truncated_content
        result["raw_content_truncated"] = was_truncated
    return result


def normalize_user_profile(user: Dict[str, Any], *, resolved_via: str) -> Dict[str, Any]:
    return {
        "resolved_via": resolved_via,
        "user_id": user.get("user_id"),
        "open_id": user.get("open_id"),
        "union_id": user.get("union_id"),
        "name": user.get("name") or user.get("display_name") or user.get("en_name"),
        "email": user.get("enterprise_email") or user.get("email"),
        "mobile": user.get("mobile"),
        "job_title": user.get("job_title"),
        "department_ids": user.get("department_ids") or [],
        "status": user.get("status") or {},
    }


def detect_upload_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _FEISHU_OPUS_UPLOAD_EXTENSIONS:
        return "opus"
    if ext in _FEISHU_MEDIA_UPLOAD_EXTENSIONS:
        return "mp4"
    if ext in _FEISHU_DOC_UPLOAD_TYPES:
        return _FEISHU_DOC_UPLOAD_TYPES[ext]
    return _FEISHU_FILE_UPLOAD_TYPE


def coerce_local_file_path(value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("file_path is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Local file not found: {path}")
    return path


def extract_filename_from_headers(headers: httpx.Headers, default_name: str) -> str:
    disposition = str(headers.get("content-disposition", "") or "")
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        return match.group(1)
    return default_name


def build_plain_post_payload(text: str, *, title: str | None = None) -> str:
    return json.dumps(
        {
            "zh_cn": {
                "title": title or "Hermes",
                "content": [[{"tag": "text", "text": text}]],
            }
        },
        ensure_ascii=False,
    )


def markdown_to_doc_blocks(markdown_text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for raw_line in str(markdown_text or "").splitlines():
        line = raw_line.rstrip()
        blocks.append(
            {
                "block_type": 2,
                "paragraph": {
                    "elements": [
                        {
                            "text_run": {"content": line},
                            "type": "text_run",
                        }
                    ]
                },
            }
        )
    return blocks or [
        {
            "block_type": 2,
            "paragraph": {"elements": [{"text_run": {"content": str(markdown_text or "")}, "type": "text_run"}]},
        }
    ]


class FeishuOpenApiClient:
    def __init__(
        self,
        *,
        app_id: str | None = None,
        app_secret: str | None = None,
        domain_name: str | None = None,
        timeout: float = 25.0,
    ) -> None:
        self.app_id = str(app_id or os.getenv("FEISHU_APP_ID") or "").strip()
        self.app_secret = str(app_secret or os.getenv("FEISHU_APP_SECRET") or "").strip()
        self.domain_name = str(domain_name or os.getenv("FEISHU_DOMAIN") or "feishu").strip().lower() or "feishu"
        self.timeout = timeout
        self.base_url = "https://open.larksuite.com" if self.domain_name == "lark" else "https://open.feishu.cn"
        if not self.app_id or not self.app_secret:
            raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET must both be configured")

    def get_tenant_access_token(self, *, force_refresh: bool = False) -> str:
        cache_key = f"{self.base_url}:{self.app_id}"
        now = time.time()
        with _TOKEN_CACHE_LOCK:
            cached = _TOKEN_CACHE.get(cache_key, {})
            if (
                not force_refresh
                and cached.get("token")
                and now < float(cached.get("expires_at") or 0)
            ):
                return str(cached["token"])

        response = httpx.post(
            f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code", 0) != 0:
            raise FeishuOpenApiError(
                "Feishu auth failed",
                code=payload.get("code"),
                log_id=str(payload.get("log_id") or ""),
                status_code=response.status_code,
            )
        token = str(payload.get("tenant_access_token") or "").strip()
        if not token:
            raise FeishuOpenApiError("Feishu auth succeeded but returned no tenant_access_token")
        expires_in = int(payload.get("expire", 7200) or 7200)
        with _TOKEN_CACHE_LOCK:
            _TOKEN_CACHE[cache_key] = {
                "token": token,
                "expires_at": now + max(60, expires_in - 120),
            }
        return token

    def _auth_headers(self, *, force_refresh: bool = False) -> Dict[str, str]:
        token = self.get_tenant_access_token(force_refresh=force_refresh)
        return {"Authorization": f"Bearer {token}"}

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any] | None = None,
        json_body: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        last_error: Exception | None = None
        force_refresh = False
        for attempt in range(1, retries + 1):
            try:
                request_headers = {"Content-Type": "application/json; charset=utf-8"}
                request_headers.update(self._auth_headers(force_refresh=force_refresh))
                if headers:
                    request_headers.update(headers)
                response = httpx.request(
                    method,
                    f"{self.base_url}{path}",
                    params=params,
                    json=json_body,
                    headers=request_headers,
                    timeout=self.timeout,
                )
                if response.status_code in {401, 403} and not force_refresh:
                    force_refresh = True
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                code = payload.get("code", 0)
                if code != 0:
                    if code in {99991661, 99991663} and not force_refresh:
                        force_refresh = True
                        continue
                    raise FeishuOpenApiError(
                        str(payload.get("msg") or "Feishu API error"),
                        code=code,
                        log_id=str(payload.get("log_id") or ""),
                        status_code=response.status_code,
                    )
                return payload.get("data", {}) or {}
            except Exception as exc:
                last_error = exc
                should_retry = attempt < retries and (
                    isinstance(exc, FeishuOpenApiError) and exc.status_code in {429, 500, 502, 503, 504}
                    or isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {429, 500, 502, 503, 504}
                    or isinstance(exc, httpx.TransportError)
                )
                if not should_retry:
                    break
                time.sleep(min(2.0, 0.4 * attempt))
        if isinstance(last_error, FeishuOpenApiError):
            raise last_error
        if last_error is not None:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError("Unknown Feishu JSON request failure")

    def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> tuple[bytes, httpx.Headers]:
        request_headers = self._auth_headers()
        if headers:
            request_headers.update(headers)
        response = httpx.request(
            method,
            f"{self.base_url}{path}",
            params=params,
            headers=request_headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.content, response.headers

    def upload_im_file(self, *, file_path: Path, file_name: str | None = None) -> Dict[str, Any]:
        resolved_name = str(file_name or file_path.name)
        file_type = detect_upload_file_type(Path(resolved_name))
        with file_path.open("rb") as fh:
            response = httpx.post(
                f"{self.base_url}/open-apis/im/v1/files",
                headers=self._auth_headers(),
                data={"file_type": file_type, "file_name": resolved_name},
                files={"file": (resolved_name, fh, mimetypes.guess_type(resolved_name)[0] or "application/octet-stream")},
                timeout=self.timeout,
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code", 0) != 0:
            raise FeishuOpenApiError(
                str(payload.get("msg") or "Feishu file upload failed"),
                code=payload.get("code"),
                log_id=str(payload.get("log_id") or ""),
                status_code=response.status_code,
            )
        return payload.get("data", {}) or {}

    def send_message(
        self,
        *,
        chat_id: str,
        msg_type: str,
        content: str,
    ) -> Dict[str, Any]:
        return self.request_json(
            "POST",
            "/open-apis/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            json_body={
                "receive_id": chat_id,
                "msg_type": msg_type,
                "content": content,
                "uuid": str(uuid.uuid4()),
            },
        )

    def send_uploaded_file_message(
        self,
        *,
        chat_id: str,
        file_key: str,
        caption: str | None = None,
        file_name: str | None = None,
    ) -> Dict[str, Any]:
        if caption:
            content = json.dumps(
                {
                    "zh_cn": {
                        "title": file_name or "Attachment",
                        "content": [
                            [{"tag": "text", "text": caption}],
                            [{"tag": "media", "file_key": file_key, "file_name": file_name or "attachment"}],
                        ],
                    }
                },
                ensure_ascii=False,
            )
            return self.send_message(chat_id=chat_id, msg_type="post", content=content)
        return self.send_message(
            chat_id=chat_id,
            msg_type="file",
            content=json.dumps({"file_key": file_key}, ensure_ascii=False),
        )


def build_feishu_client() -> FeishuOpenApiClient:
    return FeishuOpenApiClient()


def resolve_user_identifier(client: FeishuOpenApiClient, args: Dict[str, Any]) -> Tuple[str, str]:
    for key in ("user_id", "open_id", "union_id"):
        value = str(args.get(key, "") or "").strip()
        if value:
            return value, key

    email = str(args.get("email", "") or "").strip()
    mobile = str(args.get("mobile", "") or "").strip()
    if not email and not mobile:
        raise ValueError("Provide one of user_id, open_id, union_id, email, or mobile")

    body: Dict[str, Any] = {}
    if email:
        body["emails"] = [email]
    if mobile:
        body["mobiles"] = [mobile]
    data = client.request_json(
        "POST",
        "/open-apis/contact/v3/users/batch_get_id",
        params={"user_id_type": "open_id"},
        json_body=body,
    )
    user_list = data.get("user_list") or []
    if not user_list:
        raise RuntimeError("No Feishu user matched the supplied email/mobile")
    user_id = str(user_list[0].get("user_id") or "").strip()
    if not user_id:
        raise RuntimeError("Feishu returned an empty user_id for the supplied email/mobile")
    return user_id, "open_id"


def resolve_bitable_target(args: Dict[str, Any]) -> Tuple[str, str]:
    app_token = str(args.get("app_token") or os.getenv("FEISHU_BITABLE_APP_TOKEN") or "").strip()
    table_id = str(args.get("table_id") or os.getenv("FEISHU_BITABLE_TABLE_ID") or "").strip()
    if not app_token or not table_id:
        raise ValueError("Bitable target is not configured. Provide app_token/table_id or set FEISHU_BITABLE_APP_TOKEN and FEISHU_BITABLE_TABLE_ID.")
    return app_token, table_id


def build_model_registry(force_refresh: bool = False) -> Dict[str, Any]:
    ensure_feishu_data_dir()
    registry_path = get_model_registry_path()
    if registry_path.exists() and not force_refresh:
        cached = load_json(registry_path, {})
        if isinstance(cached, dict) and cached.get("entries"):
            return cached

    routing_state = load_json(get_routing_state_path(), {})
    providers = routing_state.get("providers") if isinstance(routing_state, dict) else {}
    provider_candidates: dict[str, list[str]] = {}
    for provider in ("openrouter", "nvidia"):
        payload = providers.get(provider) if isinstance(providers, dict) else None
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if isinstance(candidates, list) and candidates:
            provider_candidates[provider] = [str(item).strip() for item in candidates if str(item).strip()]

    if "openrouter" not in provider_candidates:
        try:
            from hermes_cli.models import OPENROUTER_MODELS

            provider_candidates["openrouter"] = [model_id for model_id, _note in OPENROUTER_MODELS[:24]]
        except Exception:
            provider_candidates["openrouter"] = ["openrouter/free"]

    if "nvidia" not in provider_candidates:
        provider_candidates["nvidia"] = [
            "qwen/qwq-32b",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
        ]

    registry_entries: list[dict[str, Any]] = []
    now = int(time.time())
    try:
        from agent.models_dev import get_model_info
    except Exception:
        get_model_info = None

    for provider, candidates in provider_candidates.items():
        for rank, model_id in enumerate(candidates, start=1):
            info = None
            if get_model_info is not None:
                try:
                    info = get_model_info(provider, model_id)
                except Exception:
                    info = None
            registry_entries.append(
                {
                    "provider": provider,
                    "model": model_id,
                    "display_name": getattr(info, "display_name", None) or model_id,
                    "is_free": provider == "nvidia" or model_id.endswith(":free") or model_id == "openrouter/free",
                    "is_available": True,
                    "rank": rank,
                    "last_probe_at": int(routing_state.get("refreshed_at") or now) if isinstance(routing_state, dict) else now,
                    "latency_ms": None,
                    "context_window": getattr(info, "context_window", 0) or None,
                    "reasoning": bool(getattr(info, "reasoning", False)) if info is not None else None,
                    "manual_pinned": rank == 1,
                    "selection_hint": "recommended" if rank == 1 else ("fallback" if rank > 3 else "candidate"),
                }
            )

    payload = {
        "status": "ok",
        "generated_at": now,
        "refreshed_at": int(routing_state.get("refreshed_at") or now) if isinstance(routing_state, dict) else now,
        "source": "routing_state" if isinstance(routing_state, dict) and routing_state.get("providers") else "curated-fallback",
        "entries": registry_entries,
    }
    atomic_write_json(registry_path, payload)
    return payload


def load_feishu_model_registry(force_refresh: bool = False) -> Dict[str, Any]:
    return build_model_registry(force_refresh=force_refresh)


def get_feishu_capability_snapshot(*, probe: bool = False) -> Dict[str, Any]:
    app_id, app_secret, domain = get_feishu_config()
    payload: Dict[str, Any] = {
        "configured": bool(app_id and app_secret),
        "domain": domain,
        "base_url": get_feishu_base_url(),
        "tool_capabilities": sorted(get_feishu_tool_capabilities()),
        "bitable_configured": bool(os.getenv("FEISHU_BITABLE_APP_TOKEN") and os.getenv("FEISHU_BITABLE_TABLE_ID")),
        "model_registry_mirror_enabled": str(os.getenv("FEISHU_MODEL_REGISTRY_MIRROR_ENABLED", "") or "").strip().lower() in {"1", "true", "yes"},
        "default_workspace": str(os.getenv("HERMES_FEISHU_DEFAULT_WORKSPACE", "") or "").strip(),
        "routing_state_present": get_routing_state_path().exists(),
        "model_registry_present": get_model_registry_path().exists(),
    }
    if not probe or not payload["configured"]:
        return payload
    try:
        token = build_feishu_client().get_tenant_access_token()
        payload["auth"] = {"ok": True, "token_prefix": token[:8]}
    except Exception as exc:
        payload["auth"] = {"ok": False, "error": str(exc)}
    home_channel = str(os.getenv("FEISHU_HOME_CHANNEL", "") or "").strip()
    if home_channel:
        try:
            chat = build_feishu_client().request_json("GET", f"/open-apis/im/v1/chats/{home_channel}")
            payload["home_channel_probe"] = {"ok": True, "chat": chat.get("chat", {})}
        except Exception as exc:
            payload["home_channel_probe"] = {"ok": False, "error": str(exc)}
    if payload["bitable_configured"]:
        try:
            app_token, table_id = resolve_bitable_target({})
            schema = build_feishu_client().request_json(
                "GET",
                f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields",
                params={"page_size": 50},
            )
            payload["bitable_probe"] = {
                "ok": True,
                "field_count": len(schema.get("items") or []),
            }
        except Exception as exc:
            payload["bitable_probe"] = {"ok": False, "error": str(exc)}
    return payload


def mirror_model_registry_to_bitable(
    client: FeishuOpenApiClient,
    registry_payload: Dict[str, Any],
    *,
    app_token: str,
    table_id: str,
) -> Dict[str, Any]:
    schema = client.request_json(
        "GET",
        f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields",
        params={"page_size": 200},
    )
    field_names = {
        str(item.get("field_name") or "").strip()
        for item in (schema.get("items") or [])
        if isinstance(item, dict)
    }
    expected_fields = {
        "Provider",
        "Model",
        "Display Name",
        "Is Free",
        "Is Available",
        "Rank",
        "Last Probe At",
        "Latency Ms",
        "Context Window",
        "Reasoning",
        "Manual Pinned",
        "Selection Hint",
    }
    missing_fields = sorted(name for name in expected_fields if name not in field_names)
    if missing_fields:
        raise RuntimeError(f"Bitable schema is missing fields required for mirroring: {', '.join(missing_fields)}")

    existing_records: list[dict[str, Any]] = []
    page_token = ""
    while True:
        params = {"page_size": 500}
        if page_token:
            params["page_token"] = page_token
        payload = client.request_json(
            "GET",
            f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records",
            params=params,
        )
        items = payload.get("items") or []
        existing_records.extend(item for item in items if isinstance(item, dict))
        if not payload.get("has_more"):
            break
        page_token = str(payload.get("page_token") or "").strip()
        if not page_token:
            break

    existing_lookup: dict[tuple[str, str], str] = {}
    for item in existing_records:
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        provider = str(fields.get("Provider") or "").strip().lower()
        model_name = str(fields.get("Model") or "").strip()
        record_id = str(item.get("record_id") or "").strip()
        if provider and model_name and record_id:
            existing_lookup[(provider, model_name)] = record_id

    created = 0
    updated = 0
    for entry in registry_payload.get("entries") or []:
        fields = {
            "Provider": entry.get("provider"),
            "Model": entry.get("model"),
            "Display Name": entry.get("display_name"),
            "Is Free": bool(entry.get("is_free")),
            "Is Available": bool(entry.get("is_available")),
            "Rank": int(entry.get("rank") or 0),
            "Last Probe At": int(entry.get("last_probe_at") or 0),
            "Latency Ms": entry.get("latency_ms"),
            "Context Window": entry.get("context_window"),
            "Reasoning": bool(entry.get("reasoning")) if entry.get("reasoning") is not None else None,
            "Manual Pinned": bool(entry.get("manual_pinned")),
            "Selection Hint": entry.get("selection_hint"),
        }
        record_id = existing_lookup.get((str(entry.get("provider") or "").strip().lower(), str(entry.get("model") or "").strip()))
        if record_id:
            client.request_json(
                "PUT",
                f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}",
                json_body={"fields": fields},
            )
            updated += 1
        else:
            client.request_json(
                "POST",
                f"/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records",
                json_body={"fields": fields},
            )
            created += 1
    return {
        "mirrored": True,
        "created": created,
        "updated": updated,
        "table_id": table_id,
        "app_token": app_token,
    }


def validate_message_resource_type(resource_type: str) -> str:
    normalized = str(resource_type or "file").strip().lower() or "file"
    if normalized not in _SUPPORTED_MESSAGE_RESOURCE_TYPES:
        raise ValueError(f"resource_type must be one of {', '.join(sorted(_SUPPORTED_MESSAGE_RESOURCE_TYPES))}")
    return normalized


def build_download_target_path(*, file_name: str) -> Path:
    download_dir = get_modal_data_root() / "feishu-downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir / file_name


def quote_range(range_name: str) -> str:
    return quote(range_name, safe="")
