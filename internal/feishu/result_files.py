from __future__ import annotations

import mimetypes
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, MutableMapping


def prune_result_files(
    store: MutableMapping[str, dict[str, Any]],
    *,
    now_ts: float | None = None,
) -> None:
    now_value = float(now_ts or time.time())
    expired_tokens = [
        token
        for token, payload in store.items()
        if float(payload.get("expires_at") or 0) <= now_value
    ]
    for token in expired_tokens:
        store.pop(token, None)


def register_result_file(
    file_path: str,
    *,
    kind: str,
    store: MutableMapping[str, dict[str, Any]],
    lock: Lock,
    default_ttl_seconds: int,
    is_voice: bool = False,
    ttl_seconds: int | None = None,
) -> dict[str, Any] | None:
    normalized_path = str(file_path or "").strip()
    if not normalized_path:
        return None

    path = Path(normalized_path).expanduser()
    if not path.exists() or not path.is_file():
        return None

    token = uuid.uuid4().hex
    ttl_value = max(60, int(ttl_seconds or default_ttl_seconds))
    expires_at = int(time.time()) + ttl_value
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    with lock:
        prune_result_files(store)
        store[token] = {
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


def lookup_result_file(
    token: str,
    *,
    store: MutableMapping[str, dict[str, Any]],
    lock: Lock,
) -> dict[str, Any] | None:
    normalized_token = str(token or "").strip()
    if not normalized_token:
        return None

    with lock:
        prune_result_files(store)
        payload = store.get(normalized_token)
        if not payload:
            return None

        path = Path(str(payload.get("path") or ""))
        if not path.exists() or not path.is_file():
            store.pop(normalized_token, None)
            return None

        return dict(payload)
