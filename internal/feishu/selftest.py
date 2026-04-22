from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Callable, Mapping


def encrypt_payload(encrypt_key: str, payload: dict[str, Any]) -> str:
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


def decrypt_payload(encrypt_key: str, encrypted_payload: str) -> dict[str, Any]:
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
        raise ValueError("decrypted Feishu payload must be a JSON object")
    return payload


def is_signature_valid(
    headers: Mapping[str, Any],
    body_bytes: bytes,
    *,
    encrypt_key: str,
    allow_when_encrypt_key_missing: bool,
    logger: Any | None = None,
) -> bool:
    if not encrypt_key:
        return allow_when_encrypt_key_missing
    timestamp = str(headers.get("x-lark-request-timestamp", "") or "")
    nonce = str(headers.get("x-lark-request-nonce", "") or "")
    signature = str(headers.get("x-lark-signature", "") or "")
    if not timestamp or not nonce or not signature:
        return False
    try:
        body_str = body_bytes.decode("utf-8", errors="replace")
        computed = hashlib.sha256(f"{timestamp}{nonce}{encrypt_key}{body_str}".encode("utf-8")).hexdigest()
        return hmac.compare_digest(computed, signature)
    except Exception:
        if logger is not None:
            logger.debug("[Feishu] Signature verification raised an exception", exc_info=True)
        return False


async def try_handle_verification_fast(
    request: Any,
    *,
    settings: Any,
    response_cls: Any,
    json_response_cls: Any,
    safe_json_loads: Callable[[str, Any], Any],
    is_signature_valid: Callable[[Mapping[str, Any], bytes, str], bool],
    decrypt_payload: Callable[[str, str], dict[str, Any]],
    logger: Any,
) -> Any | None:
    if response_cls is None or json_response_cls is None:
        return None

    body = await request.body()
    if not body:
        return None

    payload = safe_json_loads(body.decode("utf-8", errors="replace"), None)
    if not isinstance(payload, dict):
        return None

    if payload.get("type") == "url_verification":
        return json_response_cls({"challenge": payload.get("challenge", "")})

    encrypted_payload = str(payload.get("encrypt") or "").strip()
    if not encrypted_payload or not settings.feishu_encrypt_key:
        return None
    if not is_signature_valid(dict(request.headers), body, settings.feishu_encrypt_key):
        return None

    try:
        inner_payload = decrypt_payload(settings.feishu_encrypt_key, encrypted_payload)
    except Exception:
        logger.debug("Fast Feishu verification decrypt failed", exc_info=True)
        return None

    if inner_payload.get("type") != "url_verification":
        return None
    expected_token = str(settings.feishu_verification_token or "").strip()
    provided_token = str((inner_payload.get("header") or {}).get("token") or inner_payload.get("token") or "").strip()
    if expected_token and (not provided_token or not hmac.compare_digest(provided_token, expected_token)):
        return None
    return json_response_cls({"challenge": inner_payload.get("challenge", "")})


def validate_webhook(
    *,
    prepare_runtime_environment: Callable[[], None],
    settings_from_env: Callable[[], Any],
    normalize_public_https_url: Callable[[str | None], str],
    safe_json_loads: Callable[[str, Any], Any],
    encrypt_payload: Callable[[str, dict[str, Any]], str],
) -> dict[str, Any]:
    import httpx

    prepare_runtime_environment()
    settings = settings_from_env()
    webhook_url = _resolve_webhook_url(normalize_public_https_url=normalize_public_https_url)
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
    body = json.dumps({"encrypt": encrypt_payload(encrypt_key, inner_payload)}, ensure_ascii=False)
    headers = _build_signed_headers(body=body, encrypt_key=encrypt_key, nonce="hermes-feishu-selftest")

    with httpx.Client(timeout=20) as client:
        response = client.post(webhook_url, content=body.encode("utf-8"), headers=headers)

    return {
        "status": "ok" if response.status_code == 200 else "error",
        "webhook_url": webhook_url,
        "status_code": response.status_code,
        "response": safe_json_loads(response.text, response.text),
        "verification_token_configured": True,
        "encrypt_key_configured": True,
    }


def validate_message_ingress(
    *,
    message_text: str,
    target_webhook_url: str,
    public_base_url: str,
    request_only: bool,
    prepare_runtime_environment: Callable[[], None],
    settings_from_env: Callable[[], Any],
    normalize_public_https_url: Callable[[str | None], str],
    safe_json_loads: Callable[[str, Any], Any],
    resolve_message_ingress_strategy: Callable[[dict[str, Any], dict[str, Any]], str],
    default_message_ingress_strategy: str,
) -> dict[str, Any]:
    import httpx

    prepare_runtime_environment()
    settings = settings_from_env()
    webhook_url = _resolve_webhook_url(
        explicit_webhook_url=target_webhook_url,
        explicit_public_base=public_base_url,
        normalize_public_https_url=normalize_public_https_url,
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
    headers = _build_signed_headers(
        body=body,
        encrypt_key=encrypt_key,
        nonce="hermes-feishu-message-selftest",
    )
    strategy = {
        "configured": default_message_ingress_strategy,
        "effective": resolve_message_ingress_strategy({}, {}),
    }

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
            "message_ingress_strategy": strategy,
        }

    with httpx.Client(timeout=20) as client:
        response = client.post(webhook_url, content=body.encode("utf-8"), headers=headers)

    return {
        "status": "ok" if response.status_code == 200 else "error",
        "webhook_url": webhook_url,
        "status_code": response.status_code,
        "response": safe_json_loads(response.text, response.text),
        "event_id": event_id,
        "message_id": message_id,
        "verification_token_configured": bool(verification_token),
        "encrypt_key_configured": bool(encrypt_key),
        "message_ingress_strategy": strategy,
    }


def send_custom_message_ingress(
    *,
    message_text: str,
    chat_id: str,
    sender_open_id: str,
    sender_user_id: str,
    chat_type: str,
    message_id: str,
    target_webhook_url: str,
    public_base_url: str,
    request_only: bool,
    prepare_runtime_environment: Callable[[], None],
    settings_from_env: Callable[[], Any],
    normalize_public_https_url: Callable[[str | None], str],
    safe_json_loads: Callable[[str, Any], Any],
    resolve_message_ingress_strategy: Callable[[dict[str, Any], dict[str, Any]], str],
    default_message_ingress_strategy: str,
) -> dict[str, Any]:
    import httpx

    prepare_runtime_environment()
    settings = settings_from_env()
    normalized_chat_id = str(chat_id or "").strip()
    normalized_sender_open_id = str(sender_open_id or "").strip()
    normalized_sender_user_id = str(sender_user_id or "").strip()
    normalized_chat_type = str(chat_type or "p2p").strip().lower() or "p2p"
    if not normalized_chat_id:
        return {"status": "error", "message": "chat_id is required"}
    if not normalized_sender_open_id:
        return {"status": "error", "message": "sender_open_id is required"}

    webhook_url = _resolve_webhook_url(
        explicit_webhook_url=target_webhook_url,
        explicit_public_base=public_base_url,
        normalize_public_https_url=normalize_public_https_url,
    )
    verification_token = str(settings.feishu_verification_token or "").strip()
    encrypt_key = str(settings.feishu_encrypt_key or "").strip()
    event_id = f"evt_probe_{uuid.uuid4().hex[:16]}"
    normalized_message_id = str(message_id or "").strip() or f"om_probe_{uuid.uuid4().hex[:12]}"
    payload = {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": event_id,
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": normalized_sender_open_id},
                "sender_type": "user",
            },
            "message": {
                "message_id": normalized_message_id,
                "chat_id": normalized_chat_id,
                "chat_type": normalized_chat_type,
                "message_type": "text",
                "content": json.dumps({"text": str(message_text or "").strip() or "probe"}, ensure_ascii=False),
            },
        },
    }
    if verification_token:
        payload["header"]["token"] = verification_token
    if normalized_sender_user_id:
        payload["event"]["sender"]["sender_id"]["user_id"] = normalized_sender_user_id

    body = json.dumps(payload, ensure_ascii=False)
    headers = _build_signed_headers(
        body=body,
        encrypt_key=encrypt_key,
        nonce="hermes-feishu-custom-message-probe",
    )
    strategy = {
        "configured": default_message_ingress_strategy,
        "effective": resolve_message_ingress_strategy({}, {}),
    }

    if request_only:
        return {
            "status": "prepared",
            "webhook_url": webhook_url,
            "headers": headers,
            "body": body,
            "event_id": event_id,
            "message_id": normalized_message_id,
            "chat_id": normalized_chat_id,
            "sender_open_id": normalized_sender_open_id,
            "sender_user_id": normalized_sender_user_id,
            "verification_token_configured": bool(verification_token),
            "encrypt_key_configured": bool(encrypt_key),
            "message_ingress_strategy": strategy,
        }

    with httpx.Client(timeout=20) as client:
        response = client.post(webhook_url, content=body.encode("utf-8"), headers=headers)

    return {
        "status": "ok" if response.status_code == 200 else "error",
        "webhook_url": webhook_url,
        "status_code": response.status_code,
        "response": safe_json_loads(response.text, response.text),
        "event_id": event_id,
        "message_id": normalized_message_id,
        "chat_id": normalized_chat_id,
        "sender_open_id": normalized_sender_open_id,
        "sender_user_id": normalized_sender_user_id,
        "verification_token_configured": bool(verification_token),
        "encrypt_key_configured": bool(encrypt_key),
        "message_ingress_strategy": strategy,
    }


def _resolve_webhook_url(
    *,
    explicit_webhook_url: str = "",
    explicit_public_base: str = "",
    normalize_public_https_url: Callable[[str | None], str],
) -> str:
    normalized_webhook_url = str(explicit_webhook_url or "").strip()
    if normalized_webhook_url:
        return normalized_webhook_url

    normalized_public_base = normalize_public_https_url(explicit_public_base)
    runtime_public_base = normalize_public_https_url(os.getenv("HERMES_PUBLIC_BASE_URL") or os.getenv("PUBLIC_BASE_URL"))
    resolved_public_base = normalized_public_base or runtime_public_base
    if resolved_public_base:
        return f"{resolved_public_base}/feishu/webhook"
    return "https://isuyee88--hermes-agent-web-app.modal.run/feishu/webhook"


def _build_signed_headers(*, body: str, encrypt_key: str, nonce: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if not encrypt_key:
        return headers
    timestamp = str(int(time.time()))
    signature = hashlib.sha256(f"{timestamp}{nonce}{encrypt_key}{body}".encode("utf-8")).hexdigest()
    headers.update(
        {
            "x-lark-request-timestamp": timestamp,
            "x-lark-request-nonce": nonce,
            "x-lark-signature": signature,
        }
    )
    return headers
