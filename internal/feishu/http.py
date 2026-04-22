from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping

from .contracts import build_legacy_request_payload, build_response_envelope


async def handle_internal_json_route(
    request: Any,
    *,
    authorization: str | None,
    endpoint: str,
    expected_token: str | None,
    validate_bearer_token: Callable[[str | None, str | None], bool],
    log_auth_failure: Callable[..., None],
    extract_request_meta: Callable[[Mapping[str, Any] | None], dict[str, str]],
    http_exception_cls: type[Exception],
    executor: Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    if not validate_bearer_token(authorization, expected_token):
        log_auth_failure(
            endpoint=endpoint,
            authorization=authorization,
            expected_token=expected_token,
            headers=request.headers,
        )
        raise http_exception_cls(status_code=401, detail="Unauthorized")

    raw_payload = await request.json()
    if not isinstance(raw_payload, dict):
        raise http_exception_cls(status_code=400, detail="Expected JSON object payload")

    payload = build_legacy_request_payload(raw_payload)
    gateway_meta = raw_payload.get("gateway_meta")
    if isinstance(gateway_meta, Mapping):
        payload["_hermes_gateway_meta"] = dict(gateway_meta)
    payload["_hermes_gateway_request"] = extract_request_meta(request.headers)
    return build_response_envelope(await executor(payload))


def authorize_internal_file_route(
    *,
    authorization: str | None,
    endpoint: str,
    expected_token: str | None,
    validate_bearer_token: Callable[[str | None, str | None], bool],
    log_auth_failure: Callable[..., None],
    http_exception_cls: type[Exception],
) -> None:
    if validate_bearer_token(authorization, expected_token):
        return
    log_auth_failure(
        endpoint=endpoint,
        authorization=authorization,
        expected_token=expected_token,
        headers=None,
    )
    raise http_exception_cls(status_code=401, detail="Unauthorized")
