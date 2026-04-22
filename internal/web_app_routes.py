from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .runtime_status import build_gateway_health_payload, load_gateway_import_status

try:
    from fastapi import Request
except Exception:  # pragma: no cover - modal_.py guards create_web_app when FastAPI is unavailable
    class Request:  # type: ignore[no-redef]
        pass


def register_health_route(
    app: Any,
    *,
    path: str,
    app_name: str,
    default_chat_queue_name: str,
    prepare_runtime_environment: Callable[[], None],
    settings_from_env: Callable[[], Any],
    get_memory_provider_status: Callable[[], dict[str, Any]],
    safe_chat_queue_depth_async: Callable[[], Awaitable[int | None]],
    cron_status_impl: Callable[[int], dict[str, Any]],
    get_telegram_webhook_status: Callable[[Any], Awaitable[dict[str, Any]]],
    build_model_routing_debug_state: Callable[..., dict[str, Any]],
    build_feishu_sync_state_debug_state: Callable[[], dict[str, Any]],
    sync_runtime_config: Callable[[], str],
    serialize_settings_for_log: Callable[[Any], dict[str, Any]],
) -> None:
    async def healthz() -> dict[str, Any]:
        prepare_runtime_environment()
        settings = settings_from_env()
        telegram_webhook = None
        memory_provider = get_memory_provider_status()
        try:
            chat_queue_depth = await safe_chat_queue_depth_async()
        except Exception:
            chat_queue_depth = None
        cron_status = await asyncio.to_thread(cron_status_impl, 5)
        gateway_import_ok, gateway_import_error = load_gateway_import_status()
        if settings.telegram_bot_token:
            try:
                telegram_webhook = await get_telegram_webhook_status(settings)
            except Exception as exc:
                telegram_webhook = {
                    "configured": True,
                    "expected_url": settings.telegram_webhook_url,
                    "error": str(exc),
                }
        return build_gateway_health_payload(
            status="ok",
            service=app_name,
            settings=settings,
            telegram_webhook=telegram_webhook,
            chat_queue_name=default_chat_queue_name,
            chat_queue_depth=chat_queue_depth,
            cron_status=cron_status,
            memory_provider=memory_provider,
            model_routing=build_model_routing_debug_state(force_refresh=False, allow_network=False),
            feishu_sync=build_feishu_sync_state_debug_state(),
            runtime_config=sync_runtime_config(),
            settings_payload=serialize_settings_for_log(settings),
            gateway_import_ok=gateway_import_ok,
            gateway_import_error=gateway_import_error,
        )

    app.add_api_route(path, healthz, methods=["GET"])


def register_invoke_route(
    app: Any,
    *,
    path: str,
    settings_from_env: Callable[[], Any],
    validate_bearer_token: Callable[[str | None, str | None], bool],
    http_exception_cls: type[Exception],
    run_agent_task_impl: Callable[..., dict[str, Any]],
) -> None:
    async def invoke(request: Request) -> dict[str, Any]:
        settings = settings_from_env()
        if not validate_bearer_token(request.headers.get("authorization"), settings.bearer_token):
            raise http_exception_cls(status_code=401, detail="Unauthorized")

        payload = await request.json()
        task_input = str(payload.get("input") or "").strip()
        if not task_input:
            raise http_exception_cls(status_code=400, detail="Missing input")

        return run_agent_task_impl(
            task_input,
            session_key=payload.get("session_key"),
            model_name=payload.get("model_name"),
            max_tokens=payload.get("max_tokens"),
        )

    app.add_api_route(path, invoke, methods=["POST"])


def register_internal_feishu_routes(
    app: Any,
    *,
    settings_from_env: Callable[[], Any],
    handle_internal_json_route: Callable[..., Awaitable[dict[str, Any]]],
    authorize_internal_file_route: Callable[..., None],
    validate_bearer_token: Callable[[str | None, str | None], bool],
    validate_feishu_internal_bearer_token: Callable[[str | None, Any], bool],
    log_auth_failure: Callable[..., None],
    extract_request_meta: Callable[[dict[str, Any] | None], dict[str, str]],
    http_exception_cls: type[Exception],
    run_internal_agent_exec: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    run_internal_agent_plan: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    run_internal_control: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    lookup_internal_result_file: Callable[[str], dict[str, Any] | None],
    file_response_cls: Any,
    agent_exec_path: str,
    agent_plan_path: str,
    session_control_path: str,
    result_file_path: str,
) -> None:
    async def feishu_internal_agent_exec(request: Request) -> dict[str, Any]:
        settings = settings_from_env()
        return await handle_internal_json_route(
            request,
            authorization=request.headers.get("authorization"),
            endpoint=agent_exec_path,
            expected_token=settings.feishu_internal_bearer_token,
            validate_bearer_token=lambda authorization, _expected: validate_feishu_internal_bearer_token(
                authorization,
                settings,
            ),
            log_auth_failure=log_auth_failure,
            extract_request_meta=extract_request_meta,
            http_exception_cls=http_exception_cls,
            executor=run_internal_agent_exec,
        )

    async def feishu_internal_agent_plan(request: Request) -> dict[str, Any]:
        settings = settings_from_env()
        return await handle_internal_json_route(
            request,
            authorization=request.headers.get("authorization"),
            endpoint=agent_plan_path,
            expected_token=settings.feishu_internal_bearer_token,
            validate_bearer_token=lambda authorization, _expected: validate_feishu_internal_bearer_token(
                authorization,
                settings,
            ),
            log_auth_failure=log_auth_failure,
            extract_request_meta=extract_request_meta,
            http_exception_cls=http_exception_cls,
            executor=run_internal_agent_plan,
        )

    async def feishu_internal_session_control(request: Request) -> dict[str, Any]:
        settings = settings_from_env()
        return await handle_internal_json_route(
            request,
            authorization=request.headers.get("authorization"),
            endpoint=session_control_path,
            expected_token=settings.feishu_internal_bearer_token,
            validate_bearer_token=lambda authorization, _expected: validate_feishu_internal_bearer_token(
                authorization,
                settings,
            ),
            log_auth_failure=log_auth_failure,
            extract_request_meta=extract_request_meta,
            http_exception_cls=http_exception_cls,
            executor=run_internal_control,
        )

    async def feishu_internal_result_file(token: str, request: Request) -> Any:
        settings = settings_from_env()
        authorize_internal_file_route(
            authorization=request.headers.get("authorization"),
            endpoint=result_file_path.rsplit("/", 1)[0],
            expected_token=settings.feishu_internal_bearer_token,
            validate_bearer_token=lambda authorization, _expected: validate_feishu_internal_bearer_token(
                authorization,
                settings,
            ),
            log_auth_failure=log_auth_failure,
            http_exception_cls=http_exception_cls,
        )
        payload = lookup_internal_result_file(token)
        if payload is None:
            raise http_exception_cls(status_code=404, detail="Result file not found")
        return file_response_cls(
            path=str(payload.get("path") or ""),
            filename=str(payload.get("filename") or "artifact.bin"),
            media_type=str(payload.get("content_type") or "application/octet-stream"),
        )

    app.add_api_route(agent_exec_path, feishu_internal_agent_exec, methods=["POST"])
    app.add_api_route(agent_plan_path, feishu_internal_agent_plan, methods=["POST"])
    app.add_api_route(session_control_path, feishu_internal_session_control, methods=["POST"])
    app.add_api_route(result_file_path, feishu_internal_result_file, methods=["GET"])


def register_telegram_route(
    app: Any,
    *,
    path: str,
    settings_from_env: Callable[[], Any],
    validate_telegram_secret: Callable[[str | None, str | None], bool],
    http_exception_cls: type[Exception],
    mark_update_seen: Callable[[str], bool],
    extract_inline_fast_command: Callable[[dict[str, Any]], str | None],
    dispatch_telegram_update: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    send_telegram_message: Callable[[str, str | int, str], Awaitable[None]],
    extract_telegram_queue_context: Callable[[dict[str, Any]], dict[str, str]],
    enqueue_chat_event_async: Callable[..., Awaitable[dict[str, Any]]],
    spawn_chat_queue_worker_async: Callable[..., Awaitable[dict[str, Any]]],
    default_chat_queue_batch_size: int,
    logger: Any,
) -> None:
    async def telegram_webhook(request: Request) -> dict[str, Any]:
        settings = settings_from_env()
        if not settings.telegram_bot_token:
            raise http_exception_cls(status_code=503, detail="Telegram bot token is not configured")
        if not validate_telegram_secret(
            request.headers.get("x-telegram-bot-api-secret-token"),
            settings.telegram_webhook_secret,
        ):
            raise http_exception_cls(status_code=401, detail="Invalid Telegram webhook secret")

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
        if update_id is not None and not mark_update_seen(str(update_id)):
            return {"status": "duplicate", "update_id": update_id}

        inline_fast_command = extract_inline_fast_command(update)
        if inline_fast_command:
            logger.info(
                "Telegram webhook inline fast command: update_id=%s command=%s",
                update_id,
                inline_fast_command,
            )
            await dispatch_telegram_update(update)
            return {
                "status": "accepted",
                "update_id": update_id,
                "mode": "inline_fast_command",
                "command": inline_fast_command,
            }

        if settings.telegram_send_ack:
            chat_id = chat.get("id")
            if chat_id:
                await send_telegram_message(settings.telegram_bot_token, chat_id, "Thinking...")

        context = extract_telegram_queue_context(update)
        enqueue_result = await enqueue_chat_event_async(
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
        await spawn_chat_queue_worker_async(
            platform="telegram",
            partition=context["partition"],
            max_items=default_chat_queue_batch_size,
        )
        return {"status": "accepted", "update_id": update_id}

    app.add_api_route(path, telegram_webhook, methods=["POST"])


def register_feishu_route(
    app: Any,
    *,
    path: str,
    settings_from_env: Callable[[], Any],
    handle_feishu_webhook_route: Callable[..., Awaitable[Any]],
    build_feishu_webhook_deps: Callable[[], Any],
) -> None:
    async def feishu_webhook(request: Request) -> Any:
        settings = settings_from_env()
        return await handle_feishu_webhook_route(
            request,
            settings=settings,
            deps=build_feishu_webhook_deps(),
        )

    app.add_api_route(path, feishu_webhook, methods=["POST"])


def register_qq_route(
    app: Any,
    *,
    path: str,
    settings_from_env: Callable[[], Any],
    dispatch_qq_update: Callable[..., Awaitable[dict[str, Any]]],
    http_exception_cls: type[Exception],
    logger: Any,
) -> None:
    async def qq_webhook(request: Request) -> dict[str, Any]:
        settings = settings_from_env()
        if not settings.qq_app_id or not settings.qq_app_secret:
            raise http_exception_cls(status_code=503, detail="QQ bot credentials are not configured")

        payload = await request.json()
        try:
            return await dispatch_qq_update(
                payload,
                headers={"X-Bot-Appid": request.headers.get("x-bot-appid", "")},
            )
        except Exception as exc:
            try:
                from gateway.platforms.qq import QQWebhookError

                if isinstance(exc, QQWebhookError):
                    raise http_exception_cls(status_code=exc.status_code, detail=exc.message) from exc
            except ImportError:
                pass
            logger.exception("QQ webhook dispatch failed")
            raise http_exception_cls(status_code=500, detail=f"QQ dispatch failed: {exc}") from exc

    app.add_api_route(path, qq_webhook, methods=["POST"])
