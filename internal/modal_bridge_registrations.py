from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from internal.modal_bridge_factory import bind_async, bind_sync


Namespace = MutableMapping[str, Any]


def _settings_from_env(namespace: Namespace) -> Any:
    return namespace["RuntimeSettings"].from_env


def register_gateway_bridges(namespace: Namespace) -> None:
    namespace["_dispatch_telegram_update"] = bind_async(
        namespace,
        namespace["_gateway_dispatch_telegram_update"],
        dynamic={"get_runtime": "_get_telegram_gateway_runtime"},
    )
    namespace["_dispatch_qq_update"] = bind_async(
        namespace,
        namespace["_gateway_dispatch_qq_update"],
        dynamic={"get_runtime": "_get_qq_gateway_runtime"},
    )
    namespace["_parse_feishu_webhook_request"] = bind_async(
        namespace,
        namespace["_feishu_parse_webhook_request_bridge"],
        dynamic={
            "get_security_settings": "_get_feishu_webhook_security_settings",
            "is_signature_valid": lambda ns: (
                lambda headers, body: ns["_is_feishu_webhook_signature_valid"](
                    headers,
                    body,
                    encrypt_key=ns["_get_feishu_webhook_security_settings"]()[0],
                )
            ),
            "decrypt_payload": "_decrypt_feishu_webhook_payload",
            "logger": "logger",
            "http_exception_cls": lambda ns: ns["HTTPException"],
        },
    )
    namespace["_extract_feishu_event_metadata"] = bind_sync(namespace, namespace["_feishu_extract_event_metadata"])
    namespace["_extract_leading_command_text"] = bind_sync(namespace, namespace["_feishu_extract_leading_command_text"])
    namespace["_resolve_inline_fast_command"] = bind_sync(
        namespace,
        namespace["_feishu_resolve_inline_fast_command"],
        static={"canonicals": namespace["INLINE_FAST_COMMAND_CANONICALS"]},
    )
    namespace["_extract_feishu_text_content_from_raw_content"] = bind_sync(
        namespace,
        namespace["_feishu_extract_text_content_from_raw_content"],
    )
    namespace["_extract_feishu_trace_token"] = bind_sync(
        namespace,
        namespace["_feishu_extract_trace_token"],
        static={"token_pattern": namespace["_FEISHU_TRACE_TOKEN_RE"]},
    )
    namespace["_extract_feishu_inline_fast_command"] = bind_sync(
        namespace,
        namespace["_feishu_extract_inline_fast_command"],
        static={"canonicals": namespace["INLINE_FAST_COMMAND_CANONICALS"]},
    )
    namespace["_classify_feishu_chat_lane"] = bind_sync(namespace, namespace["_feishu_classify_chat_lane"])
    namespace["_extract_telegram_inline_fast_command"] = bind_sync(
        namespace,
        namespace["_feishu_extract_telegram_inline_fast_command"],
        static={"canonicals": namespace["INLINE_FAST_COMMAND_CANONICALS"]},
    )
    namespace["_should_inline_feishu_control_event"] = bind_sync(
        namespace,
        namespace["_feishu_should_inline_control_event"],
    )
    namespace["_is_feishu_session_warmup_event"] = bind_sync(
        namespace,
        namespace["_feishu_is_session_warmup_event"],
    )
    namespace["_await_feishu_pending_batches"] = bind_async(
        namespace,
        namespace["_feishu_await_pending_batches_bridge"],
        dynamic={"logger": "logger"},
    )
    namespace["_await_feishu_background_tasks"] = bind_async(
        namespace,
        namespace["_feishu_await_background_tasks_bridge"],
        dynamic={"logger": "logger"},
    )


def register_runtime_bridges(namespace: Namespace) -> None:
    namespace["_bootstrap_chat_queue_worker_context"] = bind_sync(
        namespace,
        namespace["_chat_queue_worker_bootstrap_context"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "load_routing_state": "_load_routing_state",
            "settings_from_env": _settings_from_env,
            "get_feishu_gateway_runtime": "_get_feishu_gateway_runtime",
            "chat_worker_preinit_feishu_runtime_enabled": lambda ns: ns["CHAT_WORKER_PREINIT_FEISHU_RUNTIME_ENABLED"],
            "logger": "logger",
        },
    )
    namespace["_maybe_sync_telegram_webhook"] = bind_async(
        namespace,
        namespace["_telegram_gateway_maybe_sync_webhook"],
        dynamic={
            "is_truthy": "_is_truthy",
            "get_telegram_webhook_status_fn": "_get_telegram_webhook_status",
            "save_telegram_webhook_sync_state": "_save_telegram_webhook_sync_state",
            "load_telegram_webhook_sync_state": "_load_telegram_webhook_sync_state",
            "set_telegram_webhook_fn": "_set_telegram_webhook",
            "telegram_webhook_retry_after_seconds_fn": "_telegram_webhook_retry_after_seconds",
        },
        static={"drop_pending_updates": False},
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
        return namespace["_chat_queue_worker_process_queue"](
            platform=platform,
            partition=partition,
            max_items=namespace["DEFAULT_CHAT_QUEUE_BATCH_SIZE"] if max_items is None else max_items,
            claim_token=claim_token,
            worker_context=worker_context,
            runtime_prepared=runtime_prepared,
            warmup_wait_seconds=warmup_wait_seconds,
            warmup_metadata=warmup_metadata,
            default_chat_queue_linger_seconds=namespace["DEFAULT_CHAT_QUEUE_LINGER_SECONDS"],
            build_chat_worker_observability=namespace["_build_chat_worker_observability"],
            build_inline_chat_worker_context=namespace["_build_inline_chat_worker_context"],
            pop_chat_queue_warmup_snapshot=namespace["_pop_chat_queue_warmup_snapshot"],
            first_non_empty_str=namespace["_first_non_empty_str"],
            record_chat_queue_warmup_snapshot=namespace["_record_chat_queue_warmup_snapshot"],
            safe_chat_queue_depth=namespace["_safe_chat_queue_depth"],
            claim_chat_partition=namespace["_claim_chat_partition"],
            get_chat_queue=namespace["_get_chat_queue"],
            process_chat_queue_items_async=namespace["_process_chat_queue_items_async"],
            sync_modal_volume=namespace["_sync_modal_volume"],
            release_chat_partition_claim=namespace["_release_chat_partition_claim"],
            logger=namespace["logger"],
        )

    namespace["_process_chat_queue_impl"] = _process_chat_queue_impl
    namespace["_run_agent_task_impl"] = bind_sync(
        namespace,
        namespace["_agent_invoke_run_agent_task"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "settings_from_env": _settings_from_env,
            "load_session_state": "_load_session_state",
            "resolve_primary_route": "_resolve_primary_route",
            "is_route_lease_active": "_is_route_lease_active",
            "hydrate_route_from_lease": "_hydrate_route_from_lease",
            "resolve_runtime_model_name": "_resolve_runtime_model_name",
            "determine_route_refresh_reason": "_determine_route_refresh_reason",
            "select_retry_route_for_result": "_select_retry_route_for_result",
            "build_route_lease": "_build_route_lease",
            "expire_route_lease": "_expire_route_lease",
            "refresh_route_lease": "_refresh_route_lease",
            "save_session_state": "_save_session_state",
            "extract_tool_names": "_extract_tool_names",
            "increment_route_metric": "_increment_route_metric",
            "logger": "logger",
        },
    )
    namespace["_validate_tavily_integration_impl"] = bind_sync(
        namespace,
        namespace["_modal_diagnostics_validate_tavily_integration"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "safe_json_loads": "_safe_json_loads",
        },
    )
    namespace["_approve_pairing_impl"] = bind_sync(
        namespace,
        namespace["_modal_diagnostics_approve_pairing"],
        dynamic={"prepare_runtime_environment": "_prepare_runtime_environment"},
    )

    async def _try_handle_feishu_verification_fast(request: Any, settings: Any) -> Any:
        return await namespace["_feishu_selftest_try_handle_verification_fast"](
            request,
            settings=settings,
            response_cls=namespace["Response"],
            json_response_cls=namespace["JSONResponse"],
            safe_json_loads=namespace["_safe_json_loads"],
            is_signature_valid=namespace["_is_feishu_signature_valid"],
            decrypt_payload=namespace["_decrypt_feishu_payload"],
            logger=namespace["logger"],
        )

    namespace["_try_handle_feishu_verification_fast"] = _try_handle_feishu_verification_fast
    namespace["_validate_feishu_webhook_impl"] = bind_sync(
        namespace,
        namespace["_feishu_selftest_validate_webhook"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "settings_from_env": _settings_from_env,
            "normalize_public_https_url": "_normalize_public_https_url",
            "safe_json_loads": "_safe_json_loads",
            "encrypt_payload": "_encrypt_feishu_payload",
        },
    )
    namespace["_validate_feishu_message_ingress_impl"] = bind_sync(
        namespace,
        namespace["_feishu_selftest_validate_message_ingress"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "settings_from_env": _settings_from_env,
            "normalize_public_https_url": "_normalize_public_https_url",
            "safe_json_loads": "_safe_json_loads",
            "resolve_message_ingress_strategy": "_resolve_feishu_message_ingress_strategy",
        },
        static={
            "target_webhook_url": "",
            "public_base_url": "",
            "request_only": False,
            "default_message_ingress_strategy": namespace["DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY"],
        },
    )
    namespace["_send_custom_feishu_message_ingress_impl"] = bind_sync(
        namespace,
        namespace["_feishu_selftest_send_custom_message_ingress"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "settings_from_env": _settings_from_env,
            "normalize_public_https_url": "_normalize_public_https_url",
            "safe_json_loads": "_safe_json_loads",
            "resolve_message_ingress_strategy": "_resolve_feishu_message_ingress_strategy",
        },
        static={
            "sender_user_id": "",
            "chat_type": "p2p",
            "message_id": "",
            "target_webhook_url": "",
            "public_base_url": "",
            "request_only": False,
            "default_message_ingress_strategy": namespace["DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY"],
        },
    )


def register_registry_and_cron_bridges(namespace: Namespace) -> None:
    namespace["_build_model_routing_debug_state"] = bind_sync(
        namespace,
        namespace["_session_debug_build_model_routing_debug_state"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "load_routing_state": "_load_routing_state",
            "refresh_free_model_routes": "_refresh_free_model_routes",
            "candidate_routes_from_state": "_candidate_routes_from_state",
            "build_recent_session_route_summaries_fn": "_build_recent_session_route_summaries",
            "aggregate_session_route_metrics_fn": "_aggregate_session_route_metrics",
        },
    )
    namespace["_build_feishu_capabilities_debug_state"] = bind_sync(
        namespace,
        namespace["_feishu_registry_build_capabilities_debug_state"],
        dynamic={"prepare_runtime_environment": "_prepare_runtime_environment"},
    )
    namespace["_build_feishu_model_registry_debug_state"] = bind_sync(
        namespace,
        namespace["_feishu_registry_build_model_registry_debug_state"],
        dynamic={"prepare_runtime_environment": "_prepare_runtime_environment"},
    )
    namespace["_run_feishu_registry_sync_cycle"] = bind_sync(
        namespace,
        namespace["_feishu_registry_run_sync_cycle"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "load_feishu_sync_state": "_load_feishu_sync_state",
            "save_feishu_sync_state": "_save_feishu_sync_state",
            "settings_from_env": _settings_from_env,
            "refresh_free_model_routes": "_refresh_free_model_routes",
            "mask_runtime_identifier": "_mask_runtime_identifier",
            "compact_feishu_schema_status": "_compact_feishu_schema_status",
            "logger": "logger",
        },
    )

    def _sync_feishu_model_registry_impl(
        *,
        force_refresh: bool = False,
        mirror_to_bitable: bool | None = None,
    ) -> dict[str, Any]:
        return namespace["_run_feishu_registry_sync_cycle"](
            force_refresh=force_refresh,
            mirror_to_bitable=mirror_to_bitable,
            ensure_schema=True,
        )

    namespace["_sync_feishu_model_registry_impl"] = _sync_feishu_model_registry_impl
    namespace["_prepare_feishu_model_registry_bitable_impl"] = bind_sync(
        namespace,
        namespace["_feishu_registry_prepare_bitable_impl"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "load_feishu_sync_state": "_load_feishu_sync_state",
            "save_feishu_sync_state": "_save_feishu_sync_state",
            "settings_from_env": _settings_from_env,
            "mask_runtime_identifier": "_mask_runtime_identifier",
            "compact_feishu_schema_status": "_compact_feishu_schema_status",
        },
    )
    namespace["_build_feishu_sync_state_debug_state"] = bind_sync(
        namespace,
        namespace["_feishu_registry_build_sync_state_debug_state"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "load_feishu_sync_state": "_load_feishu_sync_state",
            "settings_from_env": _settings_from_env,
            "state_file": lambda ns: ns["FEISHU_SYNC_STATE_PATH"],
            "mask_runtime_identifier": "_mask_runtime_identifier",
        },
    )
    namespace["_should_prepare_feishu_registry_schema_on_startup"] = bind_sync(
        namespace,
        namespace["_feishu_registry_should_prepare_schema_on_startup"],
        dynamic={
            "default_recheck_seconds": lambda ns: ns["DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS"],
        },
    )
    namespace["_validate_feishu_native_delivery_impl"] = bind_sync(
        namespace,
        namespace["_feishu_validate_native_delivery"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "data_root": lambda ns: ns["DATA_ROOT"],
            "get_feishu_gateway_runtime": "_get_feishu_gateway_runtime",
        },
        static={"tiny_png_base64": namespace["_TINY_PNG_BASE64"]},
    )
    namespace["_process_cron_queue_impl"] = bind_sync(
        namespace,
        namespace["_cron_process_queue_bridge"],
        dynamic={
            "prepare_runtime_environment": "_prepare_runtime_environment",
            "get_cron_queue": "_get_cron_queue",
            "process_cron_queue_item_fn": "_process_cron_queue_item",
            "safe_cron_queue_depth": "_safe_cron_queue_depth",
        },
        static={"max_jobs": 1},
    )

    def _cron_scheduler_tick_impl(
        *,
        enqueue_limit: int | None = None,
        worker_count: int | None = None,
    ) -> dict[str, Any]:
        normalized_enqueue_limit = (
            namespace["DEFAULT_CRON_QUEUE_BATCH_SIZE"] if enqueue_limit is None else enqueue_limit
        )
        normalized_worker_count = namespace["DEFAULT_CRON_QUEUE_WORKERS"] if worker_count is None else worker_count
        return namespace["_cron_scheduler_tick_bridge"](
            enqueue_limit=normalized_enqueue_limit,
            worker_count=normalized_worker_count,
            enqueue_due_cron_jobs_impl_fn=lambda normalized_limit: namespace["_enqueue_due_cron_jobs_impl"](
                limit=normalized_limit
            ),
            safe_cron_queue_depth=namespace["_safe_cron_queue_depth"],
            modal_module=namespace.get("modal"),
            process_cron_queue_spawn=getattr(namespace.get("process_cron_queue"), "spawn", None),
            default_cron_queue_max_jobs_per_worker=namespace["DEFAULT_CRON_QUEUE_MAX_JOBS_PER_WORKER"],
        )

    namespace["_cron_scheduler_tick_impl"] = _cron_scheduler_tick_impl
    namespace["_feishu_model_registry_heartbeat_impl"] = bind_sync(
        namespace,
        namespace["_feishu_registry_heartbeat_impl"],
        dynamic={
            "settings_from_env": _settings_from_env,
            "load_feishu_sync_state": "_load_feishu_sync_state",
            "run_feishu_registry_sync_cycle": "_run_feishu_registry_sync_cycle",
        },
    )

    def _maintenance_heartbeat_impl(
        *,
        enqueue_limit: int | None = None,
        worker_count: int | None = None,
    ) -> dict[str, Any]:
        normalized_enqueue_limit = (
            namespace["DEFAULT_CRON_QUEUE_BATCH_SIZE"] if enqueue_limit is None else enqueue_limit
        )
        normalized_worker_count = namespace["DEFAULT_CRON_QUEUE_WORKERS"] if worker_count is None else worker_count
        return namespace["_cron_maintenance_heartbeat_bridge"](
            enqueue_limit=normalized_enqueue_limit,
            worker_count=normalized_worker_count,
            cron_scheduler_tick_impl_fn=lambda normalized_limit, normalized_workers: namespace[
                "_cron_scheduler_tick_impl"
            ](
                enqueue_limit=normalized_limit,
                worker_count=normalized_workers,
            ),
            feishu_model_registry_heartbeat_impl_fn=namespace["_feishu_model_registry_heartbeat_impl"],
        )

    namespace["_maintenance_heartbeat_impl"] = _maintenance_heartbeat_impl
