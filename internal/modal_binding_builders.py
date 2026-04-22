from __future__ import annotations

from typing import Any, Mapping


def _runtime(runtime: Mapping[str, Any], name: str) -> Any:
    return runtime[name]


def build_modal_core_export_deps(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "include_probe_provider_request_metadata": False,
        "run_agent_task_impl": _runtime(runtime, "_run_agent_task_impl"),
        "probe_provider_request_metadata_impl": _runtime(runtime, "_probe_provider_request_metadata_impl"),
        "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB": _runtime(
            runtime,
            "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB",
        ),
        "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU": _runtime(
            runtime,
            "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU",
        ),
        "prepare_runtime_environment": _runtime(runtime, "_prepare_runtime_environment"),
        "settings_from_env": _runtime(runtime, "RuntimeSettings").from_env,
        "get_camofox_url": _runtime(runtime, "_get_camofox_url"),
        "is_local_camofox_url": _runtime(runtime, "_is_local_camofox_url"),
        "get_memory_provider_status": _runtime(runtime, "_get_memory_provider_status"),
        "load_gateway_import_status": _runtime(runtime, "_load_gateway_import_status"),
        "get_telegram_webhook_status": _runtime(runtime, "_get_telegram_webhook_status"),
        "build_gateway_health_payload": _runtime(runtime, "_build_gateway_health_payload"),
        "app_name": _runtime(runtime, "APP_NAME"),
        "default_chat_queue_name": _runtime(runtime, "DEFAULT_CHAT_QUEUE_NAME"),
        "safe_chat_queue_depth": _runtime(runtime, "_safe_chat_queue_depth"),
        "cron_status_impl": _runtime(runtime, "_cron_status_impl"),
        "build_model_routing_debug_state": _runtime(runtime, "_build_model_routing_debug_state"),
        "build_feishu_sync_state_debug_state": _runtime(runtime, "_build_feishu_sync_state_debug_state"),
        "sync_runtime_config": _runtime(runtime, "_sync_runtime_config"),
        "serialize_settings_for_log": _runtime(runtime, "_serialize_settings_for_log"),
        "data_root": _runtime(runtime, "DATA_ROOT"),
        "is_camofox_healthcheck_ready": _runtime(runtime, "_is_camofox_healthcheck_ready"),
        "build_modal_official_parity_state": _runtime(runtime, "_build_modal_official_parity_state"),
        "load_json_file": _runtime(runtime, "_load_json_file"),
    }


def build_modal_worker_registration_deps(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB": _runtime(
            runtime,
            "DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB",
        ),
        "DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU": _runtime(
            runtime,
            "DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU",
        ),
        "DEFAULT_FEISHU_ACK_REACTION_SCALEDOWN_WINDOW_SECONDS": _runtime(
            runtime,
            "DEFAULT_FEISHU_ACK_REACTION_SCALEDOWN_WINDOW_SECONDS",
        ),
        "FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED": _runtime(
            runtime,
            "FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED",
        ),
        "prepare_worker_snapshot_context": _runtime(runtime, "_prepare_worker_snapshot_context"),
        "bootstrap_feishu_ack_reaction_worker_context": _runtime(
            runtime,
            "_bootstrap_feishu_ack_reaction_worker_context",
        ),
        "add_feishu_ack_reaction_from_payload": _runtime(
            runtime,
            "_add_feishu_ack_reaction_from_payload",
        ),
        "DEFAULT_FEISHU_INGRESS_WORKER_MEMORY_MB": _runtime(
            runtime,
            "DEFAULT_FEISHU_INGRESS_WORKER_MEMORY_MB",
        ),
        "DEFAULT_FEISHU_INGRESS_WORKER_CPU": _runtime(runtime, "DEFAULT_FEISHU_INGRESS_WORKER_CPU"),
        "DEFAULT_FEISHU_INGRESS_SCALEDOWN_WINDOW_SECONDS": _runtime(
            runtime,
            "DEFAULT_FEISHU_INGRESS_SCALEDOWN_WINDOW_SECONDS",
        ),
        "FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED": _runtime(
            runtime,
            "FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED",
        ),
        "bootstrap_feishu_ingress_worker_context": _runtime(
            runtime,
            "_bootstrap_feishu_ingress_worker_context",
        ),
        "extract_feishu_warmup_context": _runtime(runtime, "_extract_feishu_warmup_context"),
        "spawn_chat_queue_worker_sync": _runtime(runtime, "_spawn_chat_queue_worker_sync"),
        "DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS": _runtime(
            runtime,
            "DEFAULT_CHAT_QUEUE_WARMUP_WAIT_SECONDS",
        ),
        "extract_feishu_queue_context": _runtime(runtime, "_extract_feishu_queue_context"),
        "enqueue_chat_event": _runtime(runtime, "_enqueue_chat_event"),
        "append_feishu_trace": _runtime(runtime, "_append_feishu_trace"),
        "extract_feishu_event_metadata": _runtime(runtime, "_extract_feishu_event_metadata"),
        "logger": _runtime(runtime, "logger"),
        "DEFAULT_CHAT_QUEUE_BATCH_SIZE": _runtime(runtime, "DEFAULT_CHAT_QUEUE_BATCH_SIZE"),
        "DEFAULT_FEISHU_INLINE_WORKER_HARD_BUDGET_MS": _runtime(
            runtime,
            "DEFAULT_FEISHU_INLINE_WORKER_HARD_BUDGET_MS",
        ),
        "bootstrap_chat_queue_worker_context": _runtime(
            runtime,
            "_bootstrap_chat_queue_worker_context",
        ),
        "spawn_feishu_background_exec_async": _runtime(
            runtime,
            "_spawn_feishu_background_exec_async",
        ),
        "DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS": _runtime(
            runtime,
            "DEFAULT_FEISHU_CHAT_WORKER_TIMEOUT_SECONDS",
        ),
        "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB": _runtime(
            runtime,
            "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB",
        ),
        "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU": _runtime(
            runtime,
            "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU",
        ),
        "DEFAULT_FEISHU_BACKGROUND_EXEC_SCALEDOWN_WINDOW_SECONDS": _runtime(
            runtime,
            "DEFAULT_FEISHU_BACKGROUND_EXEC_SCALEDOWN_WINDOW_SECONDS",
        ),
        "dispatch_feishu_payload": _runtime(runtime, "_dispatch_feishu_payload"),
        "normalize_phase_timings": _runtime(runtime, "_normalize_phase_timings"),
        "DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB": _runtime(runtime, "DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB"),
        "DEFAULT_CHAT_QUEUE_WORKER_CPU": _runtime(runtime, "DEFAULT_CHAT_QUEUE_WORKER_CPU"),
        "DEFAULT_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS": _runtime(
            runtime,
            "DEFAULT_CHAT_QUEUE_SCALEDOWN_WINDOW_SECONDS",
        ),
        "CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED": _runtime(runtime, "CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED"),
        "process_chat_queue_impl": _runtime(runtime, "_process_chat_queue_impl"),
    }


def build_modal_debug_export_deps(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prepare_runtime_environment": _runtime(runtime, "_prepare_runtime_environment"),
        "settings_from_env": _runtime(runtime, "RuntimeSettings").from_env,
        "serialize_settings_for_log": _runtime(runtime, "_serialize_settings_for_log"),
        "normalize_public_https_url": _runtime(runtime, "_normalize_public_https_url"),
        "split_csv": _runtime(runtime, "_split_csv"),
        "is_truthy": _runtime(runtime, "_is_truthy"),
        "build_feishu_menu_manifest": _runtime(runtime, "_build_feishu_menu_manifest"),
        "default_chat_queue_name": _runtime(runtime, "DEFAULT_CHAT_QUEUE_NAME"),
        "safe_chat_queue_depth": _runtime(runtime, "_safe_chat_queue_depth"),
        "default_feishu_message_ingress_strategy": _runtime(
            runtime,
            "DEFAULT_FEISHU_MESSAGE_INGRESS_STRATEGY",
        ),
        "resolve_feishu_message_ingress_strategy": _runtime(
            runtime,
            "_resolve_feishu_message_ingress_strategy",
        ),
        "sample_light_p2p_feishu_message_payload": _runtime(
            runtime,
            "_sample_light_p2p_feishu_message_payload",
        ),
        "legacy_feishu_message_ingress_aliases": _runtime(
            runtime,
            "LEGACY_FEISHU_MESSAGE_INGRESS_ALIASES",
        ),
        "supported_feishu_message_ingress_strategies": _runtime(
            runtime,
            "SUPPORTED_FEISHU_MESSAGE_INGRESS_STRATEGIES",
        ),
        "default_feishu_ingress_handoff_timeout_seconds": _runtime(
            runtime,
            "DEFAULT_FEISHU_INGRESS_HANDOFF_TIMEOUT_SECONDS",
        ),
        "default_feishu_ack_reaction_mode": _runtime(runtime, "DEFAULT_FEISHU_ACK_REACTION_MODE"),
        "supported_feishu_ack_reaction_modes": _runtime(runtime, "SUPPORTED_FEISHU_ACK_REACTION_MODES"),
        "default_feishu_ack_reaction_inline_budget_ms": _runtime(
            runtime,
            "DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS",
        ),
        "default_feishu_ack_reaction_request_timeout_seconds": _runtime(
            runtime,
            "DEFAULT_FEISHU_ACK_REACTION_REQUEST_TIMEOUT_SECONDS",
        ),
        "default_feishu_inline_first_response_budget_ms": _runtime(
            runtime,
            "DEFAULT_FEISHU_INLINE_FIRST_RESPONSE_BUDGET_MS",
        ),
        "default_feishu_provider_first_token_budget_ms": _runtime(
            runtime,
            "DEFAULT_FEISHU_PROVIDER_FIRST_TOKEN_BUDGET_MS",
        ),
        "default_feishu_inline_worker_hard_budget_ms": _runtime(
            runtime,
            "DEFAULT_FEISHU_INLINE_WORKER_HARD_BUDGET_MS",
        ),
        "default_feishu_background_exec_worker_cpu": _runtime(
            runtime,
            "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU",
        ),
        "default_feishu_background_exec_worker_memory_mb": _runtime(
            runtime,
            "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB",
        ),
        "default_chat_queue_worker_cpu": _runtime(runtime, "DEFAULT_CHAT_QUEUE_WORKER_CPU"),
        "default_chat_queue_worker_memory_mb": _runtime(runtime, "DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB"),
        "web_app_memory_snapshot_enabled": _runtime(runtime, "WEB_APP_MEMORY_SNAPSHOT_ENABLED"),
        "feishu_ingress_memory_snapshot_enabled": _runtime(
            runtime,
            "FEISHU_INGRESS_MEMORY_SNAPSHOT_ENABLED",
        ),
        "feishu_ack_reaction_memory_snapshot_enabled": _runtime(
            runtime,
            "FEISHU_ACK_REACTION_MEMORY_SNAPSHOT_ENABLED",
        ),
        "chat_queue_memory_snapshot_enabled": _runtime(runtime, "CHAT_QUEUE_MEMORY_SNAPSHOT_ENABLED"),
        "build_feishu_snapshot_profile_state": _runtime(runtime, "_build_feishu_snapshot_profile_state"),
        "build_feishu_capabilities_debug_state": _runtime(
            runtime,
            "_build_feishu_capabilities_debug_state",
        ),
        "get_feishu_gateway_runtime": _runtime(runtime, "_get_feishu_gateway_runtime"),
        "build_runtime_bootstrap_debug_state": _runtime(
            runtime,
            "_build_runtime_bootstrap_debug_state",
        ),
        "build_feishu_model_registry_debug_state": _runtime(
            runtime,
            "_build_feishu_model_registry_debug_state",
        ),
        "build_feishu_sync_state_debug_state": _runtime(
            runtime,
            "_build_feishu_sync_state_debug_state",
        ),
        "sync_feishu_model_registry_impl": _runtime(runtime, "_sync_feishu_model_registry_impl"),
        "validate_feishu_native_delivery_impl": _runtime(
            runtime,
            "_validate_feishu_native_delivery_impl",
        ),
        "prepare_feishu_model_registry_bitable_impl": _runtime(
            runtime,
            "_prepare_feishu_model_registry_bitable_impl",
        ),
        "build_model_routing_debug_state": _runtime(runtime, "_build_model_routing_debug_state"),
        "build_modal_official_parity_state": _runtime(runtime, "_build_modal_official_parity_state"),
        "read_feishu_trace": _runtime(runtime, "_read_feishu_trace"),
        "feishu_trace_path": _runtime(runtime, "FEISHU_TRACE_PATH"),
        "build_feishu_ingress_strategy_debug_state": _runtime(
            runtime,
            "_build_feishu_ingress_strategy_debug_state",
        ),
        "build_feishu_perf_summary_from_rows": _runtime(
            runtime,
            "_build_feishu_perf_summary_from_rows",
        ),
        "debug_session_route_state": _runtime(runtime, "_debug_session_route_state"),
        "debug_gateway_session_state": _runtime(runtime, "_debug_gateway_session_state"),
        "validate_feishu_webhook_impl": _runtime(runtime, "_validate_feishu_webhook_impl"),
        "validate_feishu_message_ingress_impl": _runtime(
            runtime,
            "_validate_feishu_message_ingress_impl",
        ),
        "send_custom_feishu_message_ingress_impl": _runtime(
            runtime,
            "_send_custom_feishu_message_ingress_impl",
        ),
        "is_valid_telegram_bot_token_format": _runtime(
            runtime,
            "_is_valid_telegram_bot_token_format",
        ),
        "resolve_telegram_webhook_url": _runtime(runtime, "_resolve_telegram_webhook_url"),
        "runtime_settings_cls": _runtime(runtime, "RuntimeSettings"),
        "get_telegram_webhook_status": _runtime(runtime, "_get_telegram_webhook_status"),
        "approve_pairing_impl": _runtime(runtime, "_approve_pairing_impl"),
        "validate_tavily_integration_impl": _runtime(runtime, "_validate_tavily_integration_impl"),
        "cron_status_impl": _runtime(runtime, "_cron_status_impl"),
        "process_cron_queue_impl": _runtime(runtime, "_process_cron_queue_impl"),
        "default_cron_queue_batch_size": _runtime(runtime, "DEFAULT_CRON_QUEUE_BATCH_SIZE"),
        "default_cron_queue_workers": _runtime(runtime, "DEFAULT_CRON_QUEUE_WORKERS"),
        "cron_scheduler_tick_impl": _runtime(runtime, "_cron_scheduler_tick_impl"),
        "maintenance_heartbeat_impl": _runtime(runtime, "_maintenance_heartbeat_impl"),
        "feishu_model_registry_heartbeat_impl": _runtime(
            runtime,
            "_feishu_model_registry_heartbeat_impl",
        ),
        "default_web_app_scaledown_window_seconds": _runtime(
            runtime,
            "DEFAULT_WEB_APP_SCALEDOWN_WINDOW_SECONDS",
        ),
        "create_web_app": _runtime(runtime, "create_web_app"),
        "app_name": _runtime(runtime, "APP_NAME"),
        "default_secret_name": _runtime(runtime, "DEFAULT_SECRET_NAME"),
        "default_volume_name": _runtime(runtime, "DEFAULT_VOLUME_NAME"),
    }


def build_web_app_factory_args(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fastapi_cls": _runtime(runtime, "FastAPI"),
        "lifespan_builder": _runtime(runtime, "_web_app_build_lifespan"),
        "startup_tasks_runner": _runtime(runtime, "_web_app_run_startup_tasks"),
        "feishu_webhook_route_deps_cls": _runtime(runtime, "_FeishuWebhookRouteDeps"),
        "register_health_route": _runtime(runtime, "_web_app_register_health_route"),
        "register_invoke_route": _runtime(runtime, "_web_app_register_invoke_route"),
        "register_internal_feishu_routes": _runtime(runtime, "_web_app_register_internal_feishu_routes"),
        "register_telegram_route": _runtime(runtime, "_web_app_register_telegram_route"),
        "register_feishu_route": _runtime(runtime, "_web_app_register_feishu_route"),
        "register_qq_route": _runtime(runtime, "_web_app_register_qq_route"),
        "app_name": _runtime(runtime, "APP_NAME"),
        "default_chat_queue_name": _runtime(runtime, "DEFAULT_CHAT_QUEUE_NAME"),
        "default_chat_queue_batch_size": _runtime(runtime, "DEFAULT_CHAT_QUEUE_BATCH_SIZE"),
        "default_feishu_ack_reaction_mode": _runtime(runtime, "DEFAULT_FEISHU_ACK_REACTION_MODE"),
        "supported_feishu_ack_reaction_modes": _runtime(runtime, "SUPPORTED_FEISHU_ACK_REACTION_MODES"),
        "default_feishu_ack_reaction_inline_budget_ms": _runtime(
            runtime,
            "DEFAULT_FEISHU_ACK_REACTION_INLINE_BUDGET_MS",
        ),
        "default_feishu_startup_schema_recheck_seconds": _runtime(
            runtime,
            "DEFAULT_FEISHU_STARTUP_SCHEMA_RECHECK_SECONDS",
        ),
        "prepare_runtime_environment": _runtime(runtime, "_prepare_runtime_environment"),
        "prewarm_chat_queue_async": _runtime(runtime, "_prewarm_chat_queue_async"),
        "settings_from_env": _runtime(runtime, "RuntimeSettings").from_env,
        "maybe_sync_telegram_webhook": _runtime(runtime, "_maybe_sync_telegram_webhook"),
        "load_feishu_sync_state": _runtime(runtime, "_load_feishu_sync_state"),
        "should_prepare_feishu_registry_schema_on_startup": _runtime(
            runtime,
            "_should_prepare_feishu_registry_schema_on_startup",
        ),
        "prepare_feishu_model_registry_bitable_impl": _runtime(
            runtime,
            "_prepare_feishu_model_registry_bitable_impl",
        ),
        "get_memory_provider_status": _runtime(runtime, "_get_memory_provider_status"),
        "safe_chat_queue_depth_async": _runtime(runtime, "_safe_chat_queue_depth_async"),
        "cron_status_impl": _runtime(runtime, "_cron_status_impl"),
        "get_telegram_webhook_status": _runtime(runtime, "_get_telegram_webhook_status"),
        "build_model_routing_debug_state": _runtime(runtime, "_build_model_routing_debug_state"),
        "build_feishu_sync_state_debug_state": _runtime(
            runtime,
            "_build_feishu_sync_state_debug_state",
        ),
        "sync_runtime_config": _runtime(runtime, "_sync_runtime_config"),
        "serialize_settings_for_log": _runtime(runtime, "_serialize_settings_for_log"),
        "validate_bearer_token": _runtime(runtime, "_validate_bearer_token"),
        "validate_feishu_internal_bearer_token": _runtime(runtime, "_validate_feishu_internal_bearer_token"),
        "run_agent_task_impl": _runtime(runtime, "_run_agent_task_impl"),
        "handle_internal_json_route": _runtime(runtime, "_feishu_handle_internal_json_route"),
        "authorize_internal_file_route": _runtime(runtime, "_feishu_authorize_internal_file_route"),
        "log_feishu_internal_auth_failure": _runtime(runtime, "_log_feishu_internal_auth_failure"),
        "extract_feishu_internal_request_meta": _runtime(
            runtime,
            "_extract_feishu_internal_request_meta",
        ),
        "run_feishu_internal_agent_exec": _runtime(runtime, "_run_feishu_internal_agent_exec"),
        "run_feishu_internal_agent_plan": _runtime(runtime, "_run_feishu_internal_agent_plan"),
        "run_feishu_internal_control": _runtime(runtime, "_run_feishu_internal_control"),
        "lookup_feishu_internal_result_file": _runtime(
            runtime,
            "_lookup_feishu_internal_result_file",
        ),
        "file_response_cls": _runtime(runtime, "FileResponse"),
        "validate_telegram_secret": _runtime(runtime, "_validate_telegram_secret"),
        "mark_update_seen": _runtime(runtime, "_mark_update_seen"),
        "extract_telegram_inline_fast_command": _runtime(
            runtime,
            "_extract_telegram_inline_fast_command",
        ),
        "dispatch_telegram_update": _runtime(runtime, "_dispatch_telegram_update"),
        "send_telegram_message": _runtime(runtime, "_send_telegram_message"),
        "extract_telegram_queue_context": _runtime(runtime, "_extract_telegram_queue_context"),
        "enqueue_chat_event_async": _runtime(runtime, "_enqueue_chat_event_async"),
        "spawn_chat_queue_worker_async": _runtime(runtime, "_spawn_chat_queue_worker_async"),
        "feishu_handle_webhook_route": _runtime(runtime, "_feishu_handle_webhook_route"),
        "try_handle_feishu_verification_fast": _runtime(
            runtime,
            "_try_handle_feishu_verification_fast",
        ),
        "parse_feishu_webhook_request": _runtime(runtime, "_parse_feishu_webhook_request"),
        "capture_phase_elapsed": _runtime(runtime, "_capture_phase_elapsed"),
        "append_feishu_trace": _runtime(runtime, "_append_feishu_trace"),
        "extract_feishu_event_metadata": _runtime(runtime, "_extract_feishu_event_metadata"),
        "mark_feishu_event_seen": _runtime(runtime, "_mark_feishu_event_seen"),
        "build_feishu_webhook_ack_response": _runtime(
            runtime,
            "_build_feishu_webhook_ack_response",
        ),
        "extract_feishu_trace_token": _runtime(runtime, "_extract_feishu_trace_token"),
        "is_feishu_session_warmup_event": _runtime(runtime, "_is_feishu_session_warmup_event"),
        "extract_feishu_warmup_context": _runtime(runtime, "_extract_feishu_warmup_context"),
        "spawn_feishu_ingress_warmup_async": _runtime(runtime, "_spawn_feishu_ingress_warmup_async"),
        "extract_feishu_message_read_event_info": _runtime(
            runtime,
            "_extract_feishu_message_read_event_info",
        ),
        "with_feishu_internal_meta": _runtime(runtime, "_with_feishu_internal_meta"),
        "add_feishu_ack_reaction_inline_async": _runtime(
            runtime,
            "_add_feishu_ack_reaction_inline_async",
        ),
        "spawn_feishu_ack_reaction_async": _runtime(runtime, "_spawn_feishu_ack_reaction_async"),
        "extract_feishu_inline_fast_command": _runtime(runtime, "_extract_feishu_inline_fast_command"),
        "dispatch_feishu_payload": _runtime(runtime, "_dispatch_feishu_payload"),
        "send_feishu_local_registry_menu_card": _runtime(
            runtime,
            "_send_feishu_local_registry_menu_card",
        ),
        "extract_feishu_card_action_name": _runtime(
            runtime,
            "_extract_feishu_card_action_name",
        ),
        "close_feishu_card_from_payload": _runtime(runtime, "_close_feishu_card_from_payload"),
        "enqueue_feishu_card_action_for_background": _runtime(
            runtime,
            "_enqueue_feishu_card_action_for_background",
        ),
        "build_feishu_card_action_ack_payload": _runtime(
            runtime,
            "_build_feishu_card_action_ack_payload",
        ),
        "should_inline_feishu_control_event": _runtime(
            runtime,
            "_should_inline_feishu_control_event",
        ),
        "extract_feishu_queue_context": _runtime(runtime, "_extract_feishu_queue_context"),
        "resolve_feishu_message_ingress_strategy": _runtime(
            runtime,
            "_resolve_feishu_message_ingress_strategy",
        ),
        "spawn_feishu_event_handoff_async": _runtime(
            runtime,
            "_spawn_feishu_event_handoff_async",
        ),
        "spawn_feishu_message_inline_async": _runtime(
            runtime,
            "_spawn_feishu_message_inline_async",
        ),
        "spawn_chat_queue_worker_optimistic_async": _runtime(
            runtime,
            "_spawn_chat_queue_worker_optimistic_async",
        ),
        "dispatch_qq_update": _runtime(runtime, "_dispatch_qq_update"),
        "http_exception_cls": _runtime(runtime, "HTTPException"),
        "json_response_cls": _runtime(runtime, "JSONResponse"),
        "logger": _runtime(runtime, "logger"),
    }
