from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Callable


def validate_tavily_integration(
    *,
    prepare_runtime_environment: Callable[[], None],
    safe_json_loads: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    prepare_runtime_environment()

    from tools import web_tools

    backend = web_tools._get_backend()
    backend_available = web_tools._is_backend_available(backend)

    search_payload = safe_json_loads(
        web_tools.web_search_tool("Tavily AI official website", limit=3),
        {"success": False, "error": "invalid_json"},
    )
    extract_payload = safe_json_loads(
        asyncio.run(
            web_tools.web_extract_tool(
                ["https://tavily.com/"],
                use_llm_processing=False,
            )
        ),
        {"success": False, "error": "invalid_json"},
    )
    crawl_payload = safe_json_loads(
        asyncio.run(
            web_tools.web_crawl_tool(
                "https://tavily.com/",
                "Find basic site information",
                use_llm_processing=False,
            )
        ),
        {"success": False, "error": "invalid_json"},
    )

    search_results = (
        ((search_payload.get("data") or {}).get("web") or [])
        if isinstance(search_payload, dict)
        else []
    )
    extract_results = (
        (extract_payload.get("results") or [])
        if isinstance(extract_payload, dict)
        else []
    )
    crawl_results = (
        (crawl_payload.get("results") or [])
        if isinstance(crawl_payload, dict)
        else []
    )

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


def probe_provider_request_metadata(
    *,
    provider_name: str,
    model_name: str = "",
    model: str = "",
    prompt: str = "ok",
    max_tokens: int = 64,
    prepare_runtime_environment: Callable[[], None],
    resolve_provider_runtime_binding: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any]:
    prepare_runtime_environment()
    normalized_provider = str(provider_name or "").strip().lower()
    normalized_model = str(model_name or model or "").strip()
    if not normalized_provider:
        raise ValueError("provider_name is required")
    if not normalized_model:
        raise ValueError("model_name or model is required")

    runtime_binding = resolve_provider_runtime_binding(normalized_provider)
    if runtime_binding is None:
        raise RuntimeError(f"Missing runtime binding for provider {normalized_provider}")

    api_key = str(runtime_binding.get("api_key") or "").strip()
    base_url = str(runtime_binding.get("base_url") or "").strip()
    resolved_provider = (
        str(runtime_binding.get("provider") or normalized_provider).strip().lower()
        or normalized_provider
    )

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


def approve_pairing(
    platform: str,
    code: str,
    *,
    prepare_runtime_environment: Callable[[], None],
) -> dict[str, Any]:
    prepare_runtime_environment()

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
