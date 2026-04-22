from __future__ import annotations

from internal.model_catalog import (
    DEFAULT_OPENROUTER_FREE_MODELS,
    dedupe_keep_order,
    extract_openrouter_free_model_candidates,
    refresh_free_model_routes,
    score_free_model,
)


class _LoggerStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str, *args: object) -> None:
        if args:
            message = message % args
        self.messages.append(message)


def test_extract_openrouter_free_model_candidates_prefers_concrete_models_before_alias() -> None:
    payload = {
        "data": [
            {
                "id": "openrouter/free",
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "google/gemma-3-27b-it:free",
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "qwen/qwen3-coder:free",
                "pricing": {"prompt": "0", "completion": "0"},
            },
        ]
    }

    candidates = extract_openrouter_free_model_candidates(
        payload,
        dedupe_keep_order_fn=dedupe_keep_order,
        score_free_model_fn=score_free_model,
    )

    assert candidates[:2] == ["qwen/qwen3-coder:free", "google/gemma-3-27b-it:free"]
    assert candidates[-1] == "openrouter/free"


def test_refresh_free_model_routes_uses_concrete_openrouter_fallbacks_when_refresh_fails() -> None:
    logger = _LoggerStub()
    saved: dict[str, object] = {}

    def _raise_fetch() -> list[str]:
        raise RuntimeError("boom")

    result = refresh_free_model_routes(
        force=True,
        ensure_runtime_dirs=lambda: None,
        load_routing_state=lambda: {},
        save_routing_state=lambda payload: saved.update(payload),
        resolve_provider_runtime_binding_fn=lambda provider_name: {
            "provider": provider_name,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "api_mode": "chat_completions",
        }
        if provider_name == "openrouter"
        else None,
        fetch_openrouter_free_model_candidates_fn=_raise_fetch,
        load_nvidia_free_model_candidates_fn=lambda: [],
        routing_refresh_ttl_seconds=60,
        default_openrouter_base_url="https://openrouter.ai/api/v1",
        default_nvidia_base_url="https://integrate.api.nvidia.com/v1",
        logger=logger,
    )

    assert result["providers"]["openrouter"]["candidates"] == list(DEFAULT_OPENROUTER_FREE_MODELS)
    assert saved["providers"]["openrouter"]["candidates"] == list(DEFAULT_OPENROUTER_FREE_MODELS)
    assert logger.messages
