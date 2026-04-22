from __future__ import annotations

import os
import re
import time
from typing import Any, Callable

DEFAULT_OPENROUTER_FREE_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-nano-9b-v2:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "openrouter/free",
)
DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/"
    "d1215a30b84b673ef0367010b0e78c10/affiliate-manager"
)
GATEWAY_ENFORCED_PROVIDERS = frozenset({"openrouter", "nvidia"})


def normalize_cloudflare_ai_gateway_base_url(raw_url: str | None) -> str:
    base_url = str(raw_url or "").strip().rstrip("/")
    if not base_url:
        return ""
    lower = base_url.lower()
    if lower.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
        lower = base_url.lower()
    if lower.endswith("/v1/chat/completions"):
        base_url = base_url[: -len("/v1/chat/completions")]
        lower = base_url.lower()
    if not lower.endswith("/compat"):
        base_url = f"{base_url}/compat"
    return base_url.rstrip("/")


def force_cloudflare_gateway_runtime_binding(provider_name: str, runtime: dict[str, Any]) -> dict[str, Any]:
    normalized_provider = str(provider_name or "").strip().lower()
    if normalized_provider not in GATEWAY_ENFORCED_PROVIDERS:
        return runtime
    api_mode = str(runtime.get("api_mode") or "chat_completions").strip().lower() or "chat_completions"
    if api_mode != "chat_completions":
        return runtime
    gateway_base_url = normalize_cloudflare_ai_gateway_base_url(
        os.getenv("CLOUDFLARE_AI_GATEWAY_BASE_URL", DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL)
    )
    if not gateway_base_url:
        return runtime
    forced = dict(runtime)
    forced["provider"] = normalized_provider
    forced["base_url"] = gateway_base_url
    return forced


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def score_free_model(model_id: str) -> tuple[int, int, int, str]:
    normalized = model_id.lower()
    family_rank = 0
    if "qwen3-coder" in normalized:
        family_rank = 100
    elif "qwq" in normalized:
        family_rank = 95
    elif "deepseek" in normalized and "r1" in normalized:
        family_rank = 92
    elif "glm-4.7" in normalized:
        family_rank = 90
    elif "llama-4" in normalized:
        family_rank = 88
    elif "llama-3.3" in normalized:
        family_rank = 86
    elif "llama-3.1-70b" in normalized:
        family_rank = 84
    elif "70b" in normalized:
        family_rank = 80
    elif "32b" in normalized:
        family_rank = 70
    elif "27b" in normalized:
        family_rank = 68
    elif "14b" in normalized:
        family_rank = 60
    elif "8b" in normalized:
        family_rank = 50

    parameter_rank = 0
    for size, rank in (
        ("405b", 405),
        ("480b", 400),
        ("236b", 236),
        ("120b", 120),
        ("90b", 90),
        ("70b", 70),
        ("49b", 49),
        ("32b", 32),
        ("27b", 27),
        ("14b", 14),
        ("8b", 8),
    ):
        if size in normalized:
            parameter_rank = rank
            break

    explicit_free_bonus = 1 if normalized.endswith(":free") else 0
    return (family_rank, parameter_rank, explicit_free_bonus, model_id)


def extract_openrouter_free_model_candidates(
    payload: dict[str, Any],
    *,
    dedupe_keep_order_fn: Callable[[list[str]], list[str]],
    score_free_model_fn: Callable[[str], tuple[int, int, int, str]],
) -> list[str]:
    concrete_candidates: list[str] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        pricing = item.get("pricing") or {}
        prompt_price = str(pricing.get("prompt") or "").strip()
        completion_price = str(pricing.get("completion") or "").strip()
        if model_id.endswith(":free") or (
            prompt_price in {"0", "0.0", "0.000000"} and completion_price in {"0", "0.0", "0.000000"}
        ):
            if model_id.lower() == "openrouter/free":
                continue
            concrete_candidates.append(model_id)
    ranked = sorted(dedupe_keep_order_fn(concrete_candidates), key=score_free_model_fn, reverse=True)
    if ranked:
        ranked.append("openrouter/free")
        return ranked
    return ["openrouter/free"]


def fetch_openrouter_free_model_candidates(
    *,
    default_openrouter_base_url: str,
    extract_openrouter_free_model_candidates_fn: Callable[[dict[str, Any]], list[str]],
) -> list[str]:
    import httpx

    headers = {}
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with httpx.Client(timeout=20) as client:
        response = client.get(f"{default_openrouter_base_url}/models", headers=headers)
        response.raise_for_status()
        payload = response.json()
    return extract_openrouter_free_model_candidates_fn(payload if isinstance(payload, dict) else {})


def expand_nvidia_model_variants(model_id: str, *, dedupe_keep_order_fn: Callable[[list[str]], list[str]]) -> list[str]:
    normalized = str(model_id or "").strip().lower()
    if not normalized or "/" not in normalized:
        return []
    variants = [normalized]
    dotted = re.sub(r"(\d)_(\d)", r"\1.\2", normalized)
    if dotted != normalized:
        variants.append(dotted)
    return dedupe_keep_order_fn(variants)


def is_probable_nvidia_chat_model(
    model_id: str,
    *,
    non_chat_markers: tuple[str, ...],
    chat_family_markers: tuple[str, ...],
) -> bool:
    normalized = str(model_id or "").strip().lower()
    if not normalized or "/" not in normalized:
        return False
    if any(marker in normalized for marker in non_chat_markers):
        return False
    return any(marker in normalized for marker in chat_family_markers)


def extract_nvidia_popular_model_candidates(
    html: str,
    *,
    is_probable_nvidia_chat_model_fn: Callable[[str], bool],
    expand_nvidia_model_variants_fn: Callable[[str], list[str]],
    dedupe_keep_order_fn: Callable[[list[str]], list[str]],
) -> list[str]:
    pairs = re.findall(r'href="/([a-z0-9_.-]+)/([a-z0-9_.-]+)"', html, flags=re.I)
    candidates: list[str] = []
    for publisher, slug in pairs:
        publisher = str(publisher or "").strip().lower()
        slug = str(slug or "").strip().lower()
        if not publisher or not slug:
            continue
        if publisher in {"models", "explore", "blueprints", "settings", "docs"}:
            continue
        if slug in {"models", "explore", "blueprints", "settings", "docs"}:
            continue
        candidate = f"{publisher}/{slug}"
        if not is_probable_nvidia_chat_model_fn(candidate):
            continue
        candidates.extend(expand_nvidia_model_variants_fn(candidate))
    return dedupe_keep_order_fn(candidates)


def fetch_nvidia_popular_model_candidates(
    *,
    default_nvidia_popular_models_url: str,
    extract_nvidia_popular_model_candidates_fn: Callable[[str], list[str]],
) -> list[str]:
    import httpx

    url = os.getenv("NVIDIA_POPULAR_MODELS_URL") or default_nvidia_popular_models_url
    with httpx.Client(timeout=20) as client:
        response = client.get(url)
        response.raise_for_status()
        html = response.text
    candidates = extract_nvidia_popular_model_candidates_fn(html)
    try:
        limit = int(os.getenv("NVIDIA_POPULAR_MODELS_LIMIT", "48") or "48")
    except ValueError:
        limit = 48
    if limit > 0:
        candidates = candidates[:limit]
    return candidates


def load_nvidia_free_model_candidates(
    *,
    split_csv: Callable[[str | None], list[str]],
    dedupe_keep_order_fn: Callable[[list[str]], list[str]],
    is_truthy: Callable[[str | None, bool], bool],
    fetch_nvidia_popular_model_candidates_fn: Callable[[], list[str]],
    default_nvidia_free_models: tuple[str, ...],
    logger: Any,
) -> list[str]:
    configured = dedupe_keep_order_fn(split_csv(os.getenv("NVIDIA_FREE_MODELS")))
    if configured:
        return configured
    candidates = list(default_nvidia_free_models)
    enable_popular = is_truthy(os.getenv("NVIDIA_POPULAR_MODELS_ENABLED"), default=True)
    if not enable_popular:
        return candidates
    try:
        popular = fetch_nvidia_popular_model_candidates_fn()
    except Exception as exc:
        logger.warning("Failed refreshing NVIDIA popular models: %s", exc)
        return candidates
    return dedupe_keep_order_fn(popular + candidates)


def resolve_provider_runtime_binding(provider_name: str) -> dict[str, Any] | None:
    normalized = str(provider_name or "").strip().lower()
    if not normalized:
        return None
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=normalized) or {}
        runtime = force_cloudflare_gateway_runtime_binding(normalized, runtime)
    except Exception:
        return None

    base_url = str(runtime.get("base_url") or "").strip()
    api_key = str(runtime.get("api_key") or "").strip()
    resolved_provider = str(runtime.get("provider") or normalized).strip().lower() or normalized
    if not base_url or not api_key:
        return None
    return {
        "provider": resolved_provider,
        "base_url": base_url,
        "api_key": api_key,
        "api_mode": str(runtime.get("api_mode") or "").strip(),
    }


def refresh_free_model_routes(
    *,
    force: bool,
    ensure_runtime_dirs: Callable[[], None],
    load_routing_state: Callable[[], dict[str, Any]],
    save_routing_state: Callable[[dict[str, Any]], None],
    resolve_provider_runtime_binding_fn: Callable[[str], dict[str, Any] | None],
    fetch_openrouter_free_model_candidates_fn: Callable[[], list[str]],
    load_nvidia_free_model_candidates_fn: Callable[[], list[str]],
    routing_refresh_ttl_seconds: int,
    default_openrouter_base_url: str,
    default_nvidia_base_url: str,
    logger: Any,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    existing = load_routing_state()
    now = int(time.time())
    if not force:
        refreshed_at = int(existing.get("refreshed_at") or 0)
        if refreshed_at and now - refreshed_at < routing_refresh_ttl_seconds:
            return existing

    providers: dict[str, Any] = {}
    openrouter_runtime = resolve_provider_runtime_binding_fn("openrouter")
    nvidia_runtime = resolve_provider_runtime_binding_fn("nvidia")
    try:
        providers["openrouter"] = {
            "base_url": str((openrouter_runtime or {}).get("base_url") or os.getenv("OPENROUTER_BASE_URL") or default_openrouter_base_url).strip(),
            "candidates": fetch_openrouter_free_model_candidates_fn(),
        }
    except Exception as exc:
        logger.warning("Failed refreshing OpenRouter free models: %s", exc)
        providers["openrouter"] = {
            "base_url": str((openrouter_runtime or {}).get("base_url") or os.getenv("OPENROUTER_BASE_URL") or default_openrouter_base_url).strip(),
            "candidates": list(DEFAULT_OPENROUTER_FREE_MODELS),
        }

    nvidia_candidates = load_nvidia_free_model_candidates_fn()
    if nvidia_candidates:
        providers["nvidia"] = {
            "base_url": str((nvidia_runtime or {}).get("base_url") or os.getenv("NVIDIA_BASE_URL") or default_nvidia_base_url).strip(),
            "candidates": nvidia_candidates,
        }

    payload = {"refreshed_at": now, "providers": providers}
    save_routing_state(payload)
    return payload
