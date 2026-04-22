import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

"""
Intelligent routing four-layer decision loop tests for T013-T017.

This test suite validates:
- L1: Capability filtering (T014)
- L2: Request type matching (T014)
- L3: Score ranking (T015)
- L4: Fallback & recovery (T016, T017)

Test coverage:
1. Model capability tags normalization
2. Request type to capability matching
3. Score ranking with multiple dimensions
4. Fallback chain behavior
5. Positive/negative feedback loop
6. High-performance type-matching priority
"""

# ── Mock Model Catalog ──────────────────────────────────────────────────

MOCK_MODEL_CATALOG = [
    {
        "provider": "openrouter",
        "model": "google/gemini-2.0-flash:free",
        "capabilities": ["text", "multimodal", "tool_call"],
        "hidden": False,
        "status": "active",
        "recent_success_rate": 0.95,
        "recent_p50_ms": 800,
        "recent_p95_ms": 1500,
        "recent_avg_cost_usd": 0.0001,
        "fallback_trigger_rate": 0.02,
        "payload_compat_rate": 0.98,
        "cache_friendly_score": 0.85,
    },
    {
        "provider": "openrouter",
        "model": "nvidia/nemotron-nano-9b-v2:free",
        "capabilities": ["text", "tool_call"],
        "hidden": False,
        "status": "active",
        "recent_success_rate": 0.90,
        "recent_p50_ms": 600,
        "recent_p95_ms": 1200,
        "recent_avg_cost_usd": 0.00008,
        "fallback_trigger_rate": 0.05,
        "payload_compat_rate": 0.95,
        "cache_friendly_score": 0.90,
    },
    {
        "provider": "openrouter",
        "model": "google/gemma-4-31b-it:free",
        "capabilities": ["text", "coding"],
        "hidden": False,
        "status": "active",
        "recent_success_rate": 0.85,
        "recent_p50_ms": 1200,
        "recent_p95_ms": 2500,
        "recent_avg_cost_usd": 0.0002,
        "fallback_trigger_rate": 0.08,
        "payload_compat_rate": 0.92,
        "cache_friendly_score": 0.70,
    },
    {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4",
        "capabilities": ["text", "multimodal", "tool_call", "coding", "browser_heavy"],
        "hidden": False,
        "status": "active",
        "recent_success_rate": 0.98,
        "recent_p50_ms": 1500,
        "recent_p95_ms": 3000,
        "recent_avg_cost_usd": 0.0015,
        "fallback_trigger_rate": 0.01,
        "payload_compat_rate": 0.99,
        "cache_friendly_score": 0.60,
    },
    {
        "provider": "nvidia",
        "model": "meta/llama-3.1-8b-instruct",
        "capabilities": ["text", "tool_call"],
        "hidden": False,
        "status": "active",
        "recent_success_rate": 0.88,
        "recent_p50_ms": 500,
        "recent_p95_ms": 1000,
        "recent_avg_cost_usd": 0.00005,
        "fallback_trigger_rate": 0.03,
        "payload_compat_rate": 0.96,
        "cache_friendly_score": 0.95,
    },
    {
        "provider": "openrouter",
        "model": "broken/model:free",
        "capabilities": ["text"],
        "hidden": False,
        "status": "active",
        "recent_success_rate": 0.10,
        "recent_p50_ms": 5000,
        "recent_p95_ms": 10000,
        "recent_avg_cost_usd": 0.0005,
        "fallback_trigger_rate": 0.80,
        "payload_compat_rate": 0.30,
        "cache_friendly_score": 0.20,
    },
    {
        "provider": "openrouter",
        "model": "hidden/model:free",
        "capabilities": ["text"],
        "hidden": True,
        "status": "active",
        "recent_success_rate": 0.99,
        "recent_p50_ms": 300,
        "recent_p95_ms": 600,
        "recent_avg_cost_usd": 0.00001,
        "fallback_trigger_rate": 0.00,
        "payload_compat_rate": 1.0,
        "cache_friendly_score": 1.0,
    },
]


# ── Helper Functions ─────────────────────────────────────────────────────

def filter_by_capability(models: list, required_capabilities: list) -> list:
    """L1: Filter models by required capabilities."""
    filtered = []
    for model in models:
        if model.get("hidden"):
            continue
        if model.get("status") != "active":
            continue
        model_caps = set(model.get("capabilities", []))
        required = set(required_capabilities)
        if required.issubset(model_caps):
            filtered.append(model)
    return filtered


def match_request_type(models: list, request_type: str) -> list:
    """L2: Match request type to model capabilities."""
    type_to_capabilities = {
        "text_plain": ["text"],
        "text_coding": ["text", "coding"],
        "tool_browser": ["text", "tool_call", "browser_heavy"],
        "tool_non_browser": ["text", "tool_call"],
        "image_understanding": ["text", "multimodal"],
        "media_hydration": ["text", "multimodal"],
    }
    required_caps = type_to_capabilities.get(request_type, ["text"])
    return filter_by_capability(models, required_caps)


def calculate_model_score(model: dict) -> float:
    """L3: Calculate composite score for model ranking."""
    # Weight factors
    success_rate_weight = 0.30
    latency_weight = 0.20
    cost_weight = 0.15
    fallback_weight = 0.15
    compat_weight = 0.10
    cache_weight = 0.10

    # Normalize latency (lower is better, max 10s)
    p50_ms = model.get("recent_p50_ms", 5000)
    latency_score = max(0, 1 - (p50_ms / 10000))

    # Normalize cost (lower is better, max 0.01 USD)
    cost_usd = model.get("recent_avg_cost_usd", 0.001)
    cost_score = max(0, 1 - (cost_usd / 0.01))

    score = (
        model.get("recent_success_rate", 0) * success_rate_weight
        + latency_score * latency_weight
        + cost_score * cost_weight
        + (1 - model.get("fallback_trigger_rate", 0)) * fallback_weight
        + model.get("payload_compat_rate", 0) * compat_weight
        + model.get("cache_friendly_score", 0) * cache_weight
    )

    return score


def rank_models(models: list) -> list:
    """L3: Rank models by composite score."""
    scored = []
    for model in models:
        # Skip hidden or inactive models
        if model.get("hidden"):
            continue
        if model.get("status") != "active":
            continue
        score = calculate_model_score(model)
        scored.append((model, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def select_fallback(primary_model: dict, candidates: list, max_fallbacks: int = 1) -> list:
    """L4: Select fallback models (max 1 compatible fallback)."""
    fallbacks = []
    primary_caps = set(primary_model.get("capabilities", []))

    for model, score in candidates:
        if model.get("model") == primary_model.get("model"):
            continue
        # Only allow compatible fallbacks
        model_caps = set(model.get("capabilities", []))
        if primary_caps.issubset(model_caps) or model_caps.issubset(primary_caps):
            fallbacks.append(model)
        if len(fallbacks) >= max_fallbacks:
            break

    return fallbacks


# ── Test Classes ──────────────────────────────────────────────────────────

class TestL1CapabilityFiltering:
    """T014: L1 capability filtering tests."""

    def test_filter_text_models(self):
        """Text-only requests should filter to text-capable models."""
        filtered = filter_by_capability(MOCK_MODEL_CATALOG, ["text"])
        assert len(filtered) > 0
        for model in filtered:
            assert "text" in model.get("capabilities", [])
            assert not model.get("hidden")
            assert model.get("status") == "active"

    def test_filter_multimodal_models(self):
        """Image understanding requires multimodal capability."""
        filtered = filter_by_capability(MOCK_MODEL_CATALOG, ["text", "multimodal"])
        assert len(filtered) > 0
        for model in filtered:
            assert "multimodal" in model.get("capabilities", [])
            assert "text" in model.get("capabilities", [])

    def test_filter_browser_heavy_models(self):
        """Browser tasks require browser_heavy capability."""
        filtered = filter_by_capability(MOCK_MODEL_CATALOG, ["text", "tool_call", "browser_heavy"])
        assert len(filtered) == 1  # Only Claude has browser_heavy
        assert filtered[0]["model"] == "anthropic/claude-sonnet-4"

    def test_filter_hidden_models(self):
        """Hidden models should be excluded from filtering."""
        filtered = filter_by_capability(MOCK_MODEL_CATALOG, ["text"])
        for model in filtered:
            assert not model.get("hidden")

    def test_filter_inactive_models(self):
        """Inactive models should be excluded from filtering."""
        filtered = filter_by_capability(MOCK_MODEL_CATALOG, ["text"])
        for model in filtered:
            assert model.get("status") == "active"

    def test_filter_no_match(self):
        """When no model matches, return empty list."""
        filtered = filter_by_capability(MOCK_MODEL_CATALOG, ["quantum_computing"])
        assert len(filtered) == 0


class TestL2RequestTypeMatching:
    """T014: L2 request type matching tests."""

    def test_text_plain_matching(self):
        """Text plain requests should match text-capable models."""
        matched = match_request_type(MOCK_MODEL_CATALOG, "text_plain")
        assert len(matched) > 0
        for model in matched:
            assert "text" in model.get("capabilities", [])

    def test_text_coding_matching(self):
        """Coding requests should match coding-capable models."""
        matched = match_request_type(MOCK_MODEL_CATALOG, "text_coding")
        assert len(matched) > 0
        for model in matched:
            assert "coding" in model.get("capabilities", [])
            assert "text" in model.get("capabilities", [])

    def test_browser_heavy_matching(self):
        """Browser tasks should match browser_heavy models."""
        matched = match_request_type(MOCK_MODEL_CATALOG, "tool_browser")
        assert len(matched) == 1
        assert matched[0]["model"] == "anthropic/claude-sonnet-4"

    def test_image_understanding_matching(self):
        """Image tasks should match multimodal models."""
        matched = match_request_type(MOCK_MODEL_CATALOG, "image_understanding")
        assert len(matched) > 0
        for model in matched:
            assert "multimodal" in model.get("capabilities", [])

    def test_non_browser_tool_matching(self):
        """Non-browser tools should match tool_call models."""
        matched = match_request_type(MOCK_MODEL_CATALOG, "tool_non_browser")
        assert len(matched) > 0
        for model in matched:
            assert "tool_call" in model.get("capabilities", [])


class TestL3ScoreRanking:
    """T015: L3 score ranking tests."""

    def test_score_calculation_range(self):
        """Model scores should be between 0 and 1."""
        for model in MOCK_MODEL_CATALOG:
            if model.get("hidden"):
                continue
            score = calculate_model_score(model)
            assert 0 <= score <= 1, f"Score {score} out of range for {model['model']}"

    def test_high_quality_model_ranks_higher(self):
        """High success rate models should rank higher than broken models."""
        scored = rank_models(MOCK_MODEL_CATALOG)
        # Claude should rank higher than broken model
        claude_score = None
        broken_score = None
        for model, score in scored:
            if model["model"] == "anthropic/claude-sonnet-4":
                claude_score = score
            elif model["model"] == "broken/model:free":
                broken_score = score
        assert claude_score is not None
        assert broken_score is not None
        assert claude_score > broken_score

    def test_ranking_excludes_hidden(self):
        """Hidden models should not appear in ranking."""
        scored = rank_models(MOCK_MODEL_CATALOG)
        for model, score in scored:
            assert not model.get("hidden")

    def test_ranking_order(self):
        """Ranking should be in descending order."""
        scored = rank_models(MOCK_MODEL_CATALOG)
        scores = [score for _, score in scored]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestL4FallbackRecovery:
    """T016-T017: L4 fallback and recovery tests."""

    def test_single_fallback_allowed(self):
        """Only one compatible fallback should be allowed."""
        candidates = rank_models(MOCK_MODEL_CATALOG)
        primary = candidates[0][0]
        fallbacks = select_fallback(primary, candidates, max_fallbacks=1)
        assert len(fallbacks) <= 1

    def test_fallback_compatibility(self):
        """Fallback model must be compatible with primary."""
        candidates = rank_models(MOCK_MODEL_CATALOG)
        primary = candidates[0][0]
        fallbacks = select_fallback(primary, candidates, max_fallbacks=1)
        if fallbacks:
            primary_caps = set(primary.get("capabilities", []))
            fallback_caps = set(fallbacks[0].get("capabilities", []))
            # Either fallback supports all primary capabilities or vice versa
            assert primary_caps.issubset(fallback_caps) or fallback_caps.issubset(primary_caps)

    def test_no_duplicate_fallbacks(self):
        """Fallback should not include the primary model."""
        candidates = rank_models(MOCK_MODEL_CATALOG)
        primary = candidates[0][0]
        fallbacks = select_fallback(primary, candidates, max_fallbacks=1)
        for fallback in fallbacks:
            assert fallback["model"] != primary["model"]


class TestPositiveNegativeFeedback:
    """T016: Positive/negative feedback loop tests."""

    def test_negative_feedback_reduces_score(self):
        """Failed requests should reduce model score over time."""
        model = dict(MOCK_MODEL_CATALOG[0])
        initial_score = calculate_model_score(model)

        # Simulate failures
        model["recent_success_rate"] = 0.50
        model["fallback_trigger_rate"] = 0.50
        model["payload_compat_rate"] = 0.50

        reduced_score = calculate_model_score(model)
        assert reduced_score < initial_score

    def test_positive_feedback_increases_score(self):
        """Successful requests should increase model score over time."""
        model = dict(MOCK_MODEL_CATALOG[0])
        initial_score = calculate_model_score(model)

        # Simulate continuous success
        model["recent_success_rate"] = 0.99
        model["fallback_trigger_rate"] = 0.00
        model["payload_compat_rate"] = 0.99

        increased_score = calculate_model_score(model)
        assert increased_score > initial_score

    def test_recovery_requires_continuous_success(self):
        """Model recovery requires continuous success, not single success."""
        # Simulate a model that failed recently
        model = {
            "provider": "openrouter",
            "model": "recovery/test:free",
            "capabilities": ["text"],
            "hidden": False,
            "status": "active",
            "recent_success_rate": 0.60,
            "recent_p50_ms": 3000,
            "recent_p95_ms": 6000,
            "recent_avg_cost_usd": 0.0005,
            "fallback_trigger_rate": 0.30,
            "payload_compat_rate": 0.70,
            "cache_friendly_score": 0.50,
        }

        low_score = calculate_model_score(model)

        # Single success shouldn't fully recover
        model["recent_success_rate"] = 0.65  # Small improvement
        partial_score = calculate_model_score(model)

        # Should improve but not reach top level
        assert partial_score > low_score
        assert partial_score < 0.9  # Not fully recovered


class TestHighPerformanceTypeMatching:
    """T017: High-performance type-matching priority tests."""

    def test_prefer_high_performance_type_match(self):
        """When multiple models match, prefer high-performance type-matched model."""
        # For coding tasks, prefer coding-capable models
        coding_models = match_request_type(MOCK_MODEL_CATALOG, "text_coding")
        assert len(coding_models) > 0

        # Score and rank
        scored = rank_models(coding_models)
        # Top model should have coding capability
        assert "coding" in scored[0][0].get("capabilities", [])

    def test_text_model_not_selected_for_browser(self):
        """Text-only models should not be selected for browser tasks."""
        browser_models = match_request_type(MOCK_MODEL_CATALOG, "tool_browser")
        for model in browser_models:
            assert "browser_heavy" in model.get("capabilities", [])

    def test_coding_model_preferred_for_coding(self):
        """Coding-capable models should be preferred for coding tasks."""
        coding_models = match_request_type(MOCK_MODEL_CATALOG, "text_coding")
        # At least one model should have coding capability
        has_coding = any("coding" in m.get("capabilities", []) for m in coding_models)
        assert has_coding
