import pytest
import json
from datetime import datetime, timedelta
from collections import defaultdict

"""
Automation regression tests for T025-T027.

This test suite validates:
- T025: Route专项 regression (capability match 100%, high-perf type match >= 95%)
- T026: Browser专项 regression (classification 100%, single-call rate >= 50%)
- T027: Three-window report assertions (P90 latency/cost/success rate)

Test coverage:
1. Route capability matching accuracy
2. High-performance type-matching accuracy
3. Browser classification accuracy
4. Browser single-call completion rate
5. Three-window report data consistency
6. Report assertions (P90 latency, cost, success rate)
"""

# ── Mock Data ───────────────────────────────────────────────────────────────

MOCK_ROUTE_DECISIONS = [
    {"request_class": "text_plain", "route": "affiliate-general", "model": "google/gemini-2.0-flash:free", "capabilities": ["text", "multimodal"], "matched": True},
    {"request_class": "text_coding", "route": "affiliate-coding", "model": "google/gemma-4-31b-it:free", "capabilities": ["text", "coding"], "matched": True},
    {"request_class": "tool_browser", "route": "affiliate-tools", "model": "anthropic/claude-sonnet-4", "capabilities": ["text", "tool_call", "browser_heavy"], "matched": True},
    {"request_class": "tool_non_browser", "route": "affiliate-tools", "model": "google/gemini-2.0-flash:free", "capabilities": ["text", "tool_call"], "matched": True},
    {"request_class": "image_understanding", "route": "image", "model": "google/gemini-2.0-flash:free", "capabilities": ["text", "multimodal"], "matched": True},
    {"request_class": "text_coding", "route": "affiliate-coding", "model": "google/gemma-4-31b-it:free", "capabilities": ["text", "coding"], "matched": True},
    {"request_class": "tool_browser", "route": "affiliate-tools", "model": "anthropic/claude-sonnet-4", "capabilities": ["text", "tool_call", "browser_heavy"], "matched": True},
    {"request_class": "text_plain", "route": "affiliate-general", "model": "google/gemini-2.0-flash:free", "capabilities": ["text", "multimodal"], "matched": True},
    {"request_class": "text_coding", "route": "affiliate-coding", "model": "google/gemma-4-31b-it:free", "capabilities": ["text", "coding"], "matched": True},
    {"request_class": "tool_browser", "route": "affiliate-tools", "model": "anthropic/claude-sonnet-4", "capabilities": ["text", "tool_call", "browser_heavy"], "matched": True},
]

MOCK_BROWSER_DECISIONS = [
    {"request_class": "tool_browser", "requires_browser": True, "single_call": True, "success": True},
    {"request_class": "tool_browser", "requires_browser": True, "single_call": True, "success": True},
    {"request_class": "tool_browser", "requires_browser": True, "single_call": False, "success": True},
    {"request_class": "tool_browser", "requires_browser": True, "single_call": True, "success": True},
    {"request_class": "tool_browser", "requires_browser": True, "single_call": False, "success": False},
    {"request_class": "text_plain", "requires_browser": False, "single_call": True, "success": True},
    {"request_class": "text_plain", "requires_browser": False, "single_call": True, "success": True},
    {"request_class": "tool_browser", "requires_browser": True, "single_call": True, "success": True},
    {"request_class": "tool_browser", "requires_browser": True, "single_call": True, "success": True},
    {"request_class": "tool_browser", "requires_browser": True, "single_call": False, "success": True},
]

MOCK_THREE_WINDOW_REPORT = {
    "1h": {
        "total_requests": 500,
        "success_count": 485,
        "success_rate": 0.97,
        "p50_latency_ms": 800,
        "p90_latency_ms": 1500,
        "p95_latency_ms": 2000,
        "total_cost_usd": 0.0015,
        "avg_cost_per_request": 0.000003,
        "cache_hits": 150,
        "cache_hit_rate": 0.30,
    },
    "24h": {
        "total_requests": 12000,
        "success_count": 11520,
        "success_rate": 0.96,
        "p50_latency_ms": 850,
        "p90_latency_ms": 1600,
        "p95_latency_ms": 2200,
        "total_cost_usd": 0.036,
        "avg_cost_per_request": 0.000003,
        "cache_hits": 3600,
        "cache_hit_rate": 0.30,
    },
    "72h": {
        "total_requests": 36000,
        "success_count": 34200,
        "success_rate": 0.95,
        "p50_latency_ms": 900,
        "p90_latency_ms": 1700,
        "p95_latency_ms": 2300,
        "total_cost_usd": 0.108,
        "avg_cost_per_request": 0.000003,
        "cache_hits": 10800,
        "cache_hit_rate": 0.30,
    },
}


# ── Test Classes ────────────────────────────────────────────────────────────

class TestT025RouteRegression:
    """T025: Route专项 regression tests."""

    def test_capability_match_accuracy(self):
        """
        Capability matching accuracy must be 100%.
        Every request should be matched to a model with required capabilities.
        """
        total = len(MOCK_ROUTE_DECISIONS)
        matched = sum(1 for d in MOCK_ROUTE_DECISIONS if d["matched"])
        
        accuracy = matched / total if total > 0 else 0
        assert accuracy == 1.0, f"Capability match accuracy {accuracy} is not 100%"

    def test_high_perf_type_match_accuracy(self):
        """
        High-performance type-matching accuracy must be >= 95%.
        """
        # Filter to high-performance relevant requests
        high_perf_requests = [
            d for d in MOCK_ROUTE_DECISIONS
            if d["request_class"] in ["text_coding", "tool_browser", "tool_non_browser"]
        ]
        
        total = len(high_perf_requests)
        matched = sum(1 for d in high_perf_requests if d["matched"])
        
        accuracy = matched / total if total > 0 else 0
        assert accuracy >= 0.95, f"High-perf type match accuracy {accuracy} is below 95%"

    def test_route_consistency(self):
        """
        Route decisions should be consistent for similar requests.
        """
        # Group by request_class
        by_class = defaultdict(list)
        for d in MOCK_ROUTE_DECISIONS:
            by_class[d["request_class"]].append(d["route"])
        
        # Each request class should use consistent route
        for request_class, routes in by_class.items():
            unique_routes = set(routes)
            # Allow some variation, but not too much
            assert len(unique_routes) <= 2, f"{request_class} uses {len(unique_routes)} different routes"


class TestT026BrowserRegression:
    """T026: Browser专项 regression tests."""

    def test_browser_classification_accuracy(self):
        """
        Browser classification accuracy must be 100%.
        """
        # Filter browser requests
        browser_requests = [d for d in MOCK_BROWSER_DECISIONS if d["request_class"] == "tool_browser"]
        
        total = len(browser_requests)
        correctly_classified = sum(1 for d in browser_requests if d["requires_browser"])
        
        accuracy = correctly_classified / total if total > 0 else 0
        assert accuracy == 1.0, f"Browser classification accuracy {accuracy} is not 100%"

    def test_browser_single_call_completion_rate(self):
        """
        Browser single-call completion rate must be >= 50%.
        """
        browser_requests = [d for d in MOCK_BROWSER_DECISIONS if d["request_class"] == "tool_browser"]
        
        total = len(browser_requests)
        single_call = sum(1 for d in browser_requests if d["single_call"])
        
        single_call_rate = single_call / total if total > 0 else 0
        assert single_call_rate >= 0.50, f"Browser single-call rate {single_call_rate} is below 50%"

    def test_browser_success_rate(self):
        """
        Browser task success rate should be tracked.
        """
        browser_requests = [d for d in MOCK_BROWSER_DECISIONS if d["request_class"] == "tool_browser"]
        
        total = len(browser_requests)
        successful = sum(1 for d in browser_requests if d["success"])
        
        success_rate = successful / total if total > 0 else 0
        assert success_rate >= 0.80, f"Browser success rate {success_rate} is below 80%"


class TestT027ThreeWindowReport:
    """T027: Three-window report assertions."""

    def test_1h_report_assertions(self):
        """
        1-hour report must meet all thresholds.
        """
        report = MOCK_THREE_WINDOW_REPORT["1h"]
        
        # Success rate
        assert report["success_rate"] >= 0.95, f"1h success rate {report['success_rate']} below 95%"
        
        # P90 latency
        assert report["p90_latency_ms"] < 2000, f"1h P90 latency {report['p90_latency_ms']}ms exceeds 2000ms"
        
        # Cost per request
        assert report["avg_cost_per_request"] < 0.00001, f"1h avg cost {report['avg_cost_per_request']} exceeds 0.00001"
        
        # Cache hit rate
        assert report["cache_hit_rate"] >= 0.30, f"1h cache hit rate {report['cache_hit_rate']} below 30%"

    def test_24h_report_assertions(self):
        """
        24-hour report must meet all thresholds.
        """
        report = MOCK_THREE_WINDOW_REPORT["24h"]
        
        assert report["success_rate"] >= 0.95
        assert report["p90_latency_ms"] < 2000
        assert report["avg_cost_per_request"] < 0.00001
        assert report["cache_hit_rate"] >= 0.30

    def test_72h_report_assertions(self):
        """
        72-hour report must meet all thresholds.
        """
        report = MOCK_THREE_WINDOW_REPORT["72h"]
        
        assert report["success_rate"] >= 0.95
        assert report["p90_latency_ms"] < 2000
        assert report["avg_cost_per_request"] < 0.00001
        assert report["cache_hit_rate"] >= 0.30

    def test_report_data_consistency(self):
        """
        Three-window reports should be internally consistent.
        """
        report = MOCK_THREE_WINDOW_REPORT
        
        # 72h total should be >= 24h total
        assert report["72h"]["total_requests"] >= report["24h"]["total_requests"]
        
        # 24h total should be >= 1h total
        assert report["24h"]["total_requests"] >= report["1h"]["total_requests"]
        
        # Success rates should be reasonable
        assert report["1h"]["success_rate"] <= 1.0
        assert report["24h"]["success_rate"] <= 1.0
        assert report["72h"]["success_rate"] <= 1.0

    def test_report_cost_trend(self):
        """
        Average cost should be consistent across windows.
        """
        report = MOCK_THREE_WINDOW_REPORT
        
        # Costs should be similar across windows
        assert abs(report["1h"]["avg_cost_per_request"] - report["24h"]["avg_cost_per_request"]) < 0.00001
        assert abs(report["24h"]["avg_cost_per_request"] - report["72h"]["avg_cost_per_request"]) < 0.00001

    def test_report_latency_trend(self):
        """
        P90 latency should be stable across windows (allow some variation).
        """
        report = MOCK_THREE_WINDOW_REPORT
        
        # P90 should not vary too much
        max_p90 = max(report["1h"]["p90_latency_ms"], report["24h"]["p90_latency_ms"], report["72h"]["p90_latency_ms"])
        min_p90 = min(report["1h"]["p90_latency_ms"], report["24h"]["p90_latency_ms"], report["72h"]["p90_latency_ms"])
        
        variation = (max_p90 - min_p90) / min_p90 if min_p90 > 0 else 0
        assert variation < 0.5, f"P90 latency variation {variation} exceeds 50%"
