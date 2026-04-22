import { describe, expect, it } from "vitest";
import {
  classifyGatewayFeedbackErrorKind,
  deriveGatewayFeedbackScore,
  summarizeFeedbackWindow,
} from "../src/model-catalog/feedback";

describe("model catalog feedback aggregation", () => {
  it("summarizes feedback rows into hourly health stats", () => {
    const summary = summarizeFeedbackWindow([
      {
        feedback: 1,
        score: 92,
        latency_ms: 850,
        ttfb_ms: 850,
        cache_hit: 1,
        request_tokens: 120,
        response_tokens: 80,
        estimated_cost: 0.002,
      },
      {
        feedback: -1,
        score: 10,
        error_kind: "provider_rate_exhausted",
        latency_ms: 1600,
        ttfb_ms: 1600,
        cache_hit: 0,
        request_tokens: 100,
        response_tokens: 0,
        estimated_cost: 0.001,
      },
    ]);

    expect(summary.requestCount).toBe(2);
    expect(summary.successCount).toBe(1);
    expect(summary.errorCount).toBe(1);
    expect(summary.rateLimitCount).toBe(1);
    expect(summary.cacheHitCount).toBe(1);
    expect(summary.cacheMissCount).toBe(1);
    expect(summary.scoreAvg).toBe(51);
    expect(summary.inputTokensSum).toBe(220);
    expect(summary.outputTokensSum).toBe(80);
    expect(summary.healthScore).toBeGreaterThan(0);
    expect(summary.healthScore).toBeLessThan(100);
  });

  it("classifies and scores negative feedback for hard failures", () => {
    expect(classifyGatewayFeedbackErrorKind(new Error("failed:401 invalid auth"))).toBe("provider_permission_denied");
    expect(classifyGatewayFeedbackErrorKind(new Error("failed:429 over limit"))).toBe("rate_limited");
    expect(deriveGatewayFeedbackScore(-1, 1200, "MISS", "provider_permission_denied")).toBe(0);
    expect(deriveGatewayFeedbackScore(1, 400, "HIT")).toBeGreaterThan(80);
  });
});
