import { describe, expect, it } from "vitest";
import { buildRouteElements, candidateSort, resolveGatewayProviderSlug } from "../src/model-catalog/routes";

describe("model catalog dynamic routes", () => {
  it("prioritizes pinned and healthier models", () => {
    const rows = [
      {
        provider: "openrouter",
        model: "model-b",
        manual_pinned: 0,
        selection_hint: "fallback",
        health_score: 90,
        error_rate: 0.02,
        cache_hit_rate: 0.1,
        priority_rank: 10,
      },
      {
        provider: "openrouter",
        model: "model-a",
        manual_pinned: 1,
        selection_hint: "recommended",
        health_score: 40,
        error_rate: 0.3,
        cache_hit_rate: 0.2,
        priority_rank: 20,
      },
      {
        provider: "openrouter",
        model: "model-c",
        manual_pinned: 0,
        selection_hint: "recommended",
        health_score: 95,
        error_rate: 0.01,
        cache_hit_rate: 0.6,
        priority_rank: 30,
      },
    ];

    const sorted = [...rows].sort(candidateSort);
    expect(sorted.map((item) => item.model)).toEqual(["model-a", "model-c", "model-b"]);
  });

  it("builds provider-level fallback chains with provider rate guards", () => {
    const elements = buildRouteElements([
      {
        provider: "openrouter",
        model: "model-a",
        gateway_route_name: "text-general",
        task_kind: "general",
        request_timeout_ms: 4500,
        max_attempts: 0,
        rate_limit_rpm: 60,
        burst_limit: 6,
        rate_limit_window_seconds: 60,
      },
      {
        provider: "openrouter",
        model: "model-b",
        gateway_route_name: "text-general",
        task_kind: "general",
        request_timeout_ms: 5000,
        max_attempts: 1,
      },
      {
        provider: "nvidia",
        model: "model-c",
        gateway_route_name: "text-general",
        task_kind: "general",
        request_timeout_ms: 6000,
        max_attempts: 0,
      },
    ], {
      gatewayProviderSlugs: {
        openrouter: "openrouter",
        nvidia: "custom-nvidia",
      },
      providerRateLimits: {
        nvidia: {
          limit: 20,
          interval: 60,
          key: "metadata.gateway_route_name",
        },
      },
    });

    expect(elements[0]?.type).toBe("start");
    expect(elements.filter((item) => item.type === "rate")).toHaveLength(2);
    expect(elements.filter((item) => item.type === "model").map((item) => item.properties?.model)).toEqual([
      "model-a",
      "model-b",
      "model-c",
    ]);
    expect(elements.filter((item) => item.type === "model").map((item) => item.properties?.provider)).toEqual([
      "openrouter",
      "openrouter",
      "custom-nvidia",
    ]);
    const firstModel = elements.find((item) => item.id === "model_1");
    expect(firstModel?.outputs.fallback.elementId).toBe("model_2");
    const secondModel = elements.find((item) => item.id === "model_2");
    expect(secondModel?.outputs.fallback.elementId).toBe("provider_2_rate_limit");
    const firstRateLimit = elements.find((item) => item.id === "provider_1_rate_limit");
    expect(firstRateLimit?.properties).toMatchObject({
      limit: 60,
      limitType: "count",
      window: 60,
      key: "metadata.gateway_route_name",
    });
    const secondRateLimit = elements.find((item) => item.id === "provider_2_rate_limit");
    expect(secondRateLimit?.properties).toMatchObject({
      limit: 20,
      window: 60,
      key: "metadata.gateway_route_name",
    });
  });

  it("keeps native nvidia by default and only uses nvidia-integrate when explicitly overridden", () => {
    expect(resolveGatewayProviderSlug("nvidia", {}, new Set(["nvidia-integrate"]))).toBe("nvidia");
    expect(resolveGatewayProviderSlug("nvidia", { nvidia: "nvidia-integrate" }, new Set())).toBe("nvidia-integrate");
    expect(resolveGatewayProviderSlug("openrouter", {}, new Set(["nvidia-integrate"]))).toBe("openrouter");
    expect(resolveGatewayProviderSlug("bigmodel", {}, new Set(["bigmodel"]))).toBe("custom-bigmodel");
    expect(resolveGatewayProviderSlug("nvidia", {}, new Set())).toBe("nvidia");
  });
});
