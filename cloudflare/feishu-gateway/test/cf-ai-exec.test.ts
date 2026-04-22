import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Env, FeishuNormalizedPayload, JsonValue, ModalInternalResponse } from "../src/runtime";

const recordGatewayFeedbackMock = vi.fn(async () => {});

vi.mock("../src/model-catalog/feedback", () => ({
  classifyGatewayFeedbackErrorKind: vi.fn((value: unknown) => {
    const message = value instanceof Error ? value.message : String(value ?? "");
    if (message.includes("429")) return "provider_rate_exhausted";
    return "provider_http_error";
  }),
  deriveGatewayFeedbackScore: vi.fn((feedback: number) => (feedback > 0 ? 95 : 10)),
  recordGatewayFeedback: recordGatewayFeedbackMock,
}));

const { executeCloudflareAiExec } = await import("../src/gateway/cf-ai-exec");

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function parseBoolean(value: unknown, fallback = false): boolean {
  const normalized = trim(value).toLowerCase();
  if (!normalized) return fallback;
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return fallback;
}

function parsePositiveInt(value: unknown, fallback: number, min = 1, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = Number.parseInt(trim(value), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function isFreeishModelName(value: unknown): boolean {
  const normalized = trim(value).toLowerCase();
  return normalized === "free" || normalized === "openrouter/free" || normalized.includes(":free");
}

function buildNormalizedPayload(): FeishuNormalizedPayload {
  return {
    correlation_id: "corr-1",
    session_key: "session-1",
    lane: "agent",
    route_hint: "modal_heavy_exec",
    site_category: "none",
    site_intent: "general",
    target_url: "",
    target_domain: "",
    task_kind: "general",
    request_class: "text_plain",
    content_modalities: ["text"],
    route_family: "gateway_text",
    gateway_route_name: "text-general",
    gateway_eligible: true,
    requires_tools: false,
    requires_browser: false,
    requires_media_hydration: false,
    requires_modal_runtime: false,
    modality_profile: "text",
    toolset: [],
    reason_code: "plain_text",
    event_id: "evt-1",
    event_type: "im.message.receive_v1",
    chat_id: "chat-1",
    chat_type: "dm",
    chat_name: "chat",
    user_id: "user-1",
    user_name: "user",
    message_id: "msg-1",
    message_type: "text",
    text: "hello",
    attachment_refs: [],
    raw_payload: {},
  };
}

function buildPlan(): ModalInternalResponse {
  return {
    status: "ok",
    external_exec_candidate: true,
    gateway_eligible: true,
    gateway_route_name: "text-general",
    request_class: "text_plain",
    route_family: "gateway_text",
    modality_profile: "text",
    toolset: [],
    content_modalities: ["text"],
    provider_async_eligible: true,
    provider_async_observed: true,
    route_decision_reason: "plain_text_without_attachments_or_browser",
    provider_plan: {
      provider: "openrouter",
      model: "openai/gpt-4.1-mini",
      fallback_model: "deepseek/deepseek-chat-v3-0324",
      fallback_provider: "openrouter",
      byok_alias: "default",
      request_timeout_ms: 4000,
      max_attempts: 1,
      retry_delay_ms: 250,
      backoff: "linear",
      cache_mode: "ttl",
      cache_scope: "chat",
      cache_ttl_seconds: 300,
    } satisfies Record<string, JsonValue>,
    llm_request: {
      model: "openai/gpt-4.1-mini",
      messages: [{ role: "user", content: "hello" }],
    } satisfies Record<string, JsonValue>,
  };
}

function buildDeps() {
  return {
    trim,
    log: vi.fn(),
    parseBoolean,
    parsePositiveInt,
    isFreeishModelName,
    isTextGatewayRequestClass: (value: string) => value === "text_plain" || value === "text_coding",
    getCloudflareAiGatewayBase: (env: Env) => trim(env.CLOUDFLARE_AI_GATEWAY_BASE_URL).replace(/\/$/, ""),
    getCloudflareAiGatewayRoot: (env: Env) =>
      trim(env.CLOUDFLARE_AI_GATEWAY_BASE_URL).replace(/\/compat$/i, "").replace(/\/$/, ""),
    sha256Hex: async (input: string) => `sha:${input.length}`,
    buildAiGatewayMetadata: () => ({ source: "test" }),
    buildRoutingLogFields: () => ({}),
    buildTextSendPlan: (content: string) => [{ type: "text", content }],
    inferRouteDecisionReason: () => "plain_text_without_attachments_or_browser",
    classifyCfAiExecFallbackReason: (error: unknown) => {
      const message = error instanceof Error ? error.message : String(error ?? "");
      return message.includes("429") ? "rate_limited" : "provider_http_error";
    },
  };
}

describe("executeCloudflareAiExec dynamic route execution", () => {
  beforeEach(() => {
    recordGatewayFeedbackMock.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("routes text requests through dynamic/<route> and records the resolved model", async () => {
    const deps = buildDeps();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe("https://gateway.ai.cloudflare.com/v1/account/gateway/compat/chat/completions");
      const headers = new Headers(init?.headers);
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, JsonValue>;
      expect(body.model).toBe("dynamic/text-general");
      expect(headers.get("cf-aig-byok-alias")).toBeNull();
      expect(headers.get("cf-aig-cache-key")?.startsWith("feishu:global:")).toBe(true);
      return new Response(
        JSON.stringify({
          model: "anthropic/claude-3.5-sonnet",
          choices: [{ message: { content: "dynamic route answer" } }],
          usage: { prompt_tokens: 12, completion_tokens: 8 },
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
            "cf-aig-provider": "openrouter",
            "cf-aig-model": "anthropic/claude-3.5-sonnet",
            "cf-aig-log-id": "log-1",
            "cf-aig-cache-status": "HIT",
          },
        },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await executeCloudflareAiExec(
      {
        CLOUDFLARE_AI_GATEWAY_API_KEY: "token",
        CLOUDFLARE_AI_GATEWAY_BASE_URL: "https://gateway.ai.cloudflare.com/v1/account/gateway/compat",
        HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT: "true",
        HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED: "true",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED: "false",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS: "12000",
      } as Env,
      buildNormalizedPayload(),
      buildPlan(),
      deps,
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.final_response).toBe("dynamic route answer");
    expect(result.cache_eligible).toBe(true);
    expect(result.cache_status).toBe("HIT");
    expect(result.ai_call_count).toBe(1);
    expect(result.capability_match).toBe(true);
    expect(result.preferred_model_selected).toBe(true);
    expect(result.provider_usage?.provider).toBe("openrouter");
    expect(result.provider_usage?.response_model).toBe("anthropic/claude-3.5-sonnet");
    expect(result.provider_usage?.requested_model).toBe("dynamic/text-general");
    expect(deps.log).toHaveBeenCalledWith(
      "feishu.cf_ai_exec.done",
      expect.objectContaining({
        correlation_id: "corr-1",
        provider: "openrouter",
        model: "anthropic/claude-3.5-sonnet",
        cf_cache_status: "HIT",
        cache_eligible: true,
        cache_scope: "global",
        cache_fingerprint: expect.any(String),
        ai_call_count: 1,
        capability_match: true,
        preferred_model_selected: true,
      }),
    );
    expect(recordGatewayFeedbackMock).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        provider: "openrouter",
        model: "anthropic/claude-3.5-sonnet",
        cacheStatus: "HIT",
      }),
    );
  });

  it("skips AI Gateway cache for context-dependent follow-up prompts", async () => {
    const deps = buildDeps();
    const normalized = buildNormalizedPayload();
    normalized.text = "继续上一个话题";
    const plan = buildPlan();
    plan.llm_request = {
      model: "openai/gpt-4.1-mini",
      messages: [
        { role: "user", content: "请介绍一下你自己" },
        { role: "assistant", content: "我是 Hermes。" },
        { role: "user", content: "继续上一个话题" },
      ],
    } satisfies Record<string, JsonValue>;

    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      expect(headers.get("cf-aig-cache-key")).toBeNull();
      expect(headers.get("cf-aig-skip-cache")).toBe("true");
      return new Response(
        JSON.stringify({
          model: "openai/gpt-4.1-mini",
          choices: [{ message: { content: "follow-up answer" } }],
          usage: { prompt_tokens: 14, completion_tokens: 6 },
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
            "cf-aig-provider": "openrouter",
            "cf-aig-model": "openai/gpt-4.1-mini",
            "cf-aig-cache-status": "MISS",
          },
        },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await executeCloudflareAiExec(
      {
        CLOUDFLARE_AI_GATEWAY_API_KEY: "token",
        CLOUDFLARE_AI_GATEWAY_BASE_URL: "https://gateway.ai.cloudflare.com/v1/account/gateway/compat",
        HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT: "true",
        HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED: "true",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED: "false",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS: "12000",
      } as Env,
      normalized,
      plan,
      deps,
    );

    expect(result.cache_eligible).toBe(false);
    expect(result.cache_status).toBe("MISS");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("keeps stateless cache keys stable even when the conversation history grows", async () => {
    const deps = buildDeps();
    const normalized = buildNormalizedPayload();
    const plan = buildPlan();
    plan.llm_request = {
      model: "openai/gpt-4.1-mini",
      messages: [
        { role: "user", content: "你好" },
        { role: "assistant", content: "你好，我是 Hermes。" },
        { role: "user", content: "请介绍一下你自己，你的核心竞争力是什么" },
        { role: "assistant", content: "我是一个多工具协同代理。" },
        { role: "user", content: "请介绍一下你自己，你的核心竞争力是什么" },
      ],
    } satisfies Record<string, JsonValue>;

    const cacheKeys: string[] = [];
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      cacheKeys.push(headers.get("cf-aig-cache-key") ?? "");
      expect(headers.get("cf-aig-skip-cache")).toBeNull();
      return new Response(
        JSON.stringify({
          model: "openai/gpt-4.1-mini",
          choices: [{ message: { content: "stable cache answer" } }],
          usage: { prompt_tokens: 18, completion_tokens: 8 },
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
            "cf-aig-provider": "openrouter",
            "cf-aig-model": "openai/gpt-4.1-mini",
            "cf-aig-cache-status": "MISS",
          },
        },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const first = await executeCloudflareAiExec(
      {
        CLOUDFLARE_AI_GATEWAY_API_KEY: "token",
        CLOUDFLARE_AI_GATEWAY_BASE_URL: "https://gateway.ai.cloudflare.com/v1/account/gateway/compat",
        HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT: "true",
        HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED: "true",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED: "false",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS: "12000",
      } as Env,
      normalized,
      plan,
      deps,
    );

    plan.llm_request = {
      model: "openai/gpt-4.1-mini",
      messages: [
        { role: "user", content: "你好" },
        { role: "assistant", content: "你好，我是 Hermes。" },
        { role: "user", content: "请介绍一下你自己，你的核心竞争力是什么" },
        { role: "assistant", content: "我是一个多工具协同代理。" },
        { role: "user", content: "再简单一点" },
        { role: "assistant", content: "我是一个执行型 AI 代理。" },
        { role: "user", content: "请介绍一下你自己，你的核心竞争力是什么" },
      ],
    } satisfies Record<string, JsonValue>;

    const second = await executeCloudflareAiExec(
      {
        CLOUDFLARE_AI_GATEWAY_API_KEY: "token",
        CLOUDFLARE_AI_GATEWAY_BASE_URL: "https://gateway.ai.cloudflare.com/v1/account/gateway/compat",
        HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT: "true",
        HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED: "true",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED: "false",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS: "12000",
      } as Env,
      normalized,
      plan,
      deps,
    );

    expect(first.cache_eligible).toBe(true);
    expect(second.cache_eligible).toBe(true);
    expect(cacheKeys).toHaveLength(2);
    expect(cacheKeys[0]).toBe(cacheKeys[1]);
    expect(cacheKeys[0]).toContain("feishu:global:");
  });

  it("does not perform application-side fallback when dynamic route execution is enabled", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: { message: "rate limited" },
        }),
        {
          status: 429,
          headers: {
            "content-type": "application/json",
            "cf-aig-provider": "openrouter",
            "cf-aig-model": "openai/gpt-4.1-mini",
          },
        },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      executeCloudflareAiExec(
        {
          CLOUDFLARE_AI_GATEWAY_API_KEY: "token",
          CLOUDFLARE_AI_GATEWAY_BASE_URL: "https://gateway.ai.cloudflare.com/v1/account/gateway/compat",
          HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT: "true",
          HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED: "true",
          HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED: "false",
          HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS: "12000",
        } as Env,
        buildNormalizedPayload(),
        buildPlan(),
        buildDeps(),
      ),
    ).rejects.toThrow(/cf_ai_exec_failed:429/);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("marks preferred_model_selected false and increments ai_call_count when direct fallback succeeds", async () => {
    const deps = buildDeps();
    const plan = buildPlan();
    plan.gateway_route_name = "";
    plan.provider_plan = {
      ...(plan.provider_plan as Record<string, JsonValue>),
      cache_mode: "skip",
    };
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(async () => {
        return new Response(
          JSON.stringify({
            error: { message: "rate limited" },
          }),
          {
            status: 429,
            headers: {
              "content-type": "application/json",
              "cf-aig-provider": "openrouter",
              "cf-aig-model": "openai/gpt-4.1-mini",
            },
          },
        );
      })
      .mockImplementationOnce(async () => {
        return new Response(
          JSON.stringify({
            model: "deepseek/deepseek-chat-v3-0324",
            choices: [{ message: { content: "fallback answer" } }],
            usage: { prompt_tokens: 10, completion_tokens: 5 },
          }),
          {
            status: 200,
            headers: {
              "content-type": "application/json",
              "cf-aig-provider": "openrouter",
              "cf-aig-model": "deepseek/deepseek-chat-v3-0324",
              "cf-aig-cache-status": "MISS",
            },
          },
        );
      });
    vi.stubGlobal("fetch", fetchMock);

    const result = await executeCloudflareAiExec(
      {
        CLOUDFLARE_AI_GATEWAY_API_KEY: "token",
        CLOUDFLARE_AI_GATEWAY_BASE_URL: "https://gateway.ai.cloudflare.com/v1/account/gateway/compat",
        HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT: "true",
        HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED: "false",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED: "false",
        HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS: "12000",
      } as Env,
      buildNormalizedPayload(),
      plan,
      deps,
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.final_response).toBe("fallback answer");
    expect(result.ai_call_count).toBe(2);
    expect(result.cache_eligible).toBe(false);
    expect(result.cache_status).toBe("MISS");
    expect(result.preferred_model_selected).toBe(false);
    expect(result.provider_usage?.response_model).toBe("deepseek/deepseek-chat-v3-0324");
    expect(deps.log).toHaveBeenCalledWith(
      "feishu.cf_ai_exec.done",
      expect.objectContaining({
        ai_call_count: 2,
        cache_eligible: false,
        preferred_model_selected: false,
      }),
    );
  });
});
