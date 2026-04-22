import { describe, expect, it, vi } from "vitest";
import type { Env, FeishuNormalizedPayload, JsonValue, ModalInternalResponse } from "../src/runtime";
import { buildEdgeDirectPlan, invokeAgentExec } from "../src/gateway/agent-runtime";

function trim(value: unknown): string {
  return String(value ?? "").trim();
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

describe("invokeAgentExec", () => {
  it("forwards route observability metadata to modal", async () => {
    let capturedBody: Record<string, JsonValue> | null = null;
    const callModalWithReconciles = vi.fn(async (_env: Env, _path: string, _normalized: FeishuNormalizedPayload, body: Record<string, JsonValue>) => {
      capturedBody = body;
      return {
        status: "ok",
        route_hint: "modal_heavy_exec",
        execution_mode: "modal_heavy_exec",
        route_decision_reason: "plain_text_without_attachments_or_browser",
        fallback_reason: "",
      } satisfies ModalInternalResponse;
    });

    await invokeAgentExec(
      {} as Env,
      buildNormalizedPayload(),
      {
        route_decision_reason: "plain_text_without_attachments_or_browser",
        provider_plan: {
          provider: "openrouter",
          cache_mode: "ttl",
        } satisfies Record<string, JsonValue>,
        preferred_model_selected: true,
        capability_match: true,
      },
      undefined,
      {
        trim,
        log: vi.fn(),
        parseBoolean: () => false,
        callModalWithReconciles,
        classifyModalInternalError: () => ({ retryable: false, retryClass: "hard_fail" }),
        buildRoutingLogFields: () => ({}),
        peekSessionStateCache: async () => ({}),
        inferRouteDecisionReason: () => "plain_text_without_attachments_or_browser",
        isExternalExecCandidate: () => true,
        buildProviderPlanFromSessionProfile: () => ({}),
        buildConversationMessages: () => [],
        modalAgentExecRetryLimit: 1,
      },
    );

    expect(callModalWithReconciles).toHaveBeenCalledTimes(1);
    expect(capturedBody).toBeTruthy();
    expect(capturedBody?.route_version).toBe("cf_route_policy_v1");
    expect(capturedBody?.provider_alias).toBe("openrouter");
    expect(capturedBody?.cache_eligible).toBe(true);
    expect(capturedBody?.capability_match).toBe(true);
    expect(capturedBody?.preferred_model_selected).toBe(true);
  });
});

describe("buildEdgeDirectPlan", () => {
  it("returns a stable execution_mode for non-external text requests", async () => {
    const plan = await buildEdgeDirectPlan(
      {} as Env,
      buildNormalizedPayload(),
      null,
      {
        trim,
        log: vi.fn(),
        parseBoolean: () => false,
        peekSessionStateCache: async () => ({}),
        inferRouteDecisionReason: () => "plain_text_without_attachments_or_browser",
        isExternalExecCandidate: () => false,
        buildProviderPlanFromSessionProfile: () => ({}),
        buildConversationMessages: () => [],
        buildRoutingLogFields: () => ({}),
      },
    );

    expect(plan.status).toBe("ok");
    expect(plan.execution_mode).toBe("modal_heavy_exec");
    expect(plan.external_exec_candidate).toBe(false);
    expect(plan.reconcile_required).toBe(false);
  });
});
