import { describe, expect, it, vi } from "vitest";
import { runFeishuWorkflow } from "../src/workflow/runner";
import type { Env, FeishuNormalizedPayload, JsonValue, ModalInternalResponse } from "../src/runtime";

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

function createStepHarness() {
  const calls: Array<{ name: string; config?: Record<string, unknown> }> = [];
  const step = {
    do: vi.fn(async (name: string, configOrFn: unknown, maybeFn?: () => Promise<unknown>) => {
      const hasConfig = typeof configOrFn !== "function";
      const config = hasConfig ? (configOrFn as Record<string, unknown>) : undefined;
      const fn = (hasConfig ? maybeFn : configOrFn) as () => Promise<unknown>;
      calls.push({ name, config });
      return await fn();
    }),
  };
  return { step, calls };
}

function createDeps(overrides: Partial<Record<string, unknown>> = {}) {
  const log = vi.fn();
  const invokeAgentExec = vi.fn(
    async (
      _env: Env,
      _normalized: FeishuNormalizedPayload,
      metadata?: Partial<ModalInternalResponse>,
    ): Promise<ModalInternalResponse> => ({
      status: "ok",
      route_hint: "modal_heavy_exec",
      execution_mode: "modal_heavy_exec",
      send_plan: [
        {
          kind: "text",
          content: "fallback ok",
        } satisfies Record<string, JsonValue>,
      ],
      route_decision_reason: String(metadata?.route_decision_reason ?? ""),
      fallback_reason: String(metadata?.fallback_reason ?? ""),
      reconcile_required: false,
    }),
  );
  const executeFeishuOperations = vi.fn(async () => {});

  return {
    log,
    buildRoutingLogFields: vi.fn(() => ({})),
    inferSiteExecutionStage: vi.fn(() => "none"),
    maybeBuildSitePrefetch: vi.fn(async () => null),
    buildEdgeDirectPlan: vi.fn(async () => {
      throw new Error("planner exploded");
    }),
    trim: (value: unknown) => String(value ?? "").trim(),
    handleControlEvent: vi.fn(),
    handleAgentCommand: vi.fn(),
    invokeAgentExec,
    executeCloudflareAiExec: vi.fn(),
    classifyCfAiExecFallbackReason: vi.fn(() => "planner_failed"),
    getExecutablePlan: vi.fn((internal: ModalInternalResponse) => internal.send_plan || []),
    executeFeishuOperations,
    persistSessionCacheAfterResponse: vi.fn(async () => {}),
    buildPendingReconcile: vi.fn(() => null),
    enqueuePendingReconcile: vi.fn(async () => {}),
    modalAgentExecTimeoutSeconds: 20,
    ...overrides,
  };
}

describe("runFeishuWorkflow", () => {
  it("falls back to modal agent execution when edge-agent-plan fails", async () => {
    const { step, calls } = createStepHarness();
    const deps = createDeps();

    const result = await runFeishuWorkflow({
      env: {} as Env,
      event: {
        payload: buildNormalizedPayload(),
        instanceId: "wf-1",
      },
      step,
      deps: deps as any,
    });

    expect(result).toMatchObject({
      status: "ok",
      route_hint: "modal_heavy_exec",
      execution_mode: "modal_heavy_exec",
    });
    expect(deps.buildEdgeDirectPlan).toHaveBeenCalledTimes(1);
    expect(deps.invokeAgentExec).toHaveBeenCalledTimes(1);
    expect(deps.executeFeishuOperations).toHaveBeenCalledTimes(1);
    expect(calls.map((call) => call.name)).toEqual([
      "emit-ingress-log",
      "site-prefetch",
      "edge-agent-plan",
      "modal-agent-exec",
      "send-feishu-response",
      "emit-final-log",
    ]);
    expect(calls.find((call) => call.name === "edge-agent-plan")?.config).toMatchObject({
      retries: {
        limit: 1,
        delay: 1000,
        backoff: "constant",
      },
      timeout: "30 seconds",
    });
    expect(deps.log).toHaveBeenCalledWith(
      "feishu.workflow.plan.fallback",
      expect.objectContaining({
        correlation_id: "corr-1",
        fallback_reason: "edge_agent_plan_failed",
        error: "planner exploded",
      }),
    );
  });

  it("writes fallback analytics when cf ai exec falls back to modal once and succeeds", async () => {
    const { step } = createStepHarness();
    const analytics = {
      writeDataPoint: vi.fn(),
    };
    const planned: ModalInternalResponse = {
      status: "ok",
      external_exec_candidate: true,
      route_hint: "modal_heavy_exec",
      execution_mode: "deferred_reconcile",
      route_decision_reason: "provider_model_not_found",
      provider_async_eligible: true,
      provider_async_observed: false,
      send_plan: [],
    };
    const deps = createDeps({
      buildEdgeDirectPlan: vi.fn(async () => planned),
      executeCloudflareAiExec: vi.fn(async () => {
        throw new Error("provider 404");
      }),
      classifyCfAiExecFallbackReason: vi.fn(() => "provider_model_not_found"),
      invokeAgentExec: vi.fn(async () => ({
        status: "ok",
        route_hint: "modal_heavy_exec",
        execution_mode: "modal_heavy_exec",
        route_decision_reason: "provider_model_not_found",
        fallback_reason: "provider_model_not_found",
        send_plan: [{ kind: "text", content: "fallback ok" }],
        reconcile_required: false,
      })),
    });

    await runFeishuWorkflow({
      env: {
        FEISHU_GATEWAY_ANALYTICS: analytics,
      } as Env,
      event: {
        payload: buildNormalizedPayload(),
        instanceId: "wf-2",
      },
      step,
      deps: deps as any,
    });

    expect(analytics.writeDataPoint).toHaveBeenCalledTimes(3);
    expect(analytics.writeDataPoint).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        indexes: ["feishu.cf_ai_exec.fallback"],
      }),
    );
    expect(analytics.writeDataPoint).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        indexes: ["feishu.cf_ai_exec.fallback.done"],
      }),
    );
    expect(deps.log).toHaveBeenCalledWith(
      "feishu.cf_ai_exec.fallback.done",
      expect.objectContaining({
        correlation_id: "corr-1",
        fallback_reason: "provider_model_not_found",
        success: true,
      }),
    );
  });
});
