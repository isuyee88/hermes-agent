import type {
  Env,
  FeishuNormalizedPayload,
  JsonValue,
  ModalInternalResponse,
  PendingReconcileItem,
  SitePrefetchManifest,
} from "../runtime";
import { writeFeishuAnalyticsEvent } from "../observability/analytics-engine";

type WorkflowDeps = {
  log: (event: string, extra: Record<string, unknown>) => void;
  buildRoutingLogFields: (
    normalized: FeishuNormalizedPayload,
    overrides?: Partial<ModalInternalResponse>,
  ) => Record<string, unknown>;
  inferSiteExecutionStage: (normalized: FeishuNormalizedPayload) => string;
  maybeBuildSitePrefetch: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    options?: { allowCrawl: boolean },
  ) => Promise<SitePrefetchManifest | null>;
  buildEdgeDirectPlan: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    sitePrefetch?: SitePrefetchManifest | null,
  ) => Promise<ModalInternalResponse>;
  trim: (value: unknown) => string;
  handleControlEvent: (env: Env, normalized: FeishuNormalizedPayload) => Promise<ModalInternalResponse>;
  handleAgentCommand: (env: Env, normalized: FeishuNormalizedPayload) => Promise<ModalInternalResponse>;
  invokeAgentExec: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    metadata?: Partial<ModalInternalResponse>,
    sitePrefetch?: SitePrefetchManifest | null,
  ) => Promise<ModalInternalResponse>;
  executeCloudflareAiExec: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    planned: ModalInternalResponse,
  ) => Promise<ModalInternalResponse>;
  classifyCfAiExecFallbackReason: (error: unknown, planned: ModalInternalResponse) => string;
  getExecutablePlan: (internal: ModalInternalResponse) => Array<Record<string, JsonValue>>;
  buildTextSendPlan: (content: string) => Array<Record<string, JsonValue>>;
  executeFeishuOperations: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    operations: Array<Record<string, JsonValue>>,
  ) => Promise<void>;
  persistSessionCacheAfterResponse: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    internal: ModalInternalResponse,
  ) => Promise<void>;
  buildPendingReconcile: (
    normalized: FeishuNormalizedPayload,
    internal: ModalInternalResponse,
  ) => PendingReconcileItem | null;
  enqueuePendingReconcile: (env: Env, sessionKey: string, item: PendingReconcileItem) => Promise<void>;
  modalAgentExecTimeoutSeconds: number;
};

export async function runFeishuWorkflow(args: {
  env: Env;
  event: { payload: FeishuNormalizedPayload; instanceId: string };
  step: any;
  deps: WorkflowDeps;
}): Promise<Record<string, JsonValue>> {
  const { env, event, step, deps } = args;
  const normalized = event.payload;

  await step.do("emit-ingress-log", async () => {
    deps.log("feishu.workflow.start", {
      correlation_id: normalized.correlation_id,
      lane: normalized.lane,
      workflow_instance_id: event.instanceId,
      ...deps.buildRoutingLogFields(normalized),
    });
    return { ok: true };
  });

  const sitePrefetch =
    normalized.lane === "agent"
      ? await step.do("site-prefetch", async () => {
          try {
            return await deps.maybeBuildSitePrefetch(env, normalized, { allowCrawl: true });
          } catch (error) {
            deps.log("feishu.site_prefetch.skipped", {
              correlation_id: normalized.correlation_id,
              site_execution_stage: deps.inferSiteExecutionStage(normalized),
              reason: "workflow_step_fail_open",
              error: error instanceof Error ? error.message : String(error),
              ...deps.buildRoutingLogFields(normalized),
            });
            return null;
          }
        })
      : null;

  let planned: ModalInternalResponse | null = null;
  if (normalized.lane === "agent" && normalized.route_hint !== "fast_control") {
    try {
      planned = await step.do(
        "edge-agent-plan",
        {
          retries: {
            limit: 1,
            delay: 1000,
            backoff: "constant",
          },
          timeout: "30 seconds",
        },
        async () => deps.buildEdgeDirectPlan(env, normalized, sitePrefetch),
      );
    } catch (error) {
      deps.log("feishu.workflow.plan.fallback", {
        correlation_id: normalized.correlation_id,
        route_hint: normalized.route_hint,
        fallback_reason: "edge_agent_plan_failed",
        error: error instanceof Error ? error.message : String(error),
        ...deps.buildRoutingLogFields(normalized),
      });
      planned = null;
    }
  }
  const effectiveRouteHint = deps.trim(planned?.route_hint) || normalized.route_hint;
  if (planned) {
    deps.log("feishu.workflow.plan.done", {
      correlation_id: normalized.correlation_id,
      execution_mode: deps.trim(planned.execution_mode) || "modal_heavy_exec",
      external_exec_candidate: !!planned.external_exec_candidate,
      route_decision_reason: deps.trim(planned.route_decision_reason),
      provider_async_eligible: !!planned.provider_async_eligible,
      provider_async_observed: !!planned.provider_async_observed,
      ...deps.buildRoutingLogFields(normalized, {
        ...planned,
        route_hint: effectiveRouteHint as ModalInternalResponse["route_hint"],
      }),
    });
  }

  const internal =
    normalized.lane === "control"
      ? await step.do("fast-control-exec", async () => {
          return {
            ...(await deps.handleControlEvent(env, normalized)),
            route_hint: "fast_control",
            execution_mode: "control_complete",
          } satisfies ModalInternalResponse;
        })
      : normalized.lane === "agent" && normalized.route_hint === "fast_control"
        ? await step.do("command-control-exec", async () => {
            return {
              ...(await deps.handleAgentCommand(env, normalized)),
              route_hint: "fast_control",
              execution_mode: "control_complete",
            } satisfies ModalInternalResponse;
          })
        : normalized.lane === "agent" && effectiveRouteHint === "cf_browser_first"
          ? await step.do("cf-browser-first-route", async () => {
              deps.log("feishu.workflow.browser_route", {
                correlation_id: normalized.correlation_id,
                browser_provider: "modal_fallback_pending_cf_browser",
                ...deps.buildRoutingLogFields(normalized, {
                  ...planned,
                  route_hint: effectiveRouteHint as ModalInternalResponse["route_hint"],
                }),
              });
              return deps.invokeAgentExec(
                env,
                normalized,
                {
                  ...planned,
                  route_decision_reason: deps.trim(planned?.route_decision_reason),
                  fallback_reason: deps.trim(planned?.route_decision_reason),
                  provider_async_eligible: !!planned?.provider_async_eligible,
                  provider_async_observed: !!planned?.provider_async_observed,
                },
                sitePrefetch,
              );
            })
          : normalized.lane === "agent"
            ? await step.do(
                planned?.external_exec_candidate ? "cf-ai-exec" : "modal-agent-exec",
                { timeout: `${deps.modalAgentExecTimeoutSeconds} seconds` },
                async () => {
                  if (planned?.external_exec_candidate) {
                    deps.log("feishu.workflow.exec_delegation_candidate", {
                      correlation_id: normalized.correlation_id,
                      execution_mode: deps.trim(planned.execution_mode) || "deferred_reconcile",
                      route_decision_reason: deps.trim(planned.route_decision_reason),
                      provider_async_eligible: !!planned.provider_async_eligible,
                      provider_async_observed: !!planned.provider_async_observed,
                      provider_plan: planned.provider_plan || {},
                      ...deps.buildRoutingLogFields(normalized, {
                        ...planned,
                        route_hint: effectiveRouteHint as ModalInternalResponse["route_hint"],
                      }),
                    });
                    try {
                      return await deps.executeCloudflareAiExec(env, normalized, planned);
                    } catch (error) {
                      const fallbackReason = deps.classifyCfAiExecFallbackReason(error, planned);
                      const fallbackPayload = {
                        correlation_id: normalized.correlation_id,
                        session_key: normalized.session_key,
                        event_id: normalized.event_id,
                        route_decision_reason: deps.trim(planned.route_decision_reason),
                        fallback_reason: fallbackReason,
                        fallback_path: "modal_internal_direct",
                        error: error instanceof Error ? error.message : String(error),
                        ...deps.buildRoutingLogFields(normalized, {
                          ...planned,
                          route_hint: effectiveRouteHint as ModalInternalResponse["route_hint"],
                          gateway_error_class: fallbackReason,
                          misroute_detected: fallbackReason === "misrouted_request_class",
                        }),
                      };
                      deps.log("feishu.cf_ai_exec.fallback", fallbackPayload);
                      writeFeishuAnalyticsEvent(env, "feishu.cf_ai_exec.fallback", fallbackPayload);
                      const fallbackInternal = await deps.invokeAgentExec(
                        env,
                        normalized,
                        {
                          ...planned,
                          route_decision_reason: deps.trim(planned.route_decision_reason),
                          fallback_reason: fallbackReason,
                          provider_async_eligible: !!planned.provider_async_eligible,
                          provider_async_observed: !!planned.provider_async_observed,
                          gateway_error_class: fallbackReason,
                          misroute_detected: fallbackReason === "misrouted_request_class",
                        },
                        sitePrefetch,
                      );
                      const fallbackDonePayload = {
                        correlation_id: normalized.correlation_id,
                        session_key: normalized.session_key,
                        event_id: normalized.event_id,
                        execution_mode: fallbackInternal.execution_mode || "modal_heavy_exec",
                        route_decision_reason:
                          deps.trim(fallbackInternal.route_decision_reason) || deps.trim(planned.route_decision_reason),
                        fallback_reason: deps.trim(fallbackInternal.fallback_reason) || fallbackReason,
                        fallback_path: "modal_internal_direct",
                        success: true,
                        ...deps.buildRoutingLogFields(normalized, {
                          ...planned,
                          ...fallbackInternal,
                          route_hint: (fallbackInternal.route_hint || effectiveRouteHint) as ModalInternalResponse["route_hint"],
                          gateway_error_class: deps.trim(fallbackInternal.fallback_reason) || fallbackReason,
                          misroute_detected:
                            (deps.trim(fallbackInternal.fallback_reason) || fallbackReason) ===
                            "misrouted_request_class",
                        }),
                      };
                      deps.log("feishu.cf_ai_exec.fallback.done", fallbackDonePayload);
                      writeFeishuAnalyticsEvent(env, "feishu.cf_ai_exec.fallback.done", fallbackDonePayload);
                      return fallbackInternal;
                    }
                  }
                  return deps.invokeAgentExec(
                    env,
                    normalized,
                    {
                      ...planned,
                      route_decision_reason: deps.trim(planned?.route_decision_reason),
                      fallback_reason: deps.trim(
                        planned?.external_exec_candidate ? "cf_execution_error" : planned?.route_decision_reason,
                      ),
                      provider_async_eligible: !!planned?.provider_async_eligible,
                      provider_async_observed: !!planned?.provider_async_observed,
                      gateway_error_class: planned?.external_exec_candidate ? "cf_execution_error" : "",
                      misroute_detected: false,
                    },
                    sitePrefetch,
                  );
                },
              )
            : { status: "ignored", send_plan: [], route_hint: effectiveRouteHint };

  let executablePlan = deps.getExecutablePlan(internal);
  const finalResponse = deps.trim(internal.final_response);
  
  function buildTextSendPlan(content: string): Array<Record<string, JsonValue>> {
    return [{ kind: "text", content }];
  }
  
  if (executablePlan.length === 0 && finalResponse) {
    executablePlan = buildTextSendPlan(finalResponse);
    deps.log("feishu.workflow.send_plan_fallback", {
      correlation_id: normalized.correlation_id,
      reason: "empty_send_plan_with_final_response",
      final_response_length: finalResponse.length,
      operation_count: executablePlan.length,
    });
  }
  if (executablePlan.length === 0 && normalized.lane === "agent" && !finalResponse) {
    executablePlan = buildTextSendPlan("Message received, processing...");
    deps.log("feishu.workflow.send_plan_fallback_default", {
      correlation_id: normalized.correlation_id,
      reason: "modal_stub_response_no_final_response",
      execution_mode: internal.execution_mode || "modal_heavy_exec",
      internal_status: internal.status,
    });
  }
  await step.do("send-feishu-response", async () => {
    const sendStartedAt = Date.now();
    deps.log("feishu.workflow.send.start", {
      correlation_id: normalized.correlation_id,
      chat_id: normalized.chat_id,
      user_id: normalized.user_id,
      execution_mode: internal.execution_mode || "modal_heavy_exec",
      operation_count: executablePlan.length,
      plan_contents: executablePlan.map(op => ({ kind: deps.trim(op.kind), content: deps.trim(op.content)?.slice(0, 50) || "..." })),
      final_response: deps.trim(internal.final_response)?.slice(0, 100) || "none",
    });
    try {
      await deps.executeFeishuOperations(env, normalized, executablePlan);
      const cfSendElapsedMs = Date.now() - sendStartedAt;
      await deps.persistSessionCacheAfterResponse(env, normalized, internal);
      const workflowSendPayload = {
        correlation_id: normalized.correlation_id,
        chat_id: normalized.chat_id,
        execution_mode: internal.execution_mode || "modal_heavy_exec",
        cf_send_elapsed_ms: cfSendElapsedMs,
        operation_count: executablePlan.length,
        send_status: "success",
        ...deps.buildRoutingLogFields(normalized, internal),
      };
      deps.log("feishu.workflow.send.done", workflowSendPayload);
      writeFeishuAnalyticsEvent(env, "feishu.workflow.send.done", workflowSendPayload);
    } catch (error) {
      deps.log("feishu.workflow.send.error", {
        correlation_id: normalized.correlation_id,
        chat_id: normalized.chat_id,
        execution_mode: internal.execution_mode || "modal_heavy_exec",
        operation_count: executablePlan.length,
        send_status: "failed",
        error: error instanceof Error ? error.message : String(error),
        ...deps.buildRoutingLogFields(normalized, internal),
      });
      throw error;
    }
    
    if (normalized.lane === "agent" && internal.reconcile_required !== false) {
      const pending = deps.buildPendingReconcile(normalized, internal);
      if (pending) {
        await deps.enqueuePendingReconcile(env, normalized.session_key, pending);
        deps.log("feishu.reconcile.enqueue", {
          session_key: normalized.session_key,
          correlation_id: pending.correlation_id,
          operation_count: pending.operation_kinds.length,
        });
      }
    }
    return { delivered: true };
  });

  await step.do("emit-final-log", async () => {
    deps.log("feishu.workflow.done", {
      correlation_id: normalized.correlation_id,
      lane: normalized.lane,
      execution_mode: internal.execution_mode || "modal_heavy_exec",
      retry_attempt_count: internal.retry_attempt_count || 1,
      retry_class: deps.trim(internal.retry_class) || "none",
      ...deps.buildRoutingLogFields(normalized, internal),
    });
    return { ok: true };
  });

  return {
    status: internal.status,
    correlation_id: normalized.correlation_id,
    route_hint: (internal.route_hint || normalized.route_hint) as JsonValue,
    execution_mode: (internal.execution_mode || "modal_heavy_exec") as JsonValue,
  };
}
