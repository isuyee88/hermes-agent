import type {
  Env,
  FeishuNormalizedPayload,
  ModalInternalResponse,
  PendingReconcileItem,
  SitePrefetchManifest,
} from "../runtime";

type DirectRunnerDeps = {
  trim: (value: unknown) => string;
  log: (event: string, extra: Record<string, unknown>) => void;
  toWorkflowInstanceId: (correlationId: string) => string;
  shouldAckReactionInline: (normalized: FeishuNormalizedPayload) => boolean;
  addAckReaction: (env: Env, messageId: string) => Promise<void>;
  handleControlEvent: (env: Env, normalized: FeishuNormalizedPayload) => Promise<ModalInternalResponse>;
  handleAgentCommand: (env: Env, normalized: FeishuNormalizedPayload) => Promise<ModalInternalResponse>;
  getExecutablePlan: (internal: ModalInternalResponse) => Array<Record<string, any>>;
  executeFeishuOperations: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    operations: Array<Record<string, any>>,
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
  buildRoutingLogFields: (
    normalized: FeishuNormalizedPayload,
    overrides?: Partial<ModalInternalResponse>,
  ) => Record<string, unknown>;
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
  shouldEnqueueWorkflowForPlannedPath: (planned: ModalInternalResponse) => boolean;
  executeCloudflareAiExec: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    planned: ModalInternalResponse,
  ) => Promise<ModalInternalResponse>;
  classifyCfAiExecFallbackReason: (error: unknown, planned: ModalInternalResponse) => string;
  invokeAgentExec: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    metadata?: Partial<ModalInternalResponse>,
    sitePrefetch?: SitePrefetchManifest | null,
  ) => Promise<ModalInternalResponse>;
};

function isWorkflowAlreadyExistsError(error: unknown, trim: (value: unknown) => string): boolean {
  const message = trim(error instanceof Error ? error.message : String(error)).toLowerCase();
  return message.includes("instance.already_exists") || message.includes("already exists");
}

export async function enqueueWorkflowInstance(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: Pick<DirectRunnerDeps, "log" | "toWorkflowInstanceId" | "trim">,
): Promise<void> {
  try {
    await env.FEISHU_AGENT_WORKFLOW.create({
      id: deps.toWorkflowInstanceId(normalized.correlation_id),
      params: normalized,
    });
  } catch (error) {
    if (isWorkflowAlreadyExistsError(error, deps.trim)) {
      deps.log("feishu.workflow.enqueue.duplicate", {
        correlation_id: normalized.correlation_id,
        lane: normalized.lane,
        route_hint: normalized.route_hint,
      });
      return;
    }
    throw error;
  }
}

export async function runAckReactionFastPath(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: Pick<DirectRunnerDeps, "addAckReaction" | "log" | "shouldAckReactionInline">,
): Promise<void> {
  if (!deps.shouldAckReactionInline(normalized)) {
    return;
  }
  const startedAt = Date.now();
  try {
    await deps.addAckReaction(env, normalized.message_id);
    deps.log("feishu.ack_reaction.done", {
      correlation_id: normalized.correlation_id,
      message_id: normalized.message_id,
      ack_reaction_elapsed_ms: Date.now() - startedAt,
      execution_path: "worker_direct",
    });
  } catch (error) {
    deps.log("feishu.ack_reaction.error", {
      correlation_id: normalized.correlation_id,
      message_id: normalized.message_id,
      ack_reaction_elapsed_ms: Date.now() - startedAt,
      execution_path: "worker_direct",
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

export async function runDirectWorkerFastPath(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: Pick<
    DirectRunnerDeps,
    | "log"
    | "handleControlEvent"
    | "handleAgentCommand"
    | "getExecutablePlan"
    | "executeFeishuOperations"
    | "persistSessionCacheAfterResponse"
    | "buildPendingReconcile"
    | "enqueuePendingReconcile"
  >,
): Promise<void> {
  const directStartedAt = Date.now();
  deps.log("feishu.direct_fast.start", {
    correlation_id: normalized.correlation_id,
    lane: normalized.lane,
    route_hint: normalized.route_hint,
    task_kind: normalized.task_kind,
  });
  const internal =
    normalized.lane === "control"
      ? ({
          ...(await deps.handleControlEvent(env, normalized)),
          route_hint: "fast_control",
          execution_mode: "control_complete",
        } satisfies ModalInternalResponse)
      : ({
          ...(await deps.handleAgentCommand(env, normalized)),
          route_hint: "fast_control",
          execution_mode: "control_complete",
        } satisfies ModalInternalResponse);

  const executablePlan = deps.getExecutablePlan(internal);
  const sendStartedAt = Date.now();
  await deps.executeFeishuOperations(env, normalized, executablePlan);
  const sendElapsedMs = Date.now() - sendStartedAt;
  await deps.persistSessionCacheAfterResponse(env, normalized, internal);

  if (normalized.lane === "agent" && internal.reconcile_required !== false) {
    const pending = deps.buildPendingReconcile(normalized, internal);
    if (pending) {
      await deps.enqueuePendingReconcile(env, normalized.session_key, pending);
      deps.log("feishu.reconcile.enqueue", {
        session_key: normalized.session_key,
        correlation_id: pending.correlation_id,
        operation_count: pending.operation_kinds.length,
        execution_path: "worker_direct",
      });
    }
  }

  deps.log("feishu.direct_fast.done", {
    correlation_id: normalized.correlation_id,
    lane: normalized.lane,
    route_hint: internal.route_hint || normalized.route_hint,
    execution_mode: internal.execution_mode || "control_complete",
    direct_exec_elapsed_ms: Date.now() - directStartedAt,
    direct_send_elapsed_ms: sendElapsedMs,
    operation_count: executablePlan.length,
  });
}

export async function runDirectWorkerPlannedPath(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: DirectRunnerDeps,
): Promise<void> {
  const directStartedAt = Date.now();
  let delivered = false;
  let effectiveRouteHint = normalized.route_hint;
  let fallbackPath = "none";
  deps.log("feishu.direct_planned.start", {
    correlation_id: normalized.correlation_id,
    lane: normalized.lane,
    ...deps.buildRoutingLogFields(normalized),
  });

  try {
    const sitePrefetch = await deps.maybeBuildSitePrefetch(env, normalized, { allowCrawl: false });
    const planned = await deps.buildEdgeDirectPlan(env, normalized, sitePrefetch);
    effectiveRouteHint = (deps.trim(planned.route_hint) || normalized.route_hint) as FeishuNormalizedPayload["route_hint"];
    deps.log("feishu.direct_planned.plan.done", {
      correlation_id: normalized.correlation_id,
      execution_mode: deps.trim(planned.execution_mode) || "modal_heavy_exec",
      external_exec_candidate: !!planned.external_exec_candidate,
      route_decision_reason: deps.trim(planned.route_decision_reason),
      provider_async_eligible: !!planned.provider_async_eligible,
      provider_async_observed: !!planned.provider_async_observed,
      ...deps.buildRoutingLogFields(normalized, {
        ...planned,
        route_hint: effectiveRouteHint,
      }),
    });

    if (deps.shouldEnqueueWorkflowForPlannedPath(planned)) {
      await enqueueWorkflowInstance(env, normalized, deps);
      deps.log("feishu.direct_planned.workflow_enqueued", {
        correlation_id: normalized.correlation_id,
        enqueue_elapsed_ms: Date.now() - directStartedAt,
        reason: !planned.external_exec_candidate
          ? (deps.trim(planned.route_decision_reason) || "planner_not_external_candidate")
          : "provider_sync_wait_required",
        ...deps.buildRoutingLogFields(normalized, {
          ...planned,
          route_hint: effectiveRouteHint,
        }),
      });
      return;
    }

    let internal: ModalInternalResponse;
    try {
      internal = await deps.executeCloudflareAiExec(env, normalized, planned);
    } catch (error) {
      const fallbackReason = deps.classifyCfAiExecFallbackReason(error, planned);
      deps.log("feishu.direct_planned.cf_ai_exec.fallback", {
        correlation_id: normalized.correlation_id,
        route_decision_reason: deps.trim(planned.route_decision_reason),
        fallback_reason: fallbackReason,
        fallback_path: "modal_internal_direct",
        provider_async_eligible: !!planned.provider_async_eligible,
        provider_async_observed: !!planned.provider_async_observed,
        error: error instanceof Error ? error.message : String(error),
        ...deps.buildRoutingLogFields(normalized, {
          ...planned,
          route_hint: effectiveRouteHint,
          gateway_error_class: fallbackReason,
          misroute_detected: fallbackReason === "misrouted_request_class",
        }),
      });
      try {
        internal = await deps.invokeAgentExec(
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
        fallbackPath = "modal_internal_direct";
        deps.log("feishu.direct_planned.modal_exec.fallback_done", {
          correlation_id: normalized.correlation_id,
          execution_mode: internal.execution_mode || "modal_heavy_exec",
          route_decision_reason: deps.trim(internal.route_decision_reason) || deps.trim(planned.route_decision_reason),
          fallback_reason: deps.trim(internal.fallback_reason) || fallbackReason,
          fallback_path: fallbackPath,
          ...deps.buildRoutingLogFields(normalized, {
            ...planned,
            ...internal,
            route_hint: internal.route_hint || effectiveRouteHint,
            gateway_error_class: deps.trim(internal.fallback_reason) || fallbackReason,
            misroute_detected: (deps.trim(internal.fallback_reason) || fallbackReason) === "misrouted_request_class",
          }),
        });
      } catch (modalError) {
        deps.log("feishu.direct_planned.modal_exec.fallback_error", {
          correlation_id: normalized.correlation_id,
          route_decision_reason: deps.trim(planned.route_decision_reason),
          fallback_reason: fallbackReason,
          fallback_path: "modal_internal_direct",
          error: modalError instanceof Error ? modalError.message : String(modalError),
          source_error: error instanceof Error ? error.message : String(error),
          ...deps.buildRoutingLogFields(normalized, {
            ...planned,
            route_hint: effectiveRouteHint,
            gateway_error_class: fallbackReason,
            misroute_detected: fallbackReason === "misrouted_request_class",
          }),
        });
        await enqueueWorkflowInstance(env, normalized, deps);
        deps.log("feishu.direct_planned.workflow_enqueued", {
          correlation_id: normalized.correlation_id,
          enqueue_elapsed_ms: Date.now() - directStartedAt,
          reason: "cf_ai_exec_and_modal_exec_fallback",
          ...deps.buildRoutingLogFields(normalized, {
            ...planned,
            route_hint: effectiveRouteHint,
            gateway_error_class: fallbackReason,
            misroute_detected: fallbackReason === "misrouted_request_class",
          }),
        });
        return;
      }
    }

    const executablePlan = deps.getExecutablePlan(internal);
    const sendStartedAt = Date.now();
    await deps.executeFeishuOperations(env, normalized, executablePlan);
    delivered = executablePlan.length > 0;
    const sendElapsedMs = Date.now() - sendStartedAt;
    await deps.persistSessionCacheAfterResponse(env, normalized, internal);

    if (internal.reconcile_required !== false) {
      const pending = deps.buildPendingReconcile(normalized, internal);
      if (pending) {
        await deps.enqueuePendingReconcile(env, normalized.session_key, pending);
        deps.log("feishu.reconcile.enqueue", {
          session_key: normalized.session_key,
          correlation_id: pending.correlation_id,
          operation_count: pending.operation_kinds.length,
          execution_path: "worker_direct_planned",
        });
      }
    }

    deps.log("feishu.direct_planned.done", {
      correlation_id: normalized.correlation_id,
      execution_mode: internal.execution_mode || "deferred_reconcile",
      direct_exec_elapsed_ms: Date.now() - directStartedAt,
      direct_send_elapsed_ms: sendElapsedMs,
      operation_count: executablePlan.length,
      route_decision_reason: deps.trim(internal.route_decision_reason) || deps.trim(planned.route_decision_reason),
      fallback_reason: deps.trim(internal.fallback_reason),
      fallback_path: fallbackPath,
      ...deps.buildRoutingLogFields(normalized, {
        ...planned,
        ...internal,
        route_hint: internal.route_hint || effectiveRouteHint,
        gateway_error_class: deps.trim(internal.fallback_reason),
        misroute_detected: deps.trim(internal.fallback_reason) === "misrouted_request_class",
      }),
    });
  } catch (error) {
    if (!delivered) {
      try {
        await enqueueWorkflowInstance(env, normalized, deps);
        deps.log("feishu.direct_planned.workflow_enqueued", {
          correlation_id: normalized.correlation_id,
          enqueue_elapsed_ms: Date.now() - directStartedAt,
          reason: "direct_planned_error",
          error: error instanceof Error ? error.message : String(error),
          ...deps.buildRoutingLogFields(normalized, {
            route_hint: effectiveRouteHint,
          }),
        });
      } catch (enqueueError) {
        deps.log("feishu.direct_planned.workflow_enqueue_failed", {
          correlation_id: normalized.correlation_id,
          route_hint: effectiveRouteHint,
          error: enqueueError instanceof Error ? enqueueError.message : String(enqueueError),
          source_error: error instanceof Error ? error.message : String(error),
        });
      }
    }
    throw error;
  }
}
