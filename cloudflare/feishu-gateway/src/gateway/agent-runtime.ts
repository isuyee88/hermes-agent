import type {
  Env,
  FeishuNormalizedPayload,
  JsonValue,
  ModalInternalResponse,
  SessionHistoryEntry,
  SessionProfileState,
  SitePrefetchManifest,
} from "../runtime";
import { resolveDomainSkill } from "./domain-skills";
import { normalizeGatewayErrorClass } from "./route-policy";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

type AgentRuntimeDeps = {
  trim: (value: unknown) => string;
  log: (event: string, extra: Record<string, unknown>) => void;
  parseBoolean: (value: unknown, fallback?: boolean) => boolean;
  callModalWithReconciles: <T>(
    env: Env,
    path: string,
    normalized: FeishuNormalizedPayload,
    body: Record<string, JsonValue>,
  ) => Promise<T>;
  classifyModalInternalError: (error: unknown) => { retryable: boolean; retryClass: string };
  buildRoutingLogFields: (
    normalized: FeishuNormalizedPayload,
    overrides?: Partial<ModalInternalResponse>,
  ) => Record<string, unknown>;
  peekSessionStateCache: (
    env: Env,
    sessionKey: string,
  ) => Promise<{ profile?: SessionProfileState | null; history?: SessionHistoryEntry[] }>;
  inferRouteDecisionReason: (
    normalized: FeishuNormalizedPayload,
    routeHint: FeishuNormalizedPayload["route_hint"],
  ) => string;
  isExternalExecCandidate: (
    normalized: FeishuNormalizedPayload,
    routeHint: FeishuNormalizedPayload["route_hint"],
  ) => boolean;
  buildProviderPlanFromSessionProfile: (
    env: Env,
    profile: SessionProfileState | null | undefined,
    normalized: FeishuNormalizedPayload,
  ) => Record<string, JsonValue>;
  buildConversationMessages: (
    history: SessionHistoryEntry[],
    latestUserText: string,
    sitePrefetch?: SitePrefetchManifest | null,
  ) => Array<Record<string, JsonValue>>;
  modalAgentExecRetryLimit: number;
};

function toJsonValueObject(record: Record<string, unknown>): Record<string, JsonValue> {
  const out: Record<string, JsonValue> = {};
  for (const [key, value] of Object.entries(record)) {
    if (value === undefined) {
      continue;
    }
    if (
      value === null ||
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean" ||
      Array.isArray(value) ||
      typeof value === "object"
    ) {
      out[key] = value as JsonValue;
    }
  }
  return out;
}

function attachSitePrefetchIngress(
  rawPayload: Record<string, JsonValue>,
  sitePrefetch: SitePrefetchManifest | null | undefined,
): Record<string, JsonValue> {
  if (!sitePrefetch) {
    return rawPayload;
  }
  const cloned = { ...rawPayload };
  const existingIngress = cloned._hermes_ingress;
  const ingress =
    existingIngress && typeof existingIngress === "object" && !Array.isArray(existingIngress)
      ? ({ ...(existingIngress as Record<string, JsonValue>) } as Record<string, JsonValue>)
      : {};
  ingress.site_prefetch = toJsonValueObject(sitePrefetch as unknown as Record<string, unknown>);
  cloned._hermes_ingress = ingress;
  return cloned;
}

function buildAgentIngressPayload(
  normalized: FeishuNormalizedPayload,
  rawMessage: Record<string, JsonValue>,
  sitePrefetch?: SitePrefetchManifest | null,
): Record<string, JsonValue> {
  const resolvedSiteSkill = resolveDomainSkill(normalized.target_domain, normalized.target_url);
  return {
    correlation_id: normalized.correlation_id,
    session_key: normalized.session_key,
    lane: normalized.lane,
    route_hint: normalized.route_hint,
    task_kind: normalized.task_kind,
    request_class: normalized.request_class,
    content_modalities: normalized.content_modalities as unknown as JsonValue,
    route_family: normalized.route_family,
    gateway_route_name: normalized.gateway_route_name,
    gateway_eligible: normalized.gateway_eligible,
    requires_tools: normalized.requires_tools,
    requires_browser: normalized.requires_browser,
    requires_media_hydration: normalized.requires_media_hydration,
    requires_modal_runtime: normalized.requires_modal_runtime,
    modality_profile: normalized.modality_profile,
    toolset: normalized.toolset as unknown as JsonValue,
    reason_code: normalized.reason_code,
    event_id: normalized.event_id,
    event_type: normalized.event_type,
    chat_id: normalized.chat_id,
    chat_type: normalized.chat_type,
    chat_name: normalized.chat_name,
    user_id: normalized.user_id,
    user_name: normalized.user_name,
    message_id: normalized.message_id,
    message_type: normalized.message_type,
    text: normalized.text,
    attachment_refs: normalized.attachment_refs as unknown as JsonValue,
    raw_message: rawMessage,
    site_prefetch: sitePrefetch as unknown as JsonValue,
    site_skill_name: resolvedSiteSkill?.skill_name || "",
  };
}

function buildAgentObservabilityPayload(metadata: Partial<ModalInternalResponse>): Record<string, JsonValue> {
  const providerPlan =
    metadata.provider_plan && typeof metadata.provider_plan === "object" && !Array.isArray(metadata.provider_plan)
      ? (metadata.provider_plan as Record<string, JsonValue>)
      : {};
  const explicitCacheEligible = typeof metadata.cache_eligible === "boolean" ? metadata.cache_eligible : undefined;
  const derivedCacheEligible =
    explicitCacheEligible ?? (trim(providerPlan.cache_mode).toLowerCase() ? trim(providerPlan.cache_mode).toLowerCase() !== "skip" : undefined);
  const explicitProviderAlias = trim(metadata.provider_alias);
  const providerAlias = explicitProviderAlias || trim(providerPlan.provider);
  const explicitPreferredSelected =
    typeof metadata.preferred_model_selected === "boolean" ? metadata.preferred_model_selected : undefined;
  const preferredModelSelected =
    explicitPreferredSelected ?? (metadata.provider_fallback_used === true ? false : undefined);
  const capabilityMatch =
    typeof metadata.capability_match === "boolean"
      ? metadata.capability_match
      : metadata.gateway_eligible === true && metadata.requires_browser !== true && metadata.requires_media_hydration !== true;
  return {
    route_version: trim(metadata.route_version) || "cf_route_policy_v1",
    provider_alias: providerAlias,
    cache_eligible: derivedCacheEligible,
    cache_status: trim(metadata.cache_status),
    ai_call_count: typeof metadata.ai_call_count === "number" ? metadata.ai_call_count : undefined,
    capability_match: capabilityMatch,
    preferred_model_selected: preferredModelSelected,
    model_catalog_version: trim(metadata.model_catalog_version),
    feedback_score_before:
      typeof metadata.feedback_score_before === "number" ? metadata.feedback_score_before : undefined,
    feedback_score_after:
      typeof metadata.feedback_score_after === "number" ? metadata.feedback_score_after : undefined,
    gateway_error_class: normalizeGatewayErrorClass(trim(metadata.gateway_error_class) || trim(metadata.fallback_reason)),
    misroute_detected: !!metadata.misroute_detected,
  };
}

export async function invokeAgentExec(
  env: Env,
  normalized: FeishuNormalizedPayload,
  metadata: Partial<ModalInternalResponse> = {},
  sitePrefetch: SitePrefetchManifest | null | undefined,
  deps: AgentRuntimeDeps,
): Promise<ModalInternalResponse> {
  let attempt = 0;
  let lastRetryClass = "";
  const rawMessage = attachSitePrefetchIngress(normalized.raw_payload, sitePrefetch);
  while (true) {
    attempt += 1;
    try {
      const internal = await deps.callModalWithReconciles<ModalInternalResponse>(
        env,
        "/internal/feishu/agent-exec",
        normalized,
        {
          ...buildAgentIngressPayload(normalized, rawMessage, sitePrefetch),
          ...buildAgentObservabilityPayload(metadata),
          route_decision_reason: deps.trim(metadata.route_decision_reason),
          fallback_reason: deps.trim(metadata.fallback_reason),
          provider_async_eligible: !!metadata.provider_async_eligible,
          provider_async_observed: !!metadata.provider_async_observed,
        },
      );
      return {
        ...internal,
        retry_class: deps.trim(internal.retry_class) || lastRetryClass,
        retry_attempt_count: attempt,
        route_decision_reason:
          deps.trim(internal.route_decision_reason) || deps.trim(metadata.route_decision_reason),
        fallback_reason: deps.trim(internal.fallback_reason) || deps.trim(metadata.fallback_reason),
      };
    } catch (error) {
      const retryInfo = deps.classifyModalInternalError(error);
      lastRetryClass = retryInfo.retryClass;
      deps.log("feishu.modal_internal.error", {
        correlation_id: normalized.correlation_id,
        route_decision_reason: deps.trim(metadata.route_decision_reason),
        fallback_reason: deps.trim(metadata.fallback_reason),
        retry_class: retryInfo.retryClass,
        retryable: retryInfo.retryable,
        retry_attempt_count: attempt,
        error: error instanceof Error ? error.message : String(error),
        ...deps.buildRoutingLogFields(normalized, {
          ...metadata,
          gateway_error_class: deps.trim(metadata.fallback_reason),
          misroute_detected: deps.trim(metadata.fallback_reason) === "misrouted_request_class",
        }),
      });
      if (!retryInfo.retryable || attempt > deps.modalAgentExecRetryLimit) {
        throw error;
      }
      deps.log("feishu.modal_internal.retry", {
        correlation_id: normalized.correlation_id,
        route_decision_reason: deps.trim(metadata.route_decision_reason),
        fallback_reason: deps.trim(metadata.fallback_reason),
        retry_class: retryInfo.retryClass,
        retry_attempt_count: attempt,
        ...deps.buildRoutingLogFields(normalized, metadata),
      });
    }
  }
}

export async function invokeAgentPlan(
  env: Env,
  normalized: FeishuNormalizedPayload,
  sitePrefetch: SitePrefetchManifest | null | undefined,
  deps: Pick<AgentRuntimeDeps, "callModalWithReconciles">,
): Promise<ModalInternalResponse> {
  const rawMessage = attachSitePrefetchIngress(normalized.raw_payload, sitePrefetch);
  return deps.callModalWithReconciles<ModalInternalResponse>(env, "/internal/feishu/agent-plan", normalized, {
    ...buildAgentIngressPayload(normalized, rawMessage, sitePrefetch),
  });
}

export async function buildEdgeDirectPlan(
  env: Env,
  normalized: FeishuNormalizedPayload,
  sitePrefetch: SitePrefetchManifest | null | undefined,
  deps: Pick<
    AgentRuntimeDeps,
    | "trim"
    | "log"
    | "parseBoolean"
    | "peekSessionStateCache"
    | "inferRouteDecisionReason"
    | "isExternalExecCandidate"
    | "buildProviderPlanFromSessionProfile"
    | "buildConversationMessages"
    | "buildRoutingLogFields"
  >,
): Promise<ModalInternalResponse> {
  const cached = await deps.peekSessionStateCache(env, normalized.session_key);
  const routeHint = normalized.route_hint;
  const routeDecisionReason = deps.inferRouteDecisionReason(normalized, routeHint);
  const externalExecCandidate = deps.isExternalExecCandidate(normalized, routeHint);
  const providerPlan = externalExecCandidate
    ? deps.buildProviderPlanFromSessionProfile(env, cached.profile, normalized)
    : {};
  const requiresBrowser = normalized.requires_browser;
  const requiresMediaHydration = normalized.requires_media_hydration;
  const requiresTools = normalized.requires_tools;
  const requiresModalRuntime = normalized.requires_modal_runtime;
  const providerAsyncEligible = externalExecCandidate
    ? deps.parseBoolean(providerPlan.provider_async_eligible, false)
    : false;
  const providerAsyncObserved = externalExecCandidate
    ? deps.parseBoolean(providerPlan.provider_async_observed, false)
    : false;
  const requestedModel = deps.trim(providerPlan.model);
  const llmRequest: Record<string, JsonValue> = externalExecCandidate
    ? {
        model: requestedModel,
        messages: deps.buildConversationMessages(cached.history ?? [], normalized.text, sitePrefetch) as unknown as JsonValue,
      }
    : {};
  const executionMode =
    routeHint === "fast_control"
      ? "control_complete"
      : routeHint === "cf_browser_first"
        ? "cf_browser_first"
        : externalExecCandidate
          ? "deferred_reconcile"
          : "modal_heavy_exec";

  deps.log("feishu.edge_local_plan.done", {
    correlation_id: normalized.correlation_id,
    execution_mode: executionMode,
    external_exec_candidate: externalExecCandidate,
    route_decision_reason: routeDecisionReason,
    provider_async_eligible: providerAsyncEligible,
    provider_async_observed: providerAsyncObserved,
    cached_history_count: (cached.history ?? []).length,
    cached_model: deps.trim(cached.profile?.current_model) || "none",
    cached_personality: deps.trim(cached.profile?.current_personality) || "none",
    ...deps.buildRoutingLogFields(normalized, {
      route_hint: routeHint,
      request_class: normalized.request_class,
      content_modalities: normalized.content_modalities,
      route_family: normalized.route_family,
      gateway_route_name: normalized.gateway_route_name,
      gateway_eligible: normalized.gateway_eligible,
      requires_modal_runtime: requiresModalRuntime,
      requires_tools: requiresTools,
      requires_browser: requiresBrowser,
      requires_media_hydration: requiresMediaHydration,
      modality_profile: normalized.modality_profile,
      toolset: normalized.toolset,
      reason_code: normalized.reason_code,
    }),
  });

  return {
    status: "ok",
    route_hint: routeHint,
    execution_mode: executionMode,
    reconcile_required: externalExecCandidate,
    external_exec_candidate: externalExecCandidate,
    request_class: normalized.request_class,
    content_modalities: normalized.content_modalities,
    route_family: normalized.route_family,
    gateway_route_name: normalized.gateway_route_name,
    gateway_eligible: normalized.gateway_eligible,
    requires_modal_runtime: requiresModalRuntime,
    requires_tools: requiresTools,
    requires_browser: requiresBrowser,
    requires_media_hydration: requiresMediaHydration,
    modality_profile: normalized.modality_profile,
    toolset: normalized.toolset,
    reason_code: normalized.reason_code,
    provider_async_eligible: providerAsyncEligible,
    provider_async_observed: providerAsyncObserved,
    route_decision_reason: routeDecisionReason,
    site_prefetch: sitePrefetch ?? undefined,
    provider_plan: providerPlan,
    llm_request: llmRequest,
    session_state_after: cached.profile
      ? {
          current_model: deps.trim(cached.profile.current_model),
          current_provider: deps.trim(cached.profile.current_provider),
          current_personality: deps.trim(cached.profile.current_personality),
          route_status_lines: cached.profile.route_status_lines ?? [],
        }
      : undefined,
    send_plan: [],
    action_plan: [],
    final_response: "",
  };
}
