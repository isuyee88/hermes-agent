import type { Env, FeishuNormalizedPayload, JsonValue, ModalInternalResponse } from "../runtime";
import { writeFeishuAnalyticsEvent } from "../observability/analytics-engine";
import {
  classifyGatewayFeedbackErrorKind,
  deriveGatewayFeedbackScore,
  recordGatewayFeedback,
} from "../model-catalog/feedback";

type CfAiExecDeps = {
  trim: (value: unknown) => string;
  log: (event: string, extra: Record<string, unknown>) => void;
  parseBoolean: (value: unknown, fallback?: boolean) => boolean;
  parsePositiveInt: (value: unknown, fallback: number, min?: number, max?: number) => number;
  isFreeishModelName: (value: unknown) => boolean;
  isTextGatewayRequestClass: (value: string) => boolean;
  getCloudflareAiGatewayBase: (env: Env) => string;
  getCloudflareAiGatewayRoot: (env: Env) => string;
  sha256Hex: (input: string) => Promise<string>;
  buildAiGatewayMetadata: (
    normalized: FeishuNormalizedPayload,
    planned: Partial<ModalInternalResponse>,
  ) => Record<string, JsonValue>;
  buildRoutingLogFields: (
    normalized: FeishuNormalizedPayload,
    overrides?: Partial<ModalInternalResponse>,
  ) => Record<string, unknown>;
  buildTextSendPlan: (content: string) => Array<Record<string, JsonValue>>;
  inferRouteDecisionReason: (
    normalized: FeishuNormalizedPayload,
    routeHint: FeishuNormalizedPayload["route_hint"],
  ) => string;
  classifyCfAiExecFallbackReason: (error: unknown, planned?: Partial<ModalInternalResponse>) => string;
};

function normalizeChatMessages(raw: unknown): Array<Record<string, JsonValue>> {
  if (!Array.isArray(raw)) {
    return [];
  }
  const out: Array<Record<string, JsonValue>> = [];
  for (const item of raw) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }
    const role = typeof (item as Record<string, JsonValue>).role === "string" ? (item as Record<string, JsonValue>).role : "";
    const content =
      typeof (item as Record<string, JsonValue>).content === "string" ? (item as Record<string, JsonValue>).content : "";
    if (!role.trim() || !content.trim()) {
      continue;
    }
    out.push({ role: role.trim(), content: content.trim() });
  }
  return out;
}

function extractLatestUserText(messages: Array<Record<string, JsonValue>>): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const item = messages[index];
    if (item.role === "user" && typeof item.content === "string" && item.content.trim()) {
      return item.content.trim();
    }
  }
  return "";
}

function isContextDependentPrompt(text: string): boolean {
  const normalized = text.trim().toLowerCase();
  if (!normalized) {
    return true;
  }
  return /(https?:\/\/|继续|刚才|上面|上一|上述|前面|这个|那个|those|that one|this one|continue|previous|above|based on)/i.test(
    normalized,
  );
}

function extractAssistantTextFromChatCompletion(
  payload: Record<string, JsonValue>,
  trim: (value: unknown) => string,
): string {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return "";
  }
  const first = choices[0];
  if (!first || typeof first !== "object" || Array.isArray(first)) {
    return "";
  }
  const message = (first as Record<string, JsonValue>).message;
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return "";
  }
  const content = (message as Record<string, JsonValue>).content;
  if (typeof content === "string") {
    return trim(content);
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .map((entry) => {
      if (typeof entry === "string") {
        return trim(entry);
      }
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
        return "";
      }
      return trim((entry as Record<string, JsonValue>).text);
    })
    .filter(Boolean)
    .join("\n")
    .trim();
}

function resolveGatewayResponseProvider(
  response: Response,
  payload: Record<string, JsonValue>,
  trim: (value: unknown) => string,
  fallbackProvider: string,
): string {
  return (
    trim(response.headers.get("cf-aig-provider")) ||
    trim((payload.provider as string | undefined) ?? "") ||
    trim((payload.provider_name as string | undefined) ?? "") ||
    fallbackProvider
  );
}

function resolveGatewayResponseModel(
  response: Response,
  payload: Record<string, JsonValue>,
  trim: (value: unknown) => string,
  fallbackModel: string,
): string {
  return trim(response.headers.get("cf-aig-model")) || trim(payload.model) || fallbackModel;
}

export async function executeCloudflareAiExec(
  env: Env,
  normalized: FeishuNormalizedPayload,
  planned: ModalInternalResponse,
  deps: CfAiExecDeps,
): Promise<ModalInternalResponse> {
  const token = deps.trim(env.CLOUDFLARE_AI_GATEWAY_API_KEY) || deps.trim(env.CLOUDFLARE_API_TOKEN);
  const llmRequest = (planned.llm_request ?? {}) as Record<string, JsonValue>;
  const providerPlan = (planned.provider_plan ?? {}) as Record<string, JsonValue>;
  const routeDecisionReason =
    deps.trim(planned.route_decision_reason) ||
    deps.inferRouteDecisionReason(normalized, planned.route_hint || normalized.route_hint);
  const providerAsyncEligible = !!planned.provider_async_eligible;
  const providerAsyncObserved = !!planned.provider_async_observed;
  const requestClass = deps.trim(planned.request_class) || normalized.request_class;
  const routeFamily = deps.trim(planned.route_family) || normalized.route_family;
  const gatewayRouteName = deps.trim(planned.gateway_route_name) || normalized.gateway_route_name;
  const gatewayEligible =
    typeof planned.gateway_eligible === "boolean" ? planned.gateway_eligible : normalized.gateway_eligible;
  const modalityProfile = deps.trim(planned.modality_profile) || normalized.modality_profile;
  const toolset =
    Array.isArray(planned.toolset) && planned.toolset.length > 0 ? planned.toolset : normalized.toolset;
  const contentModalities =
    Array.isArray(planned.content_modalities) && planned.content_modalities.length > 0
      ? planned.content_modalities
      : normalized.content_modalities;
  const strictClassEnforcement = deps.parseBoolean(env.HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT, true);
  const dynamicRouteExecutionEnabled = deps.parseBoolean(env.HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED, false);
  const requestedModel = deps.trim(llmRequest.model);
  const plannedModel = deps.trim(providerPlan.model);
  const model =
    requestedModel && !(plannedModel && deps.isFreeishModelName(requestedModel) && !deps.isFreeishModelName(plannedModel))
      ? requestedModel
      : plannedModel || requestedModel || "openrouter/free";
  const messages = normalizeChatMessages(llmRequest.messages);

  if (!token || messages.length === 0) {
    throw new Error(`cf_ai_exec_unavailable:${token ? "missing_messages" : "missing_token"}`);
  }
  if (strictClassEnforcement && (!gatewayEligible || !deps.isTextGatewayRequestClass(requestClass))) {
    throw new Error(`cf_ai_exec_misrouted_request_class:${requestClass || "unknown"}`);
  }

  const startedAt = Date.now();
  const gatewayBase = deps.trim(providerPlan.base_url) || deps.getCloudflareAiGatewayBase(env);
  const provider = deps.trim(providerPlan.provider).toLowerCase();
  const byokAlias = deps.trim(providerPlan.byok_alias) || "default";
  const requestTimeoutMs = deps.parsePositiveInt(providerPlan.request_timeout_ms, 4500, 1000, 30000);
  const externalWaitEnabled = deps.parseBoolean(env.HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED, false);
  const externalWaitMaxMs = deps.parsePositiveInt(env.HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS, 12000, 1000, 120000);
  const maxAttempts = deps.parsePositiveInt(providerPlan.max_attempts, 1, 1, 5);
  const retryDelayMs = deps.parsePositiveInt(providerPlan.retry_delay_ms, 250, 0, 5000);
  const backoff = (() => {
    const candidate = deps.trim(providerPlan.backoff).toLowerCase();
    return candidate === "constant" || candidate === "exponential" ? candidate : "linear";
  })();
  const useDynamicRoute =
    dynamicRouteExecutionEnabled && gatewayEligible && deps.isTextGatewayRequestClass(requestClass) && !!gatewayRouteName;
  const cacheMode = (() => {
    const candidate = deps.trim(providerPlan.cache_mode).toLowerCase();
    return candidate === "ttl" ? "ttl" : "skip";
  })();
  const cacheScope = (() => {
    const candidate = deps.trim(providerPlan.cache_scope).toLowerCase();
    return candidate === "chat" || candidate === "global" ? candidate : "user";
  })();
  const cacheTtlSeconds = deps.parsePositiveInt(providerPlan.cache_ttl_seconds, 300, 60, 31 * 24 * 3600);
  const fallbackModel = deps.trim(providerPlan.fallback_model);
  const fallbackProvider = deps.trim(providerPlan.fallback_provider).toLowerCase() || provider;
  const latestUserText = extractLatestUserText(messages);
  const statelessCacheCandidate =
    cacheMode === "ttl" &&
    gatewayEligible === true &&
    deps.isTextGatewayRequestClass(requestClass) &&
    normalized.requires_browser !== true &&
    normalized.requires_media_hydration !== true &&
    normalized.requires_tools !== true &&
    latestUserText.length > 0 &&
    latestUserText.length <= 220 &&
    messages.length <= 12 &&
    !isContextDependentPrompt(latestUserText);
  const derivedCacheEligible = statelessCacheCandidate;
  const effectiveCacheScope = statelessCacheCandidate ? "global" : cacheScope;
  const derivedCapabilityMatch =
    typeof planned.capability_match === "boolean"
      ? planned.capability_match
      : gatewayEligible && normalized.requires_browser !== true && normalized.requires_media_hydration !== true;
  let aiCallCount = 0;

  if (externalWaitEnabled && !providerAsyncObserved && requestTimeoutMs > externalWaitMaxMs) {
    throw new Error(`cf_ai_exec_wait_budget_exceeded:${requestTimeoutMs}`);
  }

  const requestOnce = async (
    targetProvider: string,
    targetModel: string,
    reason: "primary" | "fallback",
  ): Promise<{
    response: Response;
    payload: Record<string, JsonValue>;
    requestUrl: string;
    cacheStatus: string;
    gatewayLogId: string;
    elapsedMs: number;
    targetProvider: string;
    targetModel: string;
    resolvedProvider: string;
    resolvedModel: string;
    cacheFingerprint: string;
    cacheScope: string;
    latestUserTextLength: number;
  }> => {
    aiCallCount += 1;
    const requestUrl = useDynamicRoute
      ? `${gatewayBase}/chat/completions`
      : targetProvider === "openrouter"
        ? `${deps.getCloudflareAiGatewayRoot(env)}/openrouter/chat/completions`
        : `${gatewayBase}/chat/completions`;
    const requestModel = useDynamicRoute ? `dynamic/${gatewayRouteName}` : targetModel;
    const attemptStartedAt = Date.now();
    const metadata = deps.buildAiGatewayMetadata(normalized, {
      ...planned,
      request_class: requestClass as ModalInternalResponse["request_class"],
      route_family: routeFamily as ModalInternalResponse["route_family"],
      gateway_route_name: gatewayRouteName,
      gateway_eligible: gatewayEligible,
      modality_profile: modalityProfile,
      toolset,
      content_modalities: contentModalities,
      route_decision_reason: routeDecisionReason,
      provider_usage: {
        execution_target: useDynamicRoute ? "dynamic_route" : "direct_provider",
        planned_provider: targetProvider || "openrouter",
        planned_model: targetModel,
        requested_model: requestModel,
      },
    });
    const headers: Record<string, string> = {
      "content-type": "application/json",
      "cf-aig-authorization": `Bearer ${token}`,
      "cf-aig-collect-log-payload": "false",
      "cf-aig-metadata": JSON.stringify(metadata),
      "cf-aig-request-timeout": String(requestTimeoutMs),
      "cf-aig-max-attempts": String(maxAttempts),
      "cf-aig-retry-delay": String(retryDelayMs),
      "cf-aig-backoff": backoff,
    };
    let cacheFingerprint = "";
    if (derivedCacheEligible) {
      headers["cf-aig-cache-ttl"] = String(cacheTtlSeconds);
      const cacheIdentity =
        effectiveCacheScope === "global"
          ? "global"
          : effectiveCacheScope === "chat"
            ? normalized.chat_id || normalized.session_key || "anon-chat"
            : normalized.user_id || normalized.session_key || "anon-user";
      const requestFingerprint = await deps.sha256Hex(
        JSON.stringify({
          provider: useDynamicRoute ? "dynamic_route" : targetProvider,
          model: useDynamicRoute ? gatewayRouteName : targetModel,
          latest_user_text: latestUserText.toLowerCase().replace(/\s+/g, " "),
          personality: deps.trim(providerPlan.personality),
          task_kind: normalized.task_kind,
          request_class: requestClass,
          route_family: routeFamily,
          gateway_route_name: gatewayRouteName,
        }),
      );
      const identityFingerprint = await deps.sha256Hex(cacheIdentity);
      cacheFingerprint = requestFingerprint;
      headers["cf-aig-cache-key"] = `feishu:${effectiveCacheScope}:${identityFingerprint}:${requestFingerprint}`;
    } else {
      headers["cf-aig-skip-cache"] = "true";
    }
    if (!useDynamicRoute && targetProvider === "openrouter" && byokAlias) {
      headers["cf-aig-byok-alias"] = byokAlias;
    }
    try {
      const response = await fetch(requestUrl, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: requestModel,
          messages,
          stream: false,
        }),
      });
      const payload = (await response.json()) as Record<string, JsonValue>;
      const cacheStatus = deps.trim(response.headers.get("cf-aig-cache-status"));
      const gatewayLogId = deps.trim(response.headers.get("cf-aig-log-id"));
      const elapsedMs = Date.now() - attemptStartedAt;
      const resolvedProvider = resolveGatewayResponseProvider(response, payload, deps.trim, targetProvider || "openrouter");
      const resolvedModel = resolveGatewayResponseModel(response, payload, deps.trim, targetModel);
      if (!response.ok) {
        const gatewayErrorClass = deps.classifyCfAiExecFallbackReason(
          new Error(`cf_ai_exec_failed:${response.status}:${JSON.stringify(payload).slice(0, 500)}`),
          planned,
        );
        deps.log("feishu.cf_ai_exec.attempt_failed", {
          correlation_id: normalized.correlation_id,
          reason,
          route_decision_reason: routeDecisionReason,
          provider_async_eligible: providerAsyncEligible,
          provider_async_observed: providerAsyncObserved,
          provider: resolvedProvider || targetProvider || "openrouter",
          model: resolvedModel || targetModel,
          status_code: response.status,
          request_url: requestUrl,
          request_model: requestModel,
          route_execution_mode: useDynamicRoute ? "dynamic_route" : "direct_provider",
          cf_cache_status: cacheStatus || "none",
          ...deps.buildRoutingLogFields(normalized, {
            ...planned,
            request_class: requestClass as ModalInternalResponse["request_class"],
            route_family: routeFamily as ModalInternalResponse["route_family"],
            gateway_route_name: gatewayRouteName,
            gateway_eligible: gatewayEligible,
            modality_profile: modalityProfile,
            toolset,
            content_modalities: contentModalities,
            gateway_error_class: gatewayErrorClass,
            misroute_detected: gatewayErrorClass === "misrouted_request_class",
          }),
        });
        await recordGatewayFeedback(env, {
          provider: resolvedProvider || targetProvider || "openrouter",
          model: resolvedModel || targetModel,
          taskKind: normalized.task_kind || "general",
          routeFamily,
          gatewayRouteName,
          gatewayLogId,
          correlationId: normalized.correlation_id,
          feedback: -1,
          score: deriveGatewayFeedbackScore(-1, elapsedMs, cacheStatus, gatewayErrorClass),
          errorKind: classifyGatewayFeedbackErrorKind(gatewayErrorClass, reason === "fallback" ? "status_error" : ""),
          errorCode: String(response.status),
          errorMessage: JSON.stringify(payload).slice(0, 500),
          latencyMs: elapsedMs,
          ttfbMs: elapsedMs,
          cacheStatus,
          requestUrl,
          metadata: {
            reason,
            status_code: response.status,
            request_model: requestModel,
            route_execution_mode: useDynamicRoute ? "dynamic_route" : "direct_provider",
          },
        });
      }
      return {
        response,
        payload,
        requestUrl,
        cacheStatus,
        gatewayLogId,
        elapsedMs,
        targetProvider,
        targetModel,
        resolvedProvider,
        resolvedModel,
        cacheFingerprint,
        cacheScope: effectiveCacheScope,
        latestUserTextLength: latestUserText.length,
      };
    } catch (error) {
      const gatewayErrorClass = deps.classifyCfAiExecFallbackReason(error, planned);
      await recordGatewayFeedback(env, {
        provider: targetProvider || "openrouter",
        model: targetModel,
        taskKind: normalized.task_kind || "general",
        routeFamily,
        gatewayRouteName,
        correlationId: normalized.correlation_id,
        feedback: -1,
        score: deriveGatewayFeedbackScore(-1, Date.now() - attemptStartedAt, null, gatewayErrorClass),
        errorKind: classifyGatewayFeedbackErrorKind(error, reason === "fallback" ? "transport_error" : ""),
        errorMessage: error instanceof Error ? error.message : String(error),
        latencyMs: Date.now() - attemptStartedAt,
        ttfbMs: Date.now() - attemptStartedAt,
        requestUrl,
        metadata: {
          reason,
          transport_error: true,
          request_model: requestModel,
          route_execution_mode: useDynamicRoute ? "dynamic_route" : "direct_provider",
        },
      });
      throw error;
    }
  };

  let result: Awaited<ReturnType<typeof requestOnce>> | null = null;
  const canFallback = !useDynamicRoute && Boolean(fallbackModel && fallbackModel !== model);
  const tryFallbackRequest = async (
    reason: "status_error" | "empty_response" | "transport_error",
    primaryStatusCode = 0,
  ): Promise<void> => {
    deps.log("feishu.cf_ai_exec.retrying_with_fallback_model", {
      correlation_id: normalized.correlation_id,
      route_decision_reason: routeDecisionReason,
      primary_provider: result?.resolvedProvider || result?.targetProvider || provider || "openrouter",
      primary_model: result?.resolvedModel || model,
      fallback_provider: fallbackProvider || result?.targetProvider || provider || "openrouter",
      fallback_model: fallbackModel,
      status_code: primaryStatusCode || result?.response?.status || 0,
      reason,
      ...deps.buildRoutingLogFields(normalized, planned),
    });
    result = await requestOnce(
      fallbackProvider || result?.targetProvider || provider || "openrouter",
      fallbackModel,
      "fallback",
    );
  };

  try {
    result = await requestOnce(provider || "openrouter", model, "primary");
  } catch (error) {
    if (!canFallback) {
      throw error;
    }
    await tryFallbackRequest("transport_error");
  }
  if (!result) {
    throw new Error("cf_ai_exec_unavailable:missing_result");
  }
  if (!result.response.ok && canFallback && result.targetModel !== fallbackModel) {
    await tryFallbackRequest("status_error", result.response.status || 0);
  }
  if (!result.response.ok) {
    throw new Error(`cf_ai_exec_failed:${result.response.status}:${JSON.stringify(result.payload).slice(0, 500)}`);
  }

  let assistantText = extractAssistantTextFromChatCompletion(result.payload, deps.trim);
  if (!assistantText && canFallback && result.targetModel !== fallbackModel) {
    await recordGatewayFeedback(env, {
      provider: result.resolvedProvider || result.targetProvider || "openrouter",
      model: result.resolvedModel || result.targetModel,
      taskKind: normalized.task_kind || "general",
      routeFamily,
      gatewayRouteName,
      gatewayLogId: result.gatewayLogId,
      correlationId: normalized.correlation_id,
      feedback: -1,
      score: deriveGatewayFeedbackScore(-1, result.elapsedMs, result.cacheStatus, "empty_response"),
      errorKind: "empty_response",
      errorMessage: "cf_ai_exec_empty_response",
      latencyMs: result.elapsedMs,
      ttfbMs: result.elapsedMs,
      cacheStatus: result.cacheStatus,
      requestUrl: result.requestUrl,
      metadata: {
        fallback_reason: "empty_response",
      },
    });
    await tryFallbackRequest("empty_response");
    if (!result.response.ok) {
      throw new Error(`cf_ai_exec_failed:${result.response.status}:${JSON.stringify(result.payload).slice(0, 500)}`);
    }
    assistantText = extractAssistantTextFromChatCompletion(result.payload, deps.trim);
  }
  if (!assistantText) {
    await recordGatewayFeedback(env, {
      provider: result.resolvedProvider || result.targetProvider || "openrouter",
      model: result.resolvedModel || result.targetModel,
      taskKind: normalized.task_kind || "general",
      routeFamily,
      gatewayRouteName,
      gatewayLogId: result.gatewayLogId,
      correlationId: normalized.correlation_id,
      feedback: -1,
      score: deriveGatewayFeedbackScore(-1, result.elapsedMs, result.cacheStatus, "empty_response"),
      errorKind: "empty_response",
      errorMessage: "cf_ai_exec_empty_response",
      latencyMs: result.elapsedMs,
      ttfbMs: result.elapsedMs,
      cacheStatus: result.cacheStatus,
      requestUrl: result.requestUrl,
    });
    throw new Error("cf_ai_exec_empty_response");
  }

  const usage =
    result.payload.usage && typeof result.payload.usage === "object" && !Array.isArray(result.payload.usage)
      ? (result.payload.usage as Record<string, JsonValue>)
      : {};
  const usedFallbackModel = !!fallbackModel && result.targetModel === fallbackModel;
  const preferredModelSelected =
    typeof planned.preferred_model_selected === "boolean" ? planned.preferred_model_selected : !usedFallbackModel;
  const donePayload = {
    correlation_id: normalized.correlation_id,
    route_decision_reason: routeDecisionReason,
    provider_async_eligible: providerAsyncEligible,
    provider_async_observed: providerAsyncObserved,
    model: result.resolvedModel || result.targetModel,
    provider: result.resolvedProvider || result.targetProvider || "openrouter",
    request_url: result.requestUrl,
    cf_cache_status: result.cacheStatus || "none",
    route_execution_mode: useDynamicRoute ? "dynamic_route" : "direct_provider",
    cf_ai_exec_elapsed_ms: Date.now() - startedAt,
    cache_eligible: derivedCacheEligible,
    cache_scope: result.cacheScope,
    cache_fingerprint: result.cacheFingerprint || undefined,
    latest_user_text_length: result.latestUserTextLength,
    ai_call_count: aiCallCount,
    capability_match: derivedCapabilityMatch,
    preferred_model_selected: preferredModelSelected,
    ...deps.buildRoutingLogFields(normalized, {
      ...planned,
      request_class: requestClass as ModalInternalResponse["request_class"],
      route_family: routeFamily as ModalInternalResponse["route_family"],
      gateway_route_name: gatewayRouteName,
      gateway_eligible: gatewayEligible,
      modality_profile: modalityProfile,
      toolset,
      content_modalities: contentModalities,
    }),
  };
  deps.log("feishu.cf_ai_exec.done", donePayload);
  writeFeishuAnalyticsEvent(env, "feishu.cf_ai_exec.done", donePayload);

  await recordGatewayFeedback(env, {
    provider: result.resolvedProvider || result.targetProvider || "openrouter",
    model: result.resolvedModel || result.targetModel,
    taskKind: normalized.task_kind || "general",
    routeFamily,
    gatewayRouteName,
    gatewayLogId: result.gatewayLogId,
    correlationId: normalized.correlation_id,
    feedback: 1,
    score: deriveGatewayFeedbackScore(1, result.elapsedMs, result.cacheStatus),
    latencyMs: result.elapsedMs,
    ttfbMs: result.elapsedMs,
    cacheStatus: result.cacheStatus,
    requestTokens:
      typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : typeof usage.input_tokens === "number" ? usage.input_tokens : null,
    responseTokens:
      typeof usage.completion_tokens === "number"
        ? usage.completion_tokens
        : typeof usage.output_tokens === "number"
          ? usage.output_tokens
          : null,
    requestUrl: result.requestUrl,
    metadata: {
      reason: "success",
      used_fallback: usedFallbackModel,
      route_execution_mode: useDynamicRoute ? "dynamic_route" : "direct_provider",
    },
  });

  return {
    status: "ok",
    route_hint: (planned.route_hint || normalized.route_hint) as ModalInternalResponse["route_hint"],
    execution_mode: "deferred_reconcile",
    modal_avoided: true,
    final_response: assistantText,
    send_plan: deps.buildTextSendPlan(assistantText),
    action_plan: deps.buildTextSendPlan(assistantText),
    reconcile_required: true,
    request_class: requestClass as ModalInternalResponse["request_class"],
    content_modalities: contentModalities,
    route_family: routeFamily as ModalInternalResponse["route_family"],
    gateway_route_name: gatewayRouteName,
    gateway_eligible: gatewayEligible,
    requires_modal_runtime: false,
    requires_tools: false,
    requires_browser: false,
    requires_media_hydration: false,
    modality_profile: modalityProfile,
    toolset,
    reason_code: deps.trim(planned.reason_code) || normalized.reason_code,
    provider_async_eligible: providerAsyncEligible,
    provider_async_observed: providerAsyncObserved,
    route_decision_reason: routeDecisionReason,
    cache_eligible: derivedCacheEligible,
    cache_status: result.cacheStatus || "none",
    ai_call_count: aiCallCount,
    capability_match: derivedCapabilityMatch,
    preferred_model_selected: preferredModelSelected,
    provider_wait_measurement_mode: "cloudflare_gateway_elapsed_proxy",
    provider_usage: {
      provider: result.resolvedProvider || result.targetProvider || "openrouter",
      response_model: result.resolvedModel || result.targetModel,
      requested_model: useDynamicRoute ? `dynamic/${gatewayRouteName}` : result.targetModel,
      route_execution_mode: useDynamicRoute ? "dynamic_route" : "direct_provider",
      gateway_base_url: gatewayBase,
      request_url: result.requestUrl,
      cf_cache_status: result.cacheStatus || "none",
      completion_usage: usage,
    },
  };
}
