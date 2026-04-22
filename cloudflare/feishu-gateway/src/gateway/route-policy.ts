import type {
  Env,
  FeishuNormalizedPayload,
  JsonValue,
  ModalInternalResponse,
  SessionProfileState,
  SitePrefetchManifest,
} from "../runtime";
import { resolveDomainSkill } from "./domain-skills";

const SITE_PREFETCH_CONTENT_DIRECT_CONFIDENCE = 0.8;
const SITE_PREFETCH_INTERACTION_DIRECT_CONFIDENCE = 0.75;

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

export function inferSiteExecutionStage(normalized: FeishuNormalizedPayload): string {
  if (normalized.site_category === "site_content") return "content_prefetch";
  if (normalized.site_category === "site_interactive_light") return "interaction_preflight";
  if (normalized.site_category === "site_interactive_heavy") return "heavy_browser_exec";
  return "none";
}

function preferredSitePrefetchUrl(sitePrefetch: SitePrefetchManifest | null | undefined): string {
  if (!sitePrefetch) return "";
  return (
    trim(sitePrefetch.final_url) ||
    sitePrefetch.candidate_urls.map((item) => trim(item)).find(Boolean) ||
    trim(sitePrefetch.target_url)
  );
}

export function buildSitePrefetchDirectRule(sitePrefetch: SitePrefetchManifest | null | undefined): {
  active: boolean;
  mode: "none" | "content_direct" | "interaction_direct";
  target_url: string;
  instruction: string;
} {
  if (!sitePrefetch || sitePrefetch.status !== "completed") {
    return { active: false, mode: "none", target_url: "", instruction: "" };
  }
  const targetUrl = preferredSitePrefetchUrl(sitePrefetch);
  if (!targetUrl) {
    return { active: false, mode: "none", target_url: "", instruction: "" };
  }
  if (sitePrefetch.category === "site_content" && sitePrefetch.confidence >= SITE_PREFETCH_CONTENT_DIRECT_CONFIDENCE) {
    return {
      active: true,
      mode: "content_direct",
      target_url: targetUrl,
      instruction: `Start from ${targetUrl} directly. Do not explore the homepage first unless this URL fails.`,
    };
  }
  if (
    sitePrefetch.category === "site_interactive_light" &&
    sitePrefetch.confidence >= SITE_PREFETCH_INTERACTION_DIRECT_CONFIDENCE
  ) {
    return {
      active: true,
      mode: "interaction_direct",
      target_url: targetUrl,
      instruction: `Open ${targetUrl} directly before any generic site navigation or search.`,
    };
  }
  return { active: false, mode: "none", target_url: "", instruction: "" };
}

function buildDfmeaContext(
  normalized: FeishuNormalizedPayload,
  overrides: Partial<ModalInternalResponse> = {},
): {
  dfmea_failure_mode: ModalInternalResponse["dfmea_failure_mode"];
  dfmea_control_phase: ModalInternalResponse["dfmea_control_phase"];
  dfmea_detection_signal: string;
  dfmea_control_action: string;
  dfmea_severity: number;
} {
  const rawGatewayErrorClass = trim(overrides.gateway_error_class) || trim(overrides.fallback_reason);
  const gatewayErrorClass = normalizeGatewayErrorClass(rawGatewayErrorClass);
  const misrouteDetected = !!overrides.misroute_detected || gatewayErrorClass === "misrouted_request_class";
  if (misrouteDetected) {
    return {
      dfmea_failure_mode: "gateway_misroute",
      dfmea_control_phase: "classification",
      dfmea_detection_signal: "strict_request_class_route_guard",
      dfmea_control_action: "route_to_modal_and_mark_misroute",
      dfmea_severity: 9,
    };
  }
  if (gatewayErrorClass === "payload_incompatible" || gatewayErrorClass === "provider_model_not_found" || gatewayErrorClass === "catalog_stale") {
    return {
      dfmea_failure_mode: "provider_config_invalid",
      dfmea_control_phase: "gateway_preflight",
      dfmea_detection_signal: "gateway_status_code_or_model_reject",
      dfmea_control_action: "quarantine_provider_route",
      dfmea_severity: 8,
    };
  }
  if (gatewayErrorClass === "provider_permission_denied") {
    return {
      dfmea_failure_mode: "provider_auth_invalid",
      dfmea_control_phase: "gateway_preflight",
      dfmea_detection_signal: "auth_validation",
      dfmea_control_action: "fallback_to_modal_and_hold_route",
      dfmea_severity: 8,
    };
  }
  if (gatewayErrorClass === "rate_limited") {
    return {
      dfmea_failure_mode: "provider_rate_exhausted",
      dfmea_control_phase: "gateway_execution",
      dfmea_detection_signal: "provider_429_or_rate_guard",
      dfmea_control_action: "switch_route_or_modal_fallback",
      dfmea_severity: 7,
    };
  }
  if (rawGatewayErrorClass === "provider_sync_wait_required") {
    return {
      dfmea_failure_mode: "provider_sync_wait_budget_exceeded",
      dfmea_control_phase: "gateway_preflight",
      dfmea_detection_signal: "wait_budget_guard",
      dfmea_control_action: "enqueue_or_modal_runtime",
      dfmea_severity: 7,
    };
  }
  if (gatewayErrorClass === "timeout") {
    return {
      dfmea_failure_mode: "provider_timeout",
      dfmea_control_phase: "gateway_execution",
      dfmea_detection_signal: "request_timeout",
      dfmea_control_action: "retry_or_modal_fallback",
      dfmea_severity: 7,
    };
  }
  if (gatewayErrorClass === "upstream_5xx") {
    return {
      dfmea_failure_mode: "provider_http_error",
      dfmea_control_phase: "gateway_execution",
      dfmea_detection_signal: "gateway_http_status",
      dfmea_control_action: "fallback_and_observe",
      dfmea_severity: 6,
    };
  }
  const requiresModalRuntime =
    typeof overrides.requires_modal_runtime === "boolean"
      ? overrides.requires_modal_runtime
      : normalized.requires_modal_runtime;
  const gatewayEligible =
    typeof overrides.gateway_eligible === "boolean" ? overrides.gateway_eligible : normalized.gateway_eligible;
  if (requiresModalRuntime || !gatewayEligible) {
    return {
      dfmea_failure_mode: "modal_runtime_required",
      dfmea_control_phase: "classification",
      dfmea_detection_signal: "request_classifier",
      dfmea_control_action: "route_to_modal_runtime",
      dfmea_severity: 4,
    };
  }
  return {
    dfmea_failure_mode: "nominal_path",
    dfmea_control_phase: trim(overrides.execution_mode) === "deferred_reconcile" ? "gateway_execution" : "classification",
    dfmea_detection_signal: "request_classifier",
    dfmea_control_action: "continue_nominal_path",
    dfmea_severity: 1,
  };
}

export function normalizeGatewayErrorClass(rawValue: unknown): string {
  const normalized = trim(rawValue).toLowerCase();
  if (!normalized) {
    return "";
  }
  if (normalized === "catalog_stale") return "catalog_stale";
  if (
    [
      "provider_permission_denied",
      "provider_model_not_found",
      "rate_limited",
      "upstream_5xx",
      "timeout",
      "payload_incompatible",
      "misrouted_request_class",
    ].includes(normalized)
  ) {
    return normalized;
  }
  if (normalized === "auth_or_secret_missing" || normalized === "provider_auth_error") {
    return "provider_permission_denied";
  }
  if (normalized === "provider_model_invalid") {
    return "provider_model_not_found";
  }
  if (normalized === "rate_exhausted") {
    return "rate_limited";
  }
  if (normalized === "provider_timeout" || normalized === "provider_sync_wait_required") {
    return "timeout";
  }
  if (normalized === "config_hard_fail") {
    return "payload_incompatible";
  }
  if (
    normalized === "provider_http_error" ||
    normalized === "cf_execution_error" ||
    normalized === "provider_empty_response" ||
    normalized === "empty_response"
  ) {
    return "upstream_5xx";
  }
  if (normalized.includes("catalog") && normalized.includes("stale")) {
    return "catalog_stale";
  }
  return normalized;
}

export function buildRoutingLogFields(
  normalized: FeishuNormalizedPayload,
  overrides: Partial<ModalInternalResponse> = {},
): Record<string, unknown> {
  const contentModalities =
    Array.isArray(overrides.content_modalities) && overrides.content_modalities.length > 0
      ? overrides.content_modalities
      : normalized.content_modalities;
  const toolset = Array.isArray(overrides.toolset) && overrides.toolset.length > 0 ? overrides.toolset : normalized.toolset;
  const sitePrefetch = overrides.site_prefetch;
  const directRule = buildSitePrefetchDirectRule(sitePrefetch);
  const resolvedSiteSkill = resolveDomainSkill(normalized.target_domain, normalized.target_url);
  const gatewayErrorClass = normalizeGatewayErrorClass(trim(overrides.gateway_error_class) || trim(overrides.fallback_reason));
  return {
    route_hint: trim(overrides.route_hint) || normalized.route_hint,
    site_category: normalized.site_category,
    site_intent: normalized.site_intent,
    site_execution_stage: inferSiteExecutionStage(normalized),
    target_domain: normalized.target_domain,
    target_url: normalized.target_url,
    site_skill_name: trim((overrides as Record<string, unknown>).site_skill_name) || trim(resolvedSiteSkill?.skill_name),
    task_kind: normalized.task_kind,
    request_class: trim(overrides.request_class) || normalized.request_class,
    content_modalities: contentModalities,
    route_family: trim(overrides.route_family) || normalized.route_family,
    gateway_route_name: trim(overrides.gateway_route_name) || normalized.gateway_route_name,
    gateway_eligible:
      typeof overrides.gateway_eligible === "boolean" ? overrides.gateway_eligible : normalized.gateway_eligible,
    requires_modal_runtime:
      typeof overrides.requires_modal_runtime === "boolean" ? overrides.requires_modal_runtime : normalized.requires_modal_runtime,
    requires_tools: typeof overrides.requires_tools === "boolean" ? overrides.requires_tools : normalized.requires_tools,
    requires_browser: typeof overrides.requires_browser === "boolean" ? overrides.requires_browser : normalized.requires_browser,
    requires_media_hydration:
      typeof overrides.requires_media_hydration === "boolean"
        ? overrides.requires_media_hydration
        : normalized.requires_media_hydration,
    modality_profile: trim(overrides.modality_profile) || normalized.modality_profile,
    toolset,
    reason_code: trim(overrides.reason_code) || normalized.reason_code,
    site_prefetch_mode: trim(sitePrefetch?.mode),
    site_prefetch_status: trim(sitePrefetch?.status),
    site_prefetch_confidence: sitePrefetch?.confidence ?? 0,
    site_prefetch_direct_navigation: directRule.active,
    site_prefetch_direct_mode: directRule.mode,
    site_prefetch_direct_target: directRule.target_url,
    modal_avoided: !!(overrides as Record<string, unknown>).modal_avoided,
    browser_backend_selected: trim((overrides as Record<string, unknown>).browser_backend_selected),
    browser_escalation_reason: trim((overrides as Record<string, unknown>).browser_escalation_reason),
    misroute_detected: !!overrides.misroute_detected,
    gateway_error_class: gatewayErrorClass,
    ...buildDfmeaContext(normalized, overrides),
  };
}

export function buildAiGatewayMetadata(
  normalized: FeishuNormalizedPayload,
  planned: Partial<ModalInternalResponse>,
): Record<string, string | number | boolean> {
  const requestClass = trim(planned.request_class) || normalized.request_class;
  const routeFamily = trim(planned.route_family) || normalized.route_family;
  const routeTarget = trim(planned.gateway_route_name) || normalized.gateway_route_name || "modal_internal";
  const modalityProfile = trim(planned.modality_profile) || normalized.modality_profile || "text";
  const toolset = Array.isArray(planned.toolset) && planned.toolset.length > 0 ? planned.toolset : normalized.toolset;
  const sitePrefetch = planned.site_prefetch;
  const gatewayErrorClass = normalizeGatewayErrorClass(trim(planned.gateway_error_class) || trim(planned.fallback_reason));
  return {
    correlation_id: normalized.correlation_id,
    feishu_event_id: normalized.event_id,
    session_key: normalized.session_key,
    request_class: requestClass || "unclassified",
    route_family: routeFamily || "unknown",
    gateway_route_name: routeTarget,
    hermes_request: [
      `rf=${routeFamily || "unknown"}`,
      `rt=${routeTarget}`,
      `mp=${modalityProfile}`,
      `sc=${normalized.site_category || "none"}`,
      `si=${normalized.site_intent || "general"}`,
      `ts=${(toolset || []).join("+") || "none"}`,
      `ge=${planned.gateway_eligible === true || normalized.gateway_eligible ? "1" : "0"}`,
      `tb=${planned.requires_tools === true || normalized.requires_tools ? "1" : "0"}`,
      `br=${planned.requires_browser === true || normalized.requires_browser ? "1" : "0"}`,
      `mh=${planned.requires_media_hydration === true || normalized.requires_media_hydration ? "1" : "0"}`,
      `rs=${trim(planned.route_decision_reason) || normalized.reason_code || "unknown"}`,
      `spm=${trim(sitePrefetch?.mode) || "none"}`,
      `sps=${trim(sitePrefetch?.status) || "none"}`,
      `gec=${gatewayErrorClass || "none"}`,
    ].join(";"),
  };
}

export function isExternalExecCandidate(
  normalized: FeishuNormalizedPayload,
  routeHint: FeishuNormalizedPayload["route_hint"],
  isTextGatewayRequestClass: (value: string) => boolean,
  messageRequestsBrowser: (message: string) => boolean,
): boolean {
  if (typeof normalized.gateway_eligible === "boolean") {
    return normalized.lane === "agent" && normalized.gateway_eligible && isTextGatewayRequestClass(normalized.request_class);
  }
  if (routeHint !== "modal_heavy_exec") return false;
  const taskKind = trim(normalized.task_kind || normalized.message_type).toLowerCase();
  if (taskKind && taskKind !== "text") return false;
  if (normalized.attachment_refs.length > 0) return false;
  const text = trim(normalized.text);
  if (!text || text.startsWith("/")) return false;
  return !messageRequestsBrowser(text);
}

export function inferRouteDecisionReason(
  normalized: FeishuNormalizedPayload,
  routeHint: FeishuNormalizedPayload["route_hint"],
  messageRequestsBrowser: (message: string) => boolean,
): string {
  if (trim(normalized.reason_code)) return trim(normalized.reason_code);
  if (routeHint === "fast_control") return "planner_forced_modal";
  if (routeHint === "cf_browser_first") return "browser_required";
  const taskKind = trim(normalized.task_kind || normalized.message_type).toLowerCase();
  if (taskKind && taskKind !== "text") return "unsupported_task_shape";
  if (normalized.attachment_refs.length > 0) return "media_hydration_required";
  const text = trim(normalized.text);
  if (!text || text.startsWith("/")) return "unsupported_task_shape";
  return messageRequestsBrowser(text) ? "browser_required" : "plain_text_without_attachments_or_browser";
}

function isProviderAsyncEligible(providerPlan: Record<string, JsonValue>): boolean {
  const provider = trim(providerPlan.provider).toLowerCase();
  const baseUrl = trim(providerPlan.base_url).toLowerCase();
  const model = trim(providerPlan.model).toLowerCase();
  return provider === "nvidia" || baseUrl.includes("integrate.api.nvidia.com") || model.includes("nvidia/");
}

function isProviderAsyncObserved(providerPlan: Record<string, JsonValue>, env: Env): boolean {
  if (parseBoolean(providerPlan.provider_async_observed, false)) return true;
  if (!parseBoolean(env.HERMES_FEISHU_REQUIRE_OBSERVED_ASYNC_CAPABILITY, true)) {
    return parseBoolean(providerPlan.provider_async_eligible, false) || isProviderAsyncEligible(providerPlan);
  }
  return false;
}

export function classifyCfAiExecFallbackReason(
  error: unknown,
  planned: Partial<ModalInternalResponse> | undefined,
  isTextGatewayRequestClass: (value: string) => boolean,
): string {
  const message = trim(error instanceof Error ? error.message : String(error)).toLowerCase();
  const requestClass = trim(planned?.request_class).toLowerCase();
  const gatewayEligible = planned?.gateway_eligible === true;
  if (!message) return "upstream_5xx";
  if (message.includes("missing_token") || message.includes("missing secret") || message.includes("unauthorized")) {
    return "provider_permission_denied";
  }
  if (message.includes("misrouted_request_class")) return "misrouted_request_class";
  if (message.includes("catalog") && message.includes("stale")) return "catalog_stale";
  if (message.includes("empty_response")) return "upstream_5xx";
  if (message.includes("wait_budget_exceeded")) return "timeout";
  if (message.includes("failed:408") || message.includes("timeout")) return "timeout";
  if (message.includes("failed:400")) {
    return gatewayEligible && isTextGatewayRequestClass(requestClass) ? "payload_incompatible" : "misrouted_request_class";
  }
  if (message.includes("failed:401") || message.includes("failed:403")) return "provider_permission_denied";
  if (message.includes("failed:404") || message.includes("failed:422")) return "provider_model_not_found";
  if (message.includes("failed:429")) return "rate_limited";
  if (message.includes("failed:5")) return "upstream_5xx";
  return "upstream_5xx";
}

export function buildProviderPlanFromSessionProfile(
  env: Env,
  profile: SessionProfileState | null | undefined,
  normalized?: Pick<FeishuNormalizedPayload, "request_class" | "gateway_route_name" | "route_family" | "modality_profile">,
  getCloudflareAiGatewayBase?: (env: Env) => string,
): Record<string, JsonValue> {
  const currentModel = trim(profile?.current_model);
  const currentProvider = trim(profile?.current_provider).toLowerCase();
  const effectiveModel =
    currentModel && !isFreeishModelName(currentModel) ? currentModel : "mistralai/mistral-small-3.1-24b-instruct";
  const effectiveProvider = currentProvider || "openrouter";
  const fallbackModel = effectiveModel === "deepseek/deepseek-chat-v3-0324" ? "" : "deepseek/deepseek-chat-v3-0324";
  const profileLines = Array.isArray(profile?.route_status_lines) ? profile.route_status_lines.map((line) => trim(line)).filter(Boolean) : [];
  const plan: Record<string, JsonValue> = {
    mode: "cloudflare_workflow_candidate",
    wait_strategy: "cloudflare_wait",
    task_profile: "plain_text_llm",
    reason: trim(normalized?.request_class) || "plain_text_without_attachments_or_browser",
    model: effectiveModel,
    provider: effectiveProvider,
    base_url: getCloudflareAiGatewayBase ? getCloudflareAiGatewayBase(env) : "",
    request_timeout_ms: 18000,
    max_attempts: 2,
    retry_delay_ms: 250,
    backoff: "linear",
    cache_mode: "ttl",
    cache_scope: "chat",
    cache_ttl_seconds: 300,
    byok_alias: "default",
    external_wait_enabled: parseBoolean(env.HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED, false),
    external_wait_max_ms: parsePositiveInt(env.HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS, 12000, 1000, 120000),
    external_wait_poll_ms: parsePositiveInt(env.HERMES_FEISHU_CF_EXTERNAL_WAIT_POLL_MS, 1000, 100, 30000),
    gateway_route_name: trim(normalized?.gateway_route_name) || "affiliate-general",
    route_family: trim(normalized?.route_family) || "gateway_text",
    request_class: trim(normalized?.request_class) || "text_plain",
    modality_profile: trim(normalized?.modality_profile) || "text",
  };
  if (currentModel) plan.session_model = currentModel;
  if (currentModel) plan.last_model = currentModel;
  if (fallbackModel && fallbackModel !== effectiveModel) {
    plan.fallback_model = fallbackModel;
    plan.fallback_provider = effectiveProvider;
  }
  if (trim(profile?.current_personality)) {
    plan.personality = trim(profile?.current_personality).toLowerCase();
  }
  if (profileLines.length > 0) {
    plan.route_status_lines = profileLines as unknown as JsonValue;
  }
  plan.provider_async_eligible = isProviderAsyncEligible(plan);
  plan.provider_async_observed = isProviderAsyncObserved(plan, env);
  return plan;
}
