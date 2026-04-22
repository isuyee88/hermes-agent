import type {
  FeishuBitableFieldSpec,
  FeishuBitableViewSpec,
  FeishuFieldValue,
  ModelRegistrySyncEntry,
} from "./types";

const FEISHU_BITABLE_TEXT_FIELD = 1;
const FEISHU_BITABLE_NUMBER_FIELD = 2;
const FEISHU_BITABLE_CHECKBOX_FIELD = 7;
const DEFAULT_TABLE_NAME = "Hermes Model Registry";

export const MODEL_REGISTRY_FIELD_SPECS: readonly FeishuBitableFieldSpec[] = [
  { name: "Model", type: FEISHU_BITABLE_TEXT_FIELD, required: true, description: "Canonical model id used by Hermes." },
  { name: "Provider", type: FEISHU_BITABLE_TEXT_FIELD, required: true, description: "Provider slug such as openrouter or nvidia." },
  { name: "Display Name", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Human-readable model name." },
  { name: "Status", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Hermes availability state: active, degraded, inactive, invalid." },
  { name: "Hidden", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether the model should be hidden from default views." },
  { name: "Is Available", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Current Hermes availability signal." },
  { name: "Is Free", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether the model is free-tier eligible." },
  { name: "Rank", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Provider-specific ordering rank." },
  { name: "Selection Hint", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Hermes selection hint such as recommended or fallback." },
  { name: "Manual Pinned", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether the model is operator-pinned." },
  { name: "Recent Used", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether recent Hermes sessions used this model." },
  { name: "Recent Used Count", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Approximate recent usage count derived from Hermes sessions." },
  { name: "Generated Command", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Direct Hermes command used to switch to this model." },
  { name: "Last Probe At", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Unix timestamp of the latest provider probe." },
  { name: "Recent Used At", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Unix timestamp of the most recent Hermes usage." },
  { name: "Last Sync At", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Unix timestamp of the latest Bitable mirror sync." },
  { name: "Latency Ms", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Observed or estimated latency in milliseconds." },
  { name: "Context Window", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Reported context window in tokens." },
  { name: "Reasoning", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether the model supports reasoning mode." },
  { name: "Consecutive Failures", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Consecutive Hermes route failures for this model." },
  { name: "Failure Kind", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Normalized last failure category." },
  { name: "Last Error Code", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Last provider or Hermes error code." },
  { name: "Last Error Message", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Last provider or Hermes error message." },
  { name: "Last Failed At", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Unix timestamp of the most recent failure." },
  { name: "Source", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Registry source, for example routing_state or curated-fallback." },
  { name: "Function Type", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Primary function family such as chat, coding, embedding, or image." },
  { name: "Task Kinds", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Comma-separated task kinds used for dynamic route matching." },
  { name: "Modalities", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Comma-separated modalities supported by the model." },
  { name: "Model Parameters", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Supported parameter knobs summarized for operators." },
  { name: "Tool Calling", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether the model supports tool calling." },
  { name: "Streaming", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether the model supports streaming responses." },
  { name: "Structured Output", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether the model supports JSON or schema-constrained output." },
  { name: "Vision", type: FEISHU_BITABLE_CHECKBOX_FIELD, required: false, description: "Whether the model supports image or multimodal inputs." },
  { name: "Max Output Tokens", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Maximum output token budget reported by the provider." },
  { name: "Input Price / 1M", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Nominal input token price per one million tokens." },
  { name: "Output Price / 1M", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Nominal output token price per one million tokens." },
  { name: "Cache Read Price / 1M", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Provider cache-read price per one million tokens, when exposed." },
  { name: "Cache Write Price / 1M", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Provider cache-write price per one million tokens, when exposed." },
  { name: "Rate Limit RPM", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Configured request-per-minute ceiling for this model policy." },
  { name: "Rate Limit TPM", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Configured token-per-minute ceiling for this model policy." },
  { name: "Rate Limit RPD", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Configured request-per-day ceiling for this model policy." },
  { name: "Burst Limit", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Short-window burst allowance used before cooldown is applied." },
  { name: "Rate Limit Window Seconds", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Window size for burst/rate calculations." },
  { name: "Cooldown Until", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Unix timestamp until which this provider/model should stay cooled down." },
  { name: "Health Score", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Computed health score from feedback, success rate, and latency." },
  { name: "Success Rate", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Rolling success ratio for this provider/model and task." },
  { name: "Error Rate", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Rolling error ratio for this provider/model and task." },
  { name: "Cache Hit Rate", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Rolling Cloudflare AI Gateway cache-hit ratio." },
  { name: "Latency P50 Ms", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Median end-to-end latency in milliseconds." },
  { name: "Latency P95 Ms", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "P95 end-to-end latency in milliseconds." },
  { name: "TTFB P50 Ms", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Median time-to-first-byte in milliseconds." },
  { name: "TTFB P95 Ms", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "P95 time-to-first-byte in milliseconds." },
  { name: "Positive Feedback Count", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Count of positive Cloudflare AI Gateway feedback events." },
  { name: "Negative Feedback Count", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Count of negative Cloudflare AI Gateway feedback events." },
  { name: "Provider Failure Count", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Count of provider-side 5xx or transport failures." },
  { name: "Rate Limit Count", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Count of rate-limit violations observed for this model." },
  { name: "Auth Error Count", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Count of authentication or authorization failures." },
  { name: "Model Not Found Count", type: FEISHU_BITABLE_NUMBER_FIELD, required: false, description: "Count of invalid-model or unavailable-model responses." },
  { name: "Gateway Route Name", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Published Cloudflare AI Gateway route name currently attached to the model." },
  { name: "Route Family", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Hermes logical route family for this model policy." },
  { name: "Hidden Reason", type: FEISHU_BITABLE_TEXT_FIELD, required: false, description: "Operator or automation reason for hiding the model from selection." },
];

export const MODEL_REGISTRY_VIEW_SPECS: readonly FeishuBitableViewSpec[] = [
  {
    name: "All Models",
    view_type: "grid",
    description: "Complete registry sorted by rank.",
    filter_hint: "No filter. Sort by Rank ascending.",
  },
  {
    name: "Active Zero Cost",
    view_type: "grid",
    description: "Current zero-cost models that are active and visible to routing.",
    filter_hint: "Filter Hidden != true, Status = active, Is Free = true, Input Price / 1M = 0, Output Price / 1M = 0. Sort by Rank ascending.",
  },
  {
    name: "Recommended",
    view_type: "grid",
    description: "Recommended models with Hidden unchecked.",
    filter_hint: "Filter Hidden != true and Selection Hint = recommended.",
  },
  {
    name: "Recent Used",
    view_type: "grid",
    description: "Recently used models with Hidden unchecked.",
    filter_hint: "Filter Hidden != true and Recent Used = true. Sort by Recent Used At descending.",
  },
  {
    name: "Routing Health",
    view_type: "grid",
    description: "Models with routing quality metrics and cache visibility.",
    filter_hint: "Filter Hidden != true. Sort by Health Score descending and Error Rate ascending.",
  },
  {
    name: "Hidden or Inactive",
    view_type: "grid",
    description: "Hidden or unavailable models kept for audit.",
    filter_hint: "Filter Hidden = true or Status in (inactive, invalid).",
  },
  {
    name: "Cleanup Queue",
    view_type: "grid",
    description: "Rows waiting for cleanup, including hidden history and malformed records.",
    filter_hint: "Filter Hidden = true or Status in (inactive, invalid) or Provider is empty or Model is empty. Sort by Last Sync At ascending.",
  },
];

function joinList(values: readonly string[] | null | undefined): string | undefined {
  if (!values || values.length === 0) {
    return undefined;
  }
  const normalized = values
    .map((value) => String(value ?? "").trim())
    .filter(Boolean);
  return normalized.length > 0 ? normalized.join(", ") : undefined;
}

function normalizeNumber(value: number | null | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeBoolean(value: boolean | null | undefined): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

export function buildModelParameterSummary(entry: ModelRegistrySyncEntry): string | undefined {
  const parameters: string[] = [];
  if (entry.temperatureSupported) {
    parameters.push("temperature");
  }
  if (entry.topPSupported) {
    parameters.push("top_p");
  }
  if (entry.jsonModeSupported) {
    parameters.push("json_mode");
  }
  if (entry.parameterSchemaJson) {
    parameters.push("parameter_schema");
  }
  return parameters.length > 0 ? parameters.join(", ") : undefined;
}

export function buildModelRegistryFeishuFields(entry: ModelRegistrySyncEntry): Record<string, FeishuFieldValue> {
  const rawFields: Record<string, FeishuFieldValue | undefined> = {
    Provider: entry.provider,
    Model: entry.model,
    "Display Name": entry.displayName ?? undefined,
    "Is Free": normalizeBoolean(entry.isFree),
    "Is Available": normalizeBoolean(entry.isAvailable),
    Rank: normalizeNumber(entry.rank),
    "Last Probe At": normalizeNumber(entry.lastProbeAt),
    "Latency Ms": normalizeNumber(entry.latencyMs),
    "Context Window": normalizeNumber(entry.contextWindow),
    Reasoning: normalizeBoolean(entry.reasoning),
    "Manual Pinned": normalizeBoolean(entry.manualPinned),
    "Selection Hint": entry.selectionHint ?? undefined,
    Status: entry.status ?? undefined,
    Hidden: normalizeBoolean(entry.hidden),
    "Recent Used": normalizeBoolean(entry.recentUsed),
    "Recent Used Count": normalizeNumber(entry.recentUsedCount),
    "Recent Used At": normalizeNumber(entry.recentUsedAt),
    "Generated Command": entry.generatedCommand ?? undefined,
    "Last Error Code": entry.lastErrorCode ?? undefined,
    "Last Error Message": entry.lastErrorMessage ?? undefined,
    "Last Failed At": normalizeNumber(entry.lastFailedAt),
    "Consecutive Failures": normalizeNumber(entry.consecutiveFailures),
    "Failure Kind": entry.failureKind ?? undefined,
    "Last Sync At": normalizeNumber(entry.lastSyncAt),
    Source: entry.source ?? undefined,
    "Function Type": entry.functionType ?? undefined,
    "Task Kinds": joinList(entry.taskKinds),
    Modalities: joinList(entry.modalities),
    "Model Parameters": buildModelParameterSummary(entry),
    "Tool Calling": normalizeBoolean(entry.toolCalling),
    Streaming: normalizeBoolean(entry.streaming),
    "Structured Output": normalizeBoolean(entry.structuredOutput),
    Vision: normalizeBoolean(entry.vision),
    "Max Output Tokens": normalizeNumber(entry.maxOutputTokens),
    "Input Price / 1M": normalizeNumber(entry.inputPricePerMillion),
    "Output Price / 1M": normalizeNumber(entry.outputPricePerMillion),
    "Cache Read Price / 1M": normalizeNumber(entry.cacheReadPricePerMillion),
    "Cache Write Price / 1M": normalizeNumber(entry.cacheWritePricePerMillion),
    "Rate Limit RPM": normalizeNumber(entry.rateLimitRpm),
    "Rate Limit TPM": normalizeNumber(entry.rateLimitTpm),
    "Rate Limit RPD": normalizeNumber(entry.rateLimitRpd),
    "Burst Limit": normalizeNumber(entry.burstLimit),
    "Rate Limit Window Seconds": normalizeNumber(entry.rateLimitWindowSeconds),
    "Cooldown Until": normalizeNumber(entry.cooldownUntil),
    "Health Score": normalizeNumber(entry.healthScore),
    "Success Rate": normalizeNumber(entry.successRate),
    "Error Rate": normalizeNumber(entry.errorRate),
    "Cache Hit Rate": normalizeNumber(entry.cacheHitRate),
    "Latency P50 Ms": normalizeNumber(entry.latencyP50Ms),
    "Latency P95 Ms": normalizeNumber(entry.latencyP95Ms),
    "TTFB P50 Ms": normalizeNumber(entry.ttfbP50Ms),
    "TTFB P95 Ms": normalizeNumber(entry.ttfbP95Ms),
    "Positive Feedback Count": normalizeNumber(entry.positiveFeedbackCount),
    "Negative Feedback Count": normalizeNumber(entry.negativeFeedbackCount),
    "Provider Failure Count": normalizeNumber(entry.providerFailureCount),
    "Rate Limit Count": normalizeNumber(entry.rateLimitCount),
    "Auth Error Count": normalizeNumber(entry.authErrorCount),
    "Model Not Found Count": normalizeNumber(entry.modelNotFoundCount),
    "Gateway Route Name": entry.gatewayRouteName ?? undefined,
    "Route Family": entry.routeFamily ?? undefined,
    "Hidden Reason": entry.hiddenReason ?? undefined,
  };

  return Object.fromEntries(Object.entries(rawFields).filter(([, value]) => value !== undefined));
}

export function getModelRegistryBitableBlueprint(tableName = DEFAULT_TABLE_NAME): {
  table_name: string;
  fields: FeishuBitableFieldSpec[];
  views: FeishuBitableViewSpec[];
  required_field_names: string[];
} {
  return {
    table_name: tableName,
    fields: MODEL_REGISTRY_FIELD_SPECS.map((item) => ({ ...item })),
    views: MODEL_REGISTRY_VIEW_SPECS.map((item) => ({ ...item })),
    required_field_names: MODEL_REGISTRY_FIELD_SPECS.filter((item) => item.required).map((item) => item.name),
  };
}
