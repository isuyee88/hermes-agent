import type { Env } from "../runtime";
import type {
  ModelCatalogRecord,
  ModelRegistrySyncEntry,
  ModelRuntimePolicyRecord,
} from "./types";

const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";

type OpenRouterModel = {
  id?: string;
  canonical_slug?: string | null;
  name?: string | null;
  created?: number | null;
  context_length?: number | null;
  architecture?: {
    modality?: string | null;
    input_modalities?: string[] | null;
    output_modalities?: string[] | null;
    tokenizer?: string | null;
    instruct_type?: string | null;
  } | null;
  pricing?: Record<string, string | number | null> | null;
  top_provider?: {
    context_length?: number | null;
    max_completion_tokens?: number | null;
    is_moderated?: boolean | null;
  } | null;
  per_request_limits?: Record<string, unknown> | null;
  supported_parameters?: string[] | null;
  default_parameters?: Record<string, unknown> | null;
  knowledge_cutoff?: string | null;
  expiration_date?: string | null;
};

type GatewayRouteDefaults = {
  textPlainRouteName?: string;
  textCodingRouteName?: string;
};

export type OpenRouterSyncBundle = {
  catalog: ModelCatalogRecord;
  policies: ModelRuntimePolicyRecord[];
  syncEntry: ModelRegistrySyncEntry;
};

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function normalizeNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const normalized = trim(value);
  if (!normalized) {
    return null;
  }
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeInteger(value: unknown): number | null {
  const parsed = normalizeNumber(value);
  return parsed === null ? null : Math.trunc(parsed);
}

function normalizeModalities(model: OpenRouterModel): string[] {
  const architecture = model.architecture ?? {};
  const raw = [
    ...(Array.isArray(architecture.input_modalities) ? architecture.input_modalities : []),
    ...(Array.isArray(architecture.output_modalities) ? architecture.output_modalities : []),
  ];
  const deduped = new Set<string>();
  for (const item of raw) {
    const normalized = trim(item).toLowerCase();
    if (normalized) {
      deduped.add(normalized);
    }
  }
  return Array.from(deduped);
}

function hasTextOutput(model: OpenRouterModel): boolean {
  const outputs = model.architecture?.output_modalities;
  return Array.isArray(outputs) && outputs.some((item) => trim(item).toLowerCase() === "text");
}

function hasImageInput(model: OpenRouterModel): boolean {
  const inputs = model.architecture?.input_modalities;
  return Array.isArray(inputs) && inputs.some((item) => trim(item).toLowerCase() === "image");
}

function hasParameter(model: OpenRouterModel, names: string[]): boolean {
  const supported = Array.isArray(model.supported_parameters) ? model.supported_parameters : [];
  const set = new Set(supported.map((item) => trim(item).toLowerCase()).filter(Boolean));
  return names.some((name) => set.has(name));
}

export function isZeroCostOpenRouterModel(model: OpenRouterModel): boolean {
  const prompt = normalizeNumber(model.pricing?.prompt);
  const completion = normalizeNumber(model.pricing?.completion);
  return prompt === 0 && completion === 0;
}

export function filterZeroCostOpenRouterModels(models: OpenRouterModel[]): OpenRouterModel[] {
  return models.filter((model) => isZeroCostOpenRouterModel(model));
}

function inferFunctionType(model: OpenRouterModel): string {
  const modelId = trim(model.id).toLowerCase();
  const modality = trim(model.architecture?.modality).toLowerCase();
  if (modelId.includes("embed") || modality.includes("embedding")) {
    return "embedding";
  }
  if (modelId.includes("image") && !hasTextOutput(model)) {
    return "image";
  }
  if (modelId.includes("code") || modelId.includes("coder")) {
    return "coding";
  }
  if (hasParameter(model, ["reasoning", "include_reasoning"])) {
    return "reasoning";
  }
  return "chat";
}

function inferTaskKinds(model: OpenRouterModel, functionType: string): string[] {
  const taskKinds = new Set<string>();
  taskKinds.add("general");
  if (functionType === "coding") {
    taskKinds.add("coding");
  }
  if (functionType === "reasoning") {
    taskKinds.add("reasoning");
  }
  if (functionType === "embedding") {
    taskKinds.add("embedding");
  }
  if (hasImageInput(model)) {
    taskKinds.add("vision");
  }
  const contextLength =
    normalizeInteger(model.top_provider?.context_length) ??
    normalizeInteger(model.context_length) ??
    0;
  if (contextLength >= 200_000) {
    taskKinds.add("long_context");
  }
  return Array.from(taskKinds);
}

function toPricePerMillion(value: unknown): number | null {
  const parsed = normalizeNumber(value);
  if (parsed === null) {
    return null;
  }
  return Number((parsed * 1_000_000).toFixed(6));
}

function buildParameterSchemaJson(model: OpenRouterModel): string | null {
  const payload = {
    supported_parameters: Array.isArray(model.supported_parameters) ? model.supported_parameters : [],
    default_parameters:
      model.default_parameters && typeof model.default_parameters === "object" ? model.default_parameters : {},
  };
  if (payload.supported_parameters.length === 0 && Object.keys(payload.default_parameters).length === 0) {
    return null;
  }
  return JSON.stringify(payload);
}

function resolveRouteFamily(taskKind: string, functionType: string): string | null {
  if (functionType === "embedding") {
    return null;
  }
  if (taskKind === "vision" && functionType === "image") {
    return "gateway_image";
  }
  return "gateway_text";
}

function resolveGatewayRouteName(taskKind: string, defaults: GatewayRouteDefaults): string | null {
  if (taskKind === "coding") {
    return trim(defaults.textCodingRouteName) || null;
  }
  return trim(defaults.textPlainRouteName) || null;
}

function scorePriority(model: OpenRouterModel, functionType: string): number {
  const contextLength =
    normalizeInteger(model.top_provider?.context_length) ??
    normalizeInteger(model.context_length) ??
    0;
  const maxCompletionTokens = normalizeInteger(model.top_provider?.max_completion_tokens) ?? 0;
  const promptPrice = toPricePerMillion(model.pricing?.prompt) ?? 0;
  const completionPrice = toPricePerMillion(model.pricing?.completion) ?? 0;
  const freeBonus = isZeroCostOpenRouterModel(model) ? 500_000 : 0;
  const codingBonus = functionType === "coding" ? 20_000 : 0;
  const reasoningBonus = functionType === "reasoning" ? 10_000 : 0;
  const visionBonus = hasImageInput(model) ? 5_000 : 0;
  const pricePenalty = Math.round(promptPrice + completionPrice);
  return freeBonus + codingBonus + reasoningBonus + visionBonus + contextLength + maxCompletionTokens - pricePenalty;
}

function buildSelectionHint(model: OpenRouterModel, functionType: string): string {
  if (isZeroCostOpenRouterModel(model)) {
    return "recommended";
  }
  if (functionType === "coding") {
    return "candidate";
  }
  return "fallback";
}

function buildRateLimitHints(model: OpenRouterModel): {
  rateLimitRpm: number | null;
  rateLimitTpm: number | null;
  rateLimitRpd: number | null;
} {
  const raw = model.per_request_limits;
  if (!raw || typeof raw !== "object") {
    return {
      rateLimitRpm: null,
      rateLimitTpm: null,
      rateLimitRpd: null,
    };
  }
  const record = raw as Record<string, unknown>;
  return {
    rateLimitRpm: normalizeInteger(record.requests_per_minute ?? record.rpm ?? record.requestsPerMinute),
    rateLimitTpm: normalizeInteger(record.tokens_per_minute ?? record.tpm ?? record.tokensPerMinute),
    rateLimitRpd: normalizeInteger(record.requests_per_day ?? record.rpd ?? record.requestsPerDay),
  };
}

function buildCatalogRecord(
  model: OpenRouterModel,
  rank: number,
  nowSeconds: number,
): ModelCatalogRecord {
  const modelId = trim(model.id);
  const functionType = inferFunctionType(model);
  const taskKinds = inferTaskKinds(model, functionType);
  const modalities = normalizeModalities(model);
  const parameterSchemaJson = buildParameterSchemaJson(model);
  const maxOutputTokens = normalizeInteger(model.top_provider?.max_completion_tokens);
  const contextWindow =
    normalizeInteger(model.top_provider?.context_length) ??
    normalizeInteger(model.context_length);
  const active = !trim(model.expiration_date);
  return {
    id: `openrouter::${modelId}`,
    provider: "openrouter",
    model: modelId,
    providerModelKey: `openrouter::${modelId}`,
    displayName: trim(model.name) || modelId,
    status: active ? "active" : "inactive",
    hidden: !active,
    isAvailable: active,
    isFree: isZeroCostOpenRouterModel(model),
    rank,
    selectionHint: buildSelectionHint(model, functionType),
    manualPinned: false,
    recentUsed: false,
    recentUsedCount: 0,
    generatedCommand: `/model ${modelId} --provider openrouter`,
    lastProbeAt: normalizeInteger(model.created) ?? nowSeconds,
    recentUsedAt: null,
    lastSyncAt: nowSeconds,
    latencyMs: null,
    contextWindow,
    reasoning: hasParameter(model, ["reasoning", "include_reasoning"]),
    consecutiveFailures: 0,
    failureKind: null,
    lastErrorCode: null,
    lastErrorMessage: null,
    lastFailedAt: null,
    source: "openrouter_api",
    functionType,
    taskKindsJson: JSON.stringify(taskKinds),
    modalitiesJson: JSON.stringify(modalities),
    modelFamily: trim(model.canonical_slug).split("/")[0] || "openrouter",
    modelVersion: trim(model.canonical_slug) || modelId,
    vision: hasImageInput(model),
    toolCalling: hasParameter(model, ["tools", "tool_choice"]),
    structuredOutput: hasParameter(model, ["structured_outputs", "response_format"]),
    streaming: hasTextOutput(model) ? true : null,
    maxOutputTokens,
    inputPricePerMillion: toPricePerMillion(model.pricing?.prompt),
    outputPricePerMillion: toPricePerMillion(model.pricing?.completion),
    cacheReadPricePerMillion: toPricePerMillion(model.pricing?.input_cache_read),
    cacheWritePricePerMillion: toPricePerMillion(model.pricing?.input_cache_write),
    temperatureSupported: hasParameter(model, ["temperature"]),
    topPSupported: hasParameter(model, ["top_p"]),
    jsonModeSupported: hasParameter(model, ["structured_outputs", "response_format"]),
    parameterSchemaJson,
    apiFamily: "openrouter",
    providerRegion: null,
    createdAt: nowSeconds,
    updatedAt: nowSeconds,
  };
}

function buildRuntimePolicies(
  catalog: ModelCatalogRecord,
  model: OpenRouterModel,
  nowSeconds: number,
  defaults: GatewayRouteDefaults,
): ModelRuntimePolicyRecord[] {
  const taskKinds = JSON.parse(catalog.taskKindsJson) as string[];
  const { rateLimitRpm, rateLimitTpm, rateLimitRpd } = buildRateLimitHints(model);
  return taskKinds.map((taskKind, index) => {
    const routeFamily = resolveRouteFamily(taskKind, catalog.functionType ?? "chat");
    const gatewayRouteName = resolveGatewayRouteName(taskKind, defaults);
    const enabled = catalog.isAvailable && routeFamily !== null;
    return {
      id: `${catalog.provider}::${catalog.model}::${taskKind}`,
      provider: catalog.provider,
      model: catalog.model,
      taskKind,
      routeFamily,
      gatewayRouteName,
      priorityRank: (catalog.rank ?? 9999) + index,
      enabled,
      hiddenReason: enabled ? null : "route_family_unavailable",
      fallbackAllowed: true,
      rateLimitRpm,
      rateLimitTpm,
      rateLimitRpd,
      burstLimit: rateLimitRpm ? Math.max(1, Math.round(rateLimitRpm / 10)) : null,
      rateLimitWindowSeconds: rateLimitRpm ? 60 : null,
      cooldownUntil: null,
      cacheMode: taskKind === "embedding" ? "skip" : "ttl",
      cacheTtlSeconds: taskKind === "embedding" ? null : 300,
      cacheScope: "global",
      cacheKeyTemplate: null,
      requestTimeoutMs: taskKind === "reasoning" || taskKind === "long_context" ? 8000 : 4500,
      maxAttempts: 1,
      retryDelayMs: 250,
      backoff: "linear",
      notes: "seeded_from_openrouter_catalog",
      lastRoutePublishAt: null,
      createdAt: nowSeconds,
      updatedAt: nowSeconds,
    };
  });
}

function buildSyncEntry(
  catalog: ModelCatalogRecord,
  policies: ModelRuntimePolicyRecord[],
): ModelRegistrySyncEntry {
  const generalPolicy = policies.find((item) => item.taskKind === "general") ?? policies[0];
  return {
    provider: catalog.provider,
    model: catalog.model,
    displayName: catalog.displayName,
    status: catalog.status,
    hidden: catalog.hidden,
    isAvailable: catalog.isAvailable,
    isFree: catalog.isFree,
    rank: catalog.rank,
    selectionHint: catalog.selectionHint,
    manualPinned: catalog.manualPinned,
    recentUsed: catalog.recentUsed,
    recentUsedCount: catalog.recentUsedCount,
    generatedCommand: catalog.generatedCommand,
    lastProbeAt: catalog.lastProbeAt,
    recentUsedAt: catalog.recentUsedAt,
    lastSyncAt: catalog.lastSyncAt,
    latencyMs: catalog.latencyMs,
    contextWindow: catalog.contextWindow,
    reasoning: catalog.reasoning,
    consecutiveFailures: catalog.consecutiveFailures,
    failureKind: catalog.failureKind,
    lastErrorCode: catalog.lastErrorCode,
    lastErrorMessage: catalog.lastErrorMessage,
    lastFailedAt: catalog.lastFailedAt,
    source: catalog.source,
    functionType: catalog.functionType,
    taskKinds: JSON.parse(catalog.taskKindsJson) as string[],
    modalities: JSON.parse(catalog.modalitiesJson) as string[],
    toolCalling: catalog.toolCalling,
    streaming: catalog.streaming,
    structuredOutput: catalog.structuredOutput,
    vision: catalog.vision,
    maxOutputTokens: catalog.maxOutputTokens,
    inputPricePerMillion: catalog.inputPricePerMillion,
    outputPricePerMillion: catalog.outputPricePerMillion,
    cacheReadPricePerMillion: catalog.cacheReadPricePerMillion,
    cacheWritePricePerMillion: catalog.cacheWritePricePerMillion,
    rateLimitRpm: generalPolicy?.rateLimitRpm ?? null,
    rateLimitTpm: generalPolicy?.rateLimitTpm ?? null,
    rateLimitRpd: generalPolicy?.rateLimitRpd ?? null,
    burstLimit: generalPolicy?.burstLimit ?? null,
    rateLimitWindowSeconds: generalPolicy?.rateLimitWindowSeconds ?? null,
    cooldownUntil: generalPolicy?.cooldownUntil ?? null,
    gatewayRouteName: generalPolicy?.gatewayRouteName ?? null,
    routeFamily: generalPolicy?.routeFamily ?? null,
    hiddenReason: generalPolicy?.hiddenReason ?? null,
    temperatureSupported: catalog.temperatureSupported,
    topPSupported: catalog.topPSupported,
    jsonModeSupported: catalog.jsonModeSupported,
    parameterSchemaJson: catalog.parameterSchemaJson,
  };
}

export async function fetchOpenRouterModels(apiKey?: string): Promise<OpenRouterModel[]> {
  const headers = new Headers();
  const normalizedApiKey = trim(apiKey);
  if (normalizedApiKey) {
    headers.set("authorization", `Bearer ${normalizedApiKey}`);
  }
  const response = await fetch(OPENROUTER_MODELS_URL, {
    headers,
    method: "GET",
  });
  if (!response.ok) {
    throw new Error(`openrouter_models_fetch_failed:${response.status}`);
  }
  const payload = (await response.json()) as { data?: OpenRouterModel[] };
  return filterZeroCostOpenRouterModels(Array.isArray(payload.data) ? payload.data : []);
}

export function mapOpenRouterModelsToSyncBundles(
  models: OpenRouterModel[],
  env: Pick<Env, "HERMES_CF_TEXT_PLAIN_ROUTE_NAME" | "HERMES_CF_TEXT_CODING_ROUTE_NAME">,
  nowSeconds: number,
): OpenRouterSyncBundle[] {
  const ranked = [...models]
    .filter((model) => isZeroCostOpenRouterModel(model))
    .map((model) => ({ model, functionType: inferFunctionType(model) }))
    .sort((left, right) => scorePriority(right.model, right.functionType) - scorePriority(left.model, left.functionType));
  return ranked.map(({ model }, index) => {
    const catalog = buildCatalogRecord(model, index + 1, nowSeconds);
    const policies = buildRuntimePolicies(catalog, model, nowSeconds, {
      textPlainRouteName: trim(env.HERMES_CF_TEXT_PLAIN_ROUTE_NAME),
      textCodingRouteName: trim(env.HERMES_CF_TEXT_CODING_ROUTE_NAME),
    });
    return {
      catalog,
      policies,
      syncEntry: buildSyncEntry(catalog, policies),
    };
  });
}
