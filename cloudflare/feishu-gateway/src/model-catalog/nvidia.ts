import type { Env } from "../runtime";
import type {
  ModelCatalogRecord,
  ModelRegistrySyncEntry,
  ModelRuntimePolicyRecord,
} from "./types";

const NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models";

type NvidiaModel = {
  id?: string;
  object?: string | null;
  created?: number | null;
  owned_by?: string | null;
};

type GatewayRouteDefaults = {
  textPlainRouteName?: string;
  textCodingRouteName?: string;
};

export type NvidiaSyncBundle = {
  catalog: ModelCatalogRecord;
  policies: ModelRuntimePolicyRecord[];
  syncEntry: ModelRegistrySyncEntry;
};

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function normalizeInteger(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  const normalized = trim(value);
  if (!normalized) {
    return null;
  }
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : null;
}

function modelKey(model: NvidiaModel): string {
  return trim(model.id);
}

function inferFunctionType(model: NvidiaModel): string {
  const modelId = modelKey(model).toLowerCase();
  if (modelId.includes("embed") || modelId.includes("retriever")) {
    return "embedding";
  }
  if (modelId.includes("code") || modelId.includes("coder")) {
    return "coding";
  }
  if (modelId.includes("reason") || modelId.includes("thinking")) {
    return "reasoning";
  }
  return "chat";
}

function hasVisionInput(model: NvidiaModel): boolean {
  const modelId = modelKey(model).toLowerCase();
  return (
    modelId.includes("vision") ||
    modelId.includes("-vl") ||
    modelId.includes("/vl") ||
    modelId.includes("kosmos") ||
    modelId.includes("fuyu") ||
    modelId.includes("neva") ||
    modelId.includes("vila")
  );
}

function inferTaskKinds(model: NvidiaModel, functionType: string): string[] {
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
  if (hasVisionInput(model)) {
    taskKinds.add("vision");
  }
  const modelId = modelKey(model).toLowerCase();
  if (modelId.includes("128k") || modelId.includes("200k") || modelId.includes("256k")) {
    taskKinds.add("long_context");
  }
  return Array.from(taskKinds);
}

function normalizeModalities(model: NvidiaModel, functionType: string): string[] {
  if (functionType === "embedding") {
    return ["text"];
  }
  if (hasVisionInput(model)) {
    return ["text", "image"];
  }
  return ["text"];
}

function resolveRouteFamily(taskKind: string, functionType: string): string | null {
  if (functionType === "embedding") {
    return null;
  }
  if (taskKind === "vision") {
    return "gateway_text";
  }
  return "gateway_text";
}

function resolveGatewayRouteName(taskKind: string, defaults: GatewayRouteDefaults): string | null {
  if (taskKind === "coding") {
    return trim(defaults.textCodingRouteName) || null;
  }
  return trim(defaults.textPlainRouteName) || null;
}

function inferModelFamily(model: NvidiaModel): string {
  return trim(model.owned_by) || modelKey(model).split("/")[0] || "nvidia";
}

function inferModelVersion(model: NvidiaModel): string {
  return modelKey(model).split("/").slice(1).join("/") || modelKey(model);
}

function scorePriority(model: NvidiaModel, functionType: string): number {
  const publisher = trim(model.owned_by).toLowerCase();
  const publisherBonus = publisher === "nvidia" ? 10_000 : 0;
  const codingBonus = functionType === "coding" ? 2_000 : 0;
  const reasoningBonus = functionType === "reasoning" ? 1_500 : 0;
  const visionBonus = hasVisionInput(model) ? 1_000 : 0;
  const idScore = modelKey(model)
    .toLowerCase()
    .split("")
    .reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return publisherBonus + codingBonus + reasoningBonus + visionBonus - idScore;
}

function buildCatalogRecord(model: NvidiaModel, rank: number, nowSeconds: number): ModelCatalogRecord {
  const modelId = modelKey(model);
  const functionType = inferFunctionType(model);
  const taskKinds = inferTaskKinds(model, functionType);
  const modalities = normalizeModalities(model, functionType);
  return {
    id: `nvidia::${modelId}`,
    provider: "nvidia",
    model: modelId,
    providerModelKey: `nvidia::${modelId}`,
    displayName: modelId,
    status: "active",
    hidden: false,
    isAvailable: true,
    isFree: false,
    rank,
    selectionHint: "candidate",
    manualPinned: false,
    recentUsed: false,
    recentUsedCount: 0,
    generatedCommand: `/model ${modelId} --provider nvidia`,
    lastProbeAt: normalizeInteger(model.created) ?? nowSeconds,
    recentUsedAt: null,
    lastSyncAt: nowSeconds,
    latencyMs: null,
    contextWindow: null,
    reasoning: functionType === "reasoning",
    consecutiveFailures: 0,
    failureKind: null,
    lastErrorCode: null,
    lastErrorMessage: null,
    lastFailedAt: null,
    source: "nvidia_models_api",
    functionType,
    taskKindsJson: JSON.stringify(taskKinds),
    modalitiesJson: JSON.stringify(modalities),
    modelFamily: inferModelFamily(model),
    modelVersion: inferModelVersion(model),
    vision: hasVisionInput(model),
    toolCalling: null,
    structuredOutput: null,
    streaming: functionType === "embedding" ? null : true,
    maxOutputTokens: null,
    inputPricePerMillion: null,
    outputPricePerMillion: null,
    cacheReadPricePerMillion: null,
    cacheWritePricePerMillion: null,
    temperatureSupported: null,
    topPSupported: null,
    jsonModeSupported: null,
    parameterSchemaJson: null,
    apiFamily: "openai_compatible",
    providerRegion: null,
    createdAt: nowSeconds,
    updatedAt: nowSeconds,
  };
}

function buildRuntimePolicies(
  catalog: ModelCatalogRecord,
  model: NvidiaModel,
  nowSeconds: number,
  defaults: GatewayRouteDefaults,
): ModelRuntimePolicyRecord[] {
  const taskKinds = JSON.parse(catalog.taskKindsJson) as string[];
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
      rateLimitRpm: null,
      rateLimitTpm: null,
      rateLimitRpd: null,
      burstLimit: null,
      rateLimitWindowSeconds: null,
      cooldownUntil: null,
      cacheMode: taskKind === "embedding" ? "skip" : "ttl",
      cacheTtlSeconds: taskKind === "embedding" ? null : 300,
      cacheScope: "global",
      cacheKeyTemplate: null,
      requestTimeoutMs: taskKind === "reasoning" || taskKind === "vision" ? 8000 : 4500,
      maxAttempts: 1,
      retryDelayMs: 250,
      backoff: "linear",
      notes: "seeded_from_nvidia_models_api",
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

export async function fetchNvidiaModels(apiKey?: string): Promise<NvidiaModel[]> {
  const headers = new Headers({ accept: "application/json" });
  const normalizedApiKey = trim(apiKey);
  if (normalizedApiKey) {
    headers.set("authorization", `Bearer ${normalizedApiKey}`);
  }
  const response = await fetch(NVIDIA_MODELS_URL, {
    headers,
    method: "GET",
  });
  if (!response.ok) {
    throw new Error(`nvidia_models_fetch_failed:${response.status}`);
  }
  const payload = (await response.json()) as { data?: NvidiaModel[] };
  const models = Array.isArray(payload.data) ? payload.data : [];
  const seen = new Set<string>();
  const deduped: NvidiaModel[] = [];
  for (const model of models) {
    const id = modelKey(model);
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    deduped.push(model);
  }
  return deduped;
}

export function mapNvidiaModelsToSyncBundles(
  models: NvidiaModel[],
  env: Pick<Env, "HERMES_CF_TEXT_PLAIN_ROUTE_NAME" | "HERMES_CF_TEXT_CODING_ROUTE_NAME">,
  nowSeconds: number,
): NvidiaSyncBundle[] {
  const ranked = [...models]
    .filter((model) => !!modelKey(model))
    .map((model) => ({ model, functionType: inferFunctionType(model) }))
    .sort((left, right) => {
      const scoreDiff = scorePriority(right.model, right.functionType) - scorePriority(left.model, left.functionType);
      if (scoreDiff !== 0) {
        return scoreDiff;
      }
      return modelKey(left.model).localeCompare(modelKey(right.model));
    });

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
