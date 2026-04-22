import type { Env, JsonValue } from "../runtime";

export type GatewayFeedbackPayload = {
  provider: string;
  model: string;
  taskKind?: string | null;
  routeFamily?: string | null;
  gatewayRouteName?: string | null;
  gatewayLogId?: string | null;
  correlationId?: string | null;
  feedback: number;
  score?: number | null;
  errorKind?: string | null;
  errorCode?: string | null;
  errorMessage?: string | null;
  latencyMs?: number | null;
  ttfbMs?: number | null;
  cacheStatus?: string | null;
  requestTokens?: number | null;
  responseTokens?: number | null;
  estimatedCost?: number | null;
  requestUrl?: string | null;
  metadata?: Record<string, JsonValue> | null;
};

export type GatewayFeedbackRow = {
  feedback?: number | null;
  score?: number | null;
  error_kind?: string | null;
  latency_ms?: number | null;
  ttfb_ms?: number | null;
  cache_hit?: number | null;
  request_tokens?: number | null;
  response_tokens?: number | null;
  estimated_cost?: number | null;
};

export type FeedbackWindowSummary = {
  requestCount: number;
  successCount: number;
  errorCount: number;
  authErrorCount: number;
  modelNotFoundCount: number;
  providerFailureCount: number;
  rateLimitCount: number;
  timeoutCount: number;
  cacheHitCount: number;
  cacheMissCount: number;
  positiveFeedbackCount: number;
  negativeFeedbackCount: number;
  scoreAvg: number | null;
  healthScore: number;
  successRate: number;
  errorRate: number;
  cacheHitRate: number | null;
  latencyP50Ms: number | null;
  latencyP95Ms: number | null;
  latencyP99Ms: number | null;
  ttfbP50Ms: number | null;
  ttfbP95Ms: number | null;
  ttfbP99Ms: number | null;
  inputTokensSum: number;
  outputTokensSum: number;
  estimatedCostSum: number;
};

function normalizeString(value: unknown): string {
  return String(value ?? "").trim();
}

function normalizeNumber(value: unknown): number | null {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function percentile(values: number[], ratio: number): number | null {
  if (!values.length) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1));
  return sorted[index] ?? null;
}

function clampScore(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function scorePenaltyForErrorKind(errorKind: string): number {
  switch (errorKind) {
    case "provider_permission_denied":
    case "provider_auth_error":
      return 100;
    case "model_not_found":
      return 75;
    case "provider_rate_exhausted":
    case "rate_limited":
      return 55;
    case "timeout":
      return 45;
    case "empty_response":
      return 35;
    case "transport_error":
    case "provider_http_error":
      return 30;
    default:
      return 25;
  }
}

function isRateLimitKind(errorKind: string): boolean {
  return errorKind === "rate_limited" || errorKind === "provider_rate_exhausted";
}

function isAuthKind(errorKind: string): boolean {
  return errorKind === "provider_permission_denied" || errorKind === "provider_auth_error";
}

export function classifyGatewayFeedbackErrorKind(value: unknown, fallbackKind = ""): string {
  const raw = normalizeString(value).toLowerCase();
  const fallback = normalizeString(fallbackKind).toLowerCase();
  const haystack = `${raw} ${fallback}`.trim();

  if (!haystack) {
    return fallback || "provider_http_error";
  }
  if (haystack.includes("provider_permission_denied")) {
    return "provider_permission_denied";
  }
  if (haystack.includes("provider_auth_error") || haystack.includes(" auth ")) {
    return "provider_auth_error";
  }
  if (haystack.includes("401") || haystack.includes("403") || haystack.includes("invalid auth") || haystack.includes("permission")) {
    return "provider_permission_denied";
  }
  if (haystack.includes("provider_rate_exhausted")) {
    return "provider_rate_exhausted";
  }
  if (haystack.includes("rate_limited") || haystack.includes("429") || haystack.includes("over limit")) {
    return "rate_limited";
  }
  if (haystack.includes("model_not_found") || haystack.includes("404") || haystack.includes("unknown model")) {
    return "model_not_found";
  }
  if (haystack.includes("timeout")) {
    return "timeout";
  }
  if (haystack.includes("empty_response")) {
    return "empty_response";
  }
  if (haystack.includes("transport_error")) {
    return "transport_error";
  }
  if (haystack.includes("misrouted")) {
    return "provider_http_error";
  }
  if (haystack.includes("provider_http_error") || haystack.includes("status_error")) {
    return "provider_http_error";
  }
  return fallback || "provider_http_error";
}

export function deriveGatewayFeedbackScore(
  feedback: number,
  latencyMs: number | null | undefined,
  cacheStatus?: string | null,
  errorKind?: string | null,
): number {
  const normalizedLatency = Math.max(0, normalizeNumber(latencyMs) ?? 0);
  const normalizedCacheStatus = normalizeString(cacheStatus).toUpperCase();
  const normalizedErrorKind = normalizeString(errorKind);

  if (feedback <= 0) {
    return clampScore(35 - scorePenaltyForErrorKind(normalizedErrorKind || "provider_http_error"));
  }

  let score = 88;
  if (normalizedLatency <= 400) score += 8;
  else if (normalizedLatency <= 800) score += 4;
  else if (normalizedLatency >= 2500) score -= 12;
  else if (normalizedLatency >= 1500) score -= 6;

  if (normalizedCacheStatus === "HIT") score += 6;
  else if (normalizedCacheStatus === "MISS") score -= 2;

  return clampScore(score);
}

export function summarizeFeedbackWindow(rows: GatewayFeedbackRow[]): FeedbackWindowSummary {
  const latencyValues: number[] = [];
  const ttfbValues: number[] = [];

  let successCount = 0;
  let errorCount = 0;
  let authErrorCount = 0;
  let modelNotFoundCount = 0;
  let providerFailureCount = 0;
  let rateLimitCount = 0;
  let timeoutCount = 0;
  let cacheHitCount = 0;
  let cacheMissCount = 0;
  let positiveFeedbackCount = 0;
  let negativeFeedbackCount = 0;
  let scoreSum = 0;
  let scoreCount = 0;
  let inputTokensSum = 0;
  let outputTokensSum = 0;
  let estimatedCostSum = 0;

  for (const row of rows) {
    const feedback = normalizeNumber(row.feedback) ?? 0;
    const errorKind = normalizeString(row.error_kind);
    const latencyMs = normalizeNumber(row.latency_ms);
    const ttfbMs = normalizeNumber(row.ttfb_ms);
    const cacheHit = normalizeNumber(row.cache_hit);
    const requestTokens = normalizeNumber(row.request_tokens) ?? 0;
    const responseTokens = normalizeNumber(row.response_tokens) ?? 0;
    const estimatedCost = normalizeNumber(row.estimated_cost) ?? 0;
    const score = normalizeNumber(row.score);

    if (feedback > 0) {
      successCount += 1;
      positiveFeedbackCount += 1;
    } else if (feedback < 0) {
      errorCount += 1;
      negativeFeedbackCount += 1;
    }

    if (isAuthKind(errorKind)) authErrorCount += 1;
    if (errorKind === "model_not_found") modelNotFoundCount += 1;
    if (isRateLimitKind(errorKind)) rateLimitCount += 1;
    if (errorKind === "timeout") timeoutCount += 1;
    if (feedback < 0 && !isAuthKind(errorKind) && errorKind !== "model_not_found" && !isRateLimitKind(errorKind) && errorKind !== "timeout") {
      providerFailureCount += 1;
    }

    if (cacheHit === 1) cacheHitCount += 1;
    if (cacheHit === 0) cacheMissCount += 1;

    if (latencyMs !== null) latencyValues.push(latencyMs);
    if (ttfbMs !== null) ttfbValues.push(ttfbMs);
    if (score !== null) {
      scoreSum += score;
      scoreCount += 1;
    }
    inputTokensSum += requestTokens;
    outputTokensSum += responseTokens;
    estimatedCostSum += estimatedCost;
  }

  const requestCount = rows.length;
  const successRate = requestCount > 0 ? successCount / requestCount : 0;
  const errorRate = requestCount > 0 ? errorCount / requestCount : 0;
  const cacheDenominator = cacheHitCount + cacheMissCount;
  const cacheHitRate = cacheDenominator > 0 ? cacheHitCount / cacheDenominator : null;
  const scoreAvg = scoreCount > 0 ? clampScore(scoreSum / scoreCount) : null;
  const healthBaseline = scoreAvg ?? Math.round(successRate * 100);
  const healthScore = clampScore(
    healthBaseline - authErrorCount * 20 - modelNotFoundCount * 12 - rateLimitCount * 8 - timeoutCount * 6,
  );

  return {
    requestCount,
    successCount,
    errorCount,
    authErrorCount,
    modelNotFoundCount,
    providerFailureCount,
    rateLimitCount,
    timeoutCount,
    cacheHitCount,
    cacheMissCount,
    positiveFeedbackCount,
    negativeFeedbackCount,
    scoreAvg,
    healthScore,
    successRate,
    errorRate,
    cacheHitRate,
    latencyP50Ms: percentile(latencyValues, 0.5),
    latencyP95Ms: percentile(latencyValues, 0.95),
    latencyP99Ms: percentile(latencyValues, 0.99),
    ttfbP50Ms: percentile(ttfbValues, 0.5),
    ttfbP95Ms: percentile(ttfbValues, 0.95),
    ttfbP99Ms: percentile(ttfbValues, 0.99),
    inputTokensSum,
    outputTokensSum,
    estimatedCostSum,
  };
}

function buildFeedbackId(payload: GatewayFeedbackPayload): string {
  const suffix = Math.random().toString(36).slice(2, 10);
  return [
    "fb",
    Date.now().toString(36),
    normalizeString(payload.provider).slice(0, 12) || "provider",
    normalizeString(payload.model).slice(0, 18) || "model",
    suffix,
  ].join("_");
}

function buildCacheHitFlag(cacheStatus: string | null | undefined): number | null {
  const normalized = normalizeString(cacheStatus).toUpperCase();
  if (!normalized) {
    return null;
  }
  if (normalized === "HIT") {
    return 1;
  }
  if (normalized === "MISS" || normalized === "BYPASS" || normalized === "SKIP") {
    return 0;
  }
  return null;
}

export async function recordGatewayFeedback(env: Env, payload: GatewayFeedbackPayload): Promise<void> {
  const db = env.MODEL_CATALOG_DB;
  if (!db || typeof db.prepare !== "function") {
    return;
  }

  const statement = db.prepare(
    `INSERT INTO model_feedback_events (
      id,
      provider,
      model,
      task_kind,
      route_family,
      gateway_route_name,
      gateway_log_id,
      correlation_id,
      feedback,
      score,
      error_kind,
      error_code,
      error_message,
      latency_ms,
      ttfb_ms,
      cache_status,
      cache_hit,
      request_tokens,
      response_tokens,
      estimated_cost,
      metadata_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  );

  await statement
    .bind(
      buildFeedbackId(payload),
      normalizeString(payload.provider),
      normalizeString(payload.model),
      normalizeString(payload.taskKind) || "general",
      normalizeString(payload.routeFamily) || null,
      normalizeString(payload.gatewayRouteName) || null,
      normalizeString(payload.gatewayLogId) || null,
      normalizeString(payload.correlationId) || null,
      payload.feedback > 0 ? 1 : -1,
      normalizeNumber(payload.score),
      normalizeString(payload.errorKind) || null,
      normalizeString(payload.errorCode) || null,
      normalizeString(payload.errorMessage) || null,
      normalizeNumber(payload.latencyMs),
      normalizeNumber(payload.ttfbMs),
      normalizeString(payload.cacheStatus) || null,
      buildCacheHitFlag(payload.cacheStatus),
      normalizeNumber(payload.requestTokens),
      normalizeNumber(payload.responseTokens),
      normalizeNumber(payload.estimatedCost),
      payload.metadata ? JSON.stringify(payload.metadata) : null,
    )
    .run();
}
