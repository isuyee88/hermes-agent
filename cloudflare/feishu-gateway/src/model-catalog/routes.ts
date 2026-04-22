import type { Env } from "../runtime";

type PolicyCandidateRow = {
  id: string;
  provider: string;
  model: string;
  task_kind: string;
  route_family?: string | null;
  gateway_route_name?: string | null;
  priority_rank?: number | null;
  enabled?: number | null;
  hidden_reason?: string | null;
  rate_limit_rpm?: number | null;
  rate_limit_tpm?: number | null;
  burst_limit?: number | null;
  rate_limit_window_seconds?: number | null;
  cooldown_until?: number | null;
  request_timeout_ms?: number | null;
  max_attempts?: number | null;
  fallback_allowed?: number | null;
  cache_mode?: string | null;
  status?: string | null;
  hidden?: number | null;
  is_available?: number | null;
  manual_pinned?: number | null;
  selection_hint?: string | null;
  is_free?: number | null;
  health_score?: number | null;
  success_rate?: number | null;
  error_rate?: number | null;
  cache_hit_rate?: number | null;
  negative_feedback_count?: number | null;
  auth_error_count?: number | null;
  model_not_found_count?: number | null;
  rate_limit_count?: number | null;
  provider_failure_count?: number | null;
};

type RouteRow = {
  id: string;
  name: string;
  elements?: unknown[];
  version?: {
    version_id?: string;
    data?: unknown[];
  } | null;
};

type GatewayIdentity = {
  accountId: string;
  gatewayId: string;
};

type RouteElement = {
  id: string;
  type: "start" | "rate" | "model" | "end";
  outputs: Record<string, { elementId: string }>;
  properties?: Record<string, string | number>;
};

type ProviderRateLimitConfig = {
  limit: number;
  interval: number;
  key: string;
};

type RouteBuildOptions = {
  gatewayProviderSlugs?: Record<string, string>;
  providerRateLimits?: Record<string, ProviderRateLimitConfig>;
};

type PublishRouteSummary = {
  routeName: string;
  routeId: string;
  published: boolean;
  deploymentId?: string;
  versionId?: string;
  candidateModels: string[];
};

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function parseBoolean(value: unknown, fallback = false): boolean {
  const normalized = trim(value).toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return fallback;
}

function parsePositiveInt(value: unknown, fallback: number, min = 1, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = Number.parseInt(trim(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, parsed));
}

function normalizeNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const normalized = trim(value);
  if (!normalized) return null;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseJsonObject(value: unknown): Record<string, unknown> {
  const normalized = trim(value);
  if (!normalized) {
    return {};
  }
  try {
    const parsed = JSON.parse(normalized) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return parsed as Record<string, unknown>;
  } catch {
    return {};
  }
}

function parseProviderGatewaySlugOverrides(env: Env): Record<string, string> {
  const raw = parseJsonObject((env as Record<string, unknown>).HERMES_CF_GATEWAY_PROVIDER_SLUGS_JSON);
  const out: Record<string, string> = {};
  for (const [provider, slug] of Object.entries(raw)) {
    const normalizedProvider = trim(provider).toLowerCase();
    const normalizedSlug = trim(slug);
    if (!normalizedProvider || !normalizedSlug) {
      continue;
    }
    out[normalizedProvider] = normalizedSlug;
  }
  return out;
}

function parseProviderRateLimitOverrides(env: Env): Record<string, ProviderRateLimitConfig> {
  const raw = parseJsonObject((env as Record<string, unknown>).HERMES_CF_PROVIDER_ROUTE_LIMITS_JSON);
  const out: Record<string, ProviderRateLimitConfig> = {};
  for (const [provider, configValue] of Object.entries(raw)) {
    const normalizedProvider = trim(provider).toLowerCase();
    if (!normalizedProvider || !configValue || typeof configValue !== "object" || Array.isArray(configValue)) {
      continue;
    }
    const config = configValue as Record<string, unknown>;
    const limit = normalizeNumber(config.limit) ?? normalizeNumber(config.rpm) ?? normalizeNumber(config.requests);
    const interval = normalizeNumber(config.interval) ?? normalizeNumber(config.window) ?? 60;
    const key = trim(config.key) || "metadata.gateway_route_name";
    if ((limit ?? 0) <= 0 || (interval ?? 0) <= 0) {
      continue;
    }
    out[normalizedProvider] = {
      limit,
      interval,
      key,
    };
  }
  return out;
}

function deriveGatewayIdentity(env: Env): GatewayIdentity | null {
  const accountId = trim(env.CLOUDFLARE_ACCOUNT_ID);
  const candidate = trim(env.CLOUDFLARE_AI_GATEWAY_BASE_URL);
  const match = candidate.match(/\/v1\/([^/]+)\/([^/]+)/i);
  if (!match) {
    return null;
  }
  const derivedAccountId = trim(match[1]);
  const derivedGatewayId = trim(match[2]);
  if (!(accountId || derivedAccountId) || !derivedGatewayId) {
    return null;
  }
  return {
    accountId: accountId || derivedAccountId,
    gatewayId: derivedGatewayId,
  };
}

async function cfApi<T>(env: Env, path: string, init: RequestInit = {}): Promise<T> {
  const token = trim(env.CLOUDFLARE_API_TOKEN);
  if (!token) {
    throw new Error("cloudflare_api_token_missing");
  }
  const response = await fetch(`https://api.cloudflare.com/client/v4${path}`, {
    ...init,
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  const payload = (await response.json()) as {
    success?: boolean;
    errors?: Array<{ code?: number; message?: string }>;
    result?: T;
    data?: T;
  };
  if (!response.ok || payload.success === false) {
    const firstError = Array.isArray(payload.errors) && payload.errors.length > 0 ? payload.errors[0] : undefined;
    throw new Error(`cloudflare_api_failed:${response.status}:${firstError?.message ?? "unknown"}`);
  }
  if (payload.result !== undefined) {
    return payload.result;
  }
  if (payload.data !== undefined) {
    return payload.data;
  }
  return payload as T;
}

function buildCandidateReason(row: PolicyCandidateRow, nowSeconds: number, env: Env): {
  enabled: boolean;
  hiddenReason: string | null;
  catalogStatus: string;
  catalogHidden: boolean;
  cooldownUntil: number | null;
} {
  const minHealthScore = parsePositiveInt(env.HERMES_MODEL_CATALOG_MIN_HEALTH_SCORE, 45, 1, 100);
  const maxNegativeFeedback = parsePositiveInt(env.HERMES_MODEL_CATALOG_MAX_NEGATIVE_FEEDBACK_COUNT, 3, 1, 100);
  const maxAuthErrors = parsePositiveInt(env.HERMES_MODEL_CATALOG_MAX_AUTH_ERROR_COUNT, 1, 1, 100);
  const maxModelNotFound = parsePositiveInt(env.HERMES_MODEL_CATALOG_MAX_MODEL_NOT_FOUND_COUNT, 1, 1, 100);
  const maxRateLimitCount = parsePositiveInt(env.HERMES_MODEL_CATALOG_MAX_RATE_LIMIT_COUNT, 3, 1, 1000);
  const cooldownSeconds = parsePositiveInt(env.HERMES_MODEL_CATALOG_RATE_LIMIT_COOLDOWN_SECONDS, 900, 60, 86400);
  const currentCooldownUntil = normalizeNumber(row.cooldown_until);
  const hardUnavailable = row.is_available !== 1 || row.hidden === 1 || trim(row.status) === "inactive";
  if (hardUnavailable) {
    return {
      enabled: false,
      hiddenReason: trim(row.hidden_reason) || "inactive",
      catalogStatus: trim(row.status) || "inactive",
      catalogHidden: true,
      cooldownUntil: currentCooldownUntil,
    };
  }
  if ((normalizeNumber(row.auth_error_count) ?? 0) >= maxAuthErrors) {
    return {
      enabled: false,
      hiddenReason: "auth_error_threshold",
      catalogStatus: "invalid",
      catalogHidden: true,
      cooldownUntil: currentCooldownUntil,
    };
  }
  if ((normalizeNumber(row.model_not_found_count) ?? 0) >= maxModelNotFound) {
    return {
      enabled: false,
      hiddenReason: "model_not_found_threshold",
      catalogStatus: "invalid",
      catalogHidden: true,
      cooldownUntil: currentCooldownUntil,
    };
  }
  if ((normalizeNumber(row.negative_feedback_count) ?? 0) >= maxNegativeFeedback) {
    return {
      enabled: false,
      hiddenReason: "negative_feedback_threshold",
      catalogStatus: "degraded",
      catalogHidden: true,
      cooldownUntil: currentCooldownUntil,
    };
  }
  if ((normalizeNumber(row.health_score) ?? 100) < minHealthScore) {
    return {
      enabled: false,
      hiddenReason: "health_score_threshold",
      catalogStatus: "degraded",
      catalogHidden: true,
      cooldownUntil: currentCooldownUntil,
    };
  }
  if ((normalizeNumber(row.rate_limit_count) ?? 0) >= maxRateLimitCount) {
    return {
      enabled: false,
      hiddenReason: "rate_limit_cooldown",
      catalogStatus: "degraded",
      catalogHidden: false,
      cooldownUntil: nowSeconds + cooldownSeconds,
    };
  }
  if ((currentCooldownUntil ?? 0) > nowSeconds) {
    return {
      enabled: false,
      hiddenReason: "rate_limit_cooldown",
      catalogStatus: "degraded",
      catalogHidden: false,
      cooldownUntil: currentCooldownUntil,
    };
  }
  return {
    enabled: true,
    hiddenReason: null,
    catalogStatus: "active",
    catalogHidden: false,
    cooldownUntil: null,
  };
}

function candidateSort(left: PolicyCandidateRow, right: PolicyCandidateRow): number {
  const leftPinned = left.manual_pinned === 1 ? 1 : 0;
  const rightPinned = right.manual_pinned === 1 ? 1 : 0;
  if (leftPinned !== rightPinned) return rightPinned - leftPinned;
  const leftRecommended = trim(left.selection_hint) === "recommended" ? 1 : 0;
  const rightRecommended = trim(right.selection_hint) === "recommended" ? 1 : 0;
  if (leftRecommended !== rightRecommended) return rightRecommended - leftRecommended;
  const leftHealth = normalizeNumber(left.health_score) ?? -1;
  const rightHealth = normalizeNumber(right.health_score) ?? -1;
  if (leftHealth !== rightHealth) return rightHealth - leftHealth;
  const leftError = normalizeNumber(left.error_rate) ?? 1;
  const rightError = normalizeNumber(right.error_rate) ?? 1;
  if (leftError !== rightError) return leftError - rightError;
  const leftCache = normalizeNumber(left.cache_hit_rate) ?? -1;
  const rightCache = normalizeNumber(right.cache_hit_rate) ?? -1;
  if (leftCache !== rightCache) return rightCache - leftCache;
  return (normalizeNumber(left.priority_rank) ?? 9999) - (normalizeNumber(right.priority_rank) ?? 9999);
}

function deriveProviderRateLimit(
  provider: string,
  rows: PolicyCandidateRow[],
  options: RouteBuildOptions = {},
): ProviderRateLimitConfig | null {
  const normalizedProvider = trim(provider).toLowerCase();
  const override = normalizedProvider ? options.providerRateLimits?.[normalizedProvider] : undefined;
  if (override) {
    return override;
  }
  const rpm = rows.map((row) => normalizeNumber(row.rate_limit_rpm) ?? 0).find((value) => value > 0) ?? 0;
  const burst = rows.map((row) => normalizeNumber(row.burst_limit) ?? 0).find((value) => value > 0) ?? 0;
  const interval = rows.map((row) => normalizeNumber(row.rate_limit_window_seconds) ?? 0).find((value) => value > 0) ?? 0;
  const limit = rpm > 0 ? rpm : burst;
  if (limit <= 0 || interval <= 0) {
    return null;
  }
  return {
    limit,
    interval,
    key: "metadata.gateway_route_name",
  };
}

function buildRouteElements(rows: PolicyCandidateRow[], options: RouteBuildOptions = {}): RouteElement[] {
  const endId = "end";
  if (rows.length === 0) {
    return [
      {
        id: "start",
        type: "start",
        outputs: { next: { elementId: endId } },
      },
      {
        id: endId,
        type: "end",
        outputs: {},
      },
    ];
  }

  const elements: RouteElement[] = [];
  const elementById = new Map<string, RouteElement>();
  const providerOrder: string[] = [];
  const providerBuckets = new Map<string, PolicyCandidateRow[]>();
  let modelCounter = 0;

  for (const row of rows) {
    const provider = trim(row.provider).toLowerCase();
    if (!providerBuckets.has(provider)) {
      providerOrder.push(provider);
      providerBuckets.set(provider, []);
    }
    providerBuckets.get(provider)?.push(row);
  }

  const providerEntries = providerOrder.map((provider, providerIndex) => {
    const providerRows = providerBuckets.get(provider) ?? [];
    const providerRateLimit = deriveProviderRateLimit(provider, providerRows, options);
    const modelNodeIds: string[] = [];
    const rateNodeId = providerRateLimit ? `provider_${providerIndex + 1}_rate_limit` : null;

    for (const row of providerRows) {
      modelCounter += 1;
      const modelNodeId = `model_${modelCounter}`;
      const element: RouteElement = {
        id: modelNodeId,
        type: "model",
        outputs: {
          success: { elementId: endId },
          fallback: { elementId: endId },
        },
        properties: {
          provider: trim(options.gatewayProviderSlugs?.[provider]) || trim(row.provider),
          model: trim(row.model),
          timeout: normalizeNumber(row.request_timeout_ms) ?? 4500,
          retries: normalizeNumber(row.max_attempts) ?? 0,
        },
      };
      modelNodeIds.push(modelNodeId);
      elements.push(element);
      elementById.set(modelNodeId, element);
    }

    if (providerRateLimit && modelNodeIds.length > 0) {
      const rateElement: RouteElement = {
        id: rateNodeId!,
        type: "rate",
        outputs: {
          success: { elementId: modelNodeIds[0] },
          fallback: { elementId: endId },
        },
        properties: {
          limit: providerRateLimit.limit,
          limitType: "count",
          window: providerRateLimit.interval,
          key: providerRateLimit.key,
        },
      };
      elements.push(rateElement);
      elementById.set(rateElement.id, rateElement);
    }

    return {
      entryNodeId: rateNodeId ?? modelNodeIds[0] ?? endId,
      rateNodeId,
      modelNodeIds,
    };
  });

  for (let providerIndex = 0; providerIndex < providerEntries.length; providerIndex += 1) {
    const providerEntry = providerEntries[providerIndex];
    const nextProviderEntryId = providerEntries[providerIndex + 1]?.entryNodeId ?? endId;
    if (providerEntry.rateNodeId) {
      const rateNode = elementById.get(providerEntry.rateNodeId);
      if (rateNode) {
        rateNode.outputs.fallback = { elementId: nextProviderEntryId };
      }
    }
    for (let modelIndex = 0; modelIndex < providerEntry.modelNodeIds.length; modelIndex += 1) {
      const modelNode = elementById.get(providerEntry.modelNodeIds[modelIndex]);
      if (!modelNode) {
        continue;
      }
      modelNode.outputs.fallback = {
        elementId: providerEntry.modelNodeIds[modelIndex + 1] ?? nextProviderEntryId,
      };
    }
  }

  elements.unshift({
    id: "start",
    type: "start",
    outputs: { next: { elementId: providerEntries[0]?.entryNodeId ?? endId } },
  });
  elements.push({
    id: endId,
    type: "end",
    outputs: {},
  });
  return elements;
}

async function listRoutes(env: Env, identity: GatewayIdentity): Promise<RouteRow[]> {
  const result = await cfApi<{ routes?: RouteRow[] }>(
    env,
    `/accounts/${identity.accountId}/ai-gateway/gateways/${identity.gatewayId}/routes`,
  );
  return Array.isArray(result.routes) ? result.routes : [];
}

async function getRoute(env: Env, identity: GatewayIdentity, routeId: string): Promise<RouteRow> {
  return cfApi<RouteRow>(env, `/accounts/${identity.accountId}/ai-gateway/gateways/${identity.gatewayId}/routes/${routeId}`);
}

async function listCustomProviderSlugs(env: Env, accountId: string): Promise<Set<string>> {
  const result = await cfApi<Array<{ slug?: string }>>(env, `/accounts/${accountId}/ai-gateway/custom-providers?per_page=100`);
  const slugs = new Set<string>();
  for (const provider of Array.isArray(result) ? result : []) {
    const slug = trim(provider?.slug).toLowerCase();
    if (slug) {
      slugs.add(slug);
    }
  }
  return slugs;
}

function resolveGatewayProviderSlug(
  provider: string,
  gatewayProviderSlugs: Record<string, string>,
  customProviderSlugs: Set<string>,
): string | null {
  const normalizedProvider = trim(provider).toLowerCase();
  if (!normalizedProvider) {
    return null;
  }
  const configured = trim(gatewayProviderSlugs[normalizedProvider]);
  if (configured) {
    return configured;
  }
  if (normalizedProvider === "openrouter" || normalizedProvider === "nvidia" || normalizedProvider.startsWith("custom-")) {
    return normalizedProvider;
  }
  if (customProviderSlugs.has(normalizedProvider)) {
    return `custom-${normalizedProvider}`;
  }
  return null;
}

async function ensureRoutePublished(
  env: Env,
  identity: GatewayIdentity,
  routeName: string,
  elements: RouteElement[],
): Promise<PublishRouteSummary> {
  const routes = await listRoutes(env, identity);
  const existing = routes.find((route) => trim(route.name) === routeName);
  const desiredElementsJson = JSON.stringify(elements);

  if (!existing) {
    const created = await cfApi<{
      id: string;
      name: string;
      deployment?: { deployment_id?: string; version_id?: string };
    }>(
      env,
      `/accounts/${identity.accountId}/ai-gateway/gateways/${identity.gatewayId}/routes`,
      {
        method: "POST",
        body: JSON.stringify({
          name: routeName,
          elements,
        }),
      },
    );
    return {
      routeName,
      routeId: trim(created.id),
      published: true,
      deploymentId: trim(created.deployment?.deployment_id),
      versionId: trim(created.deployment?.version_id),
      candidateModels: elements
        .filter((item) => item.type === "model")
        .map((item) => trim(item.properties?.model)),
    };
  }

  const existingRoute = await getRoute(env, identity, trim(existing.id));
  const currentElementsJson = JSON.stringify(
    Array.isArray(existingRoute.version?.data)
      ? existingRoute.version.data
      : Array.isArray(existingRoute.elements)
        ? existingRoute.elements
        : [],
  );
  if (currentElementsJson === desiredElementsJson) {
    return {
      routeName,
      routeId: trim(existing.id),
      published: false,
      versionId: trim(existing.version?.version_id),
      candidateModels: elements
        .filter((item) => item.type === "model")
        .map((item) => trim(item.properties?.model)),
    };
  }

  const newVersion = await cfApi<{ version_id?: string }>(
    env,
    `/accounts/${identity.accountId}/ai-gateway/gateways/${identity.gatewayId}/routes/${trim(existing.id)}/versions`,
    {
      method: "POST",
      body: JSON.stringify({ elements }),
    },
  );
  const versionId = trim(newVersion.version_id);
  const deployment = await cfApi<{ deployment_id?: string }>(
    env,
    `/accounts/${identity.accountId}/ai-gateway/gateways/${identity.gatewayId}/routes/${trim(existing.id)}/deployments`,
    {
      method: "POST",
      body: JSON.stringify({ version_id: versionId }),
    },
  );
  return {
    routeName,
    routeId: trim(existing.id),
    published: true,
    deploymentId: trim(deployment.deployment_id),
    versionId,
    candidateModels: elements
      .filter((item) => item.type === "model")
      .map((item) => trim(item.properties?.model)),
  };
}

export async function applyModelHealthPolicies(env: Env, nowSeconds = Math.trunc(Date.now() / 1000)): Promise<number> {
  const db = env.MODEL_CATALOG_DB;
  if (!db) {
    return 0;
  }
  const rowsResult = await db
    .prepare(
      `WITH latest_health AS (
         SELECT h.*
         FROM model_health_stats h
         WHERE h.task_kind = 'general'
           AND h.window_end = (
             SELECT MAX(h2.window_end)
             FROM model_health_stats h2
             WHERE h2.provider = h.provider
               AND h2.model = h.model
               AND h2.task_kind = h.task_kind
           )
       )
       SELECT
         p.id,
         p.provider,
         p.model,
         p.task_kind,
         p.route_family,
         p.gateway_route_name,
         p.priority_rank,
         p.enabled,
         p.hidden_reason,
         p.rate_limit_rpm,
         p.rate_limit_tpm,
         p.burst_limit,
         p.rate_limit_window_seconds,
         p.cooldown_until,
         p.request_timeout_ms,
         p.max_attempts,
         p.fallback_allowed,
         p.cache_mode,
         c.status,
         c.hidden,
         c.is_available,
         c.manual_pinned,
         c.selection_hint,
         c.is_free,
         h.health_score,
         h.success_rate,
         h.error_rate,
         h.cache_hit_rate,
         h.negative_feedback_count,
         h.auth_error_count,
         h.model_not_found_count,
         h.rate_limit_count,
         h.provider_failure_count
       FROM model_runtime_policy p
       JOIN model_catalog c
         ON c.provider = p.provider
        AND c.model = p.model
       LEFT JOIN latest_health h
         ON h.provider = p.provider
        AND h.model = p.model`,
    )
    .all<PolicyCandidateRow>();
  const rows = Array.isArray(rowsResult.results) ? rowsResult.results : [];
  const statements: D1PreparedStatement[] = [];

  for (const row of rows) {
    const decision = buildCandidateReason(row, nowSeconds, env);
    statements.push(
      db
        .prepare(
          `UPDATE model_runtime_policy
           SET enabled = ?,
               hidden_reason = ?,
               cooldown_until = ?,
               updated_at = ?
           WHERE id = ?`,
        )
        .bind(decision.enabled ? 1 : 0, decision.hiddenReason, decision.cooldownUntil, nowSeconds, row.id),
    );
    statements.push(
      db
        .prepare(
          `UPDATE model_catalog
           SET status = ?,
               hidden = ?,
               updated_at = ?
           WHERE provider = ?
             AND model = ?`,
        )
        .bind(decision.catalogStatus, decision.catalogHidden ? 1 : 0, nowSeconds, row.provider, row.model),
    );
  }

  for (let index = 0; index < statements.length; index += 25) {
    await db.batch(statements.slice(index, index + 25));
  }
  return rows.length;
}

export async function publishDynamicRoutes(env: Env, nowSeconds = Math.trunc(Date.now() / 1000)): Promise<PublishRouteSummary[]> {
  if (!parseBoolean(env.HERMES_MODEL_CATALOG_ROUTE_PUBLISH_ENABLED, true)) {
    return [];
  }
  const db = env.MODEL_CATALOG_DB;
  const identity = deriveGatewayIdentity(env);
  const token = trim(env.CLOUDFLARE_API_TOKEN);
  if (!db || !identity || !token) {
    return [];
  }
  const providerSlugOverrides = parseProviderGatewaySlugOverrides(env);
  const providerRateLimitOverrides = parseProviderRateLimitOverrides(env);
  const customProviderSlugs = await listCustomProviderSlugs(env, identity.accountId);

  const routeRowsResult = await db
    .prepare(
      `WITH latest_health AS (
         SELECT h.*
         FROM model_health_stats h
         WHERE h.task_kind = 'general'
           AND h.window_end = (
             SELECT MAX(h2.window_end)
             FROM model_health_stats h2
             WHERE h2.provider = h.provider
               AND h2.model = h.model
               AND h2.task_kind = h.task_kind
           )
       )
       SELECT
         p.id,
         p.provider,
         p.model,
         p.task_kind,
         p.route_family,
         p.gateway_route_name,
         p.priority_rank,
         p.enabled,
         p.hidden_reason,
         p.rate_limit_rpm,
         p.rate_limit_tpm,
         p.burst_limit,
         p.rate_limit_window_seconds,
         p.cooldown_until,
         p.request_timeout_ms,
         p.max_attempts,
         p.fallback_allowed,
         p.cache_mode,
         c.status,
         c.hidden,
         c.is_available,
         c.manual_pinned,
         c.selection_hint,
         c.is_free,
         h.health_score,
         h.success_rate,
         h.error_rate,
         h.cache_hit_rate,
         h.negative_feedback_count,
         h.auth_error_count,
         h.model_not_found_count,
         h.rate_limit_count,
         h.provider_failure_count
       FROM model_runtime_policy p
       JOIN model_catalog c
         ON c.provider = p.provider
        AND c.model = p.model
       LEFT JOIN latest_health h
         ON h.provider = p.provider
        AND h.model = p.model
       WHERE p.enabled = 1
         AND c.hidden = 0
         AND c.is_available = 1
         AND COALESCE(p.gateway_route_name, '') <> ''`,
    )
    .all<PolicyCandidateRow>();
  const routeRows = Array.isArray(routeRowsResult.results) ? routeRowsResult.results : [];
  const maxModels = parsePositiveInt(env.HERMES_MODEL_CATALOG_ROUTE_MAX_MODELS, 4, 1, 12);
  const grouped = new Map<string, PolicyCandidateRow[]>();
  for (const row of routeRows) {
    const routeName = trim(row.gateway_route_name);
    if (!routeName) continue;
    const resolvedGatewayProvider = resolveGatewayProviderSlug(row.provider, providerSlugOverrides, customProviderSlugs);
    if (!resolvedGatewayProvider) continue;
    const bucket = grouped.get(routeName) ?? [];
    bucket.push({
      ...row,
      provider: trim(row.provider).toLowerCase(),
    });
    grouped.set(routeName, bucket);
  }

  const published: PublishRouteSummary[] = [];
  for (const [routeName, rows] of grouped.entries()) {
    const selected = [...rows].sort(candidateSort).slice(0, maxModels);
    const gatewayProviderSlugs = Object.fromEntries(
      Array.from(new Set(selected.map((row) => trim(row.provider).toLowerCase()))).map((provider) => [
        provider,
        resolveGatewayProviderSlug(provider, providerSlugOverrides, customProviderSlugs) ?? provider,
      ]),
    );
    const elements = buildRouteElements(selected, {
      gatewayProviderSlugs,
      providerRateLimits: providerRateLimitOverrides,
    });
    const summary = await ensureRoutePublished(env, identity, routeName, elements);
    published.push(summary);
    await db
      .prepare(
        `UPDATE model_runtime_policy
         SET last_route_publish_at = ?,
             updated_at = ?
         WHERE gateway_route_name = ?`,
      )
      .bind(nowSeconds, nowSeconds, routeName)
      .run();
  }

  return published;
}

export { buildRouteElements, candidateSort, resolveGatewayProviderSlug };
