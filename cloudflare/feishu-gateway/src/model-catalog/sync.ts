import type { Env } from "../runtime";
import { fetchNvidiaModels, mapNvidiaModelsToSyncBundles } from "./nvidia";
import { fetchOpenRouterModels, mapOpenRouterModelsToSyncBundles } from "./openrouter";
import type { ModelCatalogRecord, ModelRuntimePolicyRecord } from "./types";

type ProviderSyncSummary = {
  provider: string;
  runId: string;
  discoveredModelCount: number;
  upsertedModelCount: number;
  hiddenModelCount: number;
  status: string;
  errorSummary?: string;
};

type CatalogSyncBundle = {
  catalog: ModelCatalogRecord;
  policies: ModelRuntimePolicyRecord[];
};

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function toDbBoolean(value: boolean | null | undefined): number | null {
  if (typeof value !== "boolean") {
    return null;
  }
  return value ? 1 : 0;
}

async function executeInChunks(db: D1Database, statements: D1PreparedStatement[], chunkSize = 25): Promise<void> {
  for (let index = 0; index < statements.length; index += chunkSize) {
    await db.batch(statements.slice(index, index + chunkSize));
  }
}

async function insertProviderSyncRun(db: D1Database, provider: string, nowSeconds: number): Promise<string> {
  const runId = crypto.randomUUID();
  await db
    .prepare(
      `INSERT INTO provider_sync_runs (
         id,
         provider,
         source,
         run_kind,
         status,
         discovered_model_count,
         upserted_model_count,
         hidden_model_count,
         error_count,
         error_summary,
         started_at,
         completed_at,
         created_at
       ) VALUES (?, ?, 'scheduled', 'catalog_refresh', 'running', 0, 0, 0, 0, NULL, ?, NULL, ?)`,
    )
    .bind(runId, provider, nowSeconds, nowSeconds)
    .run();
  return runId;
}

function buildCatalogUpsertStatement(db: D1Database, catalog: ModelCatalogRecord): D1PreparedStatement {
  return db
    .prepare(
      `INSERT INTO model_catalog (
         id,
         provider,
         model,
         provider_model_key,
         display_name,
         status,
         hidden,
         is_available,
         is_free,
         rank,
         selection_hint,
         manual_pinned,
         recent_used,
         recent_used_count,
         generated_command,
         last_probe_at,
         recent_used_at,
         last_sync_at,
         latency_ms,
         context_window,
         reasoning,
         consecutive_failures,
         failure_kind,
         last_error_code,
         last_error_message,
         last_failed_at,
         source,
         function_type,
         task_kinds_json,
         modalities_json,
         model_family,
         model_version,
         vision,
         tool_calling,
         structured_output,
         streaming,
         max_output_tokens,
         input_price_per_million,
         output_price_per_million,
         cache_read_price_per_million,
         cache_write_price_per_million,
         temperature_supported,
         top_p_supported,
         json_mode_supported,
         parameter_schema_json,
         api_family,
         provider_region,
         created_at,
         updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(provider, model) DO UPDATE SET
         provider_model_key = excluded.provider_model_key,
         display_name = excluded.display_name,
         status = excluded.status,
         hidden = excluded.hidden,
         is_available = excluded.is_available,
         is_free = excluded.is_free,
         rank = excluded.rank,
         selection_hint = excluded.selection_hint,
         generated_command = excluded.generated_command,
         last_probe_at = excluded.last_probe_at,
         last_sync_at = excluded.last_sync_at,
         context_window = excluded.context_window,
         reasoning = excluded.reasoning,
         source = excluded.source,
         function_type = excluded.function_type,
         task_kinds_json = excluded.task_kinds_json,
         modalities_json = excluded.modalities_json,
         model_family = excluded.model_family,
         model_version = excluded.model_version,
         vision = excluded.vision,
         tool_calling = excluded.tool_calling,
         structured_output = excluded.structured_output,
         streaming = excluded.streaming,
         max_output_tokens = excluded.max_output_tokens,
         input_price_per_million = excluded.input_price_per_million,
         output_price_per_million = excluded.output_price_per_million,
         cache_read_price_per_million = excluded.cache_read_price_per_million,
         cache_write_price_per_million = excluded.cache_write_price_per_million,
         temperature_supported = excluded.temperature_supported,
         top_p_supported = excluded.top_p_supported,
         json_mode_supported = excluded.json_mode_supported,
         parameter_schema_json = excluded.parameter_schema_json,
         api_family = excluded.api_family,
         provider_region = excluded.provider_region,
         updated_at = excluded.updated_at`,
    )
    .bind(
      catalog.id,
      catalog.provider,
      catalog.model,
      catalog.providerModelKey,
      catalog.displayName ?? null,
      catalog.status,
      toDbBoolean(catalog.hidden),
      toDbBoolean(catalog.isAvailable),
      toDbBoolean(catalog.isFree),
      catalog.rank ?? null,
      catalog.selectionHint ?? null,
      toDbBoolean(catalog.manualPinned),
      toDbBoolean(catalog.recentUsed),
      catalog.recentUsedCount,
      catalog.generatedCommand ?? null,
      catalog.lastProbeAt ?? null,
      catalog.recentUsedAt ?? null,
      catalog.lastSyncAt ?? null,
      catalog.latencyMs ?? null,
      catalog.contextWindow ?? null,
      toDbBoolean(catalog.reasoning),
      catalog.consecutiveFailures,
      catalog.failureKind ?? null,
      catalog.lastErrorCode ?? null,
      catalog.lastErrorMessage ?? null,
      catalog.lastFailedAt ?? null,
      catalog.source ?? null,
      catalog.functionType ?? null,
      catalog.taskKindsJson,
      catalog.modalitiesJson,
      catalog.modelFamily ?? null,
      catalog.modelVersion ?? null,
      toDbBoolean(catalog.vision),
      toDbBoolean(catalog.toolCalling),
      toDbBoolean(catalog.structuredOutput),
      toDbBoolean(catalog.streaming),
      catalog.maxOutputTokens ?? null,
      catalog.inputPricePerMillion ?? null,
      catalog.outputPricePerMillion ?? null,
      catalog.cacheReadPricePerMillion ?? null,
      catalog.cacheWritePricePerMillion ?? null,
      toDbBoolean(catalog.temperatureSupported),
      toDbBoolean(catalog.topPSupported),
      toDbBoolean(catalog.jsonModeSupported),
      catalog.parameterSchemaJson ?? null,
      catalog.apiFamily ?? null,
      catalog.providerRegion ?? null,
      catalog.createdAt,
      catalog.updatedAt,
    );
}

function buildPolicyUpsertStatement(db: D1Database, policy: ModelRuntimePolicyRecord): D1PreparedStatement {
  return db
    .prepare(
      `INSERT INTO model_runtime_policy (
         id,
         provider,
         model,
         task_kind,
         route_family,
         gateway_route_name,
         priority_rank,
         enabled,
         hidden_reason,
         fallback_allowed,
         rate_limit_rpm,
         rate_limit_tpm,
         rate_limit_rpd,
         burst_limit,
         rate_limit_window_seconds,
         cooldown_until,
         cache_mode,
         cache_ttl_seconds,
         cache_scope,
         cache_key_template,
         request_timeout_ms,
         max_attempts,
         retry_delay_ms,
         backoff,
         notes,
         last_route_publish_at,
         created_at,
         updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(provider, model, task_kind) DO UPDATE SET
         route_family = excluded.route_family,
         gateway_route_name = excluded.gateway_route_name,
         priority_rank = excluded.priority_rank,
         enabled = excluded.enabled,
         hidden_reason = excluded.hidden_reason,
         fallback_allowed = excluded.fallback_allowed,
         rate_limit_rpm = excluded.rate_limit_rpm,
         rate_limit_tpm = excluded.rate_limit_tpm,
         rate_limit_rpd = excluded.rate_limit_rpd,
         burst_limit = excluded.burst_limit,
         rate_limit_window_seconds = excluded.rate_limit_window_seconds,
         cooldown_until = excluded.cooldown_until,
         cache_mode = excluded.cache_mode,
         cache_ttl_seconds = excluded.cache_ttl_seconds,
         cache_scope = excluded.cache_scope,
         request_timeout_ms = excluded.request_timeout_ms,
         max_attempts = excluded.max_attempts,
         retry_delay_ms = excluded.retry_delay_ms,
         backoff = excluded.backoff,
         notes = excluded.notes,
         updated_at = excluded.updated_at`,
    )
    .bind(
      policy.id,
      policy.provider,
      policy.model,
      policy.taskKind,
      policy.routeFamily ?? null,
      policy.gatewayRouteName ?? null,
      policy.priorityRank,
      toDbBoolean(policy.enabled),
      policy.hiddenReason ?? null,
      toDbBoolean(policy.fallbackAllowed),
      policy.rateLimitRpm ?? null,
      policy.rateLimitTpm ?? null,
      policy.rateLimitRpd ?? null,
      policy.burstLimit ?? null,
      policy.rateLimitWindowSeconds ?? null,
      policy.cooldownUntil ?? null,
      policy.cacheMode,
      policy.cacheTtlSeconds ?? null,
      policy.cacheScope ?? null,
      policy.cacheKeyTemplate ?? null,
      policy.requestTimeoutMs ?? null,
      policy.maxAttempts ?? null,
      policy.retryDelayMs ?? null,
      policy.backoff ?? null,
      policy.notes ?? null,
      policy.lastRoutePublishAt ?? null,
      policy.createdAt,
      policy.updatedAt,
    );
}

async function upsertCatalogBundles(db: D1Database, bundles: CatalogSyncBundle[]): Promise<void> {
  const statements: D1PreparedStatement[] = [];
  for (const bundle of bundles) {
    statements.push(buildCatalogUpsertStatement(db, bundle.catalog));
    for (const policy of bundle.policies) {
      statements.push(buildPolicyUpsertStatement(db, policy));
    }
  }
  await executeInChunks(db, statements, 25);
}

async function deleteStaleProviderRows(db: D1Database, provider: string, nowSeconds: number): Promise<number> {
  const result = await db
    .prepare(
      `DELETE FROM model_catalog
       WHERE provider = ?
         AND COALESCE(last_sync_at, 0) < ?`,
    )
    .bind(provider, nowSeconds)
    .run();
  return result.meta.changes ?? 0;
}

async function completeProviderSyncRun(
  db: D1Database,
  runId: string,
  discoveredCount: number,
  upsertedCount: number,
  hiddenCount: number,
  nowSeconds: number,
): Promise<void> {
  await db
    .prepare(
      `UPDATE provider_sync_runs
       SET status = 'succeeded',
           discovered_model_count = ?,
           upserted_model_count = ?,
           hidden_model_count = ?,
           completed_at = ?
       WHERE id = ?`,
    )
    .bind(discoveredCount, upsertedCount, hiddenCount, nowSeconds, runId)
    .run();
}

async function failProviderSyncRun(db: D1Database, runId: string, errorSummary: string, nowSeconds: number): Promise<void> {
  await db
    .prepare(
      `UPDATE provider_sync_runs
       SET status = 'failed',
           error_count = 1,
           error_summary = ?,
           completed_at = ?
       WHERE id = ?`,
    )
    .bind(errorSummary, nowSeconds, runId)
    .run();
}

async function syncCatalogProvider(
  env: Env,
  provider: string,
  nowSeconds: number,
  loader: () => Promise<CatalogSyncBundle[]>,
  emptyErrorCode?: string,
): Promise<ProviderSyncSummary> {
  const db = env.MODEL_CATALOG_DB;
  if (!db) {
    throw new Error("model_catalog_db_missing");
  }

  const runId = await insertProviderSyncRun(db, provider, nowSeconds);

  try {
    const bundles = await loader();
    if (bundles.length === 0 && emptyErrorCode) {
      throw new Error(emptyErrorCode);
    }

    await upsertCatalogBundles(db, bundles);
    const deletedCount = await deleteStaleProviderRows(db, provider, nowSeconds);
    await completeProviderSyncRun(db, runId, bundles.length, bundles.length, deletedCount, nowSeconds);

    return {
      provider,
      runId,
      discoveredModelCount: bundles.length,
      upsertedModelCount: bundles.length,
      hiddenModelCount: deletedCount,
      status: "succeeded",
    };
  } catch (error) {
    const errorSummary = error instanceof Error ? error.message : trim(error);
    await failProviderSyncRun(db, runId, errorSummary, nowSeconds);
    throw error;
  }
}

export async function syncOpenRouterCatalog(env: Env, nowSeconds = Math.trunc(Date.now() / 1000)): Promise<ProviderSyncSummary> {
  return syncCatalogProvider(
    env,
    "openrouter",
    nowSeconds,
    async () => {
      const models = await fetchOpenRouterModels(env.OPENROUTER_API_KEY);
      return mapOpenRouterModelsToSyncBundles(models, env, nowSeconds);
    },
    "openrouter_zero_cost_models_empty",
  );
}

export async function syncNvidiaCatalog(env: Env, nowSeconds = Math.trunc(Date.now() / 1000)): Promise<ProviderSyncSummary> {
  return syncCatalogProvider(
    env,
    "nvidia",
    nowSeconds,
    async () => {
      const models = await fetchNvidiaModels(env.NVIDIA_API_KEY);
      return mapNvidiaModelsToSyncBundles(models, env, nowSeconds);
    },
    "nvidia_models_empty",
  );
}
