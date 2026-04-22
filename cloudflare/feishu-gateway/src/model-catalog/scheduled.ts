import type { Env } from "../runtime";
import { syncModelRegistryToFeishuBitable } from "./feishu-sync";
import { applyModelHealthPolicies, publishDynamicRoutes } from "./routes";
import { syncNvidiaCatalog, syncOpenRouterCatalog } from "./sync";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function parseBoolean(value: unknown, fallback: boolean): boolean {
  const normalized = trim(value).toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

function parseProviders(value: unknown): string[] {
  const normalized = trim(value);
  if (!normalized) {
    return ["openrouter", "nvidia"];
  }
  const providers = normalized
    .split(",")
    .map((item) => trim(item).toLowerCase())
    .filter(Boolean);
  return providers.length > 0 ? Array.from(new Set(providers)) : ["openrouter", "nvidia"];
}

type CatalogSummaryRow = {
  total_models?: number;
  hidden_models?: number;
  available_models?: number;
};

export type ModelCatalogTickSummary = {
  syncEnabled: boolean;
  providers: string[];
  healthUpdates: number;
  publishedRoutes: Array<{
    routeName: string;
    published: boolean;
    candidateModels: number;
  }>;
  feishuMirror: Awaited<ReturnType<typeof syncModelRegistryToFeishuBitable>>;
  totalModels: number;
  hiddenModels: number;
  availableModels: number;
};

export async function runModelCatalogScheduledTick(env: Env, controller: ScheduledController): Promise<ModelCatalogTickSummary | null> {
  const db = env.MODEL_CATALOG_DB;
  const cron = trim(controller.cron) || "manual";
  const scheduledAt = Math.trunc((controller.scheduledTime || Date.now()) / 1000);
  const syncEnabled = parseBoolean(env.HERMES_MODEL_CATALOG_SYNC_ENABLED, true);
  const providers = parseProviders(env.HERMES_MODEL_CATALOG_SYNC_PROVIDERS);

  if (!db) {
    console.log(
      JSON.stringify({
        event: "model_catalog.tick.skipped",
        reason: "db_binding_missing",
        cron,
        scheduled_time: scheduledAt,
        sync_enabled: syncEnabled,
        providers,
      }),
    );
    return null;
  }

  if (syncEnabled) {
    if (providers.includes("openrouter")) {
      await syncOpenRouterCatalog(env, scheduledAt);
    }
    if (providers.includes("nvidia")) {
      await syncNvidiaCatalog(env, scheduledAt);
    }
  } else {
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
           hidden_model_count,
           error_count,
           error_summary,
           started_at,
           completed_at,
           created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .bind(
        runId,
        "system",
        "scheduled",
        "control_plane_tick",
        "paused",
        0,
        0,
        0,
        "catalog sync disabled by HERMES_MODEL_CATALOG_SYNC_ENABLED",
        scheduledAt,
        scheduledAt,
        scheduledAt,
      )
      .run();
  }

  const healthUpdates = await applyModelHealthPolicies(env, scheduledAt);
  const publishedRoutes = await publishDynamicRoutes(env, scheduledAt);
  const feishuMirror = await syncModelRegistryToFeishuBitable(env, scheduledAt);

  const summary = await db
    .prepare(
      `SELECT
         COUNT(*) AS total_models,
         COALESCE(SUM(CASE WHEN hidden = 1 THEN 1 ELSE 0 END), 0) AS hidden_models,
         COALESCE(SUM(CASE WHEN is_available = 1 THEN 1 ELSE 0 END), 0) AS available_models
       FROM model_catalog`,
    )
    .first<CatalogSummaryRow>();

  console.log(
    JSON.stringify({
      event: "model_catalog.tick.completed",
      cron,
      scheduled_time: scheduledAt,
      sync_enabled: syncEnabled,
      providers,
      health_policy_rows: healthUpdates,
      published_routes: publishedRoutes.map((item) => ({
        route_name: item.routeName,
        published: item.published,
        candidate_models: item.candidateModels,
      })),
      feishu_mirror: feishuMirror,
      total_models: summary?.total_models ?? 0,
      hidden_models: summary?.hidden_models ?? 0,
      available_models: summary?.available_models ?? 0,
    }),
  );

  return {
    syncEnabled,
    providers,
    healthUpdates,
    publishedRoutes: publishedRoutes.map((item) => ({
      routeName: item.routeName,
      published: item.published,
      candidateModels: item.candidateModels,
    })),
    feishuMirror,
    totalModels: summary?.total_models ?? 0,
    hiddenModels: summary?.hidden_models ?? 0,
    availableModels: summary?.available_models ?? 0,
  };
}
