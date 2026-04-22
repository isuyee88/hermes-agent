import {
  buildSitePrefetchKey,
  claimSitePrefetchManifest,
  getSitePrefetchManifest,
  storeSitePrefetchManifest,
} from "../cache";
import type { Env, FeishuNormalizedPayload, SitePrefetchManifest } from "../../runtime";
import { buildContentSitePrefetch } from "./content";
import { isSitePrefetchFresh } from "./manifest";
import { buildInteractionLightPrefetch } from "./interactive";
import { getCloudflareAccountId } from "./shared";

const BROWSER_RUN_DAILY_BUDGET_MS = 10 * 60 * 1000;
const BROWSER_RUN_CRAWL_LIMIT_PER_DAY = 5;
const BROWSER_RUN_QUICK_ACTION_INTERVAL_MS = 10 * 1000;

type PrefetchOrchestratorDeps = {
  trim: (value: unknown) => string;
  log: (event: string, extra: Record<string, unknown>) => void;
  buildRoutingLogFields: (
    normalized: FeishuNormalizedPayload,
    overrides?: Partial<{ site_prefetch: SitePrefetchManifest | null }>,
  ) => Record<string, unknown>;
  inferSiteExecutionStage: (normalized: FeishuNormalizedPayload) => string;
  buildSitePrefetchErrorManifest: (
    normalized: FeishuNormalizedPayload,
    mode: SitePrefetchManifest["mode"],
    errorClass: string,
    summary: string,
  ) => SitePrefetchManifest;
  getCloudflareAiGatewayBase: (env: Env) => string;
};

function canUseBrowserRun(env: Env, deps: PrefetchOrchestratorDeps): boolean {
  return !!deps.trim(env.CLOUDFLARE_API_TOKEN) && !!getCloudflareAccountId(env, deps.getCloudflareAiGatewayBase(env));
}

function browserBudgetOptions(env: Env, deps: PrefetchOrchestratorDeps): {
  browserRunConfigured: boolean;
  dailyBudgetMs: number;
  crawlLimitPerDay: number;
  quickActionIntervalMs: number;
} {
  return {
    browserRunConfigured: canUseBrowserRun(env, deps),
    dailyBudgetMs: BROWSER_RUN_DAILY_BUDGET_MS,
    crawlLimitPerDay: BROWSER_RUN_CRAWL_LIMIT_PER_DAY,
    quickActionIntervalMs: BROWSER_RUN_QUICK_ACTION_INTERVAL_MS,
  };
}

/**
 * Edge prefetch orchestration owns cache claim/store and fail-open logging.
 * Content extraction and interactive preflight stay delegated to their dedicated executors.
 */
export async function maybeBuildSitePrefetch(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: PrefetchOrchestratorDeps,
  options: { allowCrawl: boolean } = { allowCrawl: false },
): Promise<SitePrefetchManifest | null> {
  if (
    normalized.lane !== "agent" ||
    !normalized.target_domain ||
    !normalized.target_url ||
    normalized.site_category === "none" ||
    normalized.site_category === "site_interactive_heavy"
  ) {
    return null;
  }
  try {
    if (!env.SITE_PREFETCH_CACHE || typeof env.SITE_PREFETCH_CACHE.get !== "function") {
      deps.log("feishu.site_prefetch.skipped", {
        correlation_id: normalized.correlation_id,
        site_execution_stage: deps.inferSiteExecutionStage(normalized),
        reason: "site_prefetch_cache_binding_unavailable",
        ...deps.buildRoutingLogFields(normalized),
      });
      return null;
    }

    const key = buildSitePrefetchKey(normalized);
    const cached = await getSitePrefetchManifest(env, key);
    if (isSitePrefetchFresh(cached) && cached.status === "completed") {
      deps.log("feishu.site_prefetch.cache_hit", {
        correlation_id: normalized.correlation_id,
        key,
        site_execution_stage: deps.inferSiteExecutionStage(normalized),
        mode: cached.mode,
        status: cached.status,
        ...deps.buildRoutingLogFields(normalized, { site_prefetch: cached }),
      });
      return cached;
    }

    const claim = await claimSitePrefetchManifest(
      env,
      key,
      normalized.site_category,
      normalized.site_intent,
      normalized.target_domain,
      normalized.target_url,
    );
    if (!claim.claimed) {
      return isSitePrefetchFresh(claim.manifest) && claim.manifest?.status === "completed" ? claim.manifest : null;
    }

    let manifest: SitePrefetchManifest;
    try {
      manifest =
        normalized.site_category === "site_content"
          ? await buildContentSitePrefetch(env, normalized, {
              allowCrawl: options.allowCrawl,
              accountId: getCloudflareAccountId(env, deps.getCloudflareAiGatewayBase(env)),
              apiToken: deps.trim(env.CLOUDFLARE_API_TOKEN),
              budgetOptions: browserBudgetOptions(env, deps),
            })
          : await buildInteractionLightPrefetch(env, normalized);
    } catch (error) {
      manifest = deps.buildSitePrefetchErrorManifest(
        normalized,
        normalized.site_category === "site_content" ? "browser_markdown" : "playwright_preflight",
        "prefetch_failed",
        error instanceof Error ? error.message : String(error),
      );
    }

    await storeSitePrefetchManifest(env, key, manifest);
    deps.log("feishu.site_prefetch.done", {
      correlation_id: normalized.correlation_id,
      key,
      site_execution_stage: deps.inferSiteExecutionStage(normalized),
      mode: manifest.mode,
      status: manifest.status,
      confidence: manifest.confidence,
      browser_ms_used: manifest.browser_ms_used ?? 0,
      ...deps.buildRoutingLogFields(normalized, { site_prefetch: manifest }),
    });
    return manifest.status === "completed" ? manifest : null;
  } catch (error) {
    deps.log("feishu.site_prefetch.skipped", {
      correlation_id: normalized.correlation_id,
      site_execution_stage: deps.inferSiteExecutionStage(normalized),
      reason: "prefetch_fail_open",
      error: error instanceof Error ? error.message : String(error),
      ...deps.buildRoutingLogFields(normalized),
    });
    return null;
  }
}
