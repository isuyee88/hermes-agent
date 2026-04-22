import type { Env, JsonValue, PrefetchBudgetState } from "../runtime";
import { todayKey } from "./prefetch/manifest";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

async function sitePrefetchCacheFetch(
  env: Env,
  key: string,
  path: string,
  init?: RequestInit,
): Promise<Record<string, JsonValue>> {
  const stub = env.SITE_PREFETCH_CACHE.get(env.SITE_PREFETCH_CACHE.idFromName(key));
  const response = await stub.fetch(`https://site-prefetch${path}`, init);
  const payload = (await response.json()) as Record<string, JsonValue>;
  if (!response.ok) {
    throw new Error(`site_prefetch_cache_failed:${response.status}:${trim(payload.error)}`);
  }
  return payload;
}

async function getPrefetchBudgetState(env: Env): Promise<PrefetchBudgetState> {
  const payload = await sitePrefetchCacheFetch(env, "budget:global", "/budget");
  return {
    day: trim(payload.day) || todayKey(),
    browser_ms_used: Number(payload.browser_ms_used ?? 0) || 0,
    crawl_jobs: Number(payload.crawl_jobs ?? 0) || 0,
    last_quick_action_at: Number(payload.last_quick_action_at ?? 0) || 0,
  };
}

async function updatePrefetchBudgetState(
  env: Env,
  updates: Partial<PrefetchBudgetState> & { browser_ms_delta?: number; crawl_jobs_delta?: number },
): Promise<PrefetchBudgetState> {
  const payload = await sitePrefetchCacheFetch(env, "budget:global", "/budget", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(updates),
  });
  return {
    day: trim(payload.day) || todayKey(),
    browser_ms_used: Number(payload.browser_ms_used ?? 0) || 0,
    crawl_jobs: Number(payload.crawl_jobs ?? 0) || 0,
    last_quick_action_at: Number(payload.last_quick_action_at ?? 0) || 0,
  };
}

export async function checkQuickActionBudget(
  env: Env,
  kind: "markdown" | "content" | "crawl",
  options: {
    browserRunConfigured: boolean;
    dailyBudgetMs: number;
    crawlLimitPerDay: number;
    quickActionIntervalMs: number;
  },
): Promise<{ allowed: boolean; state: PrefetchBudgetState; reason: string }> {
  const state = await getPrefetchBudgetState(env);
  const now = Date.now();
  if (!options.browserRunConfigured) {
    return { allowed: false, state, reason: "browser_run_not_configured" };
  }
  if (state.browser_ms_used >= options.dailyBudgetMs * 0.9) {
    return { allowed: false, state, reason: "browser_run_daily_budget_exhausted" };
  }
  if (kind === "crawl") {
    if (state.browser_ms_used >= options.dailyBudgetMs * 0.7) {
      return { allowed: false, state, reason: "crawl_budget_guard" };
    }
    if (state.crawl_jobs >= options.crawlLimitPerDay - 1) {
      return { allowed: false, state, reason: "crawl_jobs_guard" };
    }
  }
  if (state.last_quick_action_at > 0 && now - state.last_quick_action_at < options.quickActionIntervalMs) {
    return { allowed: false, state, reason: "quick_action_rate_guard" };
  }
  return { allowed: true, state, reason: "allowed" };
}

export async function recordQuickActionUsage(
  env: Env,
  browserMsUsed: number,
  crawlJobUsed = false,
): Promise<void> {
  await updatePrefetchBudgetState(env, {
    browser_ms_delta: Math.max(0, browserMsUsed),
    crawl_jobs_delta: crawlJobUsed ? 1 : 0,
    last_quick_action_at: Date.now(),
  });
}
