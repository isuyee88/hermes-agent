import type { Env, FeishuNormalizedPayload, JsonValue, PrefetchBudgetState, SitePrefetchManifest } from "../runtime";
import { coerceSitePrefetchManifest } from "./prefetch/manifest";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

export function buildSitePrefetchKey(normalized: FeishuNormalizedPayload): string {
  return `${normalized.target_domain || "unknown"}|${normalized.site_intent || "general"}|${normalized.site_category || "none"}`;
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

export async function getSitePrefetchManifest(env: Env, key: string): Promise<SitePrefetchManifest | null> {
  const payload = await sitePrefetchCacheFetch(env, key, "/manifest");
  return coerceSitePrefetchManifest(payload.manifest);
}

export async function claimSitePrefetchManifest(
  env: Env,
  key: string,
  category: SitePrefetchManifest["category"],
  intent: SitePrefetchManifest["intent"],
  domain: string,
  targetUrl: string,
): Promise<{ claimed: boolean; manifest: SitePrefetchManifest | null }> {
  const payload = await sitePrefetchCacheFetch(env, key, "/claim", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ key, category, intent, domain, target_url: targetUrl }),
  });
  return {
    claimed: payload.claimed === true,
    manifest: coerceSitePrefetchManifest(payload.manifest),
  };
}

export async function storeSitePrefetchManifest(env: Env, key: string, manifest: SitePrefetchManifest): Promise<void> {
  await sitePrefetchCacheFetch(env, key, "/manifest", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ manifest }),
  });
}

export type BrowserBudgetPatch = Partial<PrefetchBudgetState> & {
  browser_ms_delta?: number;
  crawl_jobs_delta?: number;
};
