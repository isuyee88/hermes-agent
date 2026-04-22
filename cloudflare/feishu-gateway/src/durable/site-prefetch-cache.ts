import { DurableObject } from "cloudflare:workers";
import type { Env, JsonValue, PrefetchBudgetState, SitePrefetchManifest } from "../runtime";
import { SITE_PREFETCH_ERROR_TTL_MS, jsonResponse, parseJsonSafely, trim } from "../runtime";
import { coerceSitePrefetchManifest, isSitePrefetchFresh, todayKey } from "../browser/prefetch/manifest";

function emptyBudgetState(day: string): PrefetchBudgetState {
  return {
    day,
    browser_ms_used: 0,
    crawl_jobs: 0,
    last_quick_action_at: 0,
  };
}

/**
 * Durable Object for short-lived site prefetch manifests and shared browser budget state.
 */
export class SitePrefetchCache extends DurableObject<Env> {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const payload = request.method === "POST" ? parseJsonSafely(await request.text()) : {};

    if (url.pathname === "/manifest" && request.method === "GET") {
      const manifest = coerceSitePrefetchManifest(await this.ctx.storage.get("manifest"));
      return jsonResponse({ ok: true, manifest: manifest as unknown as JsonValue });
    }

    if (url.pathname === "/manifest" && request.method === "POST") {
      const manifest = coerceSitePrefetchManifest(payload.manifest);
      if (!manifest) {
        return jsonResponse({ ok: false, error: "invalid_manifest" }, { status: 400 });
      }
      await this.ctx.storage.put("manifest", manifest);
      return jsonResponse({ ok: true, manifest: manifest as unknown as JsonValue });
    }

    if (url.pathname === "/claim" && request.method === "POST") {
      const now = Date.now();
      const existing = coerceSitePrefetchManifest(await this.ctx.storage.get("manifest"));
      if (isSitePrefetchFresh(existing) && (existing.status === "completed" || existing.status === "pending")) {
        return jsonResponse({ ok: true, claimed: false, manifest: existing as unknown as JsonValue });
      }
      const pendingManifest: SitePrefetchManifest = {
        key: trim(payload.key) || trim(existing?.key),
        category: (trim(payload.category) as SitePrefetchManifest["category"]) || "none",
        mode: "none",
        domain: trim(payload.domain),
        intent: (trim(payload.intent) as SitePrefetchManifest["intent"]) || "general",
        target_url: trim(payload.target_url),
        final_url: trim(payload.target_url),
        page_title: "",
        candidate_urls: [],
        top_nav_links: [],
        summary: "",
        sections: [],
        page_kind: "",
        primary_actions: [],
        forms_summary: [],
        input_fields: [],
        dialog_or_banner: [],
        auth_required_guess: false,
        a11y_snapshot_summary: "",
        confidence: 0,
        source_evidence: ["pending"],
        status: "pending",
        fetched_at: new Date(now).toISOString(),
        expires_at: new Date(now + SITE_PREFETCH_ERROR_TTL_MS).toISOString(),
      };
      await this.ctx.storage.put("manifest", pendingManifest);
      return jsonResponse({ ok: true, claimed: true, manifest: pendingManifest as unknown as JsonValue });
    }

    if (url.pathname === "/budget" && request.method === "GET") {
      const currentDay = todayKey();
      const stored = (await this.ctx.storage.get<PrefetchBudgetState>("budget_state")) ?? emptyBudgetState(currentDay);
      const budget = stored.day === currentDay ? stored : emptyBudgetState(currentDay);
      if (budget.day !== stored.day) {
        await this.ctx.storage.put("budget_state", budget);
      }
      return jsonResponse({ ok: true, ...budget });
    }

    if (url.pathname === "/budget" && request.method === "POST") {
      const currentDay = todayKey();
      const stored = (await this.ctx.storage.get<PrefetchBudgetState>("budget_state")) ?? emptyBudgetState(currentDay);
      const next: PrefetchBudgetState =
        stored.day === currentDay
          ? { ...stored }
          : emptyBudgetState(currentDay);
      if (payload.browser_ms_used !== undefined) {
        next.browser_ms_used = Math.max(0, Number(payload.browser_ms_used) || 0);
      }
      if (payload.crawl_jobs !== undefined) {
        next.crawl_jobs = Math.max(0, Number(payload.crawl_jobs) || 0);
      }
      if (payload.last_quick_action_at !== undefined) {
        next.last_quick_action_at = Math.max(0, Number(payload.last_quick_action_at) || 0);
      }
      next.browser_ms_used += Math.max(0, Number(payload.browser_ms_delta) || 0);
      next.crawl_jobs += Math.max(0, Number(payload.crawl_jobs_delta) || 0);
      await this.ctx.storage.put("budget_state", next);
      return jsonResponse({ ok: true, ...next });
    }

    return jsonResponse({ ok: false, error: "not_found" }, { status: 404 });
  }
}
