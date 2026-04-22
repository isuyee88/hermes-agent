import type { SiteCategory, SiteIntent, SitePrefetchManifest, SitePrefetchMode } from "../../contracts/gateway";
import { SITE_PREFETCH_MAX_ARIA_CHARS, SITE_PREFETCH_MAX_SUMMARY_CHARS, compactList, compactText } from "./shared";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

export function isSitePrefetchFresh(
  manifest: SitePrefetchManifest | null | undefined,
): manifest is SitePrefetchManifest {
  if (!manifest) {
    return false;
  }
  const expiresAt = Date.parse(trim(manifest.expires_at));
  return Number.isFinite(expiresAt) && expiresAt > Date.now();
}

export function coerceSitePrefetchManifest(value: unknown): SitePrefetchManifest | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  return {
    key: trim(record.key),
    category: (trim(record.category) as SiteCategory) || "none",
    mode: (trim(record.mode) as SitePrefetchMode) || "none",
    domain: trim(record.domain),
    intent: (trim(record.intent) as SiteIntent) || "general",
    target_url: trim(record.target_url),
    final_url: trim(record.final_url),
    page_title: compactText(trim(record.page_title), 160),
    candidate_urls: compactList(Array.isArray(record.candidate_urls) ? record.candidate_urls.map((item) => trim(item)) : []),
    top_nav_links: compactList(Array.isArray(record.top_nav_links) ? record.top_nav_links.map((item) => trim(item)) : [], 12, 180),
    summary: compactText(trim(record.summary), SITE_PREFETCH_MAX_SUMMARY_CHARS),
    sections: compactList(Array.isArray(record.sections) ? record.sections.map((item) => trim(item)) : [], 12, 120),
    page_kind: trim(record.page_kind),
    primary_actions: compactList(Array.isArray(record.primary_actions) ? record.primary_actions.map((item) => trim(item)) : []),
    forms_summary: compactList(Array.isArray(record.forms_summary) ? record.forms_summary.map((item) => trim(item)) : []),
    input_fields: compactList(Array.isArray(record.input_fields) ? record.input_fields.map((item) => trim(item)) : [], 16, 120),
    dialog_or_banner: compactList(Array.isArray(record.dialog_or_banner) ? record.dialog_or_banner.map((item) => trim(item)) : [], 6, 180),
    auth_required_guess: record.auth_required_guess === true,
    a11y_snapshot_summary: compactText(trim(record.a11y_snapshot_summary), SITE_PREFETCH_MAX_ARIA_CHARS),
    confidence: Number.isFinite(Number(record.confidence)) ? Number(record.confidence) : 0,
    source_evidence: compactList(Array.isArray(record.source_evidence) ? record.source_evidence.map((item) => trim(item)) : [], 12, 120),
    status: (trim(record.status) as SitePrefetchManifest["status"]) || "error",
    fetched_at: trim(record.fetched_at) || new Date().toISOString(),
    expires_at: trim(record.expires_at) || new Date().toISOString(),
    error_class: trim(record.error_class),
    token_estimate: Number.isFinite(Number(record.token_estimate)) ? Number(record.token_estimate) : undefined,
    browser_ms_used: Number.isFinite(Number(record.browser_ms_used)) ? Number(record.browser_ms_used) : undefined,
    crawl_job_id: trim(record.crawl_job_id) || undefined,
  };
}

export function todayKey(): string {
  return new Date().toISOString().slice(0, 10);
}
