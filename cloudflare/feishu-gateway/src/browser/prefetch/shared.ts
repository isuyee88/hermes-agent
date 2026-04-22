import { buildSitePrefetchKey } from "../cache";
import type { Env, FeishuNormalizedPayload, SitePrefetchManifest } from "../../runtime";

const SITE_PREFETCH_ERROR_TTL_MS = 10 * 60 * 1000;
export const SITE_PREFETCH_MAX_SUMMARY_CHARS = 1600;
export const SITE_PREFETCH_MAX_ARIA_CHARS = 2500;

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => trim(value)).filter(Boolean)));
}

function normalizeTargetUrl(rawValue: string): string {
  const candidate = trim(rawValue);
  if (!candidate) {
    return "";
  }
  try {
    const url = new URL(candidate);
    if (!/^https?:$/i.test(url.protocol)) {
      return "";
    }
    url.hash = "";
    return url.toString();
  } catch {
    return "";
  }
}

export function compactText(value: string, maxChars = SITE_PREFETCH_MAX_SUMMARY_CHARS): string {
  const normalized = trim(value).replace(/\s+/g, " ");
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxChars - 3))}...`;
}

export function compactList(values: string[], maxItems = 10, maxItemChars = 160): string[] {
  return uniqueStrings(values.map((value) => compactText(value, maxItemChars))).slice(0, maxItems);
}

function buildSitePrefetchExpiry(ttlMs: number): string {
  return new Date(Date.now() + ttlMs).toISOString();
}

export function parseBrowserMsUsed(headers: Headers, fallbackMs: number): number {
  const headerCandidates = ["x-browser-ms-used", "X-Browser-Ms-Used", "cf-browser-ms-used"];
  for (const headerName of headerCandidates) {
    const value = Number(headers.get(headerName) ?? "");
    if (Number.isFinite(value) && value >= 0) {
      return value;
    }
  }
  return Math.max(0, fallbackMs);
}

export function getCloudflareAccountId(env: Env, fallbackGatewayBaseUrl = ""): string {
  const explicit = trim(env.CLOUDFLARE_ACCOUNT_ID);
  if (explicit) {
    return explicit;
  }
  const match = fallbackGatewayBaseUrl.match(/\/v1\/([^/]+)\//i);
  return match?.[1] ?? "";
}

export function estimateTokenCountFromMarkdown(markdown: string): number {
  const normalized = trim(markdown);
  if (!normalized) {
    return 0;
  }
  return Math.ceil(normalized.length / 4);
}

function extractMarkdownLinks(markdown: string): string[] {
  const matches = markdown.match(/\[[^\]]+\]\((https?:\/\/[^)\s]+)\)/gi) ?? [];
  return compactList(
    matches
      .map((match) => match.match(/\((https?:\/\/[^)\s]+)\)/i)?.[1] ?? "")
      .map((value) => normalizeTargetUrl(value))
      .filter(Boolean),
    12,
    200,
  );
}

export function summarizeMarkdownContent(markdown: string): {
  summary: string;
  sections: string[];
  candidateUrls: string[];
} {
  const lines = markdown
    .split(/\r?\n/)
    .map((line) => trim(line))
    .filter((line) => line && !line.startsWith("```"));
  const sections = compactList(
    lines
      .filter((line) => /^#{1,4}\s+/.test(line))
      .map((line) => line.replace(/^#{1,4}\s+/, "")),
    10,
    120,
  );
  const summary = compactText(lines.filter((line) => !/^#{1,4}\s+/.test(line)).slice(0, 8).join(" "));
  return {
    summary,
    sections,
    candidateUrls: extractMarkdownLinks(markdown),
  };
}

export function inferPageKindFromSignals(
  url: string,
  title: string,
  formsSummary: string[],
  primaryActions: string[],
  summary = "",
): string {
  const combined = [url, title, summary, ...formsSummary, ...primaryActions].join(" ").toLowerCase();
  if (/\bsign[ -]?in\b|\blogin\b|鐧诲綍|鐧婚檰/.test(combined)) {
    return "login";
  }
  if (/\bsign[ -]?up\b|\bregister\b|娉ㄥ唽/.test(combined)) {
    return "signup";
  }
  if (/\bdocs?\b|\bdocumentation\b|\bapi\b|鏂囨。|鎺ュ彛/.test(combined)) {
    return "docs";
  }
  if (/\bpricing\b|\bplan\b|浠锋牸|濂楅|璁¤垂/.test(combined)) {
    return "pricing";
  }
  if (formsSummary.length > 0) {
    return "form";
  }
  return "generic";
}

export function buildSitePrefetchManifestBase(
  normalized: FeishuNormalizedPayload,
  mode: SitePrefetchManifest["mode"],
  ttlMs: number,
): SitePrefetchManifest {
  return {
    key: buildSitePrefetchKey(normalized),
    category: normalized.site_category,
    mode,
    domain: normalized.target_domain,
    intent: normalized.site_intent,
    target_url: normalized.target_url,
    final_url: normalized.target_url,
    page_title: "",
    candidate_urls: normalized.target_url ? [normalized.target_url] : [],
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
    source_evidence: [],
    status: "error",
    fetched_at: new Date().toISOString(),
    expires_at: buildSitePrefetchExpiry(ttlMs),
  };
}

export function buildSitePrefetchErrorManifest(
  normalized: FeishuNormalizedPayload,
  mode: SitePrefetchManifest["mode"],
  errorClass: string,
  summary: string,
): SitePrefetchManifest {
  return {
    ...buildSitePrefetchManifestBase(normalized, mode, SITE_PREFETCH_ERROR_TTL_MS),
    summary: compactText(summary),
    status: errorClass === "rate_limited" ? "rate_limited" : "error",
    error_class: errorClass,
    source_evidence: compactList([errorClass]),
  };
}

export async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}
