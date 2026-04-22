import { Readability } from "@mozilla/readability";
import { DOMParser } from "linkedom";
import { checkQuickActionBudget, recordQuickActionUsage } from "../budget";
import {
  buildSitePrefetchErrorManifest,
  buildSitePrefetchManifestBase,
  compactList,
  compactText,
  estimateTokenCountFromMarkdown,
  inferPageKindFromSignals,
  parseBrowserMsUsed,
  sleep,
  summarizeMarkdownContent,
} from "./shared";
import type { Env, FeishuNormalizedPayload, JsonValue, SitePrefetchManifest } from "../../runtime";

const SITE_PREFETCH_CONTENT_TTL_MS = 6 * 60 * 60 * 1000;

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function parsePositiveInt(value: unknown, fallback: number, min = 1, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = Number.parseInt(trim(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, parsed));
}

function toReadabilityDocument(document: ReturnType<DOMParser["parseFromString"]>): Document {
  return document as unknown as Document;
}

async function browserRunFetch(
  options: {
    accountId: string;
    apiToken: string;
  },
  path: string,
  body: Record<string, JsonValue>,
  query?: Record<string, string | number>,
): Promise<{ payload: Record<string, JsonValue>; browserMsUsed: number }> {
  const url = new URL(`https://api.cloudflare.com/client/v4/accounts/${options.accountId}/browser-rendering/${path}`);
  for (const [key, value] of Object.entries(query ?? {})) {
    url.searchParams.set(key, String(value));
  }
  const startedAt = Date.now();
  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      authorization: `Bearer ${options.apiToken}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const payload = (await response.json()) as Record<string, JsonValue>;
  const browserMsUsed = parseBrowserMsUsed(response.headers, Date.now() - startedAt);
  if (!response.ok || payload.success === false) {
    const errorMessage = Array.isArray(payload.errors)
      ? payload.errors
          .map((item) => {
            if (!item || typeof item !== "object" || Array.isArray(item)) {
              return "";
            }
            return trim((item as Record<string, JsonValue>).message);
          })
          .filter(Boolean)
          .join(",")
      : "";
    throw new Error(`browser_run_${path}_failed:${response.status}:${trim(errorMessage)}`);
  }
  return { payload, browserMsUsed };
}

async function fetchMarkdownForAgents(targetUrl: string): Promise<{ markdown: string; tokenEstimate: number } | null> {
  const response = await fetch(targetUrl, {
    headers: {
      accept: "text/markdown",
    },
  });
  if (!response.ok) {
    return null;
  }
  const contentType = trim(response.headers.get("content-type")).toLowerCase();
  if (!contentType.includes("text/markdown")) {
    return null;
  }
  const markdown = await response.text();
  return {
    markdown,
    tokenEstimate: parsePositiveInt(response.headers.get("x-markdown-tokens"), estimateTokenCountFromMarkdown(markdown), 0, 1_000_000),
  };
}

export async function buildContentSitePrefetch(
  env: Env,
  normalized: FeishuNormalizedPayload,
  options: {
    allowCrawl: boolean;
    accountId: string;
    apiToken: string;
    budgetOptions: {
      browserRunConfigured: boolean;
      dailyBudgetMs: number;
      crawlLimitPerDay: number;
      quickActionIntervalMs: number;
    };
  },
): Promise<SitePrefetchManifest> {
  const markdownForAgents = await fetchMarkdownForAgents(normalized.target_url);
  if (markdownForAgents && trim(markdownForAgents.markdown)) {
    const summaryPayload = summarizeMarkdownContent(markdownForAgents.markdown);
    return {
      ...buildSitePrefetchManifestBase(normalized, "markdown_for_agents", SITE_PREFETCH_CONTENT_TTL_MS),
      final_url: normalized.target_url,
      page_title: summaryPayload.sections[0] ?? "",
      candidate_urls: compactList([normalized.target_url, ...summaryPayload.candidateUrls], 12, 200),
      summary: summaryPayload.summary,
      sections: summaryPayload.sections,
      page_kind: inferPageKindFromSignals(normalized.target_url, "", [], [], summaryPayload.summary),
      confidence: 0.92,
      source_evidence: ["markdown_for_agents"],
      status: "completed",
      token_estimate: markdownForAgents.tokenEstimate,
    };
  }

  const markdownBudget = await checkQuickActionBudget(env, "markdown", options.budgetOptions);
  if (!markdownBudget.allowed) {
    return buildSitePrefetchErrorManifest(normalized, "browser_markdown", "rate_limited", markdownBudget.reason);
  }

  const markdownResult = await browserRunFetch(
    options,
    "markdown",
    {
      url: normalized.target_url,
      gotoOptions: { waitUntil: "domcontentloaded", timeout: 25000 },
      rejectResourceTypes: ["image", "media", "font"],
      setJavaScriptEnabled: false,
      bestAttempt: true,
    },
    { cacheTTL: 0 },
  );
  await recordQuickActionUsage(env, markdownResult.browserMsUsed, false);
  const markdown = trim(markdownResult.payload.result);
  if (markdown) {
    const summaryPayload = summarizeMarkdownContent(markdown);
    return {
      ...buildSitePrefetchManifestBase(normalized, "browser_markdown", SITE_PREFETCH_CONTENT_TTL_MS),
      final_url: normalized.target_url,
      page_title: summaryPayload.sections[0] ?? "",
      candidate_urls: compactList([normalized.target_url, ...summaryPayload.candidateUrls], 12, 200),
      summary: summaryPayload.summary,
      sections: summaryPayload.sections,
      page_kind: inferPageKindFromSignals(normalized.target_url, "", [], [], summaryPayload.summary),
      confidence: 0.86,
      source_evidence: ["browser_markdown"],
      status: "completed",
      token_estimate: estimateTokenCountFromMarkdown(markdown),
      browser_ms_used: markdownResult.browserMsUsed,
    };
  }

  if (options.allowCrawl && ["docs", "help", "blog"].includes(normalized.site_intent)) {
    const crawlBudget = await checkQuickActionBudget(env, "crawl", options.budgetOptions);
    if (crawlBudget.allowed) {
      const crawlStart = await browserRunFetch(
        options,
        "crawl",
        {
          url: normalized.target_url,
          limit: 12,
          depth: 1,
          formats: ["markdown"],
          render: false,
          options: {
            includePatterns: [`${normalized.target_url.replace(/\/$/, "")}/**`],
          },
          respectRobotsTxt: true,
        },
      );
      await recordQuickActionUsage(env, crawlStart.browserMsUsed, true);
      const crawlJobId = trim(crawlStart.payload.result);
      if (crawlJobId) {
        let crawlPayload: Record<string, JsonValue> | null = null;
        for (let attempt = 0; attempt < 3; attempt += 1) {
          await sleep(2000);
          const response = await fetch(
            `https://api.cloudflare.com/client/v4/accounts/${options.accountId}/browser-rendering/crawl/${encodeURIComponent(crawlJobId)}`,
            {
              headers: {
                authorization: `Bearer ${options.apiToken}`,
              },
            },
          );
          const payload = (await response.json()) as Record<string, JsonValue>;
          if (response.ok && payload.success !== false) {
            crawlPayload = payload;
          }
          const result = (crawlPayload?.result as Record<string, JsonValue> | undefined) ?? {};
          if (trim(result.status).toLowerCase() === "completed") {
            const records = Array.isArray(result.records) ? result.records : [];
            const markdownChunks = records
              .map((record) => {
                if (!record || typeof record !== "object" || Array.isArray(record)) {
                  return "";
                }
                return trim((record as Record<string, JsonValue>).markdown);
              })
              .filter(Boolean);
            if (markdownChunks.length > 0) {
              const combinedMarkdown = markdownChunks.join("\n\n");
              const summaryPayload = summarizeMarkdownContent(combinedMarkdown);
              return {
                ...buildSitePrefetchManifestBase(normalized, "browser_crawl", SITE_PREFETCH_CONTENT_TTL_MS),
                final_url: normalized.target_url,
                page_title: summaryPayload.sections[0] ?? "",
                candidate_urls: compactList([normalized.target_url, ...summaryPayload.candidateUrls], 16, 200),
                summary: summaryPayload.summary,
                sections: summaryPayload.sections,
                page_kind: inferPageKindFromSignals(normalized.target_url, "", [], [], summaryPayload.summary),
                confidence: 0.82,
                source_evidence: ["browser_crawl"],
                status: "completed",
                token_estimate: estimateTokenCountFromMarkdown(combinedMarkdown),
                browser_ms_used: Number(result.browserSecondsUsed ?? 0) * 1000 || crawlStart.browserMsUsed,
                crawl_job_id: crawlJobId,
              };
            }
            break;
          }
        }
      }
    }
  }

  const contentBudget = await checkQuickActionBudget(env, "content", options.budgetOptions);
  if (!contentBudget.allowed) {
    return buildSitePrefetchErrorManifest(normalized, "readability_fallback", "rate_limited", contentBudget.reason);
  }
  const contentResult = await browserRunFetch(
    options,
    "content",
    {
      url: normalized.target_url,
      gotoOptions: { waitUntil: "domcontentloaded", timeout: 25000 },
      rejectResourceTypes: ["image", "media", "font"],
      setJavaScriptEnabled: true,
      bestAttempt: true,
    },
    { cacheTTL: 0 },
  );
  await recordQuickActionUsage(env, contentResult.browserMsUsed, false);
  const html = trim(contentResult.payload.result);
  if (!html) {
    return buildSitePrefetchErrorManifest(normalized, "readability_fallback", "empty_content", "browser content returned empty html");
  }
  const document = new DOMParser().parseFromString(html, "text/html");
  const article = new Readability(toReadabilityDocument(document)).parse();
  const articleText = trim(article?.textContent ?? article?.excerpt ?? "");
  const articleTitle = trim(article?.title);
  if (!articleText) {
    return buildSitePrefetchErrorManifest(normalized, "readability_fallback", "empty_content", "readability returned no article");
  }
  const summary = compactText(articleText);
  const sections = compactList(
    articleText
      .split(/\r?\n/)
      .map((line) => trim(line))
      .filter((line) => line.length > 8)
      .slice(0, 8),
    8,
    120,
  );
  return {
    ...buildSitePrefetchManifestBase(normalized, "readability_fallback", SITE_PREFETCH_CONTENT_TTL_MS),
    final_url: normalized.target_url,
    page_title: articleTitle,
    summary,
    sections,
    page_kind: inferPageKindFromSignals(normalized.target_url, articleTitle, [], [], summary),
    confidence: 0.68,
    source_evidence: ["browser_content", "readability_fallback"],
    status: "completed",
    token_estimate: estimateTokenCountFromMarkdown(articleText),
    browser_ms_used: contentResult.browserMsUsed,
  };
}
