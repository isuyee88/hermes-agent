import { compactText } from "../browser/prefetch/shared";
import { buildSitePrefetchDirectRule } from "./route-policy";
import type { JsonValue, SessionHistoryEntry, SitePrefetchManifest } from "../contracts/gateway";
import { buildDomainSkillPrompt } from "./domain-skills";

type ConversationDeps = {
  trim: (value: unknown) => string;
  compactSessionHistory: (entries: SessionHistoryEntry[]) => SessionHistoryEntry[];
};

function buildDirectConversationHistory(
  history: SessionHistoryEntry[],
  latestUserText: string,
  deps: ConversationDeps,
): Array<Record<string, JsonValue>> {
  const messages = deps
    .compactSessionHistory(history)
    .map((entry) => ({
      role: entry.role,
      content: entry.content,
    }))
    .filter((entry) => deps.trim(entry.content));
  const latest = deps.trim(latestUserText);
  if (latest) {
    messages.push({ role: "user", content: latest });
  }
  return messages as Array<Record<string, JsonValue>>;
}

export function buildDirectConversationMessages(
  history: SessionHistoryEntry[],
  latestUserText: string,
  sitePrefetch: SitePrefetchManifest | null | undefined,
  deps: ConversationDeps,
): Array<Record<string, JsonValue>> {
  const messages = buildDirectConversationHistory(history, latestUserText, deps);
  const domainSkillPrompt = sitePrefetch ? buildDomainSkillPrompt(sitePrefetch.domain, sitePrefetch.final_url || sitePrefetch.target_url) : "";
  if (!sitePrefetch && !domainSkillPrompt) {
    return messages;
  }

  const directRule = buildSitePrefetchDirectRule(sitePrefetch);
  const prefetchPrompt = compactText(
    [
      "Cloudflare site prefetch (internal hint, do not quote unless useful).",
      `category=${sitePrefetch.category}`,
      `mode=${sitePrefetch.mode}`,
      `domain=${sitePrefetch.domain}`,
      `intent=${sitePrefetch.intent}`,
      sitePrefetch.final_url ? `final_url=${sitePrefetch.final_url}` : "",
      sitePrefetch.page_title ? `page_title=${sitePrefetch.page_title}` : "",
      sitePrefetch.page_kind ? `page_kind=${sitePrefetch.page_kind}` : "",
      sitePrefetch.summary ? `summary=${sitePrefetch.summary}` : "",
      sitePrefetch.candidate_urls.length > 0 ? `candidate_urls=${sitePrefetch.candidate_urls.join(", ")}` : "",
      sitePrefetch.top_nav_links.length > 0 ? `top_nav_links=${sitePrefetch.top_nav_links.join(" | ")}` : "",
      sitePrefetch.sections.length > 0 ? `sections=${sitePrefetch.sections.join(" | ")}` : "",
      sitePrefetch.primary_actions.length > 0 ? `primary_actions=${sitePrefetch.primary_actions.join(" | ")}` : "",
      sitePrefetch.forms_summary.length > 0 ? `forms=${sitePrefetch.forms_summary.join(" | ")}` : "",
      sitePrefetch.dialog_or_banner.length > 0 ? `dialog_or_banner=${sitePrefetch.dialog_or_banner.join(" | ")}` : "",
      directRule.active
        ? `direct_navigation_rule=${directRule.mode}; direct_navigation_target=${directRule.target_url}; ${directRule.instruction}`
        : "",
    ]
      .filter(Boolean)
      .join("\n"),
    2400,
  );

  const systemPrompts = [domainSkillPrompt, prefetchPrompt].filter(Boolean).map((content) => ({ role: "system", content }));
  return [...systemPrompts, ...messages] as Array<Record<string, JsonValue>>;
}
