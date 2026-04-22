import registry from "../../../../config/domain-skills.registry.json";

type DomainSkillEntry = {
  skill_name?: string;
  template?: string;
  entry_url?: string;
  browser_strategy?: string;
  description?: string;
  preferred_actions?: string[];
  avoid_actions?: string[];
  common_actions?: string[];
  disabled_actions?: string[];
  notes?: string[];
};

type TemplateEntry = DomainSkillEntry;

type DomainSkillRegistry = {
  templates?: Record<string, TemplateEntry>;
  domains?: Record<string, DomainSkillEntry>;
};

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

export function normalizeDomain(value: unknown): string {
  const lowered = trim(value).toLowerCase().replace(/\.+$/g, "");
  return lowered.startsWith("www.") ? lowered.slice(4) : lowered;
}

export function normalizeUrl(value: unknown): string {
  const raw = trim(value);
  if (!raw) return "";
  const candidate = raw.startsWith("www.") ? `https://${raw}` : raw;
  try {
    const parsed = new URL(candidate);
    if (!/^https?:$/i.test(parsed.protocol)) {
      return "";
    }
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return "";
  }
}

function domainCandidates(domain: string): string[] {
  const normalized = normalizeDomain(domain);
  if (!normalized) {
    return [];
  }
  const parts = normalized.split(".");
  const out: string[] = [];
  for (let index = 0; index < parts.length - 1; index += 1) {
    out.push(parts.slice(index).join("."));
  }
  return out;
}

export function resolveDomainSkill(targetDomain: string, targetUrl = ""): (DomainSkillEntry & { match_domain?: string }) | null {
  const payload = registry as DomainSkillRegistry;
  const domains = payload.domains ?? {};
  const templates = payload.templates ?? {};
  const normalizedDomain =
    normalizeDomain(targetDomain) ||
    (() => {
      try {
        return normalizeDomain(new URL(normalizeUrl(targetUrl)).hostname);
      } catch {
        return "";
      }
    })();
  for (const candidate of domainCandidates(normalizedDomain)) {
    const entry = domains[candidate];
    if (!entry) {
      continue;
    }
    const template = entry.template ? templates[entry.template] ?? {} : {};
    return {
      ...template,
      ...entry,
      match_domain: candidate,
    };
  }
  return null;
}

function joinList(values: unknown, separator = "; "): string {
  if (!Array.isArray(values)) {
    return "";
  }
  return values.map((item) => trim(item)).filter(Boolean).slice(0, 4).join(separator);
}

export function buildDomainSkillPrompt(targetDomain: string, targetUrl = ""): string {
  const resolved = resolveDomainSkill(targetDomain, targetUrl);
  if (!resolved) {
    return "";
  }
  const parts = [
    "Domain skill hint (internal, not user-authored).",
    `site_skill_name=${trim(resolved.skill_name)}`,
    `match_domain=${trim(resolved.match_domain)}`,
    `browser_strategy=${trim(resolved.browser_strategy) || "local_preferred"}`,
    trim(resolved.description) ? `description=${trim(resolved.description)}` : "",
    trim(resolved.entry_url) ? `entry_url=${trim(resolved.entry_url)}` : "",
    joinList(resolved.notes) ? `notes=${joinList(resolved.notes)}` : "",
    joinList(resolved.preferred_actions) ? `preferred_actions=${joinList(resolved.preferred_actions)}` : "",
    joinList(resolved.common_actions) ? `common_actions=${joinList(resolved.common_actions)}` : "",
    joinList(resolved.avoid_actions) ? `avoid_actions=${joinList(resolved.avoid_actions)}` : "",
    joinList(resolved.disabled_actions) ? `disabled_actions=${joinList(resolved.disabled_actions)}` : "",
  ].filter(Boolean);
  return parts.join("\n");
}
