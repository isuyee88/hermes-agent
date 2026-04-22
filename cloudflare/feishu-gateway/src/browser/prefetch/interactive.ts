import { launch } from "@cloudflare/playwright";
import {
  SITE_PREFETCH_MAX_ARIA_CHARS,
  buildSitePrefetchErrorManifest,
  buildSitePrefetchManifestBase,
  compactList,
  compactText,
  inferPageKindFromSignals,
} from "./shared";
import type { Env, FeishuNormalizedPayload, SitePrefetchManifest } from "../../runtime";

const SITE_PREFETCH_INTERACTION_TTL_MS = 30 * 60 * 1000;

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

export async function buildInteractionLightPrefetch(
  env: Env,
  normalized: FeishuNormalizedPayload,
): Promise<SitePrefetchManifest> {
  if (!env.BROWSER) {
    return buildSitePrefetchErrorManifest(normalized, "playwright_preflight", "browser_binding_missing", "Cloudflare browser binding is not configured");
  }
  const startedAt = Date.now();
  const browser = await launch(env.BROWSER as any, { keep_alive: 60_000 });
  try {
    const page = await browser.newPage();
    await page.goto(normalized.target_url, { waitUntil: "domcontentloaded", timeout: 25_000 });
    const finalUrl = page.url();
    const pageTitle = trim(await page.title());
    const a11ySnapshot = compactText(await page.locator("body").ariaSnapshot({ timeout: 5_000 }).catch(() => ""), SITE_PREFETCH_MAX_ARIA_CHARS);
    const pageData = await page.evaluate(() => {
      const textOf = (value: string | null | undefined) => String(value ?? "").replace(/\s+/g, " ").trim();
      const asAbsolute = (href: string | null | undefined) => {
        try {
          return href ? new URL(href, window.location.href).toString() : "";
        } catch {
          return "";
        }
      };
      const topNavLinks = Array.from(document.querySelectorAll("a[href]"))
        .map((node) => ({
          text: textOf(node.textContent),
          href: asAbsolute(node.getAttribute("href")),
        }))
        .filter((item) => item.text && item.href)
        .slice(0, 16);
      const primaryActions = Array.from(document.querySelectorAll("a[href], button, input[type='submit'], [role='button']"))
        .map((node) =>
          textOf(
            node.textContent ||
              (node instanceof HTMLInputElement ? node.value : "") ||
              node.getAttribute("aria-label") ||
              node.getAttribute("title"),
          ),
        )
        .filter(Boolean)
        .slice(0, 12);
      const formsSummary = Array.from(document.forms)
        .map((form, index) => {
          const inputs = Array.from(form.querySelectorAll("input, textarea, select"))
            .map((field) =>
              textOf(
                field.getAttribute("name") ||
                  field.getAttribute("id") ||
                  field.getAttribute("placeholder") ||
                  field.getAttribute("aria-label"),
              ),
            )
            .filter(Boolean)
            .slice(0, 8);
          return textOf(form.getAttribute("id") || form.getAttribute("name")) || `form_${index + 1}: ${inputs.join(", ")}`;
        })
        .filter(Boolean)
        .slice(0, 6);
      const inputFields = Array.from(document.querySelectorAll("input, textarea, select"))
        .map((field) => {
          const element = field as HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement;
          return textOf(
            element.name ||
              element.id ||
              element.getAttribute("placeholder") ||
              element.getAttribute("aria-label") ||
              element.getAttribute("type"),
          );
        })
        .filter(Boolean)
        .slice(0, 16);
      const dialogOrBanner = Array.from(document.querySelectorAll("[role='dialog'], dialog, [role='alert'], .modal, .dialog, .banner"))
        .map((node) => textOf(node.textContent))
        .filter(Boolean)
        .slice(0, 3);
      return { topNavLinks, primaryActions, formsSummary, inputFields, dialogOrBanner };
    });
    const candidateUrls = compactList(
      [
        finalUrl,
        ...pageData.topNavLinks
          .map((item) => item.href)
          .filter((href) => /login|sign[-_]?in|register|sign[-_]?up|docs|documentation|pricing|help/i.test(href)),
      ],
      16,
      200,
    );
    const pageKind = inferPageKindFromSignals(finalUrl, pageTitle, pageData.formsSummary, pageData.primaryActions, a11ySnapshot);
    return {
      ...buildSitePrefetchManifestBase(normalized, "playwright_preflight", SITE_PREFETCH_INTERACTION_TTL_MS),
      target_url: finalUrl || normalized.target_url,
      final_url: finalUrl || normalized.target_url,
      page_title: pageTitle,
      candidate_urls: candidateUrls,
      top_nav_links: compactList(
        pageData.topNavLinks.map((item) => (item.text ? `${item.text} -> ${item.href}` : item.href)),
        12,
        180,
      ),
      summary: compactText(
        [
          pageTitle ? `title=${pageTitle}` : "",
          pageData.primaryActions.length > 0 ? `actions=${pageData.primaryActions.join("; ")}` : "",
          pageData.formsSummary.length > 0 ? `forms=${pageData.formsSummary.join("; ")}` : "",
        ]
          .filter(Boolean)
          .join(" | "),
      ),
      sections: compactList(pageData.topNavLinks.map((item) => item.text), 10, 120),
      page_kind: pageKind,
      primary_actions: compactList(pageData.primaryActions, 12, 120),
      forms_summary: compactList(pageData.formsSummary, 8, 140),
      input_fields: compactList(pageData.inputFields, 16, 120),
      dialog_or_banner: compactList(pageData.dialogOrBanner, 4, 180),
      auth_required_guess: pageKind === "login" || pageKind === "signup",
      a11y_snapshot_summary: a11ySnapshot,
      confidence: candidateUrls.length > 0 || pageData.primaryActions.length > 0 ? 0.8 : 0.62,
      source_evidence: ["playwright_preflight", "aria_snapshot"],
      status: "completed",
      browser_ms_used: Date.now() - startedAt,
    };
  } finally {
    await browser.close();
  }
}
