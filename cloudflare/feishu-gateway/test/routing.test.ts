import { describe, expect, it } from "vitest";
import { buildDirectConversationMessages } from "../src/gateway/conversation-messages";
import {
  classifyNormalizedRequest,
  messageExplicitlyRequestsBrowserTools,
} from "../src/gateway/routing";
import type { Env, FeishuNormalizedPayload, SitePrefetchManifest } from "../src/runtime";

const env = {
  HERMES_CF_REQUEST_CLASSIFIER_ENABLED: "true",
  HERMES_CF_IMAGE_GATEWAY_ENABLED: "false",
  HERMES_CF_TEXT_PLAIN_ROUTE_NAME: "affiliate-general",
  HERMES_CF_TEXT_CODING_ROUTE_NAME: "affiliate-coding",
  HERMES_CF_IMAGE_ROUTE_NAME: "image-understanding",
} as unknown as Env;

function makeInput(
  overrides: Partial<
    Pick<FeishuNormalizedPayload, "lane" | "task_kind" | "message_type" | "text" | "attachment_refs" | "target_url" | "target_domain">
  > = {},
) {
  return {
    lane: "agent",
    task_kind: "text",
    message_type: "text",
    text: "",
    attachment_refs: [],
    target_url: "",
    target_domain: "",
    ...overrides,
  } satisfies Pick<
    FeishuNormalizedPayload,
    "lane" | "task_kind" | "message_type" | "text" | "attachment_refs" | "target_url" | "target_domain"
  >;
}

describe("gateway routing classification", () => {
  it("keeps docs URLs on the content path instead of browser tools", () => {
    const result = classifyNormalizedRequest(
      env,
      makeInput({
        text: "https://developers.cloudflare.com/workers/",
        target_url: "https://developers.cloudflare.com/workers/",
        target_domain: "developers.cloudflare.com",
      }),
    );

    expect(result.site_category).toBe("site_content");
    expect(result.site_intent).toBe("docs");
    expect(result.request_class).toBe("text_plain");
    expect(result.requires_browser).toBe(false);
    expect(result.route_family).toBe("gateway_text");
  });

  it("keeps light interactive login discovery off the heavy browser path without explicit browser actions", () => {
    const result = classifyNormalizedRequest(
      env,
      makeInput({
        text: "帮我看看登录入口在哪里",
        target_url: "https://github.com/login",
        target_domain: "github.com",
      }),
    );

    expect(result.site_category).toBe("site_interactive_light");
    expect(result.request_class).toBe("text_plain");
    expect(result.requires_browser).toBe(false);
  });

  it("promotes explicit Chinese browser requests to tool_browser", () => {
    const result = classifyNormalizedRequest(
      env,
      makeInput({
        text: "请打开 https://example.com 并截图给我",
        target_url: "https://example.com/",
        target_domain: "example.com",
      }),
    );

    expect(messageExplicitlyRequestsBrowserTools("请打开 https://example.com 并截图给我")).toBe(true);
    expect(result.request_class).toBe("tool_browser");
    expect(result.requires_browser).toBe(true);
    expect(result.route_hint).toBe("cf_browser_first");
  });

  it("treats dashboard and console flows as heavy browser work", () => {
    const result = classifyNormalizedRequest(
      env,
      makeInput({
        text: "请登录后台控制台并上传报表",
        target_url: "https://dashboard.stripe.com/login",
        target_domain: "dashboard.stripe.com",
      }),
    );

    expect(result.site_category).toBe("site_interactive_heavy");
    expect(result.request_class).toBe("tool_browser");
    expect(result.requires_browser).toBe(true);
  });
});

describe("domain skill prompt injection", () => {
  it("prepends domain skill guidance for known high-frequency domains", () => {
    const sitePrefetch = {
      key: "k1",
      category: "site_content",
      mode: "browser_markdown",
      domain: "developers.cloudflare.com",
      intent: "docs",
      target_url: "https://developers.cloudflare.com/workers/",
      final_url: "https://developers.cloudflare.com/workers/",
      page_title: "Workers",
      page_kind: "docs",
      status: "completed",
      confidence: 0.92,
      candidate_urls: ["https://developers.cloudflare.com/workers/"],
      top_nav_links: [],
      sections: [],
      primary_actions: [],
      forms_summary: [],
      dialog_or_banner: [],
      summary: "Workers docs landing page",
      fetched_at: Date.now(),
      expires_at: Date.now() + 60_000,
    } satisfies SitePrefetchManifest;

    const messages = buildDirectConversationMessages([], "总结一下 Workers 文档", sitePrefetch, {
      trim: (value) => String(value ?? "").trim(),
      compactSessionHistory: (entries) => entries,
    });

    expect(messages[0]).toMatchObject({
      role: "system",
    });
    expect(String(messages[0]?.content || "")).toContain("site_skill_name=site.cloudflare-developers-docs");
    expect(String(messages[1]?.content || "")).toContain("direct_navigation_rule=content_direct");
  });
});
