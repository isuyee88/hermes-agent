import type { Env, FeishuNormalizedPayload } from "../runtime";

type AttachmentRef = { resource_type: "image" | "file" | "audio" | "media" };
type SiteCategory = FeishuNormalizedPayload["site_category"];
type SiteIntent = FeishuNormalizedPayload["site_intent"];
type RequestClass = FeishuNormalizedPayload["request_class"];
type RouteHint = FeishuNormalizedPayload["route_hint"];
type RouteFamily = FeishuNormalizedPayload["route_family"];

const CONTENT_INTENT_PATTERNS: Array<[SiteIntent, RegExp[]]> = [
  ["api", [/\bapi\b/, /\bsdk\b/, /\bendpoint\b/, /\bopenapi\b/, /\u63a5\u53e3/, /\u5f00\u653e\u63a5\u53e3/]],
  [
    "docs",
    [
      /\bdocs?\b/,
      /\bdocumentation\b/,
      /\bguide\b/,
      /\bmanual\b/,
      /\breference\b/,
      /\u6587\u6863/,
      /\u6559\u7a0b/,
      /\u6307\u5357/,
      /\u53c2\u8003/,
    ],
  ],
  ["pricing", [/\bpricing\b/, /\bprice\b/, /\bplan(s)?\b/, /\u4ef7\u683c/, /\u5957\u9910/, /\u8ba1\u8d39/]],
  ["help", [/\bhelp\b/, /\bsupport\b/, /\bfaq\b/, /\bkb\b/, /\u5e2e\u52a9/, /\u652f\u6301/, /\u5e38\u89c1\u95ee\u9898/]],
  ["blog", [/\bblog\b/, /\bchangelog\b/, /\bnews\b/, /\u535a\u5ba2/, /\u66f4\u65b0\u65e5\u5fd7/, /\u65b0\u95fb/]],
];

const INTERACTIVE_INTENT_PATTERNS: Array<[SiteIntent, RegExp[]]> = [
  ["login", [/\blogin\b/, /\blog[ -]?in\b/, /\bsign[ -]?in\b/, /\u767b\u5f55/, /\u767b\u5f55\u9875/]],
  ["signup", [/\bregister\b/, /\bsign[ -]?up\b/, /\bcreate account\b/, /\u6ce8\u518c/, /\u521b\u5efa\u8d26\u53f7/]],
  ["navigation", [/\bnavigation\b/, /\bnav\b/, /\bmenu\b/, /\u5bfc\u822a/, /\u5165\u53e3/, /\u9875\u9762\u7ed3\u6784/]],
];

const CONTENT_TARGET_HINTS: Array<[SiteIntent, RegExp[]]> = [
  ["api", [/\bapi\b/, /\bsdk\b/, /\bopenapi\b/, /\/api(\/|$)/, /\/reference(\/|$)/]],
  [
    "docs",
    [
      /\bdocs?\b/,
      /\bdeveloper(s)?\b/,
      /\bdocumentation\b/,
      /\/docs?(\/|$)/,
      /\/guide(s)?(\/|$)/,
      /\/reference(\/|$)/,
    ],
  ],
  ["pricing", [/\bpricing\b/, /\bprice\b/, /\bplan(s)?\b/, /\/pricing(\/|$)/, /\/plans?(\/|$)/, /\/billing(\/|$)/]],
  ["help", [/\bhelp\b/, /\bsupport\b/, /\bfaq\b/, /\bkb\b/, /\/help(\/|$)/, /\/support(\/|$)/, /\/faq(\/|$)/]],
  ["blog", [/\bblog\b/, /\bchangelog\b/, /\bnews\b/, /\/blog(\/|$)/, /\/changelog(\/|$)/, /\/news(\/|$)/]],
];

const HEAVY_BROWSER_PATTERNS: RegExp[] = [
  /\bfill\b/,
  /\bsubmit\b/,
  /\bclick\b/,
  /\bform\b/,
  /\bcheckout\b/,
  /\bapply\b/,
  /\bupload\b/,
  /\bdashboard\b/,
  /\bconsole\b/,
  /\badmin\b/,
  /\blog in to\b/,
  /\bsign in to\b/,
  /\bcaptcha\b/,
  /\bbrowser\b/,
  /\bplaywright\b/,
  /\u586b\u5199/,
  /\u63d0\u4ea4/,
  /\u70b9\u51fb/,
  /\u8868\u5355/,
  /\u63a7\u5236\u53f0/,
  /\u540e\u53f0/,
  /\u9a8c\u8bc1\u7801/,
  /\u4e0a\u4f20/,
  /\/dashboard(\/|$)/,
  /\/console(\/|$)/,
  /\/admin(\/|$)/,
];

const EXPLICIT_BROWSER_PATTERNS: RegExp[] = [
  /\bbrowser\b/,
  /\bplaywright\b/,
  /\bcamofox\b/,
  /\bbrowserbase\b/,
  /\bbrowser use\b/,
  /\bopen\b/,
  /\bvisit\b/,
  /\bnavigate\b/,
  /\bclick\b/,
  /\bscroll\b/,
  /\bscreenshot\b/,
  /\bfill\b/,
  /\bsubmit\b/,
  /\bupload\b/,
  /\bcaptcha\b/,
  /\blog in to\b/,
  /\bsign in to\b/,
  /\bsign up on\b/,
  /\bregister on\b/,
  /\u6d4f\u89c8\u5668/,
  /\u6253\u5f00/,
  /\u8bbf\u95ee/,
  /\u5bfc\u822a\u5230/,
  /\u70b9\u51fb/,
  /\u6eda\u52a8/,
  /\u622a\u56fe/,
  /\u586b\u5199/,
  /\u63d0\u4ea4/,
  /\u4e0a\u4f20/,
  /\u9a8c\u8bc1\u7801/,
  /\u5e2e\u6211\u767b\u5f55/,
  /\u767b\u5f55\u5230/,
  /\u6ce8\u518c\u5230/,
  /\u63a7\u5236\u53f0/,
  /\u540e\u53f0/,
];

const IMAGE_GENERATION_PATTERNS: RegExp[] = [
  /\bgenerate image\b/,
  /\bcreate image\b/,
  /\bmake image\b/,
  /\billustration\b/,
  /\bposter\b/,
  /\blogo\b/,
  /\brender\b/,
  /\bdraw\b/,
  /\bimage generation\b/,
  /\u751f\u6210\u56fe\u7247/,
  /\u751f\u6210\u4e00\u5f20\u56fe/,
  /\u753b\u4e00\u5f20/,
  /\u505a\u4e00\u5f20\u56fe/,
  /\u6d77\u62a5/,
  /\u914d\u56fe/,
  /\u63d2\u753b/,
];

const CODING_PATTERNS: RegExp[] = [
  /```/,
  /\bcode\b/,
  /\bcoding\b/,
  /\bdebug\b/,
  /\bfix\b/,
  /\bbug\b/,
  /\brefactor\b/,
  /\bimplement\b/,
  /\btypescript\b/,
  /\bjavascript\b/,
  /\bpython\b/,
  /\bjava\b/,
  /\bc\+\+\b/,
  /\bsql\b/,
  /\bregex\b/,
  /\bapi\b/,
  /\bfunction\b/,
  /\bclass\b/,
  /\bscript\b/,
  /\u62a5\u9519/,
  /\u4ee3\u7801/,
  /\u51fd\u6570/,
  /\u811a\u672c/,
  /\u8c03\u8bd5/,
  /\u4fee\u590d/,
  /\u5b9e\u73b0/,
  /\u91cd\u6784/,
];

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function parseBoolean(value: unknown, fallback = false): boolean {
  const normalized = trim(value).toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = trim(value);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function uniqueLowercaseStrings(values: string[]): string[] {
  return uniqueStrings(values.map((value) => trim(value).toLowerCase()).filter(Boolean));
}

function anyPatternMatch(text: string, patterns: RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(text));
}

function matchIntent(text: string, entries: Array<[SiteIntent, RegExp[]]>): SiteIntent | "" {
  return entries.find(([, patterns]) => anyPatternMatch(text, patterns))?.[0] ?? "";
}

function extractUrlCandidates(message: string): string[] {
  const text = trim(message);
  if (!text) {
    return [];
  }
  const matches = text.match(/(?:https?:\/\/|www\.)[^\s<>"'`)]+/gi) ?? [];
  return uniqueStrings(
    matches.map((match) => {
      const normalized = trim(match).replace(/[),.;!?]+$/g, "");
      if (!normalized) {
        return "";
      }
      return normalized.startsWith("http://") || normalized.startsWith("https://")
        ? normalized
        : `https://${normalized}`;
    }),
  );
}

function extractDomainCandidates(message: string): string[] {
  const text = trim(message);
  if (!text) {
    return [];
  }
  const matches = text.match(
    /\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:\/[^\s<>"'`)]*)?/gi,
  ) ?? [];
  return uniqueStrings(
    matches
      .map((match) => trim(match).replace(/[),.;!?]+$/g, ""))
      .map((match) => (match.startsWith("http://") || match.startsWith("https://") ? match : `https://${match}`)),
  );
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

export function extractTargetUrlAndDomain(message: string): { targetUrl: string; targetDomain: string } {
  const candidate = [...extractUrlCandidates(message), ...extractDomainCandidates(message)]
    .map((value) => normalizeTargetUrl(value))
    .find(Boolean);
  if (!candidate) {
    return { targetUrl: "", targetDomain: "" };
  }
  try {
    const url = new URL(candidate);
    return { targetUrl: url.toString(), targetDomain: url.hostname.toLowerCase() };
  } catch {
    return { targetUrl: "", targetDomain: "" };
  }
}

function messageLooksLikeSiteContentIntent(message: string): SiteIntent | "" {
  const text = trim(message).toLowerCase();
  return text ? matchIntent(text, CONTENT_INTENT_PATTERNS) : "";
}

function inferSiteContentIntentFromTarget(targetUrl: string, targetDomain: string): SiteIntent | "" {
  const combined = `${trim(targetDomain).toLowerCase()} ${trim(targetUrl).toLowerCase()}`.trim();
  return combined ? matchIntent(combined, CONTENT_TARGET_HINTS) : "";
}

function messageLooksLikeInteractiveIntent(message: string): SiteIntent | "" {
  const text = trim(message).toLowerCase();
  return text ? matchIntent(text, INTERACTIVE_INTENT_PATTERNS) : "";
}

function inferInteractiveIntentFromTarget(targetUrl: string, targetDomain: string): SiteIntent | "" {
  const combined = `${trim(targetDomain).toLowerCase()} ${trim(targetUrl).toLowerCase()}`;
  if (!combined.trim()) {
    return "";
  }
  if ([/\/login(\/|$)/, /\/signin(\/|$)/, /\bauth\b/, /\bsso\b/].some((pattern) => pattern.test(combined))) {
    return "login";
  }
  if ([/\/signup(\/|$)/, /\/register(\/|$)/, /create-account/].some((pattern) => pattern.test(combined))) {
    return "signup";
  }
  if ([/\/dashboard(\/|$)/, /\/console(\/|$)/, /\/admin(\/|$)/, /\/settings(\/|$)/].some((pattern) => pattern.test(combined))) {
    return "navigation";
  }
  return "";
}

function messageLooksLikeHeavyBrowserTask(message: string, targetUrl = "", targetDomain = ""): boolean {
  const combined = `${trim(message).toLowerCase()} ${trim(targetUrl).toLowerCase()} ${trim(targetDomain).toLowerCase()}`.trim();
  return combined ? anyPatternMatch(combined, HEAVY_BROWSER_PATTERNS) : false;
}

function inferSiteCategory(
  message: string,
  targetUrl: string,
  attachmentRefs: AttachmentRef[],
): { siteCategory: SiteCategory; siteIntent: SiteIntent } {
  if (!targetUrl || attachmentRefs.length > 0) {
    return { siteCategory: "none", siteIntent: "general" };
  }

  const { targetDomain } = extractTargetUrlAndDomain(targetUrl);
  const contentIntent = messageLooksLikeSiteContentIntent(message) || inferSiteContentIntentFromTarget(targetUrl, targetDomain);
  if (contentIntent) {
    return { siteCategory: "site_content", siteIntent: contentIntent };
  }

  const interactiveIntent = messageLooksLikeInteractiveIntent(message) || inferInteractiveIntentFromTarget(targetUrl, targetDomain);
  if (interactiveIntent) {
    return {
      siteCategory: messageLooksLikeHeavyBrowserTask(message, targetUrl, targetDomain)
        ? "site_interactive_heavy"
        : "site_interactive_light",
      siteIntent: interactiveIntent,
    };
  }

  if (messageLooksLikeHeavyBrowserTask(message, targetUrl, targetDomain)) {
    return { siteCategory: "site_interactive_heavy", siteIntent: "general" };
  }

  return { siteCategory: "none", siteIntent: "general" };
}

export function messageExplicitlyRequestsBrowserTools(message: string): boolean {
  const text = trim(message).toLowerCase();
  if (!text) {
    return false;
  }

  if (anyPatternMatch(text, EXPLICIT_BROWSER_PATTERNS)) {
    return true;
  }

  const { targetUrl } = extractTargetUrlAndDomain(text);
  if (!targetUrl) {
    return false;
  }

  return [
    /\u770b\u4e00\u4e0b\u9996\u9875\u5e76\u622a\u56fe/,
    /\u6253\u5f00\u8fd9\u4e2a\u7f51\u7ad9/,
    /\u8bbf\u95ee\u8fd9\u4e2a\u7f51\u7ad9/,
    /\bopen this url\b/,
    /\bopen this site\b/,
    /\bgo to this site\b/,
  ].some((pattern) => pattern.test(text));
}

function messageLooksLikeCodingRequest(message: string): boolean {
  const text = trim(message).toLowerCase();
  return text ? anyPatternMatch(text, CODING_PATTERNS) : false;
}

function messageLooksLikeImageGenerationRequest(message: string): boolean {
  const text = trim(message).toLowerCase();
  return text ? anyPatternMatch(text, IMAGE_GENERATION_PATTERNS) : false;
}

function inferContentModalities(
  messageType: FeishuNormalizedPayload["message_type"],
  text: string,
  attachmentRefs: AttachmentRef[],
): string[] {
  const modalities: string[] = [];
  if (trim(text) || messageType === "text" || messageType === "command") {
    modalities.push("text");
  }
  if (messageType === "photo") {
    modalities.push("image");
  } else if (messageType === "audio") {
    modalities.push("audio");
  } else if (messageType === "video") {
    modalities.push("video");
  } else if (messageType === "document") {
    modalities.push("file");
  }
  for (const attachment of attachmentRefs) {
    modalities.push(attachment.resource_type);
    if (attachment.resource_type === "media") {
      modalities.push("video");
    }
  }
  return uniqueLowercaseStrings(modalities);
}

function inferToolset(
  message: string,
  lane: FeishuNormalizedPayload["lane"],
  siteCategory?: SiteCategory,
): string[] {
  if (lane !== "agent") {
    return [];
  }
  const toolset: string[] = [];
  const text = trim(message);
  if (!text) {
    return toolset;
  }
  if (siteCategory !== "site_content" && messageExplicitlyRequestsBrowserTools(text)) {
    toolset.push("browser");
  }
  if (text.includes("@")) {
    toolset.push("mention_dispatch");
  }
  return uniqueLowercaseStrings(toolset);
}

function inferRouteHintFromRequestClass(lane: FeishuNormalizedPayload["lane"], requestClass: RequestClass): RouteHint {
  if (lane === "control" || requestClass === "session_mutation_heavy") {
    return "fast_control";
  }
  if (lane !== "agent") {
    return "modal_heavy_exec";
  }
  if (requestClass === "tool_browser") {
    return "cf_browser_first";
  }
  return "modal_heavy_exec";
}

export function isTextGatewayRequestClass(value: string): boolean {
  return value === "text_plain" || value === "text_coding";
}

function buildRouteDecisionReasonFromRequestClass(requestClass: RequestClass, gatewayEligible: boolean): string {
  if (gatewayEligible && requestClass === "text_coding") {
    return "text_coding_gateway_candidate";
  }
  if (gatewayEligible) {
    return "plain_text_without_attachments_or_browser";
  }
  if (requestClass === "session_mutation_heavy") {
    return "planner_forced_modal";
  }
  if (requestClass === "tool_browser") {
    return "browser_required";
  }
  if (requestClass === "tool_non_browser") {
    return "tool_required";
  }
  if (requestClass === "image_understanding" || requestClass === "media_hydration" || requestClass === "file_or_attachment") {
    return "media_hydration_required";
  }
  return "unsupported_task_shape";
}

export function classifyNormalizedRequest(
  env: Env,
  input: Pick<
    FeishuNormalizedPayload,
    "lane" | "task_kind" | "message_type" | "text" | "attachment_refs" | "target_url" | "target_domain"
  >,
): Pick<
  FeishuNormalizedPayload,
  | "route_hint"
  | "site_category"
  | "site_intent"
  | "request_class"
  | "content_modalities"
  | "route_family"
  | "gateway_route_name"
  | "gateway_eligible"
  | "requires_tools"
  | "requires_browser"
  | "requires_media_hydration"
  | "requires_modal_runtime"
  | "modality_profile"
  | "toolset"
  | "reason_code"
> {
  const classifierEnabled = parseBoolean(env.HERMES_CF_REQUEST_CLASSIFIER_ENABLED, true);
  const imageGatewayEnabled = parseBoolean(env.HERMES_CF_IMAGE_GATEWAY_ENABLED, false);
  const attachments = input.attachment_refs as AttachmentRef[];
  const contentModalities = inferContentModalities(input.message_type, input.text, attachments);
  const { siteCategory, siteIntent } = inferSiteCategory(input.text, input.target_url, attachments);
  const toolset = inferToolset(input.text, input.lane, siteCategory);
  const hasAttachments = attachments.length > 0;
  const hasImageAttachment = attachments.some((item) => item.resource_type === "image");
  const hasAudioAttachment = attachments.some((item) => item.resource_type === "audio");
  const hasMediaAttachment = attachments.some((item) => item.resource_type === "media");
  const hasFileAttachment = attachments.some((item) => item.resource_type === "file");
  const explicitBrowserRequest = messageExplicitlyRequestsBrowserTools(input.text);
  const heavyBrowserRequest = messageLooksLikeHeavyBrowserTask(input.text, input.target_url, input.target_domain);
  const requiresBrowser = classifierEnabled
    ? siteCategory === "site_interactive_heavy"
      ? true
      : siteCategory === "site_interactive_light"
        ? explicitBrowserRequest || heavyBrowserRequest || toolset.includes("browser")
        : siteCategory === "site_content"
          ? false
          : explicitBrowserRequest || heavyBrowserRequest || toolset.includes("browser")
    : false;
  const requiresTools = classifierEnabled ? toolset.length > 0 : false;
  const requiresMediaHydration = classifierEnabled ? hasAttachments : attachments.length > 0;

  let requestClass: RequestClass;
  if (input.lane === "control" || input.task_kind === "command" || input.message_type === "command") {
    requestClass = "session_mutation_heavy";
  } else if (siteCategory === "site_content") {
    requestClass = messageLooksLikeCodingRequest(input.text) ? "text_coding" : "text_plain";
  } else if (siteCategory === "site_interactive_light" && !requiresBrowser) {
    requestClass = "text_plain";
  } else if (requiresBrowser) {
    requestClass = "tool_browser";
  } else if (hasImageAttachment) {
    requestClass = "image_understanding";
  } else if (hasAudioAttachment || hasMediaAttachment) {
    requestClass = "media_hydration";
  } else if (hasFileAttachment) {
    requestClass = "file_or_attachment";
  } else if (messageLooksLikeImageGenerationRequest(input.text)) {
    requestClass = "image_generation";
  } else if (requiresTools) {
    requestClass = "tool_non_browser";
  } else if (messageLooksLikeCodingRequest(input.text)) {
    requestClass = "text_coding";
  } else {
    requestClass = "text_plain";
  }

  const routeFamily: RouteFamily =
    requestClass === "session_mutation_heavy"
      ? "modal_control"
      : requestClass === "tool_browser" || requestClass === "tool_non_browser"
        ? "modal_tools"
        : requestClass === "image_understanding" && imageGatewayEnabled && !requiresMediaHydration
          ? "gateway_image"
          : isTextGatewayRequestClass(requestClass)
            ? "gateway_text"
            : "modal_runtime";
  const gatewayRouteName =
    routeFamily === "gateway_text"
      ? requestClass === "text_coding"
        ? trim(env.HERMES_CF_TEXT_CODING_ROUTE_NAME) || "affiliate-coding"
        : trim(env.HERMES_CF_TEXT_PLAIN_ROUTE_NAME) || "affiliate-general"
      : routeFamily === "gateway_image"
        ? trim(env.HERMES_CF_IMAGE_ROUTE_NAME) || "image-understanding"
        : "";
  const gatewayEligible =
    input.lane === "agent" &&
    ((routeFamily === "gateway_text" && isTextGatewayRequestClass(requestClass)) ||
      (routeFamily === "gateway_image" && requestClass === "image_understanding"));
  const requiresModalRuntime =
    input.lane !== "agent" ||
    routeFamily === "modal_control" ||
    routeFamily === "modal_tools" ||
    routeFamily === "modal_runtime" ||
    requiresMediaHydration ||
    requiresTools;

  return {
    route_hint: inferRouteHintFromRequestClass(input.lane, requestClass),
    site_category: siteCategory,
    site_intent: siteIntent,
    request_class: requestClass,
    content_modalities: contentModalities,
    route_family: routeFamily,
    gateway_route_name: gatewayRouteName,
    gateway_eligible: gatewayEligible && !requiresModalRuntime,
    requires_tools: requiresTools,
    requires_browser: requiresBrowser,
    requires_media_hydration: requiresMediaHydration,
    requires_modal_runtime: requiresModalRuntime,
    modality_profile: contentModalities.length > 0 ? contentModalities.join("+") : "text",
    toolset,
    reason_code: buildRouteDecisionReasonFromRequestClass(requestClass, gatewayEligible),
  };
}
