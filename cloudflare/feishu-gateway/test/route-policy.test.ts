import { describe, expect, it } from "vitest";
import {
  buildRoutingLogFields,
  classifyCfAiExecFallbackReason,
  normalizeGatewayErrorClass,
} from "../src/gateway/route-policy";
import type { FeishuNormalizedPayload } from "../src/runtime";

function buildPayload(): FeishuNormalizedPayload {
  return {
    correlation_id: "corr-1",
    session_key: "session-1",
    lane: "agent",
    route_hint: "modal_heavy_exec",
    site_category: "none",
    site_intent: "general",
    target_url: "",
    target_domain: "",
    task_kind: "general",
    request_class: "text_plain",
    content_modalities: ["text"],
    route_family: "gateway_text",
    gateway_route_name: "text-general",
    gateway_eligible: true,
    requires_tools: false,
    requires_browser: false,
    requires_media_hydration: false,
    requires_modal_runtime: false,
    modality_profile: "text",
    toolset: [],
    reason_code: "plain_text_without_attachments_or_browser",
    event_id: "evt-1",
    event_type: "im.message.receive_v1",
    chat_id: "chat-1",
    chat_type: "dm",
    chat_name: "chat",
    user_id: "user-1",
    user_name: "user",
    message_id: "msg-1",
    message_type: "text",
    text: "hello",
    attachment_refs: [],
    raw_payload: {},
  };
}

describe("route policy gateway error normalization", () => {
  it("normalizes legacy Cloudflare fallback names to the shared taxonomy", () => {
    expect(normalizeGatewayErrorClass("auth_or_secret_missing")).toBe("provider_permission_denied");
    expect(normalizeGatewayErrorClass("provider_model_invalid")).toBe("provider_model_not_found");
    expect(normalizeGatewayErrorClass("rate_exhausted")).toBe("rate_limited");
    expect(normalizeGatewayErrorClass("cf_execution_error")).toBe("upstream_5xx");
  });

  it("classifies fallback reasons directly into the normalized runtime taxonomy", () => {
    expect(
      classifyCfAiExecFallbackReason(
        new Error("cf_ai_exec_failed:429:{\"error\":{\"message\":\"rate limit\"}}"),
        { gateway_eligible: true, request_class: "text_plain" },
        (value) => value === "text_plain",
      ),
    ).toBe("rate_limited");
    expect(
      classifyCfAiExecFallbackReason(
        new Error("cf_ai_exec_failed:404:{\"error\":{\"message\":\"not found\"}}"),
        { gateway_eligible: true, request_class: "text_plain" },
        (value) => value === "text_plain",
      ),
    ).toBe("provider_model_not_found");
  });

  it("emits normalized gateway_error_class in routing log fields", () => {
    const fields = buildRoutingLogFields(buildPayload(), {
      gateway_error_class: "provider_auth_error",
    });

    expect(fields.gateway_error_class).toBe("provider_permission_denied");
  });
});
