import { afterEach, describe, expect, it, vi } from "vitest";
import type { Env, FeishuNormalizedPayload } from "../src/runtime";
import { classifyFeishuSendFailure, sendLoggedFeishuOperation } from "../src/services/feishu/messages";

function buildNormalized(): FeishuNormalizedPayload {
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
    reason_code: "plain_text",
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

describe("classifyFeishuSendFailure", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("classifies bot-not-in-chat failures", () => {
    expect(
      classifyFeishuSendFailure({
        status: 400,
        code: 230001,
        message: "Bot/User can NOT be out of the chat.",
        receiveId: "oc_deadbeef",
        receiveIdType: "chat_id",
      }),
    ).toBe("target_not_in_chat");
  });

  it("classifies cross-app identity failures", () => {
    expect(
      classifyFeishuSendFailure({
        status: 400,
        code: 230002,
        message: "open_id cross app",
        receiveId: "ou_deadbeef",
        receiveIdType: "open_id",
      }),
    ).toBe("cross_app_identity");
  });

  it("classifies invalid chat targets when receive_id_type is chat_id", () => {
    expect(
      classifyFeishuSendFailure({
        status: 400,
        code: 230003,
        message: "bad request",
        receiveId: "ou_deadbeef",
        receiveIdType: "chat_id",
      }),
    ).toBe("invalid_receive_target");
  });

  it("writes analytics for send.operation.error failures", async () => {
    const env = {
      FEISHU_API_BASE: "https://open.feishu.cn",
      FEISHU_APP_ID: "cli_xxx",
      FEISHU_APP_SECRET: "secret",
      FEISHU_GATEWAY_ANALYTICS: {
        writeDataPoint: vi.fn(),
      },
    } as unknown as Env;
    const log = vi.fn();
    const fetchMock = vi
      .fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/tenant_access_token/internal")) {
          return new Response(
            JSON.stringify({
              code: 0,
              tenant_access_token: "tenant-token",
              expire: 7200,
            }),
            { status: 200 },
          );
        }
        return new Response(
          JSON.stringify({
            code: 230429,
            msg: "rate limit exceeded",
          }),
          { status: 429 },
        );
      });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      sendLoggedFeishuOperation(
        env,
        buildNormalized(),
        "reply_text",
        {
          receiveId: "oc_chat_1",
          msgType: "text",
          content: { text: "hello" },
        },
        log,
      ),
    ).rejects.toThrow("send_message_failed:429");

    expect(log).toHaveBeenCalledWith(
      "feishu.send.operation.error",
      expect.objectContaining({
        correlation_id: "corr-1",
        feishu_error_class: "rate_limited",
        status: "error",
      }),
    );
    expect(env.FEISHU_GATEWAY_ANALYTICS.writeDataPoint).toHaveBeenCalledWith(
      expect.objectContaining({
        indexes: ["feishu.send.operation.error"],
      }),
    );
  });
});
