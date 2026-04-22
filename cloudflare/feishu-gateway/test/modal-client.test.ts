import { afterEach, describe, expect, it, vi } from "vitest";
import type { Env, FeishuNormalizedPayload } from "../src/runtime";
import { fetchModalInternal } from "../src/services/modal/client";

function buildNormalizedPayload(): FeishuNormalizedPayload {
  return {
    correlation_id: "corr-1",
    session_key: "session-1",
    lane: "control",
    route_hint: "fast_control",
    site_category: "none",
    site_intent: "general",
    target_url: "",
    target_domain: "",
    task_kind: "text",
    request_class: "session_mutation_heavy",
    content_modalities: ["text"],
    route_family: "modal_control",
    gateway_route_name: "",
    gateway_eligible: false,
    requires_tools: false,
    requires_browser: false,
    requires_media_hydration: false,
    requires_modal_runtime: true,
    modality_profile: "text",
    toolset: [],
    reason_code: "planner_forced_modal",
    event_id: "evt-1",
    event_type: "application.bot.menu_v6",
    chat_id: "",
    chat_type: "group",
    chat_name: "chat",
    user_id: "ou_test",
    user_name: "tester",
    message_id: "",
    message_type: "text",
    text: "",
    attachment_refs: [],
    raw_payload: {},
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("fetchModalInternal", () => {
  it("sends the legacy payload under payload for the modal web handler", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({
          contract_version: "feishu_internal.v1",
          result: {
            status: "ok",
            route_hint: "fast_control",
            execution_mode: "control_complete",
            action: "get_session_state",
          },
          send_plan: {},
          session_patch: {},
          reconcile: {},
          provider_metrics: {},
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await fetchModalInternal<{ status: string }>(
      {
        MODAL_INTERNAL_BASE_URL: "https://isuyee88--hermes-agent-web-handler.modal.run",
        MODAL_INTERNAL_BEARER_TOKEN: "token-1",
      } as Env,
      "/internal/feishu/session-control",
      {
        action: "get_session_state",
      },
      buildNormalizedPayload(),
      [],
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://isuyee88--hermes-agent-web-handler.modal.run");
    const parsed = JSON.parse(String(init.body)) as Record<string, unknown>;
    expect(parsed.__path).toBe("/internal/feishu/session-control");
    expect(parsed.payload).toBeTruthy();
    expect(parsed.payload).toMatchObject({
      action: "get_session_state",
      session_key: "session-1",
    });
  });
});
