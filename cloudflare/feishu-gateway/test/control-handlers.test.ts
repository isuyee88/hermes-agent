import { describe, expect, it, vi } from "vitest";
import type { Env, FeishuNormalizedPayload, JsonValue, ModalInternalResponse } from "../src/runtime";
import { handleAgentCommand, handleCardAction } from "../src/gateway/control-handlers";

function buildNormalizedPayload(
  rawPayload: Record<string, JsonValue> = {},
  overrides: Partial<FeishuNormalizedPayload> = {},
): FeishuNormalizedPayload {
  return {
    correlation_id: "corr-1",
    session_key: "agent:main:feishu:p2p:oc_chat:ou_user",
    lane: "control",
    route_hint: "fast_control",
    site_category: "none",
    site_intent: "general",
    target_url: "",
    target_domain: "",
    task_kind: "command",
    request_class: "session_mutation_heavy",
    content_modalities: ["text"],
    route_family: "modal_control",
    gateway_route_name: "fast-control",
    gateway_eligible: false,
    requires_tools: false,
    requires_browser: false,
    requires_media_hydration: false,
    requires_modal_runtime: true,
    modality_profile: "text",
    toolset: [],
    reason_code: "control_event",
    event_id: "evt-1",
    event_type: "card.action.trigger",
    chat_id: "oc_chat",
    chat_type: "p2p",
    chat_name: "Hermes DM",
    user_id: "ou_user",
    user_name: "tester",
    message_id: "om_msg",
    message_type: "text",
    text: "/provider",
    attachment_refs: [],
    raw_payload: rawPayload,
    ...overrides,
  };
}

function buildDeps(responseByPath: Record<string, ModalInternalResponse | Error>) {
  const callModalWithReconciles = vi.fn(
    async (_env: Env, path: string): Promise<ModalInternalResponse> => {
      const response = responseByPath[path];
      if (response instanceof Error) {
        throw response;
      }
      if (!response) {
        throw new Error(`unexpected_path:${path}`);
      }
      return response;
    },
  );
  return {
    trim: (value: unknown) => String(value ?? "").trim(),
    readEvent: (payload: Record<string, JsonValue>) =>
      (payload.event as Record<string, JsonValue> | undefined) ?? {},
    readRecord: (parent: Record<string, JsonValue>, key: string) =>
      (parent[key] as Record<string, JsonValue> | undefined) ?? {},
    readString: (parent: Record<string, JsonValue>, key: string) => String(parent[key] ?? ""),
    buildTextSendPlan: (content: string) => [{ type: "text", text: content }],
    buildInteractiveCardSendPlan: (card: Record<string, JsonValue>, receiveId: string, receiveIdType: string) => [
      { type: "interactive", card, receiveId, receiveIdType },
    ],
    buildDeleteMessageSendPlan: (messageId: string) => [{ type: "delete", messageId }],
    callModalWithReconciles,
  };
}

describe("control-handlers fallback", () => {
  it("routes slash command dispatch directly through agent-exec", async () => {
    const deps = buildDeps({
      "/internal/feishu/agent-exec": {
        status: "ok",
        send_plan: [{ type: "text", text: "provider ok" }],
      },
    });

    const result = await handleAgentCommand({} as Env, buildNormalizedPayload(), deps);

    expect(result.status).toBe("ok");
    expect(result.send_plan).toEqual([{ type: "text", text: "provider ok" }]);
    expect(deps.callModalWithReconciles).toHaveBeenCalledTimes(1);
    expect(deps.callModalWithReconciles).toHaveBeenCalledWith(
      {} as Env,
      "/internal/feishu/agent-exec",
      expect.any(Object),
      expect.objectContaining({
        text: "/provider",
        message_type: "command",
        internal: true,
        fallback_reason: "session_control_dispatch_failed",
      }),
    );
  });

  it("falls back to agent-exec for card command actions when session-control returns an error payload", async () => {
    const rawPayload = {
      event: {
        action: {
          value: {
            hermes_action: "command_run",
            command_text: "/status",
          },
        },
      },
    } as Record<string, JsonValue>;
    const deps = buildDeps({
      "/internal/feishu/agent-exec": {
        status: "ok",
        send_plan: [{ type: "text", text: "status ok" }],
      },
    });

    const result = await handleCardAction(
      {} as Env,
      buildNormalizedPayload(rawPayload, { raw_payload: rawPayload, text: "" }),
      deps,
    );

    expect(result.status).toBe("ok");
    expect(result.send_plan).toEqual([{ type: "text", text: "status ok" }]);
    expect(deps.callModalWithReconciles).toHaveBeenCalledTimes(1);
    expect(deps.callModalWithReconciles).toHaveBeenCalledWith(
      {} as Env,
      "/internal/feishu/agent-exec",
      expect.any(Object),
      expect.objectContaining({
        text: "/status",
        message_type: "command",
        raw_message: rawPayload,
      }),
    );
  });

  it("routes skill combo card actions directly through agent-exec text payloads", async () => {
    const rawPayload = {
      event: {
        action: {
          value: {
            hermes_action: "skill_combo_apply",
            combo_id: "cto_ship",
            combo_label: "CTO 交付",
            skills: ["ship", "gov"],
            suggested_personality: "cto",
          },
        },
      },
    } as Record<string, JsonValue>;
    const deps = buildDeps({
      "/internal/feishu/agent-exec": {
        status: "ok",
        send_plan: [{ type: "text", text: "combo ok" }],
      },
    });

    const result = await handleCardAction(
      {} as Env,
      buildNormalizedPayload(rawPayload, { raw_payload: rawPayload, text: "" }),
      deps,
    );

    expect(result.status).toBe("ok");
    expect(result.send_plan).toEqual([{ type: "text", text: "combo ok" }]);
    expect(deps.callModalWithReconciles).toHaveBeenCalledTimes(1);
    expect(deps.callModalWithReconciles).toHaveBeenCalledWith(
      {} as Env,
      "/internal/feishu/agent-exec",
      expect.any(Object),
      expect.objectContaining({
        message_type: "text",
        internal: true,
        fallback_reason: "session_control_skill_combo_failed",
        route_decision_reason: "worker_control_fallback_agent_exec_text",
      }),
    );
    const payload = deps.callModalWithReconciles.mock.calls[0]?.[3] as Record<string, JsonValue>;
    expect(String(payload.text ?? "")).toContain("CTO 交付");
    expect(String(payload.text ?? "")).toContain("ship, gov");
    expect(String(payload.text ?? "")).toContain("/personality cto");
  });
});
