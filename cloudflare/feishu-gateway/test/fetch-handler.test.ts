import { describe, expect, it } from "vitest";
import type { JsonValue } from "../src/runtime";
import { buildFeishuAcceptedBody, extractFeishuControlLogFields, normalizeQueueBody } from "../src/entry/fetch-handler";

describe("fetch-handler control helpers", () => {
  it("returns toast ack for regular card actions", () => {
    const payload = {
      event: {
        action: {
          value: {
            hermes_action: "command_run",
            command_text: "/status",
          },
        },
      },
    } satisfies Record<string, JsonValue>;

    expect(buildFeishuAcceptedBody({ event_type: "card.action.trigger" }, payload)).toEqual({
      toast: {
        type: "info",
        content: "\u5df2\u6536\u5230\uff0c\u6b63\u5728\u5904\u7406",
      },
    });
  });

  it("returns close toast for registry close actions", () => {
    const payload = {
      event: {
        action: {
          value: {
            hermes_action: "registry_close_card",
          },
        },
      },
    } satisfies Record<string, JsonValue>;

    expect(buildFeishuAcceptedBody({ event_type: "card.action.trigger" }, payload)).toEqual({
      toast: {
        type: "info",
        content: "\u5df2\u5173\u95ed",
      },
    });
  });

  it("keeps generic accepted payload for non-card events", () => {
    expect(buildFeishuAcceptedBody({ event_type: "im.message.receive_v1" }, {})).toEqual({
      code: 0,
      msg: "accepted",
    });
  });

  it("extracts menu event details for observability", () => {
    const payload = {
      event: {
        event_key: "provider_status",
        context: {
          open_message_id: "om_menu_1",
        },
      },
    } satisfies Record<string, JsonValue>;

    expect(extractFeishuControlLogFields(payload)).toEqual({
      control_event_key: "provider_status",
      control_hermes_action: undefined,
      control_command_text_preview: undefined,
      control_message_id: "om_menu_1",
    });
  });

  it("falls back to header event_key for menu observability", () => {
    const payload = {
      header: {
        event_key: "model_picker",
      },
      event: {
        context: {
          open_message_id: "om_menu_2",
        },
      },
    } satisfies Record<string, JsonValue>;

    expect(extractFeishuControlLogFields(payload)).toEqual({
      control_event_key: "model_picker",
      control_hermes_action: undefined,
      control_command_text_preview: undefined,
      control_message_id: "om_menu_2",
    });
  });

  it("extracts card command details for observability", () => {
    const payload = {
      event: {
        context: {
          message_id: "om_card_1",
        },
        action: {
          value: {
            hermes_action: "command_run",
            command_text: "/status --verbose",
          },
        },
      },
    } satisfies Record<string, JsonValue>;

    expect(extractFeishuControlLogFields(payload)).toEqual({
      control_event_key: undefined,
      control_hermes_action: "command_run",
      control_command_text_preview: "/status --verbose",
      control_message_id: "om_card_1",
    });
  });

  it("parses stringified queue message bodies", () => {
    expect(normalizeQueueBody('{"job":"feishu_kpi_rollups","reason":"bootstrap"}')).toEqual({
      job: "feishu_kpi_rollups",
      reason: "bootstrap",
    });
  });

  it("returns an empty object for unsupported queue payloads", () => {
    expect(normalizeQueueBody(123)).toEqual({});
    expect(normalizeQueueBody(null)).toEqual({});
  });
});
