import { describe, expect, it } from "vitest";
import { classifyFeishuSendFailure } from "../src/services/feishu/messages";

describe("classifyFeishuSendFailure", () => {
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
});
