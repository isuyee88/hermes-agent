import { describe, expect, it } from "vitest";
import { extractMessageReadInfo } from "../src/core/feishu-webhook";

describe("extractMessageReadInfo", () => {
  it("prefers payload read_time when present", () => {
    const result = extractMessageReadInfo({
      event: {
        read_time: "1776770156863",
        message_id_list: ["om_reply_1"],
      },
    });

    expect(result.readTime).toBe("1776770156863");
    expect(result.messageIdList).toEqual(["om_reply_1"]);
  });

  it("falls back to event_time style fields and normalizes seconds to milliseconds", () => {
    const result = extractMessageReadInfo({
      event: {
        event_time: "1776770155",
        message_id_list: ["om_reply_1", "om_reply_1", "om_reply_2"],
      },
    });

    expect(result.readTime).toBe("1776770155000");
    expect(result.messageIdList).toEqual(["om_reply_1", "om_reply_2"]);
  });

  it("falls back to header create_time when event timestamp is absent", () => {
    const result = extractMessageReadInfo({
      header: {
        create_time: "1776770156",
      },
      event: {
        message_id_list: ["om_reply_3"],
      },
    });

    expect(result.readTime).toBe("1776770156000");
  });
});
