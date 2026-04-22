import { describe, expect, it } from "vitest";
import {
  compactOutboundMessageCorrelationRefs,
  resolveOutboundMessageCorrelationRefs,
} from "../src/durable/message-correlation";

describe("message correlation index", () => {
  it("keeps the newest ref per send message id and trims stale refs", () => {
    const now = Date.parse("2026-04-19T12:00:00.000Z");
    const refs = compactOutboundMessageCorrelationRefs(
      [
        {
          send_message_id: "om_stale",
          correlation_id: "corr-stale",
          session_key: "session-1",
          event_id: "evt-stale",
          source_message_id: "msg-stale",
          kind: "text",
          created_at: "2026-04-14T11:59:59.000Z",
        },
        {
          send_message_id: "om_1",
          correlation_id: "corr-old",
          session_key: "session-1",
          event_id: "evt-old",
          source_message_id: "msg-old",
          kind: "text",
          created_at: "2026-04-19T11:58:00.000Z",
        },
        {
          send_message_id: "om_1",
          correlation_id: "corr-new",
          session_key: "session-1",
          event_id: "evt-new",
          source_message_id: "msg-new",
          kind: "text",
          created_at: "2026-04-19T11:59:00.000Z",
        },
      ],
      now,
    );

    expect(refs).toHaveLength(1);
    expect(refs[0]?.correlation_id).toBe("corr-new");
  });

  it("resolves read receipts back to the stored outbound refs in request order", () => {
    const resolution = resolveOutboundMessageCorrelationRefs(
      [
        {
          send_message_id: "om_2",
          correlation_id: "corr-2",
          session_key: "session-1",
          event_id: "evt-2",
          source_message_id: "msg-2",
          kind: "text",
          created_at: "2026-04-19T11:59:00.000Z",
        },
        {
          send_message_id: "om_1",
          correlation_id: "corr-1",
          session_key: "session-1",
          event_id: "evt-1",
          source_message_id: "msg-1",
          kind: "text",
          created_at: "2026-04-19T11:58:00.000Z",
        },
      ],
      ["om_1", "om_missing", "om_2"],
    );

    expect(resolution.refs.map((item) => item.correlation_id)).toEqual(["corr-1", "corr-2"]);
    expect(resolution.unmatched_message_ids).toEqual(["om_missing"]);
  });
});
