import { describe, expect, it } from "vitest";
import {
  buildModelCatalogQueueBootstrapResponse,
  classifyControlPlaneError,
  normalizeQueueMessage,
  shouldForceImmediateRun,
} from "../src/model-catalog/queue-heartbeat";

describe("model catalog queue heartbeat", () => {
  it("normalizes queue messages with defaults", () => {
    const message = normalizeQueueMessage(
      {
        reason: "bootstrap",
      },
      100,
    );

    expect(message.job).toBe("model_catalog_control_plane");
    expect(message.reason).toBe("bootstrap");
    expect(message.attempt).toBe(1);
    expect(message.enqueued_at).toBe(100);
    expect(message.scheduled_for).toBe(100);
    expect(message.trace_id).toBeTruthy();
  });

  it("classifies hard control-plane errors separately from transient ones", () => {
    expect(classifyControlPlaneError(new Error("tenant_access_token_failed:401"))).toBe("hard");
    expect(classifyControlPlaneError(new Error("network timeout"))).toBe("transient");
  });

  it("builds a bootstrap response payload for the internal endpoint", () => {
    const payload = buildModelCatalogQueueBootstrapResponse({
      job: "model_catalog_control_plane",
      reason: "bootstrap",
      attempt: 1,
      enqueued_at: 100,
      scheduled_for: 100,
      trace_id: "trace-1",
    });

    expect(payload.ok).toBe(true);
    expect(payload.queued).toBe(true);
    expect(payload.trace_id).toBe("trace-1");
  });

  it("allows bootstrap and manual messages to bypass the normal schedule gate", () => {
    expect(shouldForceImmediateRun("bootstrap")).toBe(true);
    expect(shouldForceImmediateRun("manual")).toBe(true);
    expect(shouldForceImmediateRun("self_reschedule")).toBe(false);
  });
});
