import { describe, expect, it } from "vitest";
import { buildModalRequestEnvelope, unwrapModalResponseEnvelope } from "../src/contracts/feishu-internal";

describe("feishu internal envelope", () => {
  it("builds a versioned request envelope", () => {
    const envelope = buildModalRequestEnvelope(
      "/internal/feishu/agent-plan",
      {
        correlation_id: "corr-1",
        session_key: "agent:main",
        event_id: "evt-1",
        request_class: "text_plain",
        route_hint: "modal_heavy_exec",
        route_family: "gateway_text",
        gateway_route_name: "text-general",
        text: "hello",
      },
      {
        action: "dispatch_command",
      },
      [],
    );

    expect(envelope.contract_version).toBe("feishu_internal.v1");
    expect(envelope.ingress.correlation_id).toBe("corr-1");
    expect(envelope.route.route_hint).toBe("modal_heavy_exec");
    expect(envelope.gateway_meta.legacy_payload).toBeTruthy();
  });

  it("unwraps a versioned response envelope via legacy_response", () => {
    const payload = unwrapModalResponseEnvelope<{
      route_hint: string;
      execution_mode: string;
    }>({
      contract_version: "feishu_internal.v1",
      result: {},
      send_plan: {},
      session_patch: {},
      reconcile: {},
      provider_metrics: {},
      legacy_response: {
        route_hint: "fast_control",
        execution_mode: "control_complete",
      },
    });

    expect(payload.route_hint).toBe("fast_control");
    expect(payload.execution_mode).toBe("control_complete");
  });
});
