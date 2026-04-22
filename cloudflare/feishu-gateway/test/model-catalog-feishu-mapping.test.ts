import { describe, expect, it } from "vitest";
import {
  buildModelRegistryFeishuFields,
  getModelRegistryBitableBlueprint,
  MODEL_REGISTRY_FIELD_SPECS,
} from "../src/model-catalog/feishu-mapping";

describe("model catalog feishu mapping", () => {
  it("extends the bitable blueprint with routing and health fields", () => {
    const fieldNames = MODEL_REGISTRY_FIELD_SPECS.map((field) => field.name);

    expect(fieldNames).toContain("Function Type");
    expect(fieldNames).toContain("Task Kinds");
    expect(fieldNames).toContain("Model Parameters");
    expect(fieldNames).toContain("Rate Limit RPM");
    expect(fieldNames).toContain("Health Score");
    expect(fieldNames).toContain("Gateway Route Name");
  });

  it("builds feishu fields from a flattened registry record", () => {
    const fields = buildModelRegistryFeishuFields({
      provider: "openrouter",
      model: "openrouter/sonoma",
      displayName: "Sonoma",
      isAvailable: true,
      isFree: false,
      hidden: true,
      functionType: "coding",
      taskKinds: ["coding", "long_context"],
      modalities: ["text", "vision"],
      temperatureSupported: true,
      jsonModeSupported: true,
      toolCalling: true,
      structuredOutput: true,
      vision: true,
      rateLimitRpm: 120,
      healthScore: 88.4,
      cacheHitRate: 0.42,
      gatewayRouteName: "text-coding-primary",
      routeFamily: "gateway_text",
      hiddenReason: "negative_feedback_threshold",
    });

    expect(fields.Provider).toBe("openrouter");
    expect(fields["Task Kinds"]).toBe("coding, long_context");
    expect(fields.Modalities).toBe("text, vision");
    expect(fields["Model Parameters"]).toBe("temperature, json_mode");
    expect(fields["Tool Calling"]).toBe(true);
    expect(fields["Rate Limit RPM"]).toBe(120);
    expect(fields["Gateway Route Name"]).toBe("text-coding-primary");
    expect(fields["Hidden Reason"]).toBe("negative_feedback_threshold");
  });

  it("exports a bitable blueprint compatible with the existing table contract", () => {
    const blueprint = getModelRegistryBitableBlueprint();

    expect(blueprint.table_name).toBe("Hermes Model Registry");
    expect(blueprint.required_field_names).toEqual(["Model", "Provider"]);
    expect(blueprint.views.some((view) => view.name === "Routing Health")).toBe(true);
  });
});
