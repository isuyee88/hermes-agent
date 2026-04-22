import { describe, expect, it } from "vitest";
import {
  MODEL_CATALOG_DATABASE_BINDING,
  MODEL_CATALOG_MIGRATION_TAG,
  MODEL_CATALOG_SYNC_VIEW,
  MODEL_CATALOG_TABLES,
} from "../src/model-catalog/schema";

describe("model catalog schema", () => {
  it("defines the expected control-plane binding, migration tag, tables, and sync view", () => {
    expect(MODEL_CATALOG_DATABASE_BINDING).toBe("MODEL_CATALOG_DB");
    expect(MODEL_CATALOG_MIGRATION_TAG).toBe("0001_model_catalog_control_plane");
    expect(MODEL_CATALOG_TABLES).toEqual([
      "model_catalog",
      "model_runtime_policy",
      "model_health_stats",
      "model_feedback_events",
      "provider_sync_runs",
    ]);
    expect(MODEL_CATALOG_SYNC_VIEW).toBe("model_registry_sync_view");
  });
});
