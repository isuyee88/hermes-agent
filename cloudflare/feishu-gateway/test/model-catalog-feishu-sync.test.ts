import { afterEach, describe, expect, it, vi } from "vitest";

const bitableMocks = vi.hoisted(() => ({
  createBitableField: vi.fn(),
  createBitableRecord: vi.fn(),
  createBitableTable: vi.fn(),
  createBitableView: vi.fn(),
  deleteBitableRecord: vi.fn(),
  listBitableFields: vi.fn(),
  listBitableRecords: vi.fn(),
  listBitableTables: vi.fn(),
  listBitableViews: vi.fn(),
  updateBitableRecord: vi.fn(),
}));

vi.mock("../src/services/feishu/bitable", () => bitableMocks);

import { MODEL_REGISTRY_FIELD_SPECS } from "../src/model-catalog/feishu-mapping";
import { mapModelRegistrySyncRows, syncModelRegistryToFeishuBitable } from "../src/model-catalog/feishu-sync";

function createFakeDb(rows: Record<string, unknown>[]) {
  return {
    prepare: vi.fn(() => ({
      all: vi.fn(async () => ({
        results: rows,
      })),
    })),
  };
}

function createFieldRows(names: string[]) {
  return names.map((field_name) => ({ field_name }));
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("model catalog feishu sync", () => {
  it("maps D1 sync-view rows into flattened Feishu sync entries", () => {
    const [entry] = mapModelRegistrySyncRows(
      [
        {
          provider: "openrouter",
          model: "anthropic/claude-3.5-sonnet",
          display_name: "Claude 3.5 Sonnet",
          hidden: 0,
          is_available: 1,
          is_free: 0,
          manual_pinned: 1,
          recent_used: 1,
          recent_used_count: 12,
          task_kinds_json: '["general","coding"]',
          modalities_json: '["text","vision"]',
          tool_calling: 1,
          structured_output: 1,
          streaming: 1,
          vision: 1,
          temperature_supported: 1,
          json_mode_supported: 1,
          gateway_route_name: "text-general",
          route_family: "gateway_text",
          health_score: 93.5,
          cache_hit_rate: 0.44,
        },
      ],
      1234567890,
    );

    expect(entry.provider).toBe("openrouter");
    expect(entry.model).toBe("anthropic/claude-3.5-sonnet");
    expect(entry.manualPinned).toBe(true);
    expect(entry.recentUsed).toBe(true);
    expect(entry.taskKinds).toEqual(["general", "coding"]);
    expect(entry.modalities).toEqual(["text", "vision"]);
    expect(entry.toolCalling).toBe(true);
    expect(entry.structuredOutput).toBe(true);
    expect(entry.streaming).toBe(true);
    expect(entry.vision).toBe(true);
    expect(entry.temperatureSupported).toBe(true);
    expect(entry.jsonModeSupported).toBe(true);
    expect(entry.gatewayRouteName).toBe("text-general");
    expect(entry.routeFamily).toBe("gateway_text");
    expect(entry.healthScore).toBe(93.5);
    expect(entry.lastSyncAt).toBe(1234567890);
  });

  it("skips mirroring cleanly when Feishu sync is disabled or unconfigured", async () => {
    await expect(
      syncModelRegistryToFeishuBitable(
        {
          FEISHU_MODEL_REGISTRY_MIRROR_ENABLED: "false",
        } as never,
        123,
      ),
    ).resolves.toEqual({
      mirrored: false,
      skippedReason: "mirror_disabled",
    });

    await expect(
      syncModelRegistryToFeishuBitable(
        {
          FEISHU_MODEL_REGISTRY_MIRROR_ENABLED: "true",
        } as never,
        123,
      ),
    ).resolves.toEqual({
      mirrored: false,
      skippedReason: "missing_app_token",
    });
  });

  it("prioritizes deleting stale Feishu-only rows before creating new snapshot rows", async () => {
    bitableMocks.listBitableTables.mockResolvedValue([
      { table_id: "tbl-1", name: "Hermes Model Registry Mirror" },
    ]);
    bitableMocks.listBitableFields.mockResolvedValue(
      createFieldRows(MODEL_REGISTRY_FIELD_SPECS.map((spec) => spec.name)),
    );
    bitableMocks.listBitableViews.mockResolvedValue([{ view_name: "All Models" }]);
    bitableMocks.listBitableRecords.mockResolvedValue([
      {
        record_id: "rec-stale",
        fields: {
          Provider: "openrouter",
          Model: "old/free-model",
          "Last Sync At": 100,
        },
      },
    ]);
    bitableMocks.deleteBitableRecord.mockResolvedValue(undefined);
    bitableMocks.createBitableRecord.mockResolvedValue({ record_id: "rec-created" });

    const summary = await syncModelRegistryToFeishuBitable(
      {
        FEISHU_MODEL_REGISTRY_MIRROR_ENABLED: "true",
        FEISHU_BITABLE_APP_TOKEN: "app-token",
        FEISHU_BITABLE_TABLE_NAME: "Hermes Model Registry Mirror",
        FEISHU_MODEL_REGISTRY_MAX_MUTATIONS_PER_RUN: "1",
        FEISHU_MODEL_REGISTRY_STALE_DELETE_AFTER_SECONDS: "60",
        MODEL_CATALOG_DB: createFakeDb([
          {
            provider: "openrouter",
            model: "new/free-model",
            status: "active",
            hidden: 0,
            is_available: 1,
            is_free: 1,
          },
        ]),
      } as never,
      10_000,
    );

    expect(bitableMocks.deleteBitableRecord).toHaveBeenCalledTimes(1);
    expect(bitableMocks.createBitableRecord).not.toHaveBeenCalled();
    expect(summary).toMatchObject({
      mirrored: true,
      rowCount: 1,
      existingRowCount: 1,
      historicalOnlyCount: 1,
      staleDeleteCandidateCount: 1,
      entryCreateCandidateCount: 1,
      deleted: 1,
      created: 0,
      pending: true,
      remainingEstimate: 1,
      mutationsApplied: 1,
      mutationBudget: 1,
    });
  });

  it("marks recent Feishu-only rows as hidden without refreshing Last Sync At", async () => {
    bitableMocks.listBitableTables.mockResolvedValue([
      { table_id: "tbl-1", name: "Hermes Model Registry Mirror" },
    ]);
    bitableMocks.listBitableFields.mockResolvedValue(
      createFieldRows([
        "Provider",
        "Model",
        "Status",
        "Hidden",
        "Is Available",
        "Last Sync At",
        "Failure Kind",
        "Last Error Message",
        "Hidden Reason",
      ]),
    );
    bitableMocks.listBitableViews.mockResolvedValue([{ view_name: "All Models" }]);
    bitableMocks.listBitableRecords.mockResolvedValue([
      {
        record_id: "rec-recent",
        fields: {
          Provider: "openrouter",
          Model: "recent/free-model",
          Status: "active",
          Hidden: false,
          "Is Available": true,
          "Last Sync At": 9_900,
        },
      },
    ]);
    bitableMocks.updateBitableRecord.mockResolvedValue({ record_id: "rec-recent" });

    const summary = await syncModelRegistryToFeishuBitable(
      {
        FEISHU_MODEL_REGISTRY_MIRROR_ENABLED: "true",
        FEISHU_BITABLE_APP_TOKEN: "app-token",
        FEISHU_BITABLE_TABLE_NAME: "Hermes Model Registry Mirror",
        FEISHU_MODEL_REGISTRY_STALE_DELETE_AFTER_SECONDS: "86400",
        MODEL_CATALOG_DB: createFakeDb([]),
      } as never,
      10_000,
    );

    expect(bitableMocks.updateBitableRecord).toHaveBeenCalledTimes(1);
    expect(bitableMocks.updateBitableRecord).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        recordId: "rec-recent",
        fields: {
          Status: "inactive",
          Hidden: true,
          "Is Available": false,
          "Last Error Message": "Model missing from current Hermes registry snapshot",
          "Failure Kind": "not_in_snapshot",
          "Hidden Reason": "missing_from_snapshot",
        },
      }),
    );
    const updateFields = bitableMocks.updateBitableRecord.mock.calls[0]?.[1]?.fields as Record<string, unknown>;
    expect(updateFields).not.toHaveProperty("Last Sync At");
    expect(summary).toMatchObject({
      mirrored: true,
      historicalOnlyCount: 1,
      staleDeleteCandidateCount: 0,
      staleHideCandidateCount: 1,
      hidden: 1,
      deleted: 0,
      pending: false,
    });
  });
});
