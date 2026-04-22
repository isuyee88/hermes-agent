import { describe, expect, it, vi } from "vitest";

const openRouterMocks = vi.hoisted(() => ({
  fetchOpenRouterModels: vi.fn(),
  mapOpenRouterModelsToSyncBundles: vi.fn(),
}));

const nvidiaMocks = vi.hoisted(() => ({
  fetchNvidiaModels: vi.fn(),
  mapNvidiaModelsToSyncBundles: vi.fn(),
}));

vi.mock("../src/model-catalog/openrouter", () => openRouterMocks);
vi.mock("../src/model-catalog/nvidia", () => nvidiaMocks);

import { syncNvidiaCatalog, syncOpenRouterCatalog } from "../src/model-catalog/sync";

type RecordedStatement = {
  sql: string;
  params: unknown[];
};

function createMockDb() {
  const runCalls: RecordedStatement[] = [];
  const batchCalls: RecordedStatement[][] = [];

  const db = {
    prepare: vi.fn((sql: string) => ({
      bind: (...params: unknown[]) => ({
        sql,
        params,
        run: vi.fn(async () => {
          runCalls.push({ sql, params });
          if (sql.includes("DELETE FROM model_catalog")) {
            return { meta: { changes: 3 } };
          }
          return { meta: { changes: 1 } };
        }),
      }),
    })),
    batch: vi.fn(async (statements: Array<{ sql: string; params: unknown[] }>) => {
      batchCalls.push(statements.map((statement) => ({ sql: statement.sql, params: statement.params })));
      return [];
    }),
  };

  return { db, runCalls, batchCalls };
}

describe("model catalog sync", () => {
  it("deletes stale openrouter rows instead of only hiding them", async () => {
    openRouterMocks.fetchOpenRouterModels.mockResolvedValue([{ id: "openrouter/free-model" }]);
    openRouterMocks.mapOpenRouterModelsToSyncBundles.mockReturnValue([
      {
        catalog: {
          id: "openrouter::openrouter/free-model",
          provider: "openrouter",
          model: "openrouter/free-model",
          providerModelKey: "openrouter::openrouter/free-model",
          status: "active",
          hidden: false,
          isAvailable: true,
          isFree: true,
          manualPinned: false,
          recentUsed: false,
          recentUsedCount: 0,
          taskKindsJson: '["general"]',
          modalitiesJson: '["text"]',
          createdAt: 123,
          updatedAt: 123,
        },
        policies: [],
        syncEntry: {
          provider: "openrouter",
          model: "openrouter/free-model",
        },
      },
    ]);

    const { db, runCalls } = createMockDb();

    const summary = await syncOpenRouterCatalog(
      {
        MODEL_CATALOG_DB: db,
        OPENROUTER_API_KEY: "test-key",
      } as never,
      123,
    );

    expect(summary.discoveredModelCount).toBe(1);
    expect(summary.hiddenModelCount).toBe(3);
    expect(runCalls.some((call) => call.sql.includes("DELETE FROM model_catalog"))).toBe(true);
    expect(runCalls.some((call) => call.sql.includes("UPDATE model_catalog") && call.sql.includes("inactive"))).toBe(false);
  });

  it("syncs NVIDIA catalog rows through the shared provider pipeline", async () => {
    nvidiaMocks.fetchNvidiaModels.mockResolvedValue([{ id: "meta/llama-3.1-405b-instruct" }]);
    nvidiaMocks.mapNvidiaModelsToSyncBundles.mockReturnValue([
      {
        catalog: {
          id: "nvidia::meta/llama-3.1-405b-instruct",
          provider: "nvidia",
          model: "meta/llama-3.1-405b-instruct",
          providerModelKey: "nvidia::meta/llama-3.1-405b-instruct",
          status: "active",
          hidden: false,
          isAvailable: true,
          isFree: false,
          manualPinned: false,
          recentUsed: false,
          recentUsedCount: 0,
          taskKindsJson: '["general"]',
          modalitiesJson: '["text"]',
          createdAt: 456,
          updatedAt: 456,
        },
        policies: [],
        syncEntry: {
          provider: "nvidia",
          model: "meta/llama-3.1-405b-instruct",
        },
      },
    ]);

    const { db, runCalls } = createMockDb();

    const summary = await syncNvidiaCatalog(
      {
        MODEL_CATALOG_DB: db,
      } as never,
      456,
    );

    expect(summary.provider).toBe("nvidia");
    expect(summary.discoveredModelCount).toBe(1);
    expect(runCalls.some((call) => call.params.includes("nvidia"))).toBe(true);
    expect(runCalls.some((call) => call.sql.includes("DELETE FROM model_catalog"))).toBe(true);
  });
});
