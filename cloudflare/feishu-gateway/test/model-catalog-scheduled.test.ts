import { describe, expect, it, vi } from "vitest";
import { runModelCatalogScheduledTick } from "../src/model-catalog/scheduled";

describe("model catalog scheduled tick", () => {
  it("logs and exits when the D1 binding is missing", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    await runModelCatalogScheduledTick(
      {
        HERMES_MODEL_CATALOG_SYNC_ENABLED: "true",
        HERMES_MODEL_CATALOG_SYNC_PROVIDERS: "openrouter, nvidia",
      } as never,
      {
        cron: "0 * * * *",
        scheduledTime: Date.now(),
      } as ScheduledController,
    );

    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy.mock.calls[0]?.[0]).toContain("\"event\":\"model_catalog.tick.skipped\"");

    logSpy.mockRestore();
  });

  it("writes a control-plane heartbeat row when the binding exists", async () => {
    const all = vi.fn<() => Promise<{ results: unknown[] }>>().mockResolvedValue({ results: [] });
    const first = vi
      .fn<() => Promise<{ total_models: number; hidden_models: number; available_models: number }>>()
      .mockResolvedValue({ total_models: 3, hidden_models: 1, available_models: 2 });
    const run = vi.fn<() => Promise<unknown>>().mockResolvedValue({});
    const bind = vi.fn(() => ({ run }));
    const prepare = vi
      .fn()
      .mockReturnValueOnce({ bind })
      .mockReturnValueOnce({ all })
      .mockReturnValueOnce({ first });
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    await runModelCatalogScheduledTick(
      {
        MODEL_CATALOG_DB: {
          prepare,
        },
        HERMES_MODEL_CATALOG_SYNC_ENABLED: "false",
      } as never,
      {
        cron: "0 * * * *",
        scheduledTime: Date.now(),
      } as ScheduledController,
    );

    expect(prepare).toHaveBeenCalledTimes(3);
    expect(all).toHaveBeenCalledTimes(1);
    expect(first).toHaveBeenCalledTimes(1);
    expect(bind).toHaveBeenCalledTimes(1);
    expect(run).toHaveBeenCalledTimes(1);
    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy.mock.calls[0]?.[0]).toContain("\"event\":\"model_catalog.tick.completed\"");

    logSpy.mockRestore();
  });
});
