import { afterEach, describe, expect, it, vi } from "vitest";
import type { Env } from "../src/runtime";
import { runFeishuKpiRollupScheduledTick } from "../src/observability/kpi-rollups";

function buildEnv(overrides: Record<string, unknown> = {}): Env {
  return {
    CLOUDFLARE_API_TOKEN: "token",
    CLOUDFLARE_ACCOUNT_ID: "acct-1",
    CLOUDFLARE_AI_GATEWAY_BASE_URL: "https://gateway.ai.cloudflare.com/v1/acct-1/gateway/compat",
    FEISHU_GATEWAY_KPI_ROLLUPS: {
      writeDataPoint: vi.fn(),
    },
    ...overrides,
  } as unknown as Env;
}

function buildController(cron: string, isoTime: string): ScheduledController {
  return {
    cron,
    scheduledTime: Date.parse(isoTime),
  } as ScheduledController;
}

describe("runFeishuKpiRollupScheduledTick", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("skips when the rollup dataset binding is unavailable", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    await runFeishuKpiRollupScheduledTick(
      buildEnv({ FEISHU_GATEWAY_KPI_ROLLUPS: undefined }),
      buildController("*/10 * * * *", "2026-04-23T02:20:00.000Z"),
    );

    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy.mock.calls[0]?.[0]).toContain("\"reason\":\"rollup_dataset_binding_missing\"");
  });

  it("writes a 10 minute rollup row with derived KPI rates", async () => {
    const env = buildEnv();
    const writeDataPoint = vi.mocked(env.FEISHU_GATEWAY_KPI_ROLLUPS.writeDataPoint);
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const query = String(init?.body ?? "");
      if (query.includes("FROM hermes_feishu_kpi_rollups")) {
        return new Response(JSON.stringify({ success: true, data: [{ value: 0 }] }), { status: 200 });
      }

      const responses: Array<[string, number]> = [
        ["index1 = 'feishu.webhook.accepted'", 10],
        ["index1 = 'feishu.message_read.accepted'", 4],
        ["index1 = 'feishu.cf_ai_exec.done'\nFORMAT JSON", 8],
        ["index1 = 'feishu.send.operation.done'", 7],
        ["index1 = 'feishu.send.operation.error'", 1],
        ["index1 = 'feishu.cf_ai_exec.fallback.done'", 2],
        ["index1 = 'feishu.cf_ai_exec.fallback'\nFORMAT JSON", 2],
        ["index1 = 'feishu.cf_ai_exec.fallback' AND blob18 = 'rate_limited'", 0],
        ["index1 = 'feishu.cf_ai_exec.done' AND double5 = 1 AND blob10 = 'HIT'", 2],
        ["index1 = 'feishu.cf_ai_exec.done' AND double5 = 1", 5],
        ["index1 = 'feishu.cf_ai_exec.done' AND double6 = 1", 8],
        ["index1 = 'feishu.cf_ai_exec.done' AND double7 = 1", 7],
        ["index1 = 'feishu.cf_ai_exec.done' AND double8 = 1 AND double3 = 1", 3],
        ["index1 = 'feishu.cf_ai_exec.done' AND double8 = 1", 4],
      ];

      for (const [needle, value] of responses) {
        if (query.includes(needle)) {
          return new Response(JSON.stringify({ success: true, data: [{ value }] }), { status: 200 });
        }
      }
      throw new Error(`unexpected_query:${query}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    await runFeishuKpiRollupScheduledTick(env, buildController("*/10 * * * *", "2026-04-23T02:20:00.000Z"));

    expect(writeDataPoint).toHaveBeenCalledTimes(1);
    expect(writeDataPoint).toHaveBeenCalledWith(
      expect.objectContaining({
        indexes: ["10m"],
        blobs: expect.arrayContaining([
          "2026-04-23T02:10:00.000Z",
          "2026-04-23T02:20:00.000Z",
          "10m:2026-04-23T02:10:00.000Z",
          "*/10 * * * *",
          "hermes_feishu_gateway_events",
        ]),
        doubles: expect.arrayContaining([
          10,
          4,
          8,
          7,
          1,
          0.125,
          2,
          2,
          1,
          0,
          5,
          2,
          0.4,
          8,
          1,
          7,
          0.875,
          4,
          3,
          0.75,
        ]),
      }),
    );
  });

  it("writes both 10 minute and hourly rollups on the top-of-hour 10 minute tick", async () => {
    const env = buildEnv();
    const writeDataPoint = vi.mocked(env.FEISHU_GATEWAY_KPI_ROLLUPS.writeDataPoint);
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const query = String(init?.body ?? "");
      if (query.includes("FROM hermes_feishu_kpi_rollups")) {
        return new Response(JSON.stringify({ success: true, data: [{ value: 0 }] }), { status: 200 });
      }
      return new Response(JSON.stringify({ success: true, data: [{ value: 0 }] }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    await runFeishuKpiRollupScheduledTick(env, buildController("*/10 * * * *", "2026-04-23T03:00:00.000Z"));

    expect(writeDataPoint).toHaveBeenCalledTimes(2);
    expect(writeDataPoint.mock.calls[0]?.[0]).toMatchObject({
      indexes: ["10m"],
      blobs: expect.arrayContaining([
        "2026-04-23T02:50:00.000Z",
        "2026-04-23T03:00:00.000Z",
        "10m:2026-04-23T02:50:00.000Z",
        "*/10 * * * *",
      ]),
    });
    expect(writeDataPoint.mock.calls[1]?.[0]).toMatchObject({
      indexes: ["1h"],
      blobs: expect.arrayContaining([
        "2026-04-23T02:00:00.000Z",
        "2026-04-23T03:00:00.000Z",
        "1h:2026-04-23T02:00:00.000Z",
        "*/10 * * * *",
      ]),
    });
  });
});
