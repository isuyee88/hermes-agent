import { afterEach, describe, expect, it, vi } from "vitest";
import type { Env } from "../src/runtime";
const { runFeishuKpiRollupTickMock } = vi.hoisted(() => ({
  runFeishuKpiRollupTickMock: vi.fn(),
}));

vi.mock("../src/observability/kpi-rollups", () => ({
  runFeishuKpiRollupTick: runFeishuKpiRollupTickMock,
}));

import {
  KPI_ROLLUP_JOB_NAME,
  handleKpiRollupQueueMessage,
  normalizeKpiRollupQueueMessage,
  shouldForceImmediateKpiRun,
} from "../src/observability/kpi-rollup-queue";

type ControlPlaneRow = {
  job_name: string;
  next_run_at: number;
  lease_until: number;
  last_run_token: string;
  updated_at: number;
  last_run_at?: number;
  last_run_status?: string;
  last_error?: string | null;
};

function createControlPlaneDb() {
  const rows = new Map<string, ControlPlaneRow>();

  return {
    rows,
    prepare(query: string) {
      const sql = query.replace(/\s+/g, " ").trim();
      let bindings: unknown[] = [];
      return {
        bind(...values: unknown[]) {
          bindings = values;
          return this;
        },
        async run() {
          if (sql.startsWith("INSERT INTO control_plane_jobs")) {
            const [jobName, nextRunAt, leaseUntil, runToken, updatedAt] = bindings as [
              string,
              number,
              number,
              string,
              number,
            ];
            const existing = rows.get(jobName);
            const forceImmediateRun = sql.includes("1 = 1");
            const nextRunGuardSatisfied = forceImmediateRun || (existing?.next_run_at ?? 0) <= updatedAt;
            const leaseExpired = (existing?.lease_until ?? 0) <= updatedAt;
            if (!existing) {
              rows.set(jobName, {
                job_name: jobName,
                next_run_at: nextRunAt,
                lease_until: leaseUntil,
                last_run_token: runToken,
                updated_at: updatedAt,
              });
            } else if (leaseExpired && nextRunGuardSatisfied) {
              rows.set(jobName, {
                ...existing,
                next_run_at: nextRunAt,
                lease_until: leaseUntil,
                last_run_token: runToken,
                updated_at: updatedAt,
              });
            }
            return { success: true };
          }

          if (sql.startsWith("UPDATE control_plane_jobs")) {
            const [nextRunAt, lastRunAt, lastRunStatus, lastError, runToken, updatedAt, jobName] = bindings as [
              number,
              number,
              string,
              string | null,
              string,
              number,
              string,
            ];
            const existing = rows.get(jobName);
            if (existing) {
              rows.set(jobName, {
                ...existing,
                next_run_at: nextRunAt,
                lease_until: 0,
                last_run_at: lastRunAt,
                last_run_status: lastRunStatus,
                last_error: lastError,
                last_run_token: runToken,
                updated_at: updatedAt,
              });
            }
            return { success: true };
          }

          throw new Error(`unsupported_run_query:${sql}`);
        },
        async first<T>() {
          if (sql.startsWith("SELECT last_run_token, lease_until FROM control_plane_jobs")) {
            const [jobName] = bindings as [string];
            const row = rows.get(jobName);
            return (row
              ? {
                  last_run_token: row.last_run_token,
                  lease_until: row.lease_until,
                }
              : null) as T | null;
          }
          throw new Error(`unsupported_first_query:${sql}`);
        },
      };
    },
  };
}

function buildEnv(overrides: Record<string, unknown> = {}): Env & {
  MODEL_CATALOG_QUEUE: { send: ReturnType<typeof vi.fn> };
  MODEL_CATALOG_DB: ReturnType<typeof createControlPlaneDb>;
} {
  return {
    HERMES_FEISHU_KPI_QUEUE_ENABLED: "true",
    HERMES_FEISHU_KPI_QUEUE_DELAY_SECONDS: "600",
    HERMES_FEISHU_KPI_QUEUE_RETRY_DELAY_SECONDS: "120",
    HERMES_FEISHU_KPI_QUEUE_HARD_RETRY_DELAY_SECONDS: "3600",
    HERMES_FEISHU_KPI_QUEUE_LEASE_SECONDS: "300",
    MODEL_CATALOG_QUEUE: {
      send: vi.fn(async () => undefined),
    },
    MODEL_CATALOG_DB: createControlPlaneDb(),
    ...overrides,
  } as unknown as Env & {
    MODEL_CATALOG_QUEUE: { send: ReturnType<typeof vi.fn> };
    MODEL_CATALOG_DB: ReturnType<typeof createControlPlaneDb>;
  };
}

describe("kpi-rollup queue heartbeat", () => {
  afterEach(() => {
    runFeishuKpiRollupTickMock.mockReset();
    vi.restoreAllMocks();
  });

  it("normalizes KPI queue messages with defaults", () => {
    expect(normalizeKpiRollupQueueMessage(null, 123)).toMatchObject({
      job: KPI_ROLLUP_JOB_NAME,
      reason: "manual",
      attempt: 1,
      enqueued_at: 123,
      scheduled_for: 123,
    });
  });

  it("treats bootstrap and manual as force-immediate reasons", () => {
    expect(shouldForceImmediateKpiRun("bootstrap")).toBe(true);
    expect(shouldForceImmediateKpiRun("manual")).toBe(true);
    expect(shouldForceImmediateKpiRun("self_reschedule")).toBe(false);
  });

  it("skips when the KPI queue heartbeat is disabled", async () => {
    const env = buildEnv({ HERMES_FEISHU_KPI_QUEUE_ENABLED: "false" });

    const result = await handleKpiRollupQueueMessage(env, { job: KPI_ROLLUP_JOB_NAME }, 1_713_769_200);

    expect(result).toEqual({ processed: false, skippedReason: "queue_disabled" });
    expect(runFeishuKpiRollupTickMock).not.toHaveBeenCalled();
    expect(env.MODEL_CATALOG_QUEUE.send).not.toHaveBeenCalled();
  });

  it("runs immediately for bootstrap and self-reschedules to the next 10 minute boundary", async () => {
    const env = buildEnv();
    runFeishuKpiRollupTickMock.mockResolvedValue(undefined);

    const nowSeconds = 1_713_769_380;
    vi.spyOn(Date, "now").mockReturnValue(nowSeconds * 1000);
    const result = await handleKpiRollupQueueMessage(
      env,
      {
        job: KPI_ROLLUP_JOB_NAME,
        reason: "bootstrap",
        attempt: 1,
        scheduled_for: nowSeconds,
      },
      nowSeconds,
    );

    expect(runFeishuKpiRollupTickMock).toHaveBeenCalledWith(
      env,
      expect.objectContaining({
        source: "queue",
        cron: "queue",
        scheduledTime: nowSeconds * 1000,
      }),
    );
    expect(result).toEqual({ processed: true, nextDelaySeconds: 420 });
    expect(env.MODEL_CATALOG_QUEUE.send).toHaveBeenCalledTimes(1);
    expect(env.MODEL_CATALOG_QUEUE.send).toHaveBeenCalledWith(
      expect.objectContaining({
        job: KPI_ROLLUP_JOB_NAME,
        reason: "self_reschedule",
        attempt: 1,
        scheduled_for: nowSeconds + 420,
      }),
      { delaySeconds: 420 },
    );
    expect(env.MODEL_CATALOG_DB.rows.get(KPI_ROLLUP_JOB_NAME)).toMatchObject({
      job_name: KPI_ROLLUP_JOB_NAME,
      next_run_at: nowSeconds + 420,
      lease_until: 0,
      last_run_status: "succeeded",
    });
  });

  it("uses the aligned scheduled_for time for queued hourly boundaries", async () => {
    const env = buildEnv();
    runFeishuKpiRollupTickMock.mockResolvedValue(undefined);
    vi.spyOn(Date, "now").mockReturnValue(1_713_770_401_000);

    await handleKpiRollupQueueMessage(
      env,
      {
        job: KPI_ROLLUP_JOB_NAME,
        reason: "self_reschedule",
        attempt: 1,
        scheduled_for: 1_713_770_400,
      },
      1_713_770_401,
    );

    expect(runFeishuKpiRollupTickMock).toHaveBeenCalledWith(
      env,
      expect.objectContaining({
        scheduledTime: 1_713_770_401_000,
      }),
    );
  });
});
