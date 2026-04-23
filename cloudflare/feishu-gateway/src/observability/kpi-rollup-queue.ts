import type { Env, JsonValue } from "../runtime";
import { runFeishuKpiRollupTick } from "./kpi-rollups";

export type KpiRollupQueueMessage = {
  job: string;
  reason: string;
  attempt: number;
  enqueued_at: number;
  scheduled_for: number;
  trace_id: string;
};

type QueueHandlerResult = {
  processed: boolean;
  skippedReason?: string;
  retryDelaySeconds?: number;
  nextDelaySeconds?: number;
};

const KPI_ROLLUP_JOB_NAME = "feishu_kpi_rollups";
const MINIMUM_DELAY_SECONDS = 5;

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function parseBoolean(value: unknown, fallback = false): boolean {
  const normalized = trim(value).toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return fallback;
}

function parsePositiveInt(value: unknown, fallback: number, min = 1, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = Number.parseInt(trim(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, parsed));
}

export function normalizeKpiRollupQueueMessage(
  message: Partial<KpiRollupQueueMessage> | null | undefined,
  nowSeconds = Math.trunc(Date.now() / 1000),
): KpiRollupQueueMessage {
  return {
    job: trim(message?.job) || KPI_ROLLUP_JOB_NAME,
    reason: trim(message?.reason) || "manual",
    attempt: Math.max(1, Number(message?.attempt) || 1),
    enqueued_at: Number(message?.enqueued_at) || nowSeconds,
    scheduled_for: Number(message?.scheduled_for) || nowSeconds,
    trace_id: trim(message?.trace_id) || crypto.randomUUID(),
  };
}

export function shouldForceImmediateKpiRun(reason: string): boolean {
  const normalized = trim(reason).toLowerCase();
  return normalized === "bootstrap" || normalized === "manual";
}

function classifyKpiControlPlaneError(error: unknown): "hard" | "transient" {
  const message = trim(error instanceof Error ? error.message : error).toLowerCase();
  if (
    message.includes("missing") ||
    message.includes("binding") ||
    message.includes("cloudflare_api_token") ||
    message.includes("cloudflare_account_id")
  ) {
    return "hard";
  }
  return "transient";
}

function getQueueBinding(env: Env): Queue<KpiRollupQueueMessage> | null {
  const binding = (env as Env & Record<string, unknown>).MODEL_CATALOG_QUEUE;
  if (binding && typeof binding === "object") {
    return binding as Queue<KpiRollupQueueMessage>;
  }
  return null;
}

function getQueueEnabled(env: Env): boolean {
  return parseBoolean((env as Env & Record<string, unknown>).HERMES_FEISHU_KPI_QUEUE_ENABLED, true);
}

function getQueueIntervalSeconds(env: Env): number {
  return parsePositiveInt((env as Env & Record<string, unknown>).HERMES_FEISHU_KPI_QUEUE_DELAY_SECONDS, 600, 60, 3600);
}

function getQueueRetryDelaySeconds(env: Env): number {
  return parsePositiveInt(
    (env as Env & Record<string, unknown>).HERMES_FEISHU_KPI_QUEUE_RETRY_DELAY_SECONDS,
    120,
    30,
    3600,
  );
}

function getQueueHardRetryDelaySeconds(env: Env): number {
  return parsePositiveInt(
    (env as Env & Record<string, unknown>).HERMES_FEISHU_KPI_QUEUE_HARD_RETRY_DELAY_SECONDS,
    3600,
    60,
    86400,
  );
}

function getQueueLeaseSeconds(env: Env): number {
  return parsePositiveInt((env as Env & Record<string, unknown>).HERMES_FEISHU_KPI_QUEUE_LEASE_SECONDS, 300, 60, 3600);
}

function getNextAlignedDelaySeconds(intervalSeconds: number, nowSeconds: number, scheduledForSeconds: number): number {
  const interval = Math.max(60, intervalSeconds);
  const reference = Math.max(nowSeconds, scheduledForSeconds);
  const nextRunAt = Math.floor(reference / interval) * interval + interval;
  return Math.max(MINIMUM_DELAY_SECONDS, nextRunAt - nowSeconds);
}

async function acquireControlPlaneLease(
  env: Env,
  jobName: string,
  runToken: string,
  nowSeconds: number,
  leaseSeconds: number,
  forceImmediateRun = false,
): Promise<boolean> {
  const db = env.MODEL_CATALOG_DB;
  if (!db) {
    return false;
  }
  const nextRunGuard = forceImmediateRun
    ? "1 = 1"
    : "COALESCE(control_plane_jobs.next_run_at, 0) <= excluded.updated_at";
  await db
    .prepare(
      `INSERT INTO control_plane_jobs (
         job_name,
         next_run_at,
         lease_until,
         last_run_token,
         updated_at
       ) VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(job_name) DO UPDATE SET
         next_run_at = CASE
           WHEN COALESCE(control_plane_jobs.lease_until, 0) <= excluded.updated_at
            AND ${nextRunGuard}
           THEN excluded.next_run_at
           ELSE control_plane_jobs.next_run_at
         END,
         lease_until = CASE
           WHEN COALESCE(control_plane_jobs.lease_until, 0) <= excluded.updated_at
            AND ${nextRunGuard}
           THEN excluded.lease_until
           ELSE control_plane_jobs.lease_until
         END,
         last_run_token = CASE
           WHEN COALESCE(control_plane_jobs.lease_until, 0) <= excluded.updated_at
            AND ${nextRunGuard}
           THEN excluded.last_run_token
           ELSE control_plane_jobs.last_run_token
         END,
         updated_at = CASE
           WHEN COALESCE(control_plane_jobs.lease_until, 0) <= excluded.updated_at
            AND ${nextRunGuard}
           THEN excluded.updated_at
           ELSE control_plane_jobs.updated_at
         END`,
    )
    .bind(jobName, nowSeconds, nowSeconds + leaseSeconds, runToken, nowSeconds)
    .run();

  const row = await db
    .prepare(`SELECT last_run_token, lease_until FROM control_plane_jobs WHERE job_name = ?`)
    .bind(jobName)
    .first<{ last_run_token?: string | null; lease_until?: number | null }>();

  return trim(row?.last_run_token) === runToken && Number(row?.lease_until ?? 0) >= nowSeconds;
}

async function finishControlPlaneLease(args: {
  env: Env;
  jobName: string;
  runToken: string;
  nowSeconds: number;
  nextRunAt: number;
  status: "succeeded" | "failed" | "skipped";
  errorMessage?: string;
}): Promise<void> {
  const db = args.env.MODEL_CATALOG_DB;
  if (!db) {
    return;
  }
  await db
    .prepare(
      `UPDATE control_plane_jobs
       SET next_run_at = ?,
           lease_until = 0,
           last_run_at = ?,
           last_run_status = ?,
           last_error = ?,
           last_run_token = ?,
           updated_at = ?
       WHERE job_name = ?`,
    )
    .bind(
      args.nextRunAt,
      args.nowSeconds,
      args.status,
      args.errorMessage ? args.errorMessage.slice(0, 500) : null,
      args.runToken,
      args.nowSeconds,
      args.jobName,
    )
    .run();
}

export async function enqueueKpiRollupHeartbeat(
  env: Env,
  message: Partial<KpiRollupQueueMessage>,
  delaySeconds = 0,
): Promise<KpiRollupQueueMessage> {
  const queue = getQueueBinding(env);
  if (!queue) {
    throw new Error("model_catalog_queue_binding_missing");
  }
  const nowSeconds = Math.trunc(Date.now() / 1000);
  const normalized = normalizeKpiRollupQueueMessage(
    {
      ...message,
      enqueued_at: nowSeconds,
      scheduled_for: nowSeconds + Math.max(0, delaySeconds),
    },
    nowSeconds,
  );
  await queue.send(normalized, { delaySeconds: Math.max(0, delaySeconds) });
  return normalized;
}

export async function handleKpiRollupQueueMessage(
  env: Env,
  rawMessage: Partial<KpiRollupQueueMessage> | null | undefined,
  nowSeconds = Math.trunc(Date.now() / 1000),
): Promise<QueueHandlerResult> {
  if (!getQueueEnabled(env)) {
    return { processed: false, skippedReason: "queue_disabled" };
  }
  const message = normalizeKpiRollupQueueMessage(rawMessage, nowSeconds);
  const runToken = crypto.randomUUID();
  const leaseAcquired = await acquireControlPlaneLease(
    env,
    message.job,
    runToken,
    nowSeconds,
    getQueueLeaseSeconds(env),
    shouldForceImmediateKpiRun(message.reason),
  );
  if (!leaseAcquired) {
    return { processed: false, skippedReason: "lease_not_acquired" };
  }

  try {
    const effectiveScheduledTimeSeconds = Math.max(nowSeconds, message.scheduled_for);
    await runFeishuKpiRollupTick(env, {
      source: "queue",
      cron: "queue",
      scheduledTime: effectiveScheduledTimeSeconds * 1000,
    });
    const nextDelaySeconds = getNextAlignedDelaySeconds(
      getQueueIntervalSeconds(env),
      nowSeconds,
      effectiveScheduledTimeSeconds,
    );
    await finishControlPlaneLease({
      env,
      jobName: message.job,
      runToken,
      nowSeconds,
      nextRunAt: nowSeconds + nextDelaySeconds,
      status: "succeeded",
    });
    await enqueueKpiRollupHeartbeat(
      env,
      {
        job: message.job,
        reason: "self_reschedule",
        attempt: 1,
      },
      nextDelaySeconds,
    );
    return { processed: true, nextDelaySeconds };
  } catch (error) {
    const retryDelaySeconds =
      classifyKpiControlPlaneError(error) === "hard"
        ? getQueueHardRetryDelaySeconds(env)
        : getQueueRetryDelaySeconds(env);
    await finishControlPlaneLease({
      env,
      jobName: message.job,
      runToken,
      nowSeconds,
      nextRunAt: nowSeconds + retryDelaySeconds,
      status: "failed",
      errorMessage: error instanceof Error ? error.message : String(error),
    });
    await enqueueKpiRollupHeartbeat(
      env,
      {
        job: message.job,
        reason: classifyKpiControlPlaneError(error) === "hard" ? "hard_retry" : "retry",
        attempt: message.attempt + 1,
      },
      retryDelaySeconds,
    );
    return { processed: true, retryDelaySeconds };
  }
}

export async function bootstrapKpiRollupHeartbeat(env: Env): Promise<KpiRollupQueueMessage> {
  return enqueueKpiRollupHeartbeat(
    env,
    {
      job: KPI_ROLLUP_JOB_NAME,
      reason: "bootstrap",
      attempt: 1,
    },
    0,
  );
}

export function isInternalKpiRollupRequest(request: Request): boolean {
  const url = new URL(request.url);
  return url.pathname === "/internal/feishu-kpi/queue/bootstrap";
}

export function buildKpiRollupQueueBootstrapResponse(message: KpiRollupQueueMessage): Record<string, JsonValue> {
  return {
    ok: true,
    queued: true,
    job: message.job,
    reason: message.reason,
    attempt: message.attempt,
    enqueued_at: message.enqueued_at,
    scheduled_for: message.scheduled_for,
    trace_id: message.trace_id,
  };
}

export { KPI_ROLLUP_JOB_NAME, classifyKpiControlPlaneError };
