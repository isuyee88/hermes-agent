import type { Env, JsonValue } from "../runtime";
import { runModelCatalogScheduledTick } from "./scheduled";

export type ModelCatalogQueueMessage = {
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

const MODEL_CATALOG_JOB_NAME = "model_catalog_control_plane";

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

function normalizeQueueMessage(
  message: Partial<ModelCatalogQueueMessage> | null | undefined,
  nowSeconds = Math.trunc(Date.now() / 1000),
): ModelCatalogQueueMessage {
  return {
    job: trim(message?.job) || MODEL_CATALOG_JOB_NAME,
    reason: trim(message?.reason) || "manual",
    attempt: Math.max(1, Number(message?.attempt) || 1),
    enqueued_at: Number(message?.enqueued_at) || nowSeconds,
    scheduled_for: Number(message?.scheduled_for) || nowSeconds,
    trace_id: trim(message?.trace_id) || crypto.randomUUID(),
  };
}

function shouldForceImmediateRun(reason: string): boolean {
  const normalized = trim(reason).toLowerCase();
  return normalized === "bootstrap" || normalized === "manual";
}

function getQueueBinding(env: Env): Queue<ModelCatalogQueueMessage> | null {
  const binding = (env as Env & Record<string, unknown>).MODEL_CATALOG_QUEUE;
  if (binding && typeof binding === "object") {
    return binding as Queue<ModelCatalogQueueMessage>;
  }
  return null;
}

function getQueueEnabled(env: Env): boolean {
  return parseBoolean((env as Env & Record<string, unknown>).HERMES_MODEL_CATALOG_QUEUE_ENABLED, true);
}

function getQueueIntervalSeconds(env: Env): number {
  return parsePositiveInt((env as Env & Record<string, unknown>).HERMES_MODEL_CATALOG_QUEUE_DELAY_SECONDS, 3600, 60, 86400);
}

function getQueueRetryDelaySeconds(env: Env): number {
  return parsePositiveInt(
    (env as Env & Record<string, unknown>).HERMES_MODEL_CATALOG_QUEUE_RETRY_DELAY_SECONDS,
    300,
    60,
    86400,
  );
}

function getMirrorContinuationDelaySeconds(env: Env): number {
  return parsePositiveInt(
    (env as Env & Record<string, unknown>).FEISHU_MODEL_REGISTRY_CONTINUE_DELAY_SECONDS,
    15,
    5,
    300,
  );
}

function getQueueHardRetryDelaySeconds(env: Env): number {
  return parsePositiveInt(
    (env as Env & Record<string, unknown>).HERMES_MODEL_CATALOG_QUEUE_HARD_RETRY_DELAY_SECONDS,
    21600,
    300,
    86400,
  );
}

function getQueueLeaseSeconds(env: Env): number {
  return parsePositiveInt((env as Env & Record<string, unknown>).HERMES_MODEL_CATALOG_QUEUE_LEASE_SECONDS, 900, 60, 86400);
}

function classifyControlPlaneError(error: unknown): "hard" | "transient" {
  const message = trim(error instanceof Error ? error.message : error).toLowerCase();
  if (
    message.includes("auth") ||
    message.includes("401") ||
    message.includes("403") ||
    message.includes("missing_app_token") ||
    message.includes("tenant_access_token_failed") ||
    message.includes("required_fields_missing")
  ) {
    return "hard";
  }
  return "transient";
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

export async function enqueueModelCatalogHeartbeat(
  env: Env,
  message: Partial<ModelCatalogQueueMessage>,
  delaySeconds = 0,
): Promise<ModelCatalogQueueMessage> {
  const queue = getQueueBinding(env);
  if (!queue) {
    throw new Error("model_catalog_queue_binding_missing");
  }
  const nowSeconds = Math.trunc(Date.now() / 1000);
  const normalized = normalizeQueueMessage(
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

export async function handleModelCatalogQueueMessage(
  env: Env,
  rawMessage: Partial<ModelCatalogQueueMessage> | null | undefined,
  nowSeconds = Math.trunc(Date.now() / 1000),
): Promise<QueueHandlerResult> {
  if (!getQueueEnabled(env)) {
    return { processed: false, skippedReason: "queue_disabled" };
  }
  const message = normalizeQueueMessage(rawMessage, nowSeconds);
  const runToken = crypto.randomUUID();
  const leaseAcquired = await acquireControlPlaneLease(
    env,
    message.job,
    runToken,
    nowSeconds,
    getQueueLeaseSeconds(env),
    shouldForceImmediateRun(message.reason),
  );
  if (!leaseAcquired) {
    return { processed: false, skippedReason: "lease_not_acquired" };
  }

  try {
    const tickSummary = await runModelCatalogScheduledTick(
      env,
      {
        cron: "queue",
        scheduledTime: nowSeconds * 1000,
      } as ScheduledController,
    );
    const pendingFeishuMirror = tickSummary?.feishuMirror?.pending === true;
    const nextDelaySeconds = pendingFeishuMirror ? getMirrorContinuationDelaySeconds(env) : getQueueIntervalSeconds(env);
    await finishControlPlaneLease({
      env,
      jobName: message.job,
      runToken,
      nowSeconds,
      nextRunAt: nowSeconds + nextDelaySeconds,
      status: "succeeded",
    });
    await enqueueModelCatalogHeartbeat(
      env,
      {
        job: message.job,
        reason: pendingFeishuMirror ? "mirror_continue" : "self_reschedule",
        attempt: 1,
      },
      nextDelaySeconds,
    );
    return { processed: true, nextDelaySeconds };
  } catch (error) {
    const retryDelaySeconds =
      classifyControlPlaneError(error) === "hard"
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
    await enqueueModelCatalogHeartbeat(
      env,
      {
        job: message.job,
        reason: classifyControlPlaneError(error) === "hard" ? "hard_retry" : "retry",
        attempt: message.attempt + 1,
      },
      retryDelaySeconds,
    );
    return { processed: true, retryDelaySeconds };
  }
}

export async function bootstrapModelCatalogHeartbeat(env: Env): Promise<ModelCatalogQueueMessage> {
  return enqueueModelCatalogHeartbeat(
    env,
    {
      job: MODEL_CATALOG_JOB_NAME,
      reason: "bootstrap",
      attempt: 1,
    },
    0,
  );
}

export function isInternalModelCatalogRequest(request: Request): boolean {
  const url = new URL(request.url);
  return url.pathname === "/internal/model-catalog/queue/bootstrap";
}

export function isInternalModelCatalogRequestAuthorized(env: Env, request: Request): boolean {
  const configuredToken = trim((env as Env & Record<string, unknown>).MODAL_INTERNAL_BEARER_TOKEN);
  if (!configuredToken) {
    return false;
  }
  const header = trim(request.headers.get("authorization"));
  return header === `Bearer ${configuredToken}`;
}

export function buildModelCatalogQueueBootstrapResponse(message: ModelCatalogQueueMessage): Record<string, JsonValue> {
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

export { classifyControlPlaneError, normalizeQueueMessage, shouldForceImmediateRun };
