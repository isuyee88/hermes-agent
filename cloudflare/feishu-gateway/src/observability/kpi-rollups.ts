import type { Env } from "../runtime";
import { getCloudflareAccountId } from "../browser/prefetch/shared";
import { getCloudflareAiGatewayBase } from "../core/runtime-utils";

const RAW_DATASET = "hermes_feishu_gateway_events";
const ROLLUP_DATASET = "hermes_feishu_kpi_rollups";
const SCHEMA_VERSION = "2026-04-23.rollup.v2";
const TEN_MINUTE_CRON = "*/10 * * * *";

type RollupGrain = "10m" | "1h";
type RollupTickSource = "scheduled" | "queue";

type RollupSchedule = {
  cron: string;
  grain: RollupGrain;
  periodMinutes: number;
  windowStartMs: number;
  windowEndMs: number;
  windowKey: string;
  windowStartIso: string;
  windowEndIso: string;
};

type MetricDefinition = {
  key:
    | "webhook_accepted_count"
    | "message_read_count"
    | "cf_ai_exec_done_count"
    | "send_operation_done_count"
    | "send_operation_error_count"
    | "cf_ai_exec_fallback_count"
    | "fallback_once_success_count"
    | "rate_limit_triggered_fallback_count"
    | "cache_eligible_weight"
    | "cache_hit_weight"
    | "capability_match_weight"
    | "preferred_model_weight"
    | "browser_session_weight"
    | "browser_single_ai_call_weight";
  eventName?: string;
  conditions?: string[];
};

type CloudflareSqlSuccess = {
  success?: boolean;
  result?: {
    data?: Array<Record<string, unknown>>;
  };
  data?: Array<Record<string, unknown>>;
  rows?: Array<Record<string, unknown>>;
};

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function resolveRollupBinding(env: Env): AnalyticsEngineDataset | null {
  const dataset = (env as Record<string, unknown>).FEISHU_GATEWAY_KPI_ROLLUPS as AnalyticsEngineDataset | undefined;
  if (!dataset || typeof dataset.writeDataPoint !== "function") {
    return null;
  }
  return dataset;
}

function isTopOfHour(scheduledTime: number): boolean {
  return scheduledTime > 0 && scheduledTime % (60 * 60 * 1000) === 0;
}

function buildRollupSchedule(
  args: {
    scheduledTime: number;
    cron: string;
  },
  periodMinutes: number,
): RollupSchedule {
  const cron = trim(args.cron);
  const periodMs = periodMinutes * 60 * 1000;
  const alignedWindowEndMs = Math.floor(args.scheduledTime / periodMs) * periodMs;
  const windowEndMs = alignedWindowEndMs || args.scheduledTime;
  const windowStartMs = windowEndMs - periodMs;
  const grain: RollupGrain = periodMinutes === 10 ? "10m" : "1h";
  const windowStartIso = new Date(windowStartMs).toISOString();
  const windowEndIso = new Date(windowEndMs).toISOString();
  return {
    cron,
    grain,
    periodMinutes,
    windowStartMs,
    windowEndMs,
    windowKey: `${grain}:${windowStartIso}`,
    windowStartIso,
    windowEndIso,
  };
}

function resolveRollupSchedules(args: {
  source: RollupTickSource;
  scheduledTime: number;
  cron?: string;
}): RollupSchedule[] {
  const cron = trim(args.cron);
  if (args.source === "scheduled" && cron !== TEN_MINUTE_CRON) {
    return [];
  }
  const scheduleCron = args.source === "queue" ? "queue" : cron;
  const schedules = [
    buildRollupSchedule(
      {
        scheduledTime: args.scheduledTime,
        cron: scheduleCron,
      },
      10,
    ),
  ];
  if (isTopOfHour(args.scheduledTime)) {
    schedules.push(
      buildRollupSchedule(
        {
          scheduledTime: args.scheduledTime,
          cron: scheduleCron,
        },
        60,
      ),
    );
  }
  return schedules;
}

function resolveCloudflareAccountId(env: Env): string {
  return getCloudflareAccountId(env, getCloudflareAiGatewayBase(env));
}

function resolveRows(payload: CloudflareSqlSuccess): Array<Record<string, unknown>> {
  if (Array.isArray(payload.result?.data)) {
    return payload.result.data;
  }
  if (Array.isArray(payload.data)) {
    return payload.data;
  }
  if (Array.isArray(payload.rows)) {
    return payload.rows;
  }
  return [];
}

function readNumericCell(row: Record<string, unknown> | undefined, key: string): number {
  if (!row) {
    return 0;
  }
  const value = row[key];
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function rate(numerator: number, denominator: number): number {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) {
    return -1;
  }
  return numerator / denominator;
}

function buildWindowWhereClause(schedule: RollupSchedule): string {
  const startSeconds = Math.floor(schedule.windowStartMs / 1000);
  const endSeconds = Math.floor(schedule.windowEndMs / 1000);
  return `timestamp >= toDateTime(${startSeconds}) AND timestamp < toDateTime(${endSeconds})`;
}

function buildMetricQuery(metric: MetricDefinition, schedule: RollupSchedule): string {
  const clauses = [buildWindowWhereClause(schedule)];
  if (metric.eventName) {
    clauses.push(`index1 = '${metric.eventName}'`);
  }
  clauses.push(...(metric.conditions ?? []));
  return `
SELECT SUM(_sample_interval) AS value
FROM ${RAW_DATASET}
WHERE ${clauses.join(" AND ")}
FORMAT JSON
`.trim();
}

async function executeSqlValueQuery(env: Env, accountId: string, query: string): Promise<number> {
  const token = trim((env as Record<string, unknown>).CLOUDFLARE_API_TOKEN);
  const response = await fetch(`https://api.cloudflare.com/client/v4/accounts/${accountId}/analytics_engine/sql`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "text/plain",
      accept: "application/json",
    },
    body: query,
  });
  const rawText = await response.text();
  let payload: (CloudflareSqlSuccess & { errors?: Array<{ message?: string }> }) | null = null;
  try {
    payload = JSON.parse(rawText) as CloudflareSqlSuccess & {
      errors?: Array<{ message?: string }>;
    };
  } catch {
    if (!response.ok) {
      throw new Error(`analytics_engine_sql_failed:${response.status}:${rawText.slice(0, 200)}`);
    }
    throw new Error(`analytics_engine_sql_invalid_json:${rawText.slice(0, 200)}`);
  }
  if (!response.ok || payload.success === false) {
    const firstError = Array.isArray(payload.errors) ? payload.errors[0] : undefined;
    throw new Error(`analytics_engine_sql_failed:${response.status}:${firstError?.message ?? "unknown"}`);
  }
  return readNumericCell(resolveRows(payload)[0], "value");
}

async function hasExistingRollup(env: Env, accountId: string, schedule: RollupSchedule): Promise<boolean> {
  const query = `
SELECT SUM(_sample_interval) AS value
FROM ${ROLLUP_DATASET}
WHERE index1 = '${schedule.grain}'
  AND blob1 = '${schedule.windowStartIso}'
  AND blob2 = '${schedule.windowEndIso}'
  AND blob6 = '${SCHEMA_VERSION}'
  AND timestamp > NOW() - INTERVAL '7' DAY
FORMAT JSON
`.trim();
  return (await executeSqlValueQuery(env, accountId, query)) > 0;
}

function metricDefinitions(): MetricDefinition[] {
  return [
    { key: "webhook_accepted_count", eventName: "feishu.webhook.accepted" },
    { key: "message_read_count", eventName: "feishu.message_read.accepted" },
    { key: "cf_ai_exec_done_count", eventName: "feishu.cf_ai_exec.done" },
    { key: "send_operation_done_count", eventName: "feishu.send.operation.done" },
    { key: "send_operation_error_count", eventName: "feishu.send.operation.error" },
    { key: "cf_ai_exec_fallback_count", eventName: "feishu.cf_ai_exec.fallback" },
    { key: "fallback_once_success_count", eventName: "feishu.cf_ai_exec.fallback.done" },
    {
      key: "rate_limit_triggered_fallback_count",
      eventName: "feishu.cf_ai_exec.fallback",
      conditions: ["blob18 = 'rate_limited'"],
    },
    {
      key: "cache_eligible_weight",
      eventName: "feishu.cf_ai_exec.done",
      conditions: ["double5 = 1"],
    },
    {
      key: "cache_hit_weight",
      eventName: "feishu.cf_ai_exec.done",
      conditions: ["double5 = 1", "blob10 = 'HIT'"],
    },
    {
      key: "capability_match_weight",
      eventName: "feishu.cf_ai_exec.done",
      conditions: ["double6 = 1"],
    },
    {
      key: "preferred_model_weight",
      eventName: "feishu.cf_ai_exec.done",
      conditions: ["double7 = 1"],
    },
    {
      key: "browser_session_weight",
      eventName: "feishu.cf_ai_exec.done",
      conditions: ["double8 = 1"],
    },
    {
      key: "browser_single_ai_call_weight",
      eventName: "feishu.cf_ai_exec.done",
      conditions: ["double8 = 1", "double3 = 1"],
    },
  ];
}

export async function runFeishuKpiRollupTick(
  env: Env,
  args: {
    source: RollupTickSource;
    scheduledTime: number;
    cron?: string;
  },
): Promise<void> {
  const schedules = resolveRollupSchedules(args);
  if (schedules.length === 0) {
    console.log(
      JSON.stringify({
        event: "feishu.kpi_rollup.skipped",
        reason: "unsupported_cron",
        source: args.source,
        cron: args.cron,
        scheduled_time: args.scheduledTime,
      }),
    );
    return;
  }

  const rollupDataset = resolveRollupBinding(env);
  if (!rollupDataset) {
    console.log(
      JSON.stringify({
        event: "feishu.kpi_rollup.skipped",
        reason: "rollup_dataset_binding_missing",
        source: args.source,
        cron: args.cron,
      }),
    );
    return;
  }

  const token = trim((env as Record<string, unknown>).CLOUDFLARE_API_TOKEN);
  const accountId = resolveCloudflareAccountId(env);
  if (!token || !accountId) {
    console.log(
      JSON.stringify({
        event: "feishu.kpi_rollup.skipped",
        reason: !token ? "cloudflare_api_token_missing" : "cloudflare_account_id_missing",
        source: args.source,
        cron: args.cron,
      }),
    );
    return;
  }

  for (const schedule of schedules) {
    if (await hasExistingRollup(env, accountId, schedule)) {
      console.log(
        JSON.stringify({
          event: "feishu.kpi_rollup.skipped",
          reason: "window_already_aggregated",
          cron: schedule.cron,
          grain: schedule.grain,
          window_key: schedule.windowKey,
        }),
      );
      continue;
    }

    const metrics = Object.fromEntries(
      await Promise.all(
        metricDefinitions().map(async (metric) => [
          metric.key,
          await executeSqlValueQuery(env, accountId, buildMetricQuery(metric, schedule)),
        ]),
      ),
    ) as Record<MetricDefinition["key"], number>;

    const sendOperationTotal = metrics.send_operation_done_count + metrics.send_operation_error_count;
    const overallCacheHitRate = rate(metrics.cache_hit_weight, metrics.cache_eligible_weight);
    const capabilityMatchRate = rate(metrics.capability_match_weight, metrics.cf_ai_exec_done_count);
    const preferredModelSelectionAccuracy = rate(metrics.preferred_model_weight, metrics.cf_ai_exec_done_count);
    const browserSingleAiCallCompletionRate = rate(
      metrics.browser_single_ai_call_weight,
      metrics.browser_session_weight,
    );
    const fallbackOnceSuccessRate = rate(metrics.fallback_once_success_count, metrics.cf_ai_exec_fallback_count);
    const sendOperationErrorRate = rate(metrics.send_operation_error_count, sendOperationTotal);

    rollupDataset.writeDataPoint({
      indexes: [schedule.grain],
      blobs: [
        schedule.windowStartIso,
        schedule.windowEndIso,
        schedule.windowKey,
        schedule.cron,
        RAW_DATASET,
        SCHEMA_VERSION,
        "feishu-kpi-rollup",
      ],
      doubles: [
        metrics.webhook_accepted_count,
        metrics.message_read_count,
        metrics.cf_ai_exec_done_count,
        metrics.send_operation_done_count,
        metrics.send_operation_error_count,
        sendOperationErrorRate,
        metrics.cf_ai_exec_fallback_count,
        metrics.fallback_once_success_count,
        fallbackOnceSuccessRate,
        metrics.rate_limit_triggered_fallback_count,
        metrics.cache_eligible_weight,
        metrics.cache_hit_weight,
        overallCacheHitRate,
        metrics.capability_match_weight,
        capabilityMatchRate,
        metrics.preferred_model_weight,
        preferredModelSelectionAccuracy,
        metrics.browser_session_weight,
        metrics.browser_single_ai_call_weight,
        browserSingleAiCallCompletionRate,
      ],
    });

    console.log(
      JSON.stringify({
        event: "feishu.kpi_rollup.completed",
        source: args.source,
        cron: schedule.cron,
        grain: schedule.grain,
        window_key: schedule.windowKey,
        window_start: schedule.windowStartIso,
        window_end: schedule.windowEndIso,
        metrics: {
          cf_ai_exec_done_count: metrics.cf_ai_exec_done_count,
          overall_cache_hit_rate: overallCacheHitRate,
          fallback_once_success_rate: fallbackOnceSuccessRate,
          rate_limit_triggered_fallback_count: metrics.rate_limit_triggered_fallback_count,
        },
      }),
    );
  }
}

export async function runFeishuKpiRollupScheduledTick(
  env: Env,
  controller: ScheduledController,
): Promise<void> {
  return runFeishuKpiRollupTick(env, {
    source: "scheduled",
    cron: controller.cron,
    scheduledTime: controller.scheduledTime,
  });
}
