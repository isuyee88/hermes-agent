import type { Env } from "../runtime";

const MAX_INDEX_LENGTH = 64;
const MAX_BLOB_LENGTH = 160;

function trimValue(value: unknown, maxLength = MAX_BLOB_LENGTH): string | null {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return null;
  }
  return normalized.slice(0, maxLength);
}

function numericValue(value: unknown, fallback = -1): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nonNegativeNumericValue(value: unknown, fallback = 0): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  return value >= 0 ? value : fallback;
}

function boolValue(value: unknown): number {
  return typeof value === "boolean" ? (value ? 1 : 0) : -1;
}

function resolveElapsedMs(payload: Record<string, unknown>): number {
  return numericValue(
    payload.cf_ai_exec_elapsed_ms ??
      payload.cf_send_elapsed_ms ??
      payload.send_elapsed_ms ??
      payload.ack_elapsed_ms,
  );
}

export function writeFeishuAnalyticsEvent(
  env: Env,
  eventName: string,
  payload: Record<string, unknown> = {},
): boolean {
  const dataset = (env as Record<string, unknown>).FEISHU_GATEWAY_ANALYTICS as
    | AnalyticsEngineDataset
    | undefined;
  if (!dataset || typeof dataset.writeDataPoint !== "function") {
    return false;
  }

  try {
    dataset.writeDataPoint({
      indexes: [trimValue(eventName, MAX_INDEX_LENGTH)],
      blobs: [
        trimValue(payload.correlation_id),
        trimValue(payload.session_key),
        trimValue(payload.event_id),
        trimValue(payload.message_id),
        trimValue(payload.request_class),
        trimValue(payload.route_family),
        trimValue(payload.gateway_route_name),
        trimValue(payload.provider),
        trimValue(payload.model),
        trimValue(payload.cf_cache_status ?? payload.cache_status),
        trimValue(payload.execution_mode),
        trimValue(payload.lane),
        trimValue(payload.chat_type),
        trimValue(payload.route_decision_reason),
        trimValue(payload.status ?? payload.send_status),
        trimValue(payload.kind),
        trimValue(payload.msg_type),
        trimValue(payload.gateway_error_class),
        trimValue(payload.site_category),
        trimValue(payload.modality_profile),
      ],
      doubles: [
        numericValue(payload.read_time),
        resolveElapsedMs(payload),
        numericValue(payload.ai_call_count),
        boolValue(payload.gateway_eligible),
        boolValue(payload.cache_eligible),
        boolValue(payload.capability_match),
        boolValue(payload.preferred_model_selected),
        boolValue(payload.requires_browser),
        boolValue(payload.requires_tools),
        boolValue(payload.requires_media_hydration),
        boolValue(
          payload.status === "ok" ||
            payload.send_status === "success" ||
            payload.success === true,
        ),
        boolValue(payload.provider_async_eligible),
        boolValue(payload.provider_async_observed),
        numericValue(payload.dfmea_severity),
        nonNegativeNumericValue(payload.cost),
      ],
    });
    return true;
  } catch (error) {
    console.warn(
      JSON.stringify({
        event: "feishu.analytics_engine.write_failed",
        analytics_event: eventName,
        error: error instanceof Error ? error.message : String(error),
      }),
    );
    return false;
  }
}
