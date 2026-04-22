import { buildModalRequestEnvelope, unwrapModalResponseEnvelope } from "../../contracts/feishu-internal";
import type { Env, FeishuNormalizedPayload, JsonValue } from "../../runtime";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

/**
 * Modal internal HTTP client kept version-aware but route-path compatible.
 */
export async function fetchModalInternal<T>(
  env: Env,
  path: string,
  body: Record<string, JsonValue>,
  normalized?: FeishuNormalizedPayload,
  pendingReconcile: JsonValue = [],
  log?: (event: string, extra: Record<string, unknown>) => void,
): Promise<T> {
  const requestStartedAt = Date.now();
  const envelope = buildModalRequestEnvelope(path, normalized, body, pendingReconcile);
  const requestPayload = {
    ...(typeof envelope.gateway_meta?.legacy_payload === "object" && envelope.gateway_meta.legacy_payload !== null
      ? (envelope.gateway_meta.legacy_payload as Record<string, JsonValue>)
      : body),
  };

  const requestBody = {
    __path: path,
    payload: requestPayload,
  };

  if (log && normalized) {
    log("modal.request.start", {
      correlation_id: trim(normalized.correlation_id),
      event_id: trim(normalized.event_id),
      session_key: trim(normalized.session_key),
      modal_path: path,
      modal_url: trim(env.MODAL_INTERNAL_BASE_URL),
      has_pending_reconcile: Array.isArray(pendingReconcile) ? pendingReconcile.length > 0 : false,
      body_keys: Object.keys(body),
    });
  }

  const response = await fetch(trim(env.MODAL_INTERNAL_BASE_URL), {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.MODAL_INTERNAL_BEARER_TOKEN}`,
      "content-type": "application/json",
      "x-hermes-gateway-hop": "cloudflare-worker",
      "x-hermes-gateway-script": "hermes-feishu-gateway",
      "x-hermes-correlation-id": trim(normalized?.correlation_id),
      "x-hermes-event-id": trim(normalized?.event_id),
      "x-hermes-session-key": trim(normalized?.session_key),
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const text = (await response.text()).slice(0, 800);
    if (log && normalized) {
      log("modal.request.error", {
        correlation_id: trim(normalized.correlation_id),
        event_id: trim(normalized.event_id),
        modal_path: path,
        modal_status: response.status,
        modal_elapsed_ms: Date.now() - requestStartedAt,
        modal_response_preview: text.slice(0, 200),
        error_type: `modal_internal_failed:${response.status}`,
      });
    }
    throw new Error(`modal_internal_failed:${response.status}:${text}`);
  }

  const rawJson = await response.json();
  const unwrapped = unwrapModalResponseEnvelope<T>(rawJson);

  if (log && normalized) {
    const rawRecord = typeof rawJson === "object" && rawJson !== null ? rawJson as Record<string, unknown> : {};
    const rawSendPlan = rawRecord.send_plan;
    const rawResult = rawRecord.result;
    
    let unwrappedSendPlanLength = 0;
    let unwrappedActionPlanLength = 0;
    let unwrappedFinalResponse = "";
    let unwrappedExecutionMode = "";
    
    if (typeof unwrapped === "object" && unwrapped !== null) {
      const u = unwrapped as Record<string, unknown>;
      unwrappedSendPlanLength = Array.isArray(u.send_plan) ? u.send_plan.length : Array.isArray(u.action_plan) ? u.action_plan.length : 0;
      unwrappedActionPlanLength = Array.isArray(u.action_plan) ? u.action_plan.length : 0;
      unwrappedFinalResponse = typeof u.final_response === "string" ? u.final_response.slice(0, 100) : "";
      unwrappedExecutionMode = typeof u.execution_mode === "string" ? u.execution_mode : "";
    }
    
    log("modal.response.done", {
      correlation_id: trim(normalized.correlation_id),
      event_id: trim(normalized.event_id),
      modal_path: path,
      modal_status: response.status,
      modal_elapsed_ms: Date.now() - requestStartedAt,
      raw_contract_version: trim(rawRecord.contract_version),
      raw_send_plan_type: Array.isArray(rawSendPlan) ? "array" : typeof rawSendPlan === "object" && rawSendPlan !== null ? "object" : "undefined",
      raw_send_plan_length: Array.isArray(rawSendPlan) ? rawSendPlan.length : "N/A",
      unwrapped_send_plan_length: unwrappedSendPlanLength,
      unwrapped_action_plan_length: unwrappedActionPlanLength,
      unwrapped_final_response_preview: unwrappedFinalResponse,
      unwrapped_execution_mode: unwrappedExecutionMode,
      raw_result_status: typeof rawResult === "object" && rawResult !== null ? trim((rawResult as Record<string, unknown>).status) : "N/A",
    });
  }

  return unwrapped;
}

export function classifyModalInternalError(error: unknown): { retryClass: string; retryable: boolean } {
  const message = error instanceof Error ? error.message : String(error);
  if (/modal_internal_failed:(429|502|503|504|524):/i.test(message) || /timeout/i.test(message)) {
    return { retryClass: "retryable_upstream", retryable: true };
  }
  if (/modal_internal_failed:401:/i.test(message)) {
    return { retryClass: "auth_error", retryable: false };
  }
  if (/modal_internal_failed:400:/i.test(message)) {
    return { retryClass: "bad_request", retryable: false };
  }
  return { retryClass: "fatal", retryable: false };
}

export async function fetchModalResultFile(env: Env, token: string): Promise<{ bytes: ArrayBuffer; fileName: string }> {
  const response = await fetch(`${trim(env.MODAL_INTERNAL_BASE_URL)}/internal/feishu/result-file/${encodeURIComponent(token)}`, {
    headers: {
      authorization: `Bearer ${env.MODAL_INTERNAL_BEARER_TOKEN}`,
    },
  });
  if (!response.ok) {
    const text = (await response.text()).slice(0, 400);
    throw new Error(`modal_result_file_failed:${response.status}:${text}`);
  }
  const disposition = response.headers.get("content-disposition") ?? "";
  const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
  return {
    bytes: await response.arrayBuffer(),
    fileName: filenameMatch?.[1] ?? "artifact.bin",
  };
}
