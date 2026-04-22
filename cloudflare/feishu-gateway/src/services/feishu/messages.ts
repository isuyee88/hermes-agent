import type { Env, FeishuNormalizedPayload, JsonValue, OutboundMessageCorrelationRef } from "../../runtime";
import { writeFeishuAnalyticsEvent } from "../../observability/analytics-engine";
import { feishuApi } from "./client";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

type FeishuSendFailureClass =
  | "target_not_in_chat"
  | "cross_app_identity"
  | "permission_denied"
  | "rate_limited"
  | "invalid_receive_target"
  | "unknown_send_failure";

function readRecord(parent: Record<string, JsonValue>, key: string): Record<string, JsonValue> {
  const value = parent[key];
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, JsonValue>)
    : {};
}

function readString(parent: Record<string, JsonValue>, key: string): string {
  return trim(parent[key]);
}

export function classifyFeishuSendFailure(options: {
  status: number;
  code: number;
  message: string;
  receiveId: string;
  receiveIdType: string;
}): FeishuSendFailureClass {
  const message = trim(options.message).toLowerCase();
  const receiveId = trim(options.receiveId);
  const receiveIdType = trim(options.receiveIdType).toLowerCase();

  if (message.includes("bot/user can not be out of the chat")) {
    return "target_not_in_chat";
  }
  if (message.includes("open_id cross app")) {
    return "cross_app_identity";
  }
  if (message.includes("permission") || message.includes("no permission") || message.includes("forbidden")) {
    return "permission_denied";
  }
  if (options.status === 429 || message.includes("rate limit") || message.includes("too many requests")) {
    return "rate_limited";
  }
  if (!receiveId || (receiveIdType === "chat_id" && !receiveId.startsWith("oc_"))) {
    return "invalid_receive_target";
  }
  return "unknown_send_failure";
}

function extractFeishuSendErrorMeta(error: unknown): Record<string, unknown> {
  if (!error || typeof error !== "object") {
    return {};
  }
  const meta = error as Record<string, unknown>;
  const feishuStatus = typeof meta.feishuStatus === "number" ? meta.feishuStatus : undefined;
  const feishuCode = typeof meta.feishuCode === "number" ? meta.feishuCode : undefined;
  const feishuErrorClass = trim(meta.feishuErrorClass);
  const feishuMessage = trim(meta.feishuMessage);
  const feishuReceiveId = trim(meta.feishuReceiveId);
  const feishuReceiveIdType = trim(meta.feishuReceiveIdType);
  return {
    feishu_status: feishuStatus,
    feishu_code: feishuCode,
    feishu_error_class: feishuErrorClass || undefined,
    feishu_error_message: feishuMessage || undefined,
    feishu_receive_id: feishuReceiveId || undefined,
    feishu_receive_id_type: feishuReceiveIdType || undefined,
  };
}

export async function sendFeishuMessage(
  env: Env,
  options: {
    receiveId: string;
    receiveIdType?: string;
    msgType: "text" | "interactive" | "image" | "file" | "audio" | "media" | "post";
    content: Record<string, JsonValue> | string;
  },
): Promise<Record<string, JsonValue>> {
  const query = new URLSearchParams({
    receive_id_type: options.receiveIdType ?? "chat_id",
  });
  const payload = {
    receive_id: options.receiveId,
    msg_type: options.msgType,
    content: typeof options.content === "string" ? options.content : JSON.stringify(options.content),
    uuid: crypto.randomUUID(),
  };
  const response = await feishuApi(env, "POST", "/open-apis/im/v1/messages", {
    query,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = (await response.json()) as Record<string, JsonValue>;
  const rawCode = Number(body.code ?? 0);
  const code = Number.isFinite(rawCode) ? rawCode : -1;
  if (!response.ok || code !== 0) {
    const message = trim(body.msg) || trim(body.message) || response.statusText || "unknown_feishu_error";
    const errorClass = classifyFeishuSendFailure({
      status: response.status,
      code,
      message,
      receiveId: options.receiveId,
      receiveIdType: options.receiveIdType ?? "chat_id",
    });
    const error = new Error(
      `send_message_failed:${response.status}:${message} [class=${errorClass};code=${code};receive_id_type=${options.receiveIdType ?? "chat_id"}]`,
    ) as Error & Record<string, unknown>;
    error.feishuStatus = response.status;
    error.feishuCode = code;
    error.feishuMessage = message;
    error.feishuErrorClass = errorClass;
    error.feishuReceiveId = options.receiveId;
    error.feishuReceiveIdType = options.receiveIdType ?? "chat_id";
    throw error;
  }
  return body;
}

export async function sendLoggedFeishuOperation(
  env: Env,
  normalized: FeishuNormalizedPayload,
  kind: string,
  options: {
    receiveId: string;
    receiveIdType?: string;
    msgType: "text" | "interactive" | "image" | "file" | "audio" | "media" | "post";
    content: Record<string, JsonValue> | string;
  },
  log: (event: string, extra: Record<string, unknown>) => void,
  persistMessageRef?: (
    env: Env,
    sessionKey: string,
    ref: OutboundMessageCorrelationRef,
  ) => Promise<void>,
): Promise<Record<string, JsonValue>> {
  const startedAt = Date.now();
  log("feishu.send.operation.start", {
    correlation_id: normalized.correlation_id,
    event_id: normalized.event_id,
    kind,
    msg_type: options.msgType,
    receive_id: options.receiveId,
    receive_id_type: options.receiveIdType ?? "chat_id",
  });
  try {
    const response = await sendFeishuMessage(env, options);
    const data = readRecord(response, "data");
    const sendMessageId = readString(data, "message_id");
    if (sendMessageId && persistMessageRef) {
      try {
        await persistMessageRef(env, normalized.session_key, {
          send_message_id: sendMessageId,
          correlation_id: normalized.correlation_id,
          session_key: normalized.session_key,
          event_id: normalized.event_id,
          source_message_id: normalized.message_id,
          kind: trim(kind),
          created_at: new Date().toISOString(),
        });
      } catch (error) {
        log("feishu.send.operation.index_error", {
          correlation_id: normalized.correlation_id,
          session_key: normalized.session_key,
          event_id: normalized.event_id,
          send_message_id: sendMessageId,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
    const donePayload = {
      correlation_id: normalized.correlation_id,
      session_key: normalized.session_key,
      event_id: normalized.event_id,
      kind,
      msg_type: options.msgType,
      receive_id: options.receiveId,
      receive_id_type: options.receiveIdType ?? "chat_id",
      send_message_id: sendMessageId,
      send_elapsed_ms: Date.now() - startedAt,
      status: "ok",
    };
    log("feishu.send.operation.done", donePayload);
    writeFeishuAnalyticsEvent(env, "feishu.send.operation.done", donePayload);
    return response;
  } catch (error) {
    log("feishu.send.operation.error", {
      correlation_id: normalized.correlation_id,
      session_key: normalized.session_key,
      event_id: normalized.event_id,
      kind,
      msg_type: options.msgType,
      receive_id: options.receiveId,
      receive_id_type: options.receiveIdType ?? "chat_id",
      send_elapsed_ms: Date.now() - startedAt,
      error: error instanceof Error ? error.message : String(error),
      ...extractFeishuSendErrorMeta(error),
      status: "error",
    });
    throw error;
  }
}

export async function deleteFeishuMessage(env: Env, messageId: string): Promise<void> {
  if (!trim(messageId)) {
    return;
  }
  const response = await feishuApi(env, "DELETE", `/open-apis/im/v1/messages/${messageId}`);
  if (!response.ok) {
    const body = (await response.text()).slice(0, 400);
    throw new Error(`delete_message_failed:${response.status}:${body}`);
  }
}

export async function addAckReaction(env: Env, messageId: string): Promise<void> {
  if (!trim(messageId)) {
    return;
  }
  const response = await feishuApi(env, "POST", `/open-apis/im/v1/messages/${messageId}/reactions`, {
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      reaction_type: {
        emoji_type: trim(env.FEISHU_ACK_REACTION_EMOJI) || "OK",
      },
    }),
  });
  if (!response.ok) {
    const body = (await response.text()).slice(0, 300);
    throw new Error(`ack_reaction_failed:${response.status}:${body}`);
  }
}
