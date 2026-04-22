import type {
  Env,
  FeishuNormalizedPayload,
  JsonValue,
  OutboundMessageCorrelationRef,
  PendingReconcileItem,
  SessionHistoryEntry,
  SessionProfileState,
  SessionStateCache,
} from "../runtime";
import {
  compactSessionHistory,
  normalizeSessionHistoryEntries,
  normalizeSessionProfileState,
  trimHistoryContent,
} from "./session-state";
import { normalizeOutboundMessageCorrelationRefs } from "./message-correlation";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => trim(value)).filter(Boolean)));
}

export async function reconcileQueueRequest<T extends JsonValue>(
  env: Env,
  sessionKey: string,
  path: "/peek" | "/enqueue" | "/ack" | "/state_peek" | "/state_update" | "/message_ref_upsert" | "/message_ref_resolve",
  body?: Record<string, JsonValue>,
): Promise<T> {
  const stub = env.FEISHU_RECONCILE_QUEUE.getByName(trim(sessionKey) || "agent:main:feishu:unknown");
  const response = await stub.fetch(`https://reconcile${path}`, {
    method: "POST",
    headers: { "content-type": "application/json; charset=utf-8" },
    body: JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    const text = (await response.text()).slice(0, 400);
    throw new Error(`reconcile_queue_failed:${path}:${response.status}:${text}`);
  }
  return (await response.json()) as T;
}

export async function peekSessionStateCache(env: Env, sessionKey: string): Promise<SessionStateCache> {
  const payload = await reconcileQueueRequest<Record<string, JsonValue>>(env, sessionKey, "/state_peek");
  return {
    profile: normalizeSessionProfileState(payload.profile),
    history: normalizeSessionHistoryEntries(payload.history),
  };
}

export async function updateSessionStateCache(
  env: Env,
  sessionKey: string,
  patch: {
    profile?: SessionProfileState | null;
    historyEntries?: SessionHistoryEntry[];
  },
): Promise<void> {
  const body: Record<string, JsonValue> = {};
  if (patch.profile) {
    body.profile = {
      current_model: trim(patch.profile.current_model),
      current_provider: trim(patch.profile.current_provider),
      current_personality: trim(patch.profile.current_personality),
      route_status_lines: (patch.profile.route_status_lines ?? []).map((line) => trim(line)).filter(Boolean) as unknown as JsonValue,
      updated_at: trim(patch.profile.updated_at) || new Date().toISOString(),
    };
  }
  if (patch.historyEntries && patch.historyEntries.length > 0) {
    body.history_entries = patch.historyEntries.map((entry) => ({
      id: trim(entry.id),
      role: entry.role,
      content: trimHistoryContent(entry.content),
      created_at: trim(entry.created_at) || new Date().toISOString(),
    })) as unknown as JsonValue;
  }
  if (Object.keys(body).length === 0) {
    return;
  }
  await reconcileQueueRequest(env, sessionKey, "/state_update", body);
}

export async function peekPendingReconciles(env: Env, sessionKey: string): Promise<PendingReconcileItem[]> {
  const payload = await reconcileQueueRequest<Record<string, JsonValue>>(env, sessionKey, "/peek");
  const items = Array.isArray(payload.items) ? payload.items : [];
  return items.filter((item): item is PendingReconcileItem => Boolean(item && typeof item === "object"));
}

export async function enqueuePendingReconcile(env: Env, sessionKey: string, item: PendingReconcileItem): Promise<void> {
  await reconcileQueueRequest(env, sessionKey, "/enqueue", { item: item as unknown as JsonValue });
}

export async function upsertOutboundMessageCorrelationRef(
  env: Env,
  sessionKey: string,
  ref: OutboundMessageCorrelationRef,
): Promise<void> {
  await reconcileQueueRequest(env, sessionKey, "/message_ref_upsert", {
    ref: ref as unknown as JsonValue,
  });
}

export async function resolveOutboundMessageCorrelationForRead(
  env: Env,
  sessionKey: string,
  messageIds: string[],
): Promise<{
  refs: OutboundMessageCorrelationRef[];
  unmatched_message_ids: string[];
}> {
  const payload = await reconcileQueueRequest<Record<string, JsonValue>>(env, sessionKey, "/message_ref_resolve", {
    message_ids: uniqueStrings(messageIds) as unknown as JsonValue,
  });
  return {
    refs: normalizeOutboundMessageCorrelationRefs(payload.refs),
    unmatched_message_ids: uniqueStrings(
      Array.isArray(payload.unmatched_message_ids) ? payload.unmatched_message_ids.map((value) => trim(value)) : [],
    ),
  };
}

export async function ackPendingReconciles(env: Env, sessionKey: string, correlationIds: string[]): Promise<void> {
  const ids = uniqueStrings(correlationIds);
  if (ids.length === 0) {
    return;
  }
  await reconcileQueueRequest(env, sessionKey, "/ack", {
    correlation_ids: ids as unknown as JsonValue,
  });
}

export async function fetchModalInternalWithPendingReconciles<T>(
  env: Env,
  path: string,
  normalized: FeishuNormalizedPayload,
  body: Record<string, JsonValue>,
  fetchModalInternal: (
    env: Env,
    path: string,
    body: Record<string, JsonValue>,
    normalized?: FeishuNormalizedPayload,
    pendingReconcile?: JsonValue,
    log?: (event: string, extra: Record<string, unknown>) => void,
  ) => Promise<T>,
  log: (event: string, extra: Record<string, unknown>) => void,
): Promise<T> {
  const pending = await peekPendingReconciles(env, normalized.session_key);
  if (pending.length > 0) {
    log("feishu.reconcile.peek", {
      session_key: normalized.session_key,
      correlation_id: normalized.correlation_id,
      pending_count: pending.length,
    });
  }
  const response = await fetchModalInternal(
    env,
    path,
    {
      ...body,
      session_key: normalized.session_key,
    },
    normalized,
    pending as unknown as JsonValue,
    log,
  );
  if (pending.length > 0) {
    await ackPendingReconciles(
      env,
      normalized.session_key,
      pending.map((item) => item.correlation_id),
    );
    log("feishu.reconcile.ack", {
      session_key: normalized.session_key,
      correlation_id: normalized.correlation_id,
      ack_count: pending.length,
    });
  }
  return response;
}
