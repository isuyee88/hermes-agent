import type { OutboundMessageCorrelationRef } from "../contracts/gateway";

const MAX_OUTBOUND_MESSAGE_REF_ITEMS = 128;
const MAX_OUTBOUND_MESSAGE_REF_AGE_MS = 72 * 3600 * 1000;

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function toEpochMs(value: string): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function trimOutboundMessageCorrelationRef(
  value: OutboundMessageCorrelationRef,
  fallbackCreatedAt = new Date().toISOString(),
): OutboundMessageCorrelationRef {
  return {
    send_message_id: trim(value.send_message_id),
    correlation_id: trim(value.correlation_id),
    session_key: trim(value.session_key),
    event_id: trim(value.event_id),
    source_message_id: trim(value.source_message_id),
    kind: trim(value.kind),
    created_at: trim(value.created_at) || fallbackCreatedAt,
  };
}

export function normalizeOutboundMessageCorrelationRefs(value: unknown): OutboundMessageCorrelationRef[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is OutboundMessageCorrelationRef => Boolean(item && typeof item === "object" && !Array.isArray(item)))
    .map((item) => trimOutboundMessageCorrelationRef(item))
    .filter((item) => !!item.send_message_id);
}

export function compactOutboundMessageCorrelationRefs(
  refs: OutboundMessageCorrelationRef[],
  nowMs = Date.now(),
): OutboundMessageCorrelationRef[] {
  const deduped = new Map<string, OutboundMessageCorrelationRef>();
  for (const candidate of refs) {
    const normalized = trimOutboundMessageCorrelationRef(candidate);
    if (!normalized.send_message_id) {
      continue;
    }
    const createdAtMs = toEpochMs(normalized.created_at);
    if (createdAtMs > 0 && nowMs - createdAtMs > MAX_OUTBOUND_MESSAGE_REF_AGE_MS) {
      continue;
    }
    deduped.set(normalized.send_message_id, normalized);
  }
  return Array.from(deduped.values())
    .sort((left, right) => toEpochMs(left.created_at) - toEpochMs(right.created_at))
    .slice(-MAX_OUTBOUND_MESSAGE_REF_ITEMS);
}

export function resolveOutboundMessageCorrelationRefs(
  refs: OutboundMessageCorrelationRef[],
  messageIds: string[],
): {
  refs: OutboundMessageCorrelationRef[];
  unmatched_message_ids: string[];
} {
  const normalizedMessageIds = Array.from(new Set(messageIds.map((item) => trim(item)).filter(Boolean)));
  if (normalizedMessageIds.length === 0) {
    return { refs: [], unmatched_message_ids: [] };
  }
  const indexed = new Map<string, OutboundMessageCorrelationRef>();
  for (const ref of refs) {
    const normalized = trimOutboundMessageCorrelationRef(ref);
    if (normalized.send_message_id) {
      indexed.set(normalized.send_message_id, normalized);
    }
  }
  const matched: OutboundMessageCorrelationRef[] = [];
  const unmatched: string[] = [];
  for (const messageId of normalizedMessageIds) {
    const resolved = indexed.get(messageId);
    if (resolved) {
      matched.push(resolved);
    } else {
      unmatched.push(messageId);
    }
  }
  return {
    refs: matched,
    unmatched_message_ids: unmatched,
  };
}
