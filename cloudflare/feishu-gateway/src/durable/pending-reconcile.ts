import type { FeishuNormalizedPayload, JsonValue, ModalInternalResponse, PendingReconcileItem } from "../contracts/gateway";

const MAX_PENDING_RECONCILE_ITEMS = 50;
const MAX_PENDING_RECONCILE_BYTES = 64 * 1024;
const MAX_PENDING_RECONCILE_TEXT_LENGTH = 4000;

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = trim(value);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

export function summarizeOperationsForHermes(sendPlan: Array<Record<string, JsonValue>>): {
  assistantText: string;
  operationKinds: string[];
} {
  const operationKinds = uniqueStrings(sendPlan.map((item) => trim(item.kind).toLowerCase()).filter(Boolean));
  const textParts = uniqueStrings(sendPlan.map((item) => trim(item.content)).filter(Boolean));
  const placeholderMap: Record<string, string> = {
    image: "[Sent image]",
    image_url: "[Sent image]",
    image_file: "[Sent image]",
    file: "[Sent file]",
    document_file: "[Sent file]",
    audio: "[Sent audio]",
    audio_file: "[Sent audio]",
    media: "[Sent media]",
    video_file: "[Sent media]",
    interactive: "[Sent card]",
  };
  const placeholders = uniqueStrings(operationKinds.map((kind) => placeholderMap[kind] ?? "").filter(Boolean));
  return {
    assistantText: uniqueStrings([...textParts, ...placeholders]).join("\n").trim(),
    operationKinds,
  };
}

export function buildPendingReconcile(
  normalized: FeishuNormalizedPayload,
  internal: ModalInternalResponse,
): PendingReconcileItem | null {
  const sendPlan = Array.isArray(internal.action_plan)
    ? internal.action_plan
    : Array.isArray(internal.send_plan)
      ? internal.send_plan
      : [];
  const summary = summarizeOperationsForHermes(sendPlan);
  const finalResponse = trim(internal.final_response);
  const assistantText = trim(summary.assistantText || finalResponse);
  if (!assistantText && summary.operationKinds.length === 0) {
    return null;
  }
  return {
    correlation_id: normalized.correlation_id,
    assistant_text: assistantText,
    final_response: finalResponse,
    operation_kinds: summary.operationKinds,
    delivered_at: new Date().toISOString(),
  };
}

export function trimPendingReconcileItem(item: PendingReconcileItem): PendingReconcileItem {
  return {
    ...item,
    assistant_text: trim(item.assistant_text).slice(0, MAX_PENDING_RECONCILE_TEXT_LENGTH),
    final_response: trim(item.final_response).slice(0, MAX_PENDING_RECONCILE_TEXT_LENGTH),
    operation_kinds: uniqueStrings(Array.isArray(item.operation_kinds) ? item.operation_kinds : []),
    delivered_at: trim(item.delivered_at) || new Date().toISOString(),
  };
}

function estimatePendingReconcileBytes(items: PendingReconcileItem[]): number {
  return new TextEncoder().encode(JSON.stringify(items)).length;
}

function buildPendingReconcileSummary(items: PendingReconcileItem[]): PendingReconcileItem | null {
  if (items.length === 0) {
    return null;
  }
  const operationKinds = uniqueStrings(items.flatMap((item) => item.operation_kinds || []));
  const references = uniqueStrings(items.map((item) => trim(item.correlation_id)).filter(Boolean)).slice(0, 5);
  const assistantText = [
    `[Pending reconcile summary] ${items.length} earlier Cloudflare-delivered results were compacted.`,
    operationKinds.length > 0 ? `Kinds: ${operationKinds.join(", ")}` : "",
    references.length > 0 ? `Refs: ${references.join(", ")}` : "",
  ]
    .filter(Boolean)
    .join("\n");
  return {
    correlation_id: `summary:${Date.now()}`,
    assistant_text: assistantText.slice(0, MAX_PENDING_RECONCILE_TEXT_LENGTH),
    final_response: "",
    operation_kinds: operationKinds,
    delivered_at: new Date().toISOString(),
  };
}

export function compactPendingReconcileItems(items: PendingReconcileItem[]): {
  items: PendingReconcileItem[];
  compacted: number;
} {
  const nextItems = items.map((item) => trimPendingReconcileItem(item));
  const removed: PendingReconcileItem[] = [];
  while (
    nextItems.length > MAX_PENDING_RECONCILE_ITEMS ||
    estimatePendingReconcileBytes(nextItems) > MAX_PENDING_RECONCILE_BYTES
  ) {
    if (nextItems.length <= 1) {
      break;
    }
    const dropped = nextItems.shift();
    if (dropped) {
      removed.push(dropped);
    }
  }
  if (removed.length > 0) {
    const summary = buildPendingReconcileSummary(removed);
    if (summary) {
      nextItems.unshift(summary);
    }
    while (
      nextItems.length > MAX_PENDING_RECONCILE_ITEMS ||
      estimatePendingReconcileBytes(nextItems) > MAX_PENDING_RECONCILE_BYTES
    ) {
      nextItems.pop();
    }
  }
  return {
    items: nextItems,
    compacted: removed.length,
  };
}
