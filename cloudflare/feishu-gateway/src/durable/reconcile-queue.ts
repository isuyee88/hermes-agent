import { DurableObject } from "cloudflare:workers";
import type { Env, JsonValue, OutboundMessageCorrelationRef, PendingReconcileItem } from "../runtime";
import {
  compactOutboundMessageCorrelationRefs,
  normalizeOutboundMessageCorrelationRefs,
  resolveOutboundMessageCorrelationRefs,
  trimOutboundMessageCorrelationRef,
} from "./message-correlation";
import { compactPendingReconcileItems, trimPendingReconcileItem } from "./pending-reconcile";
import { compactSessionHistory, normalizeSessionHistoryEntries, normalizeSessionProfileState } from "./session-state";
import {
  jsonResponse,
  parseJsonSafely,
  trim,
} from "../runtime";

/**
 * Durable Object for short-lived reconcile artifacts and session cache hints.
 */
export class FeishuReconcileQueue extends DurableObject<Env> {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const payload = request.method === "POST" ? parseJsonSafely(await request.text()) : {};
    const stored = (await this.ctx.storage.get<PendingReconcileItem[]>("items")) ?? [];
    const items = Array.isArray(stored) ? stored : [];
    const storedMessageRefs = normalizeOutboundMessageCorrelationRefs(await this.ctx.storage.get("message_refs"));
    const storedProfile = normalizeSessionProfileState(await this.ctx.storage.get("session_profile"));
    const storedHistory = normalizeSessionHistoryEntries(await this.ctx.storage.get("conversation_history"));

    if (url.pathname === "/peek") {
      return jsonResponse({ items: items as unknown as JsonValue });
    }

    if (url.pathname === "/state_peek") {
      return jsonResponse({
        profile: (storedProfile ?? {}) as unknown as JsonValue,
        history: storedHistory as unknown as JsonValue,
      });
    }

    if (url.pathname === "/message_ref_upsert") {
      const candidate = payload.ref;
      if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
        return jsonResponse({ ok: false, error: "invalid_ref" }, { status: 400 });
      }
      const ref = trimOutboundMessageCorrelationRef(candidate as unknown as OutboundMessageCorrelationRef);
      if (!ref.send_message_id) {
        return jsonResponse({ ok: false, error: "missing_send_message_id" }, { status: 400 });
      }
      const nextRefs = compactOutboundMessageCorrelationRefs([...storedMessageRefs, ref]);
      await this.ctx.storage.put("message_refs", nextRefs);
      return jsonResponse({ ok: true, count: nextRefs.length });
    }

    if (url.pathname === "/message_ref_resolve") {
      const rawMessageIds = Array.isArray(payload.message_ids) ? payload.message_ids : [];
      const resolution = resolveOutboundMessageCorrelationRefs(
        storedMessageRefs,
        rawMessageIds.map((value) => trim(value)),
      );
      return jsonResponse({
        refs: resolution.refs as unknown as JsonValue,
        unmatched_message_ids: resolution.unmatched_message_ids as unknown as JsonValue,
      });
    }

    if (url.pathname === "/enqueue") {
      const candidate = payload.item;
      if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
        return jsonResponse({ ok: false, error: "invalid_item" }, { status: 400 });
      }
      const item = trimPendingReconcileItem(candidate as unknown as PendingReconcileItem);
      const correlationId = trim(item.correlation_id);
      if (!correlationId) {
        return jsonResponse({ ok: false, error: "missing_correlation_id" }, { status: 400 });
      }
      const nextItems = items.filter((entry) => trim(entry.correlation_id) !== correlationId);
      nextItems.push({
        correlation_id: correlationId,
        assistant_text: item.assistant_text,
        final_response: item.final_response,
        operation_kinds: item.operation_kinds,
        delivered_at: item.delivered_at,
      });
      const compacted = compactPendingReconcileItems(nextItems);
      await this.ctx.storage.put("items", compacted.items);
      return jsonResponse({ ok: true, count: compacted.items.length, compacted: compacted.compacted });
    }

    if (url.pathname === "/ack") {
      const rawIds = Array.isArray(payload.correlation_ids) ? payload.correlation_ids : [];
      const correlationIds = Array.from(new Set(rawIds.map((value) => trim(value)).filter(Boolean)));
      if (correlationIds.length === 0) {
        return jsonResponse({ ok: true, count: items.length });
      }
      const nextItems = items.filter((entry) => !correlationIds.includes(trim(entry.correlation_id)));
      await this.ctx.storage.put("items", nextItems);
      return jsonResponse({ ok: true, count: nextItems.length });
    }

    if (url.pathname === "/state_update") {
      const nextProfile = normalizeSessionProfileState(payload.profile) ?? storedProfile;
      const appendedHistory = normalizeSessionHistoryEntries(payload.history_entries);
      const nextHistory = compactSessionHistory([...storedHistory, ...appendedHistory]);
      if (nextProfile) {
        await this.ctx.storage.put("session_profile", nextProfile);
      }
      await this.ctx.storage.put("conversation_history", nextHistory);
      return jsonResponse({
        ok: true,
        profile_updated: Boolean(nextProfile),
        history_count: nextHistory.length,
      });
    }

    return jsonResponse({ ok: false, error: "not_found" }, { status: 404 });
  }
}
