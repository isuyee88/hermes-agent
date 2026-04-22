import type { JsonValue, SessionHistoryEntry, SessionProfileState } from "../contracts/gateway";

const MAX_HISTORY_ENTRY_CHARS = 4000;
const MAX_SESSION_HISTORY_ITEMS = 16;
const MAX_SESSION_HISTORY_CHARS = 12000;

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

export function trimHistoryContent(content: string): string {
  return trim(content).slice(0, MAX_HISTORY_ENTRY_CHARS);
}

export function trimSessionHistoryEntry(entry: SessionHistoryEntry): SessionHistoryEntry {
  return {
    id: trim(entry.id),
    role: entry.role === "system" || entry.role === "assistant" ? entry.role : "user",
    content: trimHistoryContent(entry.content),
    created_at: trim(entry.created_at) || new Date().toISOString(),
  };
}

export function compactSessionHistory(entries: SessionHistoryEntry[]): SessionHistoryEntry[] {
  const deduped = new Map<string, SessionHistoryEntry>();
  for (const entry of entries) {
    const normalized = trimSessionHistoryEntry(entry);
    if (!normalized.id || !normalized.content) {
      continue;
    }
    deduped.set(normalized.id, normalized);
  }
  const ordered = Array.from(deduped.values()).sort((left, right) => left.created_at.localeCompare(right.created_at));
  const tail = ordered.slice(-MAX_SESSION_HISTORY_ITEMS);
  let totalChars = 0;
  const kept: SessionHistoryEntry[] = [];
  for (let index = tail.length - 1; index >= 0; index -= 1) {
    const entry = tail[index];
    totalChars += entry.content.length;
    if (kept.length > 0 && totalChars > MAX_SESSION_HISTORY_CHARS) {
      continue;
    }
    kept.push(entry);
  }
  return kept.reverse();
}

export function normalizeSessionProfileState(value: unknown): SessionProfileState | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, JsonValue>;
  const routeStatusLines = Array.isArray(record.route_status_lines)
    ? record.route_status_lines.map((item) => trim(item)).filter(Boolean)
    : [];
  return {
    current_model: trim(record.current_model),
    current_provider: trim(record.current_provider),
    current_personality: trim(record.current_personality),
    route_status_lines: routeStatusLines,
    updated_at: trim(record.updated_at) || new Date().toISOString(),
  };
}

export function normalizeSessionHistoryEntries(value: unknown): SessionHistoryEntry[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const entries = value
    .filter((item): item is Record<string, JsonValue> => Boolean(item && typeof item === "object" && !Array.isArray(item)))
    .map((item) =>
      trimSessionHistoryEntry({
        id: trim(item.id),
        role: trim(item.role) === "system" ? "system" : trim(item.role) === "assistant" ? "assistant" : "user",
        content: trim(item.content),
        created_at: trim(item.created_at) || new Date().toISOString(),
      }),
    );
  return compactSessionHistory(entries);
}
