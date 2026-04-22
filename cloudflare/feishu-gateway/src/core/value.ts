import type { JsonValue } from "../contracts/gateway";

export const SITE_PREFETCH_ERROR_TTL_MS = 10 * 60 * 1000;

export function jsonResponse(body: JsonValue, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "content-type": "application/json; charset=utf-8",
      ...(init.headers ?? {}),
    },
  });
}

export function log(event: string, extra: Record<string, unknown>): void {
  console.log(JSON.stringify({ event, ...extra }));
}

export function trim(value: unknown): string {
  return String(value ?? "").trim();
}

export function parseJsonSafely(raw: string): Record<string, JsonValue> {
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? (parsed as Record<string, JsonValue>) : {};
  } catch {
    return {};
  }
}
