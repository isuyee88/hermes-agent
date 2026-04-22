import { trim } from "./value";

const DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL =
  "https://gateway.ai.cloudflare.com/v1/d1215a30b84b673ef0367010b0e78c10/affiliate-manager/compat";

export function isFreeishModelName(value: unknown): boolean {
  const normalized = trim(value).toLowerCase();
  if (!normalized) {
    return false;
  }
  return normalized === "free" || normalized === "openrouter/free" || normalized.includes(":free");
}

export function getCloudflareAiGatewayBase(env: { CLOUDFLARE_AI_GATEWAY_BASE_URL?: string }): string {
  const configured = trim(env.CLOUDFLARE_AI_GATEWAY_BASE_URL) || DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL;
  return configured.replace(/\/chat\/completions$/i, "").replace(/\/$/, "");
}

export function getCloudflareAiGatewayRoot(env: { CLOUDFLARE_AI_GATEWAY_BASE_URL?: string }): string {
  return getCloudflareAiGatewayBase(env).replace(/\/compat$/i, "").replace(/\/$/, "");
}

export function parsePositiveInt(
  value: unknown,
  fallback: number,
  min = 1,
  max = Number.MAX_SAFE_INTEGER,
): number {
  const parsed = Number.parseInt(trim(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, parsed));
}

export function parseBoolean(value: unknown, fallback = false): boolean {
  const normalized = trim(value).toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

export function toWorkflowInstanceId(correlationId: string): string {
  const normalized = trim(correlationId).replace(/[^a-zA-Z0-9_-]/g, "_");
  if (normalized) {
    return normalized.slice(0, 64);
  }
  return `feishu_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}
