import type { Env, JsonValue } from "../../runtime";
import { readTenantTokenCache, writeTenantTokenCache } from "./token-cache";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function getApiBase(env: Env): string {
  return trim(env.FEISHU_API_BASE) || "https://open.feishu.cn";
}

export async function getTenantAccessToken(env: Env): Promise<string> {
  const now = Date.now();
  const cached = readTenantTokenCache(now);
  if (cached) {
    return cached;
  }
  const appId = trim(env.FEISHU_APP_ID);
  const appSecret = trim(env.FEISHU_APP_SECRET);
  const response = await fetch(`${getApiBase(env)}/open-apis/auth/v3/tenant_access_token/internal`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      app_id: appId,
      app_secret: appSecret,
    }),
  });
  const payload = (await response.json()) as Record<string, JsonValue>;
  if (!response.ok || Number(payload.code ?? 0) !== 0) {
    throw new Error(
      `tenant_access_token_failed:${response.status}:${trim(payload.msg)}:app_id_prefix=${appId.slice(0, 8)}:app_id_len=${appId.length}:app_secret_len=${appSecret.length}`,
    );
  }
  const token = trim(payload.tenant_access_token);
  const expire = Number(payload.expire ?? 7200);
  writeTenantTokenCache(token, now + Math.max(60, expire - 60) * 1000);
  return token;
}
