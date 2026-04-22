import type { Env } from "../../runtime";
import { getTenantAccessToken } from "./auth";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function getApiBase(env: Env): string {
  return trim(env.FEISHU_API_BASE) || "https://open.feishu.cn";
}

export async function feishuApi(
  env: Env,
  method: string,
  path: string,
  init: {
    query?: URLSearchParams;
    body?: BodyInit | null;
    headers?: HeadersInit;
  } = {},
): Promise<Response> {
  const token = await getTenantAccessToken(env);
  const url = new URL(`${getApiBase(env)}${path}`);
  if (init.query) {
    url.search = init.query.toString();
  }
  return fetch(url.toString(), {
    method,
    headers: {
      authorization: `Bearer ${token}`,
      ...(init.headers ?? {}),
    },
    body: init.body ?? null,
  });
}
