type TenantTokenCache = {
  token: string;
  expiresAt: number;
};

let tenantTokenCache: TenantTokenCache | null = null;

export function readTenantTokenCache(now = Date.now()): string | null {
  if (tenantTokenCache && tenantTokenCache.expiresAt > now + 30_000) {
    return tenantTokenCache.token;
  }
  return null;
}

export function writeTenantTokenCache(token: string, expiresAt: number): void {
  tenantTokenCache = { token, expiresAt };
}
