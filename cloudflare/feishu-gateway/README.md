# Hermes Feishu Gateway

Cloudflare Worker + Workflow ingress for Feishu.

## Secrets

Set these before deploy:

```bash
wrangler secret put FEISHU_APP_ID
wrangler secret put FEISHU_APP_SECRET
wrangler secret put FEISHU_VERIFICATION_TOKEN
wrangler secret put FEISHU_ENCRYPT_KEY
wrangler secret put MODAL_INTERNAL_BEARER_TOKEN
```

Then update `MODAL_INTERNAL_BASE_URL` in `wrangler.jsonc` or via an environment override.

`MODAL_INTERNAL_BEARER_TOKEN` should match Modal's internal token. If
`HERMES_FEISHU_INTERNAL_BEARER_TOKEN` is not set explicitly, Modal now derives it from
`FEISHU_APP_ID + FEISHU_APP_SECRET`, and the Windows fallback deploy script derives the same value automatically.

## Useful commands

```bash
npm install
npm run types
npm run check
npm run deploy
```

## Windows Fallback Deploy

If `wrangler deploy` is unstable on Windows, use the API-based deploy script instead:

```powershell
pwsh .\scripts\deploy-via-api.ps1 -ProxyUrl http://127.0.0.1:12334
```

The script will:

- build the Worker bundle locally with `wrangler deploy --dry-run`
- upload the module and bindings directly through the Cloudflare API
- enable the Worker on `workers.dev`
