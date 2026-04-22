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
npx wrangler queues create hermes-model-catalog-heartbeat
npm run deploy
```

## Queue Bootstrap

The model catalog control plane now uses a self-rescheduling Queue heartbeat instead of a cron trigger.

After deploy, seed the first run once:

```bash
curl -X POST \
  -H "Authorization: Bearer <MODAL_INTERNAL_BEARER_TOKEN>" \
  https://<your-worker>/internal/model-catalog/queue/bootstrap
```

## Windows Fallback Deploy

If `wrangler deploy` is unstable on Windows, use the API-based deploy script instead:

```powershell
pwsh .\scripts\deploy-via-api.ps1 -ProxyUrl http://127.0.0.1:12334
```

If your machine exports multiple Feishu credential pairs such as `FEISHU_APP_ID`, `FEISHU_APP_ID2`, and `FEISHU_APP_ID3`, you must choose the intended bot explicitly:

```powershell
pwsh .\scripts\deploy-via-api.ps1 -FeishuAppSuffix 3
```

You can also set `HERMES_FEISHU_APP_SUFFIX=3` or `FEISHU_APP_SUFFIX=3` before running the script. The deploy script now fails fast when multiple credential pairs are present but no suffix is specified.

The script will:

- build the Worker bundle locally with `wrangler deploy --dry-run`
- upload the module and bindings directly through the Cloudflare API
- enable the Worker on `workers.dev`
