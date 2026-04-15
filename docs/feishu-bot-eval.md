# Feishu Bot Eval Helper

`scripts/feishu-bot-eval.mjs` is a Node.js helper for evaluating the OpenClaw Feishu bot with the credentials already present on this machine.

It supports:

- OAuth login to obtain a reusable `user_access_token`
- automatic refresh when OAuth was granted with `offline_access`
- retrieving the logged-in user's profile
- listing or searching chats visible to the authorized user
- reading bot-visible chat history with the app token
- sending a control message as the bot/application
- running a manual black-box evaluation flow and saving a JSON report

The helper intentionally keeps the default OAuth scope set minimal and does not require
`im:chat.group_info:readonly`. Some Feishu consoles no longer expose that exact scope,
and it is not required for the current login, user-profile, chat discovery, or bot-side
evaluation flow implemented by this script.

## Why there is a manual mode

Feishu OpenAPI does not expose a public `message.create` path that sends a message **as the authorized user**. The public API sends with the bot/app identity.

That means:

- `login` is still valuable because it lets us resolve the right user and chat context
- `bot-send` validates transport and app-side permissions only
- `eval --mode manual` is the closest true black-box workflow today: you send the prompt in Feishu, the script captures replies and produces a report

## Quick start

```bash
cd D:\suyee\github\hermesagent\hermes-agent
node scripts/feishu-bot-eval.mjs login --offline-access
node scripts/feishu-bot-eval.mjs whoami
node scripts/feishu-bot-eval.mjs list-chats --query "网络营销专家"
node scripts/feishu-bot-eval.mjs eval --batch read --mode manual --bot-name "网络营销专家"
```

If Feishu shows error `20029`, the redirect URL configured in the app does not match the script.
This helper now defaults to:

```text
http://localhost:3000/callback
```

You can also force a redirect URL that is already approved in the Feishu app:

```bash
node scripts/feishu-bot-eval.mjs login --offline-access --redirect-uri "http://localhost:3000/callback"
```

The redirect URL in the Feishu app security settings must match the script value exactly.

## Token sources

The script uses these environment variables by default:

- `FEISHU_APP_ID2`
- `FEISHU_APP_SECRET2`

It falls back to:

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`

For long-term use, prefer the OAuth cache instead of `FEISHU_UAT`.
`FEISHU_UAT` is a one-off bootstrap convenience and can become stale inside an existing shell session.
When `login` / `auth-url` is completed with `--offline-access`, the cache stored on disk includes a `refresh_token` and the script will automatically refresh the access token on the next run when needed.

OAuth tokens are stored outside the repo in:

```text
%USERPROFILE%\.hermes-feishu-bot-eval
```
