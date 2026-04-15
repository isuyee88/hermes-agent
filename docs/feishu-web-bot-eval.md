# Feishu Web Bot Eval Helper

`scripts/feishu-web-bot-eval.mjs` evaluates a Feishu bot through the real Feishu Web UI.

This is the fallback path when:

- Feishu OpenAPI cannot send messages as the logged-in user
- the app owner has not approved a local OAuth redirect URL
- you want a true black-box test of what happens after a user sends a message

## What it does

- opens Feishu Web in a persistent local browser profile
- lets you log in manually once
- reuses that login session on later runs
- sends prompts through the web UI as the real logged-in user
- captures screenshots, visible text, accessibility snapshots, and a JSON report

## Install

```bash
cd D:\suyee\github\hermesagent\hermes-agent
npm install
```

On Windows, the script defaults to the installed Microsoft Edge channel.
If you prefer Chrome, pass `--channel chrome`.
If neither channel works, pass `--executable-path`.

## Quick start

Open a persistent Feishu Web session:

```bash
npm run feishu:web-eval -- open
```

Send one real message as the current web user:

```bash
npm run feishu:web-eval -- send --text "Please summarize the recent discussion in this chat."
```

Run a builtin capability batch:

```bash
npm run feishu:web-eval -- eval --batch workflow
```

Use your own prompt list:

```bash
npm run feishu:web-eval -- eval --prompt-file D:\path\to\prompts.txt
```

## Manual steps during the run

The script intentionally keeps the fragile part manual:

1. Log in to Feishu Web
2. Open the target bot chat manually
3. Keep the composer visible
4. Press Enter in the terminal

That keeps the evaluation focused on the bot behavior instead of brittle sidebar selectors.

## Useful options

- `--channel msedge`
- `--channel chrome`
- `--executable-path "C:\Program Files\Google\Chrome\Application\chrome.exe"`
- `--send-shortcut enter`
- `--send-shortcut ctrl-enter`
- `--reply-wait-ms 60000`
- `--pause-between-prompts true`
- `--keep-open true`

## Output

Reports are written outside the repo:

```text
%USERPROFILE%\.hermes-feishu-web-eval\reports
```

Each run stores:

- `report.json`
- before/after screenshots
- before/after visible text captures
- before/after accessibility snapshots

`report.json` now also records:

- `sent_at`
- `trace_token`
- `prompt_sent`

The helper appends a `[trace:...]` token to evaluation prompts by default so you can
correlate Feishu Web reports with Modal-side ingress/ACK/worker logs.

## Recommended use for your bot assessment

Use this helper to validate:

- whether a message can trigger the bot reliably
- whether the bot replies with concrete actions or only suggestions
- whether multi-step workflows can be driven from message prompts
- whether replies are stable enough for mobile-first operations
