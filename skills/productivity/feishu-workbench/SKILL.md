---
name: feishu-workbench
description: Use Hermes native Feishu API tools as the only workbench backend for Docs, Sheets, Bitable, messages, files, and model registry operations.
version: 3.0.0
author: Nous Research
license: MIT
metadata:
  hermes:
    tags: [Feishu, Lark, Docs, Sheets, Bitable, ChatOps, Files, Workbench, API]
    homepage: https://open.feishu.cn
prerequisites:
  env_vars: [FEISHU_APP_ID, FEISHU_APP_SECRET]
---

# Feishu Workbench

This skill teaches Hermes to treat Feishu as a real workbench, not just a chat transport.

## Core Operating Model

1. Hermes native `feishu_*` tools are the only workspace backend.
2. Chat replies, cards, files, images, and control actions stay on the native Feishu delivery path.
3. Hermes local registry, session state, and route lease remain the truth source.
4. Bitable is an operator mirror and console, not live routing truth.

## Native Hermes Tool Map

### Docs

- `feishu_doc_create`
- `feishu_doc_get`
- `feishu_doc_append_markdown`
- `feishu_doc_replace_markdown`

### Sheets

- `feishu_sheet_create`
- `feishu_sheet_read_range`
- `feishu_sheet_write_range`

### Bitable

- `feishu_bitable_get_schema`
- `feishu_bitable_list_records`
- `feishu_bitable_upsert_records`
- `feishu_model_registry_prepare_bitable`
- `feishu_model_registry_bootstrap_bitable`

### Chat and delivery

- `feishu_message_send`
- `feishu_chat_lookup`
- `feishu_image_upload`
- `feishu_image_send`
- `feishu_audio_send`
- `feishu_video_send`
- `feishu_model_registry_publish_card`

### Files

- `feishu_file_upload`
- `feishu_file_send`
- `feishu_file_download`

### Registry and operations

- `feishu_model_registry_list`
- `feishu_model_registry_sync`
- `feishu_lookup_user`

## Mandatory Working Rules

1. Native `feishu_*` tools are the only supported workbench path.
2. Do not treat the mirrored Bitable as authoritative routing state.
3. Do not silently switch models. Model changes must come from explicit user intent, a picker or card action, or a deliberate control-plane command flow.
4. Menu clicks, card actions, and model switching are control-plane actions and should not be routed through the normal chat LLM interpretation path when a direct handler exists.
5. For large Sheets, large Docs, and large Bitables, read schema first, then page or scope the smallest useful slice.
6. For Bitable writes, verify field names before writing when schema certainty is low.
7. For generated local files, prefer `MEDIA:/absolute/path` or native file/image send tools.
8. For Feishu attachments that need AI analysis, download first, process locally, then send the result back through native delivery.
9. For destructive workspace changes, confirm before editing shared data.

## Decision Rules

### Docs and Sheets

- Use native `feishu_doc_*` and `feishu_sheet_*` tools directly.
- Prefer append when extending existing notes and replace when regenerating the full document body.
- Keep spreadsheet reads narrow and targeted.

### Bitable and Model Mirror

- Use `feishu_model_registry_list` for default model catalog questions because Hermes local registry is the source of truth.
- Use `feishu_model_registry_prepare_bitable` or `feishu_model_registry_bootstrap_bitable` before first sync when the table may be incomplete.
- Use `feishu_model_registry_sync` to mirror local registry state into Bitable.
- Use Bitable for operator visibility, recent usage review, hidden model audit, and copyable switch commands.

### Chat Delivery

- Always use native Feishu delivery for the final reply in the active conversation.
- Use interactive cards for dense model selection and operator workflows when it reduces chat friction.
- Reply with concise structured summaries after workspace mutations so the user knows what changed.

## Recommended Workflows

### Write Research Into a Doc and Reply With the Link

1. Create or load the Doc with native `feishu_doc_*` tools.
2. Append or replace content deliberately based on whether the update is additive or full-refresh.
3. Reply in chat with the link and a short change summary.

### Read a Sheet, Produce a Summary, and Write It Back

1. Read only the needed range.
2. Summarize or transform locally in Hermes.
3. Write back the minimal updated range.
4. Reply in chat with the result and, if useful, the sheet link.

### Bootstrap the Hermes Model Registry Mirror

1. If a workspace needs a new Bitable app, use `feishu_model_registry_bootstrap_bitable`.
2. If the app already exists, use `feishu_model_registry_prepare_bitable`.
3. Run `feishu_model_registry_sync` to mirror the latest local Hermes registry.
4. Publish an operator-facing card with `feishu_model_registry_publish_card` when a compact UI helps.

### Inspect the Bitable Registry Without Treating It as Truth

1. Use `feishu_model_registry_list` first for normal model questions.
2. Read Bitable schema or records only when the user explicitly asks about the mirrored table itself.
3. Do not let table contents silently drive live route changes.

### Analyze a Feishu Attachment

1. Download with `feishu_file_download`.
2. Run Hermes-native vision, OCR, transcription, or document analysis on the local file.
3. Send a concise summary plus any useful artifact back with native Feishu delivery.

### Publish a Dense Operator Card

1. Refresh registry state if needed with `feishu_model_registry_sync`.
2. Publish an interactive card with `feishu_model_registry_publish_card`.
3. Keep card content compact, action-oriented, and aligned with Hermes local state.

## Good Output Shape

- Read operations should return a compact JSON summary plus the ids or tokens needed for follow-up.
- Write operations should return success, primary ids or tokens, a URL when available, and a short change summary.
- Large results should be paginated, truncated, or scoped down instead of dumped in full.

## Avoid

- Do not mention or rely on Feishu MCP as a backend.
- Do not hardcode raw Feishu API URLs in prompts when a Hermes tool already wraps the action.
- Do not treat Bitable mirror rows as authoritative route state.
- Do not send control-plane menu or card actions back through the normal chat LLM path when a direct handler exists.
- Do not silently degrade to another model just because the current prompt is short.
