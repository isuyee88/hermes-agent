---
name: feishu-workbench
description: Use Hermes Feishu tools to work with Docs, Sheets, Bitable, chat messages, files, and model registry cards inside Feishu/Lark.
version: 1.0.0
author: Nous Research
license: MIT
metadata:
  hermes:
    tags: [Feishu, Lark, Docs, Sheets, Bitable, ChatOps, Files]
    homepage: https://open.feishu.cn
prerequisites:
  env_vars: [FEISHU_APP_ID, FEISHU_APP_SECRET]
---

# Feishu Workbench

This skill teaches Hermes how to use the native Feishu tool surface as a workbench, not just a chat channel.

## When To Use

Use this skill when the user wants Hermes to:

- create or update a Feishu Doc
- read or write a Feishu Sheet range
- inspect or upsert Bitable records
- send a native Feishu message, image, file, or card
- publish the current model registry into Feishu
- download a Feishu attachment for later vision, OCR, or transcription

## Preferred Tool Mapping

- Docs
  - `feishu_doc_create`
  - `feishu_doc_get`
  - `feishu_doc_append_markdown`

- Sheets
  - `feishu_sheet_create`
  - `feishu_sheet_read_range`
  - `feishu_sheet_write_range`

- Bitable
  - `feishu_bitable_get_schema`
  - `feishu_bitable_list_records`
  - `feishu_bitable_upsert_records`

- Chat and delivery
  - `feishu_message_send`
  - `feishu_chat_lookup`
  - `feishu_model_registry_publish_card`

- Files
  - `feishu_file_upload`
  - `feishu_file_send`
  - `feishu_file_download`

- Registry and operations
  - `feishu_model_registry_sync`
  - `feishu_lookup_user`

## Working Rules

1. Prefer native Feishu tools over raw HTTP or handwritten curl when a matching Hermes tool exists.
2. For document/report delivery, first create or update the artifact, then send the resulting link or file back to the user in Feishu.
3. For files generated locally, prefer the cross-platform `MEDIA:/absolute/path` convention or the explicit `feishu_file_send` tool.
4. For large tables, read or write the smallest range possible. Do not dump full sheets unless the user explicitly asks.
5. For Bitable writes, inspect schema first if the field names are uncertain.
6. For destructive or risky operations, ask the user before changing shared workspace data.

## Recommended Patterns

### Send a report to Feishu

1. Create or update content with `feishu_doc_create` or `feishu_doc_append_markdown`
2. Send the result with `feishu_message_send`

### Analyze an attachment from Feishu

1. Download with `feishu_file_download`
2. Run `vision_analyze` or the relevant transcription/document tool on the downloaded local file
3. Reply with summary plus native file/image output if useful

### Publish model choices to the user

1. Refresh registry with `feishu_model_registry_sync`
2. Publish an interactive summary with `feishu_model_registry_publish_card`

## Avoid

- Do not assume Feishu Drive permissions are required for normal chat attachment sending.
- Do not hardcode Feishu API URLs in prompts when a Hermes tool already wraps the operation.
- Do not switch models silently when the user is explicitly using Feishu model picker or provider controls.
