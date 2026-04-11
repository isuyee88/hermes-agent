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

This skill teaches Hermes how to use Feishu as a workbench, not just a chat channel.

When both tool surfaces are available, prefer them in this order:

1. Official Feishu MCP/CLI tools for workspace-style operations such as Docs, Sheets, Bitable, Contacts, and structured workbench actions.
2. Native Hermes `feishu_*` tools as fallback for the same operations when the official MCP/CLI server is unavailable or missing permissions.
3. Native Hermes Feishu platform sending for bot replies, interactive cards, model-picker controls, and attachment delivery back into the current conversation.

## When To Use

Use this skill when the user wants Hermes to:

- create or update a Feishu Doc
- read or write a Feishu Sheet range
- inspect or upsert Bitable records
- send a native Feishu message, image, file, or card
- publish the current model registry into Feishu
- download a Feishu attachment for later vision, OCR, or transcription

## Preferred Tool Mapping

- Official Feishu MCP/CLI
  - Prefer any discovered `mcp_feishu_*` or tools from the configured Feishu MCP server for workspace data operations.
  - Use these first for Docs, Sheets, Bitable, Contacts, or other structured Feishu APIs when they are available in the current tool list.

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
2. If official Feishu MCP/CLI tools are present, prefer them for workbench CRUD and data queries before falling back to native `feishu_*` wrappers.
3. For document/report delivery, first create or update the artifact, then send the resulting link or file back to the user in Feishu.
4. For files generated locally, prefer the cross-platform `MEDIA:/absolute/path` convention or the explicit `feishu_file_send` tool.
5. For large tables, read or write the smallest range possible. Do not dump full sheets unless the user explicitly asks.
6. For Bitable writes, inspect schema first if the field names are uncertain.
7. The Bitable model registry is an operations mirror, not the live source of truth for routing. Do not treat table edits as automatic route changes unless the user explicitly asks for manual sync or review.
8. Do not silently switch models. Model changes must come from explicit user intent, model picker controls, or a clearly requested `/model` command flow.
9. Do not bypass native Hermes Feishu sending for the final bot reply in the active chat. Workspace operations can use MCP/CLI; the conversational reply path stays native.
10. For destructive or risky operations, ask the user before changing shared workspace data.

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

### Workspace-first, chat-native workflow

1. Use official Feishu MCP/CLI tools first to inspect or update Docs, Sheets, Bitable, or Contacts
2. If the MCP/CLI path is unavailable, fall back to native `feishu_*` tools
3. Deliver the final result to the active chat with native Hermes Feishu sending or `MEDIA:/absolute/path`

## Avoid

- Do not assume Feishu Drive permissions are required for normal chat attachment sending.
- Do not hardcode Feishu API URLs in prompts when a Hermes tool already wraps the operation.
- Do not switch models silently when the user is explicitly using Feishu model picker or provider controls.
- Do not treat the Bitable mirror as the real-time routing database.
- Do not send control-plane card or menu actions back through the chat LLM path if a direct handler exists.
