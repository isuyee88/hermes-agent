# Hermes Feishu Gateway Architecture

## Overview
- `src/index.ts`: thin Worker entry that only re-exports the runtime surface.
- `src/runtime.ts`: current compatibility runtime while logic is being split into dedicated modules.
- `src/contracts/feishu-internal.ts`: versioned Worker <-> Modal envelope helpers.
- `internal/feishu/*.py`: Modal-side contract, planner, and trace helpers shared by FastAPI routes.

## Request Lifecycle
1. Cloudflare Worker validates the Feishu webhook and normalizes the ingress event.
2. Worker classifies the request into control, direct browser-prefetch candidate, or workflow execution.
3. Worker sends a `feishu_internal.v1` envelope to Modal internal endpoints.
4. Modal routes unwrap the envelope into the legacy payload shape, run the existing Hermes logic, then wrap the flat response back into `feishu_internal.v1`.
5. Worker unwraps the envelope and continues using the existing response handling path.

## Contract Shape
- Request sections:
  - `ingress`: correlation/session/chat/message identity fields.
  - `route`: request class, route hint, tool/browser requirements, gateway route metadata.
  - `site_prefetch`: optional prefetch manifest.
  - `session_hints`: session key and pending reconcile hints.
  - `gateway_meta`: endpoint, gateway hop/script, legacy rollout payload.
- Response sections:
  - `result`: status, route hint, execution mode, final response, action/error.
  - `send_plan`: send plan, action plan, card payload.
  - `session_patch`: before/after session state and related metadata.
  - `reconcile`: reconcile flags and summaries.
  - `provider_metrics`: provider usage, plan, fallback metadata.

## Rollout Notes
- The new contract is additive. Modal still runs on the existing flat internal payload after unwrapping.
- Responses also embed `legacy_response` so Worker-side rollout can unwrap safely before the runtime split is complete.
- Existing Wrangler bindings, Workflow names, and Durable Object class names remain unchanged in this stage.
