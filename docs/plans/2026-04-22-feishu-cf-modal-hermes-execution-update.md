# 2026-04-22 Feishu-CF-Modal-Hermes Execution Update

## Completed this round

- Feishu DM control-card routing now preserves the real chat type instead of hardcoding group mode.
  - Verified paths:
    - `registry_switch_model`
    - `personality_set`
    - `command_run`
    - bot menu command dispatch
- Modal runtime personality visibility repaired.
  - Remote `personality_picker` now shows:
    - `Neutral`
    - `CEO`
    - `CTO`
    - `Chief of Staff`
    - `Incident`
    - `Growth`
    - `Content`
    - `SEO`
    - `Ads`
    - `BD`
    - `Ops`
    - `Finance`
    - `Board`
- Remote `/internal/feishu/agent-exec` control command verified again.
  - `/personality cto` now succeeds.
  - session state reflects `current_personality=cto`.
- Modal fast-control session state now persists across requests.
  - model/provider lock survives a fresh `/status` call after `/model ... --provider ...`.
- KPI tooling proxy resilience repaired.
  - `tools/feishu_kpi_tools.py`
  - `scripts/feishu_triparty_pk_report.py`
  - default local proxy fallback now works when `127.0.0.1:12334` is reachable.
- KPI summary now backfills gateway metrics from Workers Analytics Engine.
  - cache eligible hit rate
  - capability match rate
  - preferred model selection accuracy
  - browser single AI call completion rate
- Worker cache policy hardened and redeployed.
  - file: `cloudflare/feishu-gateway/src/gateway/cf-ai-exec.ts`
  - worker: `hermes-feishu-gateway`
  - version: `e7e84b2c-402b-4705-9cd3-bd61dbfb3352`
  - new behavior:
    - only short/stateless text prompts are cache-eligible
    - context-dependent follow-up prompts are forced to `skip-cache`
    - stateless cache keys use `global` scope

## Verified tests

- Python:
  - `tests/test_feishu_modal_stub_personality.py`
  - `tests/tools/test_feishu_kpi_tools.py`
  - `tests/test_feishu_triparty_pk_report.py`
  - `tests/gateway/test_feishu.py`
- Worker:
  - `cloudflare/feishu-gateway/test/cf-ai-exec.test.ts`

## Current KPI truth

- Met:
  - read receipt < 5s
    - current p90: `528.3 ms`
  - reply minus AI < 20s
    - current p90: `7594.1 ms`
  - capability match = 100%
    - current: `1.00000000`
  - preferred model accuracy >= 95%
    - current: `1.00000000`
  - browser single AI call completion > 50%
    - current: `1.00000000`
- Not met:
  - AI Gateway cache eligible hit rate > 30%
    - current: `0.00000000`
  - session cost < `0.0045 USD`
    - current 24h p90: `0.01268782`
  - idle hourly cost < `0.005 USD`
    - current 24h p90: `0.01024489`
    - recent 3h p90: `0.00449266`

## Remaining blockers

- Modal debug trace still does not form a reliable shared store across `web_handler` and `debug_feishu_trace`.
  - current gap: `official_cost_without_matching_function_trace`
- Fresh live traffic is still required after the new Worker cache-policy deploy.
  - this is needed to validate whether the AI Gateway cache-hit ratio actually rises above `30%`
- Fresh live session samples are still required after the latest Modal downsizing window matures.
  - this is needed to re-evaluate:
    - 24h idle hourly cost
    - session cost
- One more real Feishu card-click sample is still needed.
  - focus:
    - model switch
    - personality switch
    - skill combo card

## 2026-04-22 07:58 Follow-up optimization update

- KPI closure improved for session-cost truthfulness.
  - `scripts/feishu_triparty_pk_report.py` no longer falls back to `official_total_div_session_count` when the active gap is `official_cost_without_matching_function_trace`.
  - current 24h report now shows:
    - `session_cost_p90_usd = null`
    - `data_gaps = ["official_cost_without_matching_function_trace"]`
  - this removes the previous misleading `0.01268782 USD` session-cost figure.
- Worker observability session enrichment improved.
  - session-level `cache_status` now flows back into the PK report session rows.
  - negative placeholder gateway costs are ignored during PK aggregation.
- AI Gateway cache policy was widened for safe stateless prompts and redeployed.
  - file: `cloudflare/feishu-gateway/src/gateway/cf-ai-exec.ts`
  - changes:
    - stateless cache candidate history window widened from `<= 6` to `<= 12` messages
    - stateless cache key no longer includes `message_count`
  - intent:
    - repeated standalone prompts in the same DM should now have a materially better chance to hit the same AI Gateway cache key
- Workers Analytics Engine write hygiene improved.
  - file: `cloudflare/feishu-gateway/src/observability/analytics-engine.ts`
  - missing/invalid `cost` is now stored as `0` instead of `-1`
- Latest Worker deploy completed through local proxy `127.0.0.1:12334`.
  - worker: `hermes-feishu-gateway`
  - version: `cee15794-07eb-41b9-b127-c8beed86f813`

## Latest live truth after this round

- WAE event counts currently show:
  - `feishu.cf_ai_exec.done = 3`
  - `feishu.workflow.send.done = 3`
  - `feishu.send.operation.done = 3`
  - `feishu.send.operation.error = 0`
- Current cache truth remains not met, but is now based on cleaner samples.
  - latest WAE sample rows:
    - eligible: `2`
    - hit: `0`
    - one recent text request remained explicitly `cache_eligible = 0`
- Current 24h KPI summary after report repair:
  - read receipt p90: `528.3 ms` (`met`)
  - reply minus AI p90: `7594.1 ms` (`met`)
  - idle hourly p90 cost: `0.01024489 USD` (`not_met`)
  - session cost p90: `null` because trace allocation is still missing (`not_met`)
  - cache eligible hit rate: `0.00000000` (`not_met`)
  - capability match rate: `1.00000000` (`met`)
  - preferred model accuracy: `1.00000000` (`met`)
  - browser single AI call completion rate: `1.00000000` (`met`)
