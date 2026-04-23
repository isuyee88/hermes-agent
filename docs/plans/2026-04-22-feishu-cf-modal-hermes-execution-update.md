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

## 2026-04-22 08:42 Execution round refresh

- This round aligned the two planning docs with a fresh local KPI baseline instead of relying on the earlier 07:58 snapshot only.
- Fresh commands executed:
  - `python scripts/feishu_triparty_pk_report.py --hours 24 --compare-days 1 2 --recent-hours 3 --recent-min-sessions 20 --strict-goal-metric p90`
  - `python scripts/feishu_perf_cost_report.py --since-hours 24`
  - `python scripts/feishu_chain_status.py --artifacts-dir D:\suyee\github\hermesagent`
- Fresh 24h KPI truth from the current repo/runtime view:
  - `reply_minus_ai_p90_ms = 5036.0` -> `met`
  - `read_receipt_p90_ms = null` -> `not_met`
  - `idle_hourly_p90_cost_usd = 0.01023596` -> `not_met`
  - `session_cost_avg_usd = 0.05619778` and `session_cost_p90_usd = null` -> `not_met`
  - `cache_eligible_hit_rate = null` -> `not_met`
  - `browser_single_ai_call_completion_rate = null` -> `not_met`
  - `capability_match_rate = null` -> `not_met`
  - `preferred_model_selection_accuracy = null` -> `not_met`
- Fresh chain-state evidence still proves the repaired historical success path exists:
  - session: `20260421_050249_d828c9f2`
  - response send: `success=True`
  - route lease: `openrouter / nvidia/nemotron-nano-12b-v2-vl:free`
- Fresh cost-report blocker remains unchanged:
  - `modal_debug_function_missing`
  - interpretation: the current main checkout still cannot independently rebuild the missing `debug_feishu_perf_summary` view from Modal without deployed-runtime parity.

## New unified execution entry

- Added a single execution snapshot entrypoint:
  - `scripts/feishu_kpi_execution_snapshot.py`
- It now combines:
  - triparty KPI report
  - chain status
  - perf-cost blocker state
  - priority blockers
  - next actions
- Default outputs:
  - `.tmp-feishu-kpi-execution-snapshot.json`
  - `.tmp-feishu-kpi-execution-snapshot.md`
- Recommended command for each PDCA round:

```bash
python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent
```

## Current next-step ordering after the refresh

1. Publish and verify app3 `im.message.message_read_v1`, then collect fresh real read-receipt samples.
2. Decide whether to restore Modal cross-function trace parity or formally treat WAE as the long-term cost source of truth.
3. Run repeated stateless DM samples on the latest Worker cache policy to re-check `cache_eligible_hit_rate > 30%`.
4. Re-run 24h/72h idle-cost validation after the latest Modal downsizing has fully aged into billing data.

## 2026-04-22 09:10 Checklistization update

- Added a task-checklist generator:
  - `scripts/feishu_execution_task_checklist.py`
- It combines:
  - unified KPI execution snapshot
  - Feishu delivery-path audit
  - task derivation with IDs / dependencies / suggested commands
- Added a fixed step-checklist document:
  - `docs/plans/2026-04-22-feishu-cf-modal-hermes-step-checklist.md`
- New default task-checklist outputs:
  - `.tmp-feishu-execution-task-checklist.json`
  - `.tmp-feishu-execution-task-checklist.md`
- Recommended command for each execution round:

```bash
python scripts/feishu_execution_task_checklist.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent --target-chat-id <target_chat_id>
```

## 2026-04-22 09:10 Cost-truth policy update

- `scripts/feishu_triparty_pk_report.py` now supports explicit session-cost source policy selection:
  - `strict`
  - `blended_total`
  - `official_average`
- The same policy now flows through:
  - `tools/feishu_kpi_tools.py`
  - `scripts/feishu_kpi_execution_snapshot.py`
- Current live diagnostic results:
  - `strict`
    - `session_cost_p90_usd = null`
    - `session_cost_measurement_mode = insufficient_function_trace`
    - `session_cost_truth_status = unavailable`
  - `blended_total`
    - still `null`
    - this proves current data is not yet rich enough even for blended session-total diagnostics
  - `official_average`
    - `session_cost_p90_usd = 0.04444649`
    - `session_cost_measurement_mode = official_total_div_session_count_forced`
    - `session_cost_truth_status = estimated`
- Operational conclusion for `FX006`:
  - default gatekeeping should remain `strict`
  - `official_average` can now be used as an explicit diagnostic-only estimate
  - the repository now supports a formal truth-policy comparison, but the real blocker is still missing runtime trace parity rather than report logic

## 2026-04-22 Delivery realignment and checklist refresh

- Delivery-path repair action ordering is now stable again.
  - root cause:
    - priority `0` actions were being sorted as `99` because of a falsy-coercion bug
  - fixed in:
    - `scripts/check_feishu_delivery_path.py`
- Matrix delivery summary now blocks on live routing topology before reporting success.
  - practical effect:
    - app3 callback readiness is no longer enough to mark delivery `ok` when the target chat still contains multiple bots or the user token is expired
- Task-checklist derivation now follows the canonical active app instead of aggregating missing callbacks from every visible app.
  - fixed in:
    - `scripts/feishu_execution_task_checklist.py`
  - practical effect:
    - `FX001` is now a verification/sampling task for app3 instead of incorrectly asking to republish `im.message.message_read_v1`

## Latest verified live truth after the realignment

- Canonical retained app remains:
  - `FEISHU_APP_ID3`
  - app id: `cli_a9525a47e4f99bc2`
  - app name: `hermes agent`
- Fresh published-version audit for app3 confirms:
  - `im.message.receive_v1` present
  - `im.message.message_read_v1` present
  - `application.bot.menu_v6` present
- Fresh target-chat audit still shows delivery is not ready for KPI closure:
  - target chat: `oc_ec86c28e66596c25377aff2ee028901c`
  - current known bot count: `8`
  - `FEISHU_USER_ACCESS_TOKEN` is expired
  - recent app replies in the target chat are still coming from external app `cli_9ded8676aefb1103`
- Current KPI truth remains:
  - met:
    - `reply_minus_ai_p90_ms`
  - not met:
    - `read_receipt_p90_ms`
    - `session_cost`
    - `idle_hourly_cost`
    - `cache_eligible_hit_rate`

## Current execution order after the refresh

1. `FX001`
   - verify app3 real `message_read` callback samples can enter the KPI path
2. `FX002`
   - converge the target chat to a single retained app3 bot identity
3. `FX003`
   - refresh `FEISHU_USER_ACCESS_TOKEN` and verify the full user-view member list
4. `FX004`
   - send one fresh real Feishu message and confirm a non-synthetic end-to-end workflow
5. `FX005`
   - recheck `read_receipt` KPI with fresh real samples
6. `FX006`
   - keep `strict` as the gate, compare `blended_total` / `official_average` only as diagnostics
7. `FX007` / `FX008` / `FX009`
   - replay cache/cost/control validation after the live delivery chain is clean

## Validation commands run in this round

- `python -m pytest tests/scripts/test_check_feishu_delivery_path.py tests/scripts/test_check_feishu_delivery_path_recent_messages.py tests/scripts/test_feishu_execution_task_checklist.py -q`
- `python scripts/check_feishu_delivery_path.py --all-env-apps --target-chat-id oc_ec86c28e66596c25377aff2ee028901c --recent-message-window-minutes 180`
- `python scripts/feishu_execution_task_checklist.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent --target-chat-id oc_ec86c28e66596c25377aff2ee028901c`

## 2026-04-22 Runtime rebinding and live-send verification

- The Cloudflare Feishu gateway has now been redeployed with the canonical app3 credentials explicitly selected.
  - command:
    - `deploy-via-api.ps1 -FeishuAppSuffix 3 -ProxyUrl http://127.0.0.1:12334`
  - result:
    - worker: `hermes-feishu-gateway`
    - bound app id env: `FEISHU_APP_ID3`
    - bound app secret env: `FEISHU_APP_SECRET3`
- Local Worker bundle issues were repaired by installing the missing gateway dependencies.
  - installed under:
    - `cloudflare/feishu-gateway/package.json`
  - missing packages that blocked deploy:
    - `@mozilla/readability`
    - `linkedom`
    - `@cloudflare/playwright`
- Fresh app3 direct-send verification to target chat now succeeds repeatedly.
  - target chat:
    - `oc_ec86c28e66596c25377aff2ee028901c`
  - fresh sent app messages observed in delivery audit:
    - 4 new app3 messages
    - all from `cli_a9525a47e4f99bc2`
- Fresh user-token verification still confirms the current user token is expired.
  - error code:
    - `99991677`
  - meaning:
    - `Authentication token expired. Please request a new one.`
- Live diagnose now more clearly shows the currently selected target-chat workflow evidence is stale historical evidence rather than a fresh post-rebind run.
  - selected instance:
    - `feishu_oc_ec86c28e66596c25377aff2ee028901c_cross_verify_17767003`
  - created_on:
    - `2026-04-20T15:53:16.850Z`
  - current interpretation:
    - the old `bot_not_in_target_chat` failure is still present in historical workflow data
    - but it is no longer enough to prove the current app3-bound gateway still has the same live send identity problem

## Current hard blockers after the rebinding

- `FX003` still cannot close without a freshly authorized `FEISHU_USER_ACCESS_TOKEN`.
- `FX004` / `FX005` still cannot close without a real user-originated message after token refresh.
- `FX002` still depends on Feishu-side group cleanup.
  - current target chat still reports `8` bots
  - the latest non-app3 reply artifact in the same recent window is still external app `cli_9ded8676aefb1103`

## 2026-04-22 Read-receipt probe update

- Added a dedicated read-receipt sampling script:
  - `scripts/feishu_read_receipt_probe.py`
- Purpose:
  - query recent app3 messages in the target chat
  - fetch `read_users` directly for each recent app message
  - quantify whether the read-receipt KPI is blocked by missing samples or by genuine zero-read state
- Fresh probe result for target chat `oc_ec86c28e66596c25377aff2ee028901c`:
  - recent app3 messages matched: `5`
  - sampled messages: `5`
  - read-receipt samples found: `0`
  - current interpretation:
    - app3 message delivery is working
    - `read_users` API is reachable with the canonical app3 tenant token
    - the current blocker is no longer API reachability, but the absence of actual read-user evidence in the sampled window

## 2026-04-22 10:55 Checklist hardening and cost fallback update

- The execution task checklist is now usable as a stable operator panel instead of a raw task dump.
  - `scripts/feishu_execution_task_checklist.py`
  - improvements:
    - repaired the previously garbled Chinese task text in the generator
    - added explicit task status buckets:
      - `done`
      - `blocked`
      - `todo`
      - `attention`
    - added task-level fields for:
      - current conclusion
      - verification criteria
      - recommended commands
      - dependencies
    - added summary-level fields for:
      - task counts
      - current focus ordering
      - execution order grouping
- The perf-cost report no longer hard-blocks when the current Python interpreter does not have the `modal` SDK installed.
  - `scripts/feishu_perf_cost_report.py`
  - new behavior:
    - if local `import modal` fails, the script now automatically falls back to the Python interpreter next to the discovered `modal.exe`
    - this lets the script query:
      - `debug_feishu_perf_summary`
      - `debug_feishu_trace`
    - without requiring the current shell interpreter to be the Modal-enabled one
- The triparty report is now more resilient to historical-window Modal billing failures.
  - `scripts/feishu_triparty_pk_report.py`
  - new behavior:
    - a failed `modal billing report` call for one window now degrades that window with a `data_gap`
    - it no longer has to abort the entire KPI report generation

## Latest verified truth after the fallback repair

- `python scripts/feishu_perf_cost_report.py --since-hours 24`
  - status: `ok`
  - trace rows: `16`
  - event count: `4`
  - official 24h app total cost: `0.08200033 USD`
- `python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent`
  - chain status: `ok`
  - cost report: `ok`
  - current priority blockers reduced to:
    - `kpi:read_receipt`
    - `kpi:session_cost`
    - `kpi:idle_cost`
    - `kpi:cache_hit_rate`
    - `gap:official_cost_without_matching_function_trace`
- `python scripts/feishu_execution_task_checklist.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent --target-chat-id oc_ec86c28e66596c25377aff2ee028901c`
  - task count: `9`
  - current split:
    - `blocked = 5`
    - `todo = 4`
    - `done = 0`
  - current top focus:
    - `FX001`
    - `FX002`
    - `FX003`
    - `FX004`
    - `FX005`

## Latest hard blockers after this round

- `FX006` is no longer blocked by local environment mismatch.
  - the remaining cost-side blocker is now the real data issue:
    - `official_cost_without_matching_function_trace`
- `FX003` remains blocked by expired user authorization.
  - `python scripts/feishu_user_token_test.py`
  - all 4 attempted user-originated test sends failed with:
    - `99991677 Authentication token expired. Please request a new one.`
- `FX004` / `FX005` still cannot close before:
  - a fresh `FEISHU_USER_ACCESS_TOKEN` is authorized
  - the target chat is cleaned to a single retained app3 bot identity

## 2026-04-22 11:30 OAuth tooling cleanup update

- The Feishu user-token refresh path is now readable and operable again.
  - rewritten:
    - `scripts/feishu_oauth_flow.py`
    - `scripts/feishu_user_token_test.py`
    - `scripts/feishu_get_test_token.py`
  - practical improvements:
    - removed the previous garbled terminal text
    - removed console-breaking emoji output
    - added a clean `--print-auth-url-only` mode for OAuth
    - added a clean `--no-open-browser` mode for OAuth
    - added a focused `--single-message` smoke test for user-token validation
- Added a lightweight OAuth helper test:
  - `tests/scripts/test_feishu_oauth_flow.py`
- Rewrote the operator instructions for manual token refresh:
  - `docs/plans/get-user-access-token.md`

## Latest verified live truth after the OAuth cleanup

- `python scripts/feishu_oauth_flow.py --print-auth-url-only`
  - now prints a valid readable authorization URL for:
    - app id `cli_a9525a47e4f99bc2`
    - redirect URI `http://localhost:8080/callback`
    - scopes:
      - `im:message`
      - `im:message:send`
      - `im:chat:readonly`
      - `im:chat.members:read`
- `python scripts/feishu_get_test_token.py --skip-send-test`
  - confirms tenant token fetch still works
  - confirms current bot info is readable
  - confirms target-chat member snapshot is readable from the app side
- `python scripts/feishu_user_token_test.py --single-message "用户态 smoke test"`
  - still fails with:
    - `99991677 Authentication token expired. Please request a new one.`
  - interpretation:
    - the remaining blocker is no longer tooling readability
    - it is still real-world token expiry
