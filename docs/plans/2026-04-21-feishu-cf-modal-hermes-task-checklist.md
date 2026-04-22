# Feishu-CF-Modal-Hermes 任务清单
更新时间：2026-04-21（UTC+8）

## 1. 当前结论

- `401 Unauthorized` 的直接原因已定位并修复：Cloudflare Worker 持有的 `MODAL_INTERNAL_BEARER_TOKEN` 与 Modal 当前期望指纹不一致。
- Worker 与 Modal 的内部契约已定位并修复：Worker 当前采用“固定 `MODAL_INTERNAL_BASE_URL` + body 内 `__path`”模式，而非“URL 直接拼接内部路径”。
- 生产 Modal 已恢复 `web_handler` 入口，当前可同时提供：
  - `web_app`
  - `web_handler`
- 使用真实 Workflow 实例参数回放到生产 `web_handler` 已返回 `200 completed`，说明最关键的 Worker -> Modal 内部入口断链已恢复。

## 2. 阶段任务

### P0. 阻塞问题止血

- [x] 校验 `MODAL_INTERNAL_BEARER_TOKEN` 指纹漂移并修正 Cloudflare Worker secret
- [x] 校验 `MODAL_INTERNAL_BASE_URL` 指向的 Modal 入口是否真实存在
- [x] 恢复与 Worker 当前契约兼容的 `web_handler`
- [x] 直连 `web_handler` 验证 `/internal/feishu/agent-exec` 可返回 200
- [x] 用真实 Workflow 实例参数回放验证兼容入口可处理历史真实请求

### P1. 活链路验证闭环

- [ ] 发送一条新的真实飞书文本消息，确认会生成新的 Workflow 实例而不是只看到历史失败实例
- [ ] 拉取最新 Workflow step 详情，确认不再出现 `modal_internal_failed:401`
- [ ] 拉取最新 Workflow step 详情，确认不再出现 `modal_internal_failed:404`
- [ ] 拉取最新 Workflow step 详情，确认 `cf-ai-exec` 或 `modal-agent-exec` 可正常收口
- [ ] 确认飞书侧已收到正式回复
- [ ] 确认飞书侧可收到已读回执事件并完成关联

### P2. 兼容面补全

- [ ] 补齐 Worker 当前可能调用的 Modal 内部路径清单
- [ ] 修复 `/internal/feishu/session-control`
  当前状态：`get_session_state` 已恢复；live Modal 旧产物在 `dispatch_command` / `activate_skill_combo` 路径仍可能触发 `ImportError`
- [x] 修复 `/internal/feishu/agent-plan`
  当前状态：2026-04-21 live probe 已返回 `200`
- [ ] 验证 `/internal/feishu/result-file/{token}` 文件回取路径是否与当前 `MODAL_INTERNAL_BASE_URL` 契约兼容
- [ ] 为每条内部路径补充最小回归探针

### P3. 诊断自动化

- [x] 新增活链路诊断脚本：
  - `scripts/feishu_cf_modal_live_diagnose.py`
- [ ] 将“最新 Workflow 实例 + Worker 配置 + Modal 回放”纳入日常排障 runbook
- [ ] 增加“token 指纹漂移”告警项
- [ ] 增加“Worker base URL 指向不存在 handler”告警项
- [ ] 增加“Worker 内部契约与 Modal 部署产物不匹配”告警项

### P4. 回归矩阵

- [ ] 快速控制面：
  - `/model`
  - `/provider`
  - `/personality`
  - 菜单卡片动作
- [ ] 一般文本会话：
  - 单轮问答
  - 多轮上下文
  - 会话锁定与人格连续性
- [ ] 重任务分类：
  - 浏览器任务识别
  - 分类预处理完整性
  - 单次 AI 网关达成率
- [ ] 多媒体任务：
  - 图片/文件/附件链路
  - 回传文件与媒体上传
- [ ] 回退策略：
  - 429 不触发 fallback
  - 模型错误一次 fallback 成功率 100%

### P5. 指标验收

- [ ] 单会话成本 `< 0.0045 USD`
- [ ] 空闲耗费 `< 0.005 USD / 小时`
- [ ] 飞书发出到收到已读 `< 5 秒`
- [ ] 扣除 AI 处理时间后正式回复 `< 20 秒`
- [ ] CF 因限流触发 fallback `= 0`
- [ ] 模型错误后一次 fallback 成功率 `= 100%`
- [ ] CF 网关缓存率 `> 30%`
- [ ] 浏览器任务分类预处理正确率 `= 100%`
- [ ] 浏览器任务单次 AI 网关达成率 `> 50%`
- [ ] 智能路由正确率 `= 100%`
- [ ] 类型能力匹配正确率 `= 100%`

## 3. 当前建议执行顺序

1. 先用 `scripts/feishu_cf_modal_live_diagnose.py` 固化当前“已修复”的证据，避免再次回到人工排查。
2. 立刻做一次新的真实飞书消息验证，确认历史 `errored` 实例不是修复后的现状。
3. 如果最新实例仍失败，再按 step 名称反推是否命中 `session-control`、`agent-plan` 或 `result-file` 缺口。
4. 活链路稳定后，再推进成本、缓存率、浏览器任务分类和单次网关达成率指标。

## 4. 本轮证据

- 历史失败实例：
  - `feishu_oc_33edcba53ad086b50f175352947750eb_0b3dc12b0ded07b34aa25`
  - 失败 step：`cf-ai-exec`
  - 错误：`modal_internal_failed:500:Internal Server Error`
- 更早自测失败实例：
  - `feishu_oc_selftest_ingress_e1a5e677362c_evt_selftest_3c923f64457`
  - 失败 step：`edge-agent-plan`
  - 错误：`execution_mode is not defined`
- 修复后回放结果：
  - 真实实例参数回放到 `web_handler` 返回 `200`
  - 返回状态：`completed`
  - 说明当前 `agent-exec` 兼容入口已可接住真实请求

## 5. 2026-04-21 12:15 Status Update

- Latest live diagnose confirms the blocking stage has moved from Modal ingress/auth to `send-feishu-response`.
- Latest selected instance:
  - `feishu_oc_ec86c28e66596c25377aff2ee028901c_test_1776702505`
  - `selected_instance_error_class=bot_not_in_target_chat`
  - `selected_instance_is_synthetic=true`
  - `selected_instance_chat_id=oc_ec86c28e66596c25377aff2ee028901c`
- Latest workflow step status:
  - `edge-agent-plan` success
  - `cf-ai-exec` success
  - `send-feishu-response` failed with `send_message_failed:400:Bot/User can NOT be out of the chat.`
- Conclusion:
  - Worker -> Modal internal path compatibility is restored.
  - Current regression blocker is a stale or invalid Feishu test target, not the earlier `401/404/500` Modal path issue.
- Remediation already applied in repo:
  - added structured Feishu send failure classification/logging
  - upgraded `scripts/feishu_cf_modal_live_diagnose.py` to expose synthetic-instance and error-class hints
  - changed common Feishu test scripts to prefer `FEISHU_TEST_CHAT_ID` / `FEISHU_HOME_CHANNEL` instead of a stale hardcoded test group
- Additional live verification notes:
  - direct Feishu app-token send to `oc_ec86c28e66596c25377aff2ee028901c` now succeeds
  - automated real-user ingress replay is currently blocked by expired `FEISHU_USER_ACCESS_TOKEN`

## 6. 2026-04-21 Delivery Config Root Cause Update

- The current production chain is no longer blocked at `Worker -> Modal` auth/path ingress.
- Multi-app audit now confirms there are **two active Feishu apps** that can currently see target chats, and **neither subscribes to text message receive events**:
  - `app_id=cli_a95274809bb95bda`
  - `app_name=网络营销专家`
  - visible chats: `2`
  - chats include:
    - `oc_ec86c28e66596c25377aff2ee028901c (CEO 工作汇报)`
    - `oc_fce2101bc2d5dfc0605b9507b7097ba2 (素依的智能体团队)`
  - callback type: `webhook`
  - subscribed callbacks: `[card.action.trigger]`
  - callback URL is currently empty in app metadata
  - `app_id=cli_a9525a47e4f99bc2`
  - `app_name=hermes agent`
  - visible chats: `1`
  - visible chat:
    - `oc_ec86c28e66596c25377aff2ee028901c (CEO 工作汇报)`
  - callback type: `webhook`
  - request URL: `https://hermes.isuyee.com/`
  - subscribed callbacks: `[card.action.trigger]`
- Live callback probe result:
  - `POST https://hermes.isuyee.com/` with `url_verification` challenge returns the challenge successfully
  - this means the configured root URL is currently reaching the deployed Worker
- The real blocker for "Feishu text message never reaches Hermes" is now identified as:
  - every active app that can see the target chat is missing `im.message.receive_v1`
  - every active app that can see the target chat is missing `im.message.message_read_v1`
  - every active app that can see the target chat is missing `application.bot.menu_v6`
- Secondary app state found during audit:
  - `app_id=cli_a96babfc89b8dcd1`
  - `callback_type=websocket`
  - visible chat count: `0`
  - also not subscribed to `im.message.receive_v1`

### Immediate Repair Tasks

- [x] Prove which Feishu apps can currently see target chats and separate them from the inactive websocket app
- [x] Prove the configured callback root URL can reach the current Worker
- [x] Upgrade `scripts/check_feishu_delivery_path.py` so it can audit all `FEISHU_APP_ID*` pairs, visible chats, and callback challenge reachability
- [ ] Enable `im.message.receive_v1` on every active target-chat app (`cli_a95274809bb95bda`, `cli_a9525a47e4f99bc2`) or explicitly converge traffic to a single retained app
- [ ] Enable `im.message.message_read_v1` on every retained target-chat app
- [ ] Enable `application.bot.menu_v6` on every retained target-chat app
- [ ] Decide and enforce which single Feishu app is the canonical Hermes ingress app for production groups
- [ ] Re-run live ingress verification after event subscription repair and confirm a fresh non-synthetic workflow instance is created
- [ ] Confirm read receipt callback correlation resumes after `im.message.message_read_v1` is enabled

## 7. 2026-04-21 Worker Credential Realignment Update

- User confirmed the production app should be `FEISHU_APP_ID3 / FEISHU_APP_SECRET3`.
- Direct Feishu API verification on the same target chat now proves the identity mismatch clearly:
  - base app (`FEISHU_APP_ID / FEISHU_APP_SECRET`) send to `oc_ec86c28e66596c25377aff2ee028901c` fails with `230002 Bot/User can NOT be out of the chat`
  - canonical app3 (`FEISHU_APP_ID3 / FEISHU_APP_SECRET3`) send to the same chat succeeds with `code=0`
- Cloudflare Worker secrets have been updated via Cloudflare API to align the runtime with the canonical app3 identity:
  - `FEISHU_APP_ID`
  - `FEISHU_APP_SECRET`
  - `MODAL_INTERNAL_BEARER_TOKEN`
- Additional guardrail applied in repo:
  - `cloudflare/feishu-gateway/scripts/deploy-via-api.ps1` now fails fast when multiple `FEISHU_APP_ID*` pairs exist but no explicit suffix is provided
  - `cloudflare/feishu-gateway/README.md` now documents `-FeishuAppSuffix 3`
- Updated diagnosis after this fix:
  - previous `send_message_failed:400:Bot/User can NOT be out of the chat` is now strongly explained by Worker using the wrong Feishu app identity
  - remaining unresolved item is fresh real webhook ingress verification for app3
  - `application/v6` callback metadata appears to under-report event subscriptions compared with the published developer-console UI, so its `subscribed_callbacks` field must no longer be treated as authoritative evidence on its own

### Remaining P0 Validation After Credential Fix

- [x] Realign Worker Feishu identity from base app to `APP_ID3`
- [x] Add deployment guardrail to prevent silent fallback to the wrong `FEISHU_APP_ID*`
- [ ] Trigger a fresh real user message to the `hermes agent` app and confirm new `feishu.webhook.accepted`
- [ ] If ingress still fails, validate whether Worker `FEISHU_VERIFICATION_TOKEN` matches the canonical app3 event subscription token
- [ ] After fresh ingress succeeds, re-check workflow send phase to confirm the previous `230002` failure is gone

## 8. 2026-04-21 13:30 Live Chain Recovery Update

- Current live production evidence now shows the Feishu group chain is restored end-to-end for the retained app3 identity.
- Verified production session:
  - `session_key=agent:main:feishu:group:oc_ec86c28e66596c25377aff2ee028901c:on_8a49b101406830c3e0876532e82be435`
  - `session_id=20260421_050249_d828c9f2`
  - route lease is now healthy again:
    - `provider=openrouter`
    - `model=nvidia/nemotron-nano-12b-v2-vl:free`
- Verified runtime readiness:
  - `provider_ready=true`
  - `home_channel=oc_ec86c28e66596c25377aff2ee028901c`
  - `default_model=openrouter/free`
- Verified latest successful response send in production logs:
  - `2026-04-21 13:30:29 CST`
    - `response ready: platform=feishu chat=oc_ec86c28e66596c25377aff2ee028901c time=21.7s api_calls=1 response=223 chars`
  - `2026-04-21 13:30:30 CST`
    - `Background task send complete ... success=True send_message_id=om_x100b5144f9a59cb4b34014ed90349f2`
  - `dispatch done event_id=codexfix_1776749392`
- Interpretation:
  - `Feishu -> Worker` is no longer blocked.
  - `Worker -> Modal internal auth` is no longer blocked.
  - `Modal runtime provider config` is no longer blocked.
  - `Hermes -> Feishu formal reply send` is now succeeding again.
  - The earlier “message never reaches Hermes” complaint is resolved for the current repaired production path.

### Status changes after live proof

- [x] Worker Feishu identity realigned to app3
- [x] Modal runtime provider credentials restored
- [x] `FEISHU_HOME_CHANNEL` restored in production runtime
- [x] Synthetic replay with a real Feishu `message_id` now completes end-to-end
- [x] Production send phase no longer fails with `230002 Bot/User can NOT be out of the chat`
- [x] Production send phase no longer fails with `401 Unauthorized`
- [ ] Fresh real-user message verification still required as the final non-synthetic ingress proof
- [ ] Read-receipt timing KPI still needs fresh live measurement
- [ ] Cost/cache/browser-preprocess KPI matrix still needs full rolling-window validation

### New automation hardening added in repo

- [x] Add artifact-based chain state parser:
  - `scripts/feishu_chain_status.py`
- [x] Add parser regression tests:
  - `tests/test_feishu_chain_status.py`
- [x] Make KPI scripts fail with structured blockers instead of raw stack traces:
  - `scripts/feishu_perf_cost_report.py`
  - `scripts/feishu_triparty_pk_report.py`

### Newly exposed tooling blocker

- The current main repo checkout is not the same artifact as the live deployed Modal runtime.
- Evidence:
  - local `modal_.py` in the main repo does not expose `_build_feishu_perf_summary_from_rows`

## 9. 2026-04-21 15:58 Live Routing Truth Update

- Repository-side read-receipt correlation gap has now been closed in code:
  - `internal/feishu/trace.py`
    - add recent trace reader
    - add `correlate_message_read_event(...)`
    - add `build_message_read_correlation_trace_extras(...)`
  - `internal/feishu/webhook_fast_paths.py`
    - `im.message.message_read_v1` fast path now emits:
      - `webhook.message_read.correlated`
      - `webhook.message_read.unmatched`
  - `gateway/platforms/feishu.py`
    - `_on_message_read_event(...)` is no longer a pure no-op and now logs correlation matches
  - targeted regression coverage added:
    - `tests/internal/test_feishu_read_receipt_trace.py`
    - `tests/internal/test_feishu_webhook_fast_paths.py`

### Fresh verification results after the code repair

- Recent 50-minute production data for target chat `oc_ec86c28e66596c25377aff2ee028901c` still shows:
  - Modal trace rows in `hermes-agent/debug_feishu_trace`: `0`
  - Cloudflare Worker observability events for that chat: `0`
- Therefore the latest real user traffic still does **not** enter the app3 Hermes ingress path.

### New strongest evidence for the real current blocker

- Direct Feishu message history query on the target group with `sort_type=ByCreateTimeDesc` shows:
  - `2026-04-21 13:48:28 +08:00` user message
  - `2026-04-21 13:51:18 +08:00` user message

## 10. 2026-04-21 16:58 Delivery Diagnosis Hardening Update

- `scripts/check_feishu_delivery_path.py` has been upgraded again so it now:
  - audits published app-version event subscriptions from `application/v6/applications/{app_id}/app_versions`
  - correlates recent visible-chat traffic with the actual replying Feishu `app_id`
  - fetches recent chat history with `sort_type=ByCreateTimeDesc` so fresh user traffic is not missed
  - forces UTF-8 console output on Windows so rich-text Feishu payloads do not crash the script
- Regression coverage added/extended:
  - `tests/scripts/test_check_feishu_delivery_path.py`
  - `tests/scripts/test_check_feishu_delivery_path_recent_messages.py`

### Latest live output for the canonical app3 (`cli_a9525a47e4f99bc2`)

- Verified target chat still visible:
  - `oc_ec86c28e66596c25377aff2ee028901c (CEO 工作汇报)`
- Verified recent real traffic in the last 180 minutes:
  - user messages seen: `3`
  - user messages targeting current app3 bot open_id `ou_c07fb4b6bcc81c5951bbde1933383088`: `0`
  - user messages mentioning another bot open_id: `ou_26c2a287784865ebc498643162c3f223`
  - recent app replies from app3: `0`
  - recent app replies from another app: `cli_a95274809bb95bda` (`3` replies)
- Published-version truth for app3:
  - `im.message.receive_v1`: enabled
  - `application.bot.menu_v6`: enabled
  - `im.message.message_read_v1`: still missing from the latest published version
- Current strongest blocker produced by the script:
  - `recent_messages_target_different_bot`
  - summary: recent user messages in the target chat are not targeting app3, and recent replies are coming from another app (`cli_a95274809bb95bda`)

### Updated repair interpretation

- The current blocking point is no longer “Feishu message cannot reach Hermes because ingress is broken”.
- The current blocking point is “the target production group is still routing real user intent to a different Feishu bot/app identity than app3”.
- App3 still needs `im.message.message_read_v1` published before the read-receipt KPI can be considered complete.

### Immediate next execution tasks

- [x] Teach `scripts/check_feishu_delivery_path.py` to use published app-version events as a stronger signal than `callback_info`
- [x] Teach `scripts/check_feishu_delivery_path.py` to surface recent replying `app_id` ownership
- [x] Fix the script so recent-message analysis actually reads newest chat traffic first
- [ ] Converge `CEO 工作汇报` so real user messages target only app3 (`cli_a9525a47e4f99bc2`)
- [ ] Remove or stop relying on conflicting bot identities in the same chat, especially `cli_a95274809bb95bda`
- [ ] Publish app3 with `im.message.message_read_v1`
- [ ] After chat routing is converged, send a fresh real user message and confirm:
  - Worker ingress event appears
  - Modal trace row appears
  - read receipt is correlated
  - formal reply is sent by app3 rather than the competing app

## 11. 2026-04-21 17:05 Target-Chat Ownership Update

- `scripts/check_feishu_delivery_path.py` now also supports:
  - `--target-chat-id` to isolate a single production group
  - known-env bot/app catalog enrichment so output marks:
    - known in current env
    - unknown external bot identity
    - replying app name / app id / env slot
- Latest focused run against:
  - `target_chat_id=oc_ec86c28e66596c25377aff2ee028901c`
  - canonical app3 `cli_a9525a47e4f99bc2`
- Latest focused live truth:
  - recent user messages targeting app3 in this chat: `0`
  - recent user messages targeting another bot open_id:
    - `ou_26c2a287784865ebc498643162c3f223`
    - `known_in_env=false`
  - recent app replies in this chat:
    - `cli_a95274809bb95bda`
    - app name: `网络营销专家`
    - env slot: `FEISHU_APP_ID2`
    - reply count in the current rolling window: `3`
- Interpretation tightened further:
  - the target group is currently affected by **two separate routing conflicts**:
    - an **unknown external bot open_id** is still being explicitly mentioned by users
    - a **known env app2 bot** (`网络营销专家`) is still the app that actually replies in the group
  - therefore app3 cannot yet be treated as the single canonical ingress bot for this production chat

### Execution implications

- [x] Teach diagnostics to distinguish unknown external bot mentions from known env apps
- [x] Teach diagnostics to isolate a single target chat during audits
- [ ] Remove the unknown external bot mention path from user-visible workflow in `CEO 工作汇报`
- [ ] Stop app2 (`网络营销专家`) from serving as the active replying bot in `CEO 工作汇报`
- [ ] Keep app3 (`hermes agent`) as the only retained replying bot before re-running end-to-end KPI validation

## 12. 2026-04-21 17:15 Chat Topology Evidence Update

- The focused delivery diagnostic now also inspects target-chat topology:
  - chat detail from `GET /open-apis/im/v1/chats/{chat_id}`
  - tenant-view member list from `GET /open-apis/im/v1/chats/{chat_id}/members`
  - optional user-view member list from `FEISHU_USER_ACCESS_TOKEN`
- Latest target-chat topology for `oc_ec86c28e66596c25377aff2ee028901c`:
  - `bot_count=8`
  - `user_count=1`
  - tenant-view members returned: `1`
  - tenant-view members currently visible:
    - `素依`
  - therefore tenant-view member enumeration is currently incomplete relative to the chat topology
- Current user-token state:
  - `FEISHU_USER_ACCESS_TOKEN` exists in env
  - current token is expired
  - user-view member enumeration now fails with:
    - `99991677 Authentication token expired. Please request a new one.`

### Tightened operational conclusion

- The current blocking problem is no longer just “wrong bot replied”.
- The target production chat is now proven to be a **multi-bot contention group**:
  - at least `8` bots are present
  - app2 (`网络营销专家`) is still the currently observed replying bot
  - tenant-view membership cannot currently enumerate all bot identities
  - user-view membership cannot currently be used until the user token is refreshed
- Therefore, before KPI re-validation, the chain must satisfy all of:
  - app3 remains in the target chat
  - conflicting bots are removed or operationally silenced
  - user token is refreshed so the final member list can be audited from a human view

### New completed items

- [x] Add target-chat topology inspection to `scripts/check_feishu_delivery_path.py`
- [x] Detect multi-bot contention from live chat detail (`bot_count`)
- [x] Detect incomplete tenant-view member enumeration
- [x] Detect expired `FEISHU_USER_ACCESS_TOKEN` during the same focused diagnostic run

### Remaining execution items

- [ ] Refresh `FEISHU_USER_ACCESS_TOKEN` and re-run focused member enumeration
- [ ] Enumerate the full 8-bot membership list from a user view
- [ ] Remove or silence non-app3 bots in `CEO 工作汇报`
- [ ] Re-send a fresh real user message only after app3 becomes the sole retained replying bot
  - `2026-04-21 14:10:32 +08:00` user message
- Those recent real-user messages were followed by bot replies from:
  - `cli_a95274809bb95bda`
- This replying bot is **not** app3 (`cli_a9525a47e4f99bc2`).
- Interpretation:
  - the latest real chat traffic is still being routed to another bot in the same group
  - continuing to debug app3-only ingress from those messages would be a false positive path

### Important API clarification discovered today

- `GET /open-apis/application/v6/applications/{app_id}` for app3 still reports:
  - `callback_info.request_url = https://hermes.isuyee.com/`
  - `subscribed_callbacks = ["card.action.trigger"]`
- But `GET /open-apis/application/v6/applications/{app_id}/app_versions` for app3 proves the currently published version already contains:
  - `im.message.receive_v1`
  - `application.bot.menu_v6`
- So:
  - `application/v6.app.callback_info` is confirmed non-authoritative for the full published event set
  - it may still be useful for callback URL clues, but not for deciding whether `im.message.receive_v1` is actually enabled
- The published app3 version currently does **not** show `im.message.message_read_v1`, so read-receipt callbacks still need subscription-level enablement on the published app version.

### Failed automation attempt worth recording

- A direct `PATCH /open-apis/application/v6/applications/{app_id}` attempt returned `code=0 success`
- However a follow-up `GET` showed no effective change in `callback_info`
- Working inference:
  - this endpoint either ignores those fields in the current state
  - or changes land in a hidden draft/publish flow instead of the active online configuration
  - so we cannot currently treat this API as a reliable one-step repair for app3 event subscription routing

### Updated P0 next actions

- [x] Implement runtime `message_read` correlation in the repository
- [x] Add targeted regression coverage for read-receipt correlation
- [ ] Add `im.message.message_read_v1` to the published app3 event subscription set
- [ ] Ensure the target group’s real user messages hit app3 instead of `cli_a95274809bb95bda`
- [ ] Re-run live ingress verification only after the target chat is actually bound to app3 traffic
- [ ] Re-measure the fresh chain for:
  - `webhook.accepted`
  - `webhook.ack`
  - `webhook.message_read.correlated`
  - `gateway.send.done`
  - remote Modal app `hermes-agent` does not currently expose `debug_feishu_perf_summary`
- Result:
- KPI/cost report automation in the main repo is still blocked until repo/deployment parity is restored
  - this is now an explicit task-list item, not an implicit hidden failure

## 9. 2026-04-21 14:30 Recent 50-Minute Cross-Validation Update

- Fresh evidence window:
  - local validation window start: `2026-04-21 13:36:58 CST`
  - validation completed around: `2026-04-21 14:30 CST`
- Target production app identity:
  - `app_id=cli_a9525a47e4f99bc2`
  - `app_name=hermes agent`
  - bot open_id: `ou_c07fb4b6bcc81c5951bbde1933383088`
- Recent target-chat messages that are now confirmed visible via Feishu API pagination:
  - `2026-04-21 13:48:28 CST`
    - user text: `介绍一下自己`
    - no mention target
  - `2026-04-21 13:51:18 CST`
    - user text: `请介绍一下你自己，你的核心竞争力是什么`
    - no mention target
  - `2026-04-21 14:10:32 CST`
    - user text with mention placeholder
    - mention target open_id: `ou_26c2a287784865ebc498643162c3f223`
    - this is **not** the production app3 bot open_id
- Cross-validated Cloudflare evidence for the same 50-minute window:
  - Worker observability count: `1`
  - the only event is a `POST /feishu/webhook -> 200`
  - user agent: `python-requests/2.33.1`
  - this matches our own diagnostics/probe traffic, not a real Feishu inbound message delivery
- Cross-validated Workflow evidence for the same 50-minute window:
  - new `hermes-feishu-agent-workflow` instances: `0`
- Cross-validated Modal evidence:
  - latest Feishu success in current artifacts is still the older repaired session
  - no new Feishu dispatch/send lines exist after the earlier successful `2026-04-21 13:30:30 CST` send completion

### Current Root Cause Refinement

- The immediate reason the two newly sent messages did not reach Hermes is **not** `Worker -> Workflow -> Modal` runtime loss.
- The current live blocker is upstream of Worker ingress:
  - the two plain text messages did not target the current production bot at all
  - the later mentioned message targeted a **different bot open_id**, not the retained production bot
- Therefore the current live production failure mode is:
  - `wrong_bot_target_or_no_target` before webhook ingress
- This blocker coexists with the previously observed app configuration drift:
  - app3 callback URL still reports root `/` instead of `/feishu/webhook`
  - app metadata still reports only `card.action.trigger`

### New Immediate Tasks

- [x] Prove recent user messages exist in the target chat during the failing window
- [x] Prove recent user messages did not target the retained production bot open_id
- [x] Prove the latest mentioned message targeted another bot open_id instead
- [x] Prove there was no matching Worker or Workflow creation for those messages
- [ ] Identify which concrete bot/app owns open_id `ou_26c2a287784865ebc498643162c3f223`
- [ ] Converge the group to a single canonical Hermes bot identity and remove or de-prioritize stale sibling bots
- [ ] Ensure test and operator instructions explicitly require either:
  - direct chat to canonical Hermes bot, or
  - explicit mention of the canonical bot open_id/name in group chats
- [ ] After identity convergence, repeat a fresh real-user mention test and confirm new Worker ingress plus Workflow creation

## 12. 2026-04-21 20:55 Control Command Direct-Routing Update

- Direct 1:1 app3 文本主链当前已通过 live diagnose 反复确认：
  - 目标会话：`oc_33edcba53ad086b50f175352947750eb`
  - 最新非 synthetic Workflow 实例持续 `complete`
  - 关键步骤 `edge-agent-plan` / `cf-ai-exec` / `send-feishu-response` 均成功
- 新发现并持续存在的控制链路根因：
  - live Modal 仍在运行旧部署产物
  - `/internal/feishu/session-control` 的 `dispatch_command` 命中 `/root/modal_.py:8042`
  - 旧产物会直接抛出：
    - `ImportError: cannot import name 'build_session_start_skills_message' from 'agent.skill_commands'`
- 为避免旧 Modal 控制产物继续阻塞飞书卡片/文本命令，本轮已在 Cloudflare Worker 直接改路：
  - 命令类控制事件不再先调用坏掉的 `/internal/feishu/session-control`
  - 直接改走健康的 `/internal/feishu/agent-exec`
  - 使用合成 `message_type=command` 的内部事件继续执行 `/provider`、`/model`、`/personality`、`command_run`
- 已完成代码与验证：
  - Worker 源码补丁：
    - `cloudflare/feishu-gateway/src/gateway/control-handlers.ts`
  - Worker 回归测试：
    - `cloudflare/feishu-gateway/test/control-handlers.test.ts`
    - `cloudflare/feishu-gateway/test/modal-client.test.ts`
    - `cloudflare/feishu-gateway/test/feishu-internal-contract.test.ts`
  - 测试结果：
    - `vitest` 相关 5 项全部通过
    - Python 诊断回归 `9 passed`
- 已完成线上发布：
  - Worker：`hermes-feishu-gateway`
  - Version ID：`c094119b-fe3c-404f-8e30-e1eb8abc93f5`
- 线上受控探针结果：
  - 直接向 Worker 发送合成 `card.action.trigger`（`/status`、`/provider`）均返回 `{"code":0,"msg":"accepted"}`
  - 重新发布后的最近 3 分钟 Modal 日志中，不再出现对应控制探针触发的 `session-control ImportError 500`
- 当前状态调整：
  - [x] 直接消息主链恢复并可持续收口
  - [x] 命令类控制事件已直接绕开旧 Modal `session-control`
  - [ ] 真实飞书卡片点击/菜单触发的线上验收仍需继续留痕验证
  - [ ] `activate_skill_combo` 在 live Modal 旧产物下仍建议继续做专门修复或重部署

## 13. 2026-04-21 21:35 Control Skill Combo + KPI Tooling Closure Update

- 本轮新增 Worker 侧兼容修复已完成并上线：
  - `skill_combo_apply` 不再依赖 live Modal 已漂移的 `/internal/feishu/session-control`
  - 直接改走健康的 `/internal/feishu/agent-exec`
  - 使用卡片 payload 自带的：
    - `combo_label`
    - `skills`
    - `suggested_personality`
  - 在 Worker 侧合成技能切换说明文本，避免旧 Modal 产物里的 `ImportError`
- 已完成代码与回归：
  - Worker 源码：
    - `cloudflare/feishu-gateway/src/gateway/control-handlers.ts`
  - Worker 测试：
    - `cloudflare/feishu-gateway/test/control-handlers.test.ts`
  - KPI 报表与读回执兼容层：
    - `scripts/feishu_triparty_pk_report.py`
    - `tests/test_feishu_triparty_pk_report.py`
  - 回归结果：
    - Worker `vitest`：`6 passed`
    - Python KPI 回归：`25 passed`
- 已完成线上发布：
  - Worker：`hermes-feishu-gateway`
  - Version ID：`034a6152-a944-4919-96e2-087d03489156`
- 已完成线上主链复核：
  - `scripts/feishu_cf_modal_live_diagnose.py --chat-filter oc_33edcba53ad086b50f175352947750eb --limit 12`
  - 最新选中真实实例：
    - `feishu_oc_33edcba53ad086b50f175352947750eb_d5ca8b5fd8852a26058cf`
    - `status=complete`
    - `is_synthetic=false`
  - Workflow 步骤保持全成功：
    - `emit-ingress-log`
    - `site-prefetch`
    - `edge-agent-plan`
    - `cf-ai-exec`
    - `send-feishu-response`
    - `emit-final-log`
- KPI/报表工具链本轮也已完成两项去噪修复：
  - `read_users` API 兼容：
    - 先识别 Feishu 要求 `user_id_type`
    - 兼容 `data.items` / `data.read_users`
  - 三方 PK 报表默认优先使用 canonical `FEISHU_APP_ID3 / FEISHU_APP_SECRET3`
  - 非致命 `read_users` 边界错误已从误报型数据缺口中剔除：
    - `230012 Bot is NOT the sender of the message`
    - `99992354 id not exist`
- 当前 12h 真实报表状态（`2026-04-21 21:37 CST`）：
  - 已达成：
    - 扣除 AI 后正式回复 p90：`7594.1 ms < 20000 ms`
  - 未达成：
    - 已读时延：`None`
    - 空闲小时成本 p90：`$0.01017350 > $0.005`
    - 其它缓存/浏览器预处理/能力匹配指标：当前仍无足够有效样本
  - 当前剩余真实数据缺口仅剩：
    - `cloudflare_interval_only_join`
    - `official_cost_without_matching_function_trace`
- 受控控制探针补充说明：
  - 已尝试直接向 Worker 发送合成 `card.action.trigger` 做 `skill_combo_apply` 验证
  - Worker 返回：
    - `401 invalid verification token`
  - 说明当前线上 webhook 入口仍要求正确 Feishu verification token
  - 因本地环境未暴露该 secret，该步骤不能伪造飞书事件完成最终线上留痕

### Status changes after this round

- [x] `skill_combo_apply` 已在 Worker 侧完成无损降级并上线
- [x] 控制链路不再被 live Modal 旧 `session-control` 产物阻塞
- [x] 三方 PK 报表可稳定运行且已剔除大部分 `read_users` 误报噪音
- [x] 主链 direct DM 文本会话继续保持真实实例全成功
- [ ] 真实飞书卡片点击/菜单触发的线上留痕仍需一次真实用户操作验证
- [ ] `im.message.message_read_v1` 的真实回执样本仍未形成有效 KPI 窗口
- [ ] Cloudflare Worker 可观测与 Modal function trace 仍未完成精确 join
- [ ] 空闲小时成本与全量验收指标仍未全部达标
## 14. 2026-04-22 07:58 Task checklist follow-up

- Closed in this round:
  - [x] KPI report no longer misclassifies session cost when only official billing exists without matching function trace allocation
  - [x] Worker session rows now carry back `cache_status` from Cloudflare observability
  - [x] Negative placeholder AI Gateway cost values are excluded from KPI aggregation
  - [x] Stateless cache-key policy was updated and redeployed on Worker version `cee15794-07eb-41b9-b127-c8beed86f813`
  - [x] Current WAE evidence shows no recent `feishu.send.operation.error`
- Still open after this round:
  - [ ] Need fresh real repeated stateless DM samples after Worker version `cee15794-07eb-41b9-b127-c8beed86f813` to validate cache eligible hit rate `> 30%`
  - [ ] Need fresh 24h window to validate idle hourly cost after the latest Modal downsizing fully ages into billing data
  - [ ] Need a reliable replacement for Modal cross-function trace persistence, or an explicit decision to treat WAE as the primary long-term source of truth
  - [ ] Need one real card/menu sample each for model switch, personality switch, and skill combo on the repaired direct-routing path
