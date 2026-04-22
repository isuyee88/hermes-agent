# CF AI Gateway 排查、测试、修复验证任务清单

更新时间：2026-04-19（UTC+8）

## 1. 当前已验证事实

### 1.1 本次真实排查窗口

- 日志排查窗口：2026-04-16 00:50 至 2026-04-19 00:50（UTC+8）
- 网关：`affiliate-manager`
- 账号：`d1215a30b84b673ef0367010b0e78c10`
- 数据来源：
  - Cloudflare AI Gateway Logs API
  - Cloudflare AI Gateway Dynamic Routes API
  - Cloudflare AI Gateway Provider Configs API
  - 本仓库运行时与 Worker 配置

### 1.2 72 小时日志结论

- 总日志数：363
- 错误日志数：51
- 最近 24 小时错误数：21
- 错误状态分布：
  - `400`: 34
  - `403`: 8
  - `404`: 4
  - `502`: 2
  - `401`: 2
  - `429`: 1

### 1.3 “是否已修复”当前结论

- 结论：`未完全修复`
- 理由：
  - 2026-04-19 00:17（UTC+8）仍有 `codex-rate-openrouter-probe` / `codex-rate-nvidia-probe` 400
  - 2026-04-19 00:13（UTC+8）仍有 `custom-nvidia::meta/llama-3.1-8b-instruct` 404
  - 2026-04-18 23:54（UTC+8）仍有 `nvidia-integrate::meta/llama-3.1-8b-instruct` 400
  - 2026-04-18 23:51（UTC+8）仍有 `openrouter/free` 502，响应摘要为 `Invalid URL`
  - 2026-04-19 22:48（UTC+8）已将线上 `codex-rate-openrouter-probe` 切换为 `openrouter / nvidia/nemotron-nano-9b-v2:free`，但受当前网络边界 `1010` 阻断，尚不能从本机直连完成 3 次连续成功复验

### 1.4 当前核心动态路由定义

- `affiliate-general`
  - primary: `custom-bigmodel / glm-4-flash`
  - fallback: `openrouter / nvidia/nemotron-nano-9b-v2:free`
- `affiliate-coding`
  - primary: `openrouter / google/gemma-4-31b-it:free`
  - fallback: `openrouter / openai/gpt-oss-20b:free`
- `affiliate-tools`
  - primary: `openrouter / openai/gpt-oss-20b:free`
  - fallback: `nvidia / openai/gpt-oss-20b`
- `codex-rate-openrouter-probe`
  - `rate(10/60, key=metadata.gateway_route_name)` -> `openrouter / nvidia/nemotron-nano-9b-v2:free`
- `codex-rate-nvidia-probe`
  - `rate(10/60, key=metadata.gateway_route_name)` -> `nvidia / meta/llama-3.1-8b-instruct`
- `codex-nvidia-integrate-probe`
  - `nvidia-integrate / meta/llama-3.1-8b-instruct`

### 1.5 当前网关/Worker 已启用能力

- 网关全局限流：`50 req / 60s / fixed`
- 网关默认缓存 TTL：`300s`
- 网关认证：开启
- 网关日志采集：开启
- 网关级 retry 参数：未显式设置
- Worker 侧补充限流：
  - `openrouter`: `45 req / 60s / sliding / key=metadata.gateway_route_name`
  - `nvidia`: `20 req / 60s / sliding / key=metadata.gateway_route_name`

### 1.6 真实故障簇

- 故障簇 A：新建 probe 动态路由零成功、立即 400
  - `codex-rate-openrouter-probe`
  - `codex-rate-nvidia-probe`
  - `codex-rate-limit-nvidia-probe`
- 故障簇 B：`nvidia-integrate` / `custom-nvidia` / `custom-nvidia-integrate` 路由与模型映射不稳定
  - `400/404` 持续出现
- 故障簇 C：`affiliate-tools` fallback 到 `nvidia::openai/gpt-oss-20b` 时存在批量 400
  - 但同一路由也存在成功日志，说明是“部分退化”而不是“全量不可用”
- 故障簇 D：`openrouter/free` 上游存在少量 502
  - 响应摘要显示 `Invalid URL`

## 2. POLANYI 默会知识原则

以下原则必须写进排查、测试、回归、告警和变更流程，而不是只停留在口头经验。

- 原则 1：新建动态路由若“零成功 + 首批即 400”，优先判定为“路由定义/Provider 映射/模型兼容性问题”，不要先归因给瞬时网络抖动。
- 原则 2：同一路由出现“错误后很快又成功”，只能说明系统具备部分自恢复或 fallback 生效，不能判定“已修复”。
- 原则 3：工具型请求、图片型请求、编码型请求的 payload 结构不同，不能假设同一 provider/model 对三类请求都兼容。
- 原则 4：Cloudflare AI Gateway metadata 只保留前 5 项，必须固定优先级，避免关键字段被低价值标签挤掉。
- 原则 5：`request_head` 可见但 `response_head` 缺失时，必须联动 Worker telemetry、Correlation ID、route version 一起定位，不能只看单条日志。
- 原则 6：凡是 probe 路由失败，都应被视为“上线前阻断信号”，而不是“测试噪声”。
- 原则 7：限流必须同时验证“固定窗口”和“滑动窗口”的真实效果差异，不能只验证是否返回 429。
- 原则 8：所有修复必须附带“回归样本、回退条件、失败特征、复验窗口”四要素，否则不能关单。

## 3. PDCA 总目标

- 目标 1：清零当前已识别的真实故障簇 A/B/C/D
- 目标 2：覆盖 Cloudflare AI Gateway 现有能力与可预见组合，形成不少于 10 组动态路由验证
- 目标 3：建立“修复 -> 复测 -> 回归 -> 再观测”的闭环
- 目标 4：达到功能完整、性能优异、体验良好，并具备持续演进能力

## 4. 任务清单

### P. Plan

- [x] P1. 固化排查基线
  - 输出 72h/24h/1h 三个窗口的错误统计、Top provider/model、Top route、Top request class
  - 退出标准：同一口径脚本可重复运行，输出稳定
- [x] P2. 固化路由资产台账
  - 盘点 gateway settings、provider configs、active routes、route versions、deployments
  - 退出标准：每条线上路由都能定位到 version_id 和 deployment_id
- 已完成产物：
  - `scripts/cf_ai_gateway_audit.py`
  - `.tmp-cf-ai-gateway-audit.json`
  - `.tmp-cf-ai-gateway-audit.md`
- [ ] P3. 建立“故障 -> 假设 -> 验证”映射
  - A 类：probe 路由 400
  - B 类：nvidia-integrate / custom-nvidia* 400/404
  - C 类：affiliate-tools fallback 400
  - D 类：openrouter/free 502
  - 退出标准：每类故障至少有 1 个明确验证动作
- [ ] P4. 定义统一通过门槛
  - 功能：无 P0/P1 功能缺失
  - 稳定性：目标路由连续通过回归
  - 性能：P95 延迟、错误率、fallback 触发率达标
  - 观测性：每类失败都能从日志和 telemetry 重建调用路径

### D. Do

- [x] D0. 建立主动审计/探针脚本
  - 已新增：
    - `scripts/cf_ai_gateway_audit.py`
    - `scripts/cf_ai_gateway_validate_routes.py`
  - 当前发现：
    - 本机直连 Gateway 主动探针被 Cloudflare `403 / error code 1010` 阻断
    - Worker 配置中的 `text-general` / `text-coding` 与线上真实路由名不一致
- [x] D0.1 修复 Worker 文本动态路由名错配
  - 已修改：
    - `cloudflare/feishu-gateway/wrangler.jsonc`
    - `cloudflare/feishu-gateway/src/gateway/routing.ts`
    - `cloudflare/feishu-gateway/src/gateway/route-policy.ts`
  - 修复结果：
    - 审计脚本复跑后 `route_alignment.missing_routes = []`
    - `wrangler.jsonc` 解析校验通过
  - 回归结果：
    - `tests/test_feishu_triparty_tail.py` 通过
    - `tests/test_feishu_triparty_pk_report.py` 通过
    - `tests/test_modal_deployment.py` 存在仓库原有失败，当前未归因于本次修改
- [ ] D1. 修复 probe 路由定义与 provider/model 兼容性
  - 优先处理：`codex-rate-openrouter-probe`
  - 优先处理：`codex-rate-nvidia-probe`
  - 优先处理：`codex-nvidia-integrate-probe`
  - 要求：每条 probe 路由至少出现 3 次连续成功
  - [x] D1.1 已通过 Cloudflare Logs Detail / Request / Response API 取证，确认线上 rate 节点实际使用 `type=rate` + `window`，本轮不再按网页文档误改 schema
  - [x] D1.2 已修正线上 `codex-rate-nvidia-probe` provider：`custom-nvidia-integrate` -> `nvidia`（先去除错误 slug，再回退到已由原生接口对照验证可用的健康 provider）
    - route_id=`de35189a-7391-458b-93ac-1813fc3a621d`
    - version_id=`759875f0-e7d0-452c-923f-ee62dec688b4`
    - deployment_id=`59b56c1f-96dc-4849-80f5-b64ab7f49f55`
  - [x] D1.3 已确认 `codex-rate-openrouter-probe` 原模型 `google/gemma-4-31b-it:free` 在当前窗口直连 OpenRouter 返回 `403 / PERMISSION_DENIED`（`Google AI Studio`），已将线上 probe 路由切换为 `openrouter / nvidia/nemotron-nano-9b-v2:free`
    - version_id=`fe37b660-b90f-46c4-b1ce-40f40846b228`
    - deployment_id=`a95eb346-10d2-4630-b864-a941fbca083b`
  - [ ] D1.4 待继续复测 `codex-rate-openrouter-probe` / `codex-rate-nvidia-probe` / `codex-nvidia-integrate-probe`
- [ ] D2. 修复 `affiliate-tools` fallback 退化
  - 重点核查：`nvidia::openai/gpt-oss-20b` 是否支持当前工具型请求载荷
  - 重点核查：image/tool-heavy 请求是否错误落到 `affiliate-tools`
  - 重点核查：primary 成功条件、fallback 触发条件、fallback 后 payload 是否被改写
  - [x] D2.1 已从日志明细确认：`affiliate-tools` 的 `nvidia::openai/gpt-oss-20b` 400 发生在 `task_kind=image` + `tools_required=True` 的大 payload 上
  - [x] D2.2 已在本地代码中收敛风险：`image + tools` 默认不再走 `affiliate-tools`，而是回到 image/general 路由，除非显式配置 `CLOUDFLARE_AI_GATEWAY_ROUTE_TOOLS_IMAGE`
- [ ] D3. 修复 nvidia 系 provider alias/slug/模型映射
  - 比对：`nvidia`、`nvidia-integrate`、`custom-nvidia`、`custom-nvidia-integrate`
  - 比对：Cloudflare provider_configs 与动态路由 provider 名是否完全一致
  - 比对：模型名在对应 provider 下是否真实存在、是否支持当前 endpoint
  - [x] D3.1 已核对 Cloudflare `provider_configs` 与 `custom-providers`：存在 `nvidia-integrate` / `nvidia`，不存在 `custom-nvidia-integrate`
  - [x] D3.2 已修正本地发布映射逻辑与 Worker 配置：默认保留原生 `nvidia`，仅在显式配置时才映射到 `nvidia-integrate`
    - `cloudflare/feishu-gateway/src/model-catalog/routes.ts`
    - `cloudflare/feishu-gateway/wrangler.jsonc`
  - [x] D3.3 已补回归测试覆盖“原生 `nvidia` 默认保留 + 显式 override 才走 `nvidia-integrate`”
  - [x] D3.4 已通过对照验证确认：`meta/llama-3.1-8b-instruct` 直连 NVIDIA 原生 `chat/completions` 返回 `200`，因此当前 `nvidia-integrate` 400 属于 Cloudflare 接入层兼容性问题，而非模型本身不可用
- [ ] D4. 修复上游 502 与无效 URL 问题
  - 重点核查：`openrouter/free` 的上游 vendor 配置
  - 重点核查：请求是否遗漏 base URL / provider alias / model 解析
  - 要求：新增健康检查与自动隔离策略
  - [x] D4.1 已取证 `openrouter/free` 当前 502 为上游 `Stealth` 返回 `Invalid URL`
  - [x] D4.2 已在本地动态 free 路由选择中改为“优先 concrete free model、`openrouter/free` 仅保底”，避免继续把不稳定别名作为首选
  - [x] D4.3 已将主动验证脚本切换到 `cf-aig-authorization` 头，避免脚本鉴权方式干扰排查结论
- [ ] D5. 增强网关回退与隔离
  - 为高风险 provider 增加 route version 回滚与临时摘流方案
  - 为连续 4xx/5xx provider 增加熔断/冷却时间
  - 为 429 增加显式 backoff 验证
- [ ] D6. 增强日志与关联追踪
  - 固化 metadata 优先级：`correlation_id` / `task_kind` / `hermes_request` / `tools_required|session_id` / `tool_names`
  - 将 route version、provider alias、fallback reason 纳入可追踪字段
  - 要求：单次失败能串起 Gateway 日志与 Worker telemetry

### 当前实施进展

- 已完成：
  - 基线审计自动化
  - 路由/Provider 资产盘点自动化
  - 主动动态路由探针脚本
  - 本地回归：`cloudflare/feishu-gateway/test/model-catalog-routes.test.ts` 3/3 通过
  - 本地回归：`tests/run_agent/test_run_agent.py` 中新增 image-tools 路由收敛用例 3/3 通过（`-n 0`）
  - 本地回归：`tests/test_model_catalog.py` 2/2 通过
  - 本地回归：`tests/test_modal_deployment.py -k "candidate_routes_from_state_prefers_runtime_binding or transient_error_triggers_retry_route_refresh or session_route_lease_sticky_hit"` 3/3 通过
  - 本地回归：`cloudflare/feishu-gateway/test/model-catalog-routes.test.ts` 3/3 通过
  - 线上修复：`codex-rate-openrouter-probe` 已切换到 `openrouter / nvidia/nemotron-nano-9b-v2:free`
  - 线上修复：`codex-rate-nvidia-probe` 已切换到 `nvidia / meta/llama-3.1-8b-instruct`
- 已识别新增问题：
  - `wrangler.jsonc` 原先引用的 `text-general` / `text-coding` 不存在于当前线上 Gateway 路由列表
  - 该命名错配已在配置层和代码兜底层修正为 `affiliate-general` / `affiliate-coding`
  - 本机直连 Gateway 主动探针仍被 `403 / error code 1010` 阻断，后续需补可在当前网络边界内执行的验证通道
  - Cloudflare AI Gateway 网页文档与 OpenAPI/线上实况在 rate 节点 schema 上存在冲突，本轮以线上实况和 Logs/Route Detail API 为准
  - `custom-nvidia` 404 的响应体已取证为 `404 page not found`
  - `openrouter/free` 502 的响应体已取证为 `Invalid URL:`，provider metadata 显示上游为 `Stealth`
  - 主动验证脚本在切换到 `cf-aig-authorization` 后再次复测 `codex-rate-openrouter-probe` / `codex-rate-nvidia-probe` / `codex-nvidia-integrate-probe`，结果仍全部为 `403 / error code 1010`，说明当前剩余复验阻塞来自本地网络边界而非脚本头部错误
  - `google/gemma-3-27b-it:free` / `google/gemma-4-31b-it:free` 在当前窗口直连 OpenRouter 分别返回 `400 / FAILED_PRECONDITION` 与 `403 / PERMISSION_DENIED`，说明 `Google AI Studio` 免费线路在当前地域/项目条件下不适合作为 probe 首选
  - `meta/llama-3.1-8b-instruct` 直连 NVIDIA 原生接口返回 `200`，但 Cloudflare `nvidia-integrate` 仍返回 `400`，当前继续按 Cloudflare 接入层兼容性缺陷跟踪

### C. Check

- [ ] C1. 执行 10 组动态路由测试
- [ ] C2. 对每组测试输出结果
  - 功能正确性
  - 失败类型
  - fallback 是否符合预期
  - 限流是否符合预期
  - 缓存是否符合预期
  - 日志是否完整
- [ ] C3. 回归窗口复查
  - 修复后 1 小时
  - 修复后 24 小时
  - 修复后 72 小时
- [ ] C4. 指标门禁校验
  - 错误率
  - P50/P95 延迟
  - 缓存命中率
  - fallback 触发率
  - 429 命中率
  - 关键 probe 成功率

### A. Act

- [ ] A1. 将本轮有效修复固化为默认 route version
- [ ] A2. 将失败修复回滚规则写入运行手册
- [ ] A3. 将 tacit knowledge 固化进自动化测试、告警阈值、变更前检查项
- [ ] A4. 关闭条件复核
  - 72h 无新增已知故障模式
  - 10 组测试全部通过
  - 关键生产路由连续稳定
  - 日志、路由、provider、回退链路可解释

## 5. 10 组 Cloudflare 网关动态路由测试矩阵

### T01. General 路由基础可用性

- 路由：`affiliate-general`
- 组合：动态路由 + primary/fallback + 认证 + 日志
- 验证点：
  - primary 可成功返回
  - fallback 只在 primary 失败时触发
  - metadata 中 `task_kind=general`
  - 日志字段完整

### T02. Coding 路由模型切换与回退

- 路由：`affiliate-coding`
- 组合：动态路由 + coding 分类 + fallback
- 验证点：
  - `task_kind=coding` 时正确命中
  - primary `gemma-4-31b-it:free` 可用
  - fallback `gpt-oss-20b:free` 可用
  - 失败后不会错误落到 general

### T03. Tools 路由主链路验证

- 路由：`affiliate-tools`
- 组合：动态路由 + tools_required + skip cache + metadata
- 验证点：
  - tool-heavy 请求默认走 tools 路由
  - `cf-aig-skip-cache=true`
  - tool_names 被正确截断并写入 metadata
  - primary `openrouter/openai/gpt-oss-20b:free` 正常

### T04. Tools 路由 fallback 兼容性

- 路由：`affiliate-tools`
- 组合：dynamic route + fallback + provider compatibility
- 验证点：
  - fallback `nvidia/openai/gpt-oss-20b` 是否真正支持当前 payload
  - 对 tool/image 请求的 system prompt、messages、max_tokens 是否兼容
  - 不兼容时必须返回可解释失败并触发摘流

### T05. Text Plain 分类与 Worker 侧动态执行

- 路由：`text-general`
- 组合：request_class=text_plain + Worker 侧 route policy + dynamic route execution
- 验证点：
  - `HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED=true` 时命中正确
  - misroute 不得静默通过
  - request_classification_error 可被识别并记录

### T06. Text Coding 分类与错路由防护

- 路由：`text-coding`
- 组合：request_class=text_coding + strict route guard
- 验证点：
  - coding 请求不误入 text-general
  - route-policy 中 misroute_detected 逻辑正常
  - 发生 misroute 时回退到安全路径而非错误 provider

### T07. Rate 节点 + OpenRouter 路由验证

- 路由：`codex-rate-openrouter-probe`
- 组合：rate node + dynamic route + OpenRouter provider
- 验证点：
  - 10/60 限流规则在 gateway_route_name 维度生效
  - 未超限时必须成功
  - 超限后行为必须可预测，且日志明确标记

### T08. Rate 节点 + Nvidia/Custom Provider 路由验证

- 路由：`codex-rate-nvidia-probe`
- 组合：rate node + dynamic route + custom provider alias
- 验证点：
  - `custom-nvidia-integrate` 与 provider_configs 对齐
  - 模型 `meta/llama-3.1-8b-instruct` 在对应 provider 下真实可调
  - 未超限前不能出现 400/404

### T09. Gateway 固定限流 vs Worker 滑动限流联测

- 路由：任选 `affiliate-*` + `text-*`
- 组合：
  - 网关固定限流 `50/60/fixed`
  - Worker 补充限流 `openrouter 45/60/sliding`
  - Worker 补充限流 `nvidia 20/60/sliding`
- 验证点：
  - 固定窗口与滑动窗口行为差异符合预期
  - provider-route 粒度不会互相污染
  - 429 与 fallback 的先后关系清晰

### T10. 故障注入与回归闭环

- 路由：`affiliate-general` / `affiliate-coding` / `affiliate-tools`
- 组合：4xx/5xx/429 注入 + retry/backoff + fallback + observability
- 验证点：
  - 400：配置/模型错误能快速暴露
  - 404：provider/model 不存在能被识别
  - 429：backoff 和限流说明准确
  - 502：provider 隔离/回退链路生效
  - 修复后 1h/24h/72h 回归通过

## 6. 未来扩展测试维度

以下维度即使当前未全部启用，也必须预留到测试模板中。

- Conditional 节点
- Percentage / A-B rollout
- Budget limit 节点
- 多 provider alias 共存
- 多模型同名但不同 provider 语义
- cache on / cache off / prompt cache key
- metadata 高低价值字段争抢
- authenticated / unauthenticated gateway
- log payload on / off
- DLP action
- byok / wholesale / compatibility mode
- route version rollback / fast-forward

## 7. 立即优先级

### P0

- [ ] 先修复并复测 `codex-rate-openrouter-probe`
- [ ] 先修复并复测 `codex-rate-nvidia-probe`
- [ ] 先修复并复测 `codex-nvidia-integrate-probe`
- [ ] 核查 `affiliate-tools` fallback 到 `nvidia::openai/gpt-oss-20b` 的 400 根因

### P1

- [ ] 核查 `openrouter/free` 的 `Invalid URL` 502
- [ ] 核查 `custom-nvidia` / `nvidia-integrate` / `custom-nvidia-integrate` provider 名与 alias 映射
- [ ] 增加 provider 健康摘流与恢复机制

### P2

- [ ] 自动化输出 route x provider x status x latency x fallback 报表
- [ ] 建立 72h 连续观察看板

## 8. 关闭定义

只有在以下条件全部满足时，本任务才可标记“完成”。

- [ ] 最近 72 小时没有复现已知故障簇 A/B/C/D
- [ ] 10 组动态路由测试全部通过
- [ ] 核心路由 `affiliate-general` / `affiliate-coding` / `affiliate-tools` 连续稳定
- [ ] probe 路由全部至少 3 次连续成功
- [ ] gateway 全局限流与 Worker 补充限流都按预期生效
- [ ] 所有失败模式都能从日志与 telemetry 中解释
- [ ] 已将本轮 tacit knowledge 反写进测试、手册、告警

## 9. 代码与配置触点

- 路由选择、缓存、metadata、header 注入：
  - [run_agent.py](D:/suyee/github/hermesagent/hermes-agent/run_agent.py#L1658)
- Worker 侧动态路由执行与分类：
  - [cf-ai-exec.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/gateway/cf-ai-exec.ts)
  - [route-policy.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/gateway/route-policy.ts)
  - [routing.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/gateway/routing.ts)
- Worker 侧 provider-route 限流配置：
  - [wrangler.jsonc](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/wrangler.jsonc#L30)

## 10. 官方参考

- Cloudflare AI Gateway Dynamic routing:
  - <https://developers.cloudflare.com/ai-gateway/features/dynamic-routing/>
- Cloudflare AI Gateway Rate limiting:
  - <https://developers.cloudflare.com/ai-gateway/features/rate-limiting/>
- Cloudflare AI Gateway Logging:
  - <https://developers.cloudflare.com/ai-gateway/observability/logging/>
- Cloudflare AI Gateway Observability:
  - <https://developers.cloudflare.com/ai-gateway/observability/>
- Cloudflare AI Gateway Limits:
  - <https://developers.cloudflare.com/ai-gateway/reference/limits/>
- Cloudflare AI Gateway Logs API:
  - <https://developers.cloudflare.com/api/resources/ai_gateway/subresources/logs/>
- Cloudflare AI Gateway Dynamic Routing API:
  - <https://developers.cloudflare.com/api/resources/ai_gateway/subresources/dynamic_routing/>

## 11. 2026-04-19 Latest Implementation Update

- [x] E1. Enforced `modal + hermes` runtime path for `openrouter` / `nvidia` to route through CF AI Gateway instead of direct provider URLs
  - `internal/modal_runtime_env.py`
  - `internal/session_routes.py`
  - `internal/model_catalog.py`
  - `gateway/run.py`
  - `modal_.py`
- [x] E2. Tightened dynamic route materialization so provider candidates without runtime gateway binding are skipped instead of falling back to direct provider endpoints
- [x] E3. Updated route probe behavior so gateway probes use `cf-aig-authorization` when probing enforced providers through CF AI Gateway
- [x] E4. Added / updated regression coverage for the CF-only invariant
  - `tests/test_modal_deployment.py`
  - `tests/gateway/test_session_route_leases.py`
  - `tests/hermes_cli/test_runtime_provider_resolution.py`
- [x] E5. Verified local regression results
  - `pytest -n 0 tests/test_model_catalog.py` -> 2 passed
  - `pytest -n 0 tests/gateway/test_session_route_leases.py` -> 8 passed
  - `pytest -n 0 tests/hermes_cli/test_runtime_provider_resolution.py` -> 62 passed
  - `pytest -n 0 tests/test_modal_deployment.py -k "runtime_api_config or candidate_routes_from_state or hydrate_route_from_lease or transient_error_triggers_retry_route_refresh or session_route_lease_sticky_hit or sync_runtime_config_materializes_dynamic_free_route"` -> passed
- [ ] E6. Pending external verification blocked by current network boundary
  - Direct active probes to `gateway.ai.cloudflare.com` from this machine still return `403 / error code 1010`
  - Remaining closure items must use an alternate validation channel or wait for an unblocked network path
- [x] E7. Narrowed manual model switching policy
  - Only Feishu session-scoped explicit model ID selection is allowed to create a manual lock
  - Non-Feishu `/model <id>` now returns guidance and keeps dynamic routing active
  - Chat-level `--global` persistence is disabled so default traffic remains dynamically routed
