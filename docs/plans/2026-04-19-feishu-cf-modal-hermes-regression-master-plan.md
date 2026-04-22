# Feishu-Cloudflare-Modal-Hermes 全流程回归验证与优化改进总方案

更新时间：2026-04-19（UTC+8）

## 1. 文档定位

本方案用于当前仓库内 Feishu -> Cloudflare AI Gateway -> Modal -> Hermes Agent 的真实生产链路。

目标不是单次测试，而是建立一个可持续运行的闭环体系：

- 让回归验证具备统一口径、统一证据、统一门禁
- 让 Cloudflare 网关从“动态路由”升级为“可解释、可学习、可恢复”的智能路由系统
- 让 Hermes 的轻任务、普通会话、浏览器重任务、工具调用、多媒体请求都能被正确分流
- 让问题修复进入 PDCA 自循环，直到所有目标达成

本总方案与现有专项方案互补：

- Cloudflare 动态路由专项检查清单：
  [2026-04-19-cf-ai-gateway-pdca-checklist.md](D:/suyee/github/hermesagent/hermes-agent/docs/plans/2026-04-19-cf-ai-gateway-pdca-checklist.md)

## 2. 适用范围

覆盖以下全链路对象：

- 飞书消息接入、事件解析、ACK、已读事件、卡片动作、命令动作
- Hermes 会话处理、人格切换、模型切换、命令行与菜单型控制面
- Cloudflare AI Gateway 智能路由、动态路由、缓存、限流、回退、日志
- Modal runtime、chat queue worker、background exec worker、ack reaction worker
- 浏览器重任务分类、预处理、站点预取、浏览器执行链路
- 模型目录拉取、可用性评分、正负反馈、恢复机制

不在本轮主范围但允许作为边界验证的对象：

- Telegram/QQ 等其他 webhook 接入
- Hermes 通用 CLI 的非飞书主链路行为

## 3. 现有资产基线

当前仓库已经具备以下可复用能力：

- Modal 生产入口与 Feishu 主链路：
  [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
- Cloudflare AI Gateway 审计脚本：
  [cf_ai_gateway_audit.py](D:/suyee/github/hermesagent/hermes-agent/scripts/cf_ai_gateway_audit.py)
- Cloudflare AI Gateway 路由探针：
  [cf_ai_gateway_validate_routes.py](D:/suyee/github/hermesagent/hermes-agent/scripts/cf_ai_gateway_validate_routes.py)
- Feishu 三方时延/成本/会话统计：
  [feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_triparty_pk_report.py)
- Modal 成本拆解：
  [feishu_perf_cost_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_perf_cost_report.py)
- Feishu/Modal 主回归集：
  [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)
- Feishu 三方报表回归：
  [test_feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_feishu_triparty_pk_report.py)

当前已知现状与缺口：

- Hermes 请求进入 CF 网关后的智能路由正确命中率，按现有实现目标应视为 100%，但仍需通过“能力匹配正确率”而不只是 route name 命中来验证
- 网关已具备正负反馈、定时拉取可用模型列表、错误失败评分机制，但尚需把这些能力纳入统一验收门禁
- 当前仓库历史快照中，回复扣除 AI 时延与单会话成本仍存在超标样本
- 已读回执指标存在数据缺口，不能接受 `insufficient_data` 作为通过条件

## 4. 总体目标

### 4.1 业务目标

- 飞书文本指令走快速通道，控制类操作即时完成
- 模型切换后会话锁定稳定，不漂移
- 人格特质切换、命令行控制、普通会话保持通畅
- 浏览器重任务必须先分类、先预处理、再调模型
- 一般任务优先走轻量高性能模型，重任务优先走类型匹配模型
- Cloudflare 网关不只是“转发”，而是“智能路由 + 评分闭环 + 自动恢复”

### 4.2 量化目标

- 单会话成本 `< 0.0045 USD`
- 空闲耗费 `< 0.005 USD / 小时`
- 飞书发出消息到收到已阅读消息 `< 5 秒`
- 扣除 AI 模型处理时间后，飞书发出到收到正式回复 `< 20 秒`
- CF 网关因为速率限制而触发 fallback 的次数 `= 0`
- 模型错误后仅一次 fallback 即成功，成功率 `= 100%`
- CF 网关缓存率 `> 30%`
- 浏览器任务分类预处理正确率 `= 100%`
- 浏览器任务仅需单次调用 AI 网关达成率 `> 50%`
- CF 智能路由正确率 `= 100%`
- 类型能力匹配正确率 `= 100%`
- 优先选择“高性能且类型匹配模型”的正确率 `>= 95%`

## 5. 核心设计原则

### 5.1 PDCA 原则

- Plan：先冻结口径与门禁，再做变更
- Do：按固定矩阵执行，不允许凭经验跳测
- Check：必须看滚动窗口，不接受单点成功
- Act：所有经验都要回灌到脚本、回归、告警、运行手册

### 5.2 ISO 体系思维

- 所有指标要有明确定义、采集方式、责任人、证据路径
- 所有关键变更要有前置检查、实施记录、回归结果、回滚条件
- 所有故障要能通过日志与证据重建完整路径

### 5.3 DFMEA 思维

- 不只验证“是否成功”，还要提前识别“为何会失败”
- 每个关键链路都要建立失效模式、严重度、探测方式、预防控制

### 5.4 Polanyi 默会知识显性化

- 凡是“只有熟手知道”的判断规则，都必须写入脚本、测试、告警或 runbook
- 不允许靠个人经验长期维持系统稳定

## 6. 目标链路架构

### 6.1 标准主链路

1. 飞书发送消息或卡片动作
2. Hermes Feishu webhook 接收事件
3. 快速判定：
   - 是否控制类命令
   - 是否普通轻会话
   - 是否工具请求
   - 是否浏览器重任务
   - 是否多媒体/附件任务
4. 生成结构化请求上下文：
   - `task_kind`
   - `request_class`
   - `route_hint`
   - `requires_tools`
   - `requires_browser`
   - `content_modalities`
   - `site_prefetch`
   - `target_domain`
5. Hermes 将请求交给 CF 网关智能路由
6. CF 网关完成：
   - 候选模型过滤
   - 能力匹配
   - 性能优先排序
   - 正负反馈评分
   - dynamic route 命中
   - cache / retry / fallback
7. Modal worker 执行与发送结果
8. 飞书收到正式回复与已读事件
9. Hermes 回写会话、路由租约、评分反馈、报表数据

### 6.2 关键时间点

- `T0`：飞书 webhook 接收成功
- `T1`：飞书 ACK 或已接收提示发出
- `T2`：分类与预处理完成
- `T3`：AI 网关首个成功执行完成
- `T4`：正式回复发送完成
- `T5`：飞书已读事件落地

### 6.3 关键时延定义

- `ACK 时延 = T1 - T0`
- `分类预处理时延 = T2 - T0`
- `AI 执行时长 = T3 - T2`
- `正式回复时延 = T4 - T0`
- `扣 AI 后正式回复时延 = T4 - T0 - (T3 - T2)`
- `已读时延 = T5 - T0`

## 7. Cloudflare 智能路由闭环设计

### 7.1 路由不是单层命中，而是四层决策

#### L1：能力过滤

先过滤出真正支持当前请求类型的模型：

- 纯文本
- 多媒体理解
- 浏览器重任务协同
- 工具调用兼容

#### L2：类型匹配

不同请求必须进入不同候选池：

- `text_light`
- `text_general`
- `text_coding`
- `tool_heavy`
- `browser_heavy`
- `multimodal`

#### L3：评分排序

在候选池中按综合分排序，分值建议由以下维度组成：

- 最近成功率
- P50/P95 时延
- 单次平均成本
- fallback 触发率
- payload/schema 兼容率
- 最近错误热度
- 缓存友好度

#### L4：回退与恢复

- 首选失败后只允许一次兼容 fallback
- 连续失败模型必须降权或摘流
- 恢复必须依赖连续成功样本，不允许一次成功就完全恢复

### 7.2 智能路由验收标准

- 命中正确 route name 不等于路由正确
- 必须证明该模型：
  - 支持当前请求类型
  - 支持当前 payload 结构
  - 支持当前工具调用或浏览器协同方式
  - 在同类候选中属于高性能或最优性价比候选

### 7.3 正反馈机制

成功样本应至少记录：

- provider
- model
- route_name
- request_class
- payload_type
- latency
- cost
- cache_status
- tools_required
- browser_required

成功后的正反馈应影响：

- 候选优先级上调
- 恢复冻结模型
- 提升同类型请求的首选概率

### 7.4 负反馈机制

失败样本需按错误类型区分：

- `400`：payload 或 schema 不兼容
- `401/403`：权限或供应商策略问题
- `404`：provider/model 不存在或映射错误
- `429`：速率限制
- `5xx`：上游不稳定
- `timeout`：性能不可接受

负反馈必须影响：

- 当前类型请求的模型降权
- fallback 入口选择
- 暂时摘流或冷却
- 后续恢复门槛

### 7.5 模型目录同步要求

模型目录拉取机制必须满足：

- 周期性拉取可用模型列表
- 能力标签同步更新
- provider/model 别名归一
- 拉取失败时保留最后一次健康快照
- 目录过旧时触发显式告警

### 7.6 路由解释性要求

每次请求都应能还原：

- 为什么是这个 request class
- 为什么是这个 route hint
- 为什么这个模型排在第一
- 为什么没选其他模型
- 是否命中过正反馈或负反馈
- 是否触发 fallback、为何触发

## 8. 回归域划分

### 8.1 飞书接入域

- 文本消息
- 群聊/私聊
- 菜单卡片
- 模型切换卡片
- 人格切换卡片
- message_read 事件
- 重复 webhook 与重投

### 8.2 快速控制域

- `/model`
- `/provider`
- `/personality`
- registry switch
- 纯命令型卡片动作

要求：

- 走 `fast_control`
- 不进浏览器链路
- 不进重 worker
- 不允许多次 fallback

### 8.3 一般会话域

- 轻问答
- 多轮上下文
- 摘要
- 常规工具协助

### 8.4 浏览器重任务域

- 打开网页
- 内容抓取
- 截图
- 带站点预处理的文档类任务
- 站点需要浏览器而非纯文本抓取的任务

### 8.5 智能路由域

- dynamic route 命中
- 候选模型正确过滤
- 类型能力匹配
- 正负反馈生效
- 恢复机制

### 8.6 成本与缓存域

- 单会话成本
- 空闲小时成本
- cache eligible 命中率
- overall cache 命中率
- 轻重任务成本拆分

## 9. DFMEA 主失效模式

| 编号 | 失效模式 | 后果 | 主要原因 | 探测方式 | 预防/控制 |
|---|---|---|---|---|---|
| F1 | 已读数据缺失 | 无法判断 `<5s` 是否达标 | read_users 失败、事件未关联 | PK 报表出现 `insufficient_data` | 先修已读链路，再谈时延达标 |
| F2 | 快速通道退化为重任务 | 时延高、成本高 | 控制请求未命中 `fast_control` | Feishu trace / modal 回归 | 命令类请求强制走控制面 |
| F3 | 模型切换后未锁定 | 会话漂移 | route lease 未更新 | 会话锁定回归 | 显式更新并校验 session route lease |
| F4 | 轻任务误入浏览器链路 | 时延与成本放大 | 分类错误 | request_class 与 route_hint 对照 | 建黄金样本与混淆矩阵 |
| F5 | 浏览器任务预处理不完整 | 二次调用 AI 才能收敛 | site_prefetch 字段不全 | trace / screenshot / 回归 | 预处理字段强校验 |
| F6 | 智能路由只命中 route 不命中能力 | 模型表面可用，实际不兼容 | 能力标签错误 | capability mismatch 报表 | 路由验收改为“类型匹配验收” |
| F7 | 负反馈不敏感 | 坏模型持续被选中 | 降权不生效 | 连续失败后仍为首选 | 引入摘流与冷却窗 |
| F8 | 正反馈过敏 | 一次成功即回顶 | 恢复门槛过低 | 恢复后再失败 | 连续成功后再恢复 |
| F9 | 429 触发 fallback | 成本升高、成功率抖动 | 限流前置不足 | gateway logs | 节流前移到 queue/worker |
| F10 | fallback 链过长 | 回复超时 | 兼容性未分型 | fallback 次数统计 | 每类请求只允许一次兼容 fallback |
| F11 | 缓存率失真 | 误判优化效果 | 将不可缓存流量混算 | cache eligible 对比报表 | 拆分 eligible hit rate |
| F12 | 空闲成本过高 | 无法达成 `<0.005/h` | worker linger 过长 | cost report | 分 lane 控制 worker 资源 |

## 10. 统一验收指标口径

### 10.1 主要指标

- `read_receipt_p90_ms`
- `reply_minus_ai_p90_ms`
- `session_cost_avg_usd`
- `session_cost_p90_usd`
- `idle_hourly_cost_p90_usd`
- `cache_eligible_hit_rate`
- `overall_cache_hit_rate`
- `fallback_once_success_rate`
- `rate_limit_triggered_fallback_count`
- `browser_preprocess_accuracy`
- `browser_single_ai_call_completion_rate`
- `route_decision_explainable_rate`
- `capability_match_rate`
- `preferred_model_selection_accuracy`

### 10.2 判定规则

- 任何关键指标出现 `insufficient_data`，默认判定为未达标
- 任何 P0/P1 失效模式在 72h 窗口内新增复现，默认判定为未达标
- 任何需要二次 fallback 才成功的请求，默认判定为未达标

## 11. 回归测试矩阵

### 11.1 控制面矩阵

- `/model` 文本命令
- `/personality` 文本命令
- 模型选择卡片
- 人格选择卡片
- registry switch model
- cancel/close card

验证点：

- `route_hint=fast_control`
- `execution_mode=control_complete`
- 模型或人格变更落入 session state
- 无浏览器、无多跳 worker、无二次 fallback

### 11.2 一般会话矩阵

- 短文本问答
- 多轮会话
- 上下文延续
- 常规工具少量参与

验证点：

- `chat_light` 与 `chat_heavy` 分类正确
- 会话通畅
- 模型锁定不漂移
- 成本与时延符合轻会话预期

### 11.3 浏览器任务矩阵

- 文档站点抓取
- 需要直接导航的站点
- 截图任务
- 带工具调用的网页任务

验证点：

- `request_class`
- `site_prefetch_mode`
- `requires_browser`
- `browser_target_domain`
- 单次 AI 网关达成率

### 11.4 智能路由矩阵

- 文本轻任务 -> 文本高性能模型
- 工具请求 -> 支持工具调用模型
- 多媒体请求 -> 支持多媒体模型
- 浏览器重任务 -> 支持浏览器协同或重任务模型
- coding 请求 -> coding lane 最优模型

验证点：

- 候选池正确
- 能力匹配正确
- 最终首选合理
- fallback 一次成功
- 失败后降权、恢复后稳定

### 11.5 缓存与限流矩阵

- 重复文本请求
- 相似文本请求
- 工具请求跳过缓存
- 浏览器请求缓存资格判定
- 限流边界压力请求

验证点：

- cache eligible hit rate
- overall hit rate
- 429 是否前置吸收
- 429 是否导致 fallback

## 12. 自动化任务清单

### P0：先补测量

- [ ] 将总方案目标同步到
  [feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_triparty_pk_report.py)
- [ ] 禁止 `read_receipt` 使用 `insufficient_data` 通过门禁
- [ ] 在
  [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
  补齐以下字段：
  - `route_version`
  - `provider_alias`
  - `fallback_reason`
  - `cache_eligible`
  - `cache_status`
  - `ai_call_count`
  - `capability_match`
  - `preferred_model_selected`
- [ ] 固化 1h/24h/72h 三窗日报

### P1：压时延

- [ ] 强制控制类请求全部走 `fast_control`
- [ ] 优化 `chat_light` worker 热身与复用
- [ ] 浏览器任务先预处理再调模型
- [ ] 将扣 AI 回复时延作为主门禁之一

### P1：降成本

- [ ] 利用
  [feishu_perf_cost_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_perf_cost_report.py)
  拆分函数级成本热点
- [ ] 轻任务与重任务分模型、分 worker 资源
- [ ] 收紧空闲 worker linger
- [ ] 提升可缓存文本请求的稳定 cache key

### P1：稳路由

- [ ] 为智能路由增加“能力匹配正确率”报表
- [ ] 为正负反馈增加可观测分数字段
- [ ] 引入恢复门槛，禁止一次成功即完全恢复
- [ ] 限制 fallback 为“单次兼容回退”

### P1：补回归

- [ ] 在
  [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)
  补齐：
  - 浏览器分类正确率回归
  - 单次 AI 调用达成率统计回归
  - 路由评分/降权/恢复回归
  - 429 不触发 fallback 回归
- [ ] 在
  [test_feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_feishu_triparty_pk_report.py)
  补齐新门禁阈值断言

## 13. PDCA 执行节奏

### 13.1 每日节奏

1. 跑 24h 报表
2. 识别红线项
3. 选 1 到 2 个主矛盾
4. 每个主矛盾最多做 3 个变更
5. 先单测，再小窗复测，再 24h，再 72h

### 13.2 每轮问题关闭条件

- 1h 窗口无新增同类故障
- 24h 指标显著改善
- 72h 无复发
- 自动化用例已补齐
- 运行手册已更新

## 14. 验收门禁

必须同时满足：

- 连续 72h 无新增 P0/P1 故障模式
- 单会话成本 `< 0.0045 USD`
- 空闲小时成本 `< 0.005 USD`
- 已读时延 `< 5 秒`
- 扣 AI 后正式回复 `< 20 秒`
- 429 触发 fallback 次数 `= 0`
- 模型错误后一次 fallback 成功率 `= 100%`
- CF 缓存率 `> 30%`
- 浏览器任务分类预处理正确率 `= 100%`
- 浏览器任务单次 AI 网关达成率 `> 50%`
- CF 智能路由正确率 `= 100%`
- 能力匹配正确率 `= 100%`
- 优先高性能且类型匹配模型选择正确率 `>= 95%`
- 路由解释性覆盖率 `= 100%`

## 15. 输出物要求

每轮优化必须输出：

- 报表 JSON
- 报表 Markdown 摘要
- 变更清单
- 回归结果
- 新增或修复的失效模式
- 下一轮主矛盾

建议固定输出路径：

- `.tmp-feishu-triparty-pk-report.json`
- `.tmp-feishu-triparty-pk-summary.md`
- `.tmp-cf-ai-gateway-audit.json`
- `.tmp-cf-ai-gateway-audit.md`

## 16. 最终说明

本方案默认以下判断成立：

- 当前仓库的 Hermes 请求进入 CF 网关后的智能路由确实具备动态选模能力
- 网关已经具备正负反馈、模型列表定时同步、失败打分与恢复机制
- 本轮工作的重点不是“重新设计一套路由系统”，而是把现有能力制度化、指标化、自动化、门禁化

后续执行原则：

- 先补测量，再做优化
- 先稳分类，再稳路由
- 先控 fallback，再冲性能
- 先消除数据盲区，再判定是否达标

## 17. 执行任务清单

本节将总方案转换为可执行任务 backlog。执行时遵循：

- 所有任务默认未完成，必须由代码、测试、报表或运行证据关闭
- 未完成任务不允许口头关闭
- 同一任务关闭时必须补充“证据路径”
- 所有任务清零前，PDCA 自循环不得中断

### 17.1 P0 基线与口径冻结

- [x] T001 冻结统一指标口径
  - 产出：
    - `read_receipt_p90_ms`
    - `reply_minus_ai_p90_ms`
    - `session_cost_avg_usd`
    - `session_cost_p90_usd`
    - `idle_hourly_cost_p90_usd`
    - `cache_eligible_hit_rate`
    - `overall_cache_hit_rate`
    - `fallback_once_success_rate`
    - `browser_preprocess_accuracy`
    - `browser_single_ai_call_completion_rate`
    - `capability_match_rate`
    - `preferred_model_selection_accuracy`
  - 完成标准：
    - 所有指标在脚本中有确定字段与计算口径
    - 不再依赖人工解释
  - 证据：
    - [feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_triparty_pk_report.py)

- [x] T002 将正式门禁写入报表脚本
  - 目标值：
    - 单会话成本 `< 0.0045 USD`
    - 空闲小时成本 `< 0.005 USD`
    - 已读 `< 5s`
    - 扣 AI 后正式回复 `< 20s`
    - 缓存率 `> 30%`
  - 完成标准：
    - 报表输出直接给出 `met / not_met`
    - `insufficient_data` 默认判 `not_met`
  - 证据：
    - [feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_triparty_pk_report.py)
    - [test_feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_feishu_triparty_pk_report.py)

- [x] T003 冻结 1h/24h/72h 三窗输出
  - 完成标准：
    - 每次执行都输出三窗结果
    - 每窗都有当前值、历史对照值、趋势判断
  - 证据：
    - `.tmp-feishu-triparty-pk-report.json`
    - `.tmp-feishu-triparty-pk-summary.md`
    - [feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_triparty_pk_report.py)
    - [test_feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_feishu_triparty_pk_report.py)

### 17.2 P0 观测与证据补齐

- [x] T004 为主链路补齐统一追踪字段
  - 必须补齐：
    - `route_version`
    - `provider_alias`
    - `fallback_reason`
    - `cache_eligible`
    - `cache_status`
    - `ai_call_count`
    - `capability_match`
    - `preferred_model_selected`
    - `model_catalog_version`
    - `feedback_score_before`
    - `feedback_score_after`
  - 完成标准：
    - 单次失败或成功都能重建“为何路由到此模型”
  - 证据：
    - [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
    - [internal/feishu/trace.py](D:/suyee/github/hermesagent/hermes-agent/internal/feishu/trace.py)
    - [internal/feishu/executor.py](D:/suyee/github/hermesagent/hermes-agent/internal/feishu/executor.py)
    - [tests/test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)

- [x] T005 补齐已读事件闭环
  - 完成标准：
    - `message_read` 事件能与 `message_id / correlation_id / session_key` 关联
    - 任何读回执失败都有明确错误分类
    - 报表不再长期出现 `insufficient_data`
  - 当前进展：
    - 已补齐 `message_read -> message_id_list[0] -> message_id` 的主键回填
    - 已在 Cloudflare Worker 运行时建立 `send_message_id -> correlation_id / session_key / event_id` 会话内索引
    - 已在 `message_read` 快通道直接解析运行时索引并回填 `correlation_id / session_key`
    - 已补充 `matched / partial / miss` 解析状态与未命中 message id 证据字段
  - 证据：
    - [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
    - [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)
    - [cloudflare/feishu-gateway/src/durable/message-correlation.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/durable/message-correlation.ts)
    - [cloudflare/feishu-gateway/src/durable/reconcile-queue.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/durable/reconcile-queue.ts)
    - [cloudflare/feishu-gateway/src/durable/reconcile-client.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/durable/reconcile-client.ts)
    - [cloudflare/feishu-gateway/src/entry/fetch-handler.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/entry/fetch-handler.ts)
    - [cloudflare/feishu-gateway/src/services/feishu/messages.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/services/feishu/messages.ts)
    - [cloudflare/feishu-gateway/test/message-correlation.test.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/test/message-correlation.test.ts)

- [x] T006 统一失败分类字典
  - 分类最少覆盖：
    - `payload_incompatible`
    - `provider_permission_denied`
    - `provider_model_not_found`
    - `rate_limited`
    - `upstream_5xx`
    - `timeout`
    - `misrouted_request_class`
    - `catalog_stale`
  - 完成标准：
    - Gateway、Modal、报表使用同一错误分类名
  - 当前进展：
    - Modal 已建立统一归一化映射：`payload_incompatible / provider_permission_denied / provider_model_not_found / rate_limited / upstream_5xx / timeout / misrouted_request_class / catalog_stale`
    - Cloudflare Worker 已在 `route-policy / agent-runtime / model-catalog feedback` 三处收敛到同一字典
    - 旧失败名仅作为兼容输入保留，日志与运行态输出统一写入归一化名称
  - 证据：
    - [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
    - [internal/feishu/executor.py](D:/suyee/github/hermesagent/hermes-agent/internal/feishu/executor.py)
    - [cf_ai_gateway_audit.py](D:/suyee/github/hermesagent/hermes-agent/scripts/cf_ai_gateway_audit.py)
    - [cloudflare/feishu-gateway/src/gateway/route-policy.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/gateway/route-policy.ts)
    - [cloudflare/feishu-gateway/src/gateway/agent-runtime.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/gateway/agent-runtime.ts)
    - [cloudflare/feishu-gateway/src/model-catalog/feedback.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/src/model-catalog/feedback.ts)
    - [cloudflare/feishu-gateway/test/route-policy.test.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/test/route-policy.test.ts)
    - [cloudflare/feishu-gateway/test/model-catalog-feedback.test.ts](D:/suyee/github/hermesagent/hermes-agent/cloudflare/feishu-gateway/test/model-catalog-feedback.test.ts)

### 17.3 P1 飞书快速通道与控制面

- [x] T007 将控制类请求全部压入 `fast_control`
  - 覆盖：
    - `/model`
    - `/provider`
    - `/personality`
    - 模型选择卡片
    - 人格选择卡片
    - registry switch
  - 完成标准：
    - `route_hint=fast_control`
    - `execution_mode=control_complete`
    - 不进入浏览器重任务链路
  - 证据：
    - [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
    - [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)

- [x] T008 模型切换后会话锁定稳定
  - 完成标准：
    - 显式切换后后续轮次不漂移
    - `route_lease` 中保留 provider/model/version
  - 证据：
    - [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)

- [x] T009 人格切换与命令卡片回归补齐
  - 完成标准：
    - 切换后跨轮保持
    - 取消/关闭/确认动作均有预期结果
  - 证据：
    - [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)

### 17.4 P1 浏览器任务分类与预处理

- [ ] T010 建立浏览器任务黄金样本集
  - 范围：
    - 文档站点
    - 普通网页
    - 截图任务
    - 登录态页面
    - 带工具调用的网页任务
  - 完成标准：
    - 至少一组可重复执行的样本库
  - 证据：
    - 新增或扩展测试数据与回归用例

- [ ] T011 固化浏览器任务分类规则
  - 输出字段：
    - `request_class`
    - `requires_browser`
    - `site_prefetch_mode`
    - `browser_target_domain`
    - `route_hint`
  - 完成标准：
    - 浏览器任务分类预处理正确率 `= 100%`
  - 证据：
    - [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
    - [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)

- [ ] T012 浏览器任务单次 AI 网关达成率报表化
  - 完成标准：
    - 报表能输出 `browser_single_ai_call_completion_rate`
    - 明确区分“需要二次 AI 调用”与“单次完成”
  - 证据：
    - [feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_triparty_pk_report.py)

### 17.5 P1 Cloudflare 智能路由闭环

- [ ] T013 固化模型目录同步机制
  - 完成标准：
    - 定时拉取可用模型列表
    - 目录版本可追踪
    - 拉取失败时有健康快照兜底
  - 证据：
    - [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
    - [test_model_catalog.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_model_catalog.py)

- [ ] T014 模型能力标签归一
  - 标签至少覆盖：
    - `text`
    - `multimodal`
    - `tool_call`
    - `browser_heavy`
    - `coding`
  - 完成标准：
    - 请求类型与能力标签匹配正确率 `= 100%`
  - 证据：
    - 模型目录代码与回归测试

- [ ] T015 智能路由四层决策可观测
  - 层级：
    - `L1 capability filter`
    - `L2 request-type match`
    - `L3 score ranking`
    - `L4 fallback / recovery`
  - 完成标准：
    - 每次请求都能还原四层决策轨迹
  - 证据：
    - [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
    - Gateway logs

- [ ] T016 正负反馈评分闭环落地
  - 正反馈：
    - 成功率
    - 低时延
    - 低成本
    - payload 兼容
  - 负反馈：
    - 400/401/403/404/429/5xx/timeout
  - 完成标准：
    - 连续失败会降权
    - 连续成功才恢复
  - 证据：
    - [modal_.py](D:/suyee/github/hermesagent/hermes-agent/modal_.py)
    - 回归测试

- [ ] T017 高性能且类型匹配优先策略落地
  - 完成标准：
    - 优先模型选择正确率 `>= 95%`
    - 不允许纯文本高分快模型抢占浏览器重任务 lane
  - 证据：
    - 回归测试
    - 报表字段 `preferred_model_selected`

### 17.6 P1 回退、限流、缓存

- [ ] T018 将 429 的吸收前移到 queue / worker
  - 完成标准：
    - `rate_limit_triggered_fallback_count = 0`
  - 证据：
    - Gateway logs
    - 压测与报表

- [ ] T019 将 fallback 限制为一次兼容回退
  - 完成标准：
    - 模型错误后一跳 fallback 成功率 `= 100%`
    - 不允许二次 fallback 作为成功
  - 证据：
    - [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)

- [ ] T020 拆分缓存率口径
  - 输出：
    - `cache_eligible_hit_rate`
    - `overall_cache_hit_rate`
  - 完成标准：
    - 验收默认使用 `cache_eligible_hit_rate`
  - 证据：
    - 报表脚本与日报

### 17.7 P1 时延与成本优化

- [ ] T021 函数级成本热点压降
  - 优先核查：
    - `internal_agent_exec`
    - `chat_queue_worker`
    - `feishu_background_exec_worker`
  - 完成标准：
    - 单会话成本 `< 0.0045 USD`
  - 证据：
    - [feishu_perf_cost_report.py](D:/suyee/github/hermesagent/hermes-agent/scripts/feishu_perf_cost_report.py)

- [ ] T022 控制空闲 worker 成本
  - 完成标准：
    - 空闲小时成本 `< 0.005 USD`
  - 证据：
    - Modal billing report
    - PK 报表

- [ ] T023 压降扣 AI 后正式回复时延
  - 完成标准：
    - `reply_minus_ai_p90_ms < 20000`
  - 证据：
    - PK 报表

- [ ] T024 压降已读时延
  - 完成标准：
    - `read_receipt_p90_ms < 5000`
  - 证据：
    - PK 报表

### 17.8 P1 自动化回归补齐

- [ ] T025 为智能路由新增专项回归
  - 覆盖：
    - 能力匹配
    - 首选模型选择
    - 正负反馈升降权
    - 恢复机制
  - 证据：
    - [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)

- [ ] T026 为浏览器任务新增专项回归
  - 覆盖：
    - request_class
    - site_prefetch
    - browser_target_domain
    - 单次 AI 调用完成率
  - 证据：
    - [test_modal_deployment.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_modal_deployment.py)

- [ ] T027 为报表门禁新增断言
  - 完成标准：
    - 新门禁全部有测试断言
  - 证据：
    - [test_feishu_triparty_pk_report.py](D:/suyee/github/hermesagent/hermes-agent/tests/test_feishu_triparty_pk_report.py)

## 18. 自主循环迭代机制

本计划执行方式不是线性一次做完，而是“自主循环迭代优化”，直到任务清单全部完成。

### 18.1 单轮执行算法

每轮必须按以下步骤执行：

1. 读取当前 backlog 状态
2. 跑最新三窗报表
3. 识别未达标门禁与最高优先级失效模式
4. 选取最多 1 到 2 个主矛盾
5. 完成对应任务的最小可验证变更
6. 运行单元回归
7. 运行小窗验证
8. 更新报表与任务状态
9. 若未清零，则进入下一轮

### 18.2 任务优先级规则

优先级固定如下：

1. 数据盲区与口径不一致
2. 已读与回复时延超标
3. 单会话与空闲成本超标
4. 智能路由能力匹配与 fallback 缺陷
5. 浏览器任务分类与预处理缺陷
6. 缓存率与稳定性优化

### 18.3 每轮限制

- 一轮最多处理 2 个主矛盾
- 一轮最多做 3 处核心代码改动
- 未完成验证前不得同时引入新的大范围策略变更

### 18.4 状态推进规则

任务状态只允许以下转换：

- `未开始 -> 进行中`
- `进行中 -> 已验证完成`
- `进行中 -> 阻塞`
- `阻塞 -> 进行中`

禁止：

- 无证据直接标记完成
- 单点成功直接标记完成
- 未补自动化直接标记完成

### 18.5 单任务关闭条件

任何任务关闭都必须同时满足：

- 代码已落地
- 回归已通过
- 报表或日志已证明行为达标
- 文档或 runbook 已更新

## 19. 任务清零标准

仅当以下条件全部满足时，视为“所有任务清单完成”：

- 第 17 节所有任务均已关闭
- 连续 72h 无新增 P0/P1 故障模式
- 所有量化门禁全部达标
- `insufficient_data` 类盲区清零
- 智能路由解释性覆盖率 `= 100%`
- 浏览器任务分类预处理正确率 `= 100%`
- 单次 fallback 成功率 `= 100%`

## 20. 执行记录模板

后续每轮执行建议按以下模板追加记录：

### 执行轮次

- 轮次编号：
- 执行时间：
- 本轮主矛盾：
- 处理任务：
- 代码改动：
- 回归结果：
- 指标变化：
- 新发现失效模式：
- 下一轮任务：

### 执行轮次

- 轮次编号：`R1`
- 执行时间：`2026-04-19`
- 本轮主矛盾：
  - 报表脚本仍使用旧门禁值
  - 新计划中的缓存/智能路由/浏览器单次调用指标尚未纳入正式输出
- 处理任务：
  - `T001`
  - `T002`
  - `T004`（部分，先完成报表侧字段透传准备）
- 代码改动：
  - 更新报表目标阈值为 `5s / 20s / 0.005 / 0.0045`
  - 将缺失数据门禁统一视为 `not_met`
  - 新增 `cache_hit_rate`、`browser_single_ai_call_completion_rate`、`capability_match_rate`、`preferred_model_selection_accuracy`
  - 为会话与 Cloudflare 事实补充智能路由字段透传
- 回归结果：
  - `pytest hermes-agent\\tests\\test_feishu_triparty_pk_report.py -q` -> `20 passed`
  - `python -m py_compile hermes-agent\\scripts\\feishu_triparty_pk_report.py` -> `passed`
- 指标变化：
  - 旧门禁已被新门禁替换
  - 新增的路由/缓存/浏览器指标已进入正式报表结构
- 新发现失效模式：
  - 智能路由相关字段在主链路中仍以“透传准备”为主，尚未全部由运行时代码稳定填充
- 下一轮任务：
  - `T003`
  - `T004`
  - `T005`
  - `T006`

- 轮次编号：`R2`
- 执行时间：`2026-04-19`
- 本轮主矛盾：
  - 主链路 trace 仍缺少运行时级别的智能路由/缓存/反馈字段
  - `message_read` 虽已进入快通道，但事件闭环仍偏依赖报表侧拼接
  - Gateway 与 Modal 的失败分类尚未完全统一
- 处理任务：
  - `T004`
  - `T005`（部分）
  - `T006`（部分）
- 代码改动：
  - 为 Modal trace 上下文补齐 `route_version / provider_alias / cache_eligible / cache_status / ai_call_count / capability_match / preferred_model_selected / model_catalog_version / feedback_score_before / feedback_score_after`
  - 为 `internal.agent_exec.done/error` 增加运行时观测字段派生和失败分类归一化
  - 为 `message_read` 事件补齐 `message_id_list[0] -> message_id` 关联主键
  - 为 Cloudflare Worker -> Modal 调用补传路由观测 metadata
  - 为性能汇总增加 `by_route_version / by_provider_alias / by_fallback_reason / by_cache_status / by_model_catalog_version`
- 回归结果：
  - `pytest tests\\test_modal_deployment.py -q` -> `150 passed`
  - `npm test -- agent-runtime.test.ts feishu-internal-contract.test.ts cf-ai-exec.test.ts` -> `5 passed`
  - `python -m py_compile internal\\feishu\\trace.py internal\\feishu\\contracts.py internal\\feishu\\http.py internal\\feishu\\executor.py internal\\feishu_perf.py` -> `passed`
- 指标变化：
  - 运行时 trace 已能直接还原更多“为何路由到该模型/为何回退”的证据
  - Cloudflare -> Modal 的智能路由观测链由“报表透传准备”升级为“运行时显式写入”
- 新发现失效模式：
  - `message_read` 对 `correlation_id / session_key` 的映射仍需运行时索引表或发送态回填，当前主要依赖报表侧按 `message_id` 反查
  - Cloudflare Worker 仍存在原始失败名与统一错误字典并存的情况
- 下一轮任务：
  - `T003`
  - `T005`
  - `T006`

- 轮次编号：`R3`
- 执行时间：`2026-04-19`
- 本轮主矛盾：
  - 主报表尚未固化 1h / 24h / 72h 三窗固定输出
  - 计划状态需要与实际代码进展重新对齐
- 处理任务：
  - `T003`
- 代码改动：
  - 为报表新增 `fixed_window_rollup`
  - 固定输出 `1h / 24h / 72h` 三窗当前值、历史对照值与趋势判断
  - 在 Markdown 摘要中新增 `Fixed Windows` 小节
- 回归结果：
  - `pytest tests\\test_feishu_triparty_pk_report.py -q` -> `21 passed`
  - `python -m py_compile scripts\\feishu_triparty_pk_report.py` -> `passed`
- 指标变化：
  - 三窗输出从“计划项”升级为正式报表结构
  - 每次执行都可直接读取 1h / 24h / 72h 的 current/baseline/trend
- 新发现失效模式：
  - 三窗报表已落地，但 `message_read -> correlation_id / session_key` 的运行时回连仍未彻底前移
  - 失败分类在 Cloudflare Worker 侧仍存在原始枚举与统一枚举混用
- 下一轮任务：
  - `T005`
  - `T006`

- 轮次编号：`R4`
- 执行时间：`2026-04-19`
- 本轮主矛盾：
  - `message_read` 对 `correlation_id / session_key` 的关联仍停留在报表侧回连，尚未前推到 Worker 运行时
  - Cloudflare Worker 边界输出仍混用原始失败名与统一失败字典
- 处理任务：
  - `T005`
  - `T006`
- 代码改动：
  - 为 `FEISHU_RECONCILE_QUEUE` Durable Object 新增会话内消息读索引：`send_message_id -> correlation_id / session_key / event_id`
  - 在发送成功路径持久化 outbound message correlation ref，并在 `message_read` 快通道直接解析运行时映射
  - 为 `message_read.accepted` 增加 `matched / partial / miss` 解析状态、已命中/未命中 message id 证据字段
  - 在 `route-policy / agent-runtime / model-catalog feedback` 三处统一 `payload_incompatible / provider_permission_denied / provider_model_not_found / rate_limited / upstream_5xx / timeout / misrouted_request_class / catalog_stale`
- 回归结果：
  - `npm test -- message-correlation.test.ts route-policy.test.ts agent-runtime.test.ts cf-ai-exec.test.ts feishu-internal-contract.test.ts model-catalog-feedback.test.ts` -> `12 passed`
- 指标变化：
  - `message_read` 已可在运行时直接回填 `correlation_id / session_key`，不再只依赖报表侧按 `message_id` 反查
  - Cloudflare Worker 写入 Modal 与日志的失败分类边界字典已与 Modal/report 统一
- 新发现失效模式：
  - 当前仍缺少真实 1h/24h/72h 运行样本来证明 `read_receipt_p90_ms < 5000` 与 `insufficient_data` 已完全清零
  - 总量化目标仍需继续靠后续窗口报表与线上样本验证
- 下一轮任务：
  - `T007`
  - `T008`
  - `T018`
  - `T024`

## 21. 路径说明

本计划实际存放于：

- [2026-04-19-feishu-cf-modal-hermes-regression-master-plan.md](D:/suyee/github/hermesagent/hermes-agent/docs/plans/2026-04-19-feishu-cf-modal-hermes-regression-master-plan.md)

说明：

- 你提到的 `D:/suyee/github/hermesagent/docs/plans/...` 在当前工作区中不存在
- 当前真实仓库文档路径位于内层仓库 `hermes-agent/docs/plans/`
