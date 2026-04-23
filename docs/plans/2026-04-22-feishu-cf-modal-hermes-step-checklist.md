# 2026-04-22 Feishu-CF-Modal-Hermes 分步执行清单

更新时间：2026-04-22（UTC+8）

## 1. 本轮统一入口

- KPI 快照
  - `python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent`
- 任务清单
  - `python scripts/feishu_execution_task_checklist.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent --target-chat-id oc_ec86c28e66596c25377aff2ee028901c`

默认输出
- `.tmp-feishu-kpi-execution-snapshot.json`
- `.tmp-feishu-kpi-execution-snapshot.md`
- `.tmp-feishu-execution-task-checklist.json`
- `.tmp-feishu-execution-task-checklist.md`

## 2. 已核实的现场事实

- canonical app 已确认是 app3
  - `FEISHU_APP_ID3`
  - app id: `cli_a9525a47e4f99bc2`
  - app name: `hermes agent`
- Cloudflare gateway 已重新绑定到 app3 凭据
  - worker: `hermes-feishu-gateway`
  - 绑定来源: `FEISHU_APP_ID3 / FEISHU_APP_SECRET3`
- app3 最新发布版本已包含关键事件
  - `im.message.receive_v1`
  - `im.message.message_read_v1`
  - `application.bot.menu_v6`
- app3 已可直接向目标群发送真实应用消息
  - 已验证新发送消息进入目标群最近消息窗口
- app3 最近消息的 `read_users` 采样已验证可查询
  - 最近 5 条 app3 消息全部可被 `read_users` 查询
  - 但当前 5 条样本的已读人数仍全部为 `0`
- 目标群当前仍未满足真实闭环验证条件
  - target chat: `oc_ec86c28e66596c25377aff2ee028901c`
  - 当前 bot 数: `8`
  - `FEISHU_USER_ACCESS_TOKEN` 已过期
  - 最近群内 app 回复仍来自外部 app `cli_9ded8676aefb1103`

## 3. 当前 KPI 口径

- 已达标
  - `reply_minus_ai_p90_ms`
- 未达标
  - `read_receipt_p90_ms`
  - `session_cost`
  - `idle_hourly_cost`
  - `cache_eligible_hit_rate`
- 当前最重要的阻塞
  - `official_cost_without_matching_function_trace`
  - `modal_debug_function_missing`
  - 目标群多 bot 竞争
  - 用户态成员审计 token 过期

## 4. 分步任务

- [ ] FX001 验证 app3 的 `message_read` 回调并补采真实 `read_receipt` 样本
  - 目标
    - 不再把 FX001 当作“重新发布事件”
    - 改为确认 app3 的真实 `message_read` 回调是否进入 KPI 统计链路
  - 验证标准
    - app3 现场发布状态保持包含 `im.message.message_read_v1`
    - 新鲜真实样本能让 `read_receipt_p90_ms` 从 `null` 走向可计算
  - 命令
    - `python scripts/check_feishu_delivery_path.py --all-env-apps --target-chat-id oc_ec86c28e66596c25377aff2ee028901c --recent-message-window-minutes 180`
    - `python scripts/feishu_read_receipt_probe.py --chat-id oc_ec86c28e66596c25377aff2ee028901c --window-minutes 180 --page-size 20 --limit 10`

- [ ] FX002 收敛目标群到单一 app3 bot 身份
  - 目标
    - 清掉群内非 canonical bot 的竞争
    - 停止其他 app 在目标群继续回复
  - 验证标准
    - 目标群不再存在非 app3 的活跃 Hermes bot
    - 最近 app 回复只来自 app3
  - 命令
    - `python scripts/check_feishu_delivery_path.py --all-env-apps --target-chat-id oc_ec86c28e66596c25377aff2ee028901c --recent-message-window-minutes 180`

- [ ] FX003 刷新 `FEISHU_USER_ACCESS_TOKEN` 并审计完整成员列表
  - 依赖
    - `FX002`
  - 目标
    - 用用户视角确认群成员真实拓扑
    - 证明 app3 已成为唯一保留 bot
  - 验证标准
    - 用户态成员接口恢复可用
    - 成员列表与群拓扑一致
  - 当前结论
    - 当前环境中的 `FEISHU_USER_ACCESS_TOKEN` 已过期，错误码 `99991677`

- [ ] FX004 发送一条新的真实飞书消息并核实非 synthetic workflow
  - 依赖
    - `FX001`
    - `FX002`
    - `FX003`
  - 目标
    - 证明当前链路依赖的是新鲜真实流量，而不是历史样本
  - 验证标准
    - 生成新的 workflow 实例
    - `cf-ai-exec` 成功
    - `send-feishu-response` 成功
    - 飞书真实回复送达
  - 当前结论
    - app3 应用身份发消息已验证成功
    - 但“真实用户发消息”仍被过期 user token 卡住，暂时无法闭合真实 workflow 回放
  - 命令
    - `python scripts/feishu_cf_modal_live_diagnose.py --chat-filter oc_ec86c28e66596c25377aff2ee028901c --limit 12`
    - `python scripts/feishu_chain_status.py --artifacts-dir D:\suyee\github\hermesagent`

- [ ] FX005 补齐真实读回执样本并复测 `read_receipt_p90_ms`
  - 依赖
    - `FX001`
    - `FX004`
  - 目标
    - 让 `read_receipt_p90_ms < 5000`
  - 命令
    - `python scripts/feishu_read_receipt_probe.py --chat-id oc_ec86c28e66596c25377aff2ee028901c --window-minutes 180 --page-size 20 --limit 10`
    - `python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent`

- [ ] FX006 确定成本真相源
  - 目标
    - 明确成本 KPI 的可验口径
  - 当前策略
    - gatekeeping 继续使用 `strict`
    - `blended_total` 仅作数据充分性诊断
    - `official_average` 仅作估算参考，不作为 KPI 达标依据
  - 当前 blocker
    - `official_cost_without_matching_function_trace`
    - `modal_debug_function_missing`
  - 命令
    - `python scripts/feishu_perf_cost_report.py --since-hours 24`
    - `python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent`
    - `python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent --session-cost-source-policy blended_total`
    - `python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent --session-cost-source-policy official_average`

- [ ] FX007 采集重复无状态 DM 样本验证缓存命中率
  - 依赖
    - `FX004`
  - 目标
    - 验证 `cache_eligible_hit_rate > 30%`

- [ ] FX008 等待 downsizing 窗口成熟后复测 24h/72h 成本
  - 依赖
    - `FX006`
  - 目标
    - `idle_hourly_cost_p90_usd < 0.005`
    - `session_cost < 0.0045`
  - 命令
    - `python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent`
    - `python scripts/feishu_kpi_execution_snapshot.py --hours 72 --artifacts-dir D:\suyee\github\hermesagent`

- [ ] FX009 补一组真实卡片/菜单样本验证控制面
  - 目标
    - 补齐以下真实留痕
      - model switch
      - personality switch
      - skill combo
  - 命令
    - `python scripts/feishu_cf_modal_live_diagnose.py --chat-filter <direct_dm_chat_id> --limit 12`

## 5. 当前执行顺序

1. 先做 `FX001`、`FX002`、`FX003`
2. 再做 `FX004`、`FX005`
3. 然后推进 `FX006`
4. 最后复测 `FX007`、`FX008`、`FX009`
