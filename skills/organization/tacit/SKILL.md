---
name: tacit
description: Convert hard-won experience into reusable red lines, heuristics, anti-patterns, scenarios, and escalation rules.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, tacit-knowledge, heuristics, playbook]
    related_skills: [gov, retro, board-review]
---

# tacit

这个技能用于把“做过一次才真正懂”的经验，沉淀成组织可复用资产。

## 允许沉淀的 5 类资产

1. `red-lines`
   哪些情况不能继续讨论，必须立即动作。
2. `heuristics`
   哪类信号组合对应哪类优先怀疑路径。
3. `anti-patterns`
   哪些看起来合理但已经被证明会误导。
4. `scenarios`
   高频情境的处理模板。
5. `escalation`
   什么问题交给 `cto`，什么问题切 `sev`，什么问题必须升到 `board`。

## 固定输出格式

```text
asset_type:
title:
signal:
rule:
why_it_matters:
when_not_to_use:
```

## 当前项目的第一批种子资产

### red-lines

- 当 Feishu 回调或卡片动作接近超时阈值时，优先保证快速响应和后台分流，不把重逻辑塞在同步回调里。

### heuristics

- 如果 ACK 很快但最终回复很慢，优先看 worker queue latency、分区 claim 和后半段执行链路。
- 如果功能看起来“已连接”但不工作，优先核实是不是控制面仍落在普通聊天解释链路里。

### anti-patterns

- 把表格镜像当成实时路由真相源。
- 用长时间常驻容器替代精准的生命周期优化。
- 没有埋点就讨论体验是否变快。

### scenarios

- Feishu 卡片回调 `200341`
- 模型 401/403 与 fallback 路径
- Modal 冷启动与快照收益评估

### escalation

- 进入实现与代码层面：切 `cto`
- 出现线上事故或异常成本：切 `sev`
- 涉及路线、预算、是否继续投入：升 `board`

## 不该做什么

- 不要把聊天纪要原封不动塞进 tacit 资产。
- 不要沉淀未经验证的观点。
- 不要写成只有作者自己看得懂的黑话。
