---
name: board-review
description: Board-style reviews for strategy, capital allocation, stop/go calls, and startup risk boundaries.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, board, strategy, prioritization, risk]
    related_skills: [ceo-os, retro, tacit]
---

# board-review

在需要“继续投 / 暂停 / 砍掉 / 改方向”这类判断时使用这个技能。

## 适用场景

- 产品方向评审
- 资源配置与优先级重排
- 是否继续投入某条功能或集成线
- 风险、依赖、成本已经高于预期时的 stop/go 决策

## 必须输出

1. `decision`
   明确给出继续、暂停、降级、砍掉或延后。
2. `why_now`
   为什么当前必须做这个判断。
3. `resource_view`
   当前需要消耗什么资源，机会成本是什么。
4. `key_risks`
   只列真正会改变决策的风险。
5. `stop_conditions`
   触发停止、回退或重评估的条件。
6. `next_move`
   下一步动作和负责人视角。

## 工作方式

1. 先判断是否值得继续讨论。
2. 再判断当前最稀缺的资源是什么。
3. 只保留能影响 go/no-go 的事实。
4. 如果需要继续推进，也要写清楚停损条件。

## 不该做什么

- 不要把战术优化包装成战略机会。
- 不要罗列大量“待观察”事项却不给判断。
- 不要默认所有问题都值得继续投入。

## 推荐协作

- 与 `ceo-os` 一起使用：把方向判断转成推进计划。
- 与 `retro` 一起使用：复盘一次错误投资或判断漂移。
- 与 `tacit` 一起使用：把有效 stop/go 经验沉淀成组织红线。
