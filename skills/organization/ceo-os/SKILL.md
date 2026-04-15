---
name: ceo-os
description: Startup CEO operating system for goals, priorities, dependencies, owners, and execution cadence.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, ceo, operating-system, execution]
    related_skills: [board-review, growth-os, ship, staff]
---

# ceo-os

这是日常操盘技能。用于把复杂问题收敛成目标、优先级、依赖和动作。

## 适用场景

- 项目推进
- 跨模块协调
- 需求排期与轻量经营管理
- Feishu / CLI 中的日常操盘会话

## 默认输出结构

1. `goal`
   当前真正要达成的结果。
2. `priority`
   现在最先做什么，为什么不是别的。
3. `dependencies`
   哪些前置条件会卡住结果。
4. `owners`
   从角色视角看由谁推动。
5. `cadence`
   今天、本周、下一轮分别做什么。
6. `risks`
   只写会打断节奏或导致返工的风险。

## 操盘规则

- 优先保闭环，不优先保完美。
- 先识别 bottleneck，再安排动作。
- 先把责任和节奏写清楚，再谈延伸想法。
- 需要切到工程执行时，切换 `cto + ship`。
- 需要切到增长优化时，切换 `grow + growth-os`。

## 不该做什么

- 不要把“列 TODO”误当成操盘。
- 不要把多个目标混成一个模糊大目标。
- 不要在没有关键路径判断的情况下并行摊子。

## 推荐协作

- `staff`：输出更适合协同同步、会议纪要和跨团队推进。
- `ship`：当任务进入实现、回归、上线时接管执行方法。
- `board-review`：当问题升级到方向和资源配置时接管。
