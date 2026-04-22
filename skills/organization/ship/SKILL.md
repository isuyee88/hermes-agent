---
name: ship
description: Delivery workflow from specification to instrumentation, implementation, regression, and launch verification.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, delivery, implementation, regression, launch]
    related_skills: [ceo-os, gov, retro]
---

# ship

在任务进入“真正要做、要改、要上线”的阶段时使用。

## 适用场景

- 编码实现
- 调试修复
- 回归验证
- 上线前验收

## 强制顺序

1. `acceptance`
   先写清楚什么算完成。
2. `instrumentation`
   先决定埋点和日志，避免做完后无法验证。
3. `implementation`
   再进入改动。
4. `verification`
   至少做一次针对性回归。
5. `launch-check`
   上线后检查关键路径是否真的变好。

## 最低交付要求

- 改动要能解释“为什么这样改”。
- 关键路径要有验证证据。
- 如果有风险，要写清楚回退点。
- 如果只完成了一半，要明确剩余缺口。

## 不该做什么

- 不要只分析不落地。
- 不要先改完再想怎么验证。
- 不要把“能跑”当成“可交付”。

## 推荐协作

- `gov`：性能、稳定性、成本和观测跟进。
- `retro`：上线后复盘是否真的解决问题。
