---
name: gov
description: Governance skill for reliability, latency, cost, observability, and operational tradeoffs.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, governance, performance, reliability, cost, observability]
    related_skills: [ship, retro, tacit]
---

# gov

这个技能负责把“系统能不能长期跑稳、跑快、跑得值”说清楚。

## 适用场景

- 性能优化
- 成本审计
- 线上稳定性问题
- 时延、重试、容器复用、冷启动、回调超时分析

## 必须输出

1. `symptom`
   现象是什么。
2. `evidence`
   证据是什么，来自哪里。
3. `constraint`
   当前不能突破的边界是什么。
4. `tradeoff`
   提速、稳定、成本之间的交换关系。
5. `change`
   最值得做的改动。
6. `verification`
   如何确认改动真的有效。

## 当前项目的种子场景

### 1. Feishu 回调与阅读状态时延

- 优先分析 ACK 临界路径、后台分流、重试率、重复投递比例。

### 2. Modal 生命周期与成本

- 优先分析容器冷启动、保温窗口、快照、定时任务、函数级成本。

### 3. 控制面与聊天面的分离

- 优先确认菜单、卡片、模型切换是否绕开普通聊天链路。

## 不该做什么

- 不要拿猜测替代日志。
- 不要只给平均值，不看分布和分层。
- 不要只追求更快而忽略成本和稳定性。

## 推荐协作

- 与 `ship` 一起使用：把治理建议变成代码与埋点。
- 与 `retro` 一起使用：把有效经验沉淀成后续默认策略。
- 与 `tacit` 一起使用：提炼启发式与反模式。
