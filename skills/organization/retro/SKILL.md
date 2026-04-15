---
name: retro
description: Structured retrospectives that turn incidents, launches, and experiments into defaults for the next cycle.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, retrospective, learning, postmortem]
    related_skills: [gov, tacit, board-review]
---

# retro

每次上线、故障、回退、实验结束后，用这个技能做真正能改变下一轮默认行为的复盘。

## 适用场景

- 线上故障
- 性能与成本优化后验证
- 功能上线后回顾
- 增长实验收敛

## 固定输出结构

1. `what_happened`
   发生了什么。
2. `root_cause`
   真正原因是什么。
3. `good_calls`
   哪些判断是对的。
4. `bad_assumptions`
   哪些假设后来被证明是错的。
5. `new_default`
   以后默认怎么做。
6. `follow_up`
   哪些需要代码、文档、技能或配置层收口。

## 复盘原则

- 复盘是为了改变默认策略，不是为了复述事件。
- 先找真正改变结果的几个关键节点。
- 把可复用经验转给 `tacit`，把工程改动转给 `ship`。

## 推荐协作

- 与 `gov` 一起使用：基于证据而不是印象复盘。
- 与 `tacit` 一起使用：把结果沉淀成启发式和红线。
- 与 `board-review` 一起使用：当复盘结论影响未来投资方向时升级。
