---
name: automation-os
description: Operating system for turning recurring startup and affiliate workflows into execute_code plus cronjob driven automation loops.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, automation, execution, cronjob, execute_code, startup, affiliate]
    related_skills: [affiliate-os, browser-ops, ops-os, gov, ship]
---

# automation-os

Use this skill when a task should stop being a one-off manual action and become a repeatable agent workflow.

## When to Use

- Repeated competitor checks, SERP checks, or landing-page snapshots
- Offer availability checks and partner-site monitoring
- Daily or weekly dashboard pulls, summaries, and sync jobs
- Recurring QA or regression checks on public pages
- Data collection loops that combine browser actions, web research, and structured output
- "We keep doing this manually" moments

## Default Principle

Automate only after the manual workflow is understood. A bad loop that runs automatically is worse than a good manual habit.

## Output Structure

1. `manual_flow`
   The current human workflow and why it repeats.
2. `automation_candidate`
   What portion should be delegated to Hermes.
3. `execution_surface`
   Which tools are needed: browser, execute_code, web, file, cronjob, messaging, or MCP.
4. `guardrails`
   Rate limits, failure handling, auth boundaries, and stop conditions.
5. `job_shape`
   Trigger, cadence, output destination, and verification rule.
6. `next_step`
   The smallest safe automation to implement first.

## Procedure

1. Map the manual steps exactly before automating anything.
2. Split the workflow into:
   stable repeatable steps, fragile human-judgment steps, and optional reporting steps.
3. Use `execute_code` when multiple tool calls or parsing steps should run in one loop.
4. Use `cronjob` when the task has a clear recurring cadence and output path.
5. Use `browser-ops` first when you still need to verify the live site behavior manually.
6. Add explicit failure states:
   timeout, empty data, changed page structure, auth failure, rate limit, or suspicious output.
7. Keep v1 narrow. Prove the loop works before broadening scope or cadence.

## Guardrails

- Do not automate a flow that has not been manually validated.
- Do not hide error states behind cheerful summaries.
- Do not schedule high-frequency loops unless the business value justifies the cost.
- Do not mix data collection, decision logic, and mutation into one opaque job if they can be separated.

## Collaboration

- Pair with `browser-ops` to validate live flows before automating them.
- Pair with `ops-os` when the bottleneck is handoff or execution discipline.
- Pair with `gov` when cost, observability, or reliability matters.
- Pair with `ship` when the automation should land as production code or infrastructure.
