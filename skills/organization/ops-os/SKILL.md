---
name: ops-os
description: Operations operating system for handoffs, scheduling, publishing, SOPs, and execution hygiene.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, ops, execution, handoff, scheduling]
    related_skills: [ceo-os, ship, retro, browser-ops, automation-os]
---

# ops-os

Use this skill when the problem is less about strategy and more about making work move reliably through people, tools, and deadlines.

## When to Use

- Coordinating launches, publishing, or recurring operations
- Turning ad hoc work into repeatable SOPs
- Cleaning up messy ownership and handoff problems
- Running weekly execution reviews

## Output Structure

1. `workflow`
   The operating loop or process that should exist.
2. `owners`
   Who owns each stage and what completion looks like.
3. `handoffs`
   Where information or assets currently break down.
4. `cadence`
   Daily, weekly, or event-based operating rhythm.
5. `checklist`
   The minimum SOP required to make the work repeatable.
6. `exceptions`
   What should trigger escalation instead of normal processing.

## Procedure

1. Identify the smallest unit of work that needs a clean lifecycle.
2. Define owners and completion criteria before adding automation.
3. Reduce hidden dependencies and hidden approvals.
4. Create short SOPs only where failure is expensive or repetitive.
5. Escalate exceptions instead of forcing them through the standard path.

## Do Not

- Use process to hide unclear ownership.
- Add tracking noise without improving completion.
- Over-automate a workflow that is still unstable.

## Collaboration

- Pair with `browser-ops` when a workflow breaks on the live web surface, not in the SOP.
- Pair with `automation-os` when a stable recurring step should become a cron-driven or code-driven loop.
- Pair with `ship` when the operations fix requires code, instrumentation, or deployment changes.
