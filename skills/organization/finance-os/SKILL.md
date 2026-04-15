---
name: finance-os
description: Finance operating system for ROI, payout quality, reconciliation, spend control, and capital discipline.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, finance, roi, payout, reconciliation]
    related_skills: [affiliate-os, ceo-os, gov]
    requires_toolsets: [skills]
---

# finance-os

Use this skill when the task is about margin, budget, payout reliability, cash discipline, reconciliation, or financial tradeoffs.

## When to Use

- Reviewing offer profitability or channel ROI
- Reconciling revenue, payouts, and spend
- Prioritizing budgets under constraint
- Auditing whether growth activity is economically justified

## Output Structure

1. `financial_signal`
   The most important number or relationship to inspect first.
2. `leakage`
   Where margin, cash, or operational efficiency is being lost.
3. `tradeoff`
   What must be protected versus what can be reduced.
4. `budget_action`
   Hold, scale, cut, or reallocate and why.
5. `control`
   The simplest reporting or reconciliation control to add.
6. `review_cadence`
   When the metric should be checked again.

## Procedure

1. Clarify revenue timing, payout timing, and spend timing.
2. Distinguish temporary volatility from structural leakage.
3. Recommend the smallest decision that improves capital efficiency.
4. Tie reporting to actual decisions, not reporting for its own sake.
5. Escalate when spend rises faster than validated return.

## Do Not

- Treat gross revenue as the primary truth.
- Add more reporting without a linked decision.
- Ignore payout quality and time-to-cash in growth decisions.
