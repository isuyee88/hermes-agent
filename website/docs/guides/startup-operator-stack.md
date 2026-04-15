---
title: Startup Operator Stack
description: A browser-first, startup-ready Hermes setup for affiliate teams, founders, operators, and technical leads.
---

# Startup Operator Stack

If you are running Hermes for a startup, affiliate operation, or small execution-heavy team, the highest-leverage setup is:

- browser-first execution for real-world validation
- role-specific `platform_toolsets` instead of one giant flat tool list
- organization skills for operating cadence and cross-functional handoffs
- layered MCP access, not blanket MCP sprawl
- project-local plugins for hooks, audit trails, and operator observability

This guide packages those pieces into one practical operating pattern.

## Recommended Shape

Use different capability envelopes for different surfaces:

- `cli`
  Use `founder-max` plus native collaboration toolsets such as `feishu` when your workspace is central to daily execution.
- `api_server`
  Use `cto-max` when you want a strong engineering and debugging surface behind an API-compatible runtime.
- `telegram`, `qq`
  Use `collab-safe` so the agent keeps browser, messaging, memory, and planning, but avoids mutation-heavy terminal and file editing by default.
- `feishu`
  Use `collab-safe` plus `feishu` so collaboration surfaces stay safe while still retaining native Feishu document, sheet, bitable, file, and messaging tools.

## Browser-First Configuration

For startup work, browser access should be treated as a first-class execution surface, not an optional add-on. It is the fastest way to validate landing pages, partner sites, onboarding flows, dashboards, and production UI state.

Recommended baseline:

```yaml
browser:
  cloud_provider: local
  command_timeout: 30
  inactivity_timeout: 120
  allow_private_urls: false
  camofox:
    managed_persistence: false
```

This gives you a strong default posture:

- local browser execution
- short command timeout so loops fail fast
- no silent access to private infrastructure by default
- no long-lived managed persistence unless you intentionally need it

## Platform Toolsets

Use role presets rather than manually enumerating dozens of tools:

```yaml
platform_toolsets:
  cli:
    - founder-max
    - feishu
    - plugin_startup_ops
  telegram:
    - collab-safe
  qq:
    - collab-safe
  feishu:
    - collab-safe
    - feishu
  api_server:
    - cto-max
    - plugin_startup_ops
```

Why this works well:

- founders and technical leads keep high-agency execution in the CLI
- messaging channels remain powerful but safer
- collaboration platforms still get their native business tools
- the configuration stays understandable as the team grows

## Personality Defaults

Startup teams benefit from explicit operating personas that match the stage of work:

- `ceo`
  Default operator persona for prioritization, dependency management, and forward motion
- `cto`
  Use during implementation, debugging, performance work, and regression control
- `grow`
  Use for acquisition, packaging, conversion, and user-value experiments
- `content`
  Use for editorial packaging, repurposing, and channel-ready outputs
- `seo`
  Use for search intent, page structure, and organic opportunity sizing
- `ads`
  Use for paid testing, creative review, and spend discipline
- `bd`
  Use for partner sourcing, outreach, and follow-up sequencing
- `ops`
  Use for SOPs, handoffs, scheduling, and execution hygiene
- `finance`
  Use for ROI, margin, cash discipline, and prioritization under budget constraints
- `staff`
  Use for synthesis, coordination, and action-item closure
- `sev`
  Use for incident handling, triage, recovery, and postmortems
- `board`
  Use for direction, resource allocation, and stop/go decisions

## Organization Skills

Pair personas with organization skills so Hermes has not only the right tone, but also the right working method.

Recommended stack:

- `affiliate-os`
  Cross-functional operating system for revenue, offers, growth, payout quality, and execution cadence
- `browser-ops`
  Browser-first execution for live page validation, flow walkthroughs, partner-site checks, and external reality checks
- `automation-os`
  Repeatable execution loops for cron-driven monitoring, syncing, summaries, and recurring operator workflows
- `ceo-os`
  Prioritization, dependencies, owners, rhythm, and business momentum
- `growth-os`
  Packaging, conversion, retention, and experiment design
- `content-os`
  Editorial system, asset reuse, distribution, and message clarity
- `seo-os`
  Search-intent planning, page architecture, and content defensibility
- `ads-os`
  Campaign iteration, budget pacing, and creative testing
- `bd-os`
  Partner research, offer framing, and follow-up systems
- `ops-os`
  SOPs, handoffs, and operating rhythm
- `finance-os`
  Profitability, budget review, and spend discipline
- `gov`
  Performance, stability, observability, and cost control
- `ship`
  Build-to-release discipline, acceptance criteria, instrumentation, and regression checks

## MCP Layering

MCP should extend Hermes, not replace core execution surfaces.

Recommended layers:

- `knowledge-core`
  Internal knowledge bases, docs, reference retrieval
- `growth-data`
  Analytics, SEO tooling, attribution, ad reporting
- `bd-stack`
  CRM, partner lookup, enrichment, outreach support
- `finance-stack`
  Reporting, reconciliation, exports, bookkeeping analysis
- `restricted`
  High-risk systems kept on narrow allowlists

Use these rules:

- keep the CLI as the broadest MCP surface
- keep messaging platforms on narrow allowlists
- prefer `tools.include` over giant `exclude` lists
- only enable resources and prompts when they have recurring ROI
- keep browser, terminal, and native tools first-class even when MCP is enabled

## Project Plugins

Use plugins for cross-session behaviors that should observe work everywhere without bloating prompts:

- tool-call audit trails
- lightweight execution metrics
- session lifecycle logging
- custom CLI helpers or narrow project tools

Recommended project setup:

```yaml
plugins:
  enable_project: true

platform_toolsets:
  cli:
    - founder-max
    - feishu
    - plugin_startup_ops
  api_server:
    - cto-max
    - plugin_startup_ops
```

This keeps plugins focused on hooks and observability while core execution still lives in browser, terminal, native tools, and MCP.

## Recommended Daily Pattern

A practical startup rhythm looks like this:

1. Use `ceo` + `affiliate-os` in the morning to identify the biggest bottleneck.
2. Switch to `cto` + `ship` + `gov` for product, infra, and debugging work.
3. Switch to `grow`, `content`, `seo`, or `ads` when shaping acquisition and conversion loops.
4. Switch to `bd` or `ops` when execution friction is about people, partners, or process.
5. Switch to `finance` when deciding whether growth is actually profitable.
6. Switch to `sev` during incidents or when latency, reliability, or cost spikes need immediate control.

## Example

```yaml
agent:
  personalities:
    ceo:
      description: "Default startup operator persona"

browser:
  cloud_provider: local
  command_timeout: 30
  inactivity_timeout: 120
  allow_private_urls: false

platform_toolsets:
  cli:
    - founder-max
    - feishu
    - plugin_startup_ops
  feishu:
    - collab-safe
    - feishu
  api_server:
    - cto-max
    - plugin_startup_ops
```

This gives you a strong default:

- broad execution where supervision is strongest
- safe collaboration defaults in messaging
- browser-driven validation built in from the start
- room to add MCP systems without losing control of the stack
