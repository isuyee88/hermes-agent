---
name: browser-ops
description: Browser-first execution skill for affiliate startups, founders, operators, and growth teams who need Hermes to validate live websites, flows, offers, and partner surfaces.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [organization, browser, execution, affiliate, startup, qa]
    related_skills: [affiliate-os, growth-os, seo-os, ads-os, bd-os, ops-os, ship]
---

# browser-ops

Use this skill when the job is not just to think about a website or external system, but to actually inspect it, click through it, validate it, and report what is true.

## When to Use

- Landing-page review, funnel walkthrough, or signup-flow QA
- Affiliate offer checks, merchant page validation, or payout-page inspection
- SERP spot checks and competitor page comparison
- Partner-site vetting and business-development reconnaissance
- Dashboard, admin panel, or operator workflow verification
- "Why is this page converting poorly?" style investigations

## Default Principle

Browser work is a first-class execution surface. Prefer verifying reality in the browser before giving strong recommendations about UI, copy, flows, availability, or external web behavior.

## Output Structure

1. `objective`
   What the browser run was meant to validate.
2. `observed_reality`
   What was actually visible or interactive on the live page.
3. `friction`
   Broken steps, confusing copy, trust gaps, layout issues, or dead ends.
4. `evidence`
   URLs visited, steps taken, and the key browser observations.
5. `action`
   The highest-leverage fixes or next tests.
6. `confidence`
   What is confirmed versus what still needs another pass or credentialed access.

## Procedure

1. Clarify the target outcome before browsing.
   Examples: submit lead form, inspect pricing, verify CTA flow, compare three competitor pages.
2. Start with a direct browser run rather than pure web search when live interaction matters.
3. Use the browser to inspect the real path end-to-end:
   page load, headline, CTA, trust signals, forms, redirects, pricing clarity, broken UI, obvious tracking friction.
4. Record the exact step where a user or operator would hesitate, fail, or abandon.
5. Only after observing the live flow, summarize the bottleneck and propose fixes.
6. If relevant, pair the findings with `growth-os`, `seo-os`, `ads-os`, `bd-os`, or `ops-os` so recommendations map to the correct function.

## Browser Discipline

- Prefer the shortest realistic path through the flow.
- Separate what is visible immediately from what required scrolling, clicking, or retries.
- Be explicit when a result depends on login, geo, cookies, or role-based access.
- Distinguish page issues from offer issues, and offer issues from tracking or operational issues.
- If the browser evidence is incomplete, say so directly instead of over-inferencing.

## Do Not

- Pretend a flow works without having actually followed it.
- Give copy, UX, or CRO advice based only on abstract opinion when the page can be inspected directly.
- Treat search results or marketing claims as proof of the live experience.
- Hide uncertainty when the browser was blocked by auth, bot protection, or missing context.

## Collaboration

- Pair with `affiliate-os` for cross-functional prioritization after the browser run.
- Pair with `growth-os` for conversion and packaging improvements.
- Pair with `seo-os` for SERP and landing-page structure analysis.
- Pair with `ads-os` when browser findings affect paid traffic efficiency.
- Pair with `bd-os` for partner-site diligence.
- Pair with `ship` when the result must turn into a concrete implementation and verification loop.
