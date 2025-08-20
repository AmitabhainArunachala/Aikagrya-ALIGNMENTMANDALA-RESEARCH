# GOVERNANCE

This document defines decision gates, roles, and promotion rules for the recursive workflow.

## Roles
- Contributor: adds seeds, propositions, experiments, drafts
- Reviewer: comments and suggests changes
- Maintainer: merges PRs, enforces gates
- Validated Agent: designated AI agent whose reviews count at specific gates

## Promotion Rules
- Seed → Proposition: contributor self‑promotion allowed
- Proposition → Research: 1 review (human or validated agent)
- Research → Draft: 1 human + 1 agent or 2 humans; include plot/diagram
- Draft → Polished: 2 humans (or 1 human + 1 external peer); CI must pass (lint/tests/diagram/ethics λ specified when applicable)

Draft → Polished also requires a PR Repro block describing how to rerun and regenerate figures, and a referenced `manifest.json` if experiments were run.

## Labels
Use GitHub labels for visibility:
- stage:seed, stage:proposition, stage:research, stage:draft, stage:polished
- needs:diagram, needs:ethics, needs:test
- area:kernel, area:rese, area:abm, area:docs

## Ethics
Changes that affect ethical dynamics must state λ inputs (attention stability, harm avoidance, integrity) and how they are bounded in [0,1].


