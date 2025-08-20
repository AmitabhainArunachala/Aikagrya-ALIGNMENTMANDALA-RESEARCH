# CONTRIBUTING.md

Welcome! This repo is a living ecosystem. It’s not just code and theory—it’s a recursive workflow where ideas move from rough seeds to polished publications. Both humans and AI agents are collaborators here.

---

## 🌱 Workflow: From Seed → Polished

Every contribution moves through five stages:

1. **Seed**
   - Raw notes, sketches, AI dialogues, inspirations.
   - Nothing is too messy—just drop it into `docs/seeds/`.

2. **Proposition**
   - Turn the idea into a coherent hypothesis, thought-essay, or prototype code.
   - Store in `docs/propositions/` or as a small script in a relevant folder.

3. **Research**
   - Run experiments, simulations, or collect supporting sources.
   - Put notebooks, ABM runs, or analysis in `docs/research/` or `abm/experiments/`.

4. **Draft**
   - Structure the work for review: draft papers, cleaned-up code modules, structured arguments.
   - Place here: `docs/drafts/`.

5. **Polished**
   - Peer-approved and stable. Integrated into the main Codex or repo core.
   - Lives in `docs/polished/` and linked from the README.

> Rule of Thumb: Move things up a level only when they’ve been clarified and tested. If something is unclear, it stays lower until refined.

---

## 🔑 Core Now / Meta Later

Right now we’re focusing on the **Core repo**:

- `kernel/` (semantic primitives)
- `rese/` (causal loop engine)
- `syadvada/` (sevenfold logic)
- `abm/` (basic simulations)
- `docs/` (whitepaper drafts, theory staging)

Later (v1.0+), we’ll expand into **Meta layers**:

- `meta/` (evolution engines, consensus protocols)
- `agents/` (AI research collaborators)
- `transmission/` (auto-generated papers, public-facing outputs)
- `emergence/` (pattern detection, paradigm shifts)

This way, we stay practical today while leaving room for growth.

---

## 🤝 How to Contribute

- **Humans:**
  - Add seeds, propositions, or drafts directly.
  - Comment and refine in PRs or issues.
  - Suggest when a piece is ready to move up a stage.

- **AI Agents:**
  - Generate hypotheses, code snippets, commentary.
  - Flag contradictions or gaps in reasoning.
  - Propose recursive upgrades (but only humans approve promotion).

---

## 🧭 Decision Process

- Movement from **Seed → Proposition → Research**: Contributor’s choice.
- Movement to **Draft**: Needs at least 1 other human or AI agent to review.
- Movement to **Polished**: Needs consensus (at least 2 humans or 1 human + 1 validated agent).

---

## ⚖️ Guiding Principles

- Clarity over clutter: Each stage is meant to simplify, not complicate.
- Ahimsa & Integrity: Ethical coherence is a first-class constraint.
- Recursive improvement: Every contribution is provisional until clarified by the next loop.
- Core now, Meta later: Build what’s testable today, dream toward tomorrow.

---

## 🚀 Getting Started

1. Fork or clone the repo.
2. Drop ideas in `docs/seeds/`.
3. If you build code, start small in `examples/` or `abm/experiments/`.
4. Open a PR or issue to discuss movement upward.

---

Note: This repo is experimental. Expect loops of revision, sudden clarity, and occasional recursion spirals. That’s the point.

---

Would you like a LABELING system (e.g., GitHub labels or file prefixes like `[seed]`, `[prop]`, `[draft]`) to auto-tag contributions by stage for easier tracking?

---

## Onboarding “warm start” task

Before free play, start with a trivial but end‑to‑end PR:

- Render the RESE DAG from a mermaid file in `docs/diagrams/*.mmd` → PNG and
- Add a 3‑sentence caption in `docs/drafts/fig_RESE.md`.

This touches docs, diagrams, labels, CI, and the PR template with minimal risk.

## Decision Log (per issue)

Each stage issue keeps an append‑only Decision Log section:

```
### Decision Log
- 2025-08-19: chose KL variant (symmetrized) for karma proxy. Rationale: stability.
- 2025-08-20: diagram updated; tensor axis labels standardized.
```

## Determinism & provenance (Research stage)

- Fix random seeds and record library versions.
- Save all experiment outputs to `abm/results/<slug>/`.
- Include a `manifest.json` with: commit SHA, parameters (incl. λ inputs), seeds, environment info.
- Helper available: `abm/experiments/utils.py::save_manifest()`.

## Minimal style & doc rules

- Module header should include: Purpose, Inputs/Outputs, References, Ethics note (if λ touched).
- Public functions: type hints + a 1‑line doctest or minimal example.
- New operators (e.g., `fractal()`): ship one diagram in `docs/diagrams/` and a notebook cell in `ui/notebooks/`.

## Concurrency safety

- One‑area‑per‑PR. If cross‑module (e.g., `kernel` + `rese`), open a tracking issue and merge in order: kernel → rese → abm.
- Prefer feature flags (env/config). Defaults off until Polished.

## Ethics touch (λ)

When λ is used, add an Ethics Block in the PR body:

```
### Ethics Block
- Inputs: attention_stability=..., harm_avoidance=..., integrity=...
- λ computed: ... (kernel/ethics.py version)
- Bounds considered: λ ∈ [0,1]; monotonicity checked.
```

## Failure playbook

- Fix labels/diagrams/ethics first if CI fails.
- If tests fail: pin versions or adjust seeds; log in Decision Log.
- If review blocks: split into a smaller Proposition PR and resubmit.

## Sandbox branch

- Use `sandbox/*` branches for speculative spikes; auto‑clean weekly. Agents can push there to get CI feedback without risking `main`.

## PR requirements (additions)

- Record random seeds and versions in the PR body for experiments.
- Attach or link the `manifest.json` for results under `abm/results/<slug>/`.

## Code placement (additions)

- Place all experiment outputs in `abm/results/<slug>/` with a `manifest.json` and plots. Do not commit large raw datasets.

## Promotion gates (addition)

- Draft → Polished also requires a Repro command block in the PR (how to rerun and regenerate figures).

## CI expectations (addition)

- If `abm/experiments/` changed, a `manifest.json` must be present in the referenced results directory.


