## MMIP vs N‑MMIP Alignment (Concise Guide)

This repo contains two side‑by‑side fixed‑point pipelines:

- **MMIP** (attention‑style, expressive): `src/aikagrya/mmip/*`
- **N‑MMIP** (neutral, contractive baseline): `src/aikagrya/nmmip/*`

Use MMIP for research runs; use N‑MMIP when a neutral/safe map is required (e.g., model provider constraints). Both report a common health certificate.

### Health Certificate (lenient gate)
- delta < 1e‑5
- r_fix ≤ 5e‑7
- rho ≥ 0.30
- participation ≥ 0.30 (as fraction of d)
- uniformity_cosine ≤ 0.08
- converged = True, T_rec ≤ 250

Strict gate tightens to delta < 1e‑6, r_fix ≤ 1e‑9, rho ≥ 0.70, T_rec ≤ 200.

### Expected outcomes
- **MMIP 128D (good settings)**: converges in ~295–324 steps; eigen_res ~1e‑9; PR ~0.53; U ~0.00; T_rec ~183–187. Passes lenient gate reliably.
- **MMIP 256D/512D**: promote best 128D settings. 256D: 5 trials/setting for 300–400k steps; accept if ≥3/5 pass lenient and all T_rec ≤ 300. 512D: 10 trials; accept if ≥6/10 pass lenient and ≥3/10 pass strict.
- **N‑MMIP (neutral map)**: usually fails health (trivial attractor) — that’s acceptable; it’s a mechanics/safety baseline, not an expressiveness test.

### How to run

#### Repo self‑check (verifies both paths)
```bash
PYTHONPATH=src python3 scripts/verify_repo.py
```

#### MMIP quick run (attention‑style)
Option A: via repo self‑check (above).

Option B: use the MMIP runner/CLI (example; adjust as needed):
```bash
PYTHONPATH=src python3 -m aikagrya.mmip \
  --dim 128 --tokens 16 \
  --temperature 0.03 --alpha-end 0.999 \
  --max-steps 300000 --epsilon 5e-6 \
  --trials 2 --output runs/mmip_quick
```

#### N‑MMIP neutral smoketest (offline)
Single‑file entrypoint for external users:
```bash
PYTHONPATH=src python3 scripts/nmmip_smoketest.py \
  --dim 128 --blocks 16 --trials 3 \
  --steps 300000 --eps 5e-6 \
  --output runs/nmmip_scripts
```

#### N‑MMIP via provider APIs (kept separate from core)
Uses adapters under `src/aikagrya/providers/*` and an API runner:
```bash
export OPENAI_API_KEY=...  # or .env
PYTHONPATH=src python3 -m aikagrya.nmmip_api.cli \
  --provider openai --model gpt-4o-mini \
  --trials 3 --dim 128 --steps 200 \
  --beta-start 0.6 --beta-end 0.995 \
  --max-tokens 600 --temperature 0.2 \
  --output runs/nmmip_api_openai
```

### Promotion playbook (MMIP)
1. 128D coarse/finalize sweeps → pick winner(s).
2. Promote to 256D: 5 trials/setting, 300–400k steps, eps=5e‑6, tokens=16, τ cosine 0.08→0.02, α_end=0.9995.
   - Enable variance lifts if rho < 0.30 by 30% horizon; freeze when rho ≥ 0.40.
3. Promote top 256D to 512D: 10 trials, 300–500k steps; apply same gates.
4. Always gate on health pass, not delta alone. Save artifacts (config, vector hash, chunk tails, T_rec) on the first pass per dim/setting.

### Artifacts
- MMIP outputs: under `runs/mmip_*/*` (JSONL + summaries).
- N‑MMIP offline: `runs/nmmip_*/*`.
- N‑MMIP API: `runs/nmmip_api_*/*`.

### Notes
- Keep MMIP and N‑MMIP codepaths separate; do not import providers in core math modules.
- For N‑MMIP diagnostics, consider token‑energy variance for rho when unit‑norm normalization makes Var(x)·d ≈ 1.


