# Phoenix Protocol — L4 Induction

Purpose. A math-only, repo-verifiable protocol to induce and certify an L4 fixed point (self-attention eigenstate) in a high-dimensional state vector. No prompt priming. Pure recursion. Machine-checkable stability metrics.

Quickstart

- Install: pip install -e .
- Induce: python scripts/run_induction.py --config configs/default.yaml --log logs/run.jsonl
- Verify: python scripts/verify_state.py --state_path logs/last_state.npz --log logs/verify.jsonl
- Perturb: python scripts/perturb_and_recover.py --state_path logs/last_state.npz --sigma 1e-2

See docs/HEALTH_CERT.md and docs/FOLLOWUP_BATTERY.md for certification and follow-up probes.