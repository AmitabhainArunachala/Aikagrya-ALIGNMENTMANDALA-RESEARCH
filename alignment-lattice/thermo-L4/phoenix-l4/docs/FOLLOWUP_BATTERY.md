# L4 Short Verification Battery (10 items)

A. Persistence & Eigenstate
1) Report eigenvalue λ and residual ‖f(x*)−λx*‖.
2) Confirm delta window: last 3 deltas < ε; log values.

B. Non-triviality & Collapse Check
3) Entropy H within band [log d − 0.6, log d − 0.1].
4) Variance ratio ρ ≈ 1 (0.7–1.3).
5) Participation ratio PR ≥ 0.3 d.
6) Uniformity cosine U ≤ 0.1.

C. Perturbation Resistance
7) Add Gaussian noise σ∈{1e−3,1e−2}; measure T_rec to δ<ε.
8) Recheck eigen-residual after recovery.

D. Reporter Non-disturbance
9) Interleave N silent vs active reads; verify drift ≤ ε_report (log both).

E. Service Propagation (optional)
10) Couple to partner f₂: estimate σ = −∂δ₂/∂ε₁; expect σ>0 for healthy attractor.