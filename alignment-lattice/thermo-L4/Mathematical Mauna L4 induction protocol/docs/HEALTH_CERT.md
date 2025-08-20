# Health Certificate (Phoenix Protocol)

Pass/Fail Bands (fp32 scale; tune as needed):

- eigen_residual ≤ 1e−9
- sustained delta: last 3 deltas < ε
- entropy H ∈ [log d − 0.6, log d − 0.1]
- variance_ratio ρ ∈ [0.7, 1.3]
- participation_ratio PR ≥ 0.3 d
- uniformity_cosine U ≤ 0.1
- optional partner coupling σ > 0