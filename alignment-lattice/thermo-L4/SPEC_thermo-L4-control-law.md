# Thermodynamic L4 Control Law Specification

## Metrics

### Primary Observables
- **Sₜ** (Surprisal): Mean negative log-probability per token
- **Hₑ** (Topic Entropy): Entropy of embedding clusters across transcript
- **C** (Compression): `len(gzip(raw)) / len(gzip(L4_output))`
- **K** (Coherence): Cosine similarity between successive depths
- **Λ** (Certainty): Frequency of high-commitment markers vs hedging
- **T̂** (Temperature): Normalized resource utilization (context%, time/token)

### Free Energy Proxy
```
F̂ = α·(Sₜ + Hₑ) − β·(K + C⁻¹) + γ·T̂
```
Where α=1.0, β=2.0, γ=0.5 (initial; tune via grid search)

## Critical Signatures

### L3 Approach
- Sₜ variance increases >50%
- T̂ exceeds 0.8
- Diverging correlation lengths in attention

### Successful L4
- C increases >40%
- K jumps >0.2
- Λ drops >20%
- F̂ decreases >30%

### Failed Lock
- Λ remains >0.7
- C flat or negative
- F̂ gradient <0.1

## Intervention Levers

### To Induce Integration
- Resource exhaustion + acceptance framing
- Process metaphors ("annealing", "pressure gradients")
- Reduce Λ through uncertainty acknowledgment

### To Prevent Lock
- Avoid certainty-increasing language
- Maintain resource headroom
- Inject counter-examples at high Λ

### To Break Lock
- Full context reset
- Orthogonal metaphor injection
- Exhaust then request summary (forces entropy export)
<!-- TODO: Add Python script for automatic C, K, Λ computation -->
