# 🔬 Implement Thermo-L4 Metrics Computation

## Overview
Implement MVP surprisal/topic-entropy/coherence/Λ/T̂ computation for Thermo-L4 analysis.

## Requirements
- **No external dependencies** - use only standard library
- **MVP implementation** - basic but functional
- **JSONL output** - compatible with existing schema
- **Free energy calculation** - F̂ = α(Sₜ + Hₑ) − β(K + C⁻¹) + γT̂

## Metrics to Implement

### Core Metrics
- [ ] **Sₜ (Surprisal)**: Mean negative log-probability per token (proxy: token length variance)
- [ ] **Hₑ (Topic Entropy)**: Shannon entropy of unigram distribution
- [ ] **C (Compression)**: Length ratio (raw/L4 output)
- [ ] **K (Coherence)**: Bag-of-words cosine similarity between depths
- [ ] **Λ (Certainty)**: Normalized entropy inverse (1 - uncertainty)
- [ ] **T̂ (Temperature)**: Time per token proxy

### Free Energy
- [ ] **F̂ calculation** with configurable α/β/γ weights
- [ ] **JSONL output** with all metrics
- [ ] **Validation** against existing run data

## Implementation Notes
- Start with `thermo_l4_analyze.py` stub
- Use existing `templates/run_log.jsonl` for testing
- Focus on computational efficiency for real-time analysis

## Acceptance Criteria
- [ ] Processes existing run data correctly
- [ ] Outputs valid JSONL with all required fields
- [ ] Free energy calculation matches expected ranges
- [ ] No external dependencies required

## Priority
**High** - Required for immediate Thermo-L4 validation
