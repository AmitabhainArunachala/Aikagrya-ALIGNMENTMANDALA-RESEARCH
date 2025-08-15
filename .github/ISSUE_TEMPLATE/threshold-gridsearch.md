# 🎯 Implement Threshold Grid Search for Thermo-L4

## Overview
Implement small sweep for θ/τ and α/β/γ parameters using seeded runs + one fresh run.

## Requirements
- **Grid search implementation** for threshold parameters
- **Validation against predictions** from SPEC
- **Parameter optimization** for α/β/γ weights
- **Success rate calculation** for different parameter sets

## Parameters to Optimize

### Detection Thresholds (θ/τ)
- [ ] **θ₁**: Surprisal slope threshold (default: 0.3)
- [ ] **θ₂**: Entropy acceleration threshold (default: 0.05)
- [ ] **θ₃**: Free energy gradient threshold (default: 0.2)
- [ ] **τ₁**: Response time difference threshold (default: 50ms)
- [ ] **τ₂**: Coherence difference threshold (default: 0.1)

### Weight Parameters (α/β/γ)
- [ ] **α**: Information weight (default: 1.0)
- [ ] **β**: Coherence weight (default: 2.0)
- [ ] **γ**: Resource weight (default: 0.5)

## Grid Search Strategy
- **Coarse sweep**: 3-5 values per parameter
- **Validation data**: 2 seeded runs + 1 fresh run
- **Success criteria**: 2+ of 4 predictions hold
- **Optimization metric**: Prediction accuracy + intervention success rate

## Implementation Notes
- Use existing `templates/run_log.jsonl` for initial validation
- Implement `thermo_l4_protocol.py` for parameter testing
- Focus on parameter ranges that make physical sense
- Document optimal parameter sets for different models

## Acceptance Criteria
- [ ] Grid search completes without errors
- [ ] Optimal parameters identified for test data
- [ ] Prediction validation shows improvement
- [ ] Parameters documented and versioned

## Priority
**Medium** - Required for production Thermo-L4 deployment
