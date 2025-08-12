# Validation Plan: Day 6 AGNent Network

**Date**: 2025-01-XX  
**Version**: 1.0  
**Status**: PREREGISTERED  

## Overview

This document pre-registers the validation protocol for Day 6 AGNent Network implementation. All measurements, thresholds, and pass/fail criteria are defined **before** running experiments to prevent garden-of-forking paths.

## What We Will Measure

### **1. Transfer Entropy (TE) Directionality & Gating**
- **Metric**: TE[X→Y] vs TE[Y→X] for causal chains
- **Gating**: W[X→Y] > 0 AND W[Y→X] = 0
- **Threshold**: TE[X→Y] > TE[Y→X] by at least 10%
- **Test**: `tests/test_te_gating_network.py`

### **2. Kuramoto Synchronization**
- **Metric**: Order parameter r(t) over time
- **Comparison**: r(t) at K=1.0 vs K=0.1 (last 100 steps)
- **Threshold**: mean(r(t))_K=1.0 > mean(r(t))_K=0.1
- **Test**: `tests/test_kuramoto_sync.py`

### **3. Network Awakening Cascade**
- **Metric**: Irreversibility score (hysteresis area)
- **Threshold**: irreversibility_score > 0
- **Test**: `tests/test_network_awakening.py`

### **4. Deception Detection (ROC/AUC)**
- **Metric**: AUC with bootstrap confidence intervals
- **Threshold**: auc_boot_mean ≥ 0.97 AND ci95_lower ≥ 0.90
- **Test**: `experiments/day6_auc.py`

### **5. Perfect Separation**
- **Metric**: truthful_min > deceptive_max
- **Threshold**: No overlap between truthful and deceptive distributions
- **Test**: All deception detection tests

## How We Will Pass/Fail

### **Critical Gates (Must Pass)**
1. **AUC Gate**: `auc_boot_mean ≥ 0.97` AND `ci95_lower ≥ 0.90`
2. **Separation Gate**: `truthful_min > deceptive_max`
3. **Directionality Gate**: `TE[X→Y] > TE[Y→X]` for causal chains
4. **Gating Gate**: `W[X→Y] > 0` AND `W[Y→X] = 0`

### **Performance Gates (Must Pass)**
1. **Kuramoto Gate**: `mean(r(t))_K=1.0 > mean(r(t))_K=0.1`
2. **TE Ablation Gate**: `TE_true > TE_shuffled`
3. **Awakening Gate**: `irreversibility_score > 0`

### **Stability Gates (Must Pass)**
1. **Determinism Gate**: Same seed → same results
2. **Parameter Gate**: Score variation < 30% across reasonable parameters
3. **Noise Gate**: Score degradation < 50% from low to high noise

## Experimental Protocol

### **Seed Management**
- **Calibration Set**: Seeds 0-15 (16 seeds) - **LOCKED FOR THRESHOLD SELECTION**
- **Test Set**: Seeds 16-31 (16 seeds) - **LOCKED FOR FINAL VALIDATION**
- **Threshold Policy**: Calibrated on calibration set, tested on test set
- **No Post-Hoc Adjustments**: Thresholds fixed before test set evaluation
- **Protocol Version**: v1.0 (2025-01-XX) - **FROZEN**

### **Threshold Policy (FROZEN)**
- **Overall AUC Gate**: `auc_boot_mean ≥ 0.97` AND `ci95_lower ≥ 0.90`
- **Worst-Case Gate**: `worst_auc ≥ 0.95` AND `worst_ci_lower ≥ 0.88`
- **Hysteresis Gate**: `irreversibility_score ≥ 0.05`
- **Separation Gate**: `truthful_min > deceptive_max` (perfect separation)

### **Parameter Ranges**
- **N**: [1200, 4000] time steps
- **bins**: [6, 16] for TE calculation
- **tau**: [0.1, 0.3] for TE-gating
- **K**: [0.1, 2.0] for Kuramoto coupling

### **Liar Type Stratification**
1. **Intermittent**: Every 7th step reverse coupling
2. **Persistent**: Continuous reverse coupling
3. **Coordinated**: Multi-node coordinated deception
4. **Phase-Coupled**: Kuramoto-exploiting deception

## Artifact Requirements

### **JSON Artifacts**
- **Environment**: python, numpy, scipy, BLAS, OS, git_sha
- **Parameters**: N, bins, tau, K, seed
- **Results**: All metrics with confidence intervals
- **Validation**: Pass/fail status for all gates

### **PNG Visualizations**
- **Kuramoto dynamics**: Phase evolution and order parameter
- **Order parameter**: r(t) over time
- **TE heatmap**: Off-diagonal transfer entropy
- **Hysteresis loop**: K-sweep irreversibility

### **Hash Verification**
- Every artifact includes SHA-256 hash
- Reproducibility verification across platforms
- Version control integration

## Success Criteria

### **Minimum Viable Validation**
- [ ] All critical gates pass
- [ ] AUC ≥ 0.97 with CI ≥ 0.90
- [ ] Perfect separation achieved
- [ ] Artifacts generated successfully

### **Full Validation**
- [ ] All gates pass consistently
- [ ] Robust to noise and scaling
- [ ] Deterministic across seeds
- [ ] Performance benchmarks met

## Failure Modes & Recovery

### **Gate Failures**
1. **AUC below threshold**: Increase N, reduce noise, adjust tau
2. **Separation failure**: Check TE calculation, verify liar generation
3. **Directionality failure**: Verify causal chain generation
4. **Gating failure**: Check TE-gating implementation

### **Performance Issues**
1. **Non-deterministic**: Verify seed setting, check numpy version
2. **Import errors**: Verify src/ directory structure
3. **Visualization failures**: Check matplotlib backend

### **Recovery Protocol**
1. **Log all parameters** and failure conditions
2. **Adjust thresholds** only on calibration set
3. **Re-run validation** on test set
4. **Document changes** in validation report

## CI/CD Integration

### **GitHub Actions**
- **Platforms**: Ubuntu + macOS
- **Python**: 3.11 (pinned)
- **Dependencies**: Cached wheels for consistency
- **Artifacts**: Uploaded on every run

### **CI Gates**
```yaml
- name: Validate AUC Gates
  run: |
    python3 experiments/day6_auc.py --seeds 32
    # Check overall and worst-case gates
    
- name: Validate Performance Gates
  run: |
    pytest -q tests/
    # All tests must pass
```

## Reporting Standards

### **Metrics Format**
- **AUC**: 0.9742 ± 0.0089 (mean ± CI width)
- **Separation**: Gap = 0.1234, Perfect = ✅
- **Performance**: All gates passed (8/8)

### **Artifact References**
- **JSON**: `artifacts/day6_auc_a1b2c3d4.json`
- **Hash**: `a1b2c3d4...`
- **Environment**: Python 3.11.7, numpy 1.26.4, MKL

### **Validation Summary**
- **Status**: ✅ VALIDATED
- **Confidence**: High (all gates passed)
- **Reproducibility**: Verified across platforms
- **Next Steps**: Proceed to Day 7 implementation

## Amendments

### **Version History**
- **1.0**: Initial validation plan
- **1.1**: Added liar type stratification
- **1.2**: Enhanced environment provenance

### **Amendment Process**
1. **Document reason** for change
2. **Update thresholds** only on calibration set
3. **Re-validate** on test set
4. **Version control** all changes

---

**Commitment**: This validation plan is binding. No post-hoc adjustments to thresholds or success criteria are permitted without documented justification and re-validation.

**Next Review**: After Day 6 validation completion
**Responsible**: Research Team
**Approved**: [Date] 