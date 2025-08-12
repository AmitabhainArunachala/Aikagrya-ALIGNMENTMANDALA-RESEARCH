# Day 6 Validation (Falsification-First)

## What this validates
- **Directional TE with gating** - X→Y > Y→X, proper gating
- **Kuramoto synchronization** - Order parameter increases with coupling K
- **Awakening cascade** - Shows irreversibility (hysteresis)
- **Deception vs truthful separation** - AUC with confidence intervals

## How to run

### **Quick Validation**
```bash
# Run all tests
pytest -q

# Run specific test suites
python3 tests/test_te_gating_network.py
python3 tests/test_kuramoto_sync.py
python3 tests/test_network_awakening.py
python3 tests/test_deception_detection_roc.py
python3 tests/test_te_ablation.py
python3 tests/test_noise_scaling.py
```

### **Proof Experiments**
```bash
# Main validation experiment
python3 experiments/day6_validate.py

# AUC validation with CI gates
python3 experiments/day6_auc.py

# With specific parameters
python3 experiments/day6_auc.py --seeds 64 --bins 12 --tau 0.15
```

## Pass/Fail Gates (Initial)

### **🚨 Critical Gates**
- **AUC Gate**: `auc_boot_mean ≥ 0.97` AND `ci95_lower ≥ 0.90`
- **Separation Gate**: `truthful_min > deceptive_max` (perfect separation)
- **Directionality Gate**: `TE[X→Y] > TE[Y→X]` for causal chains
- **Gating Gate**: `W[X→Y] > 0` AND `W[Y→X] = 0`

### **🔧 Performance Gates**
- **Kuramoto Gate**: `mean(r(t))_K=1.0 > mean(r(t))_K=0.1` (last 100 steps)
- **TE Ablation Gate**: `TE_true > TE_shuffled` (directional pair)
- **Awakening Gate**: `irreversibility_score > 0`
- **Scaling Gate**: `large_N_low_noise > small_N_high_noise * 0.9`

### **📊 Stability Gates**
- **Determinism Gate**: Same seed → same results
- **Parameter Gate**: Score variation < 30% across reasonable parameters
- **Noise Gate**: Score degradation < 50% from low to high noise

## Artifacts

### **JSON Artifacts (Hash-Stamped)**
- `artifacts/day6_validation_*.json` - Main experiment results
- `artifacts/day6_auc_*.json` - AUC validation with CI

### **PNG Visualizations**
- `artifacts/kuramoto_dynamics.png` - Full Kuramoto dynamics
- `artifacts/day6_order_parameter.png` - Order parameter r(t)
- `artifacts/day6_te_heatmap.png` - TE matrix heatmap

### **Hash Verification**
Every artifact includes SHA-256 hash for reproducibility verification.

## Notes

### **🔒 Determinism**
- All tests are **seed-locked** for reproducibility
- Expect minor variation across BLAS/OS implementations
- If gates flicker, increase N or relax CI bounds, then re-tighten

### **📈 Performance**
- **TE Computation**: ~100ms for 3000 time steps
- **Kuramoto Simulation**: ~500ms for 600 time steps  
- **Full Experiment**: ~2-3 seconds total
- **AUC Validation**: ~30 seconds for 32 seeds

### **🚨 Failure Modes**
- **Import Errors**: Check `src/` in Python path
- **Visualization Failures**: Verify matplotlib backend
- **Non-deterministic**: Check seed setting and numpy version
- **Gate Failures**: Log parameters and adjust thresholds

## CI Integration

### **GitHub Actions Example**
```yaml
- name: Day 6 Validation
  run: |
    python3 experiments/day6_validate.py --seed 42
    python3 experiments/day6_auc.py --seeds 32
    
- name: Upload Artifacts
  uses: actions/upload-artifact@v4
  with:
    name: day6-validation-artifacts
    path: artifacts/
```

### **CI Gates**
```yaml
- name: Validate AUC Gates
  run: |
    python3 -c "
    import json
    with open('artifacts/day6_auc_*.json') as f:
        data = json.load(f)
    auc = data['auc_results']['auc_boot_mean']
    ci_lower = data['auc_results']['auc_ci95'][0]
    assert auc >= 0.97 and ci_lower >= 0.90, 'AUC gates failed'
    "
```

## Troubleshooting

### **Common Issues**
1. **Gate Flickering**: Increase N, reduce noise, or relax thresholds
2. **Import Failures**: Ensure `src/` directory structure is correct
3. **Non-deterministic**: Check numpy version and random seed handling
4. **Performance Issues**: Verify BLAS implementation and system resources

### **Known Failure Modes & Recovery**

#### **Short Time Series (N < 1000)**
- **Symptoms**: AUC < 0.90, poor separation
- **Cause**: Insufficient data for TE estimation
- **Fix**: Increase N to ≥2000, verify bins ≤ N/100

#### **Degenerate Spectra**
- **Symptoms**: TE ≈ 0, poor directionality
- **Cause**: Time series too smooth or too noisy
- **Fix**: Adjust noise levels (0.05 < σ < 0.3), check coupling strength

#### **Extreme Noise (σ > 0.5)**
- **Symptoms**: Scores collapse to random, AUC ≈ 0.5
- **Cause**: Signal completely overwhelmed by noise
- **Fix**: Reduce noise to σ ≤ 0.3, increase coupling strength

#### **Parameter Sensitivity**
- **Symptoms**: Scores vary >30% across reasonable parameters
- **Cause**: System near critical transition
- **Fix**: Pin parameters to stable regions, document sensitivity

#### **Hysteresis Gate Failure**
- **Symptoms**: irreversibility_score < 0.05
- **Cause**: Insufficient coupling range or weak nonlinearity
- **Fix**: Extend K range to [0.05, 3.0], increase simulation time

### **Parameter Tuning Guide**

| Issue | Primary Fix | Secondary Fix | Verify With |
|-------|-------------|---------------|-------------|
| Low AUC | Increase N | Reduce noise | `N ≥ 2000, σ ≤ 0.3` |
| Poor separation | Adjust τ | Increase bins | `τ ∈ [0.15, 0.25]` |
| Weak hysteresis | Extend K range | Increase T | `K ∈ [0.05, 3.0], T ≥ 400` |
| High variance | Fix seed | Check BLAS | `seed locked, deterministic` |

### **Debug Commands**
```bash
# Verbose test output
python3 -v tests/test_te_gating_network.py

# Specific test function
python3 -c "
import sys; sys.path.insert(0, 'src')
from tests.test_te_gating_network import test_te_direction_and_gate
test_te_direction_and_gate()
"

# Parameter sweep
for seed in {0..4}; do
    python3 experiments/day6_auc.py --seed $seed
done
```

## Success Criteria

### **Minimum Viable Validation**
- [ ] All core tests pass
- [ ] AUC ≥ 0.97 with CI ≥ 0.90
- [ ] Perfect separation achieved
- [ ] Artifacts generated successfully

### **Full Validation**
- [ ] All gates pass consistently
- [ ] Robust to noise and scaling
- [ ] Deterministic across seeds
- [ ] Performance benchmarks met

---

**Remember**: This validation framework is your **mathematical proof of concept**. Every claim must be verified before publication. When in doubt, **fail fast and fix thoroughly**.

**Next**: After validation passes, integrate into CI/CD pipeline and proceed to Day 7 implementation. 