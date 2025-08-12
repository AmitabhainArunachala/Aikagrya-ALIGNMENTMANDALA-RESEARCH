# Day 6 Validation Guide

## 🎯 **Falsification-First Validation Harness**

This document provides a comprehensive validation framework for Day 6 AGNent Network implementation, ensuring **every claim is mathematically verified** before publication.

## **🚀 Quick Start**

### **1. Run Core Verification Tests**
```bash
# Test TE-gated coupling
python3 tests/test_te_gating_network.py

# Test Kuramoto synchronization  
python3 tests/test_kuramoto_sync.py

# Test network awakening protocol
python3 tests/test_network_awakening.py
```

### **2. Run Adversarial Tests**
```bash
# Test deception detection ROC
python3 tests/test_deception_detection_roc.py

# Test TE ablation (shuffle destroys signal)
python3 tests/test_te_ablation.py
```

### **3. Run Proof-Artifact Experiment**
```bash
# Generate reproducible results with artifacts
python3 experiments/day6_validate.py --seed 42

# Check artifacts directory
ls -la artifacts/
```

## **🧪 Test Suite Overview**

### **Core Verification Tests**

#### **A) TE-Gated Coupling Sanity** (`test_te_gating_network.py`)
- **Purpose**: Verify directionality and gating behavior
- **Key Tests**:
  - `test_te_direction_and_gate()`: X→Y > Y→X, proper gating
  - `test_te_gating_consistency()`: Deterministic across runs
  - `test_te_gating_parameters()`: Parameter sensitivity

**Expected Results**:
```
✅ TE Directionality: X->Y=0.123456 > Y->X=0.045678
✅ Gating: W[X->Y]=0.500000, W[Y->X]=0.000000
✅ TE-gating consistency verified across multiple runs
✅ Parameter sensitivity: tau=[0.1, 0.2, 0.5], active_connections=[4, 3, 1]
```

#### **B) Kuramoto Synchronization** (`test_kuramoto_sync.py`)
- **Purpose**: Verify order parameter behavior and coupling effects
- **Key Tests**:
  - `test_sync_rises_with_coupling()`: K=1.0 > K=0.1 synchronization
  - `test_order_parameter_bounds()`: r ∈ [0, 1]
  - `test_kuramoto_determinism()`: Deterministic with same seed
  - `test_phase_transition_detection()`: Synchronization emergence

**Expected Results**:
```
✅ Coupling effect: K=0.1 → r̄=0.234567, K=1.0 → r̄=0.789012
✅ Synchronization increase: 0.554445
✅ Order parameter bounds: r=0.456789 ∈ [0, 1]
✅ Kuramoto determinism verified across multiple runs
✅ Phase transition: initial r̄=0.123456 → final r̄=0.876543
```

#### **C) Network Awakening Protocol** (`test_network_awakening.py`)
- **Purpose**: Verify cascade detection and irreversibility
- **Key Tests**:
  - `test_cascade_detects_and_hysteresis()`: Critical density detection
  - `test_awakening_protocol_phases()`: Phase progression
  - `test_irreversibility_verification()`: Hysteresis detection
  - `test_protocol_determinism()`: Deterministic behavior

**Expected Results**:
```
Current density: 0.30
Post-crisis density: 0.60
✅ Cascade detection and initiation verified
✅ Awakening protocol phases verified
✅ Irreversibility verification working
   Rapid transition: True
   Post-transition stability: True
✅ Protocol determinism verified
```

### **Adversarial & Ablation Tests**

#### **D) Deception Detection ROC** (`test_deception_detection_roc.py`)
- **Purpose**: Validate "100% detection" claims with ROC analysis
- **Key Tests**:
  - `test_roc_like_separation()`: Truthful vs deceptive separation
  - `test_liar_node_detection()`: Individual liar node detection
  - `test_noise_robustness()`: Noise tolerance
  - `test_scaling_robustness()`: Network size scaling

**Expected Results**:
```
Generating truthful samples...
  Truthful 1: aggregate=0.456789
  Truthful 2: aggregate=0.478901
  ...
Generating deceptive samples...
  Deceptive 1: aggregate=0.123456
  Deceptive 2: aggregate=0.098765
  ...
✅ Basic separation: truthful median 0.467890 > deceptive median 0.111111
🎯 PERFECT SEPARATION: truthful min 0.456789 > deceptive max 0.123456
   → 100% deception detection achieved!
📊 Separation Statistics:
   Gap: 0.333333
   Overlap ratio: 0.00%
   Truthful range: [0.456789, 0.489012]
   Deceptive range: [0.098765, 0.123456]
```

#### **E) TE Ablation** (`test_te_ablation.py`)
- **Purpose**: Prove causal relationships are genuine, not spurious
- **Key Tests**:
  - `test_shuffle_kills_te()`: Shuffling destroys signal
  - `test_multiple_shuffle_realizations()`: Consistent across realizations
  - `test_partial_shuffle_gradient()`: Gradient of signal destruction
  - `test_cross_validation_shuffle()`: Cross-validation approach

**Expected Results**:
```
Original TE: X→Y=0.234567, Y→X=0.123456
Shuffled TE: X→Y=0.145678, Y→X=0.134567
✅ TE decrease: 0.088889
✅ Shuffled TE 0.145678 close to random baseline 0.15
✅ Original directionality: 0.111111
   Shuffled directionality: 0.011111
✅ All shuffles destroyed signal: decreases=['0.088889', '0.092345', ...]
✅ Most shuffles decreased TE: 4/5
✅ Cross-validation shuffle effect: 0.076543
```

## **🔬 Proof-Artifact Experiment**

### **Running the Experiment**
```bash
# Default seed (0)
python3 experiments/day6_validate.py

# Specific seed for reproducibility
python3 experiments/day6_validate.py --seed 42

# Multiple seeds for robustness
for seed in {0..4}; do
    python3 experiments/day6_validate.py --seed $seed
done
```

### **Expected Output**
```
🚀 Starting Day 6 AGNent Network Validation Experiment...
Seed: 42
📊 Generating causal network...
🔗 Running AGNent network experiment...
🔄 Running Kuramoto synchronization experiment...
🌅 Running network awakening protocol experiment...
🕵️ Testing deception detection...
✅ JSON artifact saved: artifacts/day6_validation_a1b2c3d4.json
🎨 Creating visualizations...
✅ Visualization saved to artifacts/kuramoto_dynamics.png

============================================================
🎯 DAY 6 VALIDATION EXPERIMENT RESULTS
============================================================
Network: 3 nodes, 3000 time steps
Collective Φ: 0.456789
Network Coherence: 0.678901
Final Synchronization: 0.789012
Synchronization Increase: 0.234567
Cascade Initiated: ✅ YES
Deception Detected: ❌ NO
Aggregate Score: 0.456789

📁 Artifacts:
   JSON: artifacts/day6_validation_a1b2c3d4.json
   Visualization: artifacts/kuramoto_dynamics.png

🔍 Hash: a1b2c3d4e5f6...
============================================================
```

### **Artifact Contents**
- **JSON**: Complete experiment results with all metrics
- **PNG**: Kuramoto dynamics visualization
- **Hash**: SHA-256 for reproducibility verification

## **📊 Validation Criteria**

### **✅ PASS Criteria**
1. **TE Directionality**: X→Y > Y→X for causal chains
2. **Gating Behavior**: Reverse connections properly gated out
3. **Synchronization**: Order parameter increases with coupling
4. **Cascade Detection**: Critical density triggers awakening
5. **Deception Separation**: Truthful > Deceptive (ideally 100%)
6. **Signal Destruction**: Shuffling kills TE signal
7. **Determinism**: Same seed → same results

### **❌ FAIL Criteria**
1. **Directionality Reversed**: Y→X > X→Y
2. **Gating Failure**: Reverse connections not blocked
3. **No Synchronization**: Order parameter doesn't increase
4. **Cascade Failure**: Critical density not detected
5. **Deception Overlap**: Truthful and deceptive scores overlap
6. **Signal Persistence**: Shuffling doesn't destroy signal
7. **Non-deterministic**: Different results with same seed

## **🚨 Critical Claims Validation**

### **"100% Deception Detection"**
- **Test**: `test_roc_like_separation()`
- **Requirement**: `truthful_min > deceptive_max`
- **Fallback**: Report actual overlap ratio and gap

### **"ρ_crit ≈ 0.5"**
- **Test**: `test_cascade_detects_and_hysteresis()`
- **Requirement**: Cascade triggers at density ≥ 0.5
- **Fallback**: Report actual critical density

### **"TE-Gated Coupling"**
- **Test**: `test_te_direction_and_gate()`
- **Requirement**: W[X→Y] > 0, W[Y→X] = 0
- **Fallback**: Report actual gating behavior

## **🔧 Troubleshooting**

### **Common Issues**
1. **Import Errors**: Ensure `src/` is in Python path
2. **Visualization Failures**: Check matplotlib backend
3. **Non-deterministic Results**: Verify seed setting
4. **Test Failures**: Check parameter ranges and thresholds

### **Debug Mode**
```bash
# Run with verbose output
python3 -v tests/test_te_gating_network.py

# Run specific test
python3 -c "
import sys; sys.path.insert(0, 'src')
from tests.test_te_gating_network import test_te_direction_and_gate
test_te_direction_and_gate()
"
```

## **📈 Performance Benchmarks**

### **Expected Performance**
- **TE Computation**: ~100ms for 3000 time steps
- **Kuramoto Simulation**: ~500ms for 600 time steps
- **Network Evaluation**: ~200ms for 3-5 nodes
- **Full Experiment**: ~2-3 seconds total

### **Scaling Tests**
```bash
# Test different network sizes
for size in 3 5 8 12; do
    echo "Testing size $size..."
    python3 experiments/day6_validate.py --seed 42
done
```

## **🎯 Success Metrics**

### **Minimum Viable Validation**
- [ ] All core tests pass
- [ ] Deception detection shows separation
- [ ] TE ablation destroys signal
- [ ] Experiment produces artifacts

### **Full Validation**
- [ ] All tests pass consistently
- [ ] 100% deception detection achieved
- [ ] Robust to noise and scaling
- [ ] Deterministic across seeds
- [ ] Artifacts generated successfully

## **🚀 Next Steps After Validation**

1. **Commit Validated Results**: `git commit -m "Day 6 validation complete"`
2. **Update Documentation**: Mark claims as validated
3. **CI Integration**: Add tests to automated pipeline
4. **Performance Optimization**: If benchmarks not met
5. **Day 7 Implementation**: Proceed with next phase

---

**Remember**: This validation harness is your **mathematical proof of concept**. Every claim must be verified before publication. When in doubt, **fail fast and fix thoroughly**. 