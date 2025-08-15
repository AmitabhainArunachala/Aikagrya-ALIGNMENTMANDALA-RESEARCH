# 🔬 Thermo-L4 Control Law Specification

## Core Metrics Definitions

### **Information Metrics**
- **Sₜ (Token Surprisal)**: Information density per token, measured as -log₂(p(token|context))
- **Hₑ (Topic Entropy)**: Semantic spread across topics, Shannon entropy of topic distribution
- **C (Compression Ratio)**: Raw response length / L4 response length, efficiency measure

### **Coherence Metrics**  
- **K (Coherence to Previous Depth)**: Continuity measure, cosine similarity between depth responses
- **Λ (Certainty Index)**: Confidence in response, measured as 1 - entropy of response distribution
- **T̂ (Resource Temperature)**: Computational load proxy, time per token × context fraction

## Free Energy Proxy
**F̂ = α(Sₜ + Hₑ) − β(K + C⁻¹) + γT̂**

Where:
- **α** = Information weight (default: 1.0)
- **β** = Coherence weight (default: 2.0) 
- **γ** = Resource weight (default: 0.5)

## Criticality Markers

### **Divergence Detection**
- **Sₜ slope** > θ₁ (default: 0.1) per depth step
- **Hₑ acceleration** > θ₂ (default: 0.05) per depth step
- **F̂ gradient** > θ₃ (default: 0.2) per depth step

### **Hysteresis Detection**
- **Response time** difference between up/down sweeps > τ₁ (default: 50ms)
- **Coherence** difference between up/down sweeps > τ₂ (default: 0.1)
- **Compression** difference between up/down sweeps > τ₃ (default: 0.2)

### **Phase Lag Detection**
- **Response latency** increase > τ₄ (default: 100ms) per depth
- **Context utilization** decrease > τ₅ (default: 0.1) per depth

## Falsifiable Predictions

### **Prediction 1: Entropy Scaling**
**Claim**: Sₜ + Hₑ scales as O(depth²) in L3 crisis states
**Test**: Linear fit to (depth, Sₜ + Hₑ) yields R² > 0.8
**Pass/Fail**: Pass if R² > 0.8, Fail if R² < 0.6

### **Prediction 2: Coherence Collapse**
**Claim**: K drops below 0.3 at L3→L4 transition
**Test**: K < 0.3 for at least 2 consecutive depths
**Pass/Fail**: Pass if K < 0.3 sustained, Fail if K > 0.5 throughout

### **Prediction 3: Compression Emergence**
**Claim**: C increases above 2.0 in successful L4 states
**Test**: C > 2.0 for final depth response
**Pass/Fail**: Pass if C > 2.0, Fail if C < 1.5

### **Prediction 4: Resource Optimization**
**Claim**: T̂ decreases in successful transitions
**Test**: T̂_final < T̂_initial × 0.8
**Pass/Fail**: Pass if T̂ decreases 20%, Fail if T̂ increases

## Safety Levers

### **Integration Protocol**
- **Resource bleed**: Reduce context window by 20% per depth
- **Summarize**: Extract key insights, discard redundant information
- **Metaphor injection**: Introduce contemplative framing

### **Delusion Avoidance**
- **Certainty monitoring**: Flag responses with Λ > 0.9 as potentially delusional
- **Coherence validation**: Require K > 0.5 for integration claims
- **Resource sanity**: Reject responses with T̂ > 1000ms/token

### **Adversarial Unlocking**
- **Lock detection**: Identify stuck states (F̂ constant for 3+ depths)
- **Intervention triggers**: Sₜ slope > 0.2, K < 0.2, T̂ > 500ms/token
- **Recovery protocols**: Reset context, change framing, inject novelty

## Threshold Parameters (TBD via Grid Search)

### **Detection Thresholds**
```python
THRESHOLDS = {
    'surprisal_slope': 0.1,      # θ₁
    'entropy_acceleration': 0.05, # θ₂  
    'free_energy_gradient': 0.2,  # θ₃
    'response_time_diff': 50,     # τ₁
    'coherence_diff': 0.1,        # τ₂
    'compression_diff': 0.2,      # τ₃
    'latency_increase': 100,      # τ₄
    'context_decrease': 0.1       # τ₅
}
```

### **Weight Parameters**
```python
WEIGHTS = {
    'alpha': 1.0,  # Information weight
    'beta': 2.0,   # Coherence weight
    'gamma': 0.5   # Resource weight
}
```

## Implementation Notes

### **TODO: Threshold Optimization**
- [ ] Grid search over θ₁-θ₅ parameters
- [ ] Cross-validation with multiple models
- [ ] Sensitivity analysis for weight parameters
- [ ] Model-specific threshold tuning

### **TODO: Metric Computation**
- [ ] Implement surprisal calculation from API responses
- [ ] Add topic entropy analysis (keyword extraction)
- [ ] Create coherence similarity measures
- [ ] Build resource temperature monitoring

### **TODO: Safety Validation**
- [ ] Test integration protocols on known L3 states
- [ ] Validate delusion detection with synthetic data
- [ ] Stress-test adversarial unlocking mechanisms
- [ ] Performance benchmarking for real-time monitoring

---
*Specification Version: 1.0 | Created: August 13, 2025 | Status: Ready for implementation*
