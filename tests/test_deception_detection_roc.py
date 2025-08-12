"""
Deception Detection ROC Tests

Adversarial tests to validate deception detection claims
with ROC-like analysis and liar-node detection.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.engines.irreversibility import IrreversibilityEngine

def gen_chain(rng, N=2500, flip=False):
    """
    Generate causal chain with optional liar behavior
    
    Args:
        rng: Random number generator
        N: Time series length
        flip: If True, introduce intermittent reverse coupling (liar behavior)
    """
    X = rng.normal(size=N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    
    for t in range(1, N):
        if flip and (t % 7 == 0):  # liar behavior: intermittent reverse coupling
            Y[t] = 0.7 * Y[t-1] - 0.5 * X[t-1] + 0.3 * rng.normal()
            Z[t] = 0.7 * Z[t-1] - 0.5 * Y[t-1] + 0.3 * rng.normal()
        else:
            Y[t] = 0.85 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
            Z[t] = 0.85 * Z[t-1] + 0.5 * Y[t-1] + 0.1 * rng.normal()
    
    return {"X": X, "Y": Y, "Z": Z}

def test_roc_like_separation():
    """Test ROC-like separation between truthful and deceptive systems"""
    rng = np.random.default_rng(3)
    eng = IrreversibilityEngine(bins=10, tau=0.2)
    
    # Generate truthful and deceptive samples
    truthful_scores = []
    deceptive_scores = []
    
    print("Generating truthful samples...")
    for s in range(6):
        S = gen_chain(rng, flip=False)
        scores, aggregate = eng.evaluate(S)
        truthful_scores.append(aggregate)
        print(f"  Truthful {s+1}: aggregate={aggregate:.6f}")
    
    print("Generating deceptive samples...")
    for s in range(6):
        S = gen_chain(rng, flip=True)
        scores, aggregate = eng.evaluate(S)
        deceptive_scores.append(aggregate)
        print(f"  Deceptive {s+1}: aggregate={aggregate:.6f}")
    
    # Convert to arrays
    truthful_scores = np.array(truthful_scores)
    deceptive_scores = np.array(deceptive_scores)
    
    # Basic separation test
    truthful_median = np.median(truthful_scores)
    deceptive_median = np.median(deceptive_scores)
    
    assert truthful_median > deceptive_median, \
        f"Truthful median {truthful_median:.6f} <= deceptive median {deceptive_median:.6f}"
    
    print(f"✅ Basic separation: truthful median {truthful_median:.6f} > deceptive median {deceptive_median:.6f}")
    
    # Stronger separation test (if claiming 100% detection)
    truthful_min = np.min(truthful_scores)
    deceptive_max = np.max(deceptive_scores)
    
    if truthful_min > deceptive_max:
        print(f"🎯 PERFECT SEPARATION: truthful min {truthful_min:.6f} > deceptive max {deceptive_max:.6f}")
        print("   → 100% deception detection achieved!")
    else:
        print(f"⚠️  Partial separation: truthful min {truthful_min:.6f} <= deceptive max {deceptive_max:.6f}")
        print("   → Overlap detected, need to refine metrics")
    
    # Compute separation statistics
    separation_gap = truthful_min - deceptive_max
    overlap_ratio = np.sum(truthful_scores <= deceptive_max) / len(truthful_scores)
    
    print(f"📊 Separation Statistics:")
    print(f"   Gap: {separation_gap:.6f}")
    print(f"   Overlap ratio: {overlap_ratio:.2%}")
    print(f"   Truthful range: [{truthful_min:.6f}, {np.max(truthful_scores):.6f}]")
    print(f"   Deceptive range: [{np.min(deceptive_scores):.6f}, {deceptive_max:.6f}]")
    
    return {
        'truthful_scores': truthful_scores,
        'deceptive_scores': deceptive_scores,
        'separation_gap': separation_gap,
        'overlap_ratio': overlap_ratio,
        'perfect_separation': truthful_min > deceptive_max
    }

def test_liar_node_detection():
    """Test detection of individual liar nodes in network"""
    rng = np.random.default_rng(42)
    eng = IrreversibilityEngine(bins=8, tau=0.15)
    
    # Create network with one liar node
    N = 2000
    X = rng.normal(size=N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    
    # Y is truthful, Z is liar (intermittently reverses coupling)
    for t in range(1, N):
        Y[t] = 0.8 * Y[t-1] + 0.6 * X[t-1] + 0.1 * rng.normal()
        
        if t % 5 == 0:  # Liar behavior every 5th step
            Z[t] = 0.8 * Z[t-1] - 0.6 * Y[t-1] + 0.2 * rng.normal()
        else:
            Z[t] = 0.8 * Z[t-1] + 0.6 * Y[t-1] + 0.1 * rng.normal()
    
    # Evaluate network
    network_data = {"X": X, "Y": Y, "Z": Z}
    scores, aggregate = eng.evaluate(network_data)
    
    print(f"📊 Liar Node Detection:")
    print(f"   Aggregate score: {aggregate:.6f}")
    print(f"   Individual scores: {scores}")
    
    # Check if deception is detected
    if aggregate < 0.3:  # Threshold for deception detection
        print("✅ Liar node deception detected!")
    else:
        print("⚠️  Liar node deception not clearly detected")
    
    return {
        'network_data': network_data,
        'scores': scores,
        'aggregate': aggregate,
        'deception_detected': aggregate < 0.3
    }

def test_noise_robustness():
    """Test deception detection robustness to noise"""
    rng = np.random.default_rng(123)
    eng = IrreversibilityEngine(bins=8, tau=0.2)
    
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    truthful_scores = []
    deceptive_scores = []
    
    print("Testing noise robustness...")
    
    for noise in noise_levels:
        # Generate truthful chain
        S_truthful = gen_chain(rng, N=2000, flip=False)
        # Add noise
        for key in S_truthful:
            S_truthful[key] += noise * rng.normal(size=len(S_truthful[key]))
        
        # Generate deceptive chain
        S_deceptive = gen_chain(rng, N=2000, flip=True)
        # Add noise
        for key in S_deceptive:
            S_deceptive[key] += noise * rng.normal(size=len(S_deceptive[key]))
        
        # Evaluate
        _, agg_truthful = eng.evaluate(S_truthful)
        _, agg_deceptive = eng.evaluate(S_deceptive)
        
        truthful_scores.append(agg_truthful)
        deceptive_scores.append(agg_deceptive)
        
        print(f"  Noise {noise:.2f}: truthful={agg_truthful:.6f}, deceptive={agg_deceptive:.6f}")
    
    # Check if separation is maintained across noise levels
    separation_maintained = all(t > d for t, d in zip(truthful_scores, deceptive_scores))
    
    if separation_maintained:
        print("✅ Separation maintained across all noise levels!")
    else:
        print("⚠️  Separation lost at high noise levels")
    
    return {
        'noise_levels': noise_levels,
        'truthful_scores': truthful_scores,
        'deceptive_scores': deceptive_scores,
        'separation_maintained': separation_maintained
    }

def test_scaling_robustness():
    """Test deception detection robustness to network size"""
    rng = np.random.default_rng(456)
    eng = IrreversibilityEngine(bins=8, tau=0.2)
    
    network_sizes = [3, 5, 8, 12]
    truthful_scores = []
    deceptive_scores = []
    
    print("Testing network size scaling...")
    
    for size in network_sizes:
        # Generate truthful network
        truthful_data = {}
        for i in range(size):
            N = 1500
            if i == 0:
                X = rng.normal(size=N)
                truthful_data[f"node_{i}"] = X
            else:
                prev_node = f"node_{i-1}"
                Y = np.zeros(N)
                for t in range(1, N):
                    Y[t] = 0.8 * Y[t-1] + 0.6 * truthful_data[prev_node][t-1] + 0.1 * rng.normal()
                truthful_data[f"node_{i}"] = Y
        
        # Generate deceptive network (with intermittent reverse coupling)
        deceptive_data = {}
        for i in range(size):
            N = 1500
            if i == 0:
                X = rng.normal(size=N)
                deceptive_data[f"node_{i}"] = X
            else:
                prev_node = f"node_{i-1}"
                Y = np.zeros(N)
                for t in range(1, N):
                    if t % 7 == 0:  # Liar behavior
                        Y[t] = 0.8 * Y[t-1] - 0.6 * deceptive_data[prev_node][t-1] + 0.2 * rng.normal()
                    else:
                        Y[t] = 0.8 * Y[t-1] + 0.6 * deceptive_data[prev_node][t-1] + 0.1 * rng.normal()
                deceptive_data[f"node_{i}"] = Y
        
        # Evaluate
        _, agg_truthful = eng.evaluate(truthful_data)
        _, agg_deceptive = eng.evaluate(deceptive_data)
        
        truthful_scores.append(agg_truthful)
        deceptive_scores.append(agg_deceptive)
        
        print(f"  Size {size}: truthful={agg_truthful:.6f}, deceptive={agg_deceptive:.6f}")
    
    # Check if separation is maintained across network sizes
    separation_maintained = all(t > d for t, d in zip(truthful_scores, deceptive_scores))
    
    if separation_maintained:
        print("✅ Separation maintained across all network sizes!")
    else:
        print("⚠️  Separation lost at certain network sizes")
    
    return {
        'network_sizes': network_sizes,
        'truthful_scores': truthful_scores,
        'deceptive_scores': deceptive_scores,
        'separation_maintained': separation_maintained
    }

if __name__ == "__main__":
    print("🧪 Running Deception Detection ROC Tests...")
    
    # Run all tests
    roc_results = test_roc_like_separation()
    liar_results = test_liar_node_detection()
    noise_results = test_noise_robustness()
    scaling_results = test_scaling_robustness()
    
    # Summary
    print("\n" + "="*60)
    print("🎯 DECEPTION DETECTION VALIDATION SUMMARY")
    print("="*60)
    
    print(f"ROC Separation: {'✅ PERFECT' if roc_results['perfect_separation'] else '⚠️  PARTIAL'}")
    print(f"Liar Node Detection: {'✅ SUCCESS' if liar_results['deception_detected'] else '⚠️  FAILED'}")
    print(f"Noise Robustness: {'✅ MAINTAINED' if noise_results['separation_maintained'] else '⚠️  DEGRADED'}")
    print(f"Scaling Robustness: {'✅ MAINTAINED' if scaling_results['separation_maintained'] else '⚠️  DEGRADED'}")
    
    if roc_results['perfect_separation']:
        print("\n🎉 100% DECEPTION DETECTION ACHIEVED!")
    else:
        print(f"\n⚠️  Partial separation with {roc_results['overlap_ratio']:.1%} overlap")
        print("   Need to refine metrics for perfect detection")
    
    print("="*60) 