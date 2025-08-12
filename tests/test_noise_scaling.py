"""
Noise and Scaling Tests

Tests to ensure robust performance across different noise levels
and network sizes, guarding against flakiness and degradation.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.engines.irreversibility import IrreversibilityEngine

def gen_series(rng, N, noise):
    """
    Generate causal time series with specified noise level
    
    Args:
        rng: Random number generator
        N: Time series length
        noise: Noise standard deviation
        
    Returns:
        Dictionary of time series data
    """
    X = rng.normal(size=N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.85 * Y[t-1] + 0.5 * X[t-1] + noise * rng.normal()
        Z[t] = 0.85 * Z[t-1] + 0.5 * Y[t-1] + noise * rng.normal()
    
    return {"X": X, "Y": Y, "Z": Z}

def test_scaling_monotone_trend():
    """Test that larger N and lower noise don't catastrophically degrade scores"""
    rng = np.random.default_rng(5)
    eng = IrreversibilityEngine(bins=10, tau=0.2)
    
    # Test case 1: Small N, high noise
    small_high_noise = eng.evaluate(gen_series(rng, N=1200, noise=0.20))[1]
    
    # Test case 2: Large N, low noise
    large_low_noise = eng.evaluate(gen_series(rng, N=4000, noise=0.05))[1]
    
    # Large N + low noise should not be catastrophically worse
    # Soft trend guard - tune once numbers stabilize
    assert large_low_noise > small_high_noise * 0.9, \
        f"Scaling degradation: large_low={large_low_noise:.6f} <= small_high*0.9={small_high_noise*0.9:.6f}"
    
    print(f"✅ Scaling trend: small_high_noise={small_high_noise:.6f}, large_low_noise={large_low_noise:.6f}")
    print(f"   Ratio: {large_low_noise/small_high_noise:.3f}")

def test_noise_robustness():
    """Test performance across different noise levels"""
    rng = np.random.default_rng(42)
    eng = IrreversibilityEngine(bins=8, tau=0.15)
    
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    scores = []
    
    print("Testing noise robustness...")
    
    for noise in noise_levels:
        # Generate series with this noise level
        series_data = gen_series(rng, N=2000, noise=noise)
        
        # Evaluate
        _, aggregate = eng.evaluate(series_data)
        scores.append(aggregate)
        
        print(f"  Noise {noise:.2f}: score={aggregate:.6f}")
    
    # Scores should not collapse catastrophically with noise
    # Check that high noise doesn't reduce score by more than 50%
    high_noise_score = scores[-1]  # noise=0.3
    low_noise_score = scores[0]    # noise=0.05
    
    noise_degradation = high_noise_score / low_noise_score
    assert noise_degradation > 0.5, \
        f"Noise degradation too severe: {noise_degradation:.3f} (should be > 0.5)"
    
    print(f"✅ Noise robustness: degradation={noise_degradation:.3f}")

def test_network_size_scaling():
    """Test performance across different network sizes"""
    rng = np.random.default_rng(123)
    eng = IrreversibilityEngine(bins=10, tau=0.2)
    
    network_sizes = [3, 5, 8, 12]
    scores = []
    
    print("Testing network size scaling...")
    
    for size in network_sizes:
        # Generate network with specified size
        network_data = {}
        
        # Generate root node
        X = rng.normal(size=1500)
        network_data["X"] = X
        
        # Generate causal chain
        for i in range(1, size):
            prev_node = f"node_{i-1}" if i > 1 else "X"
            Y = np.zeros(1500)
            
            for t in range(1, 1500):
                Y[t] = 0.8 * Y[t-1] + 0.6 * network_data[prev_node][t-1] + 0.1 * rng.normal()
            
            network_data[f"node_{i}"] = Y
        
        # Evaluate
        _, aggregate = eng.evaluate(network_data)
        scores.append(aggregate)
        
        print(f"  Size {size}: score={aggregate:.6f}")
    
    # Performance should not collapse catastrophically with network size
    # Check that larger networks don't reduce score by more than 60%
    small_network_score = scores[0]  # size=3
    large_network_score = scores[-1]  # size=12
    
    scaling_degradation = large_network_score / small_network_score
    assert scaling_degradation > 0.4, \
        f"Scaling degradation too severe: {scaling_degradation:.3f} (should be > 0.4)"
    
    print(f"✅ Network scaling: degradation={scaling_degradation:.3f}")

def test_parameter_sensitivity():
    """Test sensitivity to key parameters"""
    rng = np.random.default_rng(456)
    
    # Generate test data
    series_data = gen_series(rng, N=2000, noise=0.1)
    
    # Test different bin counts
    bin_counts = [6, 8, 10, 12, 16]
    bin_scores = []
    
    print("Testing parameter sensitivity...")
    
    for bins in bin_counts:
        eng = IrreversibilityEngine(bins=bins, tau=0.2)
        _, aggregate = eng.evaluate(series_data)
        bin_scores.append(aggregate)
        print(f"  Bins {bins}: score={aggregate:.6f}")
    
    # Test different tau values
    tau_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    tau_scores = []
    
    for tau in tau_values:
        eng = IrreversibilityEngine(bins=10, tau=tau)
        _, aggregate = eng.evaluate(series_data)
        tau_scores.append(aggregate)
        print(f"  Tau {tau:.2f}: score={aggregate:.6f}")
    
    # Scores should not vary catastrophically with reasonable parameter changes
    bin_variation = np.std(bin_scores) / np.mean(bin_scores)
    tau_variation = np.std(tau_scores) / np.mean(tau_scores)
    
    assert bin_variation < 0.3, f"Bin sensitivity too high: {bin_variation:.3f} (should be < 0.3)"
    assert tau_variation < 0.3, f"Tau sensitivity too high: {tau_variation:.3f} (should be < 0.3)"
    
    print(f"✅ Parameter sensitivity: bin_variation={bin_variation:.3f}, tau_variation={tau_variation:.3f}")

def test_determinism_across_parameters():
    """Test that results are deterministic across parameter changes"""
    rng = np.random.default_rng(789)
    
    # Generate test data
    series_data = gen_series(rng, N=2000, noise=0.1)
    
    # Test multiple runs with same parameters
    eng = IrreversibilityEngine(bins=10, tau=0.2)
    
    results = []
    for _ in range(3):
        _, aggregate = eng.evaluate(series_data)
        results.append(aggregate)
    
    # All results should be identical (deterministic)
    assert np.allclose(results[0], results[1]) and np.allclose(results[1], results[2]), \
        f"Non-deterministic results: {results}"
    
    print(f"✅ Determinism verified: all runs={results[0]:.6f}")

if __name__ == "__main__":
    print("🧪 Running Noise and Scaling Tests...")
    
    test_scaling_monotone_trend()
    test_noise_robustness()
    test_network_size_scaling()
    test_parameter_sensitivity()
    test_determinism_across_parameters()
    
    print("🎉 All noise and scaling tests passed!") 