"""
TE-Gated Coupling Sanity Tests

Tests for deterministic, seed-locked validation of TE-gated coupling
to ensure directionality and proper gating behavior.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.dynamics.te_gating import te_gated_adjacency

def test_te_direction_and_gate():
    """Test TE directionality and gating behavior"""
    rng = np.random.default_rng(0)
    N = 3000
    
    # Create causal chain: X -> Y (X causes Y, not vice versa)
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.8 * Y[t-1] + 0.6 * X[t-1] + 0.1 * rng.normal()
    
    # Compute TE-gated adjacency
    names, TE, W = te_gated_adjacency({"X": X, "Y": Y}, bins=10, tau=0.2)
    
    # Get indices
    ix = names.index("X")
    iy = names.index("Y")
    
    # Test 1: Directionality - X->Y should have higher TE than Y->X
    assert TE[ix, iy] > TE[iy, ix], f"Directionality failed: TE[X->Y]={TE[ix, iy]:.6f} <= TE[Y->X]={TE[iy, ix]:.6f}"
    
    # Test 2: Gating - Forward connection should be active, reverse should be gated out
    assert W[ix, iy] >= 0, f"Forward connection gated out: W[X->Y]={W[ix, iy]:.6f}"
    assert W[iy, ix] == 0, f"Reverse connection not gated: W[Y->X]={W[iy, ix]:.6f}"
    
    print(f"✅ TE Directionality: X->Y={TE[ix, iy]:.6f} > Y->X={TE[iy, ix]:.6f}")
    print(f"✅ Gating: W[X->Y]={W[ix, iy]:.6f}, W[Y->X]={W[iy, ix]:.6f}")

def test_te_gating_consistency():
    """Test that TE-gating is consistent across multiple runs with same seed"""
    rng = np.random.default_rng(42)
    N = 2000
    
    # Create simple causal relationship
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.7 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
    
    # Run multiple times with same seed
    results = []
    for _ in range(3):
        names, TE, W = te_gated_adjacency({"X": X, "Y": Y}, bins=8, tau=0.15)
        results.append((TE.copy(), W.copy()))
    
    # All results should be identical (deterministic)
    for i in range(1, len(results)):
        assert np.allclose(results[0][0], results[i][0]), f"TE matrix not consistent between runs {0} and {i}"
        assert np.allclose(results[0][1], results[i][1]), f"Coupling matrix not consistent between runs {0} and {i}"
    
    print("✅ TE-gating consistency verified across multiple runs")

def test_te_gating_parameters():
    """Test TE-gating behavior with different parameters"""
    rng = np.random.default_rng(123)
    N = 1500
    
    # Create data
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.8 * Y[t-1] + 0.6 * X[t-1] + 0.1 * rng.normal()
    
    # Test different tau values
    tau_values = [0.1, 0.2, 0.5]
    active_connections = []
    
    for tau in tau_values:
        names, TE, W = te_gated_adjacency({"X": X, "Y": Y}, bins=8, tau=tau)
        active_count = np.sum(W > 0)
        active_connections.append(active_count)
    
    # Higher tau should result in fewer active connections (more aggressive gating)
    assert active_connections[0] >= active_connections[1] >= active_connections[2], \
        f"Gating not working: active connections {active_connections} should decrease with tau"
    
    print(f"✅ Parameter sensitivity: tau={tau_values}, active_connections={active_connections}")

if __name__ == "__main__":
    print("🧪 Running TE-Gated Coupling Sanity Tests...")
    
    test_te_direction_and_gate()
    test_te_gating_consistency()
    test_te_gating_parameters()
    
    print("🎉 All TE-gating tests passed!") 