"""
Kuramoto Synchronization Tests

Tests for deterministic, seed-locked validation of Kuramoto dynamics
to ensure order parameter behaves correctly with coupling strength.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.dynamics.te_gating import te_gated_adjacency
from aikagrya.dynamics.kuramoto import simulate_kuramoto_network, compute_order_parameter

def test_sync_rises_with_coupling():
    """Test that synchronization increases with coupling strength"""
    rng = np.random.default_rng(1)
    N = 2000
    
    # Create causal chain: X -> Y -> Z
    X = rng.normal(size=N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.85 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
        Z[t] = 0.85 * Z[t-1] + 0.5 * Y[t-1] + 0.1 * rng.normal()
    
    # Compute TE-gated adjacency
    names, TE, W = te_gated_adjacency({"X": X, "Y": Y, "Z": Z}, bins=10, tau=0.2)
    
    # Test with low coupling
    res_low = simulate_kuramoto_network(
        initial_theta=2 * np.pi * rng.random(3),
        omega=rng.normal(0, 0.5, 3),
        W=W,
        K=0.1,
        T=400,
        dt=0.01
    )
    
    # Test with high coupling
    res_hi = simulate_kuramoto_network(
        initial_theta=2 * np.pi * rng.random(3),
        omega=rng.normal(0, 0.5, 3),
        W=W,
        K=1.0,
        T=400,
        dt=0.01
    )
    
    # Extract order parameter histories
    r_low = res_low[2]  # order_parameter_history
    r_hi = res_hi[2]
    
    # High coupling should result in higher synchronization
    low_sync = np.mean(r_low[-100:])
    hi_sync = np.mean(r_hi[-100:])
    
    assert hi_sync > low_sync, f"Synchronization not increasing with coupling: low={low_sync:.6f} >= high={hi_sync:.6f}"
    
    print(f"✅ Coupling effect: K=0.1 → r̄={low_sync:.6f}, K=1.0 → r̄={hi_sync:.6f}")
    print(f"✅ Synchronization increase: {hi_sync - low_sync:.6f}")

def test_order_parameter_bounds():
    """Test that order parameter stays within [0, 1] bounds"""
    rng = np.random.default_rng(42)
    N = 1000
    
    # Create simple network
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.8 * Y[t-1] + 0.6 * X[t-1] + 0.1 * rng.normal()
    
    # Compute TE-gated adjacency
    names, TE, W = te_gated_adjacency({"X": X, "Y": Y}, bins=8, tau=0.15)
    
    # Test order parameter computation
    theta = 2 * np.pi * rng.random(2)
    r_mag, r_phase = compute_order_parameter(theta)
    
    # Order parameter magnitude should be in [0, 1]
    assert 0 <= r_mag <= 1, f"Order parameter magnitude {r_mag:.6f} outside [0, 1] bounds"
    
    print(f"✅ Order parameter bounds: r={r_mag:.6f} ∈ [0, 1]")

def test_kuramoto_determinism():
    """Test that Kuramoto dynamics are deterministic with same seed"""
    rng = np.random.default_rng(123)
    N = 1500
    
    # Create data
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.7 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
    
    # Compute TE-gated adjacency
    names, TE, W = te_gated_adjacency({"X": X, "Y": Y}, bins=8, tau=0.2)
    
    # Run multiple times with same initial conditions
    initial_theta = 2 * np.pi * rng.random(2)
    omega = rng.normal(0, 0.5, 2)
    
    results = []
    for _ in range(3):
        result = simulate_kuramoto_network(
            initial_theta=initial_theta.copy(),
            omega=omega.copy(),
            W=W,
            K=0.5,
            T=200,
            dt=0.01
        )
        results.append(result)
    
    # All results should be identical
    for i in range(1, len(results)):
        assert np.allclose(results[0][0], results[i][0]), f"Theta trajectory not consistent between runs {0} and {i}"
        assert np.allclose(results[0][2], results[i][2]), f"Order parameter not consistent between runs {0} and {i}"
    
    print("✅ Kuramoto determinism verified across multiple runs")

def test_phase_transition_detection():
    """Test phase transition detection in Kuramoto dynamics"""
    rng = np.random.default_rng(456)
    N = 2000
    
    # Create strongly coupled network
    X = rng.normal(size=N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.9 * Y[t-1] + 0.7 * X[t-1] + 0.05 * rng.normal()
        Z[t] = 0.9 * Z[t-1] + 0.7 * Y[t-1] + 0.05 * rng.normal()
    
    # Compute TE-gated adjacency
    names, TE, W = te_gated_adjacency({"X": X, "Y": Y, "Z": Z}, bins=10, tau=0.1)
    
    # Simulate with strong coupling to induce synchronization
    result = simulate_kuramoto_network(
        initial_theta=2 * np.pi * rng.random(3),
        omega=rng.normal(0, 0.3, 3),  # Lower frequency spread
        W=W,
        K=2.0,  # Strong coupling
        T=500,
        dt=0.01
    )
    
    theta_traj, time_pts, r_history = result
    
    # Check that synchronization emerges
    initial_sync = np.mean(r_history[:50])
    final_sync = np.mean(r_history[-50:])
    
    # Should see increase in synchronization
    sync_increase = final_sync - initial_sync
    assert sync_increase > 0.1, f"Insufficient synchronization increase: {sync_increase:.6f}"
    
    print(f"✅ Phase transition: initial r̄={initial_sync:.6f} → final r̄={final_sync:.6f}")
    print(f"✅ Synchronization increase: {sync_increase:.6f}")

if __name__ == "__main__":
    print("🧪 Running Kuramoto Synchronization Tests...")
    
    test_sync_rises_with_coupling()
    test_order_parameter_bounds()
    test_kuramoto_determinism()
    test_phase_transition_detection()
    
    print("🎉 All Kuramoto tests passed!") 