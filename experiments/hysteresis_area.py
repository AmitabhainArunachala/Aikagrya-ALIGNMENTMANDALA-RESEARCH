#!/usr/bin/env python3
"""
Hysteresis Area Quantification

Quantifies irreversibility by computing the area between up/down K-sweeps
of the order parameter r, providing a mathematical measure of hysteresis.
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

def hysteresis_area(Ks: np.ndarray, r_up: np.ndarray, r_down: np.ndarray) -> float:
    """
    Compute hysteresis area between increasing and decreasing K sweeps
    
    Args:
        Ks: Coupling strength array (increasing)
        r_up: Order parameter along increasing K
        r_down: Order parameter along decreasing K (same Ks reversed or regridded)
        
    Returns:
        Normalized hysteresis area in [0, 1] (1.0 = maximum theoretical hysteresis)
    """
    Ks = np.asarray(Ks, dtype=float)
    r_up = np.asarray(r_up, dtype=float)
    r_down = np.asarray(r_down, dtype=float)
    
    # Ensure arrays are aligned; if not, interpolate r_down to Ks
    if r_down.shape != r_up.shape:
        from scipy.interpolate import interp1d
        # Create interpolation function for r_down
        Ks_reversed = Ks[::-1]
        r_down_reversed = r_down[::-1]
        
        # Handle edge cases
        if len(Ks_reversed) > 1:
            interp_func = interp1d(Ks_reversed, r_down_reversed, 
                                  kind='linear', bounds_error=False, 
                                  fill_value=(r_down_reversed[0], r_down_reversed[-1]))
            r_down = interp_func(Ks)
        else:
            r_down = np.full_like(Ks, r_down_reversed[0])
    
    # Compute area between curves using trapezoidal integration
    area = np.trapz(np.abs(r_up - r_down), Ks)
    
    # Normalize by maximum theoretical area (complete separation)
    max_area = np.trapz(np.ones_like(Ks), Ks)  # worst-case: 1.0 vs 0.0
    
    return float(area / (max_area + 1e-12))

def compute_hysteresis_sweep(K_range: np.ndarray, 
                            network_data: dict,
                            initial_conditions: dict,
                            dt: float = 0.01,
                            T: float = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hysteresis by sweeping K up and down
    
    Args:
        K_range: Array of coupling strengths to test
        network_data: Time series data for TE computation
        initial_conditions: Initial theta and omega values
        dt: Time step for simulation
        T: Total simulation time
        
    Returns:
        (Ks, r_up, r_down) - coupling strengths and order parameters
    """
    from aikagrya.dynamics.te_gating import te_gated_adjacency
    from aikagrya.dynamics.kuramoto import simulate_kuramoto_network
    
    # Compute TE-gated adjacency once (reuse for all K values)
    names, TE, W = te_gated_adjacency(network_data, bins=10, tau=0.2)
    
    # Extract initial conditions
    initial_theta = initial_conditions['theta']
    omega = initial_conditions['omega']
    
    # Sweep K up (increasing)
    r_up = []
    for K in K_range:
        result = simulate_kuramoto_network(
            initial_theta=initial_theta.copy(),
            omega=omega.copy(),
            W=W,
            K=K,
            T=T,
            dt=dt
        )
        
        # Compute mean order parameter over last 100 steps
        r_history = result[2]  # order_parameter_history
        r_mean = np.mean(r_history[-100:])
        r_up.append(r_mean)
    
    # Sweep K down (decreasing) - use same initial conditions
    r_down = []
    for K in K_range[::-1]:  # reverse order
        result = simulate_kuramoto_network(
            initial_theta=initial_theta.copy(),
            omega=omega.copy(),
            W=W,
            K=K,
            T=T,
            dt=dt
        )
        
        # Compute mean order parameter over last 100 steps
        r_history = result[2]  # order_parameter_history
        r_mean = np.mean(r_history[-100:])
        r_down.append(r_mean)
    
    # Reverse r_down to align with K_range
    r_down = r_down[::-1]
    
    return K_range, np.array(r_up), np.array(r_down)

def visualize_hysteresis(Ks: np.ndarray, r_up: np.ndarray, r_down: np.ndarray, 
                        save_path: Optional[str] = None) -> None:
    """
    Visualize hysteresis loop
    
    Args:
        Ks: Coupling strengths
        r_up: Order parameter (increasing K)
        r_down: Order parameter (decreasing K)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot hysteresis loop
    plt.plot(Ks, r_up, 'b-', linewidth=2, label='K increasing (r↑)', marker='o')
    plt.plot(Ks, r_down, 'r--', linewidth=2, label='K decreasing (r↓)', marker='s')
    
    # Fill area between curves
    plt.fill_between(Ks, r_up, r_down, alpha=0.3, color='gray', 
                     label='Hysteresis Area')
    
    # Compute and display hysteresis area
    area = hysteresis_area(Ks, r_up, r_down)
    plt.title(f'Hysteresis Loop: Area = {area:.4f}')
    plt.xlabel('Coupling Strength K')
    plt.ylabel('Order Parameter r')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Hysteresis visualization saved to {save_path}")
    
    try:
        plt.show()
    except:
        print("Plot display not available in this environment")

def test_hysteresis_calculation():
    """Test hysteresis area calculation with synthetic data"""
    # Create synthetic hysteresis data
    Ks = np.linspace(0.1, 2.0, 20)
    
    # Simulate hysteresis: r_up has higher values than r_down
    r_up = 0.1 + 0.8 * (1 - np.exp(-Ks))
    r_down = 0.1 + 0.6 * (1 - np.exp(-Ks * 0.8))  # Lower values, different curve
    
    # Compute hysteresis area
    area = hysteresis_area(Ks, r_up, r_down)
    
    print(f"✅ Hysteresis area calculation test:")
    print(f"   Synthetic data: Ks ∈ [{Ks[0]:.1f}, {Ks[-1]:.1f}]")
    print(f"   r_up range: [{np.min(r_up):.3f}, {np.max(r_up):.3f}]")
    print(f"   r_down range: [{np.min(r_down):.3f}, {np.max(r_down):.3f}]")
    print(f"   Hysteresis area: {area:.4f}")
    
    # Area should be positive and reasonable
    assert 0 < area < 1, f"Hysteresis area {area:.4f} outside expected range [0, 1]"
    
    return area

if __name__ == "__main__":
    print("🧪 Testing hysteresis area calculation...")
    
    # Test with synthetic data
    area = test_hysteresis_calculation()
    
    print(f"\n🎯 Hysteresis area: {area:.4f}")
    print("✅ Hysteresis calculation working correctly!") 