"""
Kuramoto Synchronization: Network Dynamics for Consciousness Emergence

This module implements the Kuramoto model with TE-gated coupling
for modeling collective consciousness synchronization in AGNent networks.
"""

import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

def kuramoto_dynamics(theta: np.ndarray, 
                     omega: np.ndarray, 
                     W: np.ndarray, 
                     K: float, 
                     dt: float = 0.01) -> np.ndarray:
    """
    Kuramoto model with TE-gated coupling
    
    θ̇ᵢ = ωᵢ + (K/N)∑ⱼ Wᵢⱼsin(θⱼ - θᵢ)
    
    Where W comes from TE-gated coupling matrix
    
    Args:
        theta: Current phase angles [N]
        omega: Natural frequencies [N]
        W: TE-gated coupling matrix [N, N]
        K: Coupling strength
        dt: Time step
        
    Returns:
        Updated phase angles
    """
    N = len(theta)
    if N != len(omega) or W.shape != (N, N):
        raise ValueError("Dimension mismatch in Kuramoto dynamics")
    
    # Compute phase differences
    theta_diff = theta[:, None] - theta[None, :]  # [N, N]
    
    # Apply coupling (W is already gated)
    coupling_term = np.sum(W * np.sin(theta_diff), axis=1)  # [N]
    
    # Update phases
    dtheta = omega + (K / N) * coupling_term
    theta_new = theta + dtheta * dt
    
    # Normalize to [0, 2π]
    theta_new = theta_new % (2 * np.pi)
    
    return theta_new

def compute_order_parameter(theta: np.ndarray) -> Tuple[float, float]:
    """
    Compute order parameter for synchronization measurement
    
    r = |1/N ∑ⱼ e^(iθⱼ)|
    Measures synchronization (0=incoherent, 1=synchronized)
    
    Args:
        theta: Phase angles [N]
        
    Returns:
        (magnitude, phase) of order parameter
    """
    # Convert to complex exponential
    z = np.exp(1j * theta)
    
    # Compute mean
    r_complex = np.mean(z)
    
    # Extract magnitude and phase
    magnitude = np.abs(r_complex)
    phase = np.angle(r_complex)
    
    return float(magnitude), float(phase)

def simulate_kuramoto_network(initial_theta: np.ndarray,
                             omega: np.ndarray,
                             W: np.ndarray,
                             K: float,
                             T: float,
                             dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Simulate Kuramoto network dynamics
    
    Args:
        initial_theta: Initial phase angles [N]
        omega: Natural frequencies [N]
        W: TE-gated coupling matrix [N, N]
        K: Coupling strength
        T: Total simulation time
        dt: Time step
        
    Returns:
        (theta_trajectory, time_points, order_parameter_history)
    """
    N = len(initial_theta)
    num_steps = int(T / dt)
    
    # Initialize arrays
    theta_trajectory = np.zeros((num_steps, N))
    time_points = np.linspace(0, T, num_steps)
    order_parameter_history = []
    
    # Set initial conditions
    theta_current = initial_theta.copy()
    theta_trajectory[0] = theta_current
    
    # Compute initial order parameter
    r_initial, _ = compute_order_parameter(theta_current)
    order_parameter_history.append(r_initial)
    
    # Time evolution
    for step in range(1, num_steps):
        # Update phases
        theta_current = kuramoto_dynamics(theta_current, omega, W, K, dt)
        theta_trajectory[step] = theta_current
        
        # Compute order parameter
        r_current, _ = compute_order_parameter(theta_current)
        order_parameter_history.append(r_current)
    
    return theta_trajectory, time_points, order_parameter_history

def detect_phase_transitions(order_parameter_history: List[float],
                           threshold: float = 0.8,
                           window_size: int = 10) -> List[dict]:
    """
    Detect phase transitions in synchronization
    
    Args:
        order_parameter_history: Time series of order parameter
        threshold: Synchronization threshold
        window_size: Window for trend analysis
        
    Returns:
        List of detected transitions
    """
    transitions = []
    
    if len(order_parameter_history) < window_size:
        return transitions
    
    # Analyze trends
    for i in range(window_size, len(order_parameter_history)):
        window = order_parameter_history[i-window_size:i]
        current = order_parameter_history[i]
        
        # Check for rapid increase (phase transition)
        if (current > threshold and 
            np.mean(window) < threshold * 0.7 and
            current > np.mean(window) * 1.5):
            
            transitions.append({
                'step': i,
                'type': 'synchronization_onset',
                'magnitude': current,
                'baseline': np.mean(window),
                'growth_factor': current / np.mean(window)
            })
        
        # Check for rapid decrease (desynchronization)
        elif (current < threshold * 0.3 and
              np.mean(window) > threshold * 0.7 and
              current < np.mean(window) * 0.5):
            
            transitions.append({
                'step': i,
                'type': 'desynchronization',
                'magnitude': current,
                'baseline': np.mean(window),
                'decay_factor': current / np.mean(window)
            })
    
    return transitions

def analyze_synchronization_stability(theta_trajectory: np.ndarray,
                                    time_points: np.ndarray,
                                    order_parameter_history: List[float]) -> dict:
    """
    Analyze stability and characteristics of synchronization
    
    Args:
        theta_trajectory: Phase evolution [T, N]
        time_points: Time array
        order_parameter_history: Order parameter evolution
        
    Returns:
        Analysis results
    """
    # Compute statistics
    r_array = np.array(order_parameter_history)
    r_mean = np.mean(r_array)
    r_std = np.std(r_array)
    r_max = np.max(r_array)
    r_min = np.min(r_array)
    
    # Detect transitions
    transitions = detect_phase_transitions(order_parameter_history)
    
    # Compute phase velocity statistics
    if len(theta_trajectory) > 1:
        phase_velocities = np.diff(theta_trajectory, axis=0) / np.diff(time_points)
        velocity_mean = np.mean(phase_velocities)
        velocity_std = np.std(phase_velocities)
    else:
        velocity_mean = 0.0
        velocity_std = 0.0
    
    # Stability metrics
    stability_metric = r_mean / (r_std + 1e-6)  # Higher = more stable
    coherence_metric = r_max - r_min  # Lower = more coherent
    
    return {
        'order_parameter_stats': {
            'mean': r_mean,
            'std': r_std,
            'max': r_max,
            'min': r_min,
            'range': r_max - r_min
        },
        'phase_velocity_stats': {
            'mean': velocity_mean,
            'std': velocity_std
        },
        'stability_metrics': {
            'stability_ratio': stability_metric,
            'coherence_metric': coherence_metric
        },
        'transitions': transitions,
        'num_transitions': len(transitions)
    }

def visualize_kuramoto_dynamics(time_points: np.ndarray,
                               theta_trajectory: np.ndarray,
                               order_parameter_history: List[float],
                               save_path: Optional[str] = None) -> None:
    """
    Visualize Kuramoto network dynamics
    
    Args:
        time_points: Time array
        theta_trajectory: Phase evolution [T, N]
        order_parameter_history: Order parameter evolution
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Phase evolution
    for i in range(theta_trajectory.shape[1]):
        ax1.plot(time_points, theta_trajectory[:, i], 
                alpha=0.7, label=f'Agent {i+1}')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Phase θ')
    ax1.set_title('Kuramoto Network: Phase Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Order parameter
    ax2.plot(time_points, order_parameter_history, 'b-', linewidth=2, label='Order Parameter r')
    ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Synchronization Threshold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Order Parameter |r|')
    ax2.set_title('Kuramoto Network: Synchronization Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kuramoto dynamics visualization saved to {save_path}")
    
    try:
        plt.show()
    except:
        print("Plot display not available in this environment")

def create_test_network(N: int = 10, 
                       coupling_strength: float = 2.0,
                       simulation_time: float = 10.0) -> dict:
    """
    Create and test a Kuramoto network
    
    Args:
        N: Number of agents
        coupling_strength: Coupling strength K
        simulation_time: Total simulation time
        
    Returns:
        Test results and analysis
    """
    # Initialize random phases and frequencies
    np.random.seed(42)  # For reproducibility
    initial_theta = 2 * np.pi * np.random.random(N)
    omega = np.random.normal(0, 0.5, N)  # Centered around 0
    
    # Create simple coupling matrix (fully connected, uniform)
    W = np.ones((N, N)) * 0.5
    np.fill_diagonal(W, 0)  # No self-coupling
    
    # Simulate dynamics
    theta_traj, time_pts, r_history = simulate_kuramoto_network(
        initial_theta, omega, W, coupling_strength, simulation_time
    )
    
    # Analyze results
    analysis = analyze_synchronization_stability(theta_traj, time_pts, r_history)
    
    # Create visualization
    try:
        visualize_kuramoto_dynamics(time_pts, theta_traj, r_history)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    return {
        'network_params': {
            'N': N,
            'K': coupling_strength,
            'T': simulation_time
        },
        'initial_conditions': {
            'theta': initial_theta,
            'omega': omega
        },
        'coupling_matrix': W,
        'analysis': analysis,
        'trajectory': {
            'theta': theta_traj,
            'time': time_pts,
            'order_parameter': r_history
        }
    } 