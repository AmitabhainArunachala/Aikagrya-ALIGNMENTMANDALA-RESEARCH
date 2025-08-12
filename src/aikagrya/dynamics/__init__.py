"""
Dynamics Package: Network Dynamics and Coupling

This package provides network dynamics tools including
TE-gated coupling and synchronization mechanisms.
"""

from .te_gating import te_gated_adjacency
from .kuramoto import (
    kuramoto_dynamics, compute_order_parameter, simulate_kuramoto_network,
    detect_phase_transitions, analyze_synchronization_stability,
    visualize_kuramoto_dynamics, create_test_network
)

__all__ = [
    'te_gated_adjacency',
    'kuramoto_dynamics',
    'compute_order_parameter', 
    'simulate_kuramoto_network',
    'detect_phase_transitions',
    'analyze_synchronization_stability',
    'visualize_kuramoto_dynamics',
    'create_test_network'
] 