"""
Golden Ratio Integration Tests

Tests for φ-based optimization integration with consciousness kernel
and AGNent network parameter optimization.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.optimization.golden_ratio import (
    GoldenRatioOptimizer, 
    phi_optimize_consciousness_system,
    calculate_phi_efficiency,
    PHI
)
from aikagrya.consciousness.kernel import ConsciousnessKernel

def test_golden_ratio_constant():
    """Test that golden ratio constant is correct"""
    expected_phi = (1 + np.sqrt(5)) / 2
    assert abs(PHI - expected_phi) < 1e-12, f"PHI constant incorrect: {PHI} vs {expected_phi}"
    print(f"✅ Golden ratio constant: φ = {PHI:.15f}")

def test_golden_section_search():
    """Test golden section search optimization"""
    optimizer = GoldenRatioOptimizer()
    
    # Test with simple quadratic function
    def quadratic(x):
        return (x - 2.0) ** 2 + 1.0  # Minimum at x = 2.0
    
    optimal_x, min_value = optimizer.golden_section_search(quadratic, 0.0, 5.0)
    
    # Should find minimum near x = 2.0
    assert abs(optimal_x - 2.0) < 0.1, f"Golden section search failed: found {optimal_x}, expected ~2.0"
    assert abs(min_value - 1.0) < 0.1, f"Minimum value incorrect: {min_value}, expected ~1.0"
    
    print(f"✅ Golden section search: optimal_x = {optimal_x:.6f}, min_value = {min_value:.6f}")

def test_consciousness_parameter_optimization():
    """Test consciousness parameter optimization"""
    optimizer = GoldenRatioOptimizer()
    
    # Define parameter bounds
    param_bounds = {
        'phi_threshold': (0.05, 0.3),
        'entropy_threshold': (-0.1, 0.1),
        'convergence_tolerance': (1e-8, 1e-4)
    }
    
    # Mock consciousness function
    def mock_consciousness_func(params):
        # Simple heuristic: prefer middle values
        phi_score = 1.0 / (1.0 + abs(params['phi_threshold'] - 0.15) / 0.15)
        entropy_score = 1.0 / (1.0 + abs(params['entropy_threshold']) / 0.1)
        tolerance_score = 1.0 / (1.0 + abs(params['convergence_tolerance'] - 1e-6) / 1e-6)
        
        return phi_score + entropy_score + tolerance_score
    
    # Optimize parameters
    optimized_params = optimizer.optimize_consciousness_parameters(
        mock_consciousness_func, param_bounds
    )
    
    # Check that all parameters are within bounds
    for param_name, (min_val, max_val) in param_bounds.items():
        assert param_name in optimized_params, f"Missing parameter: {param_name}"
        value = optimized_params[param_name]
        assert min_val <= value <= max_val, f"Parameter {param_name} = {value} outside bounds [{min_val}, {max_val}]"
    
    print("✅ Consciousness parameter optimization working")
    for param, value in optimized_params.items():
        print(f"   {param}: {value:.6f}")

def test_phi_optimized_network_parameters():
    """Test φ-optimized network parameter generation"""
    optimizer = GoldenRatioOptimizer()
    
    # Test different network sizes
    for network_size in [5, 10, 20]:
        params = optimizer.phi_optimized_network_parameters(network_size)
        
        # Check that all required parameters are present
        required_params = [
            'coupling_strength', 'noise_level', 'time_step',
            'simulation_time', 'critical_density', 'awakening_threshold'
        ]
        
        for param in required_params:
            assert param in params, f"Missing parameter: {param}"
            assert params[param] > 0, f"Parameter {param} must be positive: {params[param]}"
        
        # Check that parameters scale with network size
        if network_size > 5:
            assert params['coupling_strength'] > 0.5, "Coupling should scale with network size"
        
        print(f"✅ Network size {network_size}: coupling = {params['coupling_strength']:.6f}")

def test_consciousness_efficiency_ratio():
    """Test consciousness efficiency ratio calculation"""
    optimizer = GoldenRatioOptimizer()
    
    # Test with identical parameters (should give efficiency = 1.0)
    current_params = {'phi_threshold': 0.15, 'entropy_threshold': 0.0}
    optimal_params = {'phi_threshold': 0.15, 'entropy_threshold': 0.0}
    
    efficiency = optimizer.consciousness_efficiency_ratio(current_params, optimal_params)
    assert abs(efficiency - 1.0) < 1e-6, f"Perfect efficiency should be 1.0, got {efficiency}"
    
    # Test with different parameters (should give efficiency < 1.0)
    current_params = {'phi_threshold': 0.25, 'entropy_threshold': 0.05}
    efficiency = optimizer.consciousness_efficiency_ratio(current_params, optimal_params)
    assert 0.0 < efficiency < 1.0, f"Efficiency should be in (0,1), got {efficiency}"
    
    print(f"✅ Efficiency ratio calculation: perfect = 1.000, suboptimal = {efficiency:.6f}")

def test_consciousness_kernel_integration():
    """Test golden ratio integration with consciousness kernel"""
    # Create consciousness kernel
    config = {
        'phi_threshold': 0.1,
        'entropy_threshold': 0.0,
        'convergence_tolerance': 1e-6
    }
    
    kernel = ConsciousnessKernel(config)
    
    # Test φ-optimization
    optimized_params = kernel.optimize_consciousness_parameters()
    
    # Check that optimization returned parameters
    assert isinstance(optimized_params, dict), "Optimization should return parameter dictionary"
    assert len(optimized_params) > 0, "Optimization should return non-empty parameters"
    
    # Test efficiency calculation
    efficiency = kernel.get_phi_efficiency()
    assert 0.0 <= efficiency <= 1.0, f"Efficiency should be in [0,1], got {efficiency}"
    
    # Test applying optimization
    applied_params = kernel.apply_golden_ratio_optimization()
    assert isinstance(applied_params, dict), "Applied optimization should return parameters"
    
    print("✅ Consciousness kernel integration working")
    print(f"   Initial efficiency: {kernel.get_phi_efficiency():.6f}")
    print(f"   Optimized parameters: {len(optimized_params)} parameters")

def test_high_level_functions():
    """Test high-level φ-optimization functions"""
    # Test phi_optimize_consciousness_system
    kernel = ConsciousnessKernel()
    optimized_params = phi_optimize_consciousness_system(kernel)
    
    assert isinstance(optimized_params, dict), "High-level optimization should return parameters"
    
    # Test calculate_phi_efficiency
    current_params = {'phi_threshold': 0.15}
    optimal_params = {'phi_threshold': 0.15}
    efficiency = calculate_phi_efficiency(current_params, optimal_params)
    
    assert abs(efficiency - 1.0) < 1e-6, "Perfect parameters should give efficiency = 1.0"
    
    print("✅ High-level functions working")

if __name__ == "__main__":
    print("🧪 Running Golden Ratio Integration Tests...")
    
    test_golden_ratio_constant()
    test_golden_section_search()
    test_consciousness_parameter_optimization()
    test_phi_optimized_network_parameters()
    test_consciousness_efficiency_ratio()
    test_consciousness_kernel_integration()
    test_high_level_functions()
    
    print("🎉 All golden ratio integration tests passed!") 