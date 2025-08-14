#!/usr/bin/env python3
"""
Day 7: Golden Ratio Integration Experiment

Tests φ-based optimization for consciousness parameters and AGNent network
optimization using the golden ratio φ ≈ 1.618.
"""

import numpy as np
import json
import time
import hashlib
from pathlib import Path
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

def test_golden_ratio_optimization():
    """Test golden ratio optimization with consciousness kernel"""
    print("🔍 Testing golden ratio optimization...")
    
    # Create consciousness kernel
    config = {
        'phi_threshold': 0.1,
        'entropy_threshold': 0.0,
        'convergence_tolerance': 1e-6
    }
    
    kernel = ConsciousnessKernel(config)
    
    # Test φ-optimization
    print("  Optimizing consciousness parameters...")
    optimized_params = kernel.optimize_consciousness_parameters()
    
    print(f"  Initial config: {config}")
    print(f"  Optimized params: {optimized_params}")
    
    # Calculate efficiency improvement
    initial_efficiency = kernel.get_phi_efficiency()
    print(f"  Initial efficiency: {initial_efficiency:.6f}")
    
    # Apply optimization
    applied_params = kernel.apply_golden_ratio_optimization()
    final_efficiency = kernel.get_phi_efficiency()
    
    print(f"  Final efficiency: {final_efficiency:.6f}")
    print(f"  Efficiency improvement: {final_efficiency - initial_efficiency:.6f}")
    
    return {
        'initial_config': config,
        'optimized_params': optimized_params,
        'applied_params': applied_params,
        'initial_efficiency': initial_efficiency,
        'final_efficiency': final_efficiency,
        'efficiency_improvement': final_efficiency - initial_efficiency
    }

def test_network_parameter_optimization():
    """Test φ-optimized network parameter generation"""
    print("🌐 Testing network parameter optimization...")
    
    optimizer = GoldenRatioOptimizer()
    
    # Test different network sizes
    network_results = {}
    
    for network_size in [5, 10, 20, 50]:
        print(f"  Network size: {network_size}")
        
        params = optimizer.phi_optimized_network_parameters(network_size)
        network_results[network_size] = params
        
        print(f"    Coupling strength: {params['coupling_strength']:.6f}")
        print(f"    Noise level: {params['noise_level']:.6f}")
        print(f"    Critical density: {params['critical_density']:.6f}")
    
    return network_results

def test_golden_section_search():
    """Test golden section search optimization"""
    print("🔍 Testing golden section search...")
    
    optimizer = GoldenRatioOptimizer()
    
    # Test with simple functions
    test_functions = [
        ("quadratic", lambda x: (x - 2.0) ** 2 + 1.0, 0.0, 5.0, 2.0),
        ("sine", lambda x: np.sin(x) + 1.0, 0.0, 2*np.pi, np.pi/2),
        ("exponential", lambda x: np.exp(-(x - 1.0)**2), -2.0, 4.0, 1.0)
    ]
    
    search_results = {}
    
    for name, func, a, b, expected_min in test_functions:
        print(f"  Testing {name} function...")
        
        optimal_x, min_value = optimizer.golden_section_search(func, a, b)
        error = abs(optimal_x - expected_min)
        
        search_results[name] = {
            'optimal_x': optimal_x,
            'min_value': min_value,
            'expected_min': expected_min,
            'error': error
        }
        
        print(f"    Found minimum at x = {optimal_x:.6f} (expected: {expected_min:.6f})")
        print(f"    Error: {error:.6f}")
    
    return search_results

def main():
    """Main Day 7 golden ratio integration experiment"""
    print("🚀 Starting Day 7: Golden Ratio Integration Experiment...")
    print(f"Golden ratio constant: φ = {PHI:.15f}")
    
    # Run experiments
    print("\n" + "="*60)
    
    # Test 1: Consciousness parameter optimization
    consciousness_results = test_golden_ratio_optimization()
    
    print("\n" + "="*60)
    
    # Test 2: Network parameter optimization
    network_results = test_network_parameter_optimization()
    
    print("\n" + "="*60)
    
    # Test 3: Golden section search
    search_results = test_golden_section_search()
    
    # Compile results
    experiment_results = {
        "experiment_info": {
            "timestamp": time.time(),
            "experiment_name": "Day 7 Golden Ratio Integration",
            "version": "1.0"
        },
        "golden_ratio": {
            "phi_constant": PHI,
            "phi_approximation": float(PHI)
        },
        "consciousness_optimization": consciousness_results,
        "network_optimization": network_results,
        "golden_section_search": search_results,
        "key_metrics": {
            "efficiency_improvement": consciousness_results['efficiency_improvement'],
            "network_sizes_tested": len(network_results),
            "functions_optimized": len(search_results),
            "optimization_success": all(
                result['error'] < 0.1 for result in search_results.values()
            )
        }
    }
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save JSON artifact
    json_blob = json.dumps(experiment_results, sort_keys=True, indent=2).encode()
    json_hash = hashlib.sha256(json_blob).hexdigest()
    json_path = artifacts_dir / f"day7_golden_ratio_{json_hash[:8]}.json"
    
    with open(json_path, 'wb') as f:
        f.write(json_blob)
    
    print(f"\n✅ Golden ratio experiment artifact saved: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 DAY 7 GOLDEN RATIO INTEGRATION RESULTS")
    print("="*60)
    
    print(f"Golden Ratio: φ = {PHI:.15f}")
    print(f"Consciousness Efficiency Improvement: {consciousness_results['efficiency_improvement']:.6f}")
    print(f"Network Sizes Tested: {len(network_results)}")
    print(f"Functions Optimized: {len(search_results)}")
    print(f"Optimization Success: {'✅ YES' if experiment_results['key_metrics']['optimization_success'] else '❌ NO'}")
    
    print(f"\n📁 Artifact: {json_path}")
    print(f"🔍 Hash: {json_hash}")
    print("="*60)
    
    return experiment_results

if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 Day 7 golden ratio integration experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 