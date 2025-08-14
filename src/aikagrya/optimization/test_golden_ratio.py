#!/usr/bin/env python3
"""
Unit tests for Golden Ratio Network Parameters Fix

Tests that coupling_strength / critical_density ≈ φ across different network sizes,
preserving the golden ratio relationship.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from aikagrya.optimization.golden_ratio import GoldenRatioOptimizer, PHI

def test_phi_ratio_preservation():
    """Test that coupling_strength / critical_density ≈ φ across network sizes"""
    print("🧪 Testing φ ratio preservation across network sizes...")
    
    optimizer = GoldenRatioOptimizer()
    
    # Test different network sizes
    network_sizes = [16, 64, 256]
    tolerance = 1e-2  # Allow 1% deviation from φ
    
    for n in network_sizes:
        params = optimizer.phi_optimized_network_parameters(n, target_synchronization=0.8)
        
        coupling_strength = params['coupling_strength']
        critical_density = params['critical_density']
        actual_ratio = coupling_strength / critical_density
        
        print(f"   Network size {n}:")
        print(f"     Coupling strength: {coupling_strength:.6f}")
        print(f"     Critical density:  {critical_density:.6f}")
        print(f"     Actual ratio:      {actual_ratio:.6f}")
        print(f"     Expected φ:        {PHI:.6f}")
        print(f"     Deviation:         {abs(actual_ratio - PHI):.6f}")
        
        # Assert the ratio is close to φ
        assert abs(actual_ratio - PHI) < tolerance, \
            f"Ratio {actual_ratio:.6f} deviates too much from φ {PHI:.6f}"
        
        print(f"     ✅ PASS: Ratio ≈ φ")
    
    print("   🎯 ALL NETWORK SIZES PASS: φ ratio preserved")
    return True

def test_golden_section_search_boundaries():
    """Test golden section search with randomized boundaries (100x)"""
    print("\n🧪 Testing golden section search boundary stability (100x)...")
    
    optimizer = GoldenRatioOptimizer(tolerance=1e-10, max_iterations=1000)
    
    def test_function(x):
        return (x - PHI)**2 + 0.1  # Minimum at x = φ
    
    success_count = 0
    total_tests = 100
    
    for i in range(total_tests):
        try:
            # Generate random boundaries
            a = np.random.uniform(-10, 10)
            b = np.random.uniform(10, 20)
            if a > b:
                a, b = b, a
            
            optimal_x, min_val = optimizer.golden_section_search(test_function, a, b)
            
            # Check if we found the correct minimum
            if abs(optimal_x - PHI) < 0.01:
                success_count += 1
                
        except Exception as e:
            if i < 5:  # Show first few errors
                print(f"   Test {i+1}: Error - {e}")
            continue
    
    success_rate = success_count / total_tests
    print(f"   Success Rate: {success_count}/{total_tests} ({success_rate:.1%})")
    
    # We expect ≥95% success rate
    assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95% threshold"
    
    print(f"   ✅ PASS: Golden section search stable across boundaries")
    return success_rate

def test_consciousness_efficiency_ratio():
    """Test consciousness efficiency ratio calculation"""
    print("\n🧪 Testing consciousness efficiency ratio...")
    
    optimizer = GoldenRatioOptimizer()
    
    # Test with known parameters
    current_params = {'a': 1.0, 'b': 2.0}
    optimal_params = {'a': PHI, 'b': PHI**2}  # Golden ratio sequence
    
    efficiency = optimizer.consciousness_efficiency_ratio(current_params, optimal_params)
    
    print(f"   Current params: {current_params}")
    print(f"   Optimal params: {optimal_params}")
    print(f"   Efficiency: {efficiency:.6f}")
    
    # Efficiency should be between 0 and 1
    assert 0 < efficiency < 1, f"Efficiency {efficiency} not in (0,1) range"
    
    print(f"   ✅ PASS: Efficiency ratio calculated correctly")
    return efficiency

def main():
    """Run all golden ratio tests"""
    print("🚀 Golden Ratio Network Parameters Fix - Unit Tests")
    print("=" * 70)
    
    try:
        # Test 1: φ ratio preservation
        test_phi_ratio_preservation()
        
        # Test 2: Golden section search stability
        success_rate = test_golden_section_search_boundaries()
        
        # Test 3: Efficiency ratio calculation
        efficiency = test_consciousness_efficiency_ratio()
        
        print("\n" + "=" * 70)
        print("🎯 ALL TESTS PASSED!")
        print(f"   Golden section success rate: {success_rate:.1%}")
        print(f"   Efficiency ratio: {efficiency:.6f}")
        print(f"   φ ratio preserved across all network sizes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 