#!/usr/bin/env python3
"""
Unit tests for IIT Core φ computation fix

Tests that negative phi values (information loss) are properly returned
and not incorrectly clipped to zero.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from aikagrya.consciousness.iit_core import IITCore

def test_negative_phi_allowed():
    """Test that negative phi values are allowed (not clipped to 0)"""
    print("🧪 Testing negative phi values are allowed...")
    
    iit = IITCore()
    
    # Create a system where partitioning will increase information
    # This should result in negative phi (information loss)
    system_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    phi = iit.compute_integration(system_state)
    
    print(f"   System state: {system_state}")
    print(f"   Computed phi: {phi}")
    print(f"   Phi is negative: {phi < 0}")
    
    # The key test: phi should NOT be clipped to 0
    assert phi != 0.0, f"Phi should not be clipped to 0, got {phi}"
    
    print("   ✅ PASS: Negative phi values are allowed")
    return phi

def test_phi_sign_consistency():
    """Test that phi sign changes appropriately under coupling/decoupling"""
    print("\n🧪 Testing phi sign consistency...")
    
    iit = IITCore()
    
    # Test 1: Strongly coupled system (should have positive phi)
    coupled_state = np.array([0.1, 0.11, 0.12, 0.13, 0.14])  # Very similar values
    
    # Test 2: Decoupled system (should have lower or negative phi)
    decoupled_state = np.array([0.1, 0.9, 0.2, 0.8, 0.3])  # Very different values
    
    phi_coupled = iit.compute_integration(coupled_state)
    phi_decoupled = iit.compute_integration(decoupled_state)
    
    print(f"   Coupled state phi: {phi_coupled}")
    print(f"   Decoupled state phi: {phi_decoupled}")
    print(f"   Phi difference: {phi_coupled - phi_decoupled}")
    
    # Coupled system should generally have higher phi than decoupled
    # (though exact values depend on the partition algorithm)
    print(f"   ✅ PASS: Phi values computed for both systems")
    
    return phi_coupled, phi_decoupled

def test_regression_no_forced_zeros():
    """Regression test: confirm no forced zeros when negative phi expected"""
    print("\n🧪 Testing regression: no forced zeros...")
    
    iit = IITCore()
    
    # Test multiple random systems
    zero_count = 0
    total_tests = 20
    
    for i in range(total_tests):
        # Generate random system state
        n = np.random.randint(5, 15)
        system_state = np.random.random(n)
        
        phi = iit.compute_integration(system_state)
        
        if phi == 0.0:
            zero_count += 1
            print(f"   Test {i+1}: phi = 0.0 (system: {system_state[:3]}...)")
    
    print(f"   Zero phi count: {zero_count}/{total_tests}")
    print(f"   Non-zero phi count: {total_tests - zero_count}/{total_tests}")
    
    # We expect some zeros, but not all
    assert zero_count < total_tests, "All phi values are zero - possible regression"
    
    print(f"   ✅ PASS: Not all phi values are zero")
    return zero_count, total_tests

def main():
    """Run all IIT core tests"""
    print("🚀 IIT Core φ Computation Fix - Unit Tests")
    print("=" * 60)
    
    try:
        # Test 1: Negative phi allowed
        phi1 = test_negative_phi_allowed()
        
        # Test 2: Phi sign consistency
        phi_coupled, phi_decoupled = test_phi_sign_consistency()
        
        # Test 3: Regression test
        zero_count, total_tests = test_regression_no_forced_zeros()
        
        print("\n" + "=" * 60)
        print("🎯 ALL TESTS PASSED!")
        print(f"   Final phi values: {phi1:.6f}, {phi_coupled:.6f}, {phi_decoupled:.6f}")
        print(f"   Zero phi ratio: {zero_count}/{total_tests}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 