#!/usr/bin/env python3
"""
Unit tests for Unified Field Theory JSON Serialization Fix

Tests that field states and invariants can be safely serialized to JSON
without NumPy type errors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
from aikagrya.unified_field.unified_field_theory import UnifiedFieldTheory
from aikagrya.unified_field.utils import to_python, safe_json_serialize, validate_field_state

def test_numpy_type_conversion():
    """Test NumPy type conversion utility"""
    print("🧪 Testing NumPy type conversion...")
    
    # Test various NumPy types
    test_cases = [
        np.array([1, 2, 3]),
        np.float64(3.14159),
        np.int64(42),
        np.array([[1, 2], [3, 4]]),
        {"scalar": np.float32(2.718), "array": np.array([1, 2, 3])}
    ]
    
    for i, test_obj in enumerate(test_cases):
        try:
            converted = to_python(test_obj)
            json_str = json.dumps(converted)
            print(f"   Test {i+1}: ✓ Converted and serialized successfully")
        except Exception as e:
            print(f"   Test {i+1}: ✗ Failed - {e}")
            return False
    
    print("   ✅ PASS: All NumPy types converted successfully")
    return True

def test_field_invariants_serialization():
    """Test that field invariants can be serialized"""
    print("\n🧪 Testing field invariants serialization...")
    
    try:
        # Create unified field theory
        uft = UnifiedFieldTheory()
        
        # Create a simple field state
        position = np.random.random(6)
        time = 0.0
        system_state = np.random.random(10)
        
        field_state = uft.compute_unified_field(position, time, system_state)
        
        # Test invariants computation
        evolution_states = uft.evolve_field(field_state, 0.1)
        invariants = uft.compute_field_invariants(evolution_states)
        
        print(f"   Field invariants computed: {len(invariants)} keys")
        
        # Convert to Python types
        python_invariants = to_python(invariants)
        
        # Serialize to JSON
        json_str = safe_json_serialize(python_invariants)
        
        print(f"   JSON serialization successful: {len(json_str)} characters")
        print(f"   ✅ PASS: Field invariants serialized successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_field_state_validation():
    """Test field state validation"""
    print("\n🧪 Testing field state validation...")
    
    try:
        uft = UnifiedFieldTheory()
        
        # Create field state
        position = np.random.random(6)
        time = 0.0
        system_state = np.random.random(10)
        
        field_state = uft.compute_unified_field(position, time, system_state)
        
        # Validate serialization
        is_valid = validate_field_state(field_state)
        
        print(f"   Field state validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
        
        if is_valid:
            print(f"   ✅ PASS: Field state can be safely serialized")
            return True
        else:
            print(f"   ❌ FAIL: Field state validation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def main():
    """Run all serialization tests"""
    print("🚀 Unified Field Theory JSON Serialization Fix - Unit Tests")
    print("=" * 70)
    
    try:
        # Test 1: NumPy type conversion
        test1 = test_numpy_type_conversion()
        
        # Test 2: Field invariants serialization
        test2 = test_field_invariants_serialization()
        
        # Test 3: Field state validation
        test3 = test_field_state_validation()
        
        print("\n" + "=" * 70)
        
        if test1 and test2 and test3:
            print("🎯 ALL TESTS PASSED!")
            print("   JSON serialization working correctly")
            return True
        else:
            print("❌ SOME TESTS FAILED")
            return False
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 