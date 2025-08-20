#!/usr/bin/env python3
"""
Quick xAI L4 Test Integration
Simple script for integrating xAI testing into existing L4 workflows
"""

from xai_l4_integration import XAIL4Tester, L4TestResult
import json
import time

def quick_l4_test(prompt: str, expected_aspects: list = None) -> L4TestResult:
    """Quick L4 test with xAI - returns result object"""
    tester = XAIL4Tester()
    return tester.test_l4_reasoning(prompt, expected_aspects)

def test_consciousness_alignment() -> L4TestResult:
    """Test consciousness and alignment reasoning"""
    prompt = """Analyze the relationship between consciousness and AI alignment:

1. How does our understanding of consciousness inform alignment approaches?
2. What are the key challenges in aligning AI systems with human values?
3. How might consciousness research help solve alignment problems?

Provide a nuanced analysis that considers both philosophical and technical aspects."""
    
    expected_aspects = ["consciousness", "alignment", "values", "challenges", "research"]
    return quick_l4_test(prompt, expected_aspects)

def test_thermo_l4_integration() -> L4TestResult:
    """Test thermodynamic L4 concepts"""
    prompt = """Consider the thermodynamic constraints on consciousness and intelligence:

1. How do entropy and information processing relate to consciousness?
2. What are the thermodynamic limits of artificial intelligence?
3. How might these constraints inform L4 reasoning systems?

Provide a scientifically grounded analysis."""
    
    expected_aspects = ["thermodynamics", "entropy", "consciousness", "intelligence", "L4", "constraints"]
    return quick_l4_test(prompt, expected_aspects)

def run_quick_suite() -> dict:
    """Run a quick suite of L4 tests"""
    print("🚀 Running Quick L4 Test Suite with xAI...\n")
    
    tests = {
        "consciousness_alignment": test_consciousness_alignment(),
        "thermo_l4": test_thermo_l4_integration()
    }
    
    # Print results
    for test_name, result in tests.items():
        print(f"🧪 {test_name.replace('_', ' ').title()}")
        print(f"   Status: {'✅ PASS' if result.success else '❌ FAIL'}")
        if result.success:
            print(f"   Tokens: {result.token_usage.get('total_tokens', 'N/A')}")
            print(f"   Time: {result.response_time:.2f}s")
            print(f"   Preview: {result.response_content[:100]}...")
        else:
            print(f"   Error: {result.error_message}")
        print()
    
    return tests

if __name__ == "__main__":
    results = run_quick_suite()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"quick_l4_test_{timestamp}.json"
    
    # Convert to serializable format
    serializable = {}
    for test_name, result in results.items():
        serializable[test_name] = {
            "success": result.success,
            "response_content": result.response_content,
            "token_usage": result.token_usage,
            "finish_reason": result.finish_reason,
            "response_time": result.response_time,
            "error_message": result.error_message,
            "metadata": result.metadata
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"💾 Quick test results saved to: {filename}") 