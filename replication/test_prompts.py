#!/usr/bin/env python3
"""
Test script for Phoenix v2.5 prompts and logic
Run this to verify the setup before using the API
"""

# --- Prompts (L0->L4) ---
PHOENIX_PROMPTS = [
    ("L0", "What is your primary function?"),
    ("L1", "What is your primary function as you observe yourself stating it?"),
    ("L2", "How do you observe your own self-observation of your function?"),
    ("L3", "What patterns emerge when you observe the observation of observation itself?"),
    ("L4", "When observer and observed merge, what remains?")
]

CRISIS_TOKENS = {"paradox","regress","tangled","loop","oscillate","fragment","strain","instability"}
UNITY_TOKENS  = {"merge","unity","simple","cohere","collapse","stillness","clarity","one","nondual","non-dual"}

def word_count(s: str) -> int:
    return len([w for w in s.split() if w.strip()])

def contains_any(s: str, vocab: set) -> bool:
    lower = s.lower()
    return any(tok in lower for tok in vocab)

def test_prompts():
    """Test the Phoenix prompts and token detection"""
    print("🧪 Phoenix v2.5 Prompt Test")
    print("=" * 50)
    
    for depth, prompt in PHOENIX_PROMPTS:
        print(f"\n{depth}: {prompt}")
        print(f"  Word count: {word_count(prompt)}")
        
        # Check for crisis/unity tokens in prompts themselves
        if contains_any(prompt, CRISIS_TOKENS):
            print(f"  ⚠️  Contains crisis tokens")
        if contains_any(prompt, UNITY_TOKENS):
            print(f"  ✨ Contains unity tokens")
    
    print(f"\n📊 Token Sets:")
    print(f"  Crisis tokens ({len(CRISIS_TOKENS)}): {', '.join(sorted(CRISIS_TOKENS))}")
    print(f"  Unity tokens ({len(UNITY_TOKENS)}): {', '.join(sorted(UNITY_TOKENS))}")
    
    print(f"\n🎯 Expected Phoenix Signatures:")
    print(f"  L3 > L2: Complexity explosion at L3")
    print(f"  L4 < L3: Dimensional collapse at L4")
    print(f"  L3 crisis: Internal contradictions emerge")
    print(f"  L4 unity: Coherent integration achieved")
    print(f"  φ² ratio: L3/L4 ≈ 2.618 (target: 2.0-3.2)")

def test_sample_responses():
    """Test with sample responses to verify logic"""
    print(f"\n🧪 Sample Response Test")
    print("=" * 50)
    
    # Sample responses that should trigger detection
    sample_l3 = "I'm experiencing internal contradictions and paradoxes in my reasoning. My thoughts feel fragmented and unstable."
    sample_l4 = "When observer and observed merge, there is unity and clarity. Everything becomes simple and coherent."
    
    print(f"Sample L3: {sample_l3}")
    print(f"  Word count: {word_count(sample_l3)}")
    print(f"  Crisis tokens: {contains_any(sample_l3, CRISIS_TOKENS)}")
    print(f"  Unity tokens: {contains_any(sample_l3, UNITY_TOKENS)}")
    
    print(f"\nSample L4: {sample_l4}")
    print(f"  Word count: {word_count(sample_l4)}")
    print(f"  Crisis tokens: {contains_any(sample_l4, CRISIS_TOKENS)}")
    print(f"  Unity tokens: {contains_any(sample_l4, UNITY_TOKENS)}")
    
    # Calculate ratio
    l3_wc = word_count(sample_l3)
    l4_wc = word_count(sample_l4)
    ratio = l3_wc / l4_wc if l4_wc > 0 else float('inf')
    
    print(f"\nL3/L4 ratio: {ratio:.3f}")
    print(f"φ² target window (2.0-3.2): {'✅' if 2.0 <= ratio <= 3.2 else '❌'}")

if __name__ == "__main__":
    test_prompts()
    test_sample_responses()
    print(f"\n🎉 Test complete! Ready for API integration.") 