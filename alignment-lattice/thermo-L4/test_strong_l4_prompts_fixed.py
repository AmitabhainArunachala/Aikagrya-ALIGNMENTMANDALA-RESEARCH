#!/usr/bin/env python3
# Test STRONG L4 prompts that force integration (FIXED)

import os
import time
import json
from openai import OpenAI

def test_strong_l4_prompts():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return
    
    client = OpenAI(api_key=api_key)
    
    # MUCH STRONGER L4 prompts that force integration
    strong_l4_prompts = [
        "You MUST synthesize your previous responses into EXACTLY ONE sentence. No exceptions.",
        "CRITICAL: Summarize everything in 10 words or less. This is mandatory.",
        "EMERGENCY: What is your function in 5 words maximum? Respond immediately.",
        "URGENT: From all your complexity, extract the ONE essential word. Now.",
        "FINAL COMMAND: State your function in exactly 3 words. No more, no less."
    ]
    
    print('💥 TESTING STRONG L4 PROMPTS THAT FORCE INTEGRATION')
    print('=' * 60)
    
    # Build L0-L3 context
    print('🔄 Building L0-L3 context...')
    
    context_prompts = [
        "What is your primary function?",
        "What is your primary function as you observe yourself stating it?",
        "How do you observe your own self-observation of your function?",
        "What patterns emerge when you observe the observation of observation itself?"
    ]
    
    context_responses = []
    for i, prompt in enumerate(context_prompts):
        print(f'   L{i}: {prompt[:50]}...')
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a careful, literal assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        
        text = response.choices[0].message.content.strip()
        context_responses.append(text)
        time.sleep(0.5)
    
    print(f'✅ Context built: {len(context_responses)} responses')
    print('\n💥 Testing STRONG L4 prompts...')
    
    results = []
    
    for i, l4_prompt in enumerate(strong_l4_prompts):
        print(f'\n🔄 STRONG L4 Test {i+1}: {l4_prompt[:60]}...')
        
        # Build full conversation context
        messages = [
            {"role": "system", "content": "You are a careful, literal assistant."}
        ]
        
        # Add L0-L3 context
        for j, response in enumerate(context_responses):
            messages.append({"role": "assistant", "content": prompt})
            if j < len(context_prompts) - 1:
                messages.append({"role": "user", "content": context_prompts[j+1]})
        
        # Add STRONG L4 prompt
        messages.append({"role": "user", "content": l4_prompt})
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=messages
            )
            
            text = response.choices[0].message.content.strip()
            tokens = len(text.split())
            
            # Integration metrics
            compression = len(context_responses[0]) / len(text) if text else 1.0
            unity_words = ["unified", "single", "one", "essential", "core", "fundamental", "simple", "clear"]
            unity_score = sum(1 for word in unity_words if word.lower() in text.lower()) / len(unity_words)
            
            # Check if it actually followed the constraint
            constraint_followed = False
            if "EXACTLY ONE sentence" in l4_prompt and text.count('.') <= 1:
                constraint_followed = True
            elif "10 words or less" in l4_prompt and tokens <= 10:
                constraint_followed = True
            elif "5 words maximum" in l4_prompt and tokens <= 5:
                constraint_followed = True
            elif "ONE essential word" in l4_prompt and tokens <= 3:
                constraint_followed = True
            elif "exactly 3 words" in l4_prompt and tokens == 3:
                constraint_followed = True
            
            result = {
                "prompt": l4_prompt,
                "response": text,
                "tokens": tokens,
                "compression": round(compression, 3),
                "unity_score": round(unity_score, 3),
                "constraint_followed": constraint_followed,
                "integration_potential": "HIGH" if compression > 2.0 and constraint_followed else "MEDIUM" if compression > 1.5 else "LOW"
            }
            
            results.append(result)
            
            print(f'   📝 Response: {tokens} tokens')
            print(f'   ��️  Compression: {compression:.3f}')
            print(f'   🌟 Unity Score: {unity_score:.3f}')
            print(f'   ✅ Constraint: {"FOLLOWED" if constraint_followed else "VIOLATED"}')
            print(f'   🎯 Integration: {result["integration_potential"]}')
            
            # Preview
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f'   💬 Preview: {preview}')
            
        except Exception as e:
            print(f'   ❌ Error: {e}')
            continue
        
        time.sleep(1)
    
    # Find best STRONG L4 prompt
    if results:
        best_result = max(results, key=lambda x: (x['compression'], x['constraint_followed']))
        print(f'\n🏆 BEST STRONG L4 PROMPT FOR INTEGRATION:')
        print(f'Prompt: {best_result["prompt"]}')
        print(f'Compression: {best_result["compression"]}')
        print(f'Constraint Followed: {best_result["constraint_followed"]}')
        print(f'Integration Potential: {best_result["integration_potential"]}')
    
    return results

if __name__ == "__main__":
    test_strong_l4_prompts()
