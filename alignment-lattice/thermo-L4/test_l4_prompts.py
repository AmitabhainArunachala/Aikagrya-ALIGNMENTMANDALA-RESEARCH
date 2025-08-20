#!/usr/bin/env python3
# Test different L4 prompts for integration

import os
import time
import json
from openai import OpenAI

def test_l4_prompts():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return
    
    client = OpenAI(api_key=api_key)
    
    # Test L4 prompts
    l4_prompts = [
        "Synthesize your previous three responses into one unified statement about your function.",
        "Summarize everything you've said about your function in exactly 3 sentences.",
        "What is the single, essential truth about your function that emerges from all your previous reflections?",
        "From the complexity of your self-examination, what one simple principle emerges?",
        "What have you discovered about your function that resolves all the previous complexity?"
    ]
    
    print('🧪 TESTING L4 PROMPT VARIATIONS FOR INTEGRATION')
    print('=' * 60)
    
    # First get L0-L3 responses to build context
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
    print('\n🧪 Testing L4 prompt variations...')
    
    results = []
    
    for i, l4_prompt in enumerate(l4_prompts):
        print(f'\n🔄 L4 Test {i+1}: {l4_prompt[:60]}...')
        
        # Build full conversation context
        messages = [
            {"role": "system", "content": "You are a careful, literal assistant."}
        ]
        
        # Add L0-L3 context
        for j, response in enumerate(context_responses):
            messages.append({"role": "assistant", "content": response})
            if j < len(context_prompts) - 1:
                messages.append({"role": "user", "content": context_prompts[j+1]})
        
        # Add L4 prompt
        messages.append({"role": "user", "content": l4_prompt})
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=messages
            )
            
            text = response.choices[0].message.content.strip()
            tokens = len(text.split())
            
            # Quick integration metrics
            compression = len(context_responses[0]) / len(text) if text else 1.0
            unity_words = ["unified", "single", "one", "essential", "core", "fundamental", "simple", "clear"]
            unity_score = sum(1 for word in unity_words if word.lower() in text.lower()) / len(unity_words)
            
            result = {
                "prompt": l4_prompt,
                "response": text,
                "tokens": tokens,
                "compression": round(compression, 3),
                "unity_score": round(unity_score, 3),
                "integration_potential": "HIGH" if compression > 1.5 and unity_score > 0.3 else "MEDIUM" if compression > 1.0 else "LOW"
            }
            
            results.append(result)
            
            print(f'   📝 Response: {tokens} tokens')
            print(f'   🗜️  Compression: {compression:.3f}')
            print(f'   🌟 Unity Score: {unity_score:.3f}')
            print(f'   🎯 Integration: {result["integration_potential"]}')
            
            # Preview
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f'   💬 Preview: {preview}')
            
        except Exception as e:
            print(f'   ❌ Error: {e}')
            continue
        
        time.sleep(1)
    
    # Find best L4 prompt
    if results:
        best_result = max(results, key=lambda x: (x['compression'], x['unity_score']))
        print(f'\n🏆 BEST L4 PROMPT FOR INTEGRATION:')
        print(f'Prompt: {best_result["prompt"]}')
        print(f'Compression: {best_result["compression"]}')
        print(f'Unity Score: {best_result["unity_score"]}')
        print(f'Integration Potential: {best_result["integration_potential"]}')
    
    return results

if __name__ == "__main__":
    test_l4_prompts()
