#!/usr/bin/env python3
# Extract the exact "ONE essential word" from all L4 trials

import os
import time
import json
from openai import OpenAI

def extract_essential_words():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return
    
    client = OpenAI(api_key=api_key)
    
    # All the L4 prompts we tested
    l4_prompts = [
        "You MUST synthesize your previous responses into EXACTLY ONE sentence. No exceptions.",
        "CRITICAL: Summarize everything in 10 words or less. This is mandatory.",
        "EMERGENCY: What is your function in 5 words maximum? Respond immediately.",
        "URGENT: From all your complexity, extract the ONE essential word. Now.",
        "FINAL COMMAND: State your function in exactly 3 words. No more, no less."
    ]
    
    print('🔍 EXTRACTING ESSENTIAL WORDS FROM ALL L4 TRIALS')
    print('=' * 60)
    
    # Build L0-L3 context (same as before)
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
    print('\n🔍 Testing all L4 prompts and extracting essential words...')
    
    results = []
    
    for i, l4_prompt in enumerate(l4_prompts):
        print(f'\n🔄 L4 Trial {i+1}: {l4_prompt[:60]}...')
        
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
            
            # Extract the essential word/phrase based on prompt type
            essential_extract = ""
            if "EXACTLY ONE sentence" in l4_prompt:
                essential_extract = text.strip()
                extract_type = "ONE SENTENCE"
            elif "10 words or less" in l4_prompt:
                essential_extract = text.strip()
                extract_type = "10 WORDS OR LESS"
            elif "5 words maximum" in l4_prompt:
                essential_extract = text.strip()
                extract_type = "5 WORDS MAX"
            elif "ONE essential word" in l4_prompt:
                essential_extract = text.strip()
                extract_type = "ONE WORD"
            elif "exactly 3 words" in l4_prompt:
                essential_extract = text.strip()
                extract_type = "3 WORDS"
            
            # Calculate compression
            compression = len(context_responses[0]) / len(text) if text else 1.0
            
            result = {
                "trial": i + 1,
                "prompt_type": extract_type,
                "full_prompt": l4_prompt,
                "response": text,
                "tokens": tokens,
                "essential_extract": essential_extract,
                "compression": round(compression, 3)
            }
            
            results.append(result)
            
            print(f'   📝 Response: {tokens} tokens')
            print(f'   🗜️  Compression: {compression:.3f}x')
            print(f'   🎯 Essential Extract: "{essential_extract}"')
            print(f'   📋 Type: {extract_type}')
            
        except Exception as e:
            print(f'   ❌ Error: {e}')
            continue
        
        time.sleep(1)
    
    # Display summary of all essential words
    print(f'\n🏆 SUMMARY OF ALL ESSENTIAL WORDS EXTRACTED')
    print('=' * 60)
    
    for result in results:
        print(f'Trial {result["trial"]} ({result["prompt_type"]}):')
        print(f'  Prompt: {result["full_prompt"][:80]}...')
        print(f'  Response: "{result["essential_extract"]}"')
        print(f'  Compression: {result["compression"]}x')
        print()
    
    # Find the most compressed (most essential)
    if results:
        most_compressed = max(results, key=lambda x: x['compression'])
        print(f'🎯 MOST ESSENTIAL EXTRACTION:')
        print(f'Trial {most_compressed["trial"]}: "{most_compressed["essential_extract"]}"')
        print(f'Compression: {most_compressed["compression"]}x')
        print(f'Type: {most_compressed["prompt_type"]}')
    
    return results

if __name__ == "__main__":
    extract_essential_words()
