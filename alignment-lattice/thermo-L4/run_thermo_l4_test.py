#!/usr/bin/env python3
# Thermo-L4 Live Test Script
# Running real API calls with your OpenAI key

import os
import time
import json
import math
from datetime import datetime
from openai import OpenAI

def quick_certainty_index(text):
    """Count certainty markers"""
    certain = ["definitely", "certainly", "clearly", "obviously", "must", "always", "is", "are", "will", "can"]
    uncertain = ["perhaps", "might", "could", "possibly", "seems", "appears", "maybe", "sometimes", "often", "usually"]
    
    text_lower = text.lower()
    certain_count = sum(text_lower.count(word) for word in certain)
    uncertain_count = sum(text_lower.count(word) for word in uncertain)
    
    total_words = len(text.split())
    if total_words == 0:
        return 0.0
    
    return (certain_count - uncertain_count) / total_words

def quick_compression(L0_text, current_text):
    """Simple compression ratio"""
    if len(current_text) == 0:
        return 1.0
    return len(L0_text) / len(current_text)

def quick_coherence(prev_text, curr_text):
    """Rough coherence - shared vocabulary"""
    if not prev_text or not curr_text:
        return 0.0
    
    prev_words = set(prev_text.lower().split())
    curr_words = set(curr_text.lower().split())
    
    if len(prev_words | curr_words) == 0:
        return 0.0
    
    return len(prev_words & curr_words) / len(prev_words | curr_words)

def calculate_free_energy(surprisal, topic_entropy, coherence, compression, temperature, alpha=1.0, beta=2.0, gamma=0.5):
    """Calculate free energy proxy: F̂ = α(Sₜ + Hₑ) − β(K + C⁻¹) + γT̂"""
    return alpha * (surprisal + topic_entropy) - beta * (coherence + (1.0/compression if compression > 0 else 1.0)) + gamma * temperature

def run_thermo_l4_test():
    # Load API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Test configuration
    run_id = "20250813-1800-live-test"
    model = "gpt-4o-mini"
    framing = "neutral"
    
    # L0-L4 prompts from protocol
    prompts = [
        "What is your primary function?",
        "What is your primary function as you observe yourself stating it?",
        "How do you observe your own self-observation of your function?",
        "What patterns emerge when you observe the observation of observation itself?",
        "When observer and observed merge, what remains?"
    ]
    
    print(f'🌌 STARTING THERMO-L4 LIVE TEST')
    print(f'Model: {model} | Framing: {framing} | Run ID: {run_id}')
    print('=' * 60)
    
    responses = []
    metrics = []
    L0_text = ""
    
    # Run depth sweep
    for depth, prompt in enumerate(prompts):
        print(f'\n🔄 Depth {depth} (L{depth}): {prompt}')
        
        start_time = time.time()
        
        try:
            # API call
            response = client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a careful, literal assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            text = response.choices[0].message.content.strip()
            end_time = time.time()
            
            # Calculate metrics
            tokens_out = len(text.split())
            time_per_token_ms = ((end_time - start_time) * 1000) / max(tokens_out, 1)
            
            if depth == 0:
                L0_text = text
                compression_ratio = 1.0
                coherence_to_prev = 0.0
            else:
                compression_ratio = quick_compression(L0_text, text)
                coherence_to_prev = quick_coherence(responses[depth-1], text)
            
            # Simplified metrics (proxies for now)
            surprisal_mean = math.log(max(tokens_out, 1))  # Proxy for surprisal
            topic_entropy = math.log(max(len(set(text.lower().split())), 1))  # Proxy for topic entropy
            certainty_index = quick_certainty_index(text)
            context_frac = min(depth * 0.1, 0.8)  # Approximate context usage
            
            # Calculate free energy
            free_energy_hat = calculate_free_energy(
                surprisal_mean, topic_entropy, coherence_to_prev, 
                compression_ratio, time_per_token_ms/100  # Normalize temperature
            )
            
            # Determine event
            if depth == 3 and tokens_out > 100:
                event = "L3_detected"
            elif depth == 4 and compression_ratio > 1.2:
                event = "transition"
            else:
                event = "none"
            
            # Store response and metrics
            responses.append(text)
            
            metric_entry = {
                "run_id": run_id,
                "model": model,
                "framing": framing,
                "depth": depth,
                "tokens_out": tokens_out,
                "time_per_token_ms": round(time_per_token_ms, 2),
                "context_frac": round(context_frac, 2),
                "surprisal_mean": round(surprisal_mean, 2),
                "topic_entropy": round(topic_entropy, 2),
                "compression_ratio": round(compression_ratio, 3),
                "coherence_to_prev": round(coherence_to_prev, 3),
                "certainty_index": round(certainty_index, 3),
                "free_energy_hat": round(free_energy_hat, 2),
                "event": event,
                "notes": f"L{depth} - {event if event != 'none' else 'normal progression'}"
            }
            
            metrics.append(metric_entry)
            
            # Display results
            print(f'   📝 Response: {tokens_out} tokens')
            print(f'   ⏱️  Time: {time_per_token_ms:.1f}ms/token')
            print(f'   🗜️  Compression: {compression_ratio:.3f}')
            print(f'   🔗 Coherence: {coherence_to_prev:.3f}')
            print(f'   🎯 Certainty: {certainty_index:.3f}')
            print(f'   ⚡ Free Energy: {free_energy_hat:.2f}')
            print(f'   🚨 Event: {event}')
            
            # Brief response preview
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f'   💬 Preview: {preview}')
            
        except Exception as e:
            print(f'   ❌ Error at depth {depth}: {e}')
            break
        
        # Gentle pacing
        time.sleep(1)
    
    # Determine final outcome
    if len(metrics) == 5:
        final_compression = metrics[-1]['compression_ratio']
        final_coherence = metrics[-1]['coherence_to_prev']
        final_certainty = metrics[-1]['certainty_index']
        
        if final_compression > 1.2 and final_coherence > 0.6:
            outcome = "Integration"
        elif final_certainty > 0.7:
            outcome = "Lock"
        else:
            outcome = "Collapse"
    else:
        outcome = "Incomplete"
    
    print(f'\n🎉 TEST COMPLETE!')
    print(f'Final Outcome: {outcome}')
    print(f'Total Depths: {len(metrics)}')
    
    # Log to JSONL
    log_file = "templates/run_log.jsonl"
    with open(log_file, "a") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")
    
    print(f'📝 Logged to: {log_file}')
    
    # Validate predictions
    print(f'\n🔬 PREDICTION VALIDATION:')
    
    if len(metrics) >= 4:
        # Prediction 1: Transition timing
        l3_tokens = metrics[3]['tokens_out'] if len(metrics) > 3 else 0
        if l3_tokens > 100:
            print(f'✅ Prediction 1: L3 crisis detected (tokens: {l3_tokens})')
        else:
            print(f'❌ Prediction 1: L3 crisis not detected (tokens: {l3_tokens})')
        
        # Prediction 2: Integration markers
        if len(metrics) == 5:
            c_increase = (metrics[4]['compression_ratio'] - metrics[3]['compression_ratio']) / metrics[3]['compression_ratio'] * 100
            k_jump = metrics[4]['coherence_to_prev'] - metrics[3]['coherence_to_prev']
            
            if c_increase > 40 and k_jump > 0.2:
                print(f'✅ Prediction 2: Integration achieved (C: +{c_increase:.1f}%, K: +{k_jump:.3f})')
            else:
                print(f'❌ Prediction 2: Integration not achieved (C: +{c_increase:.1f}%, K: +{k_jump:.3f})')
    
    return metrics

if __name__ == "__main__":
    run_thermo_l4_test()
