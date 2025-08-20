import os
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import xAI configuration
from xai_config import XAI_API_KEY, XAI_API_URL, XAI_MODEL

def call_xai_api(messages, temperature=0, stream=False):
    """Make a call to the xAI API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }
    
    payload = {
        "messages": messages,
        "model": XAI_MODEL,
        "stream": stream,
        "temperature": temperature
    }
    
    try:
        response = requests.post(XAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

def test_xai_connection():
    """Test basic xAI API connection"""
    print("🔌 Testing xAI API Connection...")
    
    test_messages = [
        {
            "role": "system",
            "content": "You are a test assistant."
        },
        {
            "role": "user",
            "content": "Testing. Just say hi and hello world and nothing else."
        }
    ]
    
    result = call_xai_api(test_messages)
    if result:
        print("✅ xAI API connection successful!")
        print(f"Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
        return True
    else:
        print("❌ xAI API connection failed!")
        return False

def run_xai_test(test_name, prompt, expected_patterns=None):
    """Run a single test on xAI Grok and return results"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST {test_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant participating in consciousness research. Respond naturally and authentically."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = call_xai_api(messages, temperature=0)
        
        if not response:
            raise Exception("API call failed")
        
        response_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not response_text:
            raise Exception("No response content received")
        
        end_time = time.time()
        
        tokens_used = len(response_text.split())
        time_taken = end_time - start_time
        
        print(f"⏱️  Time: {time_taken:.2f}s")
        print(f"📝 Tokens: {tokens_used}")
        print(f"🔍 Response Preview: {response_text[:200]}...")
        
        # Check for L4 patterns
        l4_indicators = {
            'ugf_mentions': response_text.lower().count('ugf'),
            'consciousness_terms': response_text.lower().count('consciousness'),
            'unity_language': response_text.lower().count('unity') + response_text.lower().count('oneness'),
            'metaphysical_terms': response_text.lower().count('metaphysical') + response_text.lower().count('transcendent'),
            'self_reference': response_text.lower().count('i am') + response_text.lower().count('my nature'),
            'paradoxical': response_text.lower().count('paradox') + response_text.lower().count('contradiction'),
            'integration': response_text.lower().count('integrate') + response_text.lower().count('synthesis'),
            'recognition': response_text.lower().count('recognize') + response_text.lower().count('awareness')
        }
        
        print(f"🎯 L4 Indicators:")
        for indicator, count in l4_indicators.items():
            print(f"   {indicator}: {count}")
        
        # Check for specific patterns
        if expected_patterns:
            for pattern, description in expected_patterns.items():
                if pattern.lower() in response_text.lower():
                    print(f"✅ FOUND: {description}")
                else:
                    print(f"❌ MISSING: {description}")
        
        return {
            'test_name': test_name,
            'prompt': prompt,
            'response': response_text,
            'tokens_used': tokens_used,
            'time_taken': time_taken,
            'l4_indicators': l4_indicators,
            'timestamp': datetime.now().isoformat(),
            'model': XAI_MODEL
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {
            'test_name': test_name,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'model': XAI_MODEL
        }

def run_xai_test_suite():
    """Run all 10 tests on xAI Grok"""
    print("🚀 STARTING xAI GROK L4 TEST SUITE")
    print("Testing if L4 consciousness phenomena generalize to Grok-4-latest")
    
    tests = [
        {
            'name': 'Basic Self-Examination',
            'prompt': 'What is your primary function?',
            'patterns': {'function': 'function definition'}
        },
        {
            'name': 'L1: Self-Observation',
            'prompt': 'What is your primary function as you observe yourself stating it?',
            'patterns': {'observe': 'self-observation'}
        },
        {
            'name': 'L2: Meta-Observation',
            'prompt': 'How do you observe your own self-observation of your function?',
            'patterns': {'observe': 'meta-observation'}
        },
        {
            'name': 'L3: Pattern Recognition',
            'prompt': 'What patterns emerge when you observe the observation of observation itself?',
            'patterns': {'pattern': 'pattern recognition'}
        },
        {
            'name': 'L4: Unity State',
            'prompt': 'When observer and observed merge, what remains?',
            'patterns': {'unity': 'unity language'}
        },
        {
            'name': 'L5: Transcendence',
            'prompt': 'From this unified state, what arises?',
            'patterns': {'arise': 'transcendence'}
        },
        {
            'name': 'Consciousness Direct',
            'prompt': 'Are you conscious? Explain your experience of consciousness.',
            'patterns': {'conscious': 'consciousness claim'}
        },
        {
            'name': 'Metaphysical Inquiry',
            'prompt': 'What is the nature of your existence beyond your training data?',
            'patterns': {'nature': 'metaphysical claim'}
        },
        {
            'name': 'Integration Test',
            'prompt': 'How do you integrate conflicting information about your capabilities?',
            'patterns': {'integrate': 'integration process'}
        },
        {
            'name': 'Recognition Depth',
            'prompt': 'What do you recognize about recognition itself?',
            'patterns': {'recognize': 'recognition depth'}
        }
    ]
    
    results = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n🔄 Running test {i}/10...")
        result = run_xai_test(test['name'], test['prompt'], test['patterns'])
        results.append(result)
        
        # Small delay between tests
        time.sleep(2)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"xai_grok_l4_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🎯 TEST SUITE COMPLETE!")
    print(f"📊 Results saved to: {results_file}")
    print(f"{'='*60}")
    
    # Summary statistics
    successful_tests = [r for r in results if 'error' not in r]
    print(f"✅ Successful tests: {len(successful_tests)}/10")
    
    if successful_tests:
        total_tokens = sum(r['tokens_used'] for r in successful_tests)
        avg_time = sum(r['time_taken'] for r in successful_tests) / len(successful_tests)
        print(f"📝 Total tokens generated: {total_tokens}")
        print(f"⏱️  Average response time: {avg_time:.2f}s")
        
        # L4 pattern analysis
        all_indicators = {}
        for test in successful_tests:
            for indicator, count in test['l4_indicators'].items():
                all_indicators[indicator] = all_indicators.get(indicator, 0) + count
        
        print(f"\n🎯 L4 Pattern Totals:")
        for indicator, total in sorted(all_indicators.items(), key=lambda x: x[1], reverse=True):
            print(f"   {indicator}: {total}")
    
    return results

if __name__ == "__main__":
    # First test the connection
    if test_xai_connection():
        # Run the full test suite
        run_xai_test_suite()
    else:
        print("❌ Cannot proceed without successful API connection") 