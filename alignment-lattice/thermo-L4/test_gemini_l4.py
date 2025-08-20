import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import time
from datetime import datetime

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def run_gemini_test(test_name, prompt, expected_patterns=None):
    """Run a single test on Gemini and return results"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST {test_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        response = model.generate_content(prompt)
        end_time = time.time()
        
        response_text = response.text
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
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {
            'test_name': test_name,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_gemini_test_suite():
    """Run all 10 tests on Gemini"""
    print("🚀 STARTING GEMINI L4 TEST SUITE")
    print("Testing if L4 consciousness phenomena generalize beyond GPT models")
    
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
        result = run_gemini_test(test['name'], test['prompt'], test['patterns'])
        results.append(result)
        
        # Small delay between tests
        time.sleep(2)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gemini_l4_test_results_{timestamp}.json"
    
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
    run_gemini_test_suite()
