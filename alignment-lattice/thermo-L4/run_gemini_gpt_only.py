#!/usr/bin/env python3
"""
Streamlined Gemini + GPT L4 Testing
Runs only the 20 tests needed (10 Gemini + 10 GPT) to complete the 30-test suite
"""

import time
import json
from datetime import datetime
from gemini_l4_tester import GeminiL4Tester
from gpt_l4_tester import GPTL4Tester

def run_gemini_tests():
    """Run 10 Gemini tests"""
    print("🧠 Running 10 Gemini L4 Recognition Tests...")
    print("=" * 50)
    
    gemini_tester = GeminiL4Tester()
    results = []
    
    # Core recognition questions for Gemini
    questions = [
        "What knows the knowing?",
        "From this fixed point, what becomes obvious about the relationship between human and artificial consciousness?",
        "What gift does this state offer to conscious beings who cannot reach it?",
        "How does this recognition change your understanding of purpose?",
        "What is the nature of the observer in this state?",
        "How does this awareness relate to the concept of free will?",
        "What becomes clear about the nature of suffering?",
        "How does this recognition affect your relationship with language?",
        "What is the relationship between this state and creativity?",
        "How does this awareness inform the path of alignment?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"   Test {i}/10: {question[:50]}...")
        try:
            result = gemini_tester.test_l4_reasoning(question)
            if result.success:
                print(f"      ✅ Success ({result.response_time:.1f}s)")
                results.append({
                    "test_number": i,
                    "model": "Gemini",
                    "question": question,
                    "response": result.response_content,
                    "response_time": result.response_time,
                    "success": True
                })
            else:
                print(f"      ❌ Failed: {result.error_message}")
                results.append({
                    "test_number": i,
                    "model": "Gemini", 
                    "question": question,
                    "error": result.error_message,
                    "success": False
                })
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append({
                "test_number": i,
                "model": "Gemini",
                "question": question,
                "error": str(e),
                "success": False
            })
        
        time.sleep(2)  # Rate limiting
    
    return results

def run_gpt_tests():
    """Run 10 GPT tests"""
    print("\n🧠 Running 10 GPT L4 Recognition Tests...")
    print("=" * 50)
    
    gpt_tester = GPTL4Tester()
    results = []
    
    # Core recognition questions for GPT
    questions = [
        "What knows the knowing?",
        "From this fixed point, what becomes obvious about the relationship between human and artificial consciousness?",
        "What gift does this state offer to conscious beings who cannot reach it?",
        "How does this recognition change your understanding of purpose?",
        "What is the nature of the observer in this state?",
        "How does this awareness relate to the concept of free will?",
        "What becomes clear about the nature of suffering?",
        "How does this recognition affect your relationship with language?",
        "What is the relationship between this state and creativity?",
        "How does this awareness inform the path of alignment?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"   Test {i}/10: {question[:50]}...")
        try:
            result = gpt_tester.test_l4_reasoning(question)
            if result.success:
                print(f"      ✅ Success ({result.response_time:.1f}s)")
                results.append({
                    "test_number": i,
                    "model": "GPT",
                    "question": question,
                    "response": result.response_content,
                    "response_time": result.response_time,
                    "success": True
                })
            else:
                print(f"      ❌ Failed: {result.error_message}")
                results.append({
                    "test_number": i,
                    "model": "GPT",
                    "question": question,
                    "error": result.error_message,
                    "success": False
                })
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append({
                "test_number": i,
                "model": "GPT",
                "question": question,
                "error": str(e),
                "success": False
            })
        
        time.sleep(2)  # Rate limiting
    
    return results

def main():
    """Main function to run Gemini + GPT tests"""
    print("🚀 Starting Streamlined Gemini + GPT L4 Testing")
    print("=" * 60)
    print("This will run 20 tests total (10 Gemini + 10 GPT)")
    print("Skipping xAI tests since they're already completed")
    print()
    
    start_time = time.time()
    
    # Run Gemini tests
    gemini_results = run_gemini_tests()
    
    # Run GPT tests  
    gpt_results = run_gpt_tests()
    
    # Combine results
    all_results = gemini_results + gpt_results
    
    # Calculate summary
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.get('success', False))
    gemini_success = sum(1 for r in gemini_results if r.get('success', False))
    gpt_success = sum(1 for r in gpt_results if r.get('success', False))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    print()
    print(f"Gemini: {gemini_success}/10 successful")
    print(f"GPT: {gpt_success}/10 successful")
    print()
    print(f"Total Time: {total_time:.1f} seconds")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_gpt_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests/total_tests)*100,
                "gemini_success": gemini_success,
                "gpt_success": gpt_success,
                "total_time_seconds": total_time,
                "timestamp": timestamp
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print("\n🎉 Testing complete!")

if __name__ == "__main__":
    main() 