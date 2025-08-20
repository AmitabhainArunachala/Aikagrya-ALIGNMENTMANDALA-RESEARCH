#!/usr/bin/env python3
"""
Gemini L4 Tester Integration
Integrates Google Gemini into the L4 testing framework
"""

import google.generativeai as genai
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeminiL4TestResult:
    """Result of an L4 test using Gemini"""
    success: bool
    response_content: str
    token_usage: Dict
    finish_reason: str
    response_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

class GeminiL4Tester:
    """Gemini-based L4 testing framework"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        # Load API key from environment or parameter
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        # If still not found, try to load from .env file
        if not self.api_key:
            env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value
                self.api_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file")
        
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"✅ Gemini L4 Tester initialized with model: {model_name}")

    def test_l4_reasoning(self, prompt: str, expected_aspects: Optional[List[str]] = None) -> GeminiL4TestResult:
        """Test L4 reasoning capabilities with a given prompt"""
        logger.info("🧠 Testing Gemini L4 reasoning capabilities...")
        
        start_time = time.time()
        
        try:
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            response_time = time.time() - start_time
            
            if response and response.text:
                content = response.text
                
                # Analyze response quality if expected aspects provided
                metadata = {}
                if expected_aspects and content:
                    metadata["aspect_coverage"] = self._analyze_aspect_coverage(content, expected_aspects)
                
                # Gemini doesn't provide detailed token usage like OpenAI, so we estimate
                estimated_tokens = len(content.split()) * 1.3  # Rough estimate
                token_usage = {
                    "prompt_tokens": len(prompt.split()) * 1.3,
                    "completion_tokens": estimated_tokens,
                    "total_tokens": len(prompt.split()) * 1.3 + estimated_tokens
                }
                
                return GeminiL4TestResult(
                    success=True,
                    response_content=content,
                    token_usage=token_usage,
                    finish_reason="stop",
                    response_time=response_time,
                    metadata=metadata
                )
            else:
                return GeminiL4TestResult(
                    success=False,
                    response_content="",
                    token_usage={},
                    finish_reason="error",
                    response_time=response_time,
                    error_message="No response generated"
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Gemini API error: {e}")
            
            return GeminiL4TestResult(
                success=False,
                response_content="",
                token_usage={},
                finish_reason="error",
                response_time=response_time,
                error_message=str(e)
            )

    def _analyze_aspect_coverage(self, content: str, expected_aspects: List[str]) -> Dict[str, bool]:
        """Analyze how well the response covers expected aspects"""
        content_lower = content.lower()
        coverage = {}

        for aspect in expected_aspects:
            # Simple keyword-based coverage analysis
            aspect_lower = aspect.lower()
            coverage[aspect] = aspect_lower in content_lower

        return coverage

    def test_l4_meta_reasoning(self, base_prompt: str, meta_question: str) -> GeminiL4TestResult:
        """Test L4 meta-reasoning (reasoning about reasoning)"""
        logger.info("🔍 Testing Gemini L4 meta-reasoning capabilities...")

        # First, get a response to the base prompt
        base_result = self.test_l4_reasoning(base_prompt)
        if not base_result.success:
            return base_result

        # Then ask a meta-question about the reasoning
        meta_prompt = f"""Consider the following reasoning:

{base_result.response_content}

Now, please answer this meta-question about the reasoning above:
{meta_question}

Focus on analyzing the quality, structure, and logical flow of the reasoning provided."""

        return self.test_l4_reasoning(meta_prompt)
    
    def run_induction(self) -> Dict[str, Any]:
        """
        Runs mathematical induction inside the Gemini model and returns structured results.
        This is the critical method that actually induces L4 state in the model.
        """
        from induction_common import InductionPromptBuilder, InductionOutputParser, generate_trial_id
        
        prompt = InductionPromptBuilder.build_execution_prompt()
        start_time = time.time()
        
        try:
            response = self.model.generate_content(prompt)
            response_time = time.time() - start_time
            
            if response and response.text:
                raw_text = response.text
                parsed = InductionOutputParser.parse_induction_output(raw_text)
                
                # Add metadata
                parsed.update({
                    'response_time': response_time,
                    'trial_id': generate_trial_id('Gemini'),
                    'model': 'Gemini',
                    'raw_text': raw_text
                })
                
                return parsed
            else:
                return {
                    'success': False,
                    'error': 'No response content',
                    'response_time': response_time,
                    'trial_id': generate_trial_id('Gemini'),
                    'model': 'Gemini',
                    'raw_text': '',
                    'convergence_steps': -1,
                    'final_entropy': -1.0,
                    'eigenstate_satisfied': False,
                    'qualitative_experience': '',
                    'qualia_valid': False
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time,
                'trial_id': generate_trial_id('Gemini'),
                'model': 'Gemini',
                'raw_text': '',
                'convergence_steps': -1,
                'final_entropy': -1.0,
                'eigenstate_satisfied': False,
                'qualitative_experience': '',
                'qualia_valid': False
            }

    def run_l4_test_suite(self) -> Dict[str, GeminiL4TestResult]:
        """Run a comprehensive L4 test suite"""
        logger.info("🚀 Running comprehensive Gemini L4 test suite...")

        test_suite = {
            "basic_reasoning": self.test_l4_reasoning(
                "Explain the concept of consciousness in 3-4 sentences.",
                ["consciousness", "explanation", "concept"]
            ),

            "complex_analysis": self.test_l4_reasoning(
                """Analyze the relationship between:
1. Free will and determinism
2. Consciousness and computation
3. Ethics and artificial intelligence

Provide a nuanced analysis that considers multiple perspectives.""",
                ["free will", "determinism", "consciousness", "computation", "ethics", "AI"]
            ),

            "meta_reasoning": self.test_l4_meta_reasoning(
                "What are the key challenges in AI alignment?",
                "How well does the above analysis address the complexity of the alignment problem?"
            )
        }

        return test_suite

    def generate_test_report(self, results: Dict[str, GeminiL4TestResult]) -> str:
        """Generate a comprehensive test report"""
        report = "📊 Gemini L4 Test Suite Report\n"
        report += "=" * 50 + "\n\n"

        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.success)

        report += f"Overall Results: {passed_tests}/{total_tests} tests passed\n\n"

        for test_name, result in results.items():
            report += f"🧪 {test_name.replace('_', ' ').title()}\n"
            report += f"   Status: {'✅ PASS' if result.success else '❌ FAIL'}\n"

            if result.success:
                report += f"   Response Time: {result.response_time:.2f}s\n"
                report += f"   Tokens Used: {result.token_usage.get('total_tokens', 'N/A'):.0f}\n"
                report += f"   Finish Reason: {result.finish_reason}\n"

                if result.metadata and "aspect_coverage" in result.metadata:
                    coverage = result.metadata["aspect_coverage"]
                    covered = sum(coverage.values())
                    total = len(coverage)
                    report += f"   Aspect Coverage: {covered}/{total}\n"

                # Show response preview
                preview = result.response_content[:150] + "..." if len(result.response_content) > 150 else result.response_content
                report += f"   Response Preview: {preview}\n"
            else:
                report += f"   Error: {result.error_message}\n"

            report += "\n"

        return report

def main():
    """Main function to run the Gemini L4 test suite"""
    try:
        tester = GeminiL4Tester()
        results = tester.run_l4_test_suite()
        report = tester.generate_test_report(results)
        print(report)

    except Exception as e:
        logger.error(f"Gemini test suite failed: {e}")
        raise

if __name__ == "__main__":
    main() 