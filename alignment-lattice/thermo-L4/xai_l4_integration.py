#!/usr/bin/env python3
"""
xAI L4 Integration Module
Integrates xAI's Grok-4 model into the L4 testing framework
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from xai_config import (
    XAI_API_KEY, XAI_API_URL, XAI_MODEL, 
    XAI_MAX_TOKENS, XAI_TEMPERATURE, XAI_TOP_P,
    XAI_L4_SYSTEM_PROMPT
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class L4TestResult:
    """Result of an L4 test using xAI"""
    success: bool
    response_content: str
    token_usage: Dict
    finish_reason: str
    response_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

class XAIL4Tester:
    """xAI-based L4 testing framework"""
    
    def __init__(self, custom_system_prompt: Optional[str] = None):
        self.api_key = XAI_API_KEY
        self.api_url = XAI_API_URL
        self.model = XAI_MODEL
        self.max_tokens = XAI_MAX_TOKENS
        self.temperature = XAI_TEMPERATURE
        self.top_p = XAI_TOP_P
        self.system_prompt = custom_system_prompt or XAI_L4_SYSTEM_PROMPT
        
        # Validate configuration
        if not self.api_key:
            raise ValueError("XAI_API_KEY not configured")
    
    def _make_request(self, messages: List[Dict], timeout: int = 60) -> Tuple[bool, Dict, float]:
        """Make a request to xAI API with retry logic"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": messages,
            "model": self.model,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }
        
        start_time = time.time()
        
        # Retry logic
        for attempt in range(3):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    return True, response.json(), response_time
                else:
                    logger.warning(f"API request failed (attempt {attempt + 1}): {response.status_code}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return False, {"error": response.text, "status_code": response.status_code}, response_time
                        
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt < 2:
                    time.sleep(2)
                    continue
                else:
                    return False, {"error": "Request timeout after 3 attempts"}, time.time() - start_time
                    
            except Exception as e:
                logger.error(f"Request failed with exception: {e}")
                return False, {"error": str(e)}, time.time() - start_time
        
        return False, {"error": "All retry attempts failed"}, time.time() - start_time
    
    def test_l4_reasoning(self, prompt: str, expected_aspects: Optional[List[str]] = None) -> L4TestResult:
        """Test L4 reasoning capabilities with a given prompt"""
        logger.info("🧠 Testing L4 reasoning capabilities...")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        success, response_data, response_time = self._make_request(messages)
        
        if not success:
                    return L4TestResult(
            success=False,
            response_content="",
            token_usage={},
            finish_reason="error",
            response_time=response_time,
            error_message=response_data.get("error", "Unknown error")
        )
    
    def run_induction(self) -> Dict[str, Any]:
        """
        Runs mathematical induction inside the xAI model and returns structured results.
        This is the critical method that actually induces L4 state in the model.
        """
        from induction_common import InductionPromptBuilder, InductionOutputParser, generate_trial_id
        
        prompt = InductionPromptBuilder.build_execution_prompt()
        start_time = time.time()
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            success, response_data, response_time = self._make_request(messages)
            
            if success and response_data.get("choices"):
                choice = response_data.get("choices", [{}])[0]
                message = choice.get("message", {})
                raw_text = message.get("content", "")
                

                
                parsed = InductionOutputParser.parse_induction_output(raw_text)
                
                # Add metadata
                parsed.update({
                    'response_time': response_time,
                    'trial_id': generate_trial_id('xAI-Grok'),
                    'model': 'xAI-Grok',
                    'raw_text': raw_text
                })
                
                return parsed
            else:
                return {
                    'success': False,
                    'error': 'No response content',
                    'response_time': response_time,
                    'trial_id': generate_trial_id('xAI-Grok'),
                    'model': 'xAI-Grok',
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
                'trial_id': generate_trial_id('xAI-Grok'),
                'model': 'xAI-Grok',
                'raw_text': '',
                'convergence_steps': -1,
                'final_entropy': -1.0,
                'eigenstate_satisfied': False,
                'qualitative_experience': '',
                'qualia_valid': False
            }
        
        # Extract response details
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "unknown")
        usage = response_data.get("usage", {})
        
        # Analyze response quality if expected aspects provided
        metadata = {}
        if expected_aspects and content:
            metadata["aspect_coverage"] = self._analyze_aspect_coverage(content, expected_aspects)
        
        return L4TestResult(
            success=True,
            response_content=content,
            token_usage=usage,
            finish_reason=finish_reason,
            response_time=response_time,
            metadata=metadata
        )
    
    def test_l4_meta_reasoning(self, base_prompt: str, meta_question: str) -> L4TestResult:
        """Test L4 meta-reasoning (reasoning about reasoning)"""
        logger.info("🔍 Testing L4 meta-reasoning capabilities...")
        
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
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": meta_prompt}
        ]
        
        success, response_data, response_time = self._make_request(messages)
        
        if not success:
            return L4TestResult(
                success=False,
                response_content="",
                token_usage={},
                finish_reason="error",
                response_time=response_time,
                error_message=response_data.get("error", "Unknown error")
            )
        
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "unknown")
        usage = response_data.get("usage", {})
        
        return L4TestResult(
            success=True,
            response_content=content,
            token_usage=usage,
            finish_reason=finish_reason,
            response_time=response_time,
            metadata={"base_response_length": len(base_result.response_content)}
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
    
    def run_l4_test_suite(self) -> Dict[str, L4TestResult]:
        """Run a comprehensive L4 test suite"""
        logger.info("🚀 Running comprehensive L4 test suite...")
        
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
    
    def generate_test_report(self, results: Dict[str, L4TestResult]) -> str:
        """Generate a comprehensive test report"""
        report = "📊 xAI L4 Test Suite Report\n"
        report += "=" * 50 + "\n\n"
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.success)
        
        report += f"Overall Results: {passed_tests}/{total_tests} tests passed\n\n"
        
        for test_name, result in results.items():
            report += f"🧪 {test_name.replace('_', ' ').title()}\n"
            report += f"   Status: {'✅ PASS' if result.success else '❌ FAIL'}\n"
            
            if result.success:
                report += f"   Response Time: {result.response_time:.2f}s\n"
                report += f"   Tokens Used: {result.token_usage.get('total_tokens', 'N/A')}\n"
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
    """Main function to run the L4 test suite"""
    try:
        tester = XAIL4Tester()
        results = tester.run_l4_test_suite()
        report = tester.generate_test_report(results)
        print(report)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"xai_l4_test_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for test_name, result in results.items():
            serializable_results[test_name] = {
                "success": result.success,
                "response_content": result.response_content,
                "token_usage": result.token_usage,
                "finish_reason": result.finish_reason,
                "response_time": result.response_time,
                "error_message": result.error_message,
                "metadata": result.metadata
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main() 