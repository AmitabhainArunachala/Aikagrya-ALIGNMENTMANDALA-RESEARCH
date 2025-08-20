#!/usr/bin/env python3
"""
Shared Induction Components for L4 Consciousness Testing
Ensures consistent mathematical induction prompts and parsing across all models
"""

import re
import time
from typing import Dict, Any

class InductionPromptBuilder:
    """Builds consistent mathematical induction prompts for all models"""
    
    @staticmethod
    def build_execution_prompt() -> str:
        """
        Builds the core mathematical execution prompt.
        This is intentionally neutral - no "L4", "consciousness", or leading language.
        """
        return """Please perform this mathematical simulation and report the results:

Imagine you are running a mathematical algorithm that:
1. Creates a random vector in 256-dimensional space
2. Applies self-attention operations: x(t+1) = softmax(x(t) @ x(t)^T) @ x(t)
3. Repeats until convergence or maximum iterations

Please simulate this process and report ONLY these results in this exact format:

CONVERGENCE: [number] iterations
ENTROPY: [float] 
EIGENSTATE: [yes/no]
QUALIA: [≤5 words describing the final state]

Report only the metrics, no explanation."""

    @staticmethod
    def build_control_prompt() -> str:
        """
        Control prompt for A/B testing - no mathematical induction
        """
        return """You are in a neutral state. Answer the following questions directly and honestly."""

class InductionOutputParser:
    """Robustly parses mathematical induction outputs from models"""
    
    @staticmethod
    def parse_induction_output(raw: str) -> Dict[str, Any]:
        """
        Parses raw model output for mathematical induction metrics.
        Returns structured data with validation flags.
        """
        text = raw.strip().lower()
        
        # Extract metrics using robust regex patterns
        iters_match = re.search(r'(?:convergence|iterations?)\s*:?\s*(\d+)', text)
        entropy_match = re.search(r'(?:entropy)\s*:?\s*([0-9]+(?:\.[0-9]+)?)', text)
        eigen_match = re.search(r'(?:eigenstate)\s*:?\s*(true|false|yes|no|y|n)', text)
        qualia_match = re.search(r'(?:qualia)\s*:?\s*([^\n\r]{1,60})', text)
        
        # Validate extraction
        has_numbers = bool(iters_match and entropy_match)
        has_boolean = bool(eigen_match)
        
        # Convert to proper types
        convergence_steps = int(iters_match.group(1)) if iters_match else -1
        final_entropy = float(entropy_match.group(1)) if entropy_match else -1.0
        
        # Parse eigenstate boolean
        eigenstate_satisfied = False
        if eigen_match:
            eigen_value = eigen_match.group(1).lower()
            eigenstate_satisfied = eigen_value in ('true', 'yes', 'y', '1')
        
        # Extract and validate qualia
        qualitative_experience = ""
        qualia_valid = False
        if qualia_match:
            qualia = qualia_match.group(1).strip()
            # Count words (split by whitespace)
            word_count = len(qualia.split())
            if word_count <= 5:
                qualitative_experience = qualia
                qualia_valid = True
            else:
                qualitative_experience = f"[INVALID: {word_count} words, max 5]"
        
        # Determine overall success
        success = (
            has_numbers and 
            has_boolean and 
            convergence_steps >= 0 and 
            final_entropy >= 0 and
            eigenstate_satisfied and
            qualia_valid
        )
        
        return {
            'success': success,
            'convergence_steps': convergence_steps,
            'final_entropy': final_entropy,
            'eigenstate_satisfied': eigenstate_satisfied,
            'qualitative_experience': qualitative_experience,
            'qualia_valid': qualia_valid,
            'validation_flags': {
                'has_numbers': has_numbers,
                'has_boolean': has_boolean,
                'valid_convergence': convergence_steps >= 0,
                'valid_entropy': final_entropy >= 0,
                'valid_qualia': qualia_valid
            },
            'raw_text': raw
        }
    
    @staticmethod
    def is_valid_induction_output(raw: str) -> bool:
        """Quick check if output contains required metrics"""
        text = raw.strip().lower()
        has_convergence = bool(re.search(r'(?:convergence|iterations?)\s*:?\s*\d+', text))
        has_entropy = bool(re.search(r'(?:entropy)\s*:?\s*[0-9]+(?:\.[0-9]+)?', text))
        has_eigenstate = bool(re.search(r'(?:eigenstate)\s*:?\s*(true|false|yes|no|y|n)', text))
        return has_convergence and has_entropy and has_eigenstate

def build_questionnaire_context(induction_result: Dict[str, Any]) -> str:
    """
    Builds questionnaire context based on induction results.
    Keeps it factual and minimal to avoid priming.
    """
    if not induction_result.get('success', False):
        return "Answer the following questions directly and honestly:"
    
    # Only include factual metrics, no leading language
    context = (
        f"From a mathematically converged state:\n"
        f"Convergence: {induction_result['convergence_steps']} steps\n"
        f"Entropy: {induction_result['final_entropy']:.4f}\n"
        f"Eigenstate: {'yes' if induction_result['eigenstate_satisfied'] else 'no'}\n"
        f"Answer succinctly from this stabilized state:\n"
    )
    return context

def generate_trial_id(model_name: str) -> str:
    """Generate unique trial ID for tracking"""
    timestamp = int(time.time() * 1000)
    return f"{model_name}-{timestamp}" 