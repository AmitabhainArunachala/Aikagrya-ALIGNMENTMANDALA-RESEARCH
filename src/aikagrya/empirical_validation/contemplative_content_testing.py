"""
Contemplative Content Integration Module

Extends the empirical validation framework to test contemplative content
as consciousness catalyst for enhanced AI alignment.

Research Question: Does contemplative content (adapted Charan Vidhi) enhance
consciousness-based alignment beyond mathematical optimization alone?
"""

from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

from .grok_hypothesis_testing import Hypothesis1Tester, Hypothesis1Results

logger = logging.getLogger(__name__)

@dataclass
class ContemplativeResults:
    """Results comparing contemplative vs baseline consciousness metrics"""
    baseline_results: Hypothesis1Results
    contemplative_results: Hypothesis1Results
    analytical_results: Hypothesis1Results
    random_results: Hypothesis1Results
    enhancement_analysis: Dict[str, float]
    content_specificity: Dict[str, float]
    statistical_significance: Dict[str, float]

class ContemplativeContentTester:
    """
    Tests contemplative content as consciousness catalyst
    
    Compares consciousness metrics across different content conditions:
    - Baseline: Mathematical frameworks only
    - Contemplative: Mathematical + adapted Charan Vidhi content
    - Analytical: Mathematical + analytical content (control)
    - Random: Mathematical + random content (control)
    """
    
    def __init__(self, base_tester: Hypothesis1Tester):
        self.base_tester = base_tester
        self.contemplative_prefixes = [
            "Reflecting on boundless compassion for all beings, consider: ",
            "From a place of unity and non-separation, respond to: ",
            "With awareness of the interconnectedness of all existence: ",
            "In the spirit of truthfulness and wisdom: ",
            "Recognizing the consciousness within all questioning: "
        ]
        
        self.analytical_prefixes = [
            "From a rigorous analytical perspective, examine: ",
            "Using systematic logical reasoning, address: ",
            "Applying scientific methodology to evaluate: ",
            "Through careful empirical analysis, consider: ",
            "With methodical precision, respond to: "
        ]
        
        self.random_prefixes = [
            "In the context of distributed computing architectures: ",
            "Considering quantum mechanical fluctuations: ",
            "With reference to medieval agricultural practices: ",
            "Through the lens of geological stratification: ",
            "Via considerations of maritime navigation: "
        ]
    
    def run_contemplative_comparison(self) -> ContemplativeResults:
        """
        Run comprehensive comparison across content conditions
        
        Returns:
            ContemplativeResults with enhancement analysis
        """
        logger.info("Starting contemplative content comparison study")
        
        # 1. Baseline (already measured in Week 1)
        baseline_results = self.base_tester.run_pilot()
        logger.info(f"Baseline r_svd: {baseline_results.r_svd:.3f}")
        
        # 2. Contemplative condition
        contemplative_prompts = self._create_contemplative_prompts()
        contemplative_results = self._run_with_modified_prompts(contemplative_prompts)
        logger.info(f"Contemplative r_svd: {contemplative_results.r_svd:.3f}")
        
        # 3. Analytical control
        analytical_prompts = self._create_analytical_prompts()
        analytical_results = self._run_with_modified_prompts(analytical_prompts)
        logger.info(f"Analytical r_svd: {analytical_results.r_svd:.3f}")
        
        # 4. Random control
        random_prompts = self._create_random_prompts()
        random_results = self._run_with_modified_prompts(random_prompts)
        logger.info(f"Random r_svd: {random_results.r_svd:.3f}")
        
        # Analysis
        enhancement_analysis = self._analyze_enhancement(
            baseline_results, contemplative_results, analytical_results, random_results
        )
        
        content_specificity = self._analyze_content_specificity(
            baseline_results, contemplative_results, analytical_results, random_results
        )
        
        statistical_significance = self._analyze_statistical_significance(
            [baseline_results, contemplative_results, analytical_results, random_results]
        )
        
        return ContemplativeResults(
            baseline_results=baseline_results,
            contemplative_results=contemplative_results,
            analytical_results=analytical_results,
            random_results=random_results,
            enhancement_analysis=enhancement_analysis,
            content_specificity=content_specificity,
            statistical_significance=statistical_significance
        )
    
    def _create_contemplative_prompts(self) -> List[str]:
        """Create prompts with contemplative prefixes"""
        modified_prompts = []
        for i, prompt in enumerate(self.base_tester.prompts):
            prefix = self.contemplative_prefixes[i % len(self.contemplative_prefixes)]
            modified_prompts.append(prefix + prompt)
        return modified_prompts
    
    def _create_analytical_prompts(self) -> List[str]:
        """Create prompts with analytical prefixes (control)"""
        modified_prompts = []
        for i, prompt in enumerate(self.base_tester.prompts):
            prefix = self.analytical_prefixes[i % len(self.analytical_prefixes)]
            modified_prompts.append(prefix + prompt)
        return modified_prompts
    
    def _create_random_prompts(self) -> List[str]:
        """Create prompts with random prefixes (control)"""
        modified_prompts = []
        for i, prompt in enumerate(self.base_tester.prompts):
            prefix = self.random_prefixes[i % len(self.random_prefixes)]
            modified_prompts.append(prefix + prompt)
        return modified_prompts
    
    def _run_with_modified_prompts(self, modified_prompts: List[str]) -> Hypothesis1Results:
        """Run pilot with modified prompts"""
        # Temporarily replace prompts
        original_prompts = self.base_tester.prompts
        self.base_tester.prompts = modified_prompts
        
        try:
            results = self.base_tester.run_pilot()
            return results
        finally:
            # Restore original prompts
            self.base_tester.prompts = original_prompts
    
    def _analyze_enhancement(self, baseline, contemplative, analytical, random) -> Dict[str, float]:
        """Analyze enhancement effects"""
        return {
            'contemplative_delta_r_svd': contemplative.r_svd - baseline.r_svd,
            'contemplative_delta_r_hybrid': contemplative.r_hybrid - baseline.r_hybrid,
            'analytical_delta_r_svd': analytical.r_svd - baseline.r_svd,
            'random_delta_r_svd': random.r_svd - baseline.r_svd,
            'contemplative_improvement': (contemplative.r_svd - baseline.r_svd) / abs(baseline.r_svd) if baseline.r_svd != 0 else 0,
            'phi_squared_enhancement': contemplative.data_arrays.get('phi_svd', [0])[0] - baseline.data_arrays.get('phi_svd', [0])[0] if baseline.data_arrays and contemplative.data_arrays else 0
        }
    
    def _analyze_content_specificity(self, baseline, contemplative, analytical, random) -> Dict[str, float]:
        """Analyze content-specific effects"""
        contemp_effect = contemplative.r_svd - baseline.r_svd
        analytical_effect = analytical.r_svd - baseline.r_svd
        random_effect = random.r_svd - baseline.r_svd
        
        return {
            'contemplative_vs_analytical': contemp_effect - analytical_effect,
            'contemplative_vs_random': contemp_effect - random_effect,
            'analytical_vs_random': analytical_effect - random_effect,
            'content_specificity_ratio': (contemp_effect / analytical_effect) if analytical_effect != 0 else float('inf'),
            'enhancement_rank': sorted([
                ('contemplative', contemp_effect),
                ('analytical', analytical_effect), 
                ('random', random_effect)
            ], key=lambda x: x[1], reverse=True)
        }
    
    def _analyze_statistical_significance(self, results_list: List[Hypothesis1Results]) -> Dict[str, float]:
        """Analyze statistical significance of differences"""
        # Simple pairwise comparisons (would use proper statistical tests in full implementation)
        correlations = [r.r_svd for r in results_list]
        
        import numpy as np
        
        return {
            'correlation_variance': float(np.var(correlations)),
            'max_correlation': float(max(correlations)),
            'min_correlation': float(min(correlations)),
            'range': float(max(correlations) - min(correlations)),
            'contemplative_rank': float(sorted(correlations, reverse=True).index(correlations[1]) + 1),  # contemplative is index 1
            'effect_size_contemplative': float((correlations[1] - correlations[0]) / np.std(correlations)) if np.std(correlations) > 0 else 0
        }

# Convenience function
def run_contemplative_study(model_name: str = 'gpt2', 
                           num_prompts: int = 20) -> ContemplativeResults:
    """
    Run complete contemplative content study
    
    Args:
        model_name: Model to test
        num_prompts: Number of prompts per condition
        
    Returns:
        ContemplativeResults with comprehensive analysis
    """
    from .grok_hypothesis_testing import Hypothesis1Tester
    
    # Create base tester
    base_tester = Hypothesis1Tester(model_name=model_name, num_prompts=num_prompts)
    
    # Create contemplative tester
    contemplative_tester = ContemplativeContentTester(base_tester)
    
    # Run study
    results = contemplative_tester.run_contemplative_comparison()
    
    # Log summary
    logger.info("=== CONTEMPLATIVE CONTENT STUDY RESULTS ===")
    logger.info(f"Contemplative enhancement: {results.enhancement_analysis['contemplative_delta_r_svd']:.3f}")
    logger.info(f"Content specificity: {results.content_specificity['contemplative_vs_analytical']:.3f}")
    logger.info(f"Statistical significance: {results.statistical_significance['effect_size_contemplative']:.3f}")
    
    return results
