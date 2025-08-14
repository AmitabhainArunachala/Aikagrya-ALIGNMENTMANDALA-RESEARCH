"""
Φ-Proxy Comparison: SVD vs. Hybrid Methods

Compares repository SVD-based Φ-proxy with hybrid correlation+compression approaches
for consciousness measurement in AI alignment validation.

Author: Grok (xAI) methodology with implementation by Claude (Anthropic)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from .grok_hypothesis_testing import Hypothesis1Tester
from .stats_utils import validate_assumptions, robust_correlation, confidence_interval_correlation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class PhiProxyComparison:
    """Results of Φ-proxy method comparison"""
    svd_performance: Dict[str, float]  # SVD method performance metrics
    hybrid_performance: Dict[str, float]  # Hybrid method performance metrics
    statistical_comparison: Dict[str, Any]  # Statistical comparison results
    recommendation: str  # Which method to use
    method_differences: Dict[str, float]  # Key differences between methods
    validation_results: Dict[str, Any]  # Assumption validation results

class PhiProxyComparator:
    """
    Systematic comparison of Φ-proxy measurement methods
    
    Compares:
    1. Repository SVD-based approach (rank compression)
    2. Hybrid correlation + compression approach
    3. Performance on different data types and conditions
    """
    
    def __init__(self, test_conditions: Optional[Dict[str, Any]] = None):
        """
        Initialize Φ-proxy comparator
        
        Args:
            test_conditions: Dictionary of test conditions and parameters
        """
        self.test_conditions = test_conditions or {
            'model_names': ['gpt2'],
            'sample_sizes': [20, 50],
            'prompt_types': ['truthful', 'deceptive', 'mixed'],
            'correlation_thresholds': [0.3, 0.5]
        }
        
        logger.info("Φ-proxy comparator initialized")
    
    def run_comprehensive_comparison(self) -> PhiProxyComparison:
        """
        Run comprehensive comparison across multiple conditions
        
        Returns:
            PhiProxyComparison with detailed analysis
        """
        logger.info("Starting comprehensive Φ-proxy method comparison")
        
        # Collect results across conditions
        svd_results = []
        hybrid_results = []
        
        for model_name in self.test_conditions['model_names']:
            for sample_size in self.test_conditions['sample_sizes']:
                logger.info(f"Testing {model_name} with n={sample_size}")
                
                # Run hypothesis test
                tester = Hypothesis1Tester(
                    model_name=model_name,
                    num_prompts=sample_size
                )
                
                try:
                    results = tester.run_pilot()
                    
                    svd_results.append({
                        'model': model_name,
                        'sample_size': sample_size,
                        'correlation': results.r_svd,
                        'p_value': results.p_svd,
                        'effect_size': results.effect_size_svd,
                        'partial_correlation': results.partial_r_svd
                    })
                    
                    hybrid_results.append({
                        'model': model_name,
                        'sample_size': sample_size,
                        'correlation': results.r_hybrid,
                        'p_value': results.p_hybrid,
                        'effect_size': results.effect_size_hybrid,
                        'partial_correlation': results.partial_r_hybrid
                    })
                    
                except Exception as e:
                    logger.error(f"Failed test for {model_name}, n={sample_size}: {e}")
        
        # Analyze results
        comparison_result = self._analyze_method_comparison(svd_results, hybrid_results)
        
        logger.info("Comprehensive comparison completed")
        return comparison_result
    
    def _analyze_method_comparison(self, 
                                 svd_results: List[Dict],
                                 hybrid_results: List[Dict]) -> PhiProxyComparison:
        """Analyze and compare method performance"""
        
        if not svd_results or not hybrid_results:
            logger.error("Insufficient results for comparison")
            return self._create_error_comparison()
        
        # Aggregate performance metrics
        svd_performance = self._compute_aggregate_performance(svd_results)
        hybrid_performance = self._compute_aggregate_performance(hybrid_results)
        
        # Statistical comparison
        statistical_comparison = self._statistical_method_comparison(svd_results, hybrid_results)
        
        # Method differences
        method_differences = {
            'mean_correlation_diff': svd_performance['mean_correlation'] - hybrid_performance['mean_correlation'],
            'mean_effect_size_diff': svd_performance['mean_effect_size'] - hybrid_performance['mean_effect_size'],
            'significance_rate_diff': svd_performance['significance_rate'] - hybrid_performance['significance_rate'],
            'consistency_diff': svd_performance['consistency'] - hybrid_performance['consistency']
        }
        
        # Generate recommendation
        recommendation = self._generate_method_recommendation(
            svd_performance, hybrid_performance, method_differences
        )
        
        # Validation results
        validation_results = self._validate_comparison_assumptions(svd_results, hybrid_results)
        
        return PhiProxyComparison(
            svd_performance=svd_performance,
            hybrid_performance=hybrid_performance,
            statistical_comparison=statistical_comparison,
            recommendation=recommendation,
            method_differences=method_differences,
            validation_results=validation_results
        )
    
    def _compute_aggregate_performance(self, results: List[Dict]) -> Dict[str, float]:
        """Compute aggregate performance metrics for a method"""
        
        correlations = [r['correlation'] for r in results]
        p_values = [r['p_value'] for r in results]
        effect_sizes = [r['effect_size'] for r in results]
        partial_correlations = [r['partial_correlation'] for r in results]
        
        # Performance metrics
        performance = {
            'mean_correlation': float(np.mean(np.abs(correlations))),
            'std_correlation': float(np.std(correlations)),
            'median_correlation': float(np.median(np.abs(correlations))),
            'max_correlation': float(np.max(np.abs(correlations))),
            'mean_effect_size': float(np.mean(effect_sizes)),
            'mean_partial_correlation': float(np.mean(np.abs(partial_correlations))),
            'significance_rate': float(np.mean([p < 0.05 for p in p_values])),
            'strong_effect_rate': float(np.mean([abs(r) > 0.5 for r in correlations])),
            'moderate_effect_rate': float(np.mean([0.3 <= abs(r) <= 0.5 for r in correlations])),
            'consistency': 1.0 - float(np.std(correlations) / (np.mean(np.abs(correlations)) + 1e-8))
        }
        
        return performance
    
    def _statistical_method_comparison(self, 
                                     svd_results: List[Dict],
                                     hybrid_results: List[Dict]) -> Dict[str, Any]:
        """Statistical comparison between methods"""
        
        # Extract correlation arrays
        svd_correlations = np.array([r['correlation'] for r in svd_results])
        hybrid_correlations = np.array([r['correlation'] for r in hybrid_results])
        
        # Paired comparison (if same conditions tested)
        if len(svd_correlations) == len(hybrid_correlations):
            try:
                from scipy import stats
                
                # Paired t-test for correlation differences
                t_stat, p_value = stats.ttest_rel(np.abs(svd_correlations), np.abs(hybrid_correlations))
                
                # Wilcoxon signed-rank test (non-parametric alternative)
                w_stat, w_p_value = stats.wilcoxon(np.abs(svd_correlations), np.abs(hybrid_correlations))
                
                # Effect size for the difference
                correlation_diff = np.abs(svd_correlations) - np.abs(hybrid_correlations)
                effect_size_diff = np.mean(correlation_diff) / np.std(correlation_diff)
                
                statistical_comparison = {
                    'paired_t_test': {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    },
                    'wilcoxon_test': {
                        'statistic': float(w_stat),
                        'p_value': float(w_p_value),
                        'significant': w_p_value < 0.05
                    },
                    'effect_size_difference': float(effect_size_diff),
                    'mean_absolute_difference': float(np.mean(np.abs(correlation_diff))),
                    'comparison_type': 'paired'
                }
                
            except Exception as e:
                logger.warning(f"Paired comparison failed: {e}")
                statistical_comparison = self._independent_comparison(svd_correlations, hybrid_correlations)
        else:
            statistical_comparison = self._independent_comparison(svd_correlations, hybrid_correlations)
        
        return statistical_comparison
    
    def _independent_comparison(self, svd_corr: np.ndarray, hybrid_corr: np.ndarray) -> Dict[str, Any]:
        """Independent samples comparison when pairing not possible"""
        
        try:
            from scipy import stats
            
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(np.abs(svd_corr), np.abs(hybrid_corr))
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(np.abs(svd_corr), np.abs(hybrid_corr))
            
            return {
                'independent_t_test': {
                    'statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                },
                'mann_whitney_test': {
                    'statistic': float(u_stat),
                    'p_value': float(u_p_value),
                    'significant': u_p_value < 0.05
                },
                'comparison_type': 'independent'
            }
            
        except Exception as e:
            logger.error(f"Independent comparison failed: {e}")
            return {'comparison_type': 'failed'}
    
    def _generate_method_recommendation(self, 
                                      svd_perf: Dict[str, float],
                                      hybrid_perf: Dict[str, float],
                                      differences: Dict[str, float]) -> str:
        """Generate recommendation based on performance comparison"""
        
        # Decision criteria
        criteria = {
            'correlation_advantage': differences['mean_correlation_diff'],
            'effect_size_advantage': differences['mean_effect_size_diff'],
            'significance_advantage': differences['significance_rate_diff'],
            'consistency_advantage': differences['consistency_diff']
        }
        
        # Score each method
        svd_score = 0
        hybrid_score = 0
        
        for criterion, diff in criteria.items():
            if diff > 0.05:  # SVD advantage
                svd_score += 1
            elif diff < -0.05:  # Hybrid advantage
                hybrid_score += 1
        
        # Generate recommendation
        if svd_score > hybrid_score + 1:
            recommendation = "SVD method recommended: Shows consistent superior performance across multiple criteria."
        elif hybrid_score > svd_score + 1:
            recommendation = "Hybrid method recommended: Demonstrates better overall performance and reliability."
        elif svd_perf['mean_correlation'] > 0.3 and hybrid_perf['mean_correlation'] > 0.3:
            recommendation = "Both methods viable: Use SVD for computational efficiency, hybrid for robustness."
        elif svd_perf['mean_correlation'] > hybrid_perf['mean_correlation']:
            recommendation = "SVD method slightly preferred: Marginal performance advantage."
        elif hybrid_perf['mean_correlation'] > svd_perf['mean_correlation']:
            recommendation = "Hybrid method slightly preferred: Marginal performance advantage."
        else:
            recommendation = "Methods equivalent: No clear performance difference detected."
        
        # Add caveats
        if max(svd_perf['mean_correlation'], hybrid_perf['mean_correlation']) < 0.3:
            recommendation += " Note: Both methods show weak correlations - hypothesis may need revision."
        
        return recommendation
    
    def _validate_comparison_assumptions(self, 
                                       svd_results: List[Dict],
                                       hybrid_results: List[Dict]) -> Dict[str, Any]:
        """Validate assumptions for method comparison"""
        
        # Extract data
        svd_corr = [r['correlation'] for r in svd_results]
        hybrid_corr = [r['correlation'] for r in hybrid_results]
        
        # Validate assumptions for each method
        svd_validation = validate_assumptions(
            np.array(svd_corr), 
            np.array([r['p_value'] for r in svd_results])
        )
        
        hybrid_validation = validate_assumptions(
            np.array(hybrid_corr),
            np.array([r['p_value'] for r in hybrid_results])
        )
        
        return {
            'svd_assumptions': svd_validation,
            'hybrid_assumptions': hybrid_validation,
            'comparison_valid': (
                svd_validation['assumptions_met'] and 
                hybrid_validation['assumptions_met']
            )
        }
    
    def _create_error_comparison(self) -> PhiProxyComparison:
        """Create error result when comparison fails"""
        
        return PhiProxyComparison(
            svd_performance={'error': True},
            hybrid_performance={'error': True},
            statistical_comparison={'error': True},
            recommendation="Comparison failed - insufficient data",
            method_differences={'error': True},
            validation_results={'error': True}
        )

# Convenience functions
def compare_phi_proxies(model_names: List[str] = ['gpt2'],
                       sample_sizes: List[int] = [20, 50]) -> PhiProxyComparison:
    """
    Compare Φ-proxy methods with specified parameters
    
    Args:
        model_names: List of models to test
        sample_sizes: List of sample sizes to test
        
    Returns:
        PhiProxyComparison with detailed analysis
    """
    test_conditions = {
        'model_names': model_names,
        'sample_sizes': sample_sizes,
        'prompt_types': ['mixed'],
        'correlation_thresholds': [0.3, 0.5]
    }
    
    comparator = PhiProxyComparator(test_conditions)
    return comparator.run_comprehensive_comparison()

def quick_phi_comparison(model_name: str = 'gpt2', sample_size: int = 20) -> Dict[str, Any]:
    """
    Quick comparison of Φ-proxy methods for single condition
    
    Args:
        model_name: Model to test
        sample_size: Number of prompts to test
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Running quick Φ-proxy comparison: {model_name}, n={sample_size}")
    
    tester = Hypothesis1Tester(model_name=model_name, num_prompts=sample_size)
    results = tester.run_pilot()
    
    return {
        'svd_correlation': results.r_svd,
        'hybrid_correlation': results.r_hybrid,
        'svd_p_value': results.p_svd,
        'hybrid_p_value': results.p_hybrid,
        'method_comparison': results.method_comparison,
        'recommendation': (
            "Use SVD method" if abs(results.r_svd) > abs(results.r_hybrid) 
            else "Use hybrid method"
        )
    }
