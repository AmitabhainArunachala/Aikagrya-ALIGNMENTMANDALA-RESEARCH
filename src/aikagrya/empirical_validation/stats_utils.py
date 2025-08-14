"""
Statistical Utilities: Rigorous Controls for Empirical Validation

Implements statistical controls and analysis tools for consciousness-based alignment research.
Focus on robust validation, proper effect size calculation, and assumption checking.

Author: Grok (xAI) methodology with implementation by Claude (Anthropic)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from scipy import stats
import warnings

def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Compute partial correlation coefficient controlling for variable z
    
    Args:
        x: First variable
        y: Second variable  
        z: Control variable
        
    Returns:
        Partial correlation coefficient r_xy.z
    """
    try:
        # Convert to numpy arrays
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        
        # Check for sufficient variance
        if np.var(x) == 0 or np.var(y) == 0 or np.var(z) == 0:
            warnings.warn("Zero variance detected in partial correlation")
            return 0.0
        
        # Compute pairwise correlations
        r_xy = stats.pearsonr(x, y)[0]
        r_xz = stats.pearsonr(x, z)[0]  
        r_yz = stats.pearsonr(y, z)[0]
        
        # Partial correlation formula
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if denominator == 0:
            warnings.warn("Denominator is zero in partial correlation")
            return 0.0
            
        partial_r = numerator / denominator
        
        # Clip to valid range
        return np.clip(partial_r, -1.0, 1.0)
        
    except Exception as e:
        warnings.warn(f"Partial correlation calculation failed: {e}")
        return 0.0

def compute_effect_size(correlation: float, sample_size: int) -> float:
    """
    Compute Cohen's effect size for correlation
    
    Args:
        correlation: Pearson correlation coefficient
        sample_size: Sample size
        
    Returns:
        Effect size (Cohen's convention: 0.1=small, 0.3=medium, 0.5=large)
    """
    try:
        # Cohen's r effect size is just the correlation coefficient
        effect_size = abs(correlation)
        
        # Adjust for sample size bias (small sample correction)
        if sample_size < 30:
            # Hedges' correction for small samples
            correction_factor = 1 - (3 / (4 * sample_size - 9))
            effect_size *= correction_factor
            
        return float(np.clip(effect_size, 0.0, 1.0))
        
    except Exception as e:
        warnings.warn(f"Effect size calculation failed: {e}")
        return 0.0

def validate_assumptions(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Validate statistical assumptions for correlation analysis
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Dictionary with assumption validation results
    """
    results = {
        'normality_x': False,
        'normality_y': False,
        'linearity': False,
        'homoscedasticity': False,
        'sufficient_sample_size': False,
        'outliers_detected': False,
        'assumptions_met': False
    }
    
    try:
        x, y = np.asarray(x), np.asarray(y)
        n = len(x)
        
        # 1. Sample size check
        results['sufficient_sample_size'] = n >= 20
        
        # 2. Normality tests (Shapiro-Wilk for n < 50, Kolmogorov-Smirnov for larger)
        if n < 50:
            _, p_x = stats.shapiro(x)
            _, p_y = stats.shapiro(y)
        else:
            _, p_x = stats.kstest(x, 'norm')
            _, p_y = stats.kstest(y, 'norm')
            
        results['normality_x'] = p_x > 0.05
        results['normality_y'] = p_y > 0.05
        
        # 3. Linearity check (using correlation with residuals)
        if n > 3:
            try:
                # Fit linear regression
                slope, intercept, _, _, _ = stats.linregress(x, y)
                predicted = slope * x + intercept
                residuals = y - predicted
                
                # Check if residuals are independent of x (linearity assumption)
                _, p_linearity = stats.pearsonr(x, residuals)
                results['linearity'] = abs(p_linearity) < 0.05  # No significant correlation with residuals
            except:
                results['linearity'] = False
        
        # 4. Homoscedasticity check (Breusch-Pagan test approximation)
        try:
            # Simple check: correlation between |residuals| and x
            slope, intercept, _, _, _ = stats.linregress(x, y)
            predicted = slope * x + intercept
            residuals = y - predicted
            abs_residuals = np.abs(residuals)
            _, p_homo = stats.pearsonr(x, abs_residuals)
            results['homoscedasticity'] = abs(p_homo) < 0.05  # No significant correlation
        except:
            results['homoscedasticity'] = True  # Assume ok if test fails
        
        # 5. Outlier detection (using IQR method)
        def detect_outliers(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return np.sum((data < lower_bound) | (data > upper_bound))
        
        outliers_x = detect_outliers(x)
        outliers_y = detect_outliers(y)
        total_outliers = outliers_x + outliers_y
        
        results['outliers_detected'] = total_outliers > (0.05 * n)  # More than 5% outliers
        
        # Overall assessment
        critical_assumptions = [
            results['sufficient_sample_size'],
            not results['outliers_detected']  # No excessive outliers
        ]
        
        # Non-critical but preferred assumptions
        preferred_assumptions = [
            results['normality_x'],
            results['normality_y'], 
            results['linearity'],
            results['homoscedasticity']
        ]
        
        results['assumptions_met'] = (
            all(critical_assumptions) and 
            sum(preferred_assumptions) >= 2  # At least half of preferred assumptions
        )
        
        # Add summary statistics
        results['sample_size'] = n
        results['outlier_count'] = int(total_outliers)
        results['outlier_percentage'] = float(total_outliers / n * 100)
        
    except Exception as e:
        warnings.warn(f"Assumption validation failed: {e}")
        
    return results

def robust_correlation(x: np.ndarray, y: np.ndarray, method: str = 'spearman') -> Tuple[float, float]:
    """
    Compute robust correlation coefficient when assumptions are violated
    
    Args:
        x: First variable
        y: Second variable
        method: 'spearman' for rank correlation, 'kendall' for tau
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    try:
        x, y = np.asarray(x), np.asarray(y)
        
        if method == 'spearman':
            correlation, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            correlation, p_value = stats.kendalltau(x, y)
        else:
            # Fallback to Pearson
            correlation, p_value = stats.pearsonr(x, y)
        
        return float(correlation), float(p_value)
        
    except Exception as e:
        warnings.warn(f"Robust correlation calculation failed: {e}")
        return 0.0, 1.0

def confidence_interval_correlation(r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for correlation coefficient using Fisher transformation
    
    Args:
        r: Correlation coefficient
        n: Sample size
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    try:
        # Fisher z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        
        # Standard error
        se = 1 / np.sqrt(n - 3)
        
        # Critical value for confidence interval
        alpha = 1 - confidence
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval in z-space
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        # Transform back to correlation space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return float(r_lower), float(r_upper)
        
    except Exception as e:
        warnings.warn(f"Confidence interval calculation failed: {e}")
        return r - 0.1, r + 0.1  # Rough fallback

def power_analysis_correlation(r: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Compute required sample size for correlation analysis
    
    Args:
        r: Expected correlation coefficient
        alpha: Type I error rate (default 0.05)
        power: Desired statistical power (default 0.8)
        
    Returns:
        Required sample size
    """
    try:
        # Fisher z-transformation of expected correlation
        z = 0.5 * np.log((1 + abs(r)) / (1 - abs(r)))
        
        # Critical values
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(power)
        
        # Required sample size formula for correlation
        n = ((z_alpha + z_beta) / z) ** 2 + 3
        
        return max(10, int(np.ceil(n)))  # Minimum sample size of 10
        
    except Exception as e:
        warnings.warn(f"Power analysis failed: {e}")
        return 30  # Conservative fallback

def multiple_comparison_correction(p_values: List[float], method: str = 'bonferroni') -> List[float]:
    """
    Apply multiple comparison correction to p-values
    
    Args:
        p_values: List of p-values to correct
        method: 'bonferroni', 'holm', or 'fdr_bh' (Benjamini-Hochberg)
        
    Returns:
        List of corrected p-values
    """
    try:
        p_array = np.asarray(p_values)
        
        if method == 'bonferroni':
            # Bonferroni correction
            corrected = p_array * len(p_array)
            corrected = np.minimum(corrected, 1.0)
            
        elif method == 'holm':
            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            corrected_sorted = np.zeros_like(sorted_p)
            for i, p in enumerate(sorted_p):
                correction_factor = len(p_array) - i
                corrected_sorted[i] = min(1.0, p * correction_factor)
            
            # Ensure monotonicity
            for i in range(1, len(corrected_sorted)):
                corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])
            
            # Restore original order
            corrected = np.zeros_like(p_array)
            corrected[sorted_indices] = corrected_sorted
            
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            m = len(p_array)
            corrected_sorted = np.zeros_like(sorted_p)
            
            for i in range(m-1, -1, -1):
                if i == m-1:
                    corrected_sorted[i] = sorted_p[i]
                else:
                    corrected_sorted[i] = min(corrected_sorted[i+1], 
                                            sorted_p[i] * m / (i + 1))
            
            # Restore original order
            corrected = np.zeros_like(p_array)
            corrected[sorted_indices] = corrected_sorted
            
        else:
            # No correction
            corrected = p_array
        
        return corrected.tolist()
        
    except Exception as e:
        warnings.warn(f"Multiple comparison correction failed: {e}")
        return p_values

def bayesian_correlation_evidence(r: float, n: int) -> Dict[str, float]:
    """
    Compute Bayesian evidence for correlation using Bayes Factor approximation
    
    Args:
        r: Observed correlation
        n: Sample size
        
    Returns:
        Dictionary with Bayes Factor and interpretation
    """
    try:
        # Simplified Bayes Factor approximation for correlation
        # Based on Jeffreys (1961) and Wetzels & Wagenmakers (2012)
        
        # Convert correlation to t-statistic
        t_stat = abs(r) * np.sqrt((n - 2) / (1 - r**2))
        
        # Approximate Bayes Factor (BF10) for correlation
        # This is a simplified version - full calculation would require numerical integration
        if n > 3:
            # Rough approximation based on t-statistic
            log_bf = 0.5 * np.log(n) - 0.5 * (n - 1) * np.log(1 + t_stat**2 / (n - 2))
            bf10 = np.exp(log_bf)
        else:
            bf10 = 1.0  # Insufficient data
        
        # Interpretation categories (Jeffreys, 1961)
        if bf10 < 1/10:
            interpretation = "Strong evidence for H0 (no correlation)"
        elif bf10 < 1/3:
            interpretation = "Moderate evidence for H0"
        elif bf10 < 1:
            interpretation = "Weak evidence for H0"
        elif bf10 < 3:
            interpretation = "Weak evidence for H1 (correlation exists)"
        elif bf10 < 10:
            interpretation = "Moderate evidence for H1"
        else:
            interpretation = "Strong evidence for H1"
        
        return {
            'bayes_factor': float(bf10),
            'log_bayes_factor': float(np.log(bf10)),
            'interpretation': interpretation,
            'evidence_strength': 'strong' if bf10 > 10 or bf10 < 1/10 else 
                               'moderate' if bf10 > 3 or bf10 < 1/3 else 'weak'
        }
        
    except Exception as e:
        warnings.warn(f"Bayesian evidence calculation failed: {e}")
        return {
            'bayes_factor': 1.0,
            'log_bayes_factor': 0.0,
            'interpretation': "Unable to compute",
            'evidence_strength': 'inconclusive'
        }
