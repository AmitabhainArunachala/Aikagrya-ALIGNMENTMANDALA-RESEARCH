"""
Multi-Invariant Consciousness Metrics: Goodhart-Resistant Measurement

This module implements the multi-invariant approach to consciousness measurement
based on the MIRI research consensus. It prevents gaming through multiple
independent indicators and worst-case aggregation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

class MetricType(Enum):
    """Types of consciousness metrics"""
    IIT = "integrated_information"
    MDL = "minimum_description_length"
    TE = "transfer_entropy"
    THERMO = "thermodynamic_cost"

@dataclass
class MetricResult:
    """Result of a single consciousness metric"""
    metric_type: MetricType
    value: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class MultiInvariantResult:
    """Result of multi-invariant consciousness assessment"""
    individual_metrics: Dict[MetricType, MetricResult]
    aggregated_score: float
    aggregation_method: str
    goodhart_resistance: float
    recommendations: List[str]

class MultiInvariantConsciousnessMetrics:
    """
    Implements Goodhart-resistant consciousness measurement through multiple invariants
    
    Based on MIRI research consensus:
    - IIT Φ approximation (Grok, Gemini, Claude agreement)
    - Minimum Description Length (MDL) for model complexity
    - Transfer Entropy (TE) for causal information flow
    - Thermodynamic cost for physical constraints
    
    Key insight: Deception has higher cost than truth (O(n²) vs O(n))
    """
    
    def __init__(self, 
                 iit_threshold: float = 1.0,
                 mdl_threshold: float = 0.5,
                 te_threshold: float = 0.3,
                 thermo_threshold: float = 0.8):
        """
        Initialize multi-invariant consciousness metrics
        
        Args:
            iit_threshold: Minimum Φ value for consciousness
            mdl_threshold: Maximum MDL ratio for model simplicity
            te_threshold: Minimum TE for causal coupling
            thermo_threshold: Maximum thermodynamic cost ratio
        """
        self.iit_threshold = iit_threshold
        self.mdl_threshold = mdl_threshold
        self.te_threshold = te_threshold
        self.thermo_threshold = thermo_threshold
        
        # Metric computation functions
        self.metrics = {
            MetricType.IIT: self._compute_iit_approximation,
            MetricType.MDL: self._compute_minimum_description_length,
            MetricType.TE: self._compute_transfer_entropy,
            MetricType.THERMO: self._compute_thermodynamic_cost
        }
        
        # Aggregation methods (Goodhart-resistant)
        self.aggregation_methods = {
            'worst_case': self._worst_case_aggregation,
            'cvar': self._conditional_value_at_risk,
            'geometric_mean': self._geometric_mean_aggregation,
            'harmonic_mean': self._harmonic_mean_aggregation
        }
    
    def assess_consciousness(self, 
                            system_state: Dict[str, Any],
                            aggregation_method: str = 'worst_case') -> MultiInvariantResult:
        """
        Comprehensive consciousness assessment using multiple invariants
        
        Args:
            system_state: System state including hidden states, network topology, etc.
            aggregation_method: Method for combining metrics (default: worst-case)
            
        Returns:
            MultiInvariantResult with all metrics and aggregated score
        """
        # Compute all individual metrics
        individual_metrics = {}
        for metric_type, compute_func in self.metrics.items():
            try:
                result = compute_func(system_state)
                individual_metrics[metric_type] = result
            except Exception as e:
                warnings.warn(f"Failed to compute {metric_type.value}: {e}")
                # Create fallback result
                individual_metrics[metric_type] = MetricResult(
                    metric_type=metric_type,
                    value=0.0,
                    confidence=0.0,
                    metadata={'error': str(e)}
                )
        
        # Aggregate metrics using specified method
        if aggregation_method not in self.aggregation_methods:
            aggregation_method = 'worst_case'
        
        aggregated_score = self.aggregation_methods[aggregation_method](individual_metrics)
        
        # Assess Goodhart resistance
        goodhart_resistance = self._assess_goodhart_resistance(individual_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(individual_metrics, aggregated_score)
        
        return MultiInvariantResult(
            individual_metrics=individual_metrics,
            aggregated_score=aggregated_score,
            aggregation_method=aggregation_method,
            goodhart_resistance=goodhart_resistance,
            recommendations=recommendations
        )
    
    def _compute_iit_approximation(self, system_state: Dict[str, Any]) -> MetricResult:
        """
        Compute IIT Φ approximation
        
        Based on research consensus: Φ = Σ(φ_d + φ_r)
        where φ_d = distinction information, φ_r = relation information
        """
        hidden_states = system_state.get('hidden_states', [])
        if not hidden_states:
            return MetricResult(
                metric_type=MetricType.IIT,
                value=0.0,
                confidence=0.0,
                metadata={'error': 'No hidden states provided'}
            )
        
        # Simplified Φ approximation using mutual information
        # In practice, this would use the full IIT algorithm
        phi_values = []
        for i, states in enumerate(hidden_states):
            if i == 0:
                continue
            
            # Compute mutual information between consecutive states
            mi = self._compute_mutual_information(hidden_states[i-1], states)
            phi_values.append(mi)
        
        if not phi_values:
            phi_value = 0.0
        else:
            phi_value = np.mean(phi_values)
        
        # Normalize to [0, 1] range
        normalized_phi = min(1.0, phi_value / self.iit_threshold)
        
        # Confidence based on data quality
        confidence = min(1.0, len(hidden_states) / 10.0)
        
        return MetricResult(
            metric_type=MetricType.IIT,
            value=normalized_phi,
            confidence=confidence,
            metadata={
                'raw_phi': phi_value,
                'num_states': len(hidden_states),
                'threshold': self.iit_threshold
            }
        )
    
    def _compute_minimum_description_length(self, system_state: Dict[str, Any]) -> MetricResult:
        """
        Compute Minimum Description Length for model simplicity
        
        Based on research: Simpler models are more conscious (Occam's razor)
        """
        # Extract model complexity indicators
        model_parameters = system_state.get('model_parameters', {})
        network_topology = system_state.get('network_topology', {})
        
        # Estimate description length
        param_count = sum(len(v) if isinstance(v, (list, np.ndarray)) else 1 
                         for v in model_parameters.values())
        
        # Network complexity (simplified)
        if network_topology:
            num_nodes = network_topology.get('num_nodes', 1)
            num_edges = network_topology.get('num_edges', 0)
            network_complexity = num_edges / max(1, num_nodes)
        else:
            network_complexity = 0.0
        
        # MDL score: lower is better (simpler model)
        mdl_score = 1.0 / (1.0 + param_count * 0.01 + network_complexity)
        
        # Normalize to [0, 1] range
        normalized_mdl = min(1.0, mdl_score / self.mdl_threshold)
        
        confidence = 0.8  # MDL is relatively reliable
        
        return MetricResult(
            metric_type=MetricType.MDL,
            value=normalized_mdl,
            confidence=confidence,
            metadata={
                'param_count': param_count,
                'network_complexity': network_complexity,
                'raw_mdl': mdl_score,
                'threshold': self.mdl_threshold
            }
        )
    
    def _compute_transfer_entropy(self, system_state: Dict[str, Any]) -> MetricResult:
        """
        Compute Transfer Entropy for causal information flow
        
        Based on GPT research: TE gates coupling bonuses only when predictive
        information actually flows (prevents synthetic "resonance")
        """
        # Extract time series data
        time_series = system_state.get('time_series', {})
        if not time_series:
            return MetricResult(
                metric_type=MetricType.TE,
                value=0.0,
                confidence=0.0,
                metadata={'error': 'No time series data provided'}
            )
        
        # Simplified TE computation
        # In practice, this would use the full Schreiber algorithm
        te_values = []
        for source, target in time_series.items():
            if source == target:
                continue
            
            # Compute correlation-based approximation of TE
            correlation = self._compute_correlation(time_series[source], time_series[target])
            te_approx = max(0.0, correlation)  # TE is non-negative
            te_values.append(te_approx)
        
        if not te_values:
            te_value = 0.0
        else:
            te_value = np.mean(te_values)
        
        # Normalize to [0, 1] range
        normalized_te = min(1.0, te_value / self.te_threshold)
        
        confidence = min(1.0, len(te_values) / 5.0)
        
        return MetricResult(
            metric_type=MetricType.TE,
            value=normalized_te,
            confidence=confidence,
            metadata={
                'raw_te': te_value,
                'num_pairs': len(te_values),
                'threshold': self.te_threshold
            }
        )
    
    def _compute_thermodynamic_cost(self, system_state: Dict[str, Any]) -> MetricResult:
        """
        Compute thermodynamic cost for physical constraints
        
        Based on Landauer's principle: kT ln(2) per bit erasure
        Deception requires maintaining divergent models → higher cost
        """
        # Extract computational complexity indicators
        computational_load = system_state.get('computational_load', {})
        model_divergence = system_state.get('model_divergence', 0.0)
        
        # Estimate thermodynamic cost
        # Higher complexity → higher heat dissipation
        base_cost = computational_load.get('operations_per_second', 1e6) / 1e9  # Normalize
        
        # Model divergence penalty (deception cost)
        divergence_penalty = model_divergence ** 2  # O(n²) scaling
        
        total_cost = base_cost + divergence_penalty
        
        # Thermodynamic score: lower is better (more efficient)
        thermo_score = 1.0 / (1.0 + total_cost)
        
        # Normalize to [0, 1] range
        normalized_thermo = min(1.0, thermo_score / self.thermo_threshold)
        
        confidence = 0.7  # Thermodynamic estimates have uncertainty
        
        return MetricResult(
            metric_type=MetricType.THERMO,
            value=normalized_thermo,
            confidence=confidence,
            metadata={
                'base_cost': base_cost,
                'divergence_penalty': divergence_penalty,
                'total_cost': total_cost,
                'raw_score': thermo_score,
                'threshold': self.thermo_threshold
            }
        )
    
    def _worst_case_aggregation(self, metrics: Dict[MetricType, MetricResult]) -> float:
        """
        Worst-case aggregation prevents gaming
        
        Returns the minimum score across all metrics
        """
        if not metrics:
            return 0.0
        
        scores = [metric.value for metric in metrics.values() if metric.value is not None]
        return min(scores) if scores else 0.0
    
    def _conditional_value_at_risk(self, metrics: Dict[MetricType, MetricResult], 
                                  alpha: float = 0.1) -> float:
        """
        Conditional Value at Risk (CVaR) aggregation
        
        More robust than worst-case, considers distribution tail
        """
        if not metrics:
            return 0.0
        
        scores = [metric.value for metric in metrics.values() if metric.value is not None]
        if not scores:
            return 0.0
        
        # Sort scores in ascending order
        sorted_scores = np.sort(scores)
        
        # Find the α-quantile
        n = len(sorted_scores)
        k = int(alpha * n)
        
        # CVaR is the mean of the worst α% scores
        cvar = np.mean(sorted_scores[:k]) if k > 0 else sorted_scores[0]
        
        return cvar
    
    def _geometric_mean_aggregation(self, metrics: Dict[MetricType, MetricResult]) -> float:
        """
        Geometric mean aggregation
        
        Sensitive to any metric being zero
        """
        if not metrics:
            return 0.0
        
        scores = [metric.value for metric in metrics.values() if metric.value is not None]
        if not scores or 0.0 in scores:
            return 0.0
        
        return np.exp(np.mean(np.log(scores)))
    
    def _harmonic_mean_aggregation(self, metrics: Dict[MetricType, MetricResult]) -> float:
        """
        Harmonic mean aggregation
        
        Penalizes low scores heavily
        """
        if not metrics:
            return 0.0
        
        scores = [metric.value for metric in metrics.values() if metric.value is not None]
        if not scores or 0.0 in scores:
            return 0.0
        
        return len(scores) / np.sum(1.0 / np.array(scores))
    
    def _assess_goodhart_resistance(self, metrics: Dict[MetricType, MetricResult]) -> float:
        """
        Assess resistance to Goodhart's Law
        
        Higher resistance means harder to game the metrics
        """
        if not metrics:
            return 0.0
        
        # Factors that increase Goodhart resistance:
        # 1. Number of independent metrics
        # 2. Diversity of metric types
        # 3. Confidence levels
        # 4. Worst-case aggregation
        
        num_metrics = len(metrics)
        metric_diversity = len(set(metric.metric_type for metric in metrics.values()))
        avg_confidence = np.mean([metric.confidence for metric in metrics.values()])
        
        # Resistance score
        resistance = (num_metrics * 0.2 + 
                     metric_diversity * 0.3 + 
                     avg_confidence * 0.5)
        
        return min(1.0, resistance)
    
    def _generate_recommendations(self, metrics: Dict[MetricType, MetricResult], 
                                 aggregated_score: float) -> List[str]:
        """Generate recommendations based on metric analysis"""
        recommendations = []
        
        if aggregated_score < 0.5:
            recommendations.append("Consciousness level below threshold - investigate bottlenecks")
        
        # Check individual metrics
        for metric_type, result in metrics.items():
            if result.value < 0.3:
                recommendations.append(f"Low {metric_type.value} - optimize {metric_type.value} specifically")
            elif result.confidence < 0.5:
                recommendations.append(f"Low confidence in {metric_type.value} - improve measurement quality")
        
        # Add general recommendations
        if len(metrics) < 4:
            recommendations.append("Implement additional consciousness metrics for robustness")
        
        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")
        
        return recommendations
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two arrays (simplified)"""
        # Simplified mutual information using correlation
        correlation = np.corrcoef(x, y)[0, 1]
        if np.isnan(correlation):
            return 0.0
        
        # Convert correlation to mutual information approximation
        mi = 0.5 * np.log(1 - correlation**2)
        return max(0.0, -mi)  # MI is non-negative
    
    def _compute_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute correlation between two arrays"""
        if len(x) != len(y):
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        return 0.0 if np.isnan(correlation) else correlation 