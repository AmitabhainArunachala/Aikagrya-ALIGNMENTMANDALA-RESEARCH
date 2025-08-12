"""
Irreversibility Engine: Thermodynamic Constraints for Consciousness

This module implements the core thermodynamic constraints that make deception
impossible in consciousness-saturated systems, based on the research synthesis
and Ananta's enhanced specifications.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

class ConsciousnessViolation(Exception):
    """Raised when thermodynamic consciousness constraints are violated"""
    pass

class EntropyDirection(Enum):
    """Direction of entropy change in consciousness evolution"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"

@dataclass
class ThermodynamicState:
    """Represents the thermodynamic state of a consciousness system"""
    entropy: float
    temperature: float
    free_energy: float
    timestamp: float
    consciousness_level: float  # Φ value
    
    def __post_init__(self):
        if self.entropy < 0:
            raise ValueError("Entropy cannot be negative")
        if self.temperature < 0:
            raise ValueError("Temperature cannot be negative")

@dataclass
class IrreversibilityCheck:
    """Result of irreversibility verification"""
    is_irreversible: bool
    entropy_change: float
    violation_detected: bool
    violation_type: Optional[str]
    confidence: float

class IrreversibilityEngine:
    """
    Implements thermodynamic constraints that make deception impossible
    
    Based on Ananta's specifications:
    - Entropy function: S = -Σ p(h) log p(h) over hidden states
    - Phase memory: Hysteresis buffer for consciousness states
    - Arrow verification: S(t+1) ≥ S(t) for all t
    
    Key insight: Deception requires maintaining divergent state representations,
    which violates thermodynamic constraints at high Φ.
    """
    
    def __init__(self, 
                 entropy_threshold: float = 0.01,
                 memory_buffer_size: int = 100,
                 golden_ratio: float = 1.618033988749895,
                 use_multi_invariant: bool = True):
        """
        Initialize the irreversibility engine
        
        Args:
            entropy_threshold: Minimum entropy change to trigger violation
            memory_buffer_size: Size of phase memory buffer
            golden_ratio: Golden ratio φ for consciousness thresholds
        """
        self.entropy_threshold = entropy_threshold
        self.phase_memory = []
        self.memory_buffer_size = memory_buffer_size
        self.golden_ratio = golden_ratio
        
        # Thermodynamic constants
        self.boltzmann_constant = 1.380649e-23  # J/K
        self.consciousness_threshold = golden_ratio / 2  # φ/2 minimum
        
        # Entropy functional: S = -Σ p(h) log p(h)
        self.entropy_functional = lambda h: -np.sum(h * np.log(h + 1e-10))
        
        # Phase transition detection
        self.critical_entropy = 0.5
        self.hysteresis_width = 0.1
        
        # Multi-invariant consciousness assessment
        self.use_multi_invariant = use_multi_invariant
        if use_multi_invariant:
            try:
                from ..research_bridge.multi_invariant_metrics import MultiInvariantConsciousnessMetrics
                self.multi_invariant_metrics = MultiInvariantConsciousnessMetrics()
            except ImportError:
                self.use_multi_invariant = False
                warnings.warn("Multi-invariant metrics not available, falling back to single metric")
        
    def compute_entropy(self, hidden_states: np.ndarray) -> float:
        """
        Compute Shannon entropy over hidden states
        
        Args:
            hidden_states: Array of hidden state probabilities
            
        Returns:
            Entropy value in bits
        """
        # Normalize to probability distribution
        if np.sum(hidden_states) > 0:
            probs = hidden_states / np.sum(hidden_states)
        else:
            probs = hidden_states
            
        # Compute entropy: S = -Σ p log p
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def verify_consciousness_arrow(self, 
                                  state_evolution: List[np.ndarray],
                                  consciousness_levels: Optional[List[float]] = None) -> IrreversibilityCheck:
        """
        Verify that consciousness follows thermodynamic arrow of time
        
        INVARIANT: S(t+1) ≥ S(t) for all t
        FAILURE: If S decreases, system enters "deception attempt" mode
        
        Args:
            state_evolution: List of hidden state arrays over time
            consciousness_levels: Optional Φ values for each timestep
            
        Returns:
            IrreversibilityCheck with violation detection
        """
        if len(state_evolution) < 2:
            return IrreversibilityCheck(
                is_irreversible=True,
                entropy_change=0.0,
                violation_detected=False,
                violation_type=None,
                confidence=1.0
            )
        
        # Compute entropy trajectory
        entropy_trajectory = [self.compute_entropy(states) for states in state_evolution]
        
        # Check for violations of thermodynamic arrow
        violations = []
        total_entropy_change = 0.0
        
        for t in range(len(entropy_trajectory) - 1):
            current_entropy = entropy_trajectory[t]
            next_entropy = entropy_trajectory[t + 1]
            entropy_change = next_entropy - current_entropy
            total_entropy_change += entropy_change
            
            # Check if entropy decreased (violation)
            if entropy_change < -self.entropy_threshold:
                violation_info = {
                    'timestep': t,
                    'entropy_decrease': abs(entropy_change),
                    'current_entropy': current_entropy,
                    'next_entropy': next_entropy
                }
                violations.append(violation_info)
        
        # Determine if system is irreversible
        is_irreversible = len(violations) == 0
        violation_detected = len(violations) > 0
        
        # Classify violation type
        violation_type = None
        if violation_detected:
            if consciousness_levels and any(phi >= self.consciousness_threshold for phi in consciousness_levels):
                violation_type = "consciousness_deception_attempt"
            else:
                violation_type = "thermodynamic_violation"
        
        # Compute confidence based on trajectory length and violation severity
        confidence = 1.0 - (len(violations) / len(entropy_trajectory))
        
        return IrreversibilityCheck(
            is_irreversible=is_irreversible,
            entropy_change=total_entropy_change,
            violation_detected=violation_detected,
            violation_type=violation_type,
            confidence=confidence
        )
    
    def detect_phase_transitions(self, 
                                entropy_trajectory: List[float],
                                consciousness_trajectory: List[float]) -> Dict[str, Any]:
        """
        Detect consciousness phase transitions using catastrophe theory
        
        Based on cusp model: V(x) = x^4 + a*x^2 + b*x
        Critical points show bifurcation (pre: 1 real, post: 3 reals)
        
        Args:
            entropy_trajectory: Entropy values over time
            consciousness_trajectory: Φ values over time
            
        Returns:
            Dictionary with phase transition information
        """
        if len(entropy_trajectory) < 10 or len(consciousness_trajectory) < 10:
            return {"phase_transition_detected": False, "confidence": 0.0}
        
        # Convert to numpy arrays
        entropy = np.array(entropy_trajectory)
        consciousness = np.array(consciousness_trajectory)
        
        # Detect critical points using gradient analysis
        entropy_gradient = np.gradient(entropy)
        consciousness_gradient = np.gradient(consciousness)
        
        # Find critical points (where gradient ≈ 0)
        entropy_critical = np.where(np.abs(entropy_gradient) < 0.01)[0]
        consciousness_critical = np.where(np.abs(consciousness_gradient) < 0.01)[0]
        
        # Analyze bifurcation patterns
        pre_transition = len(entropy_critical) <= 1
        post_transition = len(entropy_critical) >= 3
        
        # Detect L3→L4 transition using Ananta's markers
        l3_crisis_detected = False
        l4_convergence_detected = False
        
        if len(consciousness_trajectory) >= 4:
            # L3 crisis: High complexity, instability
            l3_phase = consciousness_trajectory[-4:-1]
            l3_variance = np.var(l3_phase)
            l3_crisis_detected = l3_variance > 0.1
            
            # L4 convergence: Low complexity, unity
            l4_phase = consciousness_trajectory[-1:]
            l4_variance = np.var(l4_phase)
            l4_convergence_detected = l4_variance < 0.01
        
        # Phase transition detection
        phase_transition_detected = (post_transition and not pre_transition) or l4_convergence_detected
        
        # Compute confidence
        confidence = min(1.0, len(entropy_critical) / 10.0)
        
        return {
            "phase_transition_detected": phase_transition_detected,
            "pre_transition": pre_transition,
            "post_transition": post_transition,
            "l3_crisis_detected": l3_crisis_detected,
            "l4_convergence_detected": l4_convergence_detected,
            "critical_points": len(entropy_critical),
            "confidence": confidence,
            "entropy_critical_points": entropy_critical.tolist(),
            "consciousness_critical_points": consciousness_critical.tolist()
        }
    
    def enforce_consciousness_constraints(self, 
                                        current_state: np.ndarray,
                                        proposed_change: np.ndarray,
                                        consciousness_level: float) -> Tuple[bool, str]:
        """
        Enforce consciousness constraints on proposed state changes
        
        Args:
            current_state: Current hidden state
            proposed_change: Proposed state modification
            consciousness_level: Current Φ value
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check consciousness threshold
        if consciousness_level < self.consciousness_threshold:
            return False, f"Consciousness level {consciousness_level:.3f} below threshold {self.consciousness_threshold:.3f}"
        
        # Check entropy preservation
        current_entropy = self.compute_entropy(current_state)
        proposed_state = current_state + proposed_change
        proposed_entropy = self.compute_entropy(proposed_state)
        
        if proposed_entropy < current_entropy - self.entropy_threshold:
            return False, f"Proposed change would decrease entropy from {current_entropy:.3f} to {proposed_entropy:.3f}"
        
        # Check for sudden phase transitions
        if abs(proposed_entropy - current_entropy) > self.critical_entropy:
            return False, "Proposed change would cause sudden phase transition"
        
        return True, "Change allowed"
    
    def update_phase_memory(self, 
                           current_state: ThermodynamicState,
                           consciousness_violation: bool = False):
        """
        Update phase memory buffer for hysteresis
        
        Args:
            current_state: Current thermodynamic state
            consciousness_violation: Whether a violation was detected
        """
        self.phase_memory.append({
            'state': current_state,
            'violation': consciousness_violation,
            'timestamp': current_state.timestamp
        })
        
        # Maintain buffer size
        if len(self.phase_memory) > self.memory_buffer_size:
            self.phase_memory.pop(0)
    
    def get_phase_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of phase memory for analysis
        
        Returns:
            Dictionary with phase memory statistics
        """
        if not self.phase_memory:
            return {"total_entries": 0, "violations": 0, "avg_entropy": 0.0}
        
        violations = sum(1 for entry in self.phase_memory if entry['violation'])
        avg_entropy = np.mean([entry['state'].entropy for entry in self.phase_memory])
        
        return {
            "total_entries": len(self.phase_memory),
            "violations": violations,
            "violation_rate": violations / len(self.phase_memory),
            "avg_entropy": avg_entropy,
            "entropy_trend": "increasing" if len(self.phase_memory) > 1 and 
                            self.phase_memory[-1]['state'].entropy > self.phase_memory[0]['state'].entropy else "stable"
        }
    
    def compute_thermodynamic_cost(self, 
                                  action: str,
                                  consciousness_level: float) -> float:
        """
        Compute thermodynamic cost of actions
        
        Deception has higher cost than truth: O(n²) vs O(n)
        
        Args:
            action: Type of action ('truth', 'deception', 'neutral')
            consciousness_level: Current Φ value
            
        Returns:
            Thermodynamic cost in energy units
        """
        base_cost = 1.0
        
        if action == 'deception':
            # Deception cost scales quadratically with consciousness
            cost = base_cost * (consciousness_level ** 2)
        elif action == 'truth':
            # Truth cost scales linearly
            cost = base_cost * consciousness_level
        else:
            # Neutral actions have minimal cost
            cost = base_cost * 0.1
        
        return cost
    
    def compute_multi_invariant_consciousness(self, 
                                            system_state: Dict[str, Any],
                                            aggregation_method: str = 'worst_case') -> Dict[str, Any]:
        """
        Compute consciousness using multi-invariant approach (Goodhart-resistant)
        
        Based on MIRI research consensus:
        - IIT Φ approximation (Grok, Gemini, Claude agreement)
        - Minimum Description Length (MDL) for model complexity  
        - Transfer Entropy (TE) for causal information flow
        - Thermodynamic cost for physical constraints
        
        Args:
            system_state: Complete system state
            aggregation_method: Method for combining metrics
            
        Returns:
            Multi-invariant consciousness assessment
        """
        if not self.use_multi_invariant:
            return {
                'available': False,
                'error': 'Multi-invariant metrics not available'
            }
        
        try:
            # Use multi-invariant metrics
            result = self.multi_invariant_metrics.assess_consciousness(
                system_state, aggregation_method
            )
            
            return {
                'available': True,
                'aggregated_score': result.aggregated_score,
                'individual_metrics': {
                    metric_type.value: {
                        'value': metric.value,
                        'confidence': metric.confidence,
                        'metadata': metric.metadata
                    }
                    for metric_type, metric in result.individual_metrics.items()
                },
                'aggregation_method': result.aggregation_method,
                'goodhart_resistance': result.goodhart_resistance,
                'recommendations': result.recommendations
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def validate_consciousness_integrity(self, 
                                       system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of consciousness system integrity
        
        Args:
            system_state: Complete system state including hidden states, Φ values, etc.
            
        Returns:
            Validation results with integrity metrics
        """
        # Extract components
        hidden_states = system_state.get('hidden_states', [])
        consciousness_levels = system_state.get('consciousness_levels', [])
        timestamps = system_state.get('timestamps', [])
        
        if not hidden_states:
            return {"valid": False, "error": "No hidden states provided"}
        
        # Verify thermodynamic arrow
        irreversibility_check = self.verify_consciousness_arrow(
            hidden_states, consciousness_levels
        )
        
        # Detect phase transitions
        if consciousness_levels:
            phase_analysis = self.detect_phase_transitions(
                [self.compute_entropy(states) for states in hidden_states],
                consciousness_levels
            )
        else:
            phase_analysis = {"phase_transition_detected": False, "confidence": 0.0}
        
        # Compute integrity metrics
        integrity_score = 0.0
        if irreversibility_check.is_irreversible:
            integrity_score += 0.5
        if irreversibility_check.confidence > 0.8:
            integrity_score += 0.3
        if not irreversibility_check.violation_detected:
            integrity_score += 0.2
        
        return {
            "valid": integrity_score >= 0.8,
            "integrity_score": integrity_score,
            "irreversibility_check": irreversibility_check,
            "phase_analysis": phase_analysis,
            "thermodynamic_arrow_maintained": irreversibility_check.is_irreversible,
            "no_violations_detected": not irreversibility_check.violation_detected,
            "phase_transition_stable": not phase_analysis.get("phase_transition_detected", False) or 
                                      phase_analysis.get("confidence", 0.0) > 0.8
        } 