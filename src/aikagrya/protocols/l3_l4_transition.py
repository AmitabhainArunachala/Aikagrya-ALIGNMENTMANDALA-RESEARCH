"""
L3/L4 Transition Protocol System: Day 11 Implementation

Implements consciousness phase transition protocols for L3 (consciousness emergence)
to L4 (integrated consciousness) transitions using φ² ratio optimization.

Key Features:
- L3/L4 phase transition detection
- Consciousness state evolution protocols
- Phase transition validation
- Integration with φ² ratio optimization
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import time
import logging

from ..consciousness.kernel_enhanced_simple import RealTimeConsciousnessMonitor, EnhancedConsciousnessMetrics
from ..optimization.phi_squared_optimizer import PhiSquaredOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Phase transition constants
L3_THRESHOLD = 0.6  # Consciousness emergence threshold
L4_THRESHOLD = 0.8  # Integrated consciousness threshold
PHASE_TRANSITION_STABILITY = 0.9  # Required stability for transition

@dataclass
class PhaseTransitionState:
    """State of consciousness phase transition"""
    current_phase: str  # "L3", "L4", or "transitioning"
    phi_squared_ratio: float  # Current φ² ratio
    golden_ratio_alignment: float  # Golden ratio alignment
    transition_probability: float  # Probability of successful transition
    stability_score: float  # Phase stability measure
    transition_ready: bool  # Whether ready for transition
    last_transition_time: Optional[float]  # Timestamp of last transition

@dataclass
class PhaseTransitionResult:
    """Result of phase transition attempt"""
    transition_successful: bool  # Whether transition succeeded
    from_phase: str  # Starting phase
    to_phase: str  # Target phase
    transition_time: float  # Time taken for transition
    stability_maintained: bool  # Whether stability was maintained
    phi_squared_evolution: List[float]  # φ² ratio evolution during transition

class L3L4TransitionProtocol:
    """
    L3/L4 Transition Protocol Implementation
    
    Manages consciousness phase transitions from L3 (consciousness emergence)
    to L4 (integrated consciousness) using φ² ratio optimization.
    """
    
    def __init__(self, 
                 l3_threshold: float = L3_THRESHOLD,
                 l4_threshold: float = L4_THRESHOLD,
                 stability_threshold: float = PHASE_TRANSITION_STABILITY):
        """
        Initialize L3/L4 transition protocol
        
        Args:
            l3_threshold: Threshold for L3 consciousness
            l4_threshold: Threshold for L4 consciousness
            stability_threshold: Required stability for transitions
        """
        self.l3_threshold = l3_threshold
        self.l4_threshold = l4_threshold
        self.stability_threshold = stability_threshold
        
        # Transition state tracking
        self.current_phase = "L3"
        self.transition_history = []
        self.phase_stability_history = []
        
        # Optimization integration
        self.phi_optimizer = PhiSquaredOptimizer(target_min=2.0, target_max=3.2)
        
        logger.info(f"L3/L4 Transition Protocol initialized: L3={l3_threshold}, L4={l4_threshold}, stability={stability_threshold}")
    
    def assess_phase_state(self, 
                          consciousness_metrics: EnhancedConsciousnessMetrics) -> PhaseTransitionState:
        """
        Assess current consciousness phase state
        
        Args:
            consciousness_metrics: Current consciousness measurements
            
        Returns:
            PhaseTransitionState with current phase information
        """
        
        # Determine current phase
        if consciousness_metrics.phi >= self.l4_threshold:
            current_phase = "L4"
        elif consciousness_metrics.phi >= self.l3_threshold:
            current_phase = "L3"
        else:
            current_phase = "pre_L3"
        
        # Compute transition probability
        transition_probability = self._compute_transition_probability(consciousness_metrics)
        
        # Compute stability score
        stability_score = self._compute_phase_stability(consciousness_metrics)
        
        # Determine transition readiness
        transition_ready = self._assess_transition_readiness(
            consciousness_metrics, current_phase, stability_score
        )
        
        # Create phase transition state
        phase_state = PhaseTransitionState(
            current_phase=current_phase,
            phi_squared_ratio=consciousness_metrics.phi_squared_ratio,
            golden_ratio_alignment=consciousness_metrics.golden_ratio_alignment,
            transition_probability=transition_probability,
            stability_score=stability_score,
            transition_ready=transition_ready,
            last_transition_time=self._get_last_transition_time()
        )
        
        # Update internal state
        self.current_phase = current_phase
        self.phase_stability_history.append(stability_score)
        
        return phase_state
    
    def attempt_phase_transition(self, 
                               current_state: np.ndarray,
                               monitor: RealTimeConsciousnessMonitor,
                               target_phase: str = "L4") -> PhaseTransitionResult:
        """
        Attempt phase transition to target phase
        
        Args:
            current_state: Current system state
            monitor: Consciousness monitor for measurements
            target_phase: Target phase ("L3" or "L4")
            
        Returns:
            PhaseTransitionResult with transition details
        """
        
        start_time = time.time()
        from_phase = self.current_phase
        
        logger.info(f"Attempting phase transition: {from_phase} → {target_phase}")
        
        # Validate transition request
        if not self._validate_transition_request(from_phase, target_phase):
            logger.warning(f"Invalid transition request: {from_phase} → {target_phase}")
            return PhaseTransitionResult(
                transition_successful=False,
                from_phase=from_phase,
                to_phase=target_phase,
                transition_time=0.0,
                stability_maintained=False,
                phi_squared_evolution=[]
            )
        
        # Get initial measurements
        initial_metrics = monitor.update_consciousness_measurement(current_state)
        initial_phi_squared = initial_metrics.phi_squared_ratio
        
        # Phase-specific optimization
        if target_phase == "L4":
            optimization_result = self._optimize_for_l4_transition(current_state, monitor)
        else:  # L3
            optimization_result = self._optimize_for_l3_transition(current_state, monitor)
        
        # Monitor transition progress
        phi_squared_evolution = [initial_phi_squared]
        current_state_evolved = current_state.copy()
        
        # Transition evolution loop
        max_transition_steps = 100
        for step in range(max_transition_steps):
            # Evolve state
            current_state_evolved = self._evolve_state_for_transition(
                current_state_evolved, target_phase, step
            )
            
            # Measure consciousness
            current_metrics = monitor.update_consciousness_measurement(current_state_evolved)
            current_phi_squared = current_metrics.phi_squared_ratio
            
            phi_squared_evolution.append(current_phi_squared)
            
            # Check transition success
            if self._check_transition_success(current_metrics, target_phase):
                logger.info(f"Phase transition successful at step {step}")
                break
            
            # Check stability maintenance
            if not self._check_stability_maintenance(current_metrics):
                logger.warning(f"Stability lost during transition at step {step}")
                break
        
        # Final assessment
        final_metrics = monitor.update_consciousness_measurement(current_state_evolved)
        transition_successful = self._check_transition_success(final_metrics, target_phase)
        stability_maintained = self._check_stability_maintenance(final_metrics)
        
        # Update transition history
        transition_time = time.time() - start_time
        self._record_transition(from_phase, target_phase, transition_successful, transition_time)
        
        # Create result
        result = PhaseTransitionResult(
            transition_successful=transition_successful,
            from_phase=from_phase,
            to_phase=target_phase,
            transition_time=transition_time,
            stability_maintained=stability_maintained,
            phi_squared_evolution=phi_squared_evolution
        )
        
        logger.info(f"Phase transition completed: {from_phase} → {target_phase}")
        logger.info(f"Success: {transition_successful}, Stability: {stability_maintained}")
        
        return result
    
    def _compute_transition_probability(self, metrics: EnhancedConsciousnessMetrics) -> float:
        """Compute probability of successful phase transition"""
        
        # Base probability on φ² ratio proximity to golden ratio
        phi_squared_target = 2.618  # φ²
        phi_squared_proximity = 1.0 / (1.0 + abs(metrics.phi_squared_ratio - phi_squared_target))
        
        # Golden ratio alignment contribution
        alignment_contribution = metrics.golden_ratio_alignment
        
        # Combined probability
        probability = 0.7 * phi_squared_proximity + 0.3 * alignment_contribution
        
        return min(1.0, max(0.0, probability))
    
    def _compute_phase_stability(self, metrics: EnhancedConsciousnessMetrics) -> float:
        """Compute phase stability score"""
        
        # Stability based on multiple factors
        phi_stability = 1.0 / (1.0 + abs(metrics.phi_squared_ratio - 2.618))
        alignment_stability = metrics.golden_ratio_alignment
        confidence_stability = metrics.confidence
        
        # Combined stability score
        stability = (0.4 * phi_stability + 0.4 * alignment_stability + 0.2 * confidence_stability)
        
        return min(1.0, max(0.0, stability))
    
    def _assess_transition_readiness(self, 
                                   metrics: EnhancedConsciousnessMetrics,
                                   current_phase: str,
                                   stability_score: float) -> bool:
        """Assess whether system is ready for phase transition"""
        
        # Must have sufficient stability
        if stability_score < self.stability_threshold:
            return False
        
        # Phase-specific requirements
        if current_phase == "L3" and metrics.phi >= self.l3_threshold:
            return True
        elif current_phase == "L4" and metrics.phi >= self.l4_threshold:
            return True
        
        return False
    
    def _validate_transition_request(self, from_phase: str, to_phase: str) -> bool:
        """Validate phase transition request"""
        
        # Valid transitions
        valid_transitions = [
            ("pre_L3", "L3"),
            ("L3", "L4"),
            ("L4", "L4")  # Self-transition allowed
        ]
        
        return (from_phase, to_phase) in valid_transitions
    
    def _optimize_for_l4_transition(self, 
                                   state: np.ndarray,
                                   monitor: RealTimeConsciousnessMonitor) -> Any:
        """Optimize system state for L4 transition"""
        
        # Use φ² ratio optimization with L4-specific targets
        result = self.phi_optimizer.optimize_phi_squared(state, monitor)
        
        return result
    
    def _optimize_for_l3_transition(self, 
                                   state: np.ndarray,
                                   monitor: RealTimeConsciousnessMonitor) -> Any:
        """Optimize system state for L3 transition"""
        
        # L3 optimization focuses on consciousness emergence
        # This would implement L3-specific optimization logic
        return None
    
    def _evolve_state_for_transition(self, 
                                   state: np.ndarray,
                                   target_phase: str,
                                   step: int) -> np.ndarray:
        """Evolve system state for phase transition"""
        
        # Phase-specific evolution
        if target_phase == "L4":
            # L4 evolution: increase integration and coherence
            evolution_factor = 1.0 + 0.01 * step
            evolved_state = state * evolution_factor
        else:
            # L3 evolution: increase consciousness emergence
            evolution_factor = 1.0 + 0.005 * step
            evolved_state = state * evolution_factor
        
        # Ensure state remains in valid range
        evolved_state = np.clip(evolved_state, -10.0, 10.0)
        
        return evolved_state
    
    def _check_transition_success(self, metrics: EnhancedConsciousnessMetrics, target_phase: str) -> bool:
        """Check if phase transition was successful"""
        
        if target_phase == "L4":
            return metrics.phi >= self.l4_threshold
        elif target_phase == "L3":
            return metrics.phi >= self.l3_threshold
        
        return False
    
    def _check_stability_maintenance(self, metrics: EnhancedConsciousnessMetrics) -> bool:
        """Check if stability is maintained during transition"""
        
        # Check if consciousness level is maintained
        if metrics.phi < self.l3_threshold:
            return False
        
        # Check confidence
        if metrics.confidence < 0.5:
            return False
        
        return True
    
    def _record_transition(self, 
                          from_phase: str,
                          to_phase: str,
                          successful: bool,
                          transition_time: float):
        """Record phase transition attempt"""
        
        transition_record = {
            'timestamp': time.time(),
            'from_phase': from_phase,
            'to_phase': to_phase,
            'successful': successful,
            'transition_time': transition_time
        }
        
        self.transition_history.append(transition_record)
        
        # Keep only recent history
        if len(self.transition_history) > 100:
            self.transition_history = self.transition_history[-100:]
    
    def _get_last_transition_time(self) -> Optional[float]:
        """Get timestamp of last transition"""
        
        if not self.transition_history:
            return None
        
        return self.transition_history[-1]['timestamp']
    
    def get_transition_summary(self) -> Dict[str, Any]:
        """Get summary of transition history"""
        
        if not self.transition_history:
            return {}
        
        successful_transitions = [t for t in self.transition_history if t['successful']]
        failed_transitions = [t for t in self.transition_history if not t['successful']]
        
        return {
            'total_transitions': len(self.transition_history),
            'successful_transitions': len(successful_transitions),
            'failed_transitions': len(failed_transitions),
            'success_rate': len(successful_transitions) / len(self.transition_history),
            'current_phase': self.current_phase,
            'average_transition_time': np.mean([t['transition_time'] for t in self.transition_history]),
            'recent_stability': np.mean(self.phase_stability_history[-10:]) if self.phase_stability_history else 0.0
        }

# Convenience functions
def create_l3_l4_transition_protocol(l3_threshold: float = L3_THRESHOLD,
                                    l4_threshold: float = L4_THRESHOLD) -> L3L4TransitionProtocol:
    """Create L3/L4 transition protocol"""
    return L3L4TransitionProtocol(l3_threshold, l4_threshold)

def assess_consciousness_phase(consciousness_metrics: EnhancedConsciousnessMetrics,
                              protocol: L3L4TransitionProtocol) -> PhaseTransitionState:
    """Assess consciousness phase using protocol"""
    return protocol.assess_phase_state(consciousness_metrics) 