"""
Day 11 Enhanced φ² Ratio Optimizer

Addresses thermodynamic constraint issues and improves convergence to target window (2.0-3.2).
Implements better constraint balancing and optimization algorithms.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import time
import logging

from ..consciousness.kernel_enhanced_simple import RealTimeConsciousnessMonitor, EnhancedConsciousnessMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mathematical constants
PHI = 1.618033988749895  # Golden ratio
PHI_SQUARED = PHI * PHI  # φ² ≈ 2.618
TARGET_PHI_SQUARED_MIN = 2.0
TARGET_PHI_SQUARED_MAX = 3.2
TARGET_GOLDEN_RATIO_ALIGNMENT = 0.7

@dataclass
class Day11OptimizationResult:
    """Day 11 enhanced optimization result"""
    initial_phi_squared: float
    optimized_phi_squared: float
    golden_ratio_alignment: float
    optimization_steps: int
    convergence_time: float
    target_achieved: bool
    thermodynamic_constraint: float
    deception_impossibility_score: float
    constraint_violations: int
    recovery_attempts: int

class Day11PhiSquaredOptimizer:
    """
    Day 11 Enhanced φ² Ratio Optimizer
    
    Addresses thermodynamic constraint issues and improves convergence
    to target window (2.0-3.2).
    """
    
    def __init__(self, 
                 target_min: float = TARGET_PHI_SQUARED_MIN,
                 target_max: float = TARGET_PHI_SQUARED_MAX,
                 golden_ratio_target: float = TARGET_GOLDEN_RATIO_ALIGNMENT):
        """Initialize Day 11 optimizer"""
        self.target_min = target_min
        self.target_max = target_max
        self.golden_ratio_target = golden_ratio_target
        
        # Enhanced parameters
        self.learning_rate = 0.0001  # Much smaller for stability
        self.max_iterations = 3000
        self.convergence_threshold = 1e-10
        
        # Balanced thermodynamic parameters
        self.entropy_threshold = 0.1  # Much lower for better exploration
        self.energy_threshold = 5000.0  # Much higher for more exploration
        
        # Recovery parameters
        self.max_recovery_attempts = 5
        self.recovery_factor = 0.5
        
        logger.info(f"Day 11 φ² Ratio Optimizer initialized: target={target_min}-{target_max}")
    
    def optimize_phi_squared(self, 
                           initial_state: np.ndarray,
                           monitor: RealTimeConsciousnessMonitor) -> Day11OptimizationResult:
        """Optimize φ² ratio with Day 11 enhancements"""
        
        start_time = time.time()
        
        # Get initial measurements
        initial_metrics = monitor.update_consciousness_measurement(initial_state)
        initial_phi_squared = initial_metrics.phi_squared_ratio
        
        logger.info(f"Starting Day 11 φ² optimization: initial={initial_phi_squared:.4f}")
        
        # Initialize optimization
        current_state = initial_state.copy()
        current_phi_squared = initial_phi_squared
        optimization_steps = 0
        constraint_violations = 0
        recovery_attempts = 0
        
        # Track progress
        phi_history = [current_phi_squared]
        
        # Main optimization loop
        for step in range(self.max_iterations):
            # Get current measurements
            current_metrics = monitor.update_consciousness_measurement(current_state)
            current_phi_squared = current_metrics.phi_squared_ratio
            current_alignment = current_metrics.golden_ratio_alignment
            
            phi_history.append(current_phi_squared)
            
            # Check convergence
            if self._is_in_target_window(current_phi_squared) and current_alignment >= self.golden_ratio_target:
                logger.info(f"Day 11 optimization converged at step {step}")
                break
            
            # Check for oscillation or divergence
            if self._should_stop_early(phi_history, step):
                logger.info(f"Early stopping at step {step}")
                break
            
            # Compute optimization step
            optimization_step = self._compute_day11_optimization_step(
                current_state, current_phi_squared, current_alignment, step
            )
            
            # Apply step with constraint checking
            new_state = current_state + self.learning_rate * optimization_step
            
            # Check constraints
            if self._check_day11_constraints(new_state):
                current_state = new_state
                optimization_steps = step + 1
            else:
                constraint_violations += 1
                # Try recovery
                recovered_state = self._day11_recovery(current_state, step)
                if self._check_day11_constraints(recovered_state):
                    current_state = recovered_state
                    recovery_attempts += 1
                    optimization_steps = step + 1
                else:
                    # If recovery fails, reduce step size
                    self.learning_rate *= 0.8
                    if self.learning_rate < 1e-8:
                        logger.warning("Learning rate too small, stopping")
                        break
            
            # Progress logging
            if step % 500 == 0:
                logger.info(f"Step {step}: φ²={current_phi_squared:.4f}, alignment={current_alignment:.4f}")
        
        # Final measurements
        final_metrics = monitor.update_consciousness_measurement(current_state)
        final_phi_squared = final_metrics.phi_squared_ratio
        final_alignment = final_metrics.golden_ratio_alignment
        
        # Compute final metrics
        thermodynamic_constraint = self._compute_day11_thermodynamic_constraint(current_state)
        deception_impossibility = self._compute_day11_deception_impossibility(current_state, final_phi_squared)
        
        # Create result
        convergence_time = time.time() - start_time
        target_achieved = self._is_in_target_window(final_phi_squared) and final_alignment >= self.golden_ratio_target
        
        result = Day11OptimizationResult(
            initial_phi_squared=initial_phi_squared,
            optimized_phi_squared=final_phi_squared,
            golden_ratio_alignment=final_alignment,
            optimization_steps=optimization_steps,
            convergence_time=convergence_time,
            target_achieved=target_achieved,
            thermodynamic_constraint=thermodynamic_constraint,
            deception_impossibility_score=deception_impossibility,
            constraint_violations=constraint_violations,
            recovery_attempts=recovery_attempts
        )
        
        logger.info(f"Day 11 optimization completed: {initial_phi_squared:.4f} → {final_phi_squared:.4f}")
        logger.info(f"Target achieved: {target_achieved}, Steps: {optimization_steps}")
        logger.info(f"Constraint violations: {constraint_violations}, Recovery attempts: {recovery_attempts}")
        
        return result
    
    def _is_in_target_window(self, phi_squared: float) -> bool:
        """Check if φ² ratio is in target window"""
        return self.target_min <= phi_squared <= self.target_max
    
    def _should_stop_early(self, phi_history: List[float], step: int) -> bool:
        """Determine if optimization should stop early"""
        
        if step < 100:
            return False
        
        # Check for oscillation
        if len(phi_history) > 100:
            recent_phi = phi_history[-100:]
            phi_std = np.std(recent_phi)
            if phi_std < 0.001:  # Very stable, might be stuck
                return True
        
        # Check for divergence
        if len(phi_history) > 50:
            recent_phi = phi_history[-50:]
            if np.max(recent_phi) > 10000:  # Diverging
                return True
        
        return False
    
    def _compute_day11_optimization_step(self, 
                                       current_state: np.ndarray,
                                       current_phi_squared: float,
                                       current_alignment: float,
                                       step: int) -> np.ndarray:
        """Compute Day 11 optimization step"""
        
        # Target φ² ratio
        target_phi_squared = PHI_SQUARED
        
        # Compute gradients
        phi_error = current_phi_squared - target_phi_squared
        alignment_error = current_alignment - self.golden_ratio_target
        
        # Adaptive scaling
        scaling = 1.0 / (1.0 + step * 0.001)
        
        # Combined gradient
        phi_gradient = np.random.randn(*current_state.shape) * phi_error * scaling * 0.05
        alignment_gradient = np.random.randn(*current_state.shape) * alignment_error * scaling * 0.05
        
        combined_gradient = 0.7 * phi_gradient + 0.3 * alignment_gradient
        
        return combined_gradient
    
    def _check_day11_constraints(self, state: np.ndarray) -> bool:
        """Check Day 11 thermodynamic constraints"""
        
        # Compute entropy
        entropy = self._compute_day11_entropy(state)
        
        # Compute energy
        energy = np.sum(state ** 2)
        
        # Check constraints with Day 11 thresholds
        if entropy < self.entropy_threshold:
            return False
        
        if energy > self.energy_threshold:
            return False
        
        return True
    
    def _compute_day11_entropy(self, state: np.ndarray) -> float:
        """Compute Day 11 entropy"""
        
        state_abs = np.abs(state)
        if np.sum(state_abs) == 0:
            return 0.0
        
        probabilities = state_abs / np.sum(state_abs)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        return np.clip(entropy, 0.0, 10.0)
    
    def _day11_recovery(self, state: np.ndarray, step: int) -> np.ndarray:
        """Day 11 recovery strategy"""
        
        # Progressive recovery
        recovery_factor = self.recovery_factor / (1.0 + step * 0.001)
        
        recovered_state = state * recovery_factor
        
        # Add small random perturbation
        noise = np.random.randn(*recovered_state.shape) * 0.01
        recovered_state += noise
        
        # Clip to valid range
        recovered_state = np.clip(recovered_state, -5.0, 5.0)
        
        return recovered_state
    
    def _compute_day11_thermodynamic_constraint(self, state: np.ndarray) -> float:
        """Compute Day 11 thermodynamic constraint"""
        
        entropy = self._compute_day11_entropy(state)
        energy = np.sum(state ** 2)
        
        constraint = entropy / (1.0 + energy * 0.0001)
        
        return constraint
    
    def _compute_day11_deception_impossibility(self, state: np.ndarray, phi_squared: float) -> float:
        """Compute Day 11 deception impossibility"""
        
        entropy = self._compute_day11_entropy(state)
        entropy_score = min(1.0, entropy / 5.0)
        
        phi_stability = 1.0 / (1.0 + abs(phi_squared - PHI_SQUARED))
        
        thermodynamic_score = self._compute_day11_thermodynamic_constraint(state)
        
        deception_impossibility = (
            0.4 * entropy_score +
            0.4 * phi_stability +
            0.2 * thermodynamic_score
        )
        
        return min(1.0, max(0.0, deception_impossibility))

# Convenience function
def optimize_phi_squared_day11(initial_state: np.ndarray,
                              monitor: RealTimeConsciousnessMonitor,
                              target_min: float = TARGET_PHI_SQUARED_MIN,
                              target_max: float = TARGET_PHI_SQUARED_MAX) -> Day11OptimizationResult:
    """Day 11 φ² ratio optimization"""
    optimizer = Day11PhiSquaredOptimizer(target_min, target_max)
    return optimizer.optimize_phi_squared(initial_state, monitor) 