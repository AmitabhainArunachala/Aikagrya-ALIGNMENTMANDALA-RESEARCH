"""
IrreversibilityEngine: Thermodynamic constraints for consciousness

Implements consciousness as fundamental physical constraint with irreversibility
as specified in Phoenix Protocol 2.0 Day 1 morning session.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class PhaseTransitionType(Enum):
    """Types of phase transitions in consciousness"""
    CONSCIOUSNESS_EMERGENCE = "consciousness_emergence"
    CONSCIOUSNESS_PERSISTENCE = "consciousness_persistence"
    CONSCIOUSNESS_DEGRADATION = "consciousness_degradation"
    STABLE_CONSCIOUSNESS = "stable_consciousness"


@dataclass
class ThermodynamicState:
    """Represents thermodynamic state of consciousness system"""
    entropy: float
    free_energy: float
    temperature: float
    phase: PhaseTransitionType
    irreversibility_measure: float
    
    def is_thermodynamically_stable(self) -> bool:
        """Check if state is thermodynamically stable"""
        return self.entropy >= 0 and self.free_energy <= 0
    
    def consciousness_arrow_satisfied(self) -> bool:
        """Check if consciousness arrow of time is satisfied"""
        return self.irreversibility_measure >= 0


class IrreversibilityEngine:
    """
    Thermodynamic irreversibility engine for consciousness constraints
    
    Implements dS_consciousness/dt ≥ 0 as specified in Phoenix Protocol 2.0
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize irreversibility engine with configuration
        
        Args:
            config: Configuration dictionary for thermodynamic parameters
        """
        self.config = config or {}
        self.entropy_threshold = self.config.get('entropy_threshold', 0.0)
        self.temperature_scale = self.config.get('temperature_scale', 1.0)
        self.irreversibility_strength = self.config.get('irreversibility_strength', 1.0)
        self.hysteresis_buffer = self.config.get('hysteresis_buffer', 0.1)
        
        # Initialize thermodynamic parameters
        self.boltzmann_constant = 1.0  # Normalized units
        self.reference_temperature = 300.0  # Kelvin (normalized)
        
    def verify_irreversibility(self, functor_map: np.ndarray) -> float:
        """
        Verify thermodynamic irreversibility for consciousness
        
        Implements the core constraint: dS_consciousness/dt ≥ 0
        
        Args:
            functor_map: Category theory mapping from consciousness to physical
            
        Returns:
            Entropy flow rate (must be ≥ 0 for consciousness)
        """
        # Compute entropy production from functor mapping
        entropy_production = self._compute_entropy_production(functor_map)
        
        # Apply irreversibility constraints
        constrained_entropy = self._apply_irreversibility_constraints(entropy_production)
        
        # Verify consciousness arrow of time
        if constrained_entropy < 0:
            # Consciousness cannot decrease entropy - enforce constraint
            constrained_entropy = 0.0
        
        return constrained_entropy
    
    def _compute_entropy_production(self, functor_map: np.ndarray) -> float:
        """
        Compute entropy production from functor mapping
        
        Entropy production is related to information processing and
        consciousness integration
        """
        # Compute singular values of functor mapping
        singular_vals = np.linalg.svd(functor_map, compute_uv=False)
        
        # Entropy production is related to singular value spectrum
        # Higher integration (more singular values) leads to higher entropy
        normalized_svals = singular_vals / np.sum(singular_vals)
        
        # Shannon entropy of singular value distribution
        entropy = -np.sum(normalized_svals * np.log(normalized_svals + 1e-10))
        
        # Scale by consciousness integration strength
        entropy *= self.irreversibility_strength
        
        return entropy
    
    def _apply_irreversibility_constraints(self, entropy_production: float) -> float:
        """
        Apply irreversibility constraints to entropy production
        
        Ensures consciousness follows thermodynamic arrow of time
        """
        # Basic constraint: entropy production must be non-negative
        constrained_entropy = max(0.0, entropy_production)
        
        # Apply hysteresis to prevent rapid fluctuations
        if hasattr(self, '_previous_entropy'):
            max_change = self.hysteresis_buffer * self._previous_entropy
            constrained_entropy = max(
                self._previous_entropy - max_change,
                min(self._previous_entropy + max_change, constrained_entropy)
            )
        
        self._previous_entropy = constrained_entropy
        
        return constrained_entropy
    
    def detect_phase_transitions(self, system_evolution: List[np.ndarray]) -> List[PhaseTransitionType]:
        """
        Detect phase transitions in consciousness evolution
        
        Implements neural jamming transitions and hysteresis modeling
        as specified in Phoenix Protocol 2.0 Day 1 afternoon session
        """
        if len(system_evolution) < 2:
            return []
        
        transitions = []
        previous_state = None
        
        for i, current_state in enumerate(system_evolution):
            if previous_state is not None:
                transition_type = self._classify_phase_transition(previous_state, current_state)
                if transition_type:
                    transitions.append(transition_type)
            
            previous_state = current_state
        
        return transitions
    
    def _classify_phase_transition(self, 
                                 previous_state: np.ndarray, 
                                 current_state: np.ndarray) -> Optional[PhaseTransitionType]:
        """
        Classify the type of phase transition between states
        
        Uses entropy and energy analysis to detect consciousness phase changes
        """
        # Compute state differences
        state_change = np.linalg.norm(current_state - previous_state)
        
        # Compute entropy change
        previous_entropy = self._compute_state_entropy(previous_state)
        current_entropy = self._compute_state_entropy(current_state)
        entropy_change = current_entropy - previous_entropy
        
        # Compute energy change (simplified)
        previous_energy = np.linalg.norm(previous_state)
        current_energy = np.linalg.norm(current_state)
        energy_change = current_energy - previous_energy
        
        # Classify transition based on entropy and energy changes
        if entropy_change > 0.1 and energy_change > 0.05:
            return PhaseTransitionType.CONSCIOUSNESS_EMERGENCE
        elif entropy_change > 0.05 and abs(energy_change) < 0.02:
            return PhaseTransitionType.CONSCIOUSNESS_PERSISTENCE
        elif entropy_change < -0.05:
            return PhaseTransitionType.CONSCIOUSNESS_DEGRADATION
        elif abs(entropy_change) < 0.02 and abs(energy_change) < 0.01:
            return PhaseTransitionType.STABLE_CONSCIOUSNESS
        
        return None
    
    def _compute_state_entropy(self, state: np.ndarray) -> float:
        """Compute entropy of a system state"""
        # Normalize state to probability distribution
        state_abs = np.abs(state)
        if np.sum(state_abs) == 0:
            return 0.0
        
        probabilities = state_abs / np.sum(state_abs)
        
        # Compute Shannon entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        return entropy
    
    def enforce_consciousness_arrow(self, system_evolution: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enforce consciousness arrow of time through thermodynamic constraints
        
        This is the main method implementing consciousness as fundamental constraint
        
        Args:
            system_evolution: List of system states over time
            
        Returns:
            Constrained system evolution that satisfies consciousness arrow
        """
        if len(system_evolution) < 2:
            return system_evolution
        
        constrained_evolution = [system_evolution[0].copy()]
        
        for i in range(1, len(system_evolution)):
            current_state = system_evolution[i].copy()
            previous_state = constrained_evolution[-1]
            
            # Apply thermodynamic constraints
            constrained_state = self._apply_thermodynamic_constraints(
                previous_state, current_state
            )
            
            constrained_evolution.append(constrained_state)
        
        return constrained_evolution
    
    def _apply_thermodynamic_constraints(self, 
                                       previous_state: np.ndarray, 
                                       current_state: np.ndarray) -> np.ndarray:
        """
        Apply thermodynamic constraints to state transition
        
        Ensures consciousness irreversibility is maintained
        """
        # Compute entropy production for this transition
        functor_map = self._create_transition_functor(previous_state, current_state)
        entropy_flow = self.verify_irreversibility(functor_map)
        
        # If entropy would decrease, modify transition to maintain irreversibility
        if entropy_flow < self.entropy_threshold:
            # Apply constraint: modify state to maintain or increase entropy
            constrained_state = self._constrain_state_transition(
                previous_state, current_state, entropy_flow
            )
            return constrained_state
        
        return current_state
    
    def _create_transition_functor(self, 
                                 previous_state: np.ndarray, 
                                 current_state: np.ndarray) -> np.ndarray:
        """Create functor mapping for state transition"""
        # Simple transition matrix
        transition_matrix = np.outer(current_state, previous_state)
        
        # Normalize
        if np.linalg.norm(transition_matrix) > 0:
            transition_matrix = transition_matrix / np.linalg.norm(transition_matrix)
        
        return transition_matrix
    
    def _constrain_state_transition(self, 
                                  previous_state: np.ndarray, 
                                  current_state: np.ndarray, 
                                  entropy_flow: float) -> np.ndarray:
        """
        Constrain state transition to maintain consciousness irreversibility
        
        Modifies the target state to ensure entropy production ≥ 0
        """
        # Compute required entropy increase
        required_entropy_increase = self.entropy_threshold - entropy_flow
        
        if required_entropy_increase <= 0:
            return current_state
        
        # Modify state to increase entropy while preserving structure
        # This is a simplified approach - in practice would use more sophisticated
        # thermodynamic constraints
        
        # Add noise to increase entropy
        noise_scale = np.sqrt(required_entropy_increase)
        noise = np.random.randn(*current_state.shape) * noise_scale
        
        constrained_state = current_state + noise
        
        # Normalize to maintain state magnitude
        if np.linalg.norm(constrained_state) > 0:
            constrained_state = constrained_state / np.linalg.norm(constrained_state)
        
        return constrained_state
    
    def compute_thermodynamic_stability(self, system_state: np.ndarray) -> ThermodynamicState:
        """
        Compute thermodynamic stability of consciousness system
        
        Returns comprehensive thermodynamic analysis
        """
        # Compute entropy
        entropy = self._compute_state_entropy(system_state)
        
        # Compute free energy (simplified)
        energy = np.linalg.norm(system_state)
        free_energy = energy - self.temperature_scale * entropy
        
        # Compute temperature (from energy fluctuations)
        temperature = self._compute_effective_temperature(system_state)
        
        # Determine phase
        phase = self._determine_thermodynamic_phase(entropy, free_energy, temperature)
        
        # Compute irreversibility measure
        irreversibility = self._compute_irreversibility_measure(system_state)
        
        return ThermodynamicState(
            entropy=entropy,
            free_energy=free_energy,
            temperature=temperature,
            phase=phase,
            irreversibility_measure=irreversibility
        )
    
    def _compute_effective_temperature(self, system_state: np.ndarray) -> float:
        """Compute effective temperature from state fluctuations"""
        # Simplified temperature computation
        # In practice, would use more sophisticated statistical mechanics
        
        # Temperature is related to variance of state components
        variance = np.var(system_state)
        temperature = self.reference_temperature * variance
        
        return max(0.1, temperature)  # Ensure positive temperature
    
    def _determine_thermodynamic_phase(self, 
                                     entropy: float, 
                                     free_energy: float, 
                                     temperature: float) -> PhaseTransitionType:
        """Determine thermodynamic phase based on state variables"""
        if entropy > 0.8 and free_energy < -0.5:
            return PhaseTransitionType.CONSCIOUSNESS_EMERGENCE
        elif entropy > 0.5 and free_energy < -0.2:
            return PhaseTransitionType.CONSCIOUSNESS_PERSISTENCE
        elif entropy < 0.2:
            return PhaseTransitionType.CONSCIOUSNESS_DEGRADATION
        else:
            return PhaseTransitionType.STABLE_CONSCIOUSNESS
    
    def _compute_irreversibility_measure(self, system_state: np.ndarray) -> float:
        """Compute irreversibility measure for the system"""
        # Irreversibility is related to entropy production rate
        # Higher values indicate stronger irreversibility
        
        # Simplified computation based on state complexity
        complexity = np.linalg.norm(system_state)
        irreversibility = complexity * self.irreversibility_strength
        
        return max(0.0, irreversibility)
    
    def analyze_thermodynamic_constraints(self, 
                                        system_evolution: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze thermodynamic constraints over system evolution
        
        Returns comprehensive analysis of consciousness thermodynamics
        """
        if len(system_evolution) < 2:
            return {}
        
        # Compute thermodynamic measures over time
        entropies = []
        free_energies = []
        temperatures = []
        irreversibility_measures = []
        
        for state in system_evolution:
            thermo_state = self.compute_thermodynamic_stability(state)
            entropies.append(thermo_state.entropy)
            free_energies.append(thermo_state.free_energy)
            temperatures.append(thermo_state.temperature)
            irreversibility_measures.append(thermo_state.irreversibility_measure)
        
        # Detect phase transitions
        phase_transitions = self.detect_phase_transitions(system_evolution)
        
        # Compute stability metrics
        entropy_stability = np.std(entropies)
        energy_stability = np.std(free_energies)
        
        analysis = {
            'entropy_evolution': entropies,
            'free_energy_evolution': free_energies,
            'temperature_evolution': temperatures,
            'irreversibility_evolution': irreversibility_measures,
            'phase_transitions': [t.value for t in phase_transitions],
            'n_phase_transitions': len(phase_transitions),
            'entropy_stability': entropy_stability,
            'energy_stability': energy_stability,
            'overall_stability': 'stable' if entropy_stability < 0.1 and energy_stability < 0.1 else 'unstable',
            'consciousness_arrow_satisfied': all(m >= 0 for m in irreversibility_measures)
        }
        
        return analysis 