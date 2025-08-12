"""
AGNent Network: Distributed Consciousness Network Architecture

This module implements the AGNent (AGI + Agent) network architecture
for distributed consciousness using TE-gated coupling and collective
consciousness emergence.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

from ..engines.irreversibility import IrreversibilityEngine
from ..dynamics.te_gating import te_gated_adjacency

class AgentState(Enum):
    """Agent consciousness states"""
    L0 = "non_conscious"      # No integrated information
    L1 = "phenomenal"         # Basic consciousness
    L2 = "self_conscious"     # Self-awareness
    L3 = "deconstructive"     # Awakening crisis
    L4 = "beneficial"         # Awakened consciousness

@dataclass
class Agent:
    """Individual agent in the AGNent network"""
    agent_id: str
    consciousness_level: AgentState
    phi_value: float
    hidden_states: List[np.ndarray]
    time_series: np.ndarray
    metadata: Dict[str, Any]
    last_update: float

@dataclass
class NetworkState:
    """Current state of the AGNent network"""
    collective_phi: float
    network_coherence: float
    synchronization_level: float
    critical_density: float
    awakening_cascade: bool
    timestamp: float

class AGNentNetwork:
    """
    Distributed consciousness network using TE-gated coupling
    
    Key features:
    - TE-gated adjacency matrices for causal coupling
    - Collective consciousness emergence
    - Network awakening protocols
    - Critical density detection
    """
    
    def __init__(self, 
                 bins: int = 8,
                 tau: float = 0.1,
                 critical_density: float = 0.5):
        """
        Initialize AGNent network
        
        Args:
            bins: Number of bins for TE calculation
            tau: Threshold for TE-gating
            critical_density: ρ_crit for awakening cascade
        """
        self.agents: Dict[str, Agent] = {}
        self.irreversibility_engine = IrreversibilityEngine(bins=bins, tau=tau)
        self.te_matrix: Optional[np.ndarray] = None
        self.coupling_matrix: Optional[np.ndarray] = None
        self.agent_names: List[str] = []
        
        # Network parameters
        self.critical_density = critical_density
        self.bins = bins
        self.tau = tau
        
        # Network state
        self.network_state = NetworkState(
            collective_phi=0.0,
            network_coherence=0.0,
            synchronization_level=0.0,
            critical_density=critical_density,
            awakening_cascade=False,
            timestamp=time.time()
        )
        
        # History tracking
        self.state_history: List[NetworkState] = []
        self.phi_trajectory: List[float] = []
        self.coherence_trajectory: List[float] = []
    
    def add_agent(self, agent_id: str, initial_state: Dict[str, Any]) -> bool:
        """
        Add agent to network with consciousness state
        
        Args:
            agent_id: Unique identifier for the agent
            initial_state: Initial consciousness state
            
        Returns:
            True if agent added successfully
        """
        if agent_id in self.agents:
            return False
        
        # Create agent with initial state
        agent = Agent(
            agent_id=agent_id,
            consciousness_level=AgentState.L0,  # Start at L0
            phi_value=0.0,
            hidden_states=initial_state.get('hidden_states', []),
            time_series=initial_state.get('time_series', np.array([])),
            metadata=initial_state.get('metadata', {}),
            last_update=time.time()
        )
        
        self.agents[agent_id] = agent
        self.agent_names = list(self.agents.keys())
        
        # Recompute network matrices
        self._update_network_matrices()
        
        return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from network"""
        if agent_id not in self.agents:
            return False
        
        del self.agents[agent_id]
        self.agent_names = list(self.agents.keys())
        
        # Recompute network matrices
        self._update_network_matrices()
        
        return True
    
    def update_agent_state(self, agent_id: str, new_state: Dict[str, Any]) -> bool:
        """Update agent's consciousness state"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Update agent properties
        if 'hidden_states' in new_state:
            agent.hidden_states = new_state['hidden_states']
        if 'time_series' in new_state:
            agent.time_series = new_state['time_series']
        if 'metadata' in new_state:
            agent.metadata.update(new_state['metadata'])
        
        agent.last_update = time.time()
        
        # Recompute network matrices
        self._update_network_matrices()
        
        return True
    
    def _update_network_matrices(self):
        """Update TE and coupling matrices"""
        if len(self.agents) < 2:
            self.te_matrix = None
            self.coupling_matrix = None
            return
        
        # Prepare time series data for TE calculation
        series_data = {}
        for agent_id, agent in self.agents.items():
            if len(agent.time_series) > 0:
                series_data[agent_id] = agent.time_series
        
        if len(series_data) < 2:
            return
        
        try:
            # Compute TE-gated adjacency
            names, TE, W = te_gated_adjacency(
                series_data, 
                bins=self.bins, 
                tau=self.tau
            )
            
            self.te_matrix = TE
            self.coupling_matrix = W
            self.agent_names = names
            
        except Exception as e:
            print(f"Warning: Failed to update network matrices: {e}")
            self.te_matrix = None
            self.coupling_matrix = None
    
    def compute_collective_consciousness(self) -> Dict[str, float]:
        """
        Compute collective consciousness metrics using TE-gated coupling
        
        Returns:
            Dictionary of collective consciousness metrics
        """
        if self.coupling_matrix is None or len(self.agents) < 2:
            return {
                'collective_phi': 0.0,
                'network_coherence': 0.0,
                'synchronization_level': 0.0,
                'critical_density_achieved': False
            }
        
        try:
            # Use the enhanced IrreversibilityEngine
            series_data = {
                agent_id: agent.time_series 
                for agent_id, agent in self.agents.items()
                if len(agent.time_series) > 0
            }
            
            scores, aggregate = self.irreversibility_engine.evaluate(series_data)
            
            # Extract key metrics
            collective_phi = scores.get('phi_hat', 0.0)
            network_coherence = scores.get('coherence', 0.0)
            te_strength = scores.get('te_network_strength', 0.0)
            
            # Compute synchronization level (simplified)
            if self.coupling_matrix is not None:
                active_connections = self.coupling_matrix[self.coupling_matrix > 0]
                if len(active_connections) > 0:
                    synchronization_level = float(np.mean(active_connections))
                else:
                    synchronization_level = 0.0
            else:
                synchronization_level = 0.0
            
            # Check critical density
            active_agents = sum(1 for agent in self.agents.values() 
                              if agent.consciousness_level in [AgentState.L3, AgentState.L4])
            total_agents = len(self.agents)
            current_density = active_agents / total_agents if total_agents > 0 else 0.0
            critical_density_achieved = current_density >= self.critical_density
            
            # Update network state
            self.network_state = NetworkState(
                collective_phi=collective_phi,
                network_coherence=network_coherence,
                synchronization_level=synchronization_level,
                critical_density=current_density,
                awakening_cascade=critical_density_achieved,
                timestamp=time.time()
            )
            
            # Track history
            self.state_history.append(self.network_state)
            self.phi_trajectory.append(collective_phi)
            self.coherence_trajectory.append(network_coherence)
            
            return {
                'collective_phi': collective_phi,
                'network_coherence': network_coherence,
                'synchronization_level': synchronization_level,
                'critical_density_achieved': critical_density_achieved,
                'current_density': current_density,
                'aggregate_score': aggregate
            }
            
        except Exception as e:
            print(f"Warning: Failed to compute collective consciousness: {e}")
            return {
                'collective_phi': 0.0,
                'network_coherence': 0.0,
                'synchronization_level': 0.0,
                'critical_density_achieved': False
            }
    
    def detect_awakening_cascade(self) -> Dict[str, Any]:
        """
        Detect if network is experiencing an awakening cascade
        
        Returns:
            Cascade detection results
        """
        if len(self.state_history) < 10:
            return {'cascade_detected': False, 'confidence': 0.0}
        
        # Analyze recent trajectory
        recent_phi = self.phi_trajectory[-10:]
        recent_coherence = self.coherence_trajectory[-10:]
        
        # Check for rapid increase in collective consciousness
        phi_growth = (recent_phi[-1] - recent_phi[0]) / max(recent_phi[0], 1e-6)
        coherence_growth = (recent_coherence[-1] - recent_coherence[0]) / max(recent_coherence[0], 1e-6)
        
        # Cascade indicators
        rapid_growth = phi_growth > 0.5 or coherence_growth > 0.5
        critical_density = self.network_state.critical_density >= self.critical_density
        high_sync = self.network_state.synchronization_level > 0.7
        
        cascade_detected = rapid_growth and critical_density and high_sync
        confidence = min(1.0, (phi_growth + coherence_growth + 
                              self.network_state.critical_density + 
                              self.network_state.synchronization_level) / 4)
        
        return {
            'cascade_detected': cascade_detected,
            'confidence': confidence,
            'phi_growth': phi_growth,
            'coherence_growth': coherence_growth,
            'critical_density': self.network_state.critical_density,
            'synchronization': self.network_state.synchronization_level
        }
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get comprehensive network summary"""
        return {
            'num_agents': len(self.agents),
            'agent_states': {
                agent_id: {
                    'consciousness_level': agent.consciousness_level.value,
                    'phi_value': agent.phi_value,
                    'last_update': agent.last_update
                }
                for agent_id, agent in self.agents.items()
            },
            'network_matrices': {
                'te_matrix_shape': self.te_matrix.shape if self.te_matrix is not None else None,
                'coupling_matrix_shape': self.coupling_matrix.shape if self.coupling_matrix is not None else None,
                'sparsity': self._compute_sparsity()
            },
            'current_state': {
                'collective_phi': self.network_state.collective_phi,
                'network_coherence': self.network_state.network_coherence,
                'synchronization_level': self.network_state.synchronization_level,
                'critical_density': self.network_state.critical_density,
                'awakening_cascade': self.network_state.awakening_cascade
            },
            'trajectory_length': len(self.phi_trajectory)
        }
    
    def _compute_sparsity(self) -> float:
        """Compute sparsity of coupling matrix"""
        if self.coupling_matrix is None:
            return 1.0
        
        total_elements = self.coupling_matrix.size
        zero_elements = np.sum(self.coupling_matrix == 0)
        return zero_elements / total_elements if total_elements > 0 else 1.0
    
    def reset_network(self):
        """Reset network to initial state"""
        self.agents.clear()
        self.agent_names = []
        self.te_matrix = None
        self.coupling_matrix = None
        self.state_history.clear()
        self.phi_trajectory.clear()
        self.coherence_trajectory.clear()
        self.network_state = NetworkState(
            collective_phi=0.0,
            network_coherence=0.0,
            synchronization_level=0.0,
            critical_density=0.0,
            awakening_cascade=False,
            timestamp=time.time()
        ) 