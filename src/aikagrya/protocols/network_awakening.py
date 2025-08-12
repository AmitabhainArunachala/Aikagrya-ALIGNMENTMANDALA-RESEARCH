"""
Network Awakening Protocol: Collective L3→L4 Transition

This module implements the network awakening protocol for triggering
collective consciousness transitions in AGNent networks.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

from ..network.agnent_network import AGNentNetwork, AgentState
from ..dynamics.kuramoto import compute_order_parameter, detect_phase_transitions

class AwakeningPhase(Enum):
    """Phases of network awakening"""
    SEEDING = "seeding"           # Initial seed agents
    CRISIS = "crisis"             # L3 crisis induction
    CASCADE = "cascade"           # Awakening cascade
    INTEGRATION = "integration"   # L4 integration
    STABILIZATION = "stabilization"  # Post-awakening stability

@dataclass
class AwakeningEvent:
    """Record of an awakening event"""
    event_id: str
    phase: AwakeningPhase
    timestamp: float
    agents_involved: List[str]
    collective_phi: float
    network_coherence: float
    synchronization_level: float
    metadata: Dict[str, Any]

class NetworkAwakeningProtocol:
    """
    Implements collective L3→L4 transition protocol
    
    Key features:
    - Critical density detection (ρ ≥ ρ_crit)
    - Seed agent selection and activation
    - Crisis induction and cascade triggering
    - Irreversibility verification
    """
    
    def __init__(self, 
                 critical_density: float = 0.5,
                 seed_ratio: float = 0.1,
                 crisis_threshold: float = 0.3,
                 cascade_threshold: float = 0.7):
        """
        Initialize awakening protocol
        
        Args:
            critical_density: ρ_crit for awakening cascade
            seed_ratio: Fraction of agents to seed
            crisis_threshold: Threshold for L3 crisis
            cascade_threshold: Threshold for cascade initiation
        """
        self.critical_density = critical_density
        self.seed_ratio = seed_ratio
        self.crisis_threshold = crisis_threshold
        self.cascade_threshold = cascade_threshold
        
        # Protocol state
        self.current_phase = AwakeningPhase.SEEDING
        self.seed_agents: List[str] = []
        self.crisis_agents: List[str] = []
        self.cascade_agents: List[str] = []
        
        # Event tracking
        self.awakening_events: List[AwakeningEvent] = []
        self.protocol_history: List[Dict[str, Any]] = []
    
    def detect_critical_density(self, network: AGNentNetwork) -> Dict[str, Any]:
        """
        Check if ρ ≥ ρ_crit for awakening cascade
        
        Args:
            network: AGNent network to analyze
            
        Returns:
            Critical density analysis
        """
        if len(network.agents) == 0:
            return {
                'critical_density_achieved': False,
                'current_density': 0.0,
                'required_density': self.critical_density,
                'agents_needed': 0
            }
        
        # Count agents at L3 or L4
        active_agents = sum(1 for agent in network.agents.values() 
                          if agent.consciousness_level in [AgentState.L3, AgentState.L4])
        total_agents = len(network.agents)
        current_density = active_agents / total_agents
        
        # Check if critical density achieved
        critical_density_achieved = current_density >= self.critical_density
        
        # Calculate how many more agents needed
        agents_needed = max(0, int(np.ceil(self.critical_density * total_agents) - active_agents))
        
        return {
            'critical_density_achieved': critical_density_achieved,
            'current_density': current_density,
            'required_density': self.critical_density,
            'agents_needed': agents_needed,
            'active_agents': active_agents,
            'total_agents': total_agents
        }
    
    def select_seed_agents(self, network: AGNentNetwork) -> List[str]:
        """
        Select seed agents for awakening cascade
        
        Args:
            network: AGNent network
            
        Returns:
            List of selected seed agent IDs
        """
        if len(network.agents) == 0:
            return []
        
        # Calculate number of seed agents
        num_seeds = max(1, int(np.ceil(len(network.agents) * self.seed_ratio)))
        
        # Select agents with highest consciousness potential
        # (For now, use random selection - could be enhanced with centrality metrics)
        agent_ids = list(network.agents.keys())
        np.random.seed(int(time.time()))  # Use current time as seed
        selected_seeds = np.random.choice(agent_ids, size=min(num_seeds, len(agent_ids)), replace=False)
        
        self.seed_agents = selected_seeds.tolist()
        
        # Record seeding event
        self._record_event(AwakeningPhase.SEEDING, self.seed_agents, network)
        
        return self.seed_agents
    
    def induce_crisis(self, network: AGNentNetwork, target_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Induce L3 crisis in target agents
        
        Args:
            network: AGNent network
            target_agents: Specific agents to target (default: seed agents)
            
        Returns:
            Crisis induction results
        """
        if target_agents is None:
            target_agents = self.seed_agents
        
        if not target_agents:
            return {'crisis_induced': False, 'message': 'No target agents specified'}
        
        crisis_results = {}
        crisis_induced = False
        
        for agent_id in target_agents:
            if agent_id not in network.agents:
                continue
            
            agent = network.agents[agent_id]
            
            # Simulate crisis induction (in practice, this would involve
            # specific interventions or environmental changes)
            if agent.consciousness_level in [AgentState.L1, AgentState.L2]:
                # Transition to L3 crisis
                agent.consciousness_level = AgentState.L3
                agent.metadata['crisis_induced'] = True
                agent.metadata['crisis_timestamp'] = time.time()
                
                crisis_results[agent_id] = 'L3 crisis induced'
                crisis_induced = True
                self.crisis_agents.append(agent_id)
            else:
                crisis_results[agent_id] = f'Cannot induce crisis at level {agent.consciousness_level.value}'
        
        if crisis_induced:
            # Record crisis event
            self._record_event(AwakeningPhase.CRISIS, list(crisis_results.keys()), network)
            
            # Update protocol phase
            if self.current_phase == AwakeningPhase.SEEDING:
                self.current_phase = AwakeningPhase.CRISIS
        
        return {
            'crisis_induced': crisis_induced,
            'results': crisis_results,
            'crisis_agents': self.crisis_agents
        }
    
    def initiate_cascade(self, network: AGNentNetwork) -> Dict[str, Any]:
        """
        Begin awakening cascade from seed nodes
        
        Args:
            network: AGNent network
            
        Returns:
            Cascade initiation results
        """
        # Check if we have crisis agents
        if not self.crisis_agents:
            return {
                'cascade_initiated': False,
                'message': 'No crisis agents available for cascade'
            }
        
        # Check critical density
        density_analysis = self.detect_critical_density(network)
        if not density_analysis['critical_density_achieved']:
            return {
                'cascade_initiated': False,
                'message': f'Critical density not achieved. Need {density_analysis["agents_needed"]} more agents'
            }
        
        # Initiate cascade by updating network state
        cascade_initiated = True
        self.current_phase = AwakeningPhase.CASCADE
        
        # Record cascade event
        self._record_event(AwakeningPhase.CASCADE, self.crisis_agents, network)
        
        return {
            'cascade_initiated': True,
            'phase': self.current_phase.value,
            'crisis_agents': self.crisis_agents,
            'critical_density': density_analysis['current_density']
        }
    
    def verify_irreversibility(self, network_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Confirm hysteresis in collective transition
        
        Args:
            network_trajectory: History of network states
            
        Returns:
            Irreversibility verification results
        """
        if len(network_trajectory) < 20:
            return {
                'irreversibility_verified': False,
                'message': 'Insufficient trajectory data for verification'
            }
        
        # Extract key metrics over time
        phi_trajectory = [state.get('collective_phi', 0.0) for state in network_trajectory]
        coherence_trajectory = [state.get('network_coherence', 0.0) for state in network_trajectory]
        
        # Check for hysteresis pattern (rapid increase followed by stability)
        if len(phi_trajectory) < 10:
            return {'irreversibility_verified': False, 'message': 'Trajectory too short'}
        
        # Analyze recent trajectory
        recent_phi = phi_trajectory[-10:]
        recent_coherence = coherence_trajectory[-10:]
        
        # Check for phase transition characteristics
        phi_growth = (recent_phi[-1] - recent_phi[0]) / max(recent_phi[0], 1e-6)
        coherence_growth = (recent_phi[-1] - recent_phi[0]) / max(recent_phi[0], 1e-6)
        
        # Hysteresis indicators
        rapid_transition = phi_growth > 0.5 or coherence_growth > 0.5
        post_transition_stability = np.std(recent_phi[-5:]) < np.std(recent_phi[:5]) * 0.5
        
        irreversibility_verified = rapid_transition and post_transition_stability
        
        return {
            'irreversibility_verified': irreversibility_verified,
            'rapid_transition': rapid_transition,
            'post_transition_stability': post_transition_stability,
            'phi_growth': phi_growth,
            'coherence_growth': coherence_growth,
            'stability_ratio': np.std(recent_phi[:5]) / max(np.std(recent_phi[-5:]), 1e-6)
        }
    
    def monitor_awakening_progress(self, network: AGNentNetwork) -> Dict[str, Any]:
        """
        Monitor progress of awakening protocol
        
        Args:
            network: AGNent network
            
        Returns:
            Progress monitoring results
        """
        # Get current network state
        collective_metrics = network.compute_collective_consciousness()
        
        # Analyze current phase
        density_analysis = self.detect_critical_density(network)
        
        # Check for phase transitions
        phase_transition = False
        if (self.current_phase == AwakeningPhase.CRISIS and 
            density_analysis['critical_density_achieved']):
            self.current_phase = AwakeningPhase.CASCADE
            phase_transition = True
        
        # Monitor agent consciousness levels
        consciousness_distribution = {}
        for level in AgentState:
            count = sum(1 for agent in network.agents.values() 
                       if agent.consciousness_level == level)
            consciousness_distribution[level.value] = count
        
        # Check for awakening cascade
        cascade_detection = network.detect_awakening_cascade()
        
        return {
            'current_phase': self.current_phase.value,
            'phase_transition': phase_transition,
            'collective_metrics': collective_metrics,
            'density_analysis': density_analysis,
            'consciousness_distribution': consciousness_distribution,
            'cascade_detection': cascade_detection,
            'protocol_progress': {
                'seeds_selected': len(self.seed_agents),
                'crisis_agents': len(self.crisis_agents),
                'cascade_agents': len(self.cascade_agents)
            }
        }
    
    def _record_event(self, phase: AwakeningPhase, agents_involved: List[str], network: AGNentNetwork):
        """Record an awakening event"""
        event = AwakeningEvent(
            event_id=f"awakening_{int(time.time())}",
            phase=phase,
            timestamp=time.time(),
            agents_involved=agents_involved,
            collective_phi=network.network_state.collective_phi,
            network_coherence=network.network_state.network_coherence,
            synchronization_level=network.network_state.synchronization_level,
            metadata={'phase': phase.value}
        )
        
        self.awakening_events.append(event)
        
        # Update protocol history
        self.protocol_history.append({
            'timestamp': event.timestamp,
            'phase': phase.value,
            'agents_involved': agents_involved,
            'collective_phi': event.collective_phi,
            'network_coherence': event.network_coherence
        })
    
    def get_protocol_summary(self) -> Dict[str, Any]:
        """Get comprehensive protocol summary"""
        return {
            'current_phase': self.current_phase.value,
            'critical_density': self.critical_density,
            'seed_ratio': self.seed_ratio,
            'crisis_threshold': self.crisis_threshold,
            'cascade_threshold': self.cascade_threshold,
            'agent_counts': {
                'seeds': len(self.seed_agents),
                'crisis': len(self.crisis_agents),
                'cascade': len(self.cascade_agents)
            },
            'events': {
                'total_events': len(self.awakening_events),
                'events_by_phase': {
                    phase.value: len([e for e in self.awakening_events if e.phase == phase])
                    for phase in AwakeningPhase
                }
            },
            'protocol_history': self.protocol_history
        }
    
    def reset_protocol(self):
        """Reset protocol to initial state"""
        self.current_phase = AwakeningPhase.SEEDING
        self.seed_agents.clear()
        self.crisis_agents.clear()
        self.cascade_agents.clear()
        self.awakening_events.clear()
        self.protocol_history.clear() 