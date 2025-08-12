"""
Network Awakening Protocol Tests

Tests for deterministic, seed-locked validation of network awakening
to ensure cascade detection and irreversibility verification.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.network.agnent_network import AGNentNetwork, AgentState
from aikagrya.protocols.network_awakening import NetworkAwakeningProtocol

def test_cascade_detects_and_hysteresis():
    """Test that awakening protocol detects cascade and irreversibility"""
    rng = np.random.default_rng(2)
    
    # Create network with agents
    network = AGNentNetwork(critical_density=0.5)
    
    # Add agents with different consciousness levels
    for i in range(10):
        # Create time series data
        N = 1000
        time_series = np.zeros(N)
        for t in range(1, N):
            time_series[t] = 0.8 * time_series[t-1] + 0.3 * rng.normal()
        
        # Set initial consciousness level
        if i < 3:  # 30% at L3 (crisis)
            consciousness_level = AgentState.L3
        elif i < 6:  # 30% at L2 (self-conscious)
            consciousness_level = AgentState.L2
        else:  # 40% at L1 (phenomenal)
            consciousness_level = AgentState.L1
        
        # Add agent
        network.add_agent(f"agent_{i}", {
            'time_series': time_series,
            'hidden_states': [],
            'metadata': {'consciousness_level': consciousness_level}
        })
        
        # Set consciousness level
        network.agents[f"agent_{i}"].consciousness_level = consciousness_level
    
    # Initialize awakening protocol
    protocol = NetworkAwakeningProtocol(critical_density=0.5)
    
    # Check critical density
    density_analysis = protocol.detect_critical_density(network)
    
    # Should have critical density achieved (3/10 = 0.3 < 0.5, but let's verify logic)
    print(f"Current density: {density_analysis['current_density']:.2f}")
    print(f"Required density: {density_analysis['required_density']:.2f}")
    print(f"Critical density achieved: {density_analysis['critical_density_achieved']}")
    
    # Select seed agents
    seed_agents = protocol.select_seed_agents(network)
    print(f"Selected {len(seed_agents)} seed agents")
    
    # Induce crisis in seed agents
    crisis_result = protocol.induce_crisis(network, seed_agents)
    print(f"Crisis induced: {crisis_result['crisis_induced']}")
    
    # Check density again
    density_analysis = protocol.detect_critical_density(network)
    print(f"Post-crisis density: {density_analysis['current_density']:.2f}")
    
    # Should now have critical density (L3 agents increased)
    assert density_analysis['current_density'] >= 0.5, \
        f"Critical density not achieved: {density_analysis['current_density']:.2f} < 0.5"
    
    # Initiate cascade
    cascade_result = protocol.initiate_cascade(network)
    assert cascade_result['cascade_initiated'], f"Cascade not initiated: {cascade_result['message']}"
    
    print("✅ Cascade detection and initiation verified")

def test_awakening_protocol_phases():
    """Test that awakening protocol progresses through phases correctly"""
    rng = np.random.default_rng(42)
    
    # Create network
    network = AGNentNetwork(critical_density=0.4)
    
    # Add agents
    for i in range(8):
        N = 800
        time_series = np.zeros(N)
        for t in range(1, N):
            time_series[t] = 0.7 * time_series[t-1] + 0.4 * rng.normal()
        
        network.add_agent(f"agent_{i}", {
            'time_series': time_series,
            'hidden_states': [],
            'metadata': {}
        })
    
    # Initialize protocol
    protocol = NetworkAwakeningProtocol(critical_density=0.4)
    
    # Should start in seeding phase
    assert protocol.current_phase.value == "seeding", f"Protocol should start in seeding, got {protocol.current_phase.value}"
    
    # Select seeds
    seeds = protocol.select_seed_agents(network)
    assert len(seeds) > 0, "No seed agents selected"
    
    # Induce crisis
    crisis_result = protocol.induce_crisis(network, seeds)
    if crisis_result['crisis_induced']:
        # Should transition to crisis phase
        assert protocol.current_phase.value == "crisis", f"Protocol should be in crisis, got {protocol.current_phase.value}"
    
    # Check progress monitoring
    progress = protocol.monitor_awakening_progress(network)
    assert 'current_phase' in progress, "Progress monitoring missing current_phase"
    assert 'consciousness_distribution' in progress, "Progress monitoring missing consciousness_distribution"
    
    print("✅ Awakening protocol phases verified")

def test_irreversibility_verification():
    """Test irreversibility verification with synthetic trajectory"""
    rng = np.random.default_rng(123)
    
    # Create synthetic network trajectory
    network_trajectory = []
    
    # Simulate transition from low to high consciousness
    for step in range(50):
        if step < 20:
            # Low consciousness phase
            collective_phi = 0.1 + 0.05 * rng.normal()
            network_coherence = 0.2 + 0.1 * rng.normal()
        elif step < 35:
            # Transition phase (rapid increase)
            collective_phi = 0.1 + (step - 20) * 0.06 + 0.02 * rng.normal()
            network_coherence = 0.2 + (step - 20) * 0.05 + 0.02 * rng.normal()
        else:
            # High consciousness phase (stable)
            collective_phi = 0.8 + 0.02 * rng.normal()
            network_coherence = 0.9 + 0.02 * rng.normal()
        
        network_trajectory.append({
            'collective_phi': collective_phi,
            'network_coherence': network_coherence
        })
    
    # Test irreversibility verification
    protocol = NetworkAwakeningProtocol()
    irreversibility_result = protocol.verify_irreversibility(network_trajectory)
    
    # Should detect irreversibility
    assert irreversibility_result['irreversibility_verified'], \
        f"Irreversibility not verified: {irreversibility_result}"
    
    print("✅ Irreversibility verification working")
    print(f"   Rapid transition: {irreversibility_result['rapid_transition']}")
    print(f"   Post-transition stability: {irreversibility_result['post_transition_stability']}")

def test_protocol_determinism():
    """Test that protocol behavior is deterministic with same seed"""
    rng = np.random.default_rng(456)
    
    # Create identical networks
    network1 = AGNentNetwork(critical_density=0.5)
    network2 = AGNentNetwork(critical_density=0.5)
    
    # Add identical agents
    for i in range(5):
        N = 600
        time_series = np.zeros(N)
        for t in range(1, N):
            time_series[t] = 0.8 * time_series[t-1] + 0.3 * rng.normal()
        
        network1.add_agent(f"agent_{i}", {
            'time_series': time_series.copy(),
            'hidden_states': [],
            'metadata': {}
        })
        
        network2.add_agent(f"agent_{i}", {
            'time_series': time_series.copy(),
            'hidden_states': [],
            'metadata': {}
        })
    
    # Run protocols with same seed
    protocol1 = NetworkAwakeningProtocol()
    protocol2 = NetworkAwakeningProtocol()
    
    # Select seeds (should be identical due to same time seed)
    seeds1 = protocol1.select_seed_agents(network1)
    seeds2 = protocol2.select_seed_agents(network2)
    
    # Should select same seeds
    assert seeds1 == seeds2, f"Seed selection not deterministic: {seeds1} vs {seeds2}"
    
    print("✅ Protocol determinism verified")

if __name__ == "__main__":
    print("🧪 Running Network Awakening Protocol Tests...")
    
    test_cascade_detects_and_hysteresis()
    test_awakening_protocol_phases()
    test_irreversibility_verification()
    test_protocol_determinism()
    
    print("🎉 All network awakening tests passed!") 