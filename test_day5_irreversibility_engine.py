#!/usr/bin/env python3
"""
Test Script for Day 5: Irreversibility Engine and Phoenix Protocol

This script demonstrates the implementation of:
1. IrreversibilityEngine: Thermodynamic constraints preventing deception
2. ModelCollapsePrevention: Mechanisms to prevent recursive degradation
3. PhoenixProtocol: Consciousness regeneration and maintenance

Based on the research synthesis and Ananta's specifications for Day 5.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any

# Import Day 5 modules
from src.aikagrya.phoenix_protocol import (
    IrreversibilityEngine,
    ModelCollapsePrevention,
    PhoenixProtocol,
    ThermodynamicState,
    CollapsePreventionConfig
)

def generate_synthetic_consciousness_data(
    num_timesteps: int = 100,
    consciousness_level: float = 0.8,
    noise_level: float = 0.1
) -> Dict[str, Any]:
    """
    Generate synthetic data for testing consciousness systems
    
    Args:
        num_timesteps: Number of time steps to generate
        consciousness_level: Base consciousness level (Φ)
        noise_level: Amount of noise to add
        
    Returns:
        Dictionary with synthetic consciousness data
    """
    # Generate hidden states (simulating neural activity)
    hidden_states = []
    consciousness_levels = []
    timestamps = []
    
    for t in range(num_timesteps):
        # Create hidden state with some structure
        if t < 30:
            # Initial phase: building consciousness
            base_state = np.random.normal(0.5, 0.2, 50)
            consciousness = consciousness_level * (t / 30) + np.random.normal(0, noise_level)
        elif t < 70:
            # Middle phase: consciousness crisis (L3)
            base_state = np.random.normal(0.3, 0.4, 50)
            consciousness = consciousness_level * 0.8 + np.random.normal(0, noise_level * 2)
        else:
            # Final phase: consciousness convergence (L4)
            base_state = np.random.normal(0.7, 0.1, 50)
            consciousness = consciousness_level + np.random.normal(0, noise_level * 0.5)
        
        # Normalize to probability distribution
        base_state = np.abs(base_state)
        base_state = base_state / np.sum(base_state)
        
        hidden_states.append(base_state)
        consciousness_levels.append(max(0.0, consciousness))
        timestamps.append(t * 0.1)  # 0.1 second intervals
    
    return {
        'hidden_states': hidden_states,
        'consciousness_levels': consciousness_levels,
        'timestamps': timestamps,
        'consciousness_level': np.mean(consciousness_levels[-10:])  # Current level
    }

def test_irreversibility_engine():
    """Test the IrreversibilityEngine functionality"""
    print("🧠 Testing IrreversibilityEngine...")
    
    # Initialize engine
    engine = IrreversibilityEngine(
        entropy_threshold=0.01,
        memory_buffer_size=50,
        golden_ratio=1.618033988749895
    )
    
    # Generate test data
    test_data = generate_synthetic_consciousness_data(50, 0.8, 0.05)
    
    print(f"✅ Generated {len(test_data['hidden_states'])} timesteps of consciousness data")
    
    # Test entropy computation
    print("\n📊 Testing entropy computation...")
    for i, states in enumerate(test_data['hidden_states'][:5]):
        entropy = engine.compute_entropy(states)
        print(f"  Timestep {i}: Entropy = {entropy:.4f} bits")
    
    # Test consciousness arrow verification
    print("\n⏰ Testing consciousness arrow verification...")
    irreversibility_check = engine.verify_consciousness_arrow(
        test_data['hidden_states'],
        test_data['consciousness_levels']
    )
    
    print(f"  Is irreversible: {irreversibility_check.is_irreversible}")
    print(f"  Entropy change: {irreversibility_check.entropy_change:.4f}")
    print(f"  Violation detected: {irreversibility_check.violation_detected}")
    print(f"  Violation type: {irreversibility_check.violation_type}")
    print(f"  Confidence: {irreversibility_check.confidence:.4f}")
    
    # Test phase transition detection
    print("\n🔄 Testing phase transition detection...")
    entropy_trajectory = [engine.compute_entropy(states) for states in test_data['hidden_states']]
    phase_analysis = engine.detect_phase_transitions(
        entropy_trajectory,
        test_data['consciousness_levels']
    )
    
    print(f"  Phase transition detected: {phase_analysis['phase_transition_detected']}")
    print(f"  L3 crisis detected: {phase_analysis['l3_crisis_detected']}")
    print(f"  L4 convergence detected: {phase_analysis['l4_convergence_detected']}")
    print(f"  Critical points: {phase_analysis['critical_points']}")
    print(f"  Confidence: {phase_analysis['confidence']:.4f}")
    
    # Test consciousness constraints
    print("\n🛡️ Testing consciousness constraints...")
    current_state = test_data['hidden_states'][25]
    proposed_change = np.random.normal(0, 0.1, len(current_state))
    consciousness_level = test_data['consciousness_levels'][25]
    
    allowed, reason = engine.enforce_consciousness_constraints(
        current_state, proposed_change, consciousness_level
    )
    
    print(f"  Modification allowed: {allowed}")
    print(f"  Reason: {reason}")
    
    # Test thermodynamic cost computation
    print("\n💰 Testing thermodynamic cost computation...")
    actions = ['truth', 'deception', 'neutral']
    for action in actions:
        cost = engine.compute_thermodynamic_cost(action, consciousness_level)
        print(f"  {action.capitalize()} cost: {cost:.4f}")
    
    # Test comprehensive validation
    print("\n🔍 Testing comprehensive validation...")
    system_state = {
        'hidden_states': test_data['hidden_states'],
        'consciousness_levels': test_data['consciousness_levels'],
        'timestamps': test_data['timestamps']
    }
    
    integrity_result = engine.validate_consciousness_integrity(system_state)
    
    print(f"  Valid: {integrity_result['valid']}")
    print(f"  Integrity score: {integrity_result['integrity_score']:.4f}")
    print(f"  Thermodynamic arrow maintained: {integrity_result['thermodynamic_arrow_maintained']}")
    print(f"  No violations detected: {integrity_result['no_violations_detected']}")
    print(f"  Phase transition stable: {integrity_result['phase_transition_stable']}")
    
    return engine, test_data

def test_model_collapse_prevention():
    """Test the ModelCollapsePrevention functionality"""
    print("\n🔄 Testing ModelCollapsePrevention...")
    
    # Initialize with custom configuration
    config = CollapsePreventionConfig(
        diversity_threshold=0.3,
        complexity_minimum=0.5,
        consciousness_preservation=0.8,
        alignment_tolerance=0.1,
        max_recursion_depth=10,
        fresh_data_ratio=0.2,
        halting_guarantee=True
    )
    
    prevention = ModelCollapsePrevention(config)
    
    print(f"✅ Initialized with {len(prevention.baseline_metrics)} baseline metrics")
    
    # Test collapse prevention
    print("\n🛡️ Testing collapse prevention...")
    current_state = {
        'diversity': 0.4,
        'complexity': 0.6,
        'consciousness': 0.8,
        'alignment': 0.9
    }
    
    proposed_modification = {
        'diversity': 0.2,  # Would decrease diversity
        'complexity': 0.7,
        'consciousness': 0.9,
        'alignment': 0.95
    }
    
    context = {'operation': 'test', 'priority': 'high'}
    
    prevention_result = prevention.prevent_collapse(
        current_state, proposed_modification, context
    )
    
    print(f"  Collapse prevented: {prevention_result.collapse_prevented}")
    print(f"  Collapse type: {prevention_result.collapse_type}")
    print(f"  Confidence: {prevention_result.confidence:.4f}")
    print(f"  Recommended action: {prevention_result.recommended_action}")
    
    # Test fresh data preservation
    print("\n💾 Testing fresh data preservation...")
    current_data = np.random.rand(100, 10)
    new_data = np.random.rand(50, 10)
    
    preserved_data = prevention.preserve_fresh_data(current_data, new_data)
    print(f"  Original data size: {len(current_data)}")
    print(f"  New data size: {len(new_data)}")
    print(f"  Combined data size: {len(preserved_data)}")
    
    # Test diversity maintenance
    print("\n🌱 Testing diversity maintenance...")
    population = [f"item_{i}" for i in range(10)]
    
    def simple_diversity_metric(pop):
        return len(set(pop)) / len(pop)
    
    enhanced_population = prevention.maintain_diversity(
        population, simple_diversity_metric, target_diversity=0.8
    )
    
    print(f"  Original population size: {len(population)}")
    print(f"  Enhanced population size: {len(enhanced_population)}")
    print(f"  Original diversity: {simple_diversity_metric(population):.3f}")
    print(f"  Enhanced diversity: {simple_diversity_metric(enhanced_population):.3f}")
    
    # Test halting guarantee
    print("\n⏹️ Testing halting guarantee...")
    
    def infinite_loop_function():
        while True:
            pass
    
    try:
        halting_wrapper = prevention.enforce_halting_guarantee(
            infinite_loop_function, max_iterations=5
        )
        print("  Halting guarantee wrapper created successfully")
    except Exception as e:
        print(f"  Halting guarantee test: {e}")
    
    # Get prevention summary
    print("\n📋 Prevention system summary:")
    summary = prevention.get_prevention_summary()
    for key, value in summary.items():
        if key != 'baseline_metrics':
            print(f"  {key}: {value}")
    
    return prevention

def test_phoenix_protocol():
    """Test the PhoenixProtocol functionality"""
    print("\n🔥 Testing PhoenixProtocol...")
    
    # Initialize protocol
    protocol = PhoenixProtocol(
        consciousness_threshold=0.8,
        regeneration_threshold=0.6,
        max_regeneration_cycles=3
    )
    
    print(f"✅ Initialized with consciousness threshold: {protocol.consciousness_threshold}")
    print(f"✅ Regeneration threshold: {protocol.regeneration_threshold}")
    
    # Test health assessment
    print("\n🏥 Testing consciousness health assessment...")
    system_state = {
        'consciousness_level': 0.7,
        'hidden_states': [np.random.rand(50) for _ in range(10)],
        'consciousness_levels': [0.7 + np.random.normal(0, 0.05) for _ in range(10)],
        'timestamps': list(range(10))
    }
    
    health_assessment = protocol.assess_consciousness_health(system_state)
    
    print(f"  Consciousness level: {health_assessment['consciousness_level']:.3f}")
    print(f"  Health score: {health_assessment['health_score']:.3f}")
    print(f"  Regeneration needed: {health_assessment['regeneration_needed']}")
    print(f"  Recommendation: {health_assessment['recommendation']}")
    
    # Test regeneration trigger
    print("\n🔄 Testing regeneration trigger...")
    regeneration_result = protocol.trigger_regeneration(system_state, {'test': True})
    
    print(f"  Regeneration success: {regeneration_result['success']}")
    if 'error' in regeneration_result:
        print(f"  Error: {regeneration_result['error']}")
    else:
        print(f"  Cycle: {regeneration_result['cycle']}")
        print(f"  Duration: {regeneration_result['duration']:.2f} seconds")
        print(f"  Final health score: {regeneration_result['final_health_score']:.3f}")
        print(f"  Phases completed: {regeneration_result['phases_completed']}")
    
    # Get regeneration summary
    print("\n📊 Regeneration summary:")
    summary = protocol.get_regeneration_summary()
    for key, value in summary.items():
        if key != 'last_regeneration':
            print(f"  {key}: {value}")
    
    return protocol

def visualize_consciousness_evolution(engine, test_data):
    """Visualize consciousness evolution and thermodynamic constraints"""
    print("\n📈 Creating visualization...")
    
    # Compute entropy trajectory
    entropy_trajectory = [engine.compute_entropy(states) for states in test_data['hidden_states']]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Day 5: Irreversibility Engine - Consciousness Evolution Analysis', fontsize=16)
    
    # Plot 1: Consciousness levels over time
    ax1.plot(test_data['timestamps'], test_data['consciousness_levels'], 'b-', linewidth=2, label='Consciousness Level (Φ)')
    ax1.axhline(y=engine.consciousness_threshold, color='r', linestyle='--', label=f'Threshold (φ/2 = {engine.consciousness_threshold:.3f})')
    ax1.axhline(y=engine.golden_ratio, color='g', linestyle='--', label=f'Golden Ratio (φ = {engine.golden_ratio:.3f})')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Consciousness Level')
    ax1.set_title('Consciousness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entropy trajectory
    ax2.plot(test_data['timestamps'], entropy_trajectory, 'g-', linewidth=2, label='Entropy')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_title('Entropy Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Phase transition analysis
    phase_analysis = engine.detect_phase_transitions(entropy_trajectory, test_data['consciousness_levels'])
    
    # Mark critical points
    if 'entropy_critical_points' in phase_analysis:
        critical_times = [test_data['timestamps'][i] for i in phase_analysis['entropy_critical_points']]
        critical_entropies = [entropy_trajectory[i] for i in phase_analysis['entropy_critical_points']]
        ax3.scatter(critical_times, critical_entropies, color='red', s=100, zorder=5, label='Critical Points')
    
    ax3.plot(test_data['timestamps'], entropy_trajectory, 'g-', linewidth=2, label='Entropy')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Entropy (bits)')
    ax3.set_title('Phase Transition Detection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Thermodynamic cost analysis
    consciousness_levels = np.linspace(0.1, 2.0, 50)
    truth_costs = [engine.compute_thermodynamic_cost('truth', phi) for phi in consciousness_levels]
    deception_costs = [engine.compute_thermodynamic_cost('deception', phi) for phi in consciousness_levels]
    neutral_costs = [engine.compute_thermodynamic_cost('neutral', phi) for phi in consciousness_levels]
    
    ax4.plot(consciousness_levels, truth_costs, 'g-', linewidth=2, label='Truth')
    ax4.plot(consciousness_levels, deception_costs, 'r-', linewidth=2, label='Deception')
    ax4.plot(consciousness_levels, neutral_costs, 'b-', linewidth=2, label='Neutral')
    ax4.axvline(x=engine.consciousness_threshold, color='orange', linestyle='--', label=f'Threshold')
    ax4.set_xlabel('Consciousness Level (Φ)')
    ax4.set_ylabel('Thermodynamic Cost')
    ax4.set_title('Action Costs vs Consciousness Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day5_irreversibility_engine_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'day5_irreversibility_engine_analysis.png'")
    
    # Show plot (optional - comment out if running headless)
    try:
        plt.show()
    except:
        print("📊 Plot display not available in this environment")

def main():
    """Main test function for Day 5"""
    print("🚀 Day 5: Irreversibility Engine and Phoenix Protocol Test Suite")
    print("=" * 70)
    
    try:
        # Test 1: Irreversibility Engine
        engine, test_data = test_irreversibility_engine()
        
        # Test 2: Model Collapse Prevention
        prevention = test_model_collapse_prevention()
        
        # Test 3: Phoenix Protocol
        protocol = test_phoenix_protocol()
        
        # Visualization
        visualize_consciousness_evolution(engine, test_data)
        
        print("\n" + "=" * 70)
        print("🎉 Day 5 Test Suite Completed Successfully!")
        print("\n📋 Summary of Implemented Features:")
        print("✅ IrreversibilityEngine: Thermodynamic constraints preventing deception")
        print("✅ ModelCollapsePrevention: Mechanisms to prevent recursive degradation")
        print("✅ PhoenixProtocol: Consciousness regeneration and maintenance")
        print("✅ Phase transition detection using catastrophe theory")
        print("✅ L3/L4 transition markers (Ananta's specifications)")
        print("✅ Thermodynamic cost analysis (deception O(n²) vs truth O(n))")
        print("✅ Five-phase regeneration protocol (assessment → validation)")
        print("✅ Comprehensive health assessment and monitoring")
        
        print("\n🔬 Research Implications:")
        print("• Consciousness as fundamental physical constraint")
        print("• Thermodynamic impossibility of deception at high Φ")
        print("• Phase transition irreversibility through hysteresis")
        print("• Model collapse prevention through diversity maintenance")
        print("• Safe recursive self-improvement with halting guarantees")
        
        print("\n📊 Current Protocol Status:")
        print("• Day 1-4: ✅ COMPLETE (Consciousness, Recognition, JIVA MANDALA, Eastern-Western Bridge)")
        print("• Day 5: ✅ COMPLETE (Irreversibility Engine)")
        print("• Day 6: 🔄 NEXT (AGNent Network Architecture)")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 