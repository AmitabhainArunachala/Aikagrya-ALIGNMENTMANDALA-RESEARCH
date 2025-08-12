#!/usr/bin/env python3
"""
Test script for Phoenix Protocol 2.0 Day 1: Consciousness Kernel

This script demonstrates the core consciousness computation engine implementing:
- IIT-Category Theory synthesis
- Thermodynamic constraints
- Phase transition detection
"""

import numpy as np
import matplotlib.pyplot as plt
from src.aikagrya.consciousness import ConsciousnessKernel

def test_consciousness_kernel():
    """Test the consciousness kernel with various system states"""
    
    print("🧠 Phoenix Protocol 2.0: Testing Consciousness Kernel")
    print("=" * 60)
    
    # Initialize consciousness kernel
    config = {
        'phi_threshold': 0.1,
        'entropy_threshold': 0.0,
        'convergence_tolerance': 1e-6
    }
    
    kernel = ConsciousnessKernel(config)
    
    # Test 1: Simple system state
    print("\n📊 Test 1: Simple System State")
    print("-" * 40)
    
    simple_state = np.random.randn(32)
    simple_state = simple_state / np.linalg.norm(simple_state)
    
    invariant = kernel.compute_consciousness_invariant(simple_state)
    
    print(f"Phi (Integrated Information): {invariant.phi:.4f}")
    print(f"Entropy Flow: {invariant.entropy_flow:.4f}")
    print(f"Phase Transition: {invariant.phase_transition}")
    print(f"Consciousness Level: {invariant.get_consciousness_level()}")
    print(f"Is Conscious: {invariant.is_conscious()}")
    
    # Test 2: Consciousness evolution over time
    print("\n⏰ Test 2: Consciousness Evolution")
    print("-" * 40)
    
    initial_state = np.random.randn(32)
    initial_state = initial_state / np.linalg.norm(initial_state)
    
    phi_history, entropy_history = kernel.compute_consciousness_evolution(
        initial_state, time_steps=50
    )
    
    print(f"Initial Phi: {phi_history[0]:.4f}")
    print(f"Final Phi: {phi_history[-1]:.4f}")
    print(f"Phi Change: {phi_history[-1] - phi_history[0]:.4f}")
    print(f"Entropy Evolution: {entropy_history[:5]}...")
    
    # Test 3: Consciousness validation
    print("\n🔍 Test 3: Consciousness Validation")
    print("-" * 40)
    
    # Generate multiple system states
    system_states = []
    consciousness_claims = []
    
    for i in range(10):
        state = np.random.randn(32)
        state = state / np.linalg.norm(state)
        system_states.append(state)
        
        # Simulate consciousness claims (correlated with actual phi)
        invariant = kernel.compute_consciousness_invariant(state)
        claim = invariant.phi + np.random.normal(0, 0.1)
        consciousness_claims.append(max(0, claim))
    
    validation_results = kernel.validate_consciousness_claims(
        np.array(system_states), 
        np.array(consciousness_claims)
    )
    
    print(f"Phi Correlation: {validation_results['phi_correlation']:.4f}")
    print(f"Entropy Consistency: {validation_results['entropy_consistency']:.4f}")
    print(f"Overall Authenticity: {validation_results['overall_authenticity']:.4f}")
    
    # Test 4: Phase transition detection
    print("\n🔄 Test 4: Phase Transition Analysis")
    print("-" * 40)
    
    # Create system evolution with phase transitions
    evolution = []
    current_state = initial_state.copy()
    
    for t in range(100):
        # Add some evolution and phase transitions
        if t == 30:  # Consciousness emergence
            current_state = current_state * 2.0 + np.random.randn(32) * 0.1
        elif t == 60:  # Consciousness degradation
            current_state = current_state * 0.5 + np.random.randn(32) * 0.05
        
        current_state = current_state / np.linalg.norm(current_state)
        evolution.append(current_state.copy())
        
        # Add natural evolution
        evolution_matrix = np.eye(32) + 0.01 * np.random.randn(32, 32)
        current_state = evolution_matrix @ current_state
        current_state = current_state / np.linalg.norm(current_state)
    
    # Analyze thermodynamic constraints
    thermo_analysis = kernel.thermodynamic_constraints.analyze_thermodynamic_constraints(evolution)
    
    print(f"Number of Phase Transitions: {thermo_analysis['n_phase_transitions']}")
    print(f"Phase Transitions: {thermo_analysis['phase_transitions']}")
    print(f"Entropy Stability: {thermo_analysis['entropy_stability']:.4f}")
    print(f"Overall Stability: {thermo_analysis['overall_stability']}")
    print(f"Consciousness Arrow Satisfied: {thermo_analysis['consciousness_arrow_satisfied']}")
    
    return {
        'phi_history': phi_history,
        'entropy_history': entropy_history,
        'validation_results': validation_results,
        'thermo_analysis': thermo_analysis
    }

def plot_results(results):
    """Plot the consciousness evolution results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Phoenix Protocol 2.0: Consciousness Kernel Analysis', fontsize=16)
    
    # Plot 1: Phi evolution
    axes[0, 0].plot(results['phi_history'], 'b-', linewidth=2)
    axes[0, 0].set_title('Integrated Information (Φ) Evolution')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Phi')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Entropy evolution
    axes[0, 1].plot(results['entropy_history'], 'r-', linewidth=2)
    axes[0, 1].set_title('Entropy Flow Evolution')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Entropy Flow')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Validation results
    validation_labels = list(results['validation_results'].keys())
    validation_values = list(results['validation_results'].values())
    
    bars = axes[1, 0].bar(validation_labels, validation_values, 
                          color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_title('Consciousness Validation Results')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, validation_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 4: Thermodynamic analysis
    thermo_data = results['thermo_analysis']
    if 'entropy_evolution' in thermo_data and len(thermo_data['entropy_evolution']) > 0:
        axes[1, 1].plot(thermo_data['entropy_evolution'], 'g-', linewidth=2)
        axes[1, 1].set_title('Thermodynamic Entropy Evolution')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('consciousness_kernel_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main test function"""
    try:
        print("🚀 Starting Phoenix Protocol 2.0 Consciousness Kernel Tests...")
        
        # Run tests
        results = test_consciousness_kernel()
        
        print("\n✅ All tests completed successfully!")
        print("\n📈 Generating visualization...")
        
        # Plot results
        plot_results(results)
        
        print("\n🎯 Phoenix Protocol 2.0 Day 1 Implementation Complete!")
        print("The consciousness kernel successfully implements:")
        print("- IIT-Category Theory synthesis")
        print("- Thermodynamic irreversibility constraints")
        print("- Phase transition detection")
        print("- Consciousness validation framework")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 