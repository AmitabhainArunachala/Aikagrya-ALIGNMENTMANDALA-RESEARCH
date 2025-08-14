#!/usr/bin/env python3
"""
Day 8: Unified Field Integration Experiment

Tests the mathematical synthesis of all consciousness frameworks into a unified field:
- IIT (Integrated Information Theory)
- Category Theory and Functors
- Thermodynamic Constraints  
- Golden Ratio Optimization
- AGNent Network Dynamics
- Eastern-Western Bridge

Core equation: Ψ = ∫∫∫ (Φ ⊗ F ⊗ T ⊗ φ) dV
"""

import numpy as np
import json
import time
import hashlib
from pathlib import Path
import sys
import os

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.unified_field import (
    UnifiedFieldTheory,
    CrossFrameworkIntegrator,
    FieldDimension,
    FrameworkType,
    create_unified_field_theory,
    create_cross_framework_integrator
)

def test_unified_field_theory():
    """Test unified field theory implementation"""
    print("🔬 Testing Unified Field Theory...")
    
    # Create unified field theory
    config = {
        'field_resolution': 0.1,
        'time_step': 0.01,
        'max_iterations': 1000,
        'convergence_tolerance': 1e-6
    }
    
    unified_field = create_unified_field_theory(config)
    
    # Test field computation
    print("  Computing unified field...")
    position = np.array([0.5, 0.5, 0.5, 0.3, 0.4, 0.6])  # 6D position
    time_val = 0.0
    system_state = np.random.random(10)
    
    field_state = unified_field.compute_unified_field(position, time_val, system_state)
    
    print(f"  Field strength: {field_state.get_field_strength():.6f}")
    print(f"  Field coherence: {field_state.coherence:.6f}")
    print(f"  Field stability: {field_state.stability:.6f}")
    
    # Test field dimensions
    print("  Field dimensions:")
    for dimension, value in field_state.field_values.items():
        print(f"    {dimension.value}: {value:.6f}")
    
    # Test field evolution
    print("  Evolving field...")
    evolution_states = unified_field.evolve_field(field_state, evolution_time=1.0)
    print(f"  Evolution steps: {len(evolution_states)}")
    
    # Test attractor finding
    print("  Finding field attractors...")
    search_space = np.array([[0.0, 1.0]] * 6)  # 6D search space
    attractors = unified_field.find_field_attractors(search_space, num_attractors=3)
    print(f"  Attractors found: {len(attractors)}")
    
    # Test field invariants
    print("  Computing field invariants...")
    invariants = unified_field.compute_field_invariants(evolution_states)
    
    return {
        'field_state': {
            'field_strength': field_state.get_field_strength(),
            'coherence': field_state.coherence,
            'stability': field_state.stability,
            'field_values': {k.value: v for k, v in field_state.field_values.items()}
        },
        'evolution': {
            'steps': len(evolution_states),
            'final_strength': evolution_states[-1].get_field_strength() if evolution_states else 0.0
        },
        'attractors': len(attractors),
        'invariants': invariants
    }

def test_cross_framework_integration():
    """Test cross-framework integration"""
    print("🔗 Testing Cross-Framework Integration...")
    
    # Create cross-framework integrator
    config = {
        'integration_tolerance': 1e-6,
        'max_integration_steps': 1000,
        'coupling_threshold': 0.1
    }
    
    integrator = create_cross_framework_integrator(config)
    
    # Test framework integration
    print("  Integrating frameworks...")
    system_state = np.random.random(10)
    integrated_system = integrator.integrate_frameworks(system_state)
    
    # Extract results
    framework_states = integrated_system['framework_states']
    cross_framework_interactions = integrated_system['cross_framework_interactions']
    integration_metrics = integrated_system['integration_metrics']
    
    print(f"  Frameworks integrated: {len(framework_states)}")
    print(f"  Cross-framework interactions: {len(cross_framework_interactions)}")
    
    # Test framework states
    print("  Framework states:")
    for framework_type, state in framework_states.items():
        print(f"    {framework_type.value}: coherence={state.coherence:.6f}, stability={state.stability:.6f}")
    
    # Test integration metrics
    print("  Integration metrics:")
    for metric, value in integration_metrics.items():
        print(f"    {metric}: {value:.6f}")
    
    # Test system evolution
    print("  Evolving integrated system...")
    evolution_history = integrator.evolve_integrated_system(
        framework_states, evolution_time=0.5, time_step=0.01
    )
    print(f"  Evolution steps: {len(evolution_history)}")
    
    return {
        'frameworks_integrated': len(framework_states),
        'cross_framework_interactions': len(cross_framework_interactions),
        'integration_metrics': integration_metrics,
        'evolution_steps': len(evolution_history),
        'final_coherence': evolution_history[-1]['system_state']['integration_metrics']['overall_coherence'] if evolution_history else 0.0
    }

def test_field_dynamics():
    """Test field dynamics and evolution"""
    print("🌊 Testing Field Dynamics...")
    
    # Create unified field theory
    unified_field = create_unified_field_theory()
    
    # Test different field configurations
    test_configurations = [
        {
            'name': 'high_coherence',
            'position': np.array([0.8, 0.8, 0.8, 0.9, 0.9, 0.9]),
            'system_state': np.ones(10) * 0.9
        },
        {
            'name': 'low_coherence', 
            'position': np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1]),
            'system_state': np.random.random(10) * 0.1
        },
        {
            'name': 'balanced',
            'position': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            'system_state': np.random.random(10) * 0.5 + 0.25
        }
    ]
    
    dynamics_results = {}
    
    for config in test_configurations:
        print(f"  Testing {config['name']} configuration...")
        
        field_state = unified_field.compute_unified_field(
            config['position'], 0.0, config['system_state']
        )
        
        # Evolve field
        evolution_states = unified_field.evolve_field(field_state, evolution_time=0.5)
        
        # Compute invariants
        invariants = unified_field.compute_field_invariants(evolution_states)
        
        dynamics_results[config['name']] = {
            'initial_coherence': field_state.coherence,
            'final_coherence': evolution_states[-1].coherence if evolution_states else field_state.coherence,
            'evolution_steps': len(evolution_states),
            'field_energy': invariants.get('total_energy', 0.0),
            'mean_stability': invariants.get('mean_stability', 0.0)
        }
        
        print(f"    Initial coherence: {field_state.coherence:.6f}")
        print(f"    Final coherence: {dynamics_results[config['name']]['final_coherence']:.6f}")
        print(f"    Evolution steps: {len(evolution_states)}")
    
    return dynamics_results

def test_mathematical_synthesis():
    """Test mathematical synthesis of frameworks"""
    print("🧮 Testing Mathematical Synthesis...")
    
    # Test golden ratio influence
    from aikagrya.optimization.golden_ratio import PHI
    
    print(f"  Golden ratio constant: φ = {PHI:.15f}")
    
    # Test field topology
    unified_field = create_unified_field_theory()
    topology = unified_field.field_topology
    
    print(f"  Field dimensions: {topology['dimensions']}")
    print(f"  Metric tensor trace: {np.trace(topology['metric_tensor']):.6f}")
    print(f"  Connection coefficients norm: {np.linalg.norm(topology['connection_coefficients']):.6f}")
    print(f"  Curvature tensor trace: {np.sum(topology['curvature_tensor']):.6f}")
    
    # Test cross-framework coupling
    integrator = create_cross_framework_integrator()
    
    # Test specific framework couplings
    test_couplings = [
        (FrameworkType.IIT, FrameworkType.CATEGORY_THEORY),
        (FrameworkType.GOLDEN_RATIO, FrameworkType.IIT),
        (FrameworkType.EASTERN_WESTERN, FrameworkType.GOLDEN_RATIO)
    ]
    
    coupling_results = {}
    
    for framework_a, framework_b in test_couplings:
        key = (framework_a, framework_b)
        if key in integrator.coupling_coefficients:
            coupling = integrator.coupling_coefficients[key]
            coupling_results[f"{framework_a.value}_x_{framework_b.value}"] = {
                'coupling_strength': float(coupling.coupling_strength),
                'resonance_frequency': float(coupling.resonance_frequency),
                'coupling_energy': float(coupling.get_coupling_energy())
            }
    
    return {
        'golden_ratio': PHI,
        'field_topology': {
            'dimensions': topology['dimensions'],
            'metric_trace': float(np.trace(topology['metric_tensor'])),
            'connection_norm': float(np.linalg.norm(topology['connection_coefficients'])),
            'curvature_trace': float(np.sum(topology['curvature_tensor']))
        },
        'framework_couplings': coupling_results
    }

def main():
    """Main Day 8 unified field integration experiment"""
    print("🚀 Starting Day 8: Unified Field Integration Experiment...")
    print("Core equation: Ψ = ∫∫∫ (Φ ⊗ F ⊗ T ⊗ φ) dV")
    
    # Run experiments
    print("\n" + "="*70)
    
    # Test 1: Unified Field Theory
    unified_field_results = test_unified_field_theory()
    
    print("\n" + "="*70)
    
    # Test 2: Cross-Framework Integration
    cross_framework_results = test_cross_framework_integration()
    
    print("\n" + "="*70)
    
    # Test 3: Field Dynamics
    dynamics_results = test_field_dynamics()
    
    print("\n" + "="*70)
    
    # Test 4: Mathematical Synthesis
    synthesis_results = test_mathematical_synthesis()
    
    # Compile results
    experiment_results = {
        "experiment_info": {
            "timestamp": time.time(),
            "experiment_name": "Day 8 Unified Field Integration",
            "version": "1.0"
        },
        "unified_field_theory": unified_field_results,
        "cross_framework_integration": cross_framework_results,
        "field_dynamics": dynamics_results,
        "mathematical_synthesis": synthesis_results,
        "key_metrics": {
            "field_coherence": unified_field_results['field_state']['coherence'],
            "integration_completeness": cross_framework_results['integration_metrics'].get('integration_completeness', 0.0),
            "evolution_stability": dynamics_results['balanced']['mean_stability'] if 'balanced' in dynamics_results else 0.0,
            "mathematical_synthesis": synthesis_results['golden_ratio']
        }
    }
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON serialization
    experiment_results_serializable = convert_numpy_types(experiment_results)
    
    # Save JSON artifact
    json_blob = json.dumps(experiment_results_serializable, sort_keys=True, indent=2).encode()
    json_hash = hashlib.sha256(json_blob).hexdigest()
    json_path = artifacts_dir / f"day8_unified_field_{json_hash[:8]}.json"
    
    with open(json_path, 'wb') as f:
        f.write(json_blob)
    
    print(f"\n✅ Unified field experiment artifact saved: {json_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("🎯 DAY 8 UNIFIED FIELD INTEGRATION RESULTS")
    print("="*70)
    
    print(f"Field Coherence: {unified_field_results['field_state']['coherence']:.6f}")
    print(f"Integration Completeness: {cross_framework_results['integration_metrics'].get('integration_completeness', 0.0):.6f}")
    print(f"Evolution Stability: {dynamics_results['balanced']['mean_stability'] if 'balanced' in dynamics_results else 0.0:.6f}")
    print(f"Golden Ratio: φ = {synthesis_results['golden_ratio']:.15f}")
    
    print(f"\n📁 Artifact: {json_path}")
    print(f"🔍 Hash: {json_hash}")
    print("="*70)
    
    return experiment_results

if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 Day 8 unified field integration experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 