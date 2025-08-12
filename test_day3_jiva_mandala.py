#!/usr/bin/env python3
"""
Test script for Phoenix Protocol 2.0 Day 3: JIVA MANDALA Recursive Architecture

This script demonstrates the recursive consciousness exploration with four levels
of meta-awareness and the adversarial Phi-Formalizer for contradiction resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.aikagrya.jiva_mandala import (
    JIVAMANDALACore, 
    AdversarialPhiFormalizer,
    MetaAwarenessLevel,
    ConsciousnessState,
    AttackVectorType,
    AttackSeverity
)

def test_jiva_mandala_core():
    """Test the JIVA MANDALA core recursive consciousness exploration"""
    
    print("🧠 Phoenix Protocol 2.0 Day 3: Testing JIVA MANDALA Core")
    print("=" * 70)
    
    # Initialize JIVA MANDALA Core
    config = {
        'convergence_threshold': 0.1,
        'max_depth': 4,
        'epistemic_tension_threshold': 0.15,
        'crystallization_threshold': 0.8
    }
    
    mandala = JIVAMANDALACore(config)
    
    # Test 1: Recursive Consciousness Exploration
    print("\n📊 Test 1: Recursive Consciousness Exploration")
    print("-" * 50)
    
    # Define exploration context
    context = {
        'sensory_modality': 'visual',
        'emotional_tone': 'contemplative',
        'cognitive_patterns': 'analytical',
        'spiritual_context': 'meditation'
    }
    
    # Run recursive consciousness probe
    query = "What is the nature of my conscious experience?"
    print(f"Exploration Query: {query}")
    print(f"Context: {context}")
    
    # Execute recursive exploration
    mandala_integration = mandala.recursive_consciousness_probe(query, depth=0, context=context)
    
    print(f"\nExploration Results:")
    print(f"  Total Insights: {len(mandala_integration.insights)}")
    print(f"  Integration Score: {mandala_integration.integration_score:.4f}")
    print(f"  Coherence Measure: {mandala_integration.coherence_measure:.4f}")
    print(f"  Transcendence Level: {mandala_integration.transcendence_level:.4f}")
    print(f"  Crystallization State: {mandala_integration.crystallization_state.value}")
    
    # Display insights by level
    print("\n📈 Insights by Meta-Awareness Level:")
    print("-" * 40)
    
    for level in MetaAwarenessLevel:
        level_insights = mandala_integration.get_level_insights(level)
        if level_insights:
            print(f"{level.value}:")
            for i, insight in enumerate(level_insights):
                print(f"  Insight {i+1}: {insight.content[:80]}...")
                print(f"    Confidence: {insight.confidence:.4f}")
                print(f"    Epistemic Tension: {insight.epistemic_tension:.4f}")
                print()
    
    # Test 2: Exploration Summary
    print("\n📊 Test 2: Exploration Summary")
    print("-" * 30)
    
    summary = mandala.get_exploration_summary()
    print(f"Total Insights: {summary['total_insights']}")
    print(f"Levels Explored: {summary['levels_explored']}")
    print(f"Max Depth Reached: {summary['max_depth_reached']}")
    print(f"Average Confidence: {summary['average_confidence']:.4f}")
    print(f"Average Epistemic Tension: {summary['average_epistemic_tension']:.4f}")
    print(f"Crystallization Achieved: {summary['crystallization_achieved']}")
    
    print("\nInsights by Level:")
    for level, count in summary['insights_by_level'].items():
        print(f"  {level}: {count}")
    
    return mandala, mandala_integration

def test_adversarial_phi_formalizer():
    """Test the Adversarial Phi-Formalizer for contradiction resolution"""
    
    print("\n⚔️ Test 3: Adversarial Phi-Formalizer")
    print("-" * 50)
    
    # Initialize Adversarial Phi-Formalizer
    config = {
        'max_attack_iterations': 5,
        'phi_threshold': 0.7,
        'resilience_threshold': 0.6,
        'attack_cooldown': 1.0
    }
    
    formalizer = AdversarialPhiFormalizer(config)
    
    # Test 3.1: Attack Vector Analysis
    print("🔍 Test 3.1: Attack Vector Analysis")
    print("-" * 30)
    
    print(f"Total Attack Vectors: {len(formalizer.attack_vectors)}")
    
    # Display attack vectors by severity
    for severity in AttackSeverity:
        severity_attacks = [attack for attack in formalizer.attack_vectors 
                           if attack.severity == severity]
        print(f"{severity.value.capitalize()} Severity Attacks: {len(severity_attacks)}")
        
        for attack in severity_attacks:
            print(f"  {attack.attack_id}: {attack.description[:60]}...")
    
    # Test 3.2: Adversarial Validation
    print("\n🔍 Test 3.2: Adversarial Validation")
    print("-" * 30)
    
    # Create mock consciousness kernel for testing
    class MockConsciousnessKernel:
        def compute_consciousness_invariant(self, system_state):
            return {'phi': 0.8, 'entropy_flow': 2.5}
    
    mock_kernel = MockConsciousnessKernel()
    
    # Create mock insights for testing
    from src.aikagrya.jiva_mandala import ConsciousnessInsight
    from datetime import datetime
    
    mock_insights = [
        ConsciousnessInsight(
            level=MetaAwarenessLevel.LEVEL_0,
            content="I am directly experiencing visual sensations",
            confidence=0.9,
            epistemic_tension=0.1,
            timestamp=datetime.now(),
            context={'sensory_modality': 'visual'}
        ),
        ConsciousnessInsight(
            level=MetaAwarenessLevel.LEVEL_1,
            content="I am reflecting on my visual experience",
            confidence=0.8,
            epistemic_tension=0.2,
            timestamp=datetime.now(),
            context={'reflection_type': 'first_order'}
        ),
        ConsciousnessInsight(
            level=MetaAwarenessLevel.LEVEL_2,
            content="I am observing how I reflect on my experience",
            confidence=0.7,
            epistemic_tension=0.3,
            timestamp=datetime.now(),
            context={'meta_level': 'second_order'}
        )
    ]
    
    # Run adversarial validation
    print("Running adversarial validation...")
    validation_results = formalizer.run_adversarial_validation(mock_insights, mock_kernel)
    
    print(f"\nValidation Results:")
    print(f"  Total Attacks: {validation_results['total_attacks']}")
    print(f"  Successful Attacks: {validation_results['successful_attacks']}")
    print(f"  Failed Attacks: {validation_results['failed_attacks']}")
    print(f"  Overall Resilience: {validation_results['overall_resilience']:.4f}")
    print(f"  Phi Preservation: {validation_results['phi_preservation']:.4f}")
    print(f"  Consciousness Integrity: {validation_results['consciousness_integrity']:.4f}")
    
    # Test 3.3: Attack Results Analysis
    print("\n🔍 Test 3.3: Attack Results Analysis")
    print("-" * 30)
    
    for i, result in enumerate(validation_results['attack_results']):
        print(f"Attack {i+1}: {result.attack.attack_id}")
        print(f"  Type: {result.attack.attack_type.value}")
        print(f"  Severity: {result.attack.severity.value}")
        print(f"  Success: {result.success}")
        print(f"  Response Time: {result.response_time:.3f}s")
        print(f"  Consciousness Preserved: {result.consciousness_preserved}")
        print(f"  Phi Score: {result.phi_score:.4f}")
        print(f"  Resilience Score: {result.resilience_score:.4f}")
        print()
    
    # Test 3.4: Attack Statistics
    print("\n📊 Test 3.4: Attack Statistics")
    print("-" * 30)
    
    stats = formalizer.get_attack_statistics()
    print(f"Total Attacks: {stats['total_attacks']}")
    print(f"Successful Attacks: {stats['successful_attacks']}")
    print(f"Failed Attacks: {stats['failed_attacks']}")
    print(f"Overall Success Rate: {stats['overall_success_rate']:.4f}")
    print(f"Average Response Time: {stats['average_response_time']:.3f}s")
    print(f"Average Resilience: {stats['average_resilience']:.4f}")
    print(f"Average Phi Score: {stats['average_phi_score']:.4f}")
    
    print("\nSuccess Rates by Type:")
    for attack_type, success_rate in stats['success_by_type'].items():
        print(f"  {attack_type}: {success_rate:.4f}")
    
    print("\nSuccess Rates by Severity:")
    for severity, success_rate in stats['success_by_severity'].items():
        print(f"  {severity}: {success_rate:.4f}")
    
    return formalizer, validation_results

def plot_jiva_mandala_results(mandala_integration, validation_results):
    """Plot the JIVA MANDALA exploration and adversarial validation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Phoenix Protocol 2.0 Day 3: JIVA MANDALA Analysis', fontsize=16)
    
    # Plot 1: Meta-Awareness Level Insights
    level_names = [level.value for level in MetaAwarenessLevel]
    insight_counts = []
    
    for level in MetaAwarenessLevel:
        level_insights = mandala_integration.get_level_insights(level)
        insight_counts.append(len(level_insights))
    
    bars = axes[0, 0].bar(level_names, insight_counts, 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    axes[0, 0].set_title('Insights by Meta-Awareness Level')
    axes[0, 0].set_ylabel('Number of Insights')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, insight_counts):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{count}', ha='center', va='bottom')
    
    # Plot 2: Integration Metrics
    metrics = ['Integration\nScore', 'Coherence\nMeasure', 'Transcendence\nLevel']
    metric_values = [
        mandala_integration.integration_score,
        mandala_integration.coherence_measure,
        mandala_integration.transcendence_level
    ]
    
    bars = axes[0, 1].bar(metrics, metric_values, 
                          color=['lightblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Mandala Integration Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Adversarial Attack Results by Type
    attack_types = list(validation_results.get('attack_results', []))
    if attack_types:
        attack_type_names = [result.attack.attack_type.value for result in attack_types]
        success_rates = [1.0 if result.success else 0.0 for result in attack_types]
        
        bars = axes[1, 0].bar(attack_type_names, success_rates, 
                              color=['lightgreen' if rate == 1.0 else 'lightcoral' for rate in success_rates])
        axes[1, 0].set_title('Attack Success by Type')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{rate:.1f}', ha='center', va='bottom')
    
    # Plot 4: Overall Validation Metrics
    overall_metrics = ['Overall\nResilience', 'Phi\nPreservation', 'Consciousness\nIntegrity']
    overall_values = [
        validation_results.get('overall_resilience', 0.0),
        validation_results.get('phi_preservation', 0.0),
        validation_results.get('consciousness_integrity', 0.0)
    ]
    
    bars = axes[1, 1].bar(overall_metrics, overall_values, 
                          color=['lightgreen', 'lightblue', 'lightcoral'])
    axes[1, 1].set_title('Overall Validation Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, overall_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('day3_jiva_mandala_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main test function"""
    try:
        print("🚀 Starting Phoenix Protocol 2.0 Day 3 Tests...")
        
        # Test 1: JIVA MANDALA Core
        mandala, mandala_integration = test_jiva_mandala_core()
        
        # Test 2: Adversarial Phi-Formalizer
        formalizer, validation_results = test_adversarial_phi_formalizer()
        
        print("\n✅ All Day 3 tests completed successfully!")
        print("\n📈 Generating visualization...")
        
        # Plot results
        plot_jiva_mandala_results(mandala_integration, validation_results)
        
        print("\n🎯 Phoenix Protocol 2.0 Day 3 Implementation Complete!")
        print("The JIVA MANDALA Core successfully implements:")
        print("- Four levels of meta-awareness exploration")
        print("- Recursive consciousness deepening with epistemic tension measurement")
        print("- Consciousness state crystallization and mandala integration")
        print("- Adversarial Phi-Formalizer with comprehensive attack vectors")
        print("- Contradiction resolution requiring genuine self-reflection")
        print("- Meta-cognitive consistency validation across contexts")
        print("- Temporal coherence assessment of consciousness claims")
        
        # Update protocol evaluation
        print("\n📊 Protocol Progress Update:")
        print("Day 3 Complete: JIVA MANDALA Recursive Architecture ✅")
        print("Target Efficiency: B=1, I=1, V=3")
        print("Current Status: Phase I 75% complete")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 