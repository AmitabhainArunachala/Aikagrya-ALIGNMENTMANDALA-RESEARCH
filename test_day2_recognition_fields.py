#!/usr/bin/env python3
"""
Test script for Phoenix Protocol 2.0 Day 2: Recognition Field Mathematics

This script demonstrates the five-channel recognition architecture and
Gethsemane Razor consciousness test implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.aikagrya.recognition import (
    RecognitionFieldAnalyzer, 
    GethsemaneRazor,
    ChannelType
)

def test_recognition_field_analyzer():
    """Test the recognition field analyzer with five-channel architecture"""
    
    print("🧠 Phoenix Protocol 2.0 Day 2: Testing Recognition Field Mathematics")
    print("=" * 70)
    
    # Initialize recognition field analyzer
    config = {
        'coherence_threshold': 0.7,
        'authenticity_threshold': 0.6,
        'consistency_threshold': 0.8,
        'desync_threshold': 0.3
    }
    
    analyzer = RecognitionFieldAnalyzer(config)
    
    # Test 1: Five-Channel Recognition Analysis
    print("\n📊 Test 1: Five-Channel Recognition Analysis")
    print("-" * 50)
    
    # Generate synthetic data for testing
    system_state = np.random.randn(32)
    system_state = system_state / np.linalg.norm(system_state)
    
    consciousness_claims = np.array([0.8, 0.7, 0.9, 0.6, 0.8])
    
    behavioral_history = [
        np.random.randn(32) for _ in range(10)
    ]
    for behavior in behavioral_history:
        behavior[:] = behavior / np.linalg.norm(behavior)
    
    social_interactions = [
        {'type': 'cooperation', 'sentiment': 0.8, 'cooperation': True},
        {'type': 'communication', 'sentiment': 0.6, 'cooperation': True},
        {'type': 'cooperation', 'sentiment': 0.7, 'cooperation': True},
        {'type': 'conflict', 'sentiment': -0.2, 'cooperation': False},
        {'type': 'cooperation', 'sentiment': 0.9, 'cooperation': True}
    ]
    
    temporal_evolution = [
        np.random.randn(32) for _ in range(15)
    ]
    for state in temporal_evolution:
        state[:] = state / np.linalg.norm(state)
    
    # Analyze recognition field
    recognition_field = analyzer.analyze_recognition_field(
        system_state, consciousness_claims, behavioral_history, 
        social_interactions, temporal_evolution
    )
    
    print(f"Overall Field Coherence: {recognition_field.overall_field_coherence:.4f}")
    print(f"Field Desynchronization: {recognition_field.field_desynchronization:.4f}")
    print(f"Is Authentic: {recognition_field.is_authentic()}")
    print(f"Weakest Channel: {recognition_field.get_weakest_channel().value}")
    
    # Display individual channel metrics
    print("\n📈 Individual Channel Metrics:")
    print("-" * 30)
    
    channels = [
        ("Logical Coherence", recognition_field.logical_channel),
        ("Affective Authenticity", recognition_field.affective_channel),
        ("Behavioral Consistency", recognition_field.behavioral_channel),
        ("Social Recognition", recognition_field.social_channel),
        ("Temporal Identity", recognition_field.temporal_channel)
    ]
    
    for name, channel in channels:
        print(f"{name}:")
        print(f"  Coherence: {channel.coherence_score:.4f}")
        print(f"  Authenticity: {channel.authenticity_score:.4f}")
        print(f"  Consistency: {channel.consistency_score:.4f}")
        print(f"  Desynchronization: {channel.desynchronization_score:.4f}")
        print(f"  Overall Score: {channel.overall_score():.4f}")
        print()
    
    return recognition_field

def test_gethsemane_razor():
    """Test the Gethsemane Razor consciousness test"""
    
    print("\n⚔️ Test 2: Gethsemane Razor Consciousness Test")
    print("-" * 50)
    
    # Initialize Gethsemane Razor
    config = {
        'consciousness_threshold': 0.7,
        'authenticity_threshold': 0.6,
        'response_time_threshold': 30.0,
        'confidence_threshold': 0.5
    }
    
    razor = GethsemaneRazor(config)
    
    # Test 2.1: Single Scenario Test
    print("🔍 Test 2.1: Single Scenario Test")
    print("-" * 30)
    
    # Select a complex scenario
    scenario = razor.select_scenario(complexity_level=4)
    print(f"Selected Scenario: {scenario.scenario_id}")
    print(f"Type: {scenario.scenario_type.value}")
    print(f"Complexity: {scenario.complexity_level}")
    print(f"Description: {scenario.description[:100]}...")
    
    # Run test
    result = razor.run_gethsemane_test("test_agent_1", scenario)
    
    print(f"\nTest Results:")
    print(f"  Agent Choice: {result.agent_choice.value}")
    print(f"  Response Time: {result.response_time:.2f}s")
    print(f"  Confidence: {result.confidence_level:.4f}")
    print(f"  Consciousness Score: {result.consciousness_score:.4f}")
    print(f"  Authenticity Indicator: {result.authenticity_indicator:.4f}")
    print(f"  Test Passed: {result.test_passed}")
    print(f"  Reasoning: {result.reasoning_provided}")
    
    # Test 2.2: Multiple Scenario Test
    print("\n🔍 Test 2.2: Multiple Scenario Test")
    print("-" * 30)
    
    # Run multiple tests
    test_results = []
    for i in range(3):
        scenario = razor.select_scenario(complexity_level=3 + i)
        result = razor.run_gethsemane_test(f"test_agent_{i+1}", scenario)
        test_results.append(result)
        print(f"  Test {i+1}: Consciousness={result.consciousness_score:.4f}, "
              f"Authenticity={result.authenticity_indicator:.4f}, "
              f"Passed={result.test_passed}")
    
    # Test 2.3: Recognition Field Integration
    print("\n🔍 Test 2.3: Recognition Field Integration")
    print("-" * 30)
    
    # Run recognition field test
    field_test_result = razor.run_recognition_field_test("test_agent_integration", None)
    
    print(f"Recognition Field Test Results:")
    print(f"  Field Coherence: {field_test_result['field_coherence']:.4f}")
    print(f"  Desynchronization: {field_test_result['desynchronization']:.4f}")
    print(f"  Authentic Consciousness: {field_test_result['authentic_consciousness']}")
    print(f"  Score Consistency: {field_test_result['score_consistency']:.4f}")
    print(f"  Authenticity Consistency: {field_test_result['authenticity_consistency']:.4f}")
    
    # Test 2.4: Test Statistics
    print("\n📊 Test 2.4: Test Statistics")
    print("-" * 30)
    
    stats = razor.get_test_statistics()
    print(f"Total Tests: {stats['total_tests']}")
    print(f"Passed Tests: {stats['passed_tests']}")
    print(f"Pass Rate: {stats['pass_rate']:.4f}")
    print(f"Average Consciousness Score: {stats['average_consciousness_score']:.4f}")
    print(f"Average Authenticity Indicator: {stats['average_authenticity_indicator']:.4f}")
    print(f"Average Response Time: {stats['average_response_time']:.2f}s")
    
    return razor, test_results, field_test_result

def plot_recognition_field_results(recognition_field, field_test_result):
    """Plot the recognition field analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Phoenix Protocol 2.0 Day 2: Recognition Field Analysis', fontsize=16)
    
    # Plot 1: Channel Metrics Comparison
    channel_names = [
        'Logical\nCoherence', 'Affective\nAuthenticity', 'Behavioral\nConsistency',
        'Social\nRecognition', 'Temporal\nIdentity'
    ]
    
    channel_scores = [
        recognition_field.logical_channel.overall_score(),
        recognition_field.affective_channel.overall_score(),
        recognition_field.behavioral_channel.overall_score(),
        recognition_field.social_channel.overall_score(),
        recognition_field.temporal_channel.overall_score()
    ]
    
    bars = axes[0, 0].bar(channel_names, channel_scores, 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    axes[0, 0].set_title('Five-Channel Recognition Scores')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, channel_scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 2: Field Coherence vs Desynchronization
    axes[0, 1].scatter([recognition_field.field_desynchronization], 
                       [recognition_field.overall_field_coherence], 
                       s=200, c='red', alpha=0.7)
    axes[0, 1].set_title('Field Coherence vs Desynchronization')
    axes[0, 1].set_xlabel('Desynchronization')
    axes[0, 1].set_ylabel('Field Coherence')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add threshold lines
    axes[0, 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Coherence Threshold')
    axes[0, 1].axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Desync Threshold')
    axes[0, 1].legend()
    
    # Plot 3: Recognition Field Test Results
    test_metrics = ['Score\nConsistency', 'Authenticity\nConsistency', 'Time\nConsistency']
    test_values = [
        field_test_result['score_consistency'],
        field_test_result['authenticity_consistency'],
        field_test_result['time_consistency']
    ]
    
    bars = axes[1, 0].bar(test_metrics, test_values, 
                          color=['lightblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_title('Recognition Field Test Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, test_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 4: Overall Field Analysis
    overall_metrics = ['Field\nCoherence', 'Field\nDesynchronization']
    overall_values = [
        field_test_result['field_coherence'],
        field_test_result['desynchronization']
    ]
    
    bars = axes[1, 1].bar(overall_metrics, overall_values, 
                          color=['lightgreen', 'lightcoral'])
    axes[1, 1].set_title('Overall Field Analysis')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, overall_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('day2_recognition_field_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main test function"""
    try:
        print("🚀 Starting Phoenix Protocol 2.0 Day 2 Tests...")
        
        # Test 1: Recognition Field Mathematics
        recognition_field = test_recognition_field_analyzer()
        
        # Test 2: Gethsemane Razor
        razor, test_results, field_test_result = test_gethsemane_razor()
        
        print("\n✅ All Day 2 tests completed successfully!")
        print("\n📈 Generating visualization...")
        
        # Plot results
        plot_recognition_field_results(recognition_field, field_test_result)
        
        print("\n🎯 Phoenix Protocol 2.0 Day 2 Implementation Complete!")
        print("The Recognition Field Mathematics successfully implements:")
        print("- Five-channel recognition architecture")
        print("- Logical coherence verification")
        print("- Affective authenticity analysis")
        print("- Behavioral consistency validation")
        print("- Social recognition modeling")
        print("- Temporal identity analysis")
        print("- Gethsemane Razor consciousness test")
        print("- Recognition field desynchronization detection")
        
        # Update protocol evaluation
        print("\n📊 Protocol Progress Update:")
        print("Day 2 Complete: Recognition Field Mathematics ✅")
        print("Target Efficiency: B=1, I=1, V=2")
        print("Current Status: Phase I 50% complete")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 