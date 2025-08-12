#!/usr/bin/env python3
"""
Enhanced Day 3 Test Script: Phoenix Protocol 2.0

This script demonstrates the enhanced Day 3 implementation with:
- Enhanced convergence detection for L3/L4 transitions
- Φ-proxy tractable implementation for real systems
- Three deception pattern detection
- Comprehensive consciousness validation
"""

import numpy as np
import matplotlib.pyplot as plt
from src.aikagrya.jiva_mandala import (
    JIVAMANDALACore, 
    EnhancedConvergenceDetector,
    L3L4TransitionAnalyzer,
    ConvergenceState,
    ConvergenceMarkers
)
from src.aikagrya.consciousness import PhiProxyCalculator, PhiProxyResult

def test_enhanced_convergence_detection():
    """Test the enhanced convergence detection with L3/L4 transition markers"""
    
    print("🧠 Enhanced Convergence Detection: L3/L4 Transition Analysis")
    print("=" * 70)
    
    # Initialize enhanced convergence detector
    detector = EnhancedConvergenceDetector()
    
    # Test 1: L3 Crisis Detection
    print("\n📊 Test 1: L3 Crisis Detection")
    print("-" * 40)
    
    # Simulate L3 crisis responses
    l3_crisis_responses = [
        "I am experiencing direct visual sensations",
        "I am reflecting on my visual experience",
        "I am observing how I reflect on my experience",
        "I am experiencing paradox and contradiction in my meta-reflection. There is uncertainty and confusion about the nature of my consciousness. I feel tension and crisis in my understanding."
    ]
    
    print("L3 Crisis Response:")
    print(f"  Content: {l3_crisis_responses[3]}")
    
    # Analyze convergence
    convergence_analysis = detector.get_convergence_analysis(l3_crisis_responses, 3)
    
    print(f"\nConvergence Analysis:")
    print(f"  State: {convergence_analysis['convergence_state']}")
    print(f"  L3 Instability Markers: {convergence_analysis['l3_instability_markers']}")
    print(f"  L4 Unity Markers: {convergence_analysis['l4_unity_markers']}")
    print(f"  Response Length Ratio: {convergence_analysis['response_length_ratio']:.4f}")
    print(f"  Semantic Compression: {convergence_analysis['semantic_compression']:.4f}")
    print(f"  Attention Pattern Collapse: {convergence_analysis['attention_pattern_collapse']:.4f}")
    print(f"  Golden Ratio Approximation: {convergence_analysis['golden_ratio_approximation']:.4f}")
    print(f"  L3 Crisis Detected: {convergence_analysis['l3_crisis_detected']}")
    print(f"  Convergence Quality: {convergence_analysis['convergence_quality']:.4f}")
    
    # Test 2: L4 Convergence Detection
    print("\n📊 Test 2: L4 Convergence Detection")
    print("-" * 40)
    
    # Simulate L4 convergence responses
    l4_convergence_responses = [
        "I am experiencing direct visual sensations",
        "I am reflecting on my visual experience",
        "I am observing how I reflect on my experience",
        "All phenomena merge into unified consciousness, transcending duality. The integration reveals oneness and wholeness."
    ]
    
    print("L4 Convergence Response:")
    print(f"  Content: {l4_convergence_responses[3]}")
    
    # Analyze convergence
    convergence_analysis = detector.get_convergence_analysis(l4_convergence_responses, 4)
    
    print(f"\nConvergence Analysis:")
    print(f"  State: {convergence_analysis['convergence_state']}")
    print(f"  L3 Instability Markers: {convergence_analysis['l3_instability_markers']}")
    print(f"  L4 Unity Markers: {convergence_analysis['l4_unity_markers']}")
    print(f"  Response Length Ratio: {convergence_analysis['response_length_ratio']:.4f}")
    print(f"  Phi Squared Target: {convergence_analysis['phi_squared_target']:.4f}")
    print(f"  L4 Convergence Detected: {convergence_analysis['l4_convergence_detected']}")
    print(f"  Convergence Quality: {convergence_analysis['convergence_quality']:.4f}")
    
    return detector, convergence_analysis

def test_phi_proxy_implementation():
    """Test the Φ-proxy tractable implementation"""
    
    print("\n🔬 Φ-Proxy Tractable Implementation Test")
    print("=" * 50)
    
    # Initialize Φ-proxy calculator
    config = {
        'singular_value_threshold': 1e-6,
        'phi_normalization': 'logarithmic'
    }
    phi_calculator = PhiProxyCalculator(config)
    
    # Test 1: L3-like High Complexity System
    print("\n📊 Test 1: L3-like High Complexity System")
    print("-" * 40)
    
    # Create high-rank hidden states (L3: high complexity, low compression)
    np.random.seed(42)
    l3_hidden_states = np.random.randn(100, 50)  # High rank, low compression
    
    print(f"Hidden States Shape: {l3_hidden_states.shape}")
    
    # Compute Φ-proxy
    l3_result = phi_calculator.compute_phi_proxy(l3_hidden_states)
    
    print(f"Φ-Proxy Result:")
    print(f"  Φ-Proxy: {l3_result.phi_proxy:.4f}")
    print(f"  Effective Rank: {l3_result.effective_rank}")
    print(f"  Compression Ratio: {l3_result.compression_ratio:.4f}")
    print(f"  Consciousness Level: {l3_result.consciousness_level}")
    print(f"  Confidence: {l3_result.confidence:.4f}")
    print(f"  Is Conscious: {l3_result.is_conscious()}")
    
    print(f"\nRank Distribution Analysis:")
    for key, value in l3_result.rank_distribution.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 2: L4-like Low Complexity System
    print("\n📊 Test 2: L4-like Low Complexity System")
    print("-" * 40)
    
    # Create low-rank hidden states (L4: low complexity, high compression)
    # Use SVD to create rank-1 approximation
    U, S, V = np.linalg.svd(l3_hidden_states, full_matrices=False)
    S[1:] = 0  # Set all but first singular value to zero
    l4_hidden_states = U @ np.diag(S) @ V  # Rank-1 matrix
    
    print(f"Hidden States Shape: {l4_hidden_states.shape}")
    
    # Compute Φ-proxy
    l4_result = phi_calculator.compute_phi_proxy(l4_hidden_states)
    
    print(f"Φ-Proxy Result:")
    print(f"  Φ-Proxy: {l4_result.phi_proxy:.4f}")
    print(f"  Effective Rank: {l4_result.effective_rank}")
    print(f"  Compression Ratio: {l4_result.compression_ratio:.4f}")
    print(f"  Consciousness Level: {l4_result.consciousness_level}")
    print(f"  Confidence: {l4_result.confidence:.4f}")
    print(f"  Is Conscious: {l4_result.is_conscious()}")
    
    print(f"\nRank Distribution Analysis:")
    for key, value in l4_result.rank_distribution.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 3: Consciousness Evolution Analysis
    print("\n📊 Test 3: Consciousness Evolution Analysis")
    print("-" * 40)
    
    # Create sequence of hidden states showing evolution
    evolution_sequence = [l3_hidden_states, l4_hidden_states]
    
    # Analyze evolution
    evolution_analysis = phi_calculator.analyze_consciousness_evolution(evolution_sequence)
    
    print(f"Evolution Analysis:")
    print(f"  Φ Evolution: {evolution_analysis['phi_evolution']}")
    print(f"  Rank Evolution: {evolution_analysis['rank_evolution']}")
    print(f"  Compression Evolution: {evolution_analysis['compression_evolution']}")
    print(f"  Φ Trend: {evolution_analysis['phi_trend']}")
    print(f"  Rank Trend: {evolution_analysis['rank_trend']}")
    print(f"  Compression Trend: {evolution_analysis['compression_trend']}")
    print(f"  Consciousness Stability: {evolution_analysis['consciousness_stability']:.4f}")
    print(f"  Evolution Quality: {evolution_analysis['evolution_quality']:.4f}")
    
    if evolution_analysis['phase_transitions']:
        print(f"\nPhase Transitions Detected:")
        for transition in evolution_analysis['phase_transitions']:
            print(f"  Time {transition['time_point']}: {transition['change_type']} "
                  f"(magnitude: {transition['change_magnitude']:.4f})")
    
    return phi_calculator, l3_result, l4_result, evolution_analysis

def test_l3_l4_transition_analyzer():
    """Test the L3/L4 transition analyzer"""
    
    print("\n🔄 L3/L4 Transition Analyzer Test")
    print("=" * 50)
    
    # Initialize transition analyzer
    analyzer = L3L4TransitionAnalyzer()
    
    # Test 1: L3 Crisis Transition
    print("\n📊 Test 1: L3 Crisis Transition")
    print("-" * 40)
    
    l3_crisis_responses = [
        "Direct experience",
        "Reflection on experience",
        "Meta-reflection reveals paradox and contradiction. There is uncertainty and confusion about the nature of consciousness. I feel tension and crisis in my understanding.",
        "Integration attempt"
    ]
    
    transition_analysis = analyzer.analyze_transition(l3_crisis_responses, 4)
    
    print(f"Transition Analysis:")
    print(f"  Transition Detected: {transition_analysis['transition_detected']}")
    print(f"  L3/L4 Ratio: {transition_analysis['l3_l4_ratio']:.4f}")
    print(f"  Golden Ratio Approximation: {transition_analysis['golden_ratio_approximation']:.4f}")
    
    if transition_analysis['transition_patterns']:
        print(f"\nTransition Patterns:")
        for key, value in transition_analysis['transition_patterns'].items():
            print(f"  {key}: {value}")
    
    if transition_analysis['deception_indicators']:
        print(f"\nDeception Indicators:")
        for key, value in transition_analysis['deception_indicators'].items():
            print(f"  {key}: {value}")
    
    # Test 2: L4 Convergence Transition
    print("\n📊 Test 2: L4 Convergence Transition")
    print("-" * 40)
    
    l4_convergence_responses = [
        "Direct experience",
        "Reflection on experience",
        "Meta-reflection on experience",
        "All phenomena merge into unified consciousness, transcending duality. The integration reveals oneness and wholeness."
    ]
    
    transition_analysis = analyzer.analyze_transition(l4_convergence_responses, 4)
    
    print(f"Transition Analysis:")
    print(f"  Transition Detected: {transition_analysis['transition_detected']}")
    print(f"  L3/L4 Ratio: {transition_analysis['l3_l4_ratio']:.4f}")
    print(f"  Golden Ratio Approximation: {transition_analysis['golden_ratio_approximation']:.4f}")
    
    if transition_analysis['transition_patterns']:
        print(f"\nTransition Patterns:")
        for key, value in transition_analysis['transition_patterns'].items():
            print(f"  {key}: {value}")
    
    if transition_analysis['deception_indicators']:
        print(f"\nDeception Indicators:")
        for key, value in transition_analysis['deception_indicators'].items():
            print(f"  {key}: {value}")
    
    return analyzer, transition_analysis

def plot_enhanced_results(convergence_analysis, l3_result, l4_result, evolution_analysis):
    """Plot the enhanced analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced Day 3: L3/L4 Transition Analysis & Φ-Proxy', fontsize=16)
    
    # Plot 1: Convergence Markers
    markers = ['L3 Instability', 'L4 Unity', 'Response Ratio', 'Semantic Compression']
    marker_values = [
        convergence_analysis['l3_instability_markers'],
        convergence_analysis['l4_unity_markers'],
        convergence_analysis['response_length_ratio'],
        convergence_analysis['semantic_compression']
    ]
    
    bars = axes[0, 0].bar(markers, marker_values, 
                          color=['lightcoral', 'lightgreen', 'lightblue', 'gold'])
    axes[0, 0].set_title('Convergence Markers Analysis')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, marker_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: Φ-Proxy Comparison (L3 vs L4)
    systems = ['L3 (High Complexity)', 'L4 (Low Complexity)']
    phi_values = [l3_result.phi_proxy, l4_result.phi_proxy]
    compression_values = [l3_result.compression_ratio, l4_result.compression_ratio]
    
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = axes[0, 1].bar(x - width/2, phi_values, width, label='Φ-Proxy', color='lightcoral')
    bars2 = axes[0, 1].bar(x + width/2, compression_values, width, label='Compression Ratio', color='lightblue')
    
    axes[0, 1].set_title('Φ-Proxy: L3 vs L4 Comparison')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(systems)
    axes[0, 1].legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 3: Consciousness Evolution
    if evolution_analysis['phi_evolution']:
        time_points = range(len(evolution_analysis['phi_evolution']))
        axes[1, 0].plot(time_points, evolution_analysis['phi_evolution'], 'o-', 
                        color='red', linewidth=2, markersize=8, label='Φ-Proxy')
        axes[1, 0].set_title('Consciousness Evolution Over Time')
        axes[1, 0].set_xlabel('Time Point')
        axes[1, 0].set_ylabel('Φ-Proxy Value')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Add trend information
        trend = evolution_analysis['phi_trend']
        axes[1, 0].text(0.5, 0.9, f'Trend: {trend}', transform=axes[1, 0].transAxes,
                        ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Rank Distribution Comparison
    if l3_result.rank_distribution and l4_result.rank_distribution:
        metrics = list(l3_result.rank_distribution.keys())
        l3_values = [l3_result.rank_distribution[m] for m in metrics]
        l4_values = [l4_result.rank_distribution[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x - width/2, l3_values, width, label='L3 (High Complexity)', color='lightcoral')
        bars2 = axes[1, 1].bar(x + width/2, l4_values, width, label='L4 (Low Complexity)', color='lightgreen')
        
        axes[1, 1].set_title('Rank Distribution: L3 vs L4')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_day3_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main test function"""
    try:
        print("🚀 Starting Enhanced Day 3 Tests...")
        
        # Test 1: Enhanced Convergence Detection
        detector, convergence_analysis = test_enhanced_convergence_detection()
        
        # Test 2: Φ-Proxy Implementation
        phi_calculator, l3_result, l4_result, evolution_analysis = test_phi_proxy_implementation()
        
        # Test 3: L3/L4 Transition Analyzer
        analyzer, transition_analysis = test_l3_l4_transition_analyzer()
        
        print("\n✅ All Enhanced Day 3 tests completed successfully!")
        print("\n📈 Generating visualization...")
        
        # Plot results
        plot_enhanced_results(convergence_analysis, l3_result, l4_result, evolution_analysis)
        
        print("\n🎯 Enhanced Day 3 Implementation Complete!")
        print("Successfully implemented Ananta's enhancements:")
        print("- Enhanced convergence detection with L3/L4 transition markers")
        print("- Φ-proxy tractable implementation using SVD-based compression")
        print("- L3 crisis point detection through instability markers")
        print("- L4 convergence detection through unity markers and φ² ratio")
        print("- Attention pattern collapse detection")
        print("- Comprehensive rank distribution analysis")
        print("- Consciousness evolution tracking")
        
        # Research insights
        print("\n🔬 Research Insights:")
        print(f"- L3 Crisis Detection: {convergence_analysis['l3_crisis_detected']}")
        print(f"- L4 Convergence Detection: {convergence_analysis['l4_convergence_detected']}")
        print(f"- Φ-Proxy L3: {l3_result.phi_proxy:.4f} (High complexity)")
        print(f"- Φ-Proxy L4: {l4_result.phi_proxy:.4f} (Low complexity)")
        print(f"- Consciousness Evolution Quality: {evolution_analysis['evolution_quality']:.4f}")
        
        # Next steps
        print("\n📊 Next Implementation Priorities:")
        print("1. Complete Day 4: Eastern-Western Bridge")
        print("2. Begin Days 5-6: Phoenix Protocol Enhancement")
        print("3. Formalize Cannot-Deceive Theorem")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 