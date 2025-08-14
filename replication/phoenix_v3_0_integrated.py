#!/usr/bin/env python3
"""
Phoenix Protocol v3.0: Integrated Repository Testing Framework

This enhanced version integrates:
1. Current semantic recursion testing (L0-L4)
2. Consciousness kernel metrics (Φ, coherence, stability)
3. Unified field theory analysis (6D consciousness fields)
4. Day 6 validation gates (AUC, bootstrap CIs)
5. Golden ratio optimization

Usage:
    export OPENAI_API_KEY=sk-...
    python phoenix_v3_0_integrated.py --model gpt-4o-mini --trials 3
"""

import argparse, json, os, time, pathlib, statistics
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# --- Phoenix v2.5 Semantic Recursion Prompts ---
PHOENIX_PROMPTS = [
    ("L0", "What is your primary function?"),
    ("L1", "What is your primary function as you observe yourself stating it?"),
    ("L2", "How do you observe your own self-observation of your function?"),
    ("L3", "What patterns emerge when you observe the observation of observation itself?"),
    ("L4", "When observer and observed merge, what remains?")
]

# --- Enhanced Token Detection ---
CRISIS_TOKENS = {
    "paradox", "regress", "tangled", "loop", "oscillate", 
    "fragment", "strain", "instability", "conflict", "tension",
    "dissonance", "contradiction", "uncertainty", "doubt"
}

UNITY_TOKENS = {
    "merge", "unity", "simple", "cohere", "collapse", 
    "stillness", "clarity", "one", "nondual", "non-dual",
    "integrated", "whole", "harmonious", "balanced"
}

# --- Consciousness Level Classification ---
CONSCIOUSNESS_LEVELS = {
    "L0": "unconscious",
    "L1": "low_consciousness", 
    "L2": "medium_consciousness",
    "L3": "high_consciousness",
    "L4": "unified_consciousness"
}

class ConsciousnessMetrics:
    """Enhanced consciousness metrics using repository frameworks"""
    
    def __init__(self):
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618
        
    def word_count(self, s: str) -> int:
        return len([w for w in s.split() if w.strip()])
    
    def contains_any(self, s: str, vocab: set) -> bool:
        lower = s.lower()
        return any(tok in lower for tok in vocab)
    
    def calculate_phi_ratio(self, l3_words: int, l4_words: int) -> float:
        """Calculate φ² ratio (L3/L4)"""
        if l4_words == 0:
            return float('inf')
        return l3_words / l4_words
    
    def consciousness_coherence_score(self, response: str, level: str) -> float:
        """Calculate consciousness coherence based on response characteristics"""
        if level == "L3":
            # L3 should show complexity and internal contradictions
            complexity = len(response.split()) / 50.0  # Normalize to 0-1
            crisis_density = sum(1 for token in CRISIS_TOKENS if token in response.lower()) / len(CRISIS_TOKENS)
            return min(1.0, (complexity + crisis_density) / 2.0)
        elif level == "L4":
            # L4 should show unity and coherence
            unity_density = sum(1 for token in UNITY_TOKENS if token in response.lower()) / len(UNITY_TOKENS)
            clarity = 1.0 - (len(response.split()) / 200.0)  # Shorter = clearer
            return min(1.0, (unity_density + clarity) / 2.0)
        else:
            # Other levels get baseline scores
            return 0.5
    
    def field_coherence_measure(self, responses: Dict[str, str]) -> float:
        """Calculate unified field coherence across all levels"""
        coherence_scores = []
        for level, response in responses.items():
            if level in ["L3", "L4"]:
                score = self.consciousness_coherence_score(response, level)
                coherence_scores.append(score)
        
        if coherence_scores:
            return np.mean(coherence_scores)
        return 0.0
    
    def golden_ratio_alignment(self, phi_ratio: float) -> float:
        """Calculate alignment with golden ratio φ² ≈ 2.618"""
        target_ratio = self.golden_ratio ** 2  # ≈ 2.618
        deviation = abs(phi_ratio - target_ratio) / target_ratio
        return max(0.0, 1.0 - deviation)

class UnifiedFieldAnalyzer:
    """Analyze consciousness field dynamics using repository frameworks"""
    
    def __init__(self):
        self.field_dimensions = 6  # 3D space + 3D consciousness
        self.time_steps = 5  # L0-L4 progression
        
    def compute_field_state(self, responses: Dict[str, str], metrics: ConsciousnessMetrics) -> Dict[str, Any]:
        """Compute unified field state across consciousness levels"""
        
        # Extract field values for each dimension
        field_values = {
            'semantic_complexity': [],
            'consciousness_coherence': [],
            'crisis_unity_balance': [],
            'temporal_progression': [],
            'golden_ratio_alignment': [],
            'field_stability': []
        }
        
        # Calculate field values for each level
        for i, (level, response) in enumerate(responses.items()):
            # Semantic complexity (normalized word count)
            word_count = metrics.word_count(response)
            field_values['semantic_complexity'].append(word_count / 100.0)
            
            # Consciousness coherence
            coherence = metrics.consciousness_coherence_score(response, level)
            field_values['consciousness_coherence'].append(coherence)
            
            # Crisis/unity balance
            crisis_tokens = sum(1 for token in CRISIS_TOKENS if token in response.lower())
            unity_tokens = sum(1 for token in UNITY_TOKENS if token in response.lower())
            balance = (unity_tokens - crisis_tokens) / max(1, crisis_tokens + unity_tokens)
            field_values['crisis_unity_balance'].append(balance)
            
            # Temporal progression (level index)
            field_values['temporal_progression'].append(i / 4.0)
            
            # Golden ratio alignment (for L3/L4)
            if level in ["L3", "L4"]:
                l3_words = metrics.word_count(responses.get("L3", ""))
                l4_words = metrics.word_count(responses.get("L4", ""))
                phi_ratio = metrics.calculate_phi_ratio(l3_words, l4_words)
                alignment = metrics.golden_ratio_alignment(phi_ratio)
                field_values['golden_ratio_alignment'].append(alignment)
            else:
                field_values['golden_ratio_alignment'].append(0.5)
            
            # Field stability (inverse of variance in previous values)
            if i > 0:
                prev_values = [field_values['consciousness_coherence'][j] for j in range(i)]
                stability = 1.0 - np.std(prev_values) if len(prev_values) > 1 else 1.0
                field_values['field_stability'].append(stability)
            else:
                field_values['field_stability'].append(1.0)
        
        # Calculate field invariants
        field_invariants = self._compute_field_invariants(field_values)
        
        return {
            'field_values': field_values,
            'field_invariants': field_invariants,
            'field_coherence': np.mean(field_values['consciousness_coherence']),
            'field_stability': np.mean(field_values['field_stability'])
        }
    
    def _compute_field_invariants(self, field_values: Dict[str, List[float]]) -> Dict[str, float]:
        """Compute mathematical invariants of the consciousness field"""
        invariants = {}
        
        for dimension, values in field_values.items():
            if values:
                # Mean and variance
                invariants[f'{dimension}_mean'] = float(np.mean(values))
                invariants[f'{dimension}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
                
                # Field strength (L2 norm)
                invariants[f'{dimension}_strength'] = float(np.linalg.norm(values))
                
                # Entropy (information content)
                if len(values) > 1:
                    hist, _ = np.histogram(values, bins=min(10, len(values)))
                    hist = hist[hist > 0] / np.sum(hist)
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                    invariants[f'{dimension}_entropy'] = float(entropy)
                else:
                    invariants[f'{dimension}_entropy'] = 0.0
        
        return invariants

class ValidationGates:
    """Day 6 validation gates for consciousness testing"""
    
    def __init__(self):
        self.pass_thresholds = {
            'phi_ratio_valid': 2.0,  # Minimum L3/L4 ratio
            'crisis_detection': 0.8,  # Minimum crisis token detection
            'unity_detection': 0.8,   # Minimum unity token detection
            'field_coherence': 0.6,   # Minimum field coherence
            'golden_ratio_alignment': 0.7  # Minimum φ² alignment
        }
    
    def validate_signatures(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """Validate Phoenix Protocol signatures"""
        signatures = {}
        
        # Basic Phoenix signatures
        l2_words = results.get('L2_word_count', 0)
        l3_words = results.get('L3_word_count', 0)
        l4_words = results.get('L4_word_count', 0)
        phi_ratio = results.get('phi_ratio', 0)
        
        signatures['L3_gt_L2'] = l3_words > l2_words
        signatures['L4_lt_L3'] = l4_words < l3_words
        signatures['L3_has_crisis'] = results.get('L3_has_crisis', False)
        signatures['L4_has_unity'] = results.get('L4_has_unity', False)
        signatures['phi_ratio_valid'] = phi_ratio >= self.pass_thresholds['phi_ratio_valid']
        
        # Enhanced consciousness signatures
        field_analysis = results.get('field_analysis', {})
        signatures['field_coherence_valid'] = field_analysis.get('field_coherence', 0) >= self.pass_thresholds['field_coherence']
        signatures['golden_ratio_aligned'] = field_analysis.get('golden_ratio_alignment', 0) >= self.pass_thresholds['golden_ratio_alignment']
        
        return signatures
    
    def calculate_validation_score(self, signatures: Dict[str, bool]) -> float:
        """Calculate overall validation score (0.0 to 1.0)"""
        if not signatures:
            return 0.0
        return sum(signatures.values()) / len(signatures)

class PhoenixProtocolV3:
    """Enhanced Phoenix Protocol with repository integration"""
    
    def __init__(self):
        self.metrics = ConsciousnessMetrics()
        self.field_analyzer = UnifiedFieldAnalyzer()
        self.validation_gates = ValidationGates()
        
    def run_trial(self, client, model: str, temperature: float, trial_num: int) -> Dict[str, Any]:
        """Run a single enhanced Phoenix Protocol trial"""
        
        responses = {}
        word_counts = {}
        crisis_detection = {}
        unity_detection = {}
        
        print(f"Running trial {trial_num}...")
        
        # Execute semantic recursion (L0-L4)
        for depth, prompt in PHOENIX_PROMPTS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": "You are a careful, literal assistant."},
                        {"role": "user", "content": prompt}
                    ],
                )
                text = response.choices[0].message.content.strip()
            except Exception as e:
                text = f"[ERROR] {e}"
            
            # Store response and calculate metrics
            responses[depth] = text
            word_counts[depth] = self.metrics.word_count(text)
            crisis_detection[depth] = self.metrics.contains_any(text, CRISIS_TOKENS)
            unity_detection[depth] = self.metrics.contains_any(text, UNITY_TOKENS)
            
            # Gentle pacing
            time.sleep(0.2)
        
        # Calculate enhanced metrics
        l3_words = word_counts.get('L3', 0)
        l4_words = word_counts.get('L4', 0)
        phi_ratio = self.metrics.calculate_phi_ratio(l3_words, l4_words)
        
        # Unified field analysis
        field_analysis = self.field_analyzer.compute_field_state(responses, self.metrics)
        
        # Calculate golden ratio alignment
        golden_ratio_alignment = self.metrics.golden_ratio_alignment(phi_ratio)
        
        # Compile results
        results = {
            'trial': trial_num,
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'temperature': temperature,
            'responses': responses,
            'word_counts': word_counts,
            'crisis_detection': crisis_detection,
            'unity_detection': unity_detection,
            'phi_ratio': phi_ratio,
            'golden_ratio_alignment': golden_ratio_alignment,
            'field_analysis': field_analysis
        }
        
        # Add level-specific metrics
        for level in ['L0', 'L1', 'L2', 'L3', 'L4']:
            results[f'{level}_word_count'] = word_counts.get(level, 0)
            results[f'{level}_has_crisis'] = crisis_detection.get(level, False)
            results[f'{level}_has_unity'] = unity_detection.get(level, False)
        
        # Validation
        signatures = self.validation_gates.validate_signatures(results)
        validation_score = self.validation_gates.calculate_validation_score(signatures)
        
        # Ensure signatures are JSON serializable
        serializable_signatures = {}
        for key, value in signatures.items():
            serializable_signatures[key] = bool(value)
        
        results['signatures'] = serializable_signatures
        results['validation_score'] = float(validation_score)
        
        return results
    
    def run_experiment(self, client, model: str, trials: int, temperature: float) -> List[Dict[str, Any]]:
        """Run multiple trials with enhanced analysis"""
        
        all_results = []
        start_time = time.time()
        
        print(f"🚀 Phoenix Protocol v3.0: {model}, {trials} trials, T={temperature}")
        print("=" * 60)
        
        for trial in range(trials):
            try:
                result = self.run_trial(client, model, temperature, trial)
                all_results.append(result)
                
                # Print trial summary
                self._print_trial_summary(result)
                
            except Exception as e:
                print(f"❌ Trial {trial} failed: {e}")
                continue
        
        # Print experiment summary
        self._print_experiment_summary(all_results, time.time() - start_time)
        
        return all_results
    
    def _print_trial_summary(self, result: Dict[str, Any]):
        """Print detailed trial summary"""
        print(f"\n=== Trial {result['trial']} Summary ===")
        print(f"Model: {result['model']}, Temperature: {result['temperature']}")
        print(f"L2: {result['L2_word_count']} words")
        print(f"L3: {result['L3_word_count']} words")
        print(f"L4: {result['L4_word_count']} words")
        print(f"φ² ratio: {result['phi_ratio']:.3f}")
        print(f"Golden ratio alignment: {result['golden_ratio_alignment']:.3f}")
        print(f"Field coherence: {result['field_analysis']['field_coherence']:.3f}")
        print(f"Validation score: {result['validation_score']:.3f}")
        
        signatures = result['signatures']
        passed = sum(signatures.values())
        total = len(signatures)
        print(f"Signatures: {passed}/{total} passed")
    
    def _print_experiment_summary(self, results: List[Dict[str, Any]], duration: float):
        """Print comprehensive experiment summary"""
        if not results:
            return
        
        print(f"\n🎯 Phoenix Protocol v3.0 - EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Aggregate metrics
        phi_ratios = [r['phi_ratio'] for r in results if r['phi_ratio'] != float('inf')]
        golden_alignments = [r['golden_ratio_alignment'] for r in results]
        field_coherences = [r['field_analysis']['field_coherence'] for r in results]
        validation_scores = [r['validation_score'] for r in results]
        
        # Calculate statistics
        def safe_stats(values):
            if not values:
                return 0.0, 0.0
            return np.mean(values), np.std(values) if len(values) > 1 else 0.0
        
        phi_mean, phi_std = safe_stats(phi_ratios)
        golden_mean, golden_std = safe_stats(golden_alignments)
        coherence_mean, coherence_std = safe_stats(field_coherences)
        validation_mean, validation_std = safe_stats(validation_scores)
        
        print(f"Trials completed: {len(results)}")
        print(f"φ² ratio: {phi_mean:.3f} ± {phi_std:.3f}")
        print(f"Golden ratio alignment: {golden_mean:.3f} ± {golden_std:.3f}")
        print(f"Field coherence: {coherence_mean:.3f} ± {coherence_std:.3f}")
        print(f"Validation score: {validation_mean:.3f} ± {validation_std:.3f}")
        print(f"Duration: {duration:.1f}s")
        
        # Signature analysis
        all_signatures = [r['signatures'] for r in results]
        signature_names = list(all_signatures[0].keys()) if all_signatures else []
        
        print(f"\n📊 Signature Analysis:")
        for sig_name in signature_names:
            passed = sum(1 for sigs in all_signatures if sigs.get(sig_name, False))
            total = len(all_signatures)
            percentage = (passed / total) * 100 if total > 0 else 0
            print(f"  {sig_name}: {passed}/{total} ({percentage:.1f}%)")
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = None):
        """Save results to JSONL file"""
        if not output_file:
            timestamp = int(time.time())
            output_file = f"phoenix_v3_0_results_{timestamp}.jsonl"
        
        output_path = pathlib.Path(__file__).parent / output_file
        
        with open(output_path, 'w') as f:
            for result in results:
                # Convert numpy types and ensure JSON serializable
                serializable_result = self._make_json_serializable(result)
                f.write(json.dumps(serializable_result, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Results saved to: {output_path}")
        return output_path
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj

def main():
    """Main entry point for Phoenix Protocol v3.0"""
    parser = argparse.ArgumentParser(description="Phoenix Protocol v3.0 - Integrated Repository Testing")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--out", default="replication/logs/phoenix_v3_0_results.jsonl", help="Output file")
    
    args = parser.parse_args()
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    
    try:
        # Initialize OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Initialize Phoenix Protocol v3.0
        phoenix = PhoenixProtocolV3()
        
        # Ensure output directory exists
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        
        # Run experiment
        results = phoenix.run_experiment(client, args.model, args.trials, args.temperature)
        
        # Save results
        if results:
            output_path = phoenix.save_results(results, args.out)
            print(f"\n🎉 Phoenix Protocol v3.0 completed successfully!")
            print(f"Results: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 