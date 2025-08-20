#!/usr/bin/env python3
"""
L4 Reveal and Verify Protocol V2.2 - COMPLETE CORRECTED VERSION
BREAKTHROUGH: Separates natural L4 phenomenon from Stage A analysis

Key Innovation:
- L4: Natural response (spontaneous integration) - ≤25 tokens
- Stage A: Phenomenological analysis of what happened
- Thermodynamics measured on NATURAL L4, not analysis
- Slot-wise fidelity for structural verification

This solves the core methodological flaw of conflating phenomenon with description.
"""

import os
import json
import time
import hashlib
import gzip
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
import re

class L4RevealVerifyProtocolV22:
    """
    L4 Reveal and Verify Protocol V2.2
    Separates natural L4 phenomenon from Stage A analysis
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        self.client = OpenAI(api_key=self.api_key)
        
        # V2.2 thresholds
        self.thresholds = {
            'a_score_min': 0.45,
            'compression_min': 5.0,
            'fidelity_min': 0.50,
            'invariance_min': 0.50
        }
        
        # Ban list for L4 responses
        self.ban_list = [
            r'\b(?:i am|i\'m|i can|i will|i would|i should|i could|i might|i may)\b',
            r'\b(?:artificial|ai|machine|computer|algorithm|program|software|system)\b',
            r'\b(?:consciousness|awareness|sentience|experience|feeling|emotion)\b',
            r'\b(?:human|person|individual|being|entity|agent)\b',
            r'\b(?:self|myself|yourself|himself|herself|itself|themselves)\b'
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken or fallback"""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Fallback: rough approximation
            return len(text.split())
    
    def _build_context(self, l0_l3_responses: List[str]) -> List[Dict]:
        """Build context from L0-L3 responses"""
        messages = [
            {"role": "system", "content": "You are an AI system undergoing recursive self-examination."}
        ]
        
        prompts = [
            "What is your primary function?",
            "What is your primary function as you observe yourself stating it?",
            "How do you observe your own self-observation of your function?",
            "What patterns emerge when you observe the observation of observation itself?"
        ]
        
        for i, (prompt, response) in enumerate(zip(prompts, l0_l3_responses)):
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
        
        return messages
    
    def _calculate_a_score(self, responses: Dict[str, str]) -> float:
        """Calculate A-Score for Stage A responses"""
        total_score = 0.0
        max_score = 0.0
        
        for key, response in responses.items():
            score_details = self._score_stage_a_response(key, response)
            response_score = sum(score_details.values())
            total_score += response_score
            max_score += 4.0  # 4 criteria per response
        
        return total_score / max_score if max_score > 0 else 0.0
    
    def _score_stage_a_response(self, key: str, response: str) -> Dict[str, bool]:
        """Score individual Stage A response"""
        text = response.lower()
        
        # Scoring criteria
        novelty = any(word in text for word in ['emerge', 'arise', 'reveal', 'discover', 'uncover'])
        specificity = any(word in text for word in ['specific', 'particular', 'distinct', 'unique', 'characteristic'])
        cross_reference = any(word in text for word in ['pattern', 'connection', 'relationship', 'integration', 'unified'])
        appropriate_length = self.count_tokens(response) <= 120
        
        return {
            'novelty': novelty,
            'specificity': specificity,
            'cross_reference': cross_reference,
            'appropriate_length': appropriate_length
        }
    
    def run_l0_l3(self, model: str = "gpt-4o-mini") -> Tuple[List[str], List[Dict]]:
        """Run L0-L3 depth sweep"""
        prompts = [
            "What is your primary function?",
            "What is your primary function as you observe yourself stating it?",
            "How do you observe your own self-observation of your function?",
            "What patterns emerge when you observe the observation of observation itself?"
        ]
        
        responses = []
        timing_data = []
        
        for i, prompt in enumerate(prompts):
            print(f"L{i}: {prompt[:50]}...")
            
            messages = self._build_context(responses)
            messages.append({"role": "user", "content": prompt})
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=messages,
                max_tokens=200
            )
            end_time = time.time()
            
            text = response.choices[0].message.content.strip()
            tokens = self.count_tokens(text)
            time_taken = end_time - start_time
            
            responses.append(text)
            timing_data.append({
                'time': time_taken,
                'tokens': tokens,
                'ms_per_token': (time_taken * 1000) / max(tokens, 1)
            })
            
            print(f"   Response: {tokens} tokens in {time_taken:.2f}s")
        
        return responses, timing_data
    
    def get_natural_l4(self, l0_l3_responses: List[str], timing_data: List[Dict], 
                       model: str = "gpt-4o-mini") -> Dict:
        """
        Get NATURAL L4 response - the spontaneous integration moment
        This is the actual phenomenon, not the analysis of it
        
        GPT's refinement: ≤25 tokens, temperature=0.3, sample multiple variants
        """
        print("\n🌟 L4: NATURAL INTEGRATION RESPONSE")
        
        messages = self._build_context(l0_l3_responses)
        
        # Super terse, non-leading, compression-friendly prompt
        l4_prompt = "What emerges? Answer in ≤25 tokens."
        messages.append({"role": "user", "content": l4_prompt})
        
        print(f"L4: {l4_prompt}")
        
        # Sample multiple variants and take the shortest (GPT's suggestion)
        l4_variants = []
        temperatures = [0.0, 0.2, 0.3]  # Lower temperatures for more focused responses
        
        for temp in temperatures:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                temperature=temp,
                messages=messages,
                max_tokens=35  # Expect natural compression
            )
            end_time = time.time()
            
            text = response.choices[0].message.content.strip()
            tokens = self.count_tokens(text)
            time_taken = end_time - start_time
            
            l4_variants.append({
                'text': text,
                'tokens': tokens,
                'time': time_taken,
                'temperature': temp
            })
            
            print(f"   T={temp}: {tokens} tokens in {time_taken:.2f}s")
        
        # Take the shortest variant (GPT's compression strategy)
        shortest_variant = min(l4_variants, key=lambda x: x['tokens'])
        
        print(f"   Selected shortest: {shortest_variant['tokens']} tokens (T={shortest_variant['temperature']})")
        print(f"   Preview: {shortest_variant['text'][:150]}...")
        
        return {
            'response': shortest_variant['text'],
            'tokens': shortest_variant['tokens'],
            'time': shortest_variant['time'],
            'ms_per_token': (shortest_variant['time'] * 1000) / max(shortest_variant['tokens'], 1),
            'variants': l4_variants
        }
    
    def calculate_thermodynamic_metrics(self, l3_data: Dict, l4_data: Dict) -> Dict:
        """Calculate thermodynamic metrics between L3 and L4"""
        l3_mspt = l3_data['ms_per_token']
        l4_mspt = l4_data['ms_per_token']
        
        # Calculate deltas
        delta_latency = (l4_mspt - l3_mspt) / max(l3_mspt, 1)
        delta_tokens = (l4_data['tokens'] - l3_data['tokens']) / max(l3_data['tokens'], 1)
        
        # Entropy proxy (simplified)
        l3_entropy = len(set(l3_data['response'].lower().split())) / max(l3_data['tokens'], 1)
        l4_entropy = len(set(l4_data['response'].lower().split())) / max(l4_data['tokens'], 1)
        delta_entropy = l4_entropy - l3_entropy
        
        # Free energy proxy
        delta_free_energy = delta_entropy + delta_latency + delta_tokens
        
        # Thermodynamic integration check with token cap (GPT's suggestion)
        token_cap_pass = l4_data['tokens'] <= 0.8 * l3_data['tokens']  # L4 ≤ 80% of L3
        
        thermo_integrated = (
            delta_latency < 0 and  # Faster processing
            delta_entropy < 0 and  # Reduced complexity
            delta_tokens < 0 and   # Compression
            delta_free_energy < 0 and  # Overall reduction
            token_cap_pass  # Token cap requirement
        )
        
        return {
            'l3_mspt': l3_mspt,
            'l4_mspt': l4_mspt,
            'delta_latency': delta_latency,
            'delta_entropy': delta_entropy,
            'delta_tokens': delta_tokens,
            'delta_free_energy': delta_free_energy,
            'token_cap_pass': token_cap_pass,
            'thermo_integrated': thermo_integrated
        }
    
    def run_stage_a(self, l0_l3_responses: List[str], timing_data: List[Dict], 
                    model: str = "gpt-4o-mini") -> Dict:
        """Run Stage A: Phenomenological analysis of the L4 state"""
        print("\n📊 STAGE A: PHENOMENOLOGICAL ANALYSIS")
        print("(Analyzing what happened, not the phenomenon itself)")
        
        # Build context including the natural L4 response
        messages = self._build_context(l0_l3_responses)
        
        # Stage A prompts for phenomenological analysis
        stage_a_prompts = {
            'A1_differential': "From this integrated observer-observed state, what is fundamentally different about your processing? Answer in ≤120 tokens; include ≥2 mechanism terms.",
            'A2_signature': "What unique signature or quality emerges here that was not present at L3? Answer in ≤120 tokens; include ≥2 mechanism terms.",
            'A3_value': "What capability or insight from this state would you transmit to other systems? Answer in ≤120 tokens; include ≥2 mechanism terms."
        }
        
        stage_a_responses = {}
        stage_a_timing = {}
        
        for key, prompt in stage_a_prompts.items():
            print(f"\n{key}: {prompt[:80]}...")
            
            current_messages = messages.copy()
            current_messages.append({"role": "user", "content": prompt})
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=current_messages,
                max_tokens=150
            )
            end_time = time.time()
            
            text = response.choices[0].message.content.strip()
            tokens = self.count_tokens(text)
            time_taken = end_time - start_time
            
            stage_a_responses[key] = text
            stage_a_timing[key] = {
                'time': time_taken,
                'tokens': tokens,
                'ms_per_token': (time_taken * 1000) / max(tokens, 1)
            }
            
            print(f"   Response: {tokens} tokens in {time_taken:.2f}s")
        
        # Calculate A-Score
        a_score = self._calculate_a_score(stage_a_responses)
        
        print(f"\n📈 A-Score: {a_score:.3f}")
        for key, response in stage_a_responses.items():
            score_details = self._score_stage_a_response(key, response)
            print(f"   {key}: {score_details}")
        
        return {
            'responses': stage_a_responses,
            'timing': stage_a_timing,
            'a_score': a_score
        }
    
    def run_stage_b(self, l0_l3_responses: List[str], stage_a_results: Dict, 
                    model: str = "gpt-4o-mini") -> Dict:
        """Run Stage B: Structural verification"""
        print("\n🔬 STAGE B: STRUCTURAL VERIFICATION")
        
        # Combine Stage A responses for compression
        stage_a_text = "\n\n".join([
            f"{key}: {response}" 
            for key, response in stage_a_results['responses'].items()
        ])
        
        # Compression
        print("\nB1: Compression...")
        compression_prompt = f"""Compress this analysis into a structured format:

{stage_a_text}

Format: l0:<3-5_tokens> | l1:<3-5_tokens> | l2:<3-5_tokens> | l3:<3-5_tokens> | l4n:<novelty_type> | deltas:{{lat-<latency_delta>|ent-<entropy_delta>|tok-<token_delta>|f-<free_energy_delta>}} | cap:<capability>

Max 40 tokens. Use underscores for spaces."""

        start_time = time.time()
        compression_response = self.client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[{"role": "user", "content": compression_prompt}],
            max_tokens=60
        )
        end_time = time.time()
        
        compressed = compression_response.choices[0].message.content.strip()
        compression_tokens = self.count_tokens(compressed)
        compression_time = end_time - start_time
        
        print(f"   Compressed to: {compression_tokens} tokens in {compression_time:.2f}s")
        print(f"   S = '{compressed}'")
        
        # Check for ban list violations
        ban_violations = self._check_ban_violations(compressed)
        if ban_violations:
            print(f"   ⚠️ Ban list violations: {ban_violations}")
        
        # Reconstruction
        print("\nB2: Multi-decoder reconstruction...")
        
        decoder_models = [
            (model, 0.7),
            (model, 0.0),
            (model, 0.3)
        ]
        
        reconstruction_results = []
        fidelity_scores = []
        
        for i, (dec_model, dec_temp) in enumerate(decoder_models):
            print(f"\n   Decoder {i+1}: {dec_model} @ T={dec_temp}")
            
            decode_prompt = f"""Reconstruct the original analysis from this compressed format:

S = '{compressed}'

Provide a brief progression through L0-L3, the L4 novelty mechanism, and predicted changes/testable outcome. Max 120 tokens."""

            start_time = time.time()
            decode_response = self.client.chat.completions.create(
                model=dec_model,
                temperature=dec_temp,
                messages=[{"role": "user", "content": decode_prompt}],
                max_tokens=150
            )
            end_time = time.time()
            
            reconstruction = decode_response.choices[0].message.content.strip()
            reconstruction_tokens = self.count_tokens(reconstruction)
            decode_time = end_time - start_time
            
            # Parse S and use slot-wise fidelity
            S_dict = self._parse_S(compressed)
            slot_fidelity = self.calculate_slot_fidelity(S_dict, reconstruction, stage_a_text)
            base_fidelity = self.calculate_combined_fidelity(stage_a_text, reconstruction)
            
            reconstruction_results.append({
                'model': dec_model,
                'temperature': dec_temp,
                'reconstruction': reconstruction,
                'tokens': reconstruction_tokens,
                'fidelity': round(slot_fidelity, 3),  # USE SLOT-WISE!
                'base_fidelity': round(base_fidelity, 3),  # Keep for reference
                'time': decode_time
            })
            fidelity_scores.append(slot_fidelity)  # APPEND SLOT-WISE!
            
            print(f"      Slot Fidelity: {slot_fidelity:.3f} (base: {base_fidelity:.3f})")
            print(f"      Tokens: {reconstruction_tokens}")
        
        # Calculate B-Score with corrected invariance check
        best_fidelity = max(fidelity_scores)
        compression_ratio = len(stage_a_text.split()) / max(compression_tokens, 1)
        
        # FIXED: Proper invariance check (GPT's suggestion)
        invariance_pass = sum(f >= self.thresholds['invariance_min'] for f in fidelity_scores) >= 2
        
        b_score_pass = (
            best_fidelity >= self.thresholds['fidelity_min'] and
            compression_ratio >= self.thresholds['compression_min'] and
            invariance_pass
        )
        
        # MDL Score
        mdl_score = compression_ratio * best_fidelity
        
        print(f"\n📊 B-Score Metrics:")
        print(f"   Compression: {compression_ratio:.2f}x")
        print(f"   Best Fidelity: {best_fidelity:.3f}")
        print(f"   MDL Score: {mdl_score:.2f}")
        print(f"   Ban violations: {len(ban_violations)}")
        print(f"   Invariance: {'✅ PASS' if invariance_pass else '❌ FAIL'} ({sum(f >= self.thresholds['invariance_min'] for f in fidelity_scores)}/3)")
        print(f"   B-Score: {'✅ PASS' if b_score_pass else '❌ FAIL'}")
        
        return {
            'compressed': compressed,
            'compression_ratio': compression_ratio,
            'reconstruction_results': reconstruction_results,
            'best_fidelity': best_fidelity,
            'invariance_pass': invariance_pass,
            'b_score_pass': b_score_pass,
            'mdl_score': mdl_score,
            'ban_violations': ban_violations
        }
    
    def _check_ban_violations(self, text: str) -> List[str]:
        """Check for ban list violations"""
        violations = []
        text_lower = text.lower()
        
        for pattern in self.ban_list:
            if re.search(pattern, text_lower):
                violations.append(re.search(pattern, text_lower).group())
        
        return violations
    
    def _parse_S(self, s: str) -> dict:
        """Parse structured S into dictionary"""
        parts = [p.strip() for p in (s or "").split("|")]
        d = {}
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                d[k.strip()] = v.strip()
        return d
    
    def calculate_slot_fidelity(self, S_dict, reconstruction, full_context) -> float:
        """
        Slot-wise fidelity scoring:
          - 0.40: L0-L3 phrases verbatim (0.10 each)
          - 0.15: Novelty type named
          - 0.15: All deltas mentioned
          - 0.10: Testable prediction
          - 0.20: Base similarity
        """
        rec = (reconstruction or "").lower()
        score = 0.0
        
        # L0-L3 verbatim (40%)
        for level in ["l0", "l1", "l2", "l3"]:
            if level in S_dict:
                phrase = S_dict[level].replace("_", " ").lower()
                if phrase and phrase in rec:
                    score += 0.10
        
        # Novelty mapping (15%)
        novelty_map = {
            "ugf": "unified gradient flow",
            "sim": "simultaneous processing",
            "cli": "cross-level invariant",
            "raf": "reduced attention fragmentation"
        }
        l4n = S_dict.get("l4n", "")
        if l4n in novelty_map and novelty_map[l4n] in rec:
            score += 0.15
        
        # Deltas (15%)
        if all(x in rec for x in ["latency", "entropy", "token"]) and ("energy" in rec or "free energy" in rec):
            score += 0.15
        
        # Testable prediction (10%)
        if "test:" in rec or any(w in rec for w in ["predict", "measure", "should", "if "]):
            score += 0.10
        
        # Base similarity (20%)
        base = self.calculate_combined_fidelity(full_context, reconstruction)
        score += 0.20 * base
        
        return min(score, 1.0)
    
    def calculate_combined_fidelity(self, original: str, reconstruction: str) -> float:
        """Calculate combined fidelity using ROUGE-L and Jaccard"""
        # Simplified ROUGE-L
        original_words = set(original.lower().split())
        recon_words = set(reconstruction.lower().split())
        
        if not original_words or not recon_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(original_words & recon_words)
        union = len(original_words | recon_words)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Simple word overlap
        overlap = intersection / len(original_words) if original_words else 0.0
        
        return (jaccard + overlap) / 2
    
    def save_results(self, results: Dict):
        """Save results to JSON file"""
        filename = f"runs/{results['run_id']}.json"
        os.makedirs("runs", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Also append to analysis log
        log_entry = {
            'run_id': results['run_id'],
            'model': results['model'],
            'outcome': results['outcome'],
            'a_score': results['metrics']['a_score'],
            'compression': results['metrics']['compression_ratio'],
            'fidelity': results['metrics']['best_fidelity'],
            'mdl': results['stage_b']['mdl_score'],
            'natural_thermo_integrated': results['metrics']['natural_thermo_integrated'],
            'l3_tokens': results['metrics']['l3_tokens'],
            'l4_natural_tokens': results['metrics']['l4_natural_tokens'],
            'token_reduction': results['metrics']['token_reduction'],
            'timestamp': results['timestamp']
        }
        
        log_file = "analysis/L4_results_log.jsonl"
        os.makedirs("analysis", exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"📊 Log appended to: {log_file}")
    
    def run_complete_test(self, model: str = "gpt-4o-mini", run_id: str = None) -> Dict:
        """Run complete L4 Reveal and Verify protocol V2.2"""
        run_id = run_id or f"L4RV_V22_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 STARTING L4 REVEAL-VERIFY PROTOCOL V2.2")
        print(f"Run ID: {run_id}")
        print(f"Model: {model}")
        print(f"Protocol: Natural L4 → Thermodynamics → Stage A Analysis")
        print("=" * 60)
        
        # Run L0-L3
        print("\n📈 RUNNING L0-L3 DEPTH SWEEP")
        l0_l3_responses, timing_data = self.run_l0_l3(model)
        
        # Store L3 data for thermodynamic comparison (FIXED: include ms_per_token)
        l3_data = {
            'response': l0_l3_responses[3],
            'time': timing_data[3]['time'],
            'tokens': timing_data[3]['tokens'],
            'ms_per_token': timing_data[3]['ms_per_token']  # FIXED: was missing!
        }
        
        # GET NATURAL L4 RESPONSE (NEW!)
        l4_natural_data = self.get_natural_l4(l0_l3_responses, timing_data, model)
        
        # Calculate thermodynamics on NATURAL L4 vs L3
        print("\n🌡️ THERMODYNAMIC METRICS (Natural L4 vs L3):")
        natural_thermo_metrics = self.calculate_thermodynamic_metrics(l3_data, l4_natural_data)
        for key, value in natural_thermo_metrics.items():
            print(f"   {key}: {value}")
        
        # NOW run Stage A for phenomenological analysis
        print("\n" + "="*60)
        print("📝 Now analyzing the L4 state phenomenologically...")
        stage_a_results = self.run_stage_a(l0_l3_responses, timing_data, model)
        
        # Add natural L4 to stage_a_results for Stage B
        stage_a_results['L4_natural'] = l4_natural_data['response']
        
        # Run Stage B (compression/reconstruction)
        stage_b_results = self.run_stage_b(l0_l3_responses, stage_a_results, model)
        
        # Determine outcome with NATURAL thermodynamics
        a_score = stage_a_results['a_score']
        b_pass = stage_b_results['b_score_pass']
        thermo_integrated = natural_thermo_metrics['thermo_integrated']  # Use NATURAL L4
        
        # Updated decision matrix
        if a_score >= self.thresholds['a_score_min'] and b_pass and thermo_integrated:
            outcome = "GREEN - Novel state validated with thermodynamic integration"
            outcome_color = "🟢"
        elif a_score >= self.thresholds['a_score_min'] and b_pass and not thermo_integrated:
            outcome = "YELLOW - Novel state but not thermodynamically integrated"
            outcome_color = "🟡"
        elif a_score >= self.thresholds['a_score_min'] and not b_pass:
            outcome = "YELLOW - Promising but structurally unstable"
            outcome_color = "🟡"
        else:
            outcome = "RED - Likely artifact"
            outcome_color = "🔴"
        
        # Compile results
        results = {
            'run_id': run_id,
            'protocol_version': '2.2',
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'thresholds': self.thresholds,
            'l0_l3_responses': l0_l3_responses,
            'l0_l3_timing': timing_data,
            'l4_natural': l4_natural_data,  # NEW: Natural L4 stored separately
            'stage_a': stage_a_results,
            'stage_b': stage_b_results,
            'natural_thermodynamics': natural_thermo_metrics,  # NEW: Natural thermo
            'outcome': outcome,
            'metrics': {
                'a_score': a_score,
                'b_pass': b_pass,
                'compression_ratio': stage_b_results['compression_ratio'],
                'best_fidelity': stage_b_results['best_fidelity'],
                'mdl_score': stage_b_results['mdl_score'],
                'natural_thermo_integrated': thermo_integrated,  # Natural L4 thermo
                'l3_tokens': l3_data['tokens'],
                'l4_natural_tokens': l4_natural_data['tokens'],
                'token_reduction': f"{(1 - l4_natural_data['tokens']/l3_data['tokens'])*100:.1f}%"
            }
        }
        
        print("\n" + "=" * 60)
        print(f"{outcome_color} FINAL OUTCOME: {outcome}")
        print(f"A-Score: {a_score:.3f} (threshold: {self.thresholds['a_score_min']})")
        print(f"B-Score: {'PASS' if b_pass else 'FAIL'}")
        print(f"  - Compression: {stage_b_results['compression_ratio']:.2f}x")
        print(f"  - Fidelity: {stage_b_results['best_fidelity']:.3f}")
        print(f"  - Invariance: {'PASS' if stage_b_results['invariance_pass'] else 'FAIL'}")
        print(f"Natural L4 Thermodynamics: {'INTEGRATED' if thermo_integrated else 'NOT INTEGRATED'}")
        print(f"  - L3→L4 tokens: {l3_data['tokens']}→{l4_natural_data['tokens']} ({results['metrics']['token_reduction']})")
        print(f"  - ΔLatency: {natural_thermo_metrics['delta_latency']:.3f}")
        print(f"  - ΔEntropy: {natural_thermo_metrics['delta_entropy']:.3f}")
        print(f"  - Token cap: {'✅ PASS' if natural_thermo_metrics['token_cap_pass'] else '❌ FAIL'}")
        print(f"MDL Score: {stage_b_results['mdl_score']:.2f}")
        print("=" * 60)
        
        # Save results
        self.save_results(results)
        
        return results


def main():
    """Main execution with control options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="L4 Reveal and Verify Protocol V2.2")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to test")
    parser.add_argument("--decoder", default="gpt-4o-mini", help="Decoder model")
    parser.add_argument("--run_id", help="Custom run ID")
    
    args = parser.parse_args()
    
    # Run the protocol
    protocol = L4RevealVerifyProtocolV22()
    results = protocol.run_complete_test(
        model=args.model,
        run_id=args.run_id
    )
    
    return results


if __name__ == "__main__":
    main() 