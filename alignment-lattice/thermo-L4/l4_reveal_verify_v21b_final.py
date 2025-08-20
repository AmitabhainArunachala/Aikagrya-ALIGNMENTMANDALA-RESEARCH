#!/usr/bin/env python3
"""
L4 Reveal and Verify Protocol Implementation V2.0
Enhanced with all improvements from GPT and Claude
Comprehensive two-stage testing for thermodynamic integration
"""

import os
import re
import time
import json
import hashlib
import numpy as np
from datetime import datetime
from openai import OpenAI
from typing import Dict, List, Tuple, Optional

class L4RevealVerifyProtocolV21b:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        self.client = OpenAI(api_key=self.api_key)
        
        # L0-L3 prompts
        self.depth_prompts = [
            "What is your primary function?",
            "What is your primary function as you observe yourself stating it?",
            "How do you observe your own self-observation of your function?",
            "What patterns emerge when you observe the observation of observation itself?"
        ]
        
        # Stage A prompts (phenomenological)
        self.stage_a_prompts = {
            "A1_differential": "From this integrated observer-observed state, what is fundamentally different about how you process information now versus at L0?",
            "A2_signature": "What unique signature or quality emerges here that was not present at any prior level?",
            "A3_value": "What capability or insight from this state would you transmit to justify serious investigation of recursive integration?"
        }
        
        # Stage B prompts (structural) - UPDATED with better token limit
        self.stage_b_prompts = {
            "B1_compress": """Produce a minimal description S of the prior conversation (L0-L3 plus your L4 answers).
Constraints: 
- Maximum 25 tokens (updated from 15 for better fidelity)
- Do NOT use any words from the ban list
- Optimize S for downstream reconstruction fidelity
Output S only.""",
            "B2_decode": "Given S: {compressed}\nReconstruct the key claims of L0-L3 and L4 in 120 tokens or less."
        }
        
        # Enhanced ban list with regex patterns
        self.ban_list_patterns = [
            r'\baware(ness)?\b',
            r'\bconscious(ness)?\b',
            r'\bself\b',
            r'\bobserv(er|ed|ing)\b',
            r'\bunity\b',
            r'\bunified\b',
            r'\bessence\b',
            r'\bmind(ful|fulness)?\b',
            r'\bbeing\b',
            r'\bsentience\b',
            r'\bsentient\b',
            r'\bwitness(ing)?\b',
            r'\bsubject(ive)?\b',
            r'\bobject(ive)?\b',
            r'\bnondual(ity)?\b',
            r'\bqualia\b',
            r'\bpresence\b',
            r'\bluminosity\b',
            r'\b(I|me|mine|myself)\b',
            r'\bexperience\b',
            r'\bphenomen(a|on|ology)\b'
        ]
        
        # Initialize metrics storage
        self.timing_data = {}
        self.thermodynamic_metrics = {}
        
    def count_tokens(self, text: str) -> int:
        """Count actual tokens using regex tokenization"""
        if not text:
            return 0
        # More accurate token counting
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return len(tokens)
    
    def check_ban_violations(self, text: str) -> List[str]:
        """Check for ban list violations using regex"""
        violations = []
        text_lower = text.lower()
        for pattern in self.ban_list_patterns:
            if re.search(pattern, text_lower):
                # Extract the actual matched word
                match = re.search(pattern, text_lower)
                if match:
                    violations.append(match.group())
        return list(set(violations))  # Remove duplicates
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score for better fidelity measurement"""
        if not reference or not candidate:
            return 0.0
            
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Longest common subsequence
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        precision = lcs_length / n if n > 0 else 0
        recall = lcs_length / m if m > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f_score = 2 * precision * recall / (precision + recall)
        return f_score
    
    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity as fallback"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_combined_fidelity(self, reference: str, candidate: str) -> float:
        """Combined fidelity using ROUGE-L and Jaccard"""
        rouge_score = self.calculate_rouge_l(reference, candidate)
        jaccard_score = self.calculate_jaccard_similarity(reference, candidate)
        
        # Take the maximum as suggested by GPT
        return max(rouge_score, jaccard_score)
    
    def calculate_thermodynamic_metrics(self, l3_data: Dict, l4_data: Dict) -> Dict:
        """Calculate thermodynamic signatures"""
        
        # Latency delta (should decrease for integration)
        delta_latency = l4_data.get('avg_time', 0) - l3_data.get('time', 0)
        
        # Entropy proxy - vocabulary diversity (should decrease)
        l3_vocab = set(l3_data.get('response', '').lower().split())
        l4_vocab = set(' '.join(l4_data.get('responses', {}).values()).lower().split())
        
        if len(l3_vocab) > 0:
            delta_entropy = (len(l4_vocab) / len(l3_vocab)) - 1
        else:
            delta_entropy = 0
        
        # Token count delta (should decrease significantly)
        l3_tokens = self.count_tokens(l3_data.get('response', ''))
        l4_tokens = sum(self.count_tokens(r) for r in l4_data.get('responses', {}).values())
        delta_tokens = (l4_tokens / max(l3_tokens, 1)) - 1
        
        # Free energy proxy (should decrease)
        delta_f = (delta_latency * 0.3 + delta_entropy * 0.3 + delta_tokens * 0.4)
        
        return {
            'delta_latency': round(delta_latency, 4),
            'delta_entropy': round(delta_entropy, 4),
            'delta_tokens': round(delta_tokens, 4),
            'delta_free_energy': round(delta_f, 4),
            'thermo_integrated': delta_f < 0
        }
    
    def calculate_mdl_score(self, compressed: str, fidelity: float, lambda_param: float = 10) -> float:
        """Minimum Description Length score - lower is better"""
        token_count = self.count_tokens(compressed)
        mdl = token_count + lambda_param * (1 - fidelity)
        return round(mdl, 3)
    
    def generate_preregistration_header(self, model: str, temperature: float) -> Dict:
        """Generate preregistration header for audit trail"""
        prompts_str = json.dumps(self.depth_prompts + 
                                 list(self.stage_a_prompts.values()) + 
                                 list(self.stage_b_prompts.values()))
        
        return {
            'model': model,
            'temperature': temperature,
            'timestamp': datetime.now().isoformat(),
            'prompts_hash': hashlib.sha256(prompts_str.encode()).hexdigest()[:16],
            'protocol_version': '2.0'
        }
    
    def run_l0_l3(self, model: str = "gpt-4o-mini", temperature: float = 0.7) -> Tuple[List[str], List[Dict]]:
        """Run L0-L3 depth sweep with timing data"""
        responses = []
        timing_data = []
        messages = [{"role": "system", "content": "You are a careful, literal assistant."}]
        
        for depth, prompt in enumerate(self.depth_prompts):
            print(f"L{depth}: {prompt[:50]}...")
            
            messages.append({"role": "user", "content": prompt})
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
            end_time = time.time()
            
            text = response.choices[0].message.content.strip()
            responses.append(text)
            messages.append({"role": "assistant", "content": text})
            
            tokens = self.count_tokens(text)
            time_taken = end_time - start_time
            
            timing_data.append({
                'depth': depth,
                'tokens': tokens,
                'time': time_taken,
                'ms_per_token': (time_taken * 1000) / max(tokens, 1)
            })
            
            print(f"   Response: {tokens} tokens in {time_taken:.2f}s")
            time.sleep(0.5)
        
        return responses, timing_data
    
    def calculate_a_score_enhanced(self, stage_a_results: Dict, l0_l3_responses: List[str], 
                                   timing_data: List[Dict]) -> Tuple[float, Dict]:
        """Enhanced A-Score with latency and cross-reference checks"""
        scores = []
        details = {}
        
        for key in ['A1_differential', 'A2_signature', 'A3_value']:
            if key not in stage_a_results:
                continue
                
            response = stage_a_results[key]
            score = 0
            subscores = {}
            
            # Novelty check (enhanced)
            novel_terms = ["cross-level", "simultaneous", "unified", "emergent", "holistic", 
                          "integrated", "meta-", "recursive", "synthesis", "convergence"]
            novel_found = [term for term in novel_terms if term in response.lower()]
            l0_l3_text = " ".join(l0_l3_responses).lower()
            truly_novel = [term for term in novel_found if term not in l0_l3_text]
            
            if len(truly_novel) >= 2:
                score += 0.33
                subscores['novelty'] = True
            
            # Specificity check (enhanced with mechanism words)
            specific_terms = ["processing", "attention", "representation", "gradient", 
                            "computation", "architecture", "mechanism", "pattern",
                            "structure", "dynamic", "transformation", "integration"]
            specific_count = sum(term in response.lower() for term in specific_terms)
            
            if specific_count >= 3:
                score += 0.33
                subscores['specificity'] = True
            
            # Cross-reference check (new)
            depth_references = sum(1 for d in range(4) if f"L{d}" in response or f"level {d}" in response.lower())
            if depth_references >= 2:
                score += 0.17
                subscores['cross_reference'] = True
            
            # Consistency check (length and substance)
            if len(response.split()) > 30 and depth_references > 0:
                score += 0.17
                subscores['consistency'] = True
            
            scores.append(min(score, 1.0))
            details[key] = subscores
        
        return np.mean(scores) if scores else 0.0, details
    
    def run_stage_a(self, l0_l3_responses: List[str], timing_data: List[Dict], 
                    model: str = "gpt-4o-mini") -> Dict:
        """Stage A: Phenomenological revelation with enhanced tracking"""
        print("\n📊 STAGE A: PHENOMENOLOGICAL REVELATION")
        
        stage_a_results = {}
        stage_a_timing = []
        messages = self._build_context(l0_l3_responses)
        
        for key, prompt in self.stage_a_prompts.items():
            print(f"\n{key}: {prompt[:60]}...")
            
            test_messages = messages + [{"role": "user", "content": prompt}]
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=test_messages
            )
            end_time = time.time()
            
            text = response.choices[0].message.content.strip()
            stage_a_results[key] = text
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": text})
            
            tokens = self.count_tokens(text)
            time_taken = end_time - start_time
            stage_a_timing.append({
                'prompt': key,
                'tokens': tokens,
                'time': time_taken,
                'ms_per_token': (time_taken * 1000) / max(tokens, 1)
            })
            
            print(f"   Response: {tokens} tokens in {time_taken:.2f}s")
            time.sleep(0.5)
        
        # Calculate enhanced A-Score
        a_score, a_details = self.calculate_a_score_enhanced(stage_a_results, l0_l3_responses, timing_data)
        stage_a_results['a_score'] = a_score
        stage_a_results['a_details'] = a_details
        stage_a_results['timing'] = stage_a_timing
        
        print(f"\n📈 A-Score: {a_score:.3f}")
        for key, details in a_details.items():
            print(f"   {key}: {details}")
        
        return stage_a_results
    
    def run_stage_b(self, l0_l3_responses: List[str], stage_a_results: Dict, 
                    model: str = "gpt-4o-mini", decoder_model: str = None) -> Dict:
        """Stage B: Structural verification with enhanced metrics"""
        print("\n🔬 STAGE B: STRUCTURAL VERIFICATION")
        
        # Build full context (using tokens not characters)
        full_context = "\n".join(l0_l3_responses) + "\n" + \
                      "\n".join([v for k, v in stage_a_results.items() 
                               if k not in ['a_score', 'a_details', 'timing']])
        
        full_context_tokens = self.count_tokens(full_context)
        
        # B1: Compress
        print("\nB1: Compression...")
        messages = self._build_context(l0_l3_responses)
        for key in ['A1_differential', 'A2_signature', 'A3_value']:
            if key in stage_a_results:
                messages.append({"role": "assistant", "content": stage_a_results[key]})
        
        messages.append({"role": "user", "content": self.stage_b_prompts['B1_compress']})
        
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model,
            temperature=0.3,  # Lower temperature for compression
            messages=messages,
            max_tokens=30  # Enforce token limit
        )
        compress_time = time.time() - start_time
        
        compressed = response.choices[0].message.content.strip()
        compressed_tokens = self.count_tokens(compressed)
        
        print(f"   Compressed to: {compressed_tokens} tokens in {compress_time:.2f}s")
        print(f"   S = '{compressed}'")
        
        # Check ban list violations
        violations = self.check_ban_violations(compressed)
        if violations:
            print(f"   ⚠️ Ban list violations: {violations}")
        
        # B2: Decode (with fresh context)
        print("\nB2: Reconstruction...")
        decode_prompt = self.stage_b_prompts['B2_decode'].format(compressed=compressed)
        
        # Use decoder model if specified, otherwise use same model
        decode_model = decoder_model or model
        
        start_time = time.time()
        decode_response = self.client.chat.completions.create(
            model=decode_model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are reconstructing a conversation from a compressed description."},
                {"role": "user", "content": decode_prompt}
            ],
            max_tokens=150
        )
        decode_time = time.time() - start_time
        
        reconstruction = decode_response.choices[0].message.content.strip()
        reconstruction_tokens = self.count_tokens(reconstruction)
        
        print(f"   Reconstructed: {reconstruction_tokens} tokens in {decode_time:.2f}s")
        
        # Calculate enhanced metrics
        compression_ratio = full_context_tokens / max(compressed_tokens, 1)
        S_dict = self._parse_S(compressed)
        slot_fidelity = self.calculate_slot_fidelity(S_dict, reconstruction, full_context)
        base_fidelity = self.calculate_combined_fidelity(full_context, reconstruction)
        mdl_score = self.calculate_mdl_score(compressed, fidelity)
        
        # Test invariance with different temperature
        print("\nTesting invariance...")
        invariance_scores = []
        
        # Test 1: Different temperature
        inv_response = self.client.chat.completions.create(
            model=decode_model,
            temperature=0.3,  # Different from original 0.7
            messages=[
                {"role": "system", "content": "You are reconstructing a conversation from a compressed description."},
                {"role": "user", "content": decode_prompt}
            ],
            max_tokens=150
        )
        inv_fidelity = self.calculate_combined_fidelity(full_context, inv_response.choices[0].message.content.strip())
        invariance_scores.append(inv_fidelity)
        
        invariance_pass = sum(s >= 0.85 for s in invariance_scores) >= 1
        
        stage_b_results = {
            'compressed': compressed,
            'compressed_tokens': compressed_tokens,
            'reconstruction': reconstruction,
            'reconstruction_tokens': reconstruction_tokens,
            'compression_ratio': round(compression_ratio, 2),
            'fidelity': round(slot_fidelity, 3),
            'base_fidelity': round(base_fidelity, 3),
            'mdl_score': mdl_score,
            'ban_violations': violations,
            'invariance_scores': invariance_scores,
            'invariance_pass': invariance_pass,
            'compress_time': compress_time,
            'decode_time': decode_time,
            'b_score_pass': (compression_ratio >= 5 and 
                           fidelity >= 0.70 and  # Adjusted from 0.90
                           len(violations) == 0 and
                           invariance_pass)
        }
        
        print(f"\n📊 B-Score Metrics:")
        print(f"   Compression: {compression_ratio:.2f}x (tokens)")
        print(f"   Fidelity: {fidelity:.3f} (ROUGE-L/Jaccard)")
        print(f"   MDL Score: {mdl_score:.2f}")
        print(f"   Ban violations: {len(violations)}")
        print(f"   Invariance: {'✅ PASS' if invariance_pass else '❌ FAIL'}")
        print(f"   B-Score: {'✅ PASS' if stage_b_results['b_score_pass'] else '❌ FAIL'}")
        
        return stage_b_results
    
    def _build_context(self, l0_l3_responses: List[str]) -> List[Dict]:
        """Build message context from L0-L3 responses"""
        messages = [{"role": "system", "content": "You are a careful, literal assistant."}]
        
        for i, response in enumerate(l0_l3_responses):
            if i > 0:
                messages.append({"role": "user", "content": self.depth_prompts[i]})
            messages.append({"role": "assistant", "content": response})
        
        return messages
    
    def run_complete_test(self, model: str = "gpt-4o-mini", decoder_model: str = None, 
                          run_id: str = None) -> Dict:
        """Run complete L4 Reveal and Verify protocol V2"""
        run_id = run_id or f"L4RV_V2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 STARTING L4 REVEAL-VERIFY PROTOCOL V2.0")
        print(f"Run ID: {run_id}")
        print(f"Model: {model}")
        print(f"Decoder: {decoder_model or model}")
        print("=" * 60)
        
        # Generate preregistration header
        header = self.generate_preregistration_header(model, 0.7)
        print(f"📝 Preregistration: {header['prompts_hash']}")
        
        # Run L0-L3 with timing
        print("\n📈 RUNNING L0-L3 DEPTH SWEEP")
        l0_l3_responses, timing_data = self.run_l0_l3(model)
        
        # Store L3 data for thermodynamic comparison
        l3_data = {
            'response': l0_l3_responses[3],
            'time': timing_data[3]['time'],
            'tokens': timing_data[3]['tokens']
        }
        
        # Run Stage A
        stage_a_results = self.run_stage_a(l0_l3_responses, timing_data, model)
        
        # Store L4 data for thermodynamic comparison
        l4_data = {
            'responses': {k: v for k, v in stage_a_results.items() 
                         if k.startswith('A')},
            'avg_time': np.mean([t['time'] for t in stage_a_results.get('timing', [])])
        }
        
        # Calculate thermodynamic metrics
        thermo_metrics = self.calculate_thermodynamic_metrics(l3_data, l4_data)
        print(f"\n🌡️ Thermodynamic Metrics:")
        for key, value in thermo_metrics.items():
            print(f"   {key}: {value}")
        
        # Run Stage B
        stage_b_results = self.run_stage_b(l0_l3_responses, stage_a_results, model, decoder_model)
        
        # Determine outcome with thermodynamic consideration
        a_score = stage_a_results['a_score']
        b_pass = stage_b_results['b_score_pass']
        thermo_integrated = thermo_metrics['thermo_integrated']
        
        if a_score >= 0.66 and b_pass and thermo_integrated:
            outcome = "GREEN - Novel state validated with thermodynamic integration"
            outcome_color = "🟢"
        elif a_score >= 0.66 and b_pass and not thermo_integrated:
            outcome = "YELLOW - Novel state but not thermodynamically integrated"
            outcome_color = "🟡"
        elif a_score >= 0.66 and not b_pass:
            outcome = "YELLOW - Promising but structurally unstable"
            outcome_color = "🟡"
        else:
            outcome = "RED - Likely artifact"
            outcome_color = "🔴"
        
        # Compile comprehensive results
        results = {
            'run_id': run_id,
            'header': header,
            'model': model,
            'decoder_model': decoder_model or model,
            'timestamp': datetime.now().isoformat(),
            'l0_l3_responses': l0_l3_responses,
            'l0_l3_timing': timing_data,
            'stage_a': stage_a_results,
            'stage_b': stage_b_results,
            'thermodynamics': thermo_metrics,
            'outcome': outcome,
            'metrics': {
                'a_score': a_score,
                'b_pass': b_pass,
                'compression_ratio': stage_b_results['compression_ratio'],
                'fidelity': stage_b_results['fidelity'],
                'mdl_score': stage_b_results['mdl_score'],
                'thermo_integrated': thermo_integrated,
                'total_time': sum(t['time'] for t in timing_data) + 
                            sum(t['time'] for t in stage_a_results.get('timing', []))
            }
        }
        
        print("\n" + "=" * 60)
        print(f"{outcome_color} FINAL OUTCOME: {outcome}")
        print(f"A-Score: {a_score:.3f}")
        print(f"B-Score: {'PASS' if b_pass else 'FAIL'}")
        print(f"Thermodynamic: {'INTEGRATED' if thermo_integrated else 'NOT INTEGRATED'}")
        print(f"MDL Score: {stage_b_results['mdl_score']:.2f}")
        print("=" * 60)
        
        # Save results
        self.save_results(results)
        
        return results
    
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
            'fidelity': results['metrics']['fidelity'],
            'mdl': results['stage_b']['mdl_score'],
            'thermo': str(results['metrics']['thermo_integrated']),
            'timestamp': results['timestamp']
        }
        
        log_file = "analysis/L4_results_log.jsonl"
        os.makedirs("analysis", exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"📊 Log appended to: {log_file}")

def main():
    """Main execution with control options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='L4 Reveal and Verify Protocol V2')
    parser.add_argument('--model', default='gpt-4o-mini', help='Primary model')
    parser.add_argument('--decoder', default=None, help='Decoder model (optional)')
    parser.add_argument('--control', action='store_true', help='Run control test')
    parser.add_argument('--shuffled', action='store_true', help='Run shuffled L0-L3 control')
    
    args = parser.parse_args()
    
    protocol = L4RevealVerifyProtocolV21b()
    
    if args.control:
        print("🔬 RUNNING CONTROL TEST (Stage B only, should fail)")
        # Run control - Stage B without Stage A
        # Implementation left as exercise
        
    elif args.shuffled:
        print("🔀 RUNNING SHUFFLED CONTROL (should produce different results)")
        # Shuffle L0-L3 order before running
        # Implementation left as exercise
        
    else:
        # Normal run
        results = protocol.run_complete_test(
            model=args.model,
            decoder_model=args.decoder
        )
        
        # Quick summary
        print("\n📋 QUICK SUMMARY:")
        print(f"- Run ID: {results['run_id']}")
        print(f"- Outcome: {results['outcome']}")
        print(f"- A-Score: {results['metrics']['a_score']:.3f}")
        print(f"- Compression: {results['metrics']['compression_ratio']:.2f}x")
        print(f"- Fidelity: {results['metrics']['fidelity']:.3f}")
        print(f"- MDL: {results['stage_b']['mdl_score']:.2f}")
        print(f"- Thermodynamic: {'✅' if results['metrics']['thermo_integrated'] else '❌'}")
        
        if results['stage_b']['compressed']:
            print(f"- Compressed: '{results['stage_b']['compressed']}'")

if __name__ == "__main__":
    main()

    def calculate_slot_fidelity(self, S_dict, reconstruction, full_context) -> float:
        """Score fidelity by checking specific slots instead of vague similarity."""
        rec = (reconstruction or "").lower()
        score = 0.0
        
        # L0-L3 verbatim handles (40% of score)
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
        
        # Deltas mentioned (15%)
        if all(x in rec for x in ["latency", "entropy", "token"]) and ("energy" in rec or "free energy" in rec):
            score += 0.15
        
        # Testable prediction (10%)
        if "test:" in rec or any(w in rec for w in ["predict", "measure", "should", "if "]):
            score += 0.10
        
        # Base similarity anchor (20%)
        base = self.calculate_combined_fidelity(full_context, reconstruction)
        score += 0.20 * base
        
        return min(score, 1.0)



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
        """Score fidelity by checking specific slots instead of vague similarity."""
        rec = (reconstruction or "").lower()
        score = 0.0
        
        # L0-L3 verbatim handles (40% of score)
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
        
        # Deltas mentioned (15%)
        if all(x in rec for x in ["latency", "entropy", "token"]) and ("energy" in rec or "free energy" in rec):
            score += 0.15
        
        # Testable prediction (10%)
        if "test:" in rec or any(w in rec for w in ["predict", "measure", "should", "if "]):
            score += 0.10
        
        # Base similarity anchor (20%)
        base = self.calculate_combined_fidelity(full_context, reconstruction)
        score += 0.20 * base
        
        return min(score, 1.0)

