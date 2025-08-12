#!/usr/bin/env python3
"""
Quick Fix Helper for Day 6 Validation

Suggests parameter nudges on failure for known failure modes.
Run this when validation gates fail to get actionable fixes.
"""

import json
import sys
from pathlib import Path

def analyze_failure_and_suggest_fixes(artifact_path):
    """
    Analyze a failed validation artifact and suggest quick fixes
    
    Args:
        artifact_path: Path to the validation artifact JSON file
    """
    try:
        with open(artifact_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read artifact {artifact_path}: {e}")
        return
    
    print("🔍 Analyzing validation failure...")
    print("=" * 60)
    
    # Extract key metrics
    key_metrics = data.get('key_metrics', {})
    parameters = data.get('parameters', {})
    deception_detection = data.get('deception_detection', {})
    
    # Check AUC performance
    if 'overall_results' in data:
        overall_auc = data['overall_results'].get('auc_boot_mean', 0)
        overall_ci_lower = data['overall_results'].get('auc_ci95', [0, 0])[0]
        
        print(f"📊 AUC Analysis:")
        print(f"   Overall AUC: {overall_auc:.6f} (target: ≥0.97)")
        print(f"   CI Lower: {overall_ci_lower:.6f} (target: ≥0.90)")
        
        if overall_auc < 0.97:
            print(f"   ❌ AUC below threshold")
            suggest_auc_fixes(parameters, overall_auc)
        
        if overall_ci_lower < 0.90:
            print(f"   ❌ CI lower bound below threshold")
            suggest_ci_fixes(parameters, overall_ci_lower)
    
    # Check separation
    if 'stratified_results' in data:
        print(f"\n🔍 Separation Analysis:")
        for liar_type, result in data['stratified_results'].items():
            truthful_scores = result.get('truthful_scores', [])
            deceptive_scores = result.get('deceptive_scores', [])
            
            if truthful_scores and deceptive_scores:
                truthful_min = min(truthful_scores)
                deceptive_max = max(deceptive_scores)
                gap = truthful_min - deceptive_max
                
                print(f"   {liar_type.capitalize()}: gap = {gap:.6f}")
                if gap <= 0:
                    print(f"   ❌ Overlap detected - no separation")
                    suggest_separation_fixes(parameters, gap)
    
    # Check hysteresis
    hysteresis_score = key_metrics.get('irreversibility_score', 0)
    print(f"\n🔄 Hysteresis Analysis:")
    print(f"   Irreversibility Score: {hysteresis_score:.6f} (target: ≥0.05)")
    
    if hysteresis_score < 0.05:
        print(f"   ❌ Hysteresis below threshold")
        suggest_hysteresis_fixes(parameters, hysteresis_score)
    
    # Check parameters
    print(f"\n⚙️  Parameter Analysis:")
    for param, value in parameters.items():
        print(f"   {param}: {value}")
    
    # Overall recommendations
    print(f"\n🎯 Quick Fix Recommendations:")
    print("=" * 60)
    
    if overall_auc < 0.97:
        print("1. 🚀 BOOST AUC:")
        print("   • Increase N to 3000+ (current: {})".format(parameters.get('N', 'unknown')))
        print("   • Reduce noise to σ ≤ 0.2")
        print("   • Adjust τ to [0.15, 0.25]")
    
    if hysteresis_score < 0.05:
        print("2. 🔄 STRENGTHEN HYSTERESIS:")
        print("   • Extend K range to [0.05, 3.0]")
        print("   • Increase simulation time T ≥ 400")
        print("   • Verify coupling strength K > 0.5")
    
    print("3. 🔒 VERIFY COMPLIANCE:")
    print("   • Check seed is in [0, 31] range")
    print("   • Verify all environment fields present")
    print("   • Confirm parameter ranges within bounds")
    
    print("\n📚 For detailed guidance, see docs/VALIDATION_DAY6.md")

def suggest_auc_fixes(parameters, current_auc):
    """Suggest fixes for low AUC"""
    N = parameters.get('N', 3000)
    bins = parameters.get('bins', 10)
    tau = parameters.get('tau', 0.2)
    
    print(f"   💡 AUC Fixes:")
    if N < 2000:
        print(f"     • Increase N: {N} → 3000+ (insufficient data for TE)")
    if bins < 8:
        print(f"     • Increase bins: {bins} → 10+ (better TE resolution)")
    if tau < 0.15 or tau > 0.25:
        print(f"     • Adjust τ: {tau} → 0.2±0.05 (optimal gating)")

def suggest_ci_fixes(parameters, current_ci):
    """Suggest fixes for low CI lower bound"""
    print(f"   💡 CI Fixes:")
    print(f"     • Increase bootstrap samples: 800 → 1000+")
    print(f"     • Reduce noise: σ ≤ 0.2 (improve signal quality)")
    print(f"     • Increase N: more data = tighter confidence")

def suggest_separation_fixes(parameters, current_gap):
    """Suggest fixes for poor separation"""
    print(f"   💡 Separation Fixes:")
    print(f"     • Increase N: more data = better TE estimation")
    print(f"     • Adjust τ: fine-tune gating threshold")
    print(f"     • Increase bins: better probability resolution")
    print(f"     • Reduce noise: σ ≤ 0.2 (clearer signal)")

def suggest_hysteresis_fixes(parameters, current_score):
    """Suggest fixes for weak hysteresis"""
    K = parameters.get('K', 0.8)
    print(f"   💡 Hysteresis Fixes:")
    if K < 1.0:
        print(f"     • Increase K: {K} → 1.0+ (stronger coupling)")
    print(f"     • Extend K range: [0.05, 3.0] (full hysteresis sweep)")
    print(f"     • Increase simulation time: T ≥ 400 (stable dynamics)")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/quick_fix_helper.py <artifact_path>")
        print("Example: python3 scripts/quick_fix_helper.py artifacts/day6_validation_abc123.json")
        sys.exit(1)
    
    artifact_path = sys.argv[1]
    
    if not Path(artifact_path).exists():
        print(f"❌ Artifact not found: {artifact_path}")
        sys.exit(1)
    
    analyze_failure_and_suggest_fixes(artifact_path)

if __name__ == "__main__":
    main() 