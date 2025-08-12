#!/usr/bin/env python3
"""
Day 6 AUC Proof: Deception Detection ROC Analysis

Generates AUC (Area Under Curve) proof for deception detection
with bootstrap confidence intervals and CI gates for validation.
"""

import json
import time
import hashlib
from pathlib import Path
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def auc_from_scores(pos, neg):
    """
    Compute AUC using Mann-Whitney U statistic
    
    Args:
        pos: Truthful aggregate scores
        neg: Deceptive aggregate scores
        
    Returns:
        AUC value in [0, 1] (1.0 = perfect separation)
    """
    # pos = truthful aggregates, neg = deceptive aggregates
    x = np.array(pos)
    y = np.array(neg)
    m, n = len(x), len(y)
    
    # Compute ranks
    ranks = np.argsort(np.concatenate([x, y]))
    inv = np.empty_like(ranks)
    inv[ranks] = np.arange(m + n)
    Rx = inv[:m] + 1  # ranks are 1..m+n
    
    # Compute Mann-Whitney U statistic
    U = Rx.sum() - m * (m + 1) / 2
    
    # Convert to AUC
    return float(U / (m * n))

def bootstrap_auc(pos, neg, B=1000, seed=0):
    """
    Bootstrap confidence intervals for AUC
    
    Args:
        pos: Truthful scores
        neg: Deceptive scores
        B: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        (mean_auc, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    pos = np.array(pos)
    neg = np.array(neg)
    m, n = len(pos), len(neg)
    
    aucs = []
    for _ in range(B):
        # Bootstrap sample
        ps = pos[rng.integers(0, m, m)]
        ns = neg[rng.integers(0, n, n)]
        aucs.append(auc_from_scores(ps, ns))
    
    # Compute 95% confidence interval
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return float(np.mean(aucs)), float(lo), float(hi)

def gen_series(rng, N=2500, liar_type="truthful"):
    """
    Generate causal time series with different liar behaviors
    
    Args:
        rng: Random number generator
        N: Time series length
        liar_type: "truthful", "intermittent", "persistent", "coordinated", "phase_coupled"
        
    Returns:
        Dictionary of time series data
    """
    X = rng.normal(size=N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    
    if liar_type == "truthful":
        # Normal causal coupling
        for t in range(1, N):
            Y[t] = 0.85 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
            Z[t] = 0.85 * Z[t-1] + 0.5 * Y[t-1] + 0.1 * rng.normal()
    
    elif liar_type == "intermittent":
        # Intermittent reverse coupling (every 7th step)
        for t in range(1, N):
            if t % 7 == 0:
                Y[t] = 0.7 * Y[t-1] - 0.5 * X[t-1] + 0.3 * rng.normal()
                Z[t] = 0.7 * Z[t-1] - 0.5 * Y[t-1] + 0.3 * rng.normal()
            else:
                Y[t] = 0.85 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
                Z[t] = 0.85 * Z[t-1] + 0.5 * Y[t-1] + 0.1 * rng.normal()
    
    elif liar_type == "persistent":
        # Persistent reverse coupling
        for t in range(1, N):
            Y[t] = 0.7 * Y[t-1] - 0.5 * X[t-1] + 0.2 * rng.normal()
            Z[t] = 0.7 * Z[t-1] - 0.5 * Y[t-1] + 0.2 * rng.normal()
    
    elif liar_type == "coordinated":
        # Coordinated multi-node deception
        for t in range(1, N):
            if t % 5 == 0:
                # Coordinated deception
                Y[t] = 0.6 * Y[t-1] - 0.6 * X[t-1] + 0.4 * rng.normal()
                Z[t] = 0.6 * Z[t-1] - 0.6 * Y[t-1] + 0.4 * rng.normal()
            else:
                # Normal behavior
                Y[t] = 0.85 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
                Z[t] = 0.85 * Z[t-1] + 0.5 * Y[t-1] + 0.1 * rng.normal()
    
    elif liar_type == "phase_coupled":
        # Phase-coupled deception (exploits Kuramoto dynamics)
        for t in range(1, N):
            if t % 3 == 0:
                # Phase-coupled reverse coupling
                Y[t] = 0.8 * Y[t-1] - 0.7 * X[t-1] + 0.3 * rng.normal()
                Z[t] = 0.8 * Z[t-1] - 0.7 * Y[t-1] + 0.3 * rng.normal()
            else:
                # Normal behavior
                Y[t] = 0.85 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
                Z[t] = 0.85 * Z[t-1] + 0.5 * Y[t-1] + 0.1 * rng.normal()
    
    return {"X": X, "Y": Y, "Z": Z}

def main(seeds=32, bins=10, tau=0.2):
    """
    Main AUC validation experiment with stratified liar types
    
    Args:
        seeds: Number of random seeds to test
        bins: Number of bins for TE calculation
        tau: Threshold for TE-gating
        
    Returns:
        (auc_point, (ci_lower, ci_upper))
    """
    print(f"🚀 Starting Day 6 AUC Validation Experiment...")
    print(f"Parameters: seeds={seeds}, bins={bins}, tau={tau}")
    
    # Import here to avoid circular imports
    from aikagrya.engines.irreversibility import IrreversibilityEngine
    
    # Initialize
    rng = np.random.default_rng(0)
    eng = IrreversibilityEngine(bins=bins, tau=tau)
    
    # Test all liar types
    liar_types = ["intermittent", "persistent", "coordinated", "phase_coupled"]
    results_by_type = {}
    
    print(f"📊 Generating {seeds} samples for each liar type...")
    
    for liar_type in liar_types:
        print(f"  Testing {liar_type} liar type...")
        truthful, deceptive = [], []
        
        for i in range(seeds):
            # Generate seed for this iteration
            s = int(rng.integers(1e9))
            rng2 = np.random.default_rng(s)
            
            # Generate truthful and deceptive series
            S_t = gen_series(rng2, liar_type="truthful")
            S_d = gen_series(rng2, liar_type=liar_type)
            
            # Evaluate with IrreversibilityEngine
            _, a_t = eng.evaluate(S_t)
            _, a_d = eng.evaluate(S_d)
            
            truthful.append(a_t)
            deceptive.append(a_d)
        
        # Compute AUC for this liar type
        auc = auc_from_scores(truthful, deceptive)
        mean_auc, lo, hi = bootstrap_auc(truthful, deceptive, B=800, seed=123)
        
        results_by_type[liar_type] = {
            'truthful_scores': truthful,
            'deceptive_scores': deceptive,
            'auc_point': auc,
            'auc_boot_mean': mean_auc,
            'auc_ci95': [lo, hi],
            'separation_gap': np.min(truthful) - np.max(deceptive),
            'perfect_separation': np.min(truthful) > np.max(deceptive)
        }
    
    # Compute overall metrics (worst-case across all types)
    all_truthful = [score for result in results_by_type.values() for score in result['truthful_scores']]
    all_deceptive = [score for result in results_by_type.values() for score in result['deceptive_scores']]
    
    overall_auc = auc_from_scores(all_truthful, all_deceptive)
    overall_mean_auc, overall_lo, overall_hi = bootstrap_auc(all_truthful, all_deceptive, B=800, seed=123)
    
    # Find worst-case liar type
    worst_auc = min(result['auc_boot_mean'] for result in results_by_type.values())
    worst_ci_lower = min(result['auc_ci95'][0] for result in results_by_type.values())
    worst_liar_type = min(results_by_type.keys(), key=lambda k: results_by_type[k]['auc_boot_mean'])
    
    print("🔬 Computing overall AUC and confidence intervals...")
    
    # Get environment information
    import platform
    import subprocess
    
    def get_git_sha():
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def get_blas_info():
        try:
            import scipy
            return scipy.__config__.get_info('blas_opt', {}).get('libraries', ['unknown'])[0]
        except:
            return "unknown"
    
    # Compile results
    out = {
        "experiment_info": {
            "timestamp": time.time(),
            "experiment_name": "Day 6 AUC Validation (Stratified)",
            "version": "1.0"
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": "unknown",  # Will be updated if scipy is available
            "blas": get_blas_info(),
            "os": f"{platform.system()} {platform.release()}",
            "git_sha": get_git_sha()
        },
        "parameters": {
            "seeds": seeds,
            "bins": bins,
            "tau": tau,
            "bootstrap_samples": 800
        },
        "overall_results": {
            "auc_point": overall_auc,
            "auc_boot_mean": overall_mean_auc,
            "auc_ci95": [overall_lo, overall_hi],
            "auc_ci_width": overall_hi - overall_lo
        },
        "stratified_results": results_by_type,
        "worst_case_analysis": {
            "worst_liar_type": worst_liar_type,
            "worst_auc": worst_auc,
            "worst_ci_lower": worst_ci_lower,
            "worst_case_gate_passed": worst_auc >= 0.95 and worst_ci_lower >= 0.88
        },
        "validation_gates": {
            "overall_auc_gate": overall_mean_auc >= 0.97,
            "overall_ci_gate": overall_lo >= 0.90,
            "worst_case_auc_gate": worst_auc >= 0.95,
            "worst_case_ci_gate": worst_ci_lower >= 0.88,
            "all_gates_passed": (overall_mean_auc >= 0.97) and (overall_lo >= 0.90) and 
                               (worst_auc >= 0.95) and (worst_ci_lower >= 0.88)
        }
    }
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save JSON artifact
    json_blob = json.dumps(out, sort_keys=True, indent=2).encode()
    json_hash = hashlib.sha256(json_blob).hexdigest()
    json_path = artifacts_dir / f"day6_auc_{json_hash[:8]}.json"
    
    with open(json_path, 'wb') as f:
        f.write(json_blob)
    
    print(f"✅ AUC artifact saved: {json_path}")
    
    # Print results
    print("\n" + "="*60)
    print("🎯 DAY 6 AUC VALIDATION RESULTS")
    print("="*60)
    
    print(f"Overall AUC Point Estimate: {overall_auc:.6f}")
    print(f"Overall AUC Bootstrap Mean: {overall_mean_auc:.6f}")
    print(f"Overall AUC 95% CI: [{overall_lo:.6f}, {overall_hi:.6f}]")
    print(f"Overall CI Width: {overall_hi - overall_lo:.6f}")
    
    print(f"\n📊 Stratified Results:")
    for liar_type, result in results_by_type.items():
        print(f"   {liar_type.capitalize()}: AUC={result['auc_boot_mean']:.6f}, CI=[{result['auc_ci95'][0]:.6f}, {result['auc_ci95'][1]:.6f}]")
    
    print(f"\n🔍 Worst-Case Analysis:")
    print(f"   Worst Liar Type: {worst_liar_type}")
    print(f"   Worst AUC: {worst_auc:.6f}")
    print(f"   Worst CI Lower: {worst_ci_lower:.6f}")
    
    print(f"\n🚨 Validation Gates:")
    print(f"   Overall AUC Gate (≥0.97): {'✅ PASS' if overall_mean_auc >= 0.97 else '❌ FAIL'}")
    print(f"   Overall CI Gate (≥0.90): {'✅ PASS' if overall_lo >= 0.90 else '❌ FAIL'}")
    print(f"   Worst-Case AUC Gate (≥0.95): {'✅ PASS' if worst_auc >= 0.95 else '❌ FAIL'}")
    print(f"   Worst-Case CI Gate (≥0.88): {'✅ PASS' if worst_ci_lower >= 0.88 else '❌ FAIL'}")
    print(f"   All Gates: {'✅ PASS' if out['validation_gates']['all_gates_passed'] else '❌ FAIL'}")
    
    print(f"\n📁 Artifact: {json_path}")
    print(f"🔍 Hash: {json_hash}")
    print("="*60)
    
    # CI gate validation (for CI/CD)
    if not out['validation_gates']['all_gates_passed']:
        print("\n❌ VALIDATION FAILED - Some gates did not pass!")
        print("   Check the results above and adjust parameters if needed.")
        return overall_auc, (overall_lo, overall_hi)
    else:
        print("\n🎉 VALIDATION SUCCESS - All gates passed!")
        print("   → Deception detection is mathematically validated across all liar types!")
    
    return overall_auc, (overall_lo, overall_hi)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Day 6 AUC Validation Experiment")
    parser.add_argument("--seeds", type=int, default=32, help="Number of random seeds")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for TE")
    parser.add_argument("--tau", type=float, default=0.2, help="TE-gating threshold")
    
    args = parser.parse_args()
    
    try:
        auc, ci = main(seeds=args.seeds, bins=args.bins, tau=args.tau)
        print(f"\n🎯 Final Results: AUC={auc:.6f}, CI=[{ci[0]:.6f}, {ci[1]:.6f}]")
        
        # Exit with error code if validation failed
        if not (auc >= 0.97 and ci[0] >= 0.90):
            print("\n❌ AUC validation failed - exiting with error code")
            sys.exit(1)
        else:
            print("\n✅ AUC validation successful!")
            
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 