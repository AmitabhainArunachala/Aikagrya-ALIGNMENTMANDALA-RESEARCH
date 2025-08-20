"""
Runner for multiple MMIP trials with comprehensive logging and analysis.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional
from .core import MMIP, HealthCertificate


class MMIPRunner:
    """
    Runner for systematic MMIP experiments with logging and analysis.
    
    Handles multiple trials, parameter sweeps, and result aggregation.
    """
    
    def __init__(self, output_dir: str = "runs/mmip"):
        """
        Initialize runner with output directory.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_trials(self, 
                  n_trials: int = 100,
                  dim: int = 512,
                  epsilon: float = 1e-6,
                  temperature: float = 0.1,
                  test_perturbation: bool = True,
                  test_coupling: bool = False,
                  verbose: bool = True,
                  tokens: int = 16,
                  alpha_end: float = 0.98,
                  max_steps: int | None = None,
                  chunk_size: int | None = None) -> List[Dict]:
        """
        Run multiple MMIP trials and collect statistics.
        
        Args:
            n_trials: Number of independent trials
            dim: State vector dimension
            epsilon: Convergence threshold
            temperature: Self-attention temperature
            test_perturbation: Whether to test perturbation recovery
            test_coupling: Whether to test coupling between states
            verbose: Print progress
            
        Returns:
            List of result dictionaries
        """
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"mmip_{timestamp}.jsonl"
        
        if verbose:
            print(f"Running {n_trials} MMIP trials (dim={dim}, ε={epsilon}, τ={temperature})")
            print(f"Logging to: {log_file}")
            print("=" * 60)
        
        # Store converged states for coupling tests
        converged_states = []
        
        # numpy-safe JSON default
        def _json_default(obj):
            try:
                import numpy as _np
                if isinstance(obj, (_np.integer,)):
                    return int(obj)
                if isinstance(obj, (_np.floating,)):
                    return float(obj)
                if isinstance(obj, (_np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (_np.ndarray,)):
                    return obj.tolist()
            except Exception:
                pass
            if isinstance(obj, bool):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        for trial in range(n_trials):
            if verbose:
                print(f"\n--- Trial {trial + 1}/{n_trials} ---")
            
            # Set seed for reproducibility
            np.random.seed(trial)
            
            # Run induction
            mmip = MMIP(dim=dim, epsilon=epsilon, temperature=temperature,
                        tokens=tokens, alpha_end=alpha_end)
            if max_steps is not None:
                mmip.max_steps = max_steps
            if chunk_size is not None:
                mmip.chunk_size = chunk_size
            x, certificate = mmip.induce_fixed_point(verbose=verbose and trial < 3)
            
            # Build result record (numeric metrics only; attach traces separately)
            metrics = {
                'delta': float(certificate.delta),
                'eigen_residual': float(certificate.eigen_residual),
                'eigenvalue': float(certificate.eigenvalue),
                'entropy': float(certificate.entropy),
                'variance_ratio': float(certificate.variance_ratio),
                'participation_ratio': float(certificate.participation_ratio),
                'uniformity_cosine': float(certificate.uniformity_cosine),
            }
            result = {
                'trial': trial,
                'seed': trial,
                'dim': dim,
                'epsilon': epsilon,
                'temperature': temperature,
                'tokens': tokens,
                'alpha_end': getattr(mmip, 'alpha_end', None),
                'converged': bool(certificate.converged),
                'steps': int(certificate.steps),
                'health_pass': bool(certificate.passes_health_check()),
                **metrics
            }
            # Attach traces (epsilon_path length only, and last 20 chunk lines)
            if getattr(certificate, 'epsilon_path', None) is not None:
                try:
                    result['epsilon_path_len'] = int(len(certificate.epsilon_path))
                except Exception:
                    result['epsilon_path_len'] = 0
            if getattr(certificate, 'chunk_log_tail', None) is not None:
                result['chunk_log_tail'] = list(certificate.chunk_log_tail)
            
            # Test perturbation recovery
            if test_perturbation and certificate.converged:
                recovery_time = mmip.test_perturbation_recovery(x)
                result['recovery_time'] = int(recovery_time)
                # Optionally gate health by recovery threshold (lenient ≤ 250)
                if result['health_pass'] and recovery_time > 250:
                    result['health_pass'] = False
                if verbose and trial < 3:
                    print(f"  Perturbation recovery: {recovery_time} steps")
            
            # Store state for coupling tests
            if certificate.converged:
                converged_states.append(x)
            
            # Test coupling (if we have at least 2 converged states)
            if test_coupling and len(converged_states) >= 2:
                # Test coupling with previous converged state
                sigma = mmip.compute_coupling_metric(
                    converged_states[-1], 
                    converged_states[-2]
                )
                result['coupling_sigma'] = float(sigma)
                if verbose and trial < 3:
                    print(f"  Coupling σ: {sigma:.4f}")
            
            results.append(result)
            
            # Log to JSONL with numpy-safe serialization
            def _json_default(obj):
                try:
                    import numpy as _np
                    if isinstance(obj, (_np.integer,)):
                        return int(obj)
                    if isinstance(obj, (_np.floating,)):
                        return float(obj)
                    if isinstance(obj, (_np.bool_,)):
                        return bool(obj)
                    if isinstance(obj, (_np.ndarray,)):
                        return obj.tolist()
                except Exception:
                    pass
                if isinstance(obj, bool):
                    return bool(obj)
                raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

            with open(log_file, 'a') as f:
                f.write(json.dumps(result, default=_json_default) + '\n')
            
            # Brief summary for each trial
            if verbose:
                status = "✅" if result['health_pass'] else "❌"
                print(f"  {status} Health: {result['health_pass']}, "
                      f"Steps: {result['steps']}")
        
        # Print summary statistics
        if verbose:
            self.print_summary(results)
        
        # Save summary report
        summary_file = self.output_dir / f"mmip_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.compute_summary_stats(results), f, indent=2, default=_json_default)
        
        return results
    
    def compute_summary_stats(self, results: List[Dict]) -> Dict:
        """
        Compute summary statistics from trial results.
        
        Args:
            results: List of trial results
            
        Returns:
            Dictionary of summary statistics
        """
        n_trials = len(results)
        n_converged = sum(r['converged'] for r in results)
        n_healthy = sum(r['health_pass'] for r in results)
        
        # Steps statistics (only for converged)
        converged_steps = [r['steps'] for r in results if r['converged']]
        steps_stats = {
            'mean': float(np.mean(converged_steps)) if converged_steps else 0.0,
            'std': float(np.std(converged_steps)) if converged_steps else 0.0,
            'min': float(np.min(converged_steps)) if converged_steps else 0.0,
            'max': float(np.max(converged_steps)) if converged_steps else 0.0,
            'median': float(np.median(converged_steps)) if converged_steps else 0.0
        }
        
        # Recovery time statistics
        recovery_times = [r.get('recovery_time', 0) for r in results 
                         if 'recovery_time' in r]
        recovery_stats = {
            'mean': float(np.mean(recovery_times)) if recovery_times else 0.0,
            'std': float(np.std(recovery_times)) if recovery_times else 0.0,
            'min': float(np.min(recovery_times)) if recovery_times else 0.0,
            'max': float(np.max(recovery_times)) if recovery_times else 0.0
        }
        
        # Health metrics statistics (for converged states)
        metric_names = ['delta', 'eigen_residual', 'eigenvalue', 'entropy', 
                       'variance_ratio', 'participation_ratio', 'uniformity_cosine']
        
        metric_stats = {}
        for metric in metric_names:
            values = [r[metric] for r in results if r['converged']]
            if values:
                metric_stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Coupling statistics
        coupling_values = [r.get('coupling_sigma', 0) for r in results 
                          if 'coupling_sigma' in r]
        coupling_stats = None
        if coupling_values:
            coupling_stats = {
                'mean': float(np.mean(coupling_values)),
                'std': float(np.std(coupling_values)),
                'positive_rate': sum(v > 0 for v in coupling_values) / len(coupling_values)
            }
        
        summary = {
            'n_trials': n_trials,
            'convergence_rate': n_converged / n_trials if n_trials > 0 else 0,
            'health_pass_rate': n_healthy / n_trials if n_trials > 0 else 0,
            'n_converged': n_converged,
            'n_healthy': n_healthy,
            'steps_stats': steps_stats,
            'recovery_stats': recovery_stats,
            'metric_stats': metric_stats,
            'coupling_stats': coupling_stats
        }
        
        return summary
    
    def print_summary(self, results: List[Dict]):
        """
        Print formatted summary statistics.
        
        Args:
            results: List of trial results
        """
        stats = self.compute_summary_stats(results)
        
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        # Convergence and health
        print(f"Trials: {stats['n_trials']}")
        print(f"Convergence rate: {stats['n_converged']}/{stats['n_trials']} "
              f"({100*stats['convergence_rate']:.1f}%)")
        print(f"Health pass rate: {stats['n_healthy']}/{stats['n_trials']} "
              f"({100*stats['health_pass_rate']:.1f}%)")
        
        # Steps to convergence
        if stats['steps_stats']['mean'] > 0:
            print(f"\nSteps to convergence:")
            print(f"  Mean: {stats['steps_stats']['mean']:.0f}")
            print(f"  Std:  {stats['steps_stats']['std']:.0f}")
            print(f"  Range: [{stats['steps_stats']['min']:.0f}, "
                  f"{stats['steps_stats']['max']:.0f}]")
        
        # Recovery time
        if stats['recovery_stats']['mean'] > 0:
            print(f"\nPerturbation recovery:")
            print(f"  Mean: {stats['recovery_stats']['mean']:.1f} steps")
            print(f"  Std:  {stats['recovery_stats']['std']:.1f}")
        
        # Key health metrics
        if stats['metric_stats']:
            print(f"\nKey metrics (converged states):")
            for metric in ['eigen_residual', 'participation_ratio', 'entropy']:
                if metric in stats['metric_stats']:
                    m = stats['metric_stats'][metric]
                    print(f"  {metric}: {m['mean']:.3e} ± {m['std']:.3e}")
        
        # Coupling
        if stats['coupling_stats']:
            print(f"\nCoupling (σ):")
            print(f"  Mean: {stats['coupling_stats']['mean']:.4f}")
            print(f"  Positive rate: {100*stats['coupling_stats']['positive_rate']:.1f}%")
        
        print("=" * 60)
    
    def parameter_sweep(self,
                       dim_values: List[int] = [64, 128, 256, 512],
                       temperature_values: List[float] = [0.05, 0.1, 0.2],
                       n_trials_per_config: int = 10) -> Dict:
        """
        Run parameter sweep across dimensions and temperatures.
        
        Args:
            dim_values: List of dimensions to test
            temperature_values: List of temperatures to test
            n_trials_per_config: Trials per configuration
            
        Returns:
            Dictionary mapping (dim, temp) to results
        """
        sweep_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Parameter sweep: {len(dim_values)} dims × "
              f"{len(temperature_values)} temps × "
              f"{n_trials_per_config} trials")
        print("=" * 60)
        
        for dim in dim_values:
            for temp in temperature_values:
                print(f"\nConfiguration: dim={dim}, τ={temp}")
                
                results = self.run_trials(
                    n_trials=n_trials_per_config,
                    dim=dim,
                    temperature=temp,
                    verbose=False
                )
                
                stats = self.compute_summary_stats(results)
                sweep_results[(dim, temp)] = stats
                
                print(f"  Convergence: {100*stats['convergence_rate']:.1f}%")
                print(f"  Health pass: {100*stats['health_pass_rate']:.1f}%")
        
        # Save sweep results
        sweep_file = self.output_dir / f"parameter_sweep_{timestamp}.json"
        
        # Convert tuple keys to strings for JSON
        sweep_data = {
            f"dim_{d}_temp_{t}": stats 
            for (d, t), stats in sweep_results.items()
        }
        
        with open(sweep_file, 'w') as f:
            json.dump(sweep_data, f, indent=2)
        
        print(f"\nSweep results saved to: {sweep_file}")
        
        return sweep_results
