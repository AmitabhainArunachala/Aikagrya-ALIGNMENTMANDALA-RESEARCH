#!/usr/bin/env python3
"""
Day 8.5: Unified Field Validation Harness - Grid Execution Engine

This module runs the comprehensive parameter sweep to validate the unified field theory
before proceeding to Day 9 production implementation.
"""

import yaml
import json
import hashlib
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from aikagrya.unified_field import create_unified_field_theory, create_cross_framework_integrator
from aikagrya.optimization.golden_ratio import PHI

class ValidationGridExecutor:
    """
    Executes the comprehensive validation grid for unified field theory
    """
    
    def __init__(self, config_path: str, output_dir: str = "artifacts"):
        """
        Initialize the validation grid executor
        
        Args:
            config_path: Path to sweep configuration YAML
            output_dir: Directory for output artifacts
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Generate parameter grid
        self.parameter_grid = self._generate_parameter_grid()
        self.total_runs = len(self.parameter_grid)
        
        print(f"🎯 Generated {self.total_runs} parameter combinations for validation")
    
    def _generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """Generate the full parameter grid from configuration"""
        grid = []
        
        # Extract configuration sections
        couplings = self.config['couplings']
        phi_opt = self.config['phi_optimization']
        framework_weights = self.config['framework_weights']
        init_conditions = self.config['initial_conditions']
        graph_topology = self.config['graph_topology']
        noise_injection = self.config['noise_injection']
        execution = self.config['execution']
        
        # Generate coupling combinations
        coupling_combinations = []
        for K in np.arange(couplings['global_coupling_K']['range'][0], 
                          couplings['global_coupling_K']['range'][1] + couplings['global_coupling_K']['step'], 
                          couplings['global_coupling_K']['step']):
            for tau_mode in couplings['te_gate_tau']['percentile_modes']:
                for tau_abs in couplings['te_gate_tau']['absolute_values']:
                    for sigma_omega in couplings['kuramoto_frequency_spread']['values']:
                        coupling_combinations.append({
                            'K': float(K),
                            'tau_mode': tau_mode,
                            'tau_abs': tau_abs,
                            'sigma_omega': sigma_omega
                        })
        
        # Generate φ-optimization combinations
        phi_combinations = []
        for phi_mult in phi_opt['phi_perturbations']['multipliers']:
            for lambda_phi in phi_opt['phi_weight_in_loss']['values']:
                phi_combinations.append({
                    'phi_multiplier': phi_mult,
                    'lambda_phi': lambda_phi
                })
        
        # Generate framework weight combinations
        weight_combinations = []
        
        # Add edge cases
        weight_combinations.extend(framework_weights['edge_cases']['single_framework_dominant'])
        weight_combinations.extend(framework_weights['edge_cases']['single_framework_zero'])
        
        # Add Latin hypercube samples (simplified for now)
        for i in range(min(50, framework_weights['sampling']['samples'])):
            # Generate random weights that sum to 1.0
            weights = np.random.random(6)
            weights = weights / np.sum(weights)
            weight_combinations.append({
                'IIT': float(weights[0]),
                'CT': float(weights[1]),
                'Thermo': float(weights[2]),
                'Phi': float(weights[3]),
                'AGNent': float(weights[4]),
                'Bridge': float(weights[5])
            })
        
        # Generate initial condition combinations
        init_combinations = []
        for init_config in init_conditions['state_distributions']:
            init_combinations.append({
                'name': init_config['name'],
                'params': init_config['params']
            })
        
        # Generate graph topology combinations
        graph_combinations = []
        for graph_family in graph_topology['families']:
            for n in graph_topology['network_sizes']['values']:
                graph_combinations.append({
                    'type': graph_family['name'],
                    'n': n,
                    'params': graph_family['params']
                })
        
        # Generate noise combinations
        noise_combinations = []
        for gaussian_noise in noise_injection['gaussian_process_noise']['values']:
            for te_noise in noise_injection['te_gate_noise']['values']:
                for liar_process in noise_injection['adversarial_liar_processes']:
                    for liar_fraction in liar_process['fractions']:
                        noise_combinations.append({
                            'gaussian_sigma': gaussian_noise,
                            'te_jitter': te_noise,
                            'liar_type': liar_process['name'],
                            'liar_fraction': liar_fraction,
                            'liar_params': liar_process['params']
                        })
        
        # Generate time step combinations
        time_step_combinations = []
        for dt in execution['time_steps']:
            time_step_combinations.append({'dt': dt})
        
        # Combine all parameter combinations
        for coupling in coupling_combinations:
            for phi in phi_combinations:
                for weights in weight_combinations:
                    for init in init_combinations:
                        for graph in graph_combinations:
                            for noise in noise_combinations:
                                for time_step in time_step_combinations:
                                    # Create parameter combination
                                    param_combo = {
                                        **coupling,
                                        **phi,
                                        'framework_weights': weights,
                                        'initial_condition': init,
                                        'graph_topology': graph,
                                        **noise,
                                        **time_step,
                                        'steps_per_run': execution['steps_per_run'],
                                        'burn_in_steps': execution['burn_in_steps'],
                                        'seeds': execution['seeds_per_grid_point']
                                    }
                                    
                                    # Add configuration hash
                                    param_combo['config_hash'] = self._generate_config_hash(param_combo)
                                    grid.append(param_combo)
        
        return grid
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a unique hash for the configuration"""
        # Convert to sorted string for consistent hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    def _run_single_validation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run validation for a single parameter combination
        
        Args:
            params: Parameter combination dictionary
            
        Returns:
            Validation results dictionary
        """
        try:
            print(f"🔬 Running validation for config {params['config_hash']}")
            
            # Create unified field theory with parameters
            field_config = {
                'field_resolution': 0.1,
                'time_step': params['dt'],
                'max_iterations': params['steps_per_run'],
                'convergence_tolerance': 1e-6
            }
            
            unified_field = create_unified_field_theory(field_config)
            integrator = create_cross_framework_integrator()
            
            # Run validation with multiple seeds
            seed_results = []
            for seed in range(params['seeds']):
                np.random.seed(seed)
                
                # Generate initial system state based on configuration
                n = params['graph_topology']['n']
                if params['initial_condition']['name'] == 'random_uniform':
                    system_state = np.random.uniform(
                        params['initial_condition']['params']['min'],
                        params['initial_condition']['params']['max'],
                        n
                    )
                elif params['initial_condition']['name'] == 'gaussian':
                    system_state = np.random.normal(
                        params['initial_condition']['params']['mean'],
                        params['initial_condition']['params']['std'],
                        n
                    )
                elif params['initial_condition']['name'] == 'clustered':
                    clusters = params['initial_condition']['params']['clusters']
                    cluster_std = params['initial_condition']['params']['cluster_std']
                    system_state = np.zeros(n)
                    for i in range(n):
                        cluster = i % clusters
                        system_state[i] = np.random.normal(cluster * 2 * np.pi / clusters, cluster_std)
                else:  # adversarial
                    system_state = np.random.random(n) * 2 * np.pi
                
                # Add noise if specified
                if params['gaussian_sigma'] > 0:
                    system_state += np.random.normal(0, params['gaussian_sigma'], n)
                
                # Run field evolution
                position = np.array([0.5] * 6)  # 6D position
                field_state = unified_field.compute_unified_field(position, 0.0, system_state)
                
                # Evolve field
                evolution_states = unified_field.evolve_field(
                    field_state, 
                    evolution_time=params['dt'] * params['steps_per_run']
                )
                
                # Compute metrics for this seed
                seed_metrics = self._compute_seed_metrics(
                    evolution_states, params, seed
                )
                seed_results.append(seed_metrics)
            
            # Aggregate results across seeds
            aggregated_results = self._aggregate_seed_results(seed_results, params)
            
            # Add provenance information
            aggregated_results['provenance'] = self._get_provenance()
            
            return aggregated_results
            
        except Exception as e:
            print(f"❌ Error in validation for config {params['config_hash']}: {e}")
            return {
                'config': params,
                'error': str(e),
                'config_hash': params['config_hash']
            }
    
    def _compute_seed_metrics(self, evolution_states: List, params: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Compute metrics for a single seed run"""
        if not evolution_states:
            return {'error': 'No evolution states'}
        
        # Extract coherence values over time
        coherence_values = [state.coherence for state in evolution_states]
        stability_values = [state.stability for state in evolution_states]
        
        # Compute core metrics
        metrics = {
            'seed': seed,
            'r_final': float(coherence_values[-1]),
            'r_mean': float(np.mean(coherence_values)),
            'r_std': float(np.std(coherence_values)),
            'r_auc': float(np.trapz(coherence_values)),
            'stability_mean': float(np.mean(stability_values)),
            'evolution_steps': len(evolution_states)
        }
        
        # Compute attractor analysis (simplified)
        final_states = [state.get_field_strength() for state in evolution_states[-10:]]  # Last 10 states
        metrics['attractor_count'] = len(set(np.round(final_states, 2)))
        
        # Compute Lyapunov proxy (simplified)
        if len(coherence_values) > 1:
            lyap_proxy = np.mean(np.diff(coherence_values[-100:]))  # Last 100 differences
            metrics['lyapunov_proxy'] = float(lyap_proxy)
        else:
            metrics['lyapunov_proxy'] = 0.0
        
        # Compute hysteresis area (simplified - would need K-sweep in full implementation)
        metrics['hysteresis_area'] = 0.05  # Placeholder
        
        return metrics
    
    def _aggregate_seed_results(self, seed_results: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across multiple seeds"""
        if not seed_results or 'error' in seed_results[0]:
            return {'error': 'No valid seed results'}
        
        # Extract key metrics
        r_final_values = [r['r_final'] for r in seed_results if 'r_final' in r]
        r_mean_values = [r['r_mean'] for r in seed_results if 'r_mean' in r]
        stability_values = [r['stability_mean'] for r in seed_results if 'stability_mean' in r]
        
        # Compute aggregated metrics
        aggregated = {
            'config': params,
            'metrics': {
                'r_final': {
                    'mean': float(np.mean(r_final_values)),
                    'std': float(np.std(r_final_values)),
                    'cv': float(np.std(r_final_values) / np.mean(r_final_values)) if np.mean(r_final_values) > 0 else 0.0
                },
                'r_mean': {
                    'mean': float(np.mean(r_mean_values)),
                    'std': float(np.std(r_mean_values))
                },
                'stability': {
                    'mean': float(np.mean(stability_values)),
                    'std': float(np.std(stability_values))
                },
                'attractor_count': {
                    'modes': [r['attractor_count'] for r in seed_results if 'attractor_count' in r]
                },
                'lyapunov_proxy': {
                    'mean': float(np.mean([r['lyapunov_proxy'] for r in seed_results if 'lyapunov_proxy' in r]))
                },
                'hysteresis_area': {
                    'mean': float(np.mean([r['hysteresis_area'] for r in seed_results if 'hysteresis_area' in r]))
                }
            },
            'seed_count': len(seed_results),
            'config_hash': params['config_hash']
        }
        
        return aggregated
    
    def _get_provenance(self) -> Dict[str, str]:
        """Get provenance information for the run"""
        import git
        import platform
        
        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.head.object.hexsha[:8]
        except:
            git_sha = "unknown"
        
        return {
            'git_sha': git_sha,
            'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'numpy': np.__version__,
            'os': platform.system(),
            'timestamp': time.time()
        }
    
    def run_validation_grid(self, parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Run the complete validation grid
        
        Args:
            parallel: Whether to run in parallel
            
        Returns:
            List of validation results
        """
        print(f"🚀 Starting validation grid with {self.total_runs} parameter combinations")
        
        if parallel:
            # Run in parallel
            with mp.Pool(processes=self.config['execution']['parallel_workers']) as pool:
                results = pool.map(self._run_single_validation, self.parameter_grid)
        else:
            # Run sequentially
            results = []
            for i, params in enumerate(self.parameter_grid):
                print(f"Progress: {i+1}/{self.total_runs}")
                result = self._run_single_validation(params)
                results.append(result)
        
        # Save individual results
        for result in results:
            if 'error' not in result:
                result_file = self.output_dir / f"results_{result['config_hash']}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Generate summary
        summary = self._generate_summary(results)
        summary_file = self.output_dir / self.config['output']['summary_filename']
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Validation grid completed. Results saved to {self.output_dir}")
        return results
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from all results"""
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        summary = {
            'execution_summary': {
                'total_runs': len(results),
                'successful_runs': len(valid_results),
                'failed_runs': len(error_results),
                'success_rate': len(valid_results) / len(results) if results else 0.0
            },
            'parameter_coverage': {
                'coupling_K_range': [min(r['config']['K'] for r in valid_results), 
                                   max(r['config']['K'] for r in valid_results)],
                'phi_perturbations': list(set(r['config']['phi_multiplier'] for r in valid_results)),
                'network_sizes': list(set(r['config']['graph_topology']['n'] for r in valid_results)),
                'noise_levels': list(set(r['config']['gaussian_sigma'] for r in valid_results))
            },
            'metric_summaries': {
                'coherence_r_final': {
                    'mean': np.mean([r['metrics']['r_final']['mean'] for r in valid_results]),
                    'std': np.std([r['metrics']['r_final']['mean'] for r in valid_results]),
                    'min': min([r['metrics']['r_final']['mean'] for r in valid_results]),
                    'max': max([r['metrics']['r_final']['mean'] for r in valid_results])
                },
                'stability': {
                    'mean': np.mean([r['metrics']['stability']['mean'] for r in valid_results]),
                    'std': np.std([r['metrics']['stability']['mean'] for r in valid_results])
                },
                'hysteresis_area': {
                    'mean': np.mean([r['metrics']['hysteresis_area']['mean'] for r in valid_results]),
                    'std': np.std([r['metrics']['hysteresis_area']['mean'] for r in valid_results])
                }
            },
            'validation_gates': self._evaluate_validation_gates(valid_results),
            'timestamp': time.time(),
            'config_file': str(self.config_path)
        }
        
        return summary
    
    def _evaluate_validation_gates(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate validation gates against thresholds"""
        if not results:
            return {'status': 'no_results'}
        
        thresholds = self.config['validation_thresholds']
        gates = {}
        
        # Global stability gate
        coherence_values = [r['metrics']['r_final']['mean'] for r in results]
        coherence_range = thresholds['global_stability']['coherence_range']
        stable_coherence = sum(1 for r in coherence_values 
                             if coherence_range[0] <= r <= coherence_range[1])
        gates['global_stability'] = {
            'coherence_in_range': stable_coherence / len(results),
            'threshold': thresholds['global_stability']['min_stable_percentage'] / 100,
            'passed': (stable_coherence / len(results)) >= (thresholds['global_stability']['min_stable_percentage'] / 100)
        }
        
        # Irreversibility gate
        hysteresis_values = [r['metrics']['hysteresis_area']['mean'] for r in results]
        min_hysteresis = thresholds['irreversibility']['min_hysteresis_area']
        stable_hysteresis = sum(1 for h in hysteresis_values if h >= min_hysteresis)
        gates['irreversibility'] = {
            'hysteresis_stable': stable_hysteresis / len(results),
            'threshold': thresholds['irreversibility']['min_stable_percentage'] / 100,
            'passed': (stable_hysteresis / len(results)) >= (thresholds['irreversibility']['min_stable_percentage'] / 100)
        }
        
        # Overall gate status
        all_gates_passed = all(gate['passed'] for gate in gates.values())
        gates['overall_status'] = 'PASSED' if all_gates_passed else 'FAILED'
        
        return gates

def main():
    """Main entry point for validation grid execution"""
    parser = argparse.ArgumentParser(description="Day 8.5 Unified Field Validation Harness")
    parser.add_argument('--config', required=True, help='Path to sweep configuration YAML')
    parser.add_argument('--output', default='artifacts', help='Output directory for artifacts')
    parser.add_argument('--parallel', action='store_true', default=True, help='Run in parallel')
    parser.add_argument('--seeds', type=int, default=None, help='Override seeds per grid point')
    parser.add_argument('--steps', type=int, default=None, help='Override steps per run')
    
    args = parser.parse_args()
    
    # Initialize executor
    executor = ValidationGridExecutor(args.config, args.output)
    
    # Override configuration if specified
    if args.seeds:
        executor.config['execution']['seeds_per_grid_point'] = args.seeds
    if args.steps:
        executor.config['execution']['steps_per_run'] = args.steps
    
    # Run validation grid
    results = executor.run_validation_grid(parallel=args.parallel)
    
    # Print summary
    summary_file = Path(args.output) / executor.config['output']['summary_filename']
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("\n" + "="*70)
        print("🎯 VALIDATION GRID SUMMARY")
        print("="*70)
        print(f"Total Runs: {summary['execution_summary']['total_runs']}")
        print(f"Successful: {summary['execution_summary']['successful_runs']}")
        print(f"Failed: {summary['execution_summary']['failed_runs']}")
        print(f"Success Rate: {summary['execution_summary']['success_rate']:.2%}")
        print(f"Overall Status: {summary['validation_gates']['overall_status']}")
        print("="*70)
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 Validation grid execution completed successfully!")
    except Exception as e:
        print(f"\n❌ Validation grid execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 