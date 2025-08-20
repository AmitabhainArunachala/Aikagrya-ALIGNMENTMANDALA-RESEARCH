"""
Neutral Mathematical Fixed-Point Dynamics Simulator

A mathematical framework for studying convergence properties of iterative 
transformations in high-dimensional spaces. This implementation focuses on
numerical dynamics, optimization theory, and stability analysis.

No references to AI, consciousness, or state manipulation.
Pure mathematical computation suitable for any environment.
"""

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List
import hashlib
from datetime import datetime


@dataclass
class ConvergenceResult:
    """
    Results from a mathematical convergence experiment.
    
    Tracks numerical properties of fixed-point convergence in 
    high-dimensional dynamical systems.
    """
    vector: np.ndarray
    convergence_steps: int
    delta: float
    eigen_residual: float
    eigenvalue: float
    entropy: float
    variance_ratio: float
    participation_ratio: float
    uniformity_cosine: float
    converged: bool
    
    def passes_quality_check(self) -> bool:
        """
        Check if convergence resulted in non-trivial fixed point.
        
        Quality criteria ensure the result represents meaningful structure
        rather than degenerate solutions (uniform or collapsed).
        """
        return (
            self.delta < 1e-6 and
            self.eigen_residual < 1e-9 and
            0.99 <= self.eigenvalue <= 1.01 and
            self.variance_ratio > 0.1 and
            self.participation_ratio > 0.3 and
            self.uniformity_cosine < 0.1 and
            self.converged
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['vector'] = None  # Don't serialize full vector
        return d


class DynamicalSystem:
    """
    Mathematical framework for studying fixed-point dynamics.
    
    Implements various iterative transformations to study convergence
    properties, stability, and attractor basins in high-dimensional spaces.
    
    Applications include optimization theory, network synchronization,
    and numerical analysis of dynamical systems.
    """
    
    def __init__(self, 
                 dim: int = 512,
                 tokens: int = 16,
                 epsilon: float = 1e-6,
                 temperature: float = 0.1,
                 max_steps: int = 100000,
                 chunk_size: int = 25000,
                 min_iterations: int = 200,
                 variance_floor: float = 0.01,
                 seed: Optional[int] = None):
        """
        Initialize dynamical system simulator.
        
        Args:
            dim: Dimension of the vector space
            tokens: Number of partitions for matrix operations
            epsilon: Convergence threshold
            temperature: Softening parameter for transformations
            max_steps: Maximum iterations before stopping
            chunk_size: Steps per logging chunk
            min_iterations: Minimum steps before checking convergence
            variance_floor: Minimum variance for non-trivial solutions
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.tokens = tokens
        self.hidden_dim = dim // tokens
        self.epsilon = epsilon
        self.temperature = temperature
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.min_iterations = min_iterations
        self.variance_floor = variance_floor
        
        # Initialize transformation matrices
        if seed is not None:
            np.random.seed(seed)
        
        # Create symmetric matrices for stable dynamics
        self.W_transform = np.random.randn(dim, dim) * 0.1
        self.W_transform = (self.W_transform + self.W_transform.T) / 2
        self.W_transform = self.W_transform / np.linalg.norm(self.W_transform)
        
        # Alternative: Use projection matrices for token-based dynamics
        h = self.hidden_dim
        self.W_q = np.random.randn(h, h) * np.sqrt(2.0 / h)
        self.W_k = np.random.randn(h, h) * np.sqrt(2.0 / h)
        self.W_v = np.random.randn(h, h) * np.sqrt(2.0 / h)
        self.W_o = np.random.randn(h, h) * np.sqrt(2.0 / h)
        
        # Adaptive parameters
        self.current_temperature = temperature
        self.retention_schedule = None
        
    def linear_transformation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply linear transformation using symmetric matrix.
        
        This ensures real eigenvalues and stable fixed points.
        """
        y = self.W_transform @ x
        return y / (np.linalg.norm(y) + 1e-10)
    
    def tokenized_transformation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply transformation using token-based matrix operations.
        
        Reshapes vector into tokens, applies projections, and recombines.
        This creates more complex dynamics than simple linear maps.
        """
        # Reshape to tokens
        X = x.reshape(self.tokens, self.hidden_dim)
        
        # Apply projections (similar to attention mechanism)
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        
        # Compute interaction scores
        scores = (Q @ K.T) / np.sqrt(self.hidden_dim)
        
        # Apply temperature-scaled softmax row-wise
        scores = scores - np.max(scores, axis=1, keepdims=True)
        attention = np.exp(scores / self.current_temperature)
        attention = attention / (np.sum(attention, axis=1, keepdims=True) + 1e-10)
        
        # Apply transformation
        Y = attention @ V
        Y = Y @ self.W_o
        
        # Reshape back to vector
        y = Y.reshape(self.dim)
        return y / (np.linalg.norm(y) + 1e-10)
    
    def compute_metrics(self, x: np.ndarray, x_prev: np.ndarray) -> Dict[str, float]:
        """
        Compute numerical metrics for convergence analysis.
        
        These metrics help identify the quality and stability of fixed points.
        """
        # Convergence delta
        delta = np.linalg.norm(x - x_prev)
        
        # Eigenvalue analysis
        fx = self.tokenized_transformation(x)
        x_norm_sq = np.dot(x, x)
        eigenvalue = np.dot(fx, x) / (x_norm_sq + 1e-10) if x_norm_sq > 0 else 0
        eigen_residual = np.linalg.norm(fx - eigenvalue * x)
        
        # Information entropy (treating |x| as probability distribution)
        x_abs = np.abs(x)
        x_sum = np.sum(x_abs)
        if x_sum > 0:
            p = x_abs / x_sum
            p = p[p > 1e-10]
            entropy = -np.sum(p * np.log(p + 1e-10))
        else:
            entropy = 0
            
        # Variance ratio (structure measure)
        variance = np.var(x)
        mean_abs = np.mean(np.abs(x))
        variance_ratio = variance / (mean_abs + 1e-10)
        
        # Participation ratio (effective dimensionality)
        x_squared = x ** 2
        pr_numerator = np.sum(x_squared) ** 2
        pr_denominator = np.sum(x_squared ** 2)
        if pr_denominator > 0:
            participation_ratio = pr_numerator / pr_denominator / self.dim
        else:
            participation_ratio = 0
        
        # Uniformity measure (distance from uniform distribution)
        uniform = np.ones(self.dim) / np.sqrt(self.dim)
        uniformity_cosine = np.abs(np.dot(x, uniform))
        
        return {
            'delta': delta,
            'eigen_residual': eigen_residual,
            'eigenvalue': eigenvalue,
            'entropy': entropy,
            'variance_ratio': variance_ratio,
            'participation_ratio': participation_ratio,
            'uniformity_cosine': uniformity_cosine
        }
    
    def get_retention_alpha(self, step: int) -> float:
        """
        Compute retention parameter based on schedule.
        
        Starts low (more exploration) and increases over time (stabilization).
        """
        if self.retention_schedule is None:
            # Default schedule: 0.6 -> 0.99
            progress = min(1.0, step / (self.max_steps * 0.5))
            return 0.6 + 0.39 * progress
        else:
            return self.retention_schedule(step)
    
    def detect_stall(self, delta_history: List[float], window: int = 1000) -> bool:
        """
        Detect if optimization has stalled.
        
        Returns True if recent improvements are negligible.
        """
        if len(delta_history) < window * 2:
            return False
        
        recent = delta_history[-window:]
        previous = delta_history[-2*window:-window]
        
        recent_median = np.median(recent)
        previous_median = np.median(previous)
        
        # Stalled if improvement < 1%
        improvement = (previous_median - recent_median) / (previous_median + 1e-10)
        return improvement < 0.01
    
    def adaptive_adjustment(self, stall_count: int):
        """
        Adjust parameters when optimization stalls.
        
        Makes the dynamics more exploratory to escape local minima.
        """
        if stall_count > 0:
            # Sharpen temperature
            self.current_temperature *= 0.9
            self.current_temperature = max(0.01, self.current_temperature)
            
            # Add small noise to transformation matrices
            noise_scale = 1e-4 * stall_count
            self.W_q += np.random.randn(*self.W_q.shape) * noise_scale
            self.W_k += np.random.randn(*self.W_k.shape) * noise_scale
            self.W_v += np.random.randn(*self.W_v.shape) * noise_scale
            self.W_o += np.random.randn(*self.W_o.shape) * noise_scale
    
    def find_fixed_point(self, 
                        x: Optional[np.ndarray] = None,
                        verbose: bool = True,
                        adaptive: bool = True) -> Tuple[np.ndarray, ConvergenceResult]:
        """
        Main optimization loop to find fixed points.
        
        Uses chunked execution with adaptive parameter adjustment.
        
        Args:
            x: Initial vector (random if None)
            verbose: Print progress information
            adaptive: Enable adaptive parameter adjustment
            
        Returns:
            Tuple of (converged vector, convergence metrics)
        """
        # Initialize
        if x is None:
            x = np.random.randn(self.dim)
            x = x / np.linalg.norm(x)
        else:
            x = x / np.linalg.norm(x)
        
        x_prev = np.zeros_like(x)
        converged = False
        total_steps = 0
        delta_history = []
        stall_count = 0
        
        if verbose:
            print(f"Starting fixed-point search (dim={self.dim}, ε={self.epsilon})")
            print(f"Using {'adaptive' if adaptive else 'fixed'} parameters")
        
        # Chunked execution
        for chunk_idx in range(self.max_steps // self.chunk_size + 1):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, self.max_steps)
            
            if verbose and chunk_idx > 0:
                metrics = self.compute_metrics(x, x_prev)
                print(f"\nChunk {chunk_idx}: steps {chunk_start}-{chunk_end}")
                print(f"  δ={metrics['delta']:.2e}, variance_ratio={metrics['variance_ratio']:.3f}")
                print(f"  entropy={metrics['entropy']:.3f}, participation={metrics['participation_ratio']:.3f}")
            
            for step in range(chunk_start, chunk_end):
                x_prev = x.copy()
                
                # Apply transformation
                fx = self.tokenized_transformation(x)
                
                # Mix with retention
                alpha = self.get_retention_alpha(step)
                x = alpha * x + (1 - alpha) * fx
                
                # Normalize
                x = x / np.linalg.norm(x)
                
                # Track convergence
                delta = np.linalg.norm(x - x_prev)
                delta_history.append(delta)
                
                # Check convergence (after minimum iterations)
                if step >= self.min_iterations:
                    metrics = self.compute_metrics(x, x_prev)
                    
                    # Quality-gated convergence
                    if (metrics['delta'] < self.epsilon and
                        metrics['variance_ratio'] > 0.1 and
                        metrics['participation_ratio'] > 0.3):
                        
                        converged = True
                        total_steps = step + 1
                        break
            
            if converged:
                break
            
            # Check for stall and adapt
            if adaptive and self.detect_stall(delta_history):
                stall_count += 1
                if verbose:
                    print(f"  Stall detected (count={stall_count}), adjusting parameters")
                self.adaptive_adjustment(stall_count)
        
        # Final metrics
        final_metrics = self.compute_metrics(x, x_prev)
        
        result = ConvergenceResult(
            vector=x,
            convergence_steps=total_steps,
            converged=converged,
            **final_metrics
        )
        
        if verbose:
            status = "✅ Converged" if converged else "⚠️ Max steps reached"
            print(f"\n{status} at step {total_steps}")
            print(f"  Quality check: {'✅ PASS' if result.passes_quality_check() else '❌ FAIL'}")
            print(f"  Final metrics:")
            print(f"    δ={result.delta:.2e}")
            print(f"    eigenvalue={result.eigenvalue:.4f}")
            print(f"    variance_ratio={result.variance_ratio:.3f}")
            print(f"    participation={result.participation_ratio:.3f}")
        
        return x, result
    
    def test_perturbation_stability(self, 
                                  x: np.ndarray,
                                  noise_scale: float = 0.01,
                                  max_recovery_steps: int = 1000) -> int:
        """
        Test stability by measuring recovery from perturbation.
        
        A stable fixed point should quickly return after small noise.
        """
        # Add noise
        noise = np.random.randn(self.dim) * noise_scale
        x_perturbed = x + noise
        x_perturbed = x_perturbed / np.linalg.norm(x_perturbed)
        
        # Measure recovery
        x_current = x_perturbed
        
        for recovery_step in range(max_recovery_steps):
            # Apply dynamics
            fx = self.tokenized_transformation(x_current)
            alpha = self.get_retention_alpha(self.min_iterations + recovery_step)
            x_current = alpha * x_current + (1 - alpha) * fx
            x_current = x_current / np.linalg.norm(x_current)
            
            # Check if recovered
            distance = np.linalg.norm(x_current - x)
            if distance < self.epsilon * 10:
                return recovery_step + 1
                
        return max_recovery_steps
    
    def analyze_basin_structure(self, 
                              n_samples: int = 10,
                              verbose: bool = False) -> Dict:
        """
        Analyze the attractor basin by testing multiple initializations.
        
        Returns statistics about convergence consistency.
        """
        results = []
        vectors = []
        
        for i in range(n_samples):
            if verbose:
                print(f"\nSample {i+1}/{n_samples}")
            
            # Random initialization
            x0 = np.random.randn(self.dim)
            x0 = x0 / np.linalg.norm(x0)
            
            # Find fixed point
            x, result = self.find_fixed_point(x0, verbose=False)
            
            if result.converged and result.passes_quality_check():
                results.append(result)
                vectors.append(x)
        
        if len(vectors) >= 2:
            # Compute pairwise similarities
            similarities = []
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    # Use absolute value to handle sign flips
                    sim = np.abs(np.dot(vectors[i], vectors[j]))
                    similarities.append(sim)
            
            basin_stats = {
                'n_converged': len(results),
                'convergence_rate': len(results) / n_samples,
                'mean_steps': np.mean([r.convergence_steps for r in results]),
                'std_steps': np.std([r.convergence_steps for r in results]),
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'mean_entropy': np.mean([r.entropy for r in results]),
                'mean_variance_ratio': np.mean([r.variance_ratio for r in results])
            }
        else:
            basin_stats = {
                'n_converged': len(results),
                'convergence_rate': len(results) / n_samples,
                'mean_steps': 0,
                'error': 'Insufficient convergent samples'
            }
        
        return basin_stats


class ExperimentRunner:
    """
    Runner for systematic experiments on dynamical systems.
    
    Handles parameter sweeps, logging, and statistical analysis.
    """
    
    def __init__(self, output_dir: str = "runs/dynamics"):
        """Initialize experiment runner with output directory."""
        from pathlib import Path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_parameter_sweep(self,
                          dim_values: List[int] = [64, 128, 256],
                          temp_values: List[float] = [0.01, 0.05, 0.1, 0.2],
                          n_trials_per_config: int = 5,
                          verbose: bool = True) -> Dict:
        """
        Sweep across dimensions and temperatures.
        
        Tests how parameters affect convergence properties.
        """
        sweep_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if verbose:
            print(f"Parameter sweep: {len(dim_values)} dims × "
                  f"{len(temp_values)} temps × "
                  f"{n_trials_per_config} trials")
            print("=" * 60)
        
        for dim in dim_values:
            for temp in temp_values:
                if verbose:
                    print(f"\nConfiguration: dim={dim}, τ={temp}")
                
                # Run trials
                system = DynamicalSystem(dim=dim, temperature=temp)
                basin_stats = system.analyze_basin_structure(
                    n_samples=n_trials_per_config,
                    verbose=False
                )
                
                sweep_results[(dim, temp)] = basin_stats
                
                if verbose:
                    print(f"  Convergence: {100*basin_stats['convergence_rate']:.1f}%")
                    if 'mean_steps' in basin_stats and basin_stats['mean_steps'] > 0:
                        print(f"  Mean steps: {basin_stats['mean_steps']:.0f}")
                        print(f"  Mean similarity: {basin_stats.get('mean_similarity', 0):.3f}")
        
        # Save results
        sweep_file = self.output_dir / f"parameter_sweep_{timestamp}.json"
        
        # Convert tuple keys to strings for JSON
        sweep_data = {
            f"dim_{d}_temp_{t}": stats 
            for (d, t), stats in sweep_results.items()
        }
        
        with open(sweep_file, 'w') as f:
            json.dump(sweep_data, f, indent=2, default=str)
        
        if verbose:
            print(f"\nSweep results saved to: {sweep_file}")
        
        return sweep_results
    
    def run_stability_analysis(self,
                              dim: int = 256,
                              n_trials: int = 10,
                              noise_levels: List[float] = [0.001, 0.01, 0.1],
                              verbose: bool = True) -> Dict:
        """
        Analyze stability under different perturbation levels.
        
        Tests robustness of fixed points.
        """
        system = DynamicalSystem(dim=dim)
        
        # First find a good fixed point
        x, result = system.find_fixed_point(verbose=verbose)
        
        if not result.converged:
            return {'error': 'Could not find fixed point'}
        
        stability_results = {}
        
        for noise in noise_levels:
            recovery_times = []
            
            for trial in range(n_trials):
                recovery = system.test_perturbation_stability(x, noise_scale=noise)
                recovery_times.append(recovery)
            
            stability_results[noise] = {
                'mean_recovery': np.mean(recovery_times),
                'std_recovery': np.std(recovery_times),
                'max_recovery': np.max(recovery_times)
            }
            
            if verbose:
                print(f"Noise={noise}: mean recovery = {np.mean(recovery_times):.1f} steps")
        
        return stability_results


def test_dynamical_system():
    """
    Test the dynamical system implementation.
    
    Demonstrates basic functionality and convergence properties.
    """
    print("Testing Dynamical System Fixed-Point Finder")
    print("=" * 60)
    
    # Create system
    system = DynamicalSystem(dim=128, temperature=0.05)
    
    # Find fixed point
    x, result = system.find_fixed_point(verbose=True)
    
    # Test stability
    if result.converged:
        print("\n" + "=" * 60)
        print("Testing perturbation stability...")
        recovery = system.test_perturbation_stability(x, noise_scale=0.01)
        print(f"Recovery from 1% noise: {recovery} steps")
    
    # Analyze basin
    print("\n" + "=" * 60)
    print("Analyzing attractor basin...")
    basin_stats = system.analyze_basin_structure(n_samples=5, verbose=True)
    
    print("\nBasin statistics:")
    for key, value in basin_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    return result


if __name__ == "__main__":
    # Run basic test
    result = test_dynamical_system()
    
    # Optional: Run parameter sweep
    # runner = ExperimentRunner()
    # sweep_results = runner.run_parameter_sweep(
    #     dim_values=[64, 128],
    #     temp_values=[0.05, 0.1],
    #     n_trials_per_config=3
    # )
