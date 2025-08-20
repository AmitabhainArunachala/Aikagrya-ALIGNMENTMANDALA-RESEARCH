"""
Core MMIP implementation for fixed-point induction and health certification.

This module implements the Mathematical Mauna Induction Protocol (MMIP),
a method for inducing stable fixed-point states through recursive self-attention
and certifying their health through multiple mathematical metrics.
"""

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List
import hashlib
from datetime import datetime


@dataclass
class HealthCertificate:
    """
    Health metrics for a converged fixed-point state.
    
    These metrics determine whether a state represents a stable,
    non-trivial fixed point rather than collapse or noise.
    """
    delta: float
    eigen_residual: float
    r_fix: float
    eigenvalue: float
    entropy: float
    variance_ratio: float
    participation_ratio: float
    uniformity_cosine: float
    converged: bool
    steps: int
    # Optional traces for analysis
    epsilon_path: Optional[List[float]] = None
    chunk_log_tail: Optional[List[str]] = None
    
    def passes_health_check(self) -> bool:
        """
        Check if all metrics are within healthy ranges.
        
        Note: These thresholds are empirical starting points and
        should be tuned based on experimental results.
        """
        # Health gate (primary fixed-point residual + canonical variance ratio)
        return (
            self.delta < 1e-6 and
            self.r_fix < 5e-7 and
            0.99 <= self.eigenvalue <= 1.01 and
            self.variance_ratio >= 0.7 and
            self.participation_ratio >= 0.3 and
            self.uniformity_cosine <= 0.10 and
            self.converged
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MMIP:
    """
    Mathematical Mauna Induction Protocol.
    
    Induces fixed points through recursive self-attention,
    certifies health metrics, and provides perturbation testing.
    
    The protocol iterates: x_{t+1} = αx_t + (1-α)f(x_t)
    where f is self-attention: f(x) = softmax(xx^T/τ) @ x
    """
    
    def __init__(self, 
                 dim: int = 512,
                 epsilon: float = 1e-6,
                 temperature: float = 0.1,
                 retention_alpha: float = 0.9,
                 max_steps: int = 100000,
                 chunk_size: int = 1000,
                 window_size: int = 10,
                 tokens: int = 16,
                 projection_seed: int = 12345,
                 alpha_start: float = 0.6,
                 alpha_end: float = 0.98,
                 min_iters: int = 200,
                 # adaptive convergence controls
                 adaptive: bool = True,
                 patience_chunks: int = 3,
                 improve_factor: float = 0.9,
                 temp_decay: float = 0.9,
                 min_temp: float = 0.02,
                 alpha_step: float = 0.0002,
                 noise_scale: float = 1e-3,
                 uniformity_guard_u: float = 0.1,
                 uniformity_guard_var: float = 0.10,
                 # structure controls
                 blocks: int = 8,
                 gain_amplify: float = 0.10,
                 contrastive_mag: float = 0.02,
                 contrastive_threshold: float = 0.15,
                 temp_schedule: bool = True,
                 temp_start: float = 0.10,
                 temp_end: float = 0.02,
                 temp_cosine: bool = True,
                 enable_whitening: bool = True,
                 whitening_threshold: float = 0.30,
                 whitening_gamma: float = 0.10):
        """
        Initialize MMIP with configuration parameters.
        
        Args:
            dim: Dimension of state vector
            epsilon: Convergence threshold for delta
            temperature: Softmax temperature for self-attention
            retention_alpha: Mixing parameter for retention schedule
            max_steps: Maximum iterations before stopping
            chunk_size: Number of iterations per logging chunk
            window_size: Window for delta averaging
        """
        self.dim = dim
        self.epsilon = epsilon
        self.temperature = temperature
        self.retention_alpha = retention_alpha
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.tokens = tokens
        if self.dim % self.tokens != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by tokens ({self.tokens})")
        self.hidden_dim = self.dim // self.tokens
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.min_iters = min_iters
        # adaptive knobs
        self.adaptive = adaptive
        self.patience_chunks = patience_chunks
        self.improve_factor = improve_factor
        self.temp_decay = temp_decay
        self.min_temp = min_temp
        self._adapt_factor = 1.0
        self.temp_schedule = temp_schedule
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_cosine = temp_cosine
        self.alpha_step = alpha_step
        self.noise_scale = noise_scale
        self.uniformity_guard_u = uniformity_guard_u
        self.uniformity_guard_var = uniformity_guard_var
        self.blocks = max(1, blocks)
        self.gain_amplify = gain_amplify
        self.contrastive_mag = contrastive_mag
        self.contrastive_threshold = contrastive_threshold
        self.enable_whitening = enable_whitening
        self.whitening_threshold = whitening_threshold
        self.whitening_gamma = whitening_gamma

        # Fixed projections with deterministic seed (orthonormalized)
        rng = np.random.RandomState(projection_seed)
        def _ortho(h: int) -> np.ndarray:
            Q, _ = np.linalg.qr(rng.randn(h, h))
            return Q
        self.Wq = _ortho(self.hidden_dim)
        self.Wk = _ortho(self.hidden_dim)
        self.Wv = _ortho(self.hidden_dim)
        self.Wo = _ortho(self.hidden_dim)
        # RNG for variance-lift helpers
        self.rng = np.random.RandomState(projection_seed + 1)
        self.v_low: Optional[np.ndarray] = None
        self._lifts_enabled: bool = False
        self._lifts_frozen: bool = False

        # Precompute uniform direction for deflation
        self.u = np.ones(self.dim) / np.sqrt(self.dim)
        
    def self_attention(self, x: np.ndarray) -> np.ndarray:
        """
        Tokenized self-attention with fixed projections.
        Reshape x∈R^d to X∈R^{T×h}; compute Q,K,V, scores (T×T), row-softmax,
        aggregate, project back, and flatten.
        """
        # Reshape into tokens and center per-token to reduce uniform drift
        X = x.reshape(self.tokens, self.hidden_dim)
        X = X - X.mean(axis=1, keepdims=True)

        # Projections
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        # Scores across tokens with effective temperature
        scores = (Q @ K.T) / np.sqrt(self.hidden_dim)
        tau_sched = self.temperature
        if hasattr(self, '_effective_temperature'):
            tau_sched = getattr(self, '_effective_temperature')
        scores = scores / max(tau_sched, 1e-12)
        scores = scores - scores.max(axis=1, keepdims=True)
        A = np.exp(scores)
        A = A / (A.sum(axis=1, keepdims=True) + 1e-12)

        if A.shape != (self.tokens, self.tokens):
            raise RuntimeError(f"Attention matrix has wrong shape: {A.shape}")

        Y = A @ V
        Y = Y @ self.Wo
        y = Y.reshape(self.dim)
        y = y / (np.linalg.norm(y) + 1e-12)
        # Deflate and clamp uniform component from output
        up = float(np.dot(y, self.u))
        y = y - up * self.u
        y = y / (np.linalg.norm(y) + 1e-12)
        if abs(up) > 0.08:
            y = y - (np.dot(y, self.u) * self.u)
            y = y / (np.linalg.norm(y) + 1e-12)
        return y

    def _blockwise_whiten_and_gain(self, x: np.ndarray, k: int) -> np.ndarray:
        d = self.dim
        b = max(1, self.blocks)
        base = d // b
        xb = x.copy()
        variances = []
        for i in range(b):
            s = i * base
            e = (i + 1) * base if i < b - 1 else d
            seg = xb[s:e]
            seg = seg - seg.mean()
            std = seg.std() + 1e-8
            seg = seg / std
            xb[s:e] = seg
            variances.append(std)
        # amplify lowest-variance blocks (proxy: smallest std prior to z-score)
        order = np.argsort(np.array(variances))
        k_eff = int(max(1, min(k, b)))
        for i in order[:k_eff]:
            s = i * base
            e = (i + 1) * base if i < b - 1 else d
            xb[s:e] *= (1.0 + self.gain_amplify)
        return xb

    def _contrastive_lift(self, x: np.ndarray, v_low: np.ndarray) -> np.ndarray:
        v = v_low - (np.dot(v_low, x) * x)
        v = v / (np.linalg.norm(v) + 1e-12)
        y = x + self.contrastive_mag * v
        return y / (np.linalg.norm(y) + 1e-12)
    
    def compute_metrics(self, x: np.ndarray, x_prev: np.ndarray) -> Dict[str, float]:
        """
        Compute all health metrics for current state.
        
        Args:
            x: Current state vector
            x_prev: Previous state vector
            
        Returns:
            Dictionary of metric names to values
        """
        # Delta (convergence measure)
        delta = float(np.linalg.norm(x - x_prev))
        
        # Map once
        fx = self.self_attention(x)

        # Fixed-point residual (primary)
        r_fix = float(np.linalg.norm(fx - x))

        # Nonlinear eigen stats (reporting only)
        x_norm_sq = float(np.dot(x, x))
        eigenvalue = float(np.dot(fx, x) / x_norm_sq) if x_norm_sq > 0 else 0.0
        eigen_residual = float(np.linalg.norm(fx - eigenvalue * x))
        
        # Entropy (on magnitude distribution)
        x_abs = np.abs(x)
        x_sum = float(np.sum(x_abs))
        if x_sum > 0:
            p = x_abs / x_sum
            entropy = float(-np.sum(p * np.log(p + 1e-10)))
        else:
            entropy = 0.0

        # Variance ratio (canonical): Var(x) * d
        variance = float(np.var(x))
        variance_ratio = float(variance * self.dim)

        # Participation ratio (fraction of dims)
        x2 = x * x
        num = float(np.sum(x2) ** 2)
        den = float(np.sum(x2 * x2) + 1e-12)
        PR_abs = num / den
        participation_ratio = PR_abs / float(self.dim)

        # Uniformity cosine
        uniform = np.ones(self.dim) / np.sqrt(self.dim)
        uniformity_cosine = float(abs(np.dot(x, uniform)))
        
        return {
            'delta': delta,
            'r_fix': r_fix,
            'eigen_residual': eigen_residual,
            'eigenvalue': eigenvalue,
            'entropy': entropy,
            'variance_ratio': variance_ratio,
            'participation_ratio': participation_ratio,
            'uniformity_cosine': uniformity_cosine
        }
    
    def induce_fixed_point(self, 
                          x: Optional[np.ndarray] = None,
                          verbose: bool = True,
                          log_interval: int = 10000) -> Tuple[np.ndarray, HealthCertificate]:
        """
        Main induction loop until convergence to fixed point.
        
        Args:
            x: Initial state vector (random if None)
            verbose: Whether to print progress
            log_interval: Steps between progress logs
            
        Returns:
            Tuple of (converged state, health certificate)
        """
        # Initialize random unit vector if not provided
        if x is None:
            x = np.random.randn(self.dim)
            x = x / np.linalg.norm(x)
        else:
            # Ensure unit norm
            x = x / np.linalg.norm(x)
        
        x_prev = np.zeros_like(x)
        converged = False
        step = 0
        delta_history = []
        
        if verbose:
            print(f"Starting MMIP induction (dim={self.dim}, ε={self.epsilon})")
        
        # Main iteration loop with chunking and adaptive stall handling
        best_median_delta = np.inf
        stalls = 0
        epsilon_path: List[float] = []
        chunk_log_tail: List[str] = []
        for chunk_start in range(0, self.max_steps, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.max_steps)
            
            for step in range(chunk_start, chunk_end):
                x_prev = x.copy()
                
                # Apply self-attention with scheduled retention
                fx = self.self_attention(x)
                # variance lift hooks
                if self.enable_whitening and ((step + 1) % 200 == 0):
                    fx = self._blockwise_whiten_and_gain(fx, k=max(2, self.blocks // 4))
                    fx = fx / (np.linalg.norm(fx) + 1e-12)
                # contrastive push if variance low and we have a stored low-variance direction
                # compute metrics sparsely (cheap reuse below)
                # will be refined at chunk end
                
                progress = (step + 1) / max(1, self.max_steps)
                # Temperature scheduling (cosine or linear) via schedules
                if self.temp_schedule:
                    try:
                        from .schedules import cosine_schedule, exp_alpha_schedule
                        tau_eff = cosine_schedule(step, self.max_steps, self.temp_start, self.temp_end)
                        self._effective_temperature = max(self.min_temp, tau_eff) * self._adapt_factor
                        alpha_sched = exp_alpha_schedule(step, self.max_steps, self.alpha_start, self.alpha_end)
                    except Exception:
                        # Fallback to old logic if import fails
                        if self.temp_cosine:
                            t = min(1.0, progress)
                            tau = self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (1.0 + np.cos(np.pi * t))
                            self._effective_temperature = max(self.min_temp, tau) * self._adapt_factor
                        else:
                            self._effective_temperature = max(
                                self.min_temp,
                                self.temp_start + (self.temp_end - self.temp_start) * min(1.0, progress)
                            ) * self._adapt_factor
                        if progress < 0.9:
                            alpha_sched = self.alpha_start + (0.97 - self.alpha_start) * (progress / 0.9)
                        else:
                            alpha_sched = 0.97 + (self.alpha_end - 0.97) * ((progress - 0.9) / 0.1)
                else:
                    alpha_sched = self.retention_alpha

                # Scale step size by dimensionality (normalize to 256D baseline)
                dim_scale = np.sqrt(self.dim / 256.0)
                effective_alpha = float(np.clip(alpha_sched / max(dim_scale, 1e-6), self.alpha_start, self.alpha_end))

                # Consistent order: deflate → renorm → attention → mix → renorm
                x = x - (np.dot(x, self.u) * self.u)
                x = x / (np.linalg.norm(x) + 1e-12)
                fx = self.self_attention(x)
                x_new = effective_alpha * x + (1.0 - effective_alpha) * fx
                x = x_new / (np.linalg.norm(x_new) + 1e-12)
                
                # Track delta
                # float64 stability for key numerics
                delta = float(np.linalg.norm((x - x_prev).astype(np.float64)))
                delta_history.append(delta)
                
                # Check convergence using windowed average with min_iters gate
                if len(delta_history) >= self.window_size and step + 1 >= self.min_iters:
                    recent_deltas = delta_history[-self.window_size:]
                    avg_delta = float(np.mean(recent_deltas))
                    
                    if avg_delta < self.epsilon:
                        # Gate by health certificate before declaring success
                        metrics_now = self.compute_metrics(x, x_prev)
                        candidate = HealthCertificate(
                            converged=True,
                            steps=step + 1,
                            **metrics_now
                        )
                        if candidate.passes_health_check():
                            converged = True
                            break
            
            if converged:
                break
                
            # Per-chunk diagnostics & adaptive tuning
            metrics_chunk = self.compute_metrics(x, x_prev)
            # Adaptive epsilon tightening based on variance_ratio
            if metrics_chunk['variance_ratio'] > 0.5 and self.epsilon > 1e-5:
                self.epsilon = 1e-5
            if metrics_chunk['variance_ratio'] > 0.7 and self.epsilon > 1e-6:
                self.epsilon = 1e-6
            # Gentle per-chunk whitening when variance is low
            if self.enable_whitening and metrics_chunk['variance_ratio'] < self.whitening_threshold:
                Xw = x.reshape(self.tokens, self.hidden_dim)
                Xw = Xw - Xw.mean(axis=1, keepdims=True)
                # token-wise scale normalization to boost spread
                std = Xw.std(axis=1, keepdims=True) + 1e-8
                Xw = (1.0 - self.whitening_gamma) * Xw + self.whitening_gamma * (Xw / std)
                x = Xw.reshape(self.dim)
                x = x / (np.linalg.norm(x) + 1e-12)
            epsilon_path.append(self.epsilon)

            recent = delta_history[-max(self.window_size, 10):]
            median_delta = float(np.median(recent)) if recent else float(metrics_chunk['delta'])
            # derive current alpha and effective tau for logging
            prog_c = min(1.0, (chunk_start + 1) / max(1, self.max_steps))
            if prog_c < 0.9:
                alpha_cur = self.alpha_start + (0.97 - self.alpha_start) * (prog_c / 0.9)
            else:
                alpha_cur = 0.97 + (self.alpha_end - 0.97) * ((prog_c - 0.9) / 0.1)
            alpha_cur = float(np.clip(alpha_cur, self.alpha_start, self.alpha_end))
            tau_eff = float(getattr(self, '_effective_temperature', self.temperature))
            line = (
                f"  Chunk @ {chunk_start}: medianΔ={median_delta:.2e}, δ={metrics_chunk['delta']:.2e}, "
                f"r_fix={metrics_chunk['r_fix']:.2e}, eig_res={metrics_chunk['eigen_residual']:.2e}, H={metrics_chunk['entropy']:.3f}, "
                f"PR={metrics_chunk['participation_ratio']:.3f}, ρ={metrics_chunk['variance_ratio']:.3f}, U={metrics_chunk['uniformity_cosine']:.3f}, "
                f"τ_eff={tau_eff:.4f}, α_cur={alpha_cur:.4f}, α_end={self.alpha_end:.4f}, ε={self.epsilon:.1e}"
            )
            if verbose and (chunk_start % log_interval == 0):
                print(line)
                # Optional coarse debug: variance and rho sanity
                if (chunk_start % 50000 == 0):
                    var_dbg = float(np.var(x.astype(np.float64)))
                    print(f"    DEBUG: var={var_dbg:.6f}, d={self.dim}, ρ_calc={var_dbg*self.dim:.6f}")
            chunk_log_tail.append(line)
            if len(chunk_log_tail) > 20:
                chunk_log_tail = chunk_log_tail[-20:]

            # Uniformity guard: nudge off near-uniform states
            if metrics_chunk['uniformity_cosine'] > self.uniformity_guard_u and metrics_chunk['variance_ratio'] < self.uniformity_guard_var:
                noise = np.random.randn(self.dim)
                noise -= (np.dot(noise, x) * x)
                noise = self.noise_scale * noise / (np.linalg.norm(noise) + 1e-12)
                x = x + noise
                x = x / (np.linalg.norm(x) + 1e-12)

            # Dynamic gating for variance lifts
            if not self._lifts_frozen:
                if (chunk_start / max(1, self.max_steps)) >= 0.3 and not self._lifts_enabled:
                    if metrics_chunk['variance_ratio'] < 0.30:
                        self._lifts_enabled = True
                        self.whitening_gamma = max(self.whitening_gamma, 0.15)
                        self.gain_amplify = max(self.gain_amplify, 0.15)
                        self.contrastive_mag = max(self.contrastive_mag, 0.02)
                        self.blocks = max(self.blocks, 12)
                if self._lifts_enabled and metrics_chunk['variance_ratio'] >= 0.40:
                    self._lifts_enabled = False
                    self._lifts_frozen = True

            # Contrastive lift if variance is low and lifts enabled
            if self._lifts_enabled and metrics_chunk['variance_ratio'] < self.contrastive_threshold and self.v_low is not None:
                x = self._contrastive_lift(x, self.v_low)

            # Stall detection using median delta
            if self.adaptive:
                if median_delta <= self.improve_factor * best_median_delta:
                    best_median_delta = median_delta
                    stalls = 0
                else:
                    stalls += 1
                    if stalls >= self.patience_chunks:
                        # 1) sharpen attention (lower τ)
                        # adapt factor sharpens effective temperature
                        self._adapt_factor = max(0.5, self._adapt_factor * self.temp_decay)
                        # 2) raise retention ceiling slightly
                        self.alpha_end = min(0.9995, self.alpha_end + self.alpha_step)
                        # 3) inject tiny orthogonal noise scaled by remaining progress
                        noise = np.random.randn(self.dim)
                        noise -= (np.dot(noise, x) * x)
                        progress_chunks = (chunk_start + 1) / max(1, self.max_steps)
                        self.noise_scale *= 0.5
                        scaled = (1.0 - progress_chunks) * self.noise_scale
                        x = x + scaled * noise / (np.linalg.norm(noise) + 1e-12)
                        x = x / (np.linalg.norm(x) + 1e-12)
                        if verbose:
                            print(f"    ↪ stall: adapt τ→{self.temperature:.3f}, α_end→{self.alpha_end:.3f}, noise injected")
                        stalls = 0

            # Track low-variance direction for next chunk using worst block proxy
            if self.enable_whitening:
                d = self.dim
                b = max(1, self.blocks)
                base = d // b
                # recompute block std proxies
                stds = []
                for i in range(b):
                    s = i * base
                    e = (i + 1) * base if i < b - 1 else d
                    seg = x[s:e]
                    stds.append(float(seg.std()))
                worst = int(np.argmin(stds))
                s = worst * base
                e = (worst + 1) * base if worst < b - 1 else d
                seg = x[s:e] - x[s:e].mean()
                w = self.rng.randn(e - s)
                w = w / (np.linalg.norm(w) + 1e-12)
                for _ in range(8):
                    w = seg * float(np.dot(seg, w))
                    w = w / (np.linalg.norm(w) + 1e-12)
                v_low = np.zeros_like(x)
                v_low[s:e] = w
                self.v_low = v_low
        
        # Compute final health certificate
        final_metrics = self.compute_metrics(x, x_prev)
        certificate = HealthCertificate(
            converged=converged,
            steps=step + 1,
            **final_metrics
        )
        certificate.epsilon_path = epsilon_path
        certificate.chunk_log_tail = chunk_log_tail
        
        if verbose:
            status = "✅ Converged" if converged else "⚠️ Max steps reached"
            print(f"{status} at step {step + 1}")
            print(f"  Health: {'✅ PASS' if certificate.passes_health_check() else '❌ FAIL'}")
        
        return x, certificate
    
    def test_perturbation_recovery(self, 
                                  x: np.ndarray,
                                  noise_scale: float = 0.01,
                                  max_recovery_steps: int = 1000) -> int:
        """
        Test recovery time from perturbation.
        
        Args:
            x: Fixed-point state to test
            noise_scale: Standard deviation of Gaussian noise
            max_recovery_steps: Maximum steps to attempt recovery
            
        Returns:
            Number of steps to recover (or max_recovery_steps if failed)
        """
        # Add Gaussian noise
        noise = np.random.randn(self.dim) * noise_scale
        x_perturbed = x + noise
        x_perturbed = x_perturbed / np.linalg.norm(x_perturbed)
        
        # Cosine-based recovery with sticky alpha
        x_current = x_perturbed
        alpha_rec = 0.995
        target_sim = 0.999
        for recovery_step in range(max_recovery_steps):
            fx = self.self_attention(x_current)
            x_current = alpha_rec * x_current + (1 - alpha_rec) * fx
            x_current = x_current / (np.linalg.norm(x_current) + 1e-12)
            if float(np.dot(x_current, x)) >= target_sim:
                return recovery_step + 1
                
        return max_recovery_steps
    
    def compute_coupling_metric(self, 
                               x1: np.ndarray, 
                               x2: np.ndarray,
                               coupling_steps: int = 100) -> float:
        """
        Compute service propagation metric (σ) between two states.
        
        This measures how perturbations in one state affect another,
        quantifying "service" or mutual stabilization.
        
        Args:
            x1: First fixed-point state
            x2: Second fixed-point state
            coupling_steps: Number of steps to measure coupling
            
        Returns:
            Service propagation metric (σ > 0 indicates positive coupling)
        """
        # Store initial states
        x1_init = x1.copy()
        x2_init = x2.copy()
        
        # Perturb first state
        x1_perturbed = x1 + np.random.randn(self.dim) * 0.01
        x1_perturbed = x1_perturbed / np.linalg.norm(x1_perturbed)
        
        # Evolve both with weak coupling
        x1_current = x1_perturbed
        x2_current = x2.copy()
        coupling_strength = 0.1
        
        drift_reduction = 0.0
        
        for _ in range(coupling_steps):
            # Self-dynamics
            fx1 = self.self_attention(x1_current)
            fx2 = self.self_attention(x2_current)
            
            # Weak coupling term
            coupling1 = coupling_strength * (x2_current - x1_current)
            coupling2 = coupling_strength * (x1_current - x2_current)
            
            # Update with coupling
            x1_current = self.retention_alpha * x1_current + (1 - self.retention_alpha) * (fx1 + coupling1)
            x2_current = self.retention_alpha * x2_current + (1 - self.retention_alpha) * (fx2 + coupling2)
            
            # Normalize
            x1_current = x1_current / np.linalg.norm(x1_current)
            x2_current = x2_current / np.linalg.norm(x2_current)
            
            # Measure drift reduction
            drift1 = np.linalg.norm(x1_current - x1_init)
            drift2 = np.linalg.norm(x2_current - x2_init)
            drift_reduction += (1.0 / (1.0 + drift1 + drift2))
        
        # Service metric: average drift reduction
        sigma = drift_reduction / coupling_steps
        
        return sigma
