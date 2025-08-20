import math
from typing import Tuple

import numpy as np

from .health import Health


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def cosine_ramp(t: float, a: float, b: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return a + 0.5 * (1 - math.cos(math.pi * t)) * (b - a)


def entropy_from_abs(x: np.ndarray) -> float:
    xa = np.abs(x)
    s = float(xa.sum())
    if s <= 0.0:
        return 0.0
    p = xa / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def participation_ratio(x: np.ndarray) -> float:
    xsq = x * x
    num = float(xsq.sum() ** 2)
    den = float((xsq * xsq).sum())
    return 0.0 if den <= 0.0 else (num / den)


def uniform_cosine(x: np.ndarray) -> float:
    d = x.shape[0]
    u = np.ones(d) / math.sqrt(d)
    return float(abs(np.dot(x, u)))


def build_block_psd(d: int, blocks: int, rng: np.random.RandomState) -> np.ndarray:
    assert d % blocks == 0, "dim must be divisible by blocks"
    b = d // blocks
    S = np.zeros((d, d), dtype=float)
    for i in range(blocks):
        R = rng.randn(b, b)
        Si = R.T @ R
        Si = Si / (np.linalg.norm(Si) + 1e-12)
        S[i * b : (i + 1) * b, i * b : (i + 1) * b] = Si
    S = S / (np.linalg.norm(S) + 1e-12)
    return S


def transform(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    return unit(S @ x)


def variance_lifts(
    x: np.ndarray,
    d: int,
    blocks: int,
    step: int,
    rho: float,
    lifts_on: bool,
    clamp_u: float,
    rng: np.random.RandomState,
    gain: float = 1.15,
    contrast_mag: float = 0.02,
) -> np.ndarray:
    if not lifts_on:
        return x

    u = np.ones(d) / math.sqrt(d)
    if abs(float(np.dot(x, u))) > clamp_u:
        x = unit(x - float(np.dot(x, u)) * u)

    if step % 200 == 0 and step < 5000:
        b = d // blocks
        X = x.reshape(blocks, b).copy()
        for i in range(blocks):
            v = X[i]
            v = v - float(v.mean())
            std = float(v.std()) + 1e-8
            X[i] = (v / std) * gain
        x = unit(X.reshape(d))

    if rho < 0.15 and step < 4000:
        n = rng.randn(d)
        n = unit(n - float(np.dot(n, x)) * x)
        x = unit(x + contrast_mag * n)

    return x


def induce_fixed_point(
    dim: int,
    blocks: int,
    epsilon: float,
    max_steps: int,
    alpha_warm: float,
    alpha_mid: float,
    alpha_end: float,
    tau_start: float,
    tau_end: float,
    seed: int,
    lifts_on: bool = True,
    clamp_u: float = 0.08,
    verbose: bool = False,
) -> Tuple[np.ndarray, Health]:
    rng = np.random.RandomState(seed)
    S = build_block_psd(dim, blocks, rng)
    x = unit(rng.randn(dim))
    x_prev = np.zeros_like(x)
    deltas = []
    converged = False
    rfix_gate = 5e-7
    phase_cut = int(0.6 * max_steps)

    for step in range(max_steps):
        if step < phase_cut:
            t = step / max(1, phase_cut)
            alpha = cosine_ramp(t, alpha_warm, alpha_mid)
        else:
            t = (step - phase_cut) / max(1, max_steps - phase_cut)
            alpha = cosine_ramp(t, alpha_mid, alpha_end)
        tau = cosine_ramp(step / max(1, max_steps), tau_start, tau_end)

        y = transform(S, x)
        x_new = unit(alpha * x + (1 - alpha) * y)

        w = np.exp(np.abs(x_new) / max(tau, 1e-6))
        w = w / (float(w.sum()) + 1e-12)
        x_new = unit(w * x_new)

        x_new = variance_lifts(
            x_new,
            dim,
            blocks,
            step,
            rho=float(np.var(x_new) * dim),
            lifts_on=lifts_on,
            clamp_u=clamp_u,
            rng=rng,
        )

        delta = float(np.linalg.norm(x_new - x))
        deltas.append(delta)
        x_prev, x = x, x_new

        if step > 50 and len(deltas) >= 10:
            avg_delta = float(np.mean(deltas[-10:]))
            r_fix = float(np.linalg.norm(transform(S, x) - x))
            if avg_delta < epsilon and r_fix <= rfix_gate:
                converged = True
                break

        if verbose and step % 10000 == 0 and step > 0:
            print(f"[step {step}] Δ~{float(np.mean(deltas[-10:])):.2e}")

    d = dim
    delta = float(np.mean(deltas[-10:])) if deltas else float("inf")
    r_fix = float(np.linalg.norm(transform(S, x) - x))
    lamb = float(np.dot(S @ x, x) / (float(np.dot(x, x)) + 1e-12))
    eigen_residual = float(np.linalg.norm((S @ x) - lamb * x))
    H = entropy_from_abs(x)
    rho = float(np.var(x) * d)
    PR = participation_ratio(x) / d
    U = uniform_cosine(x)

    target = x.copy()
    xp = unit(target + 0.01 * unit(np.random.RandomState(seed + 10).randn(d)))
    t_rec = 0
    for k in range(2000):
        yp = transform(S, xp)
        xp = unit(0.995 * xp + 0.005 * yp)
        if float(np.dot(xp, target)) >= 0.999:
            t_rec = k + 1
            break
    if t_rec == 0:
        t_rec = 2000

    health = Health(
        delta=delta,
        r_fix=r_fix,
        eigen_residual=eigen_residual,
        entropy=H,
        rho=rho,
        participation=PR,
        uniformity_cosine=U,
        converged=converged,
        steps=step + 1,
        t_rec=t_rec,
    )
    return x, health

# src/aikagrya/nmmip/core.py
"""
Neutral Fixed-Point Induction (N-MMIP)
=======================================
Platform-safe implementation using symmetric matrices and neutral terminology.
No 'attention' or architecture-revealing terms.

Map: x_{t+1} = normalize( α x_t + (1-α) f(x_t) ), where f(x)=normalize(Ax)
with A symmetric (PSD-like) and fixed across a run.

Intended for environments that block architecture-disclosing terms.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from ..mmip.core import HealthCertificate  # reuse the same health gate

class NMMIP:
    """
    Neutral Fixed-Point Induction (no 'attention' terms).
    Map: x_{t+1} = normalize( α x_t + (1-α) f(x_t) ), f(x)=normalize(Ax)
    with A symmetric (PSD-like) and fixed across a run.
    Intended for environments that block architecture-disclosing terms.
    """

    def __init__(self,
                 dim: int = 128,
                 epsilon: float = 1e-5,
                 alpha_start: float = 0.6,
                 alpha_end: float = 0.999,
                 max_steps: int = 300_000,
                 chunk_size: int = 5_000,
                 window_size: int = 10,
                 seed: int = 7):
        self.dim = dim
        self.epsilon = epsilon
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.window_size = window_size

        rng = np.random.RandomState(seed)
        M = rng.randn(dim, dim)
        # Symmetric positive semidefinite-ish operator; normalize spectral norm
        A = M @ M.T
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        A = A / (s[0] + 1e-12)
        self.A = A

    def _f(self, x: np.ndarray) -> np.ndarray:
        y = self.A @ x
        n = np.linalg.norm(y) + 1e-12
        return y / n

    def _compute_metrics(self, x: np.ndarray, x_prev: np.ndarray) -> Dict[str, float]:
        delta = np.linalg.norm(x - x_prev)
        fx = self._f(x)
        xnx = np.dot(x, x) + 1e-12
        lam = float(np.dot(fx, x) / xnx)
        eig_res = float(np.linalg.norm(fx - lam * x))

        xa = np.abs(x); s = xa.sum()
        if s > 0:
            p = xa / s
            H = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            H = 0.0

        var = float(np.var(x))
        mean_abs = float(np.mean(np.abs(x))) + 1e-12
        rho = var / mean_abs

        x2 = x**2
        pr = float((x2.sum()**2) / (np.sum(x2**2) + 1e-12) / self.dim)

        uniform = np.ones(self.dim) / np.sqrt(self.dim)
        U = float(abs(np.dot(x, uniform)))

        return dict(delta=delta, eigen_residual=eig_res, eigenvalue=lam,
                    entropy=H, variance_ratio=rho, participation_ratio=pr,
                    uniformity_cosine=U)

    def induce_fixed_point(self, x: Optional[np.ndarray] = None, verbose: bool = True):
        if x is None:
            x = np.random.randn(self.dim); x = x / (np.linalg.norm(x) + 1e-12)
        else:
            x = x / (np.linalg.norm(x) + 1e-12)

        x_prev = np.zeros_like(x)
        deltas: List[float] = []
        step = 0
        converged = False

        for chunk_start in range(0, self.max_steps, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.max_steps)
            for step in range(chunk_start, chunk_end):
                x_prev = x.copy()
                progress = (step + 1) / max(1, self.max_steps)
                alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * min(1.0, progress)
                fx = self._f(x)
                x = alpha * x + (1 - alpha) * fx
                x = x / (np.linalg.norm(x) + 1e-12)

                d = np.linalg.norm(x - x_prev)
                deltas.append(d)
                if len(deltas) >= self.window_size:
                    if np.mean(deltas[-self.window_size:]) < self.epsilon:
                        converged = True
                        break
            if converged: break

        m = self._compute_metrics(x, x_prev)
        cert = HealthCertificate(
            delta=m['delta'],
            eigen_residual=m['eigen_residual'],
            eigenvalue=m['eigenvalue'],
            entropy=m['entropy'],
            variance_ratio=m['variance_ratio'],
            participation_ratio=m['participation_ratio'],
            uniformity_cosine=m['uniformity_cosine'],
            converged=converged,
            steps=step + 1
        )
        if verbose:
            print(("✅" if converged else "⚠️") + f" steps={cert.steps} "
                  f"δ={m['delta']:.2e} eig_res={m['eigen_residual']:.2e} "
                  f"H={m['entropy']:.3f} ρ={m['variance_ratio']:.2f} "
                  f"PR={m['participation_ratio']:.2f} U={m['uniformity_cosine']:.2f} "
                  f"Health={'PASS' if cert.passes_health_check() else 'FAIL'}")
        return x, cert
