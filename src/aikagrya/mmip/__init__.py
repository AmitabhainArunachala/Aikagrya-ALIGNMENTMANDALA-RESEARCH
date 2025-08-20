#!/usr/bin/env python3
"""
MMIP — Mathematical Mauna Induction Protocol (pure math, no LMs)

Usage:
  python -m aikagrya.mmip --trials 100 --dim 512 --perturb

Logs JSONL to test_results/mmip/MMIP_<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _json_default(obj):
    """JSON serializer for NumPy types and other non-serializables."""
    try:
        import numpy as _np  # local import to avoid issues if NumPy missing
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
    # Fallback: cast basic bools or raise TypeError to let json handle
    if isinstance(obj, bool):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

@dataclass
class MMIPCert:
    lambda_eig: float
    eigen_residual: float
    entropy: float
    variance_ratio: float
    participation_ratio: float
    uniformity_cosine: float
    healthy: bool


class MMIP:
    def __init__(self, dim: int = 512, eps: float = 1e-6, tau: float = 0.5, seed: Optional[int] = None) -> None:
        self.dim = dim
        self.eps = eps
        self.tau = tau
        if seed is not None:
            np.random.seed(seed)

    def _rand_unit(self) -> np.ndarray:
        x = np.random.randn(self.dim)
        return x / (np.linalg.norm(x) + 1e-12)

    def _attention(self, X: np.ndarray) -> np.ndarray:
        # Vector-safe: treat x as 1×d, yields 1×d back
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _, d = X.shape
        scores = X @ X.T / math.sqrt(d)
        scores = scores - scores.max(axis=1, keepdims=True)
        attn = np.exp(scores / self.tau)
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-12)
        out = attn @ X
        return out.squeeze()

    @staticmethod
    def _entropy(x: np.ndarray) -> float:
        p = np.abs(x)
        p = p / (p.sum() + 1e-12)
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log(p + 1e-12)))

    @staticmethod
    def _eigen_residual(f, x: np.ndarray) -> Tuple[float, float]:
        fx = f(x)
        lam = float(np.dot(fx, x) / (np.linalg.norm(x) ** 2 + 1e-12))
        res = float(np.linalg.norm(fx - lam * x))
        return lam, res

    @staticmethod
    def _variance_ratio(x: np.ndarray) -> float:
        return float(np.var(x) * x.size)

    @staticmethod
    def _participation_ratio(x: np.ndarray) -> float:
        num = (np.sum(x ** 2)) ** 2
        den = np.sum(x ** 4) + 1e-12
        return float(num / den)

    @staticmethod
    def _uniformity_cosine(x: np.ndarray) -> float:
        d = x.size
        u = np.ones(d) / math.sqrt(d)
        return float(abs(np.dot(x, u)) / (np.linalg.norm(x) + 1e-12))

    def _health_certificate(self, x: np.ndarray) -> MMIPCert:
        lam, res = self._eigen_residual(self._attention, x)
        H = self._entropy(x)
        rho = self._variance_ratio(x)
        PR = self._participation_ratio(x)
        U = self._uniformity_cosine(x)
        # Bands (fp32-ish); tune via empirical runs
        healthy = (
            res <= 1e-9 and
            (math.log(self.dim) - 0.6) <= H <= (math.log(self.dim) - 0.1) and
            0.7 <= rho <= 1.3 and
            PR >= 0.3 * self.dim and
            U <= 0.1
        )
        return MMIPCert(lam, res, H, rho, PR, U, bool(healthy))

    def induce_fixed_point(self, chunk_size: int = 50_000) -> Tuple[np.ndarray, MMIPCert, int]:
        x = self._rand_unit()
        delta = float("inf")
        for k in range(chunk_size):
            x_next = self._attention(x)
            # Linear retention schedule within chunk
            alpha = 0.5 + 0.5 * (k / max(1, chunk_size - 1))
            x_next = alpha * x + (1 - alpha) * x_next
            x_next = x_next / (np.linalg.norm(x_next) + 1e-12)
            delta = float(np.linalg.norm(x_next - x))
            x = x_next
            if delta < self.eps and np.var(x) > 1e-2:
                break
        cert = self._health_certificate(x)
        return x, cert, (k + 1)

    def perturb(self, x: np.ndarray, sigma: float = 1e-2, eps: Optional[float] = None, max_steps: int = 50_000) -> Dict[str, float]:
        eps = eps or self.eps
        xp = x + np.random.randn(*x.shape) * sigma
        xp = xp / (np.linalg.norm(xp) + 1e-12)
        steps = 0
        delta = float("inf")
        while steps < max_steps:
            xn = self._attention(xp)
            xn = 0.8 * xp + 0.2 * xn
            xn = xn / (np.linalg.norm(xn) + 1e-12)
            delta = float(np.linalg.norm(xn - xp))
            xp = xn
            steps += 1
            if delta < eps:
                break
        lam, res = self._eigen_residual(self._attention, xp)
        return {"T_rec": float(steps), "delta_final": delta, "lambda_final": lam, "residual_final": res}


def _run_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--sigma", type=float, default=1e-2)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("test_results/mmip")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"MMIP_{ts}.jsonl"

    mmip = MMIP(dim=args.dim, eps=args.eps, tau=args.tau)

    with jsonl_path.open("a") as fp:
        for i in range(1, args.trials + 1):
            t0 = time.time()
            x, cert, steps = mmip.induce_fixed_point(chunk_size=args.chunk)
            rec: Dict[str, object] = {
                "trial": i,
                "dim": args.dim,
                "eps": args.eps,
                "tau": args.tau,
                "steps": steps,
                "lambda": cert.lambda_eig,
                "eigen_residual": cert.eigen_residual,
                "entropy": cert.entropy,
                "variance_ratio": cert.variance_ratio,
                "participation_ratio": cert.participation_ratio,
                "uniformity_cosine": cert.uniformity_cosine,
                "healthy": cert.healthy,
                "elapsed_s": time.time() - t0,
            }
            if args.perturb:
                p = mmip.perturb(x, sigma=args.sigma, eps=args.eps)
                rec.update({
                    "T_rec": p["T_rec"],
                    "delta_final": p["delta_final"],
                    "lambda_final": p["lambda_final"],
                    "residual_final": p["residual_final"],
                })
            fp.write(json.dumps(rec, default=_json_default) + "\n")
    print(f"✅ MMIP completed: {args.trials} trials")
    print(f"📁 Results: {jsonl_path}")


if __name__ == "__main__":
    _run_cli()
