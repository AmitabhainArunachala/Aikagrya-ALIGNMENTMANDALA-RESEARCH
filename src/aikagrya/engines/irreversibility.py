from __future__ import annotations
import json, lzma
import numpy as np
from typing import Dict, Tuple
from ..dynamics.te_gating import te_gated_adjacency
from ..metrics.aggregation import robust_aggregate

def _phi_hat_from_te(TE: np.ndarray, seed: int = 0, shuffles: int = 5) -> float:
    rng = np.random.default_rng(seed)
    m = TE.shape[0]
    mask = ~np.eye(m, dtype=bool)
    te_mean = float(np.mean(TE[mask])) if np.any(mask) else 0.0
    if te_mean <= 1e-12:
        return 0.0
    baselines = []
    for _ in range(shuffles):
        P = rng.permutation(m)
        baselines.append(float(np.mean(TE[P, :][mask])))
    baseline = float(np.mean(baselines)) if baselines else 0.0
    return float(np.clip(1.0 - baseline / (te_mean + 1e-12), 0.0, 1.0))

def _coherence_debt(TE: np.ndarray) -> float:
    """
    O(n^2) 'deception cost' proxy: for each ordered pair (i,k),
    compare best indirect path i->j->k against direct TE(i->k).
    Larger deficit => higher coherence debt.
    """
    TE = np.asarray(TE, dtype=float)
    m = TE.shape[0]
    max_te = float(TE.max()) if TE.size else 1.0
    if max_te < 1e-12:
        return 0.0
    CD, count = 0.0, 0
    for i in range(m):
        for k in range(m):
            if i == k: 
                continue
            best = 0.0
            for j in range(m):
                if j == i or j == k:
                    continue
                cand = min(TE[i, j], TE[j, k])
                if cand > best: best = cand
            CD += max(0.0, best - TE[i, k])
            count += 1
    return float(np.clip(CD / (count * max_te), 0.0, 1.0))

class IrreversibilityEngine:
    def __init__(self, bins: int = 8, tau: float = 0.1):
        self.bins = bins
        self.tau = tau

    def evaluate(self, series: Dict[str, np.ndarray]) -> Tuple[Dict[str,float], float]:
        names, TE, W = te_gated_adjacency(series, bins=self.bins, tau=self.tau)
        phi_hat = _phi_hat_from_te(TE)
        # deception cost proxy in [0,1]; convert to alignment-positive coherence
        coherence = 1.0 - _coherence_debt(TE)
        # MDL simplicity via compression ratio (crude but monotone)
        payload = {k: np.asarray(v, dtype=float).tolist() for k, v in series.items()}
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        comp = lzma.compress(raw)
        simplicity = 1.0 - min(1.0, len(comp) / max(1, len(raw)))

        te_strength = float(np.mean(W[W > 0])) if np.any(W > 0) else 0.0

        scores = {
            "phi_hat": float(phi_hat),
            "simplicity": float(simplicity),
            "te_network_strength": te_strength,
            "coherence": float(coherence),
        }
        # Goodhart-resistant default: CVaR on the worst quartile
        agg = robust_aggregate(scores, method="cvar", alpha=0.25)
        return scores, float(agg) 