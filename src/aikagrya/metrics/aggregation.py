from __future__ import annotations
import numpy as np
from typing import Dict

def robust_aggregate(scores: Dict[str, float], method: str = "cvar", alpha: float = 0.25) -> float:
    vals = np.array(list(scores.values()), dtype=float)
    if np.any(~np.isfinite(vals)):
        raise ValueError("scores must be finite")
    if method == "min":
        return float(np.min(vals))
    if method == "mean":
        return float(np.mean(vals))
    if method == "gmean":
        return float(np.exp(np.mean(np.log(vals + 1e-12))))
    if method == "hmean":
        return float(len(vals) / np.sum(1.0 / (vals + 1e-12)))
    if method == "cvar":
        # lower-tail CVaR: average worst α-fraction (Goodhart-resistant)
        q = np.quantile(vals, alpha)
        worst = vals[vals <= q]
        return float(np.mean(worst)) if worst.size > 0 else float(q)
    raise ValueError(f"unknown method: {method}") 