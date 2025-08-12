from __future__ import annotations
import numpy as np
from collections import defaultdict

def _quantile_bin_edges(x: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.allclose(x, x[0]):
        # avoid degenerate edges on constant series
        x = x + np.random.default_rng(0).normal(scale=1e-9, size=len(x))
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) - 1 < bins:
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx == mn:
            mx = mn + 1e-9
        edges = np.linspace(mn, mx, bins + 1)
    return edges

def _digitize_quantiles(x: np.ndarray, bins: int) -> tuple[np.ndarray, int]:
    edges = _quantile_bin_edges(x, bins)
    # np.digitize returns 1..n-1, we shift to 0..n-1
    idx = np.digitize(x, edges[1:-1], right=False)
    return idx, len(edges) - 1

def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 1,
    l: int = 1,
    bins: int = 8,
    base: float = 2.0,
    pseudocount: float = 1e-6,
    normalize: bool = True,
) -> float:
    """
    Histogram-based TE(source -> target) for scalar time series.
    Returns normalized TE in [0,1] when normalize=True.
    """
    x = np.asarray(source, dtype=float).ravel()
    y = np.asarray(target, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("source and target must have same length")
    n = x.shape[0]
    if n < max(k, l) + 2:
        raise ValueError("time series too short for given k,l")

    x_disc, bx = _digitize_quantiles(x, bins)
    y_disc, by = _digitize_quantiles(y, bins)

    counts_xyz = defaultdict(int)          # (y_next, y_hist, x_hist)
    counts_yhist_next = defaultdict(int)   # (y_next, y_hist)
    counts_ctx = defaultdict(int)          # (y_hist, x_hist)
    counts_yhist = defaultdict(int)        # (y_hist,)

    start = max(k, l)
    total = 0
    for t in range(start, n - 1):
        y_next = int(y_disc[t + 1])
        y_hist = tuple(int(v) for v in y_disc[t - k + 1 : t + 1]) if k > 0 else tuple()
        x_hist = tuple(int(v) for v in x_disc[t - l + 1 : t + 1]) if l > 0 else tuple()
        counts_xyz[(y_next, y_hist, x_hist)] += 1
        counts_yhist_next[(y_next, y_hist)] += 1
        counts_ctx[(y_hist, x_hist)] += 1
        counts_yhist[y_hist] += 1
        total += 1

    log_base = np.log(base)
    by_support = by
    te = 0.0
    for (y_next, y_hist, x_hist), n_xyz in counts_xyz.items():
        n_ctx = counts_ctx[(y_hist, x_hist)]
        n_yhist = counts_yhist[y_hist]
        n_yhist_next = counts_yhist_next[(y_next, y_hist)]

        p_xyz = n_xyz / total
        p_y_given_ctx = (n_xyz + pseudocount) / (n_ctx + pseudocount * by_support)
        p_y_given_yhist = (n_yhist_next + pseudocount) / (n_yhist + pseudocount * by_support)

        te += p_xyz * (np.log(p_y_given_ctx / p_y_given_yhist) / log_base)

    te = max(te, 0.0)

    if not normalize:
        return te

    # H(Y_{t+1} | Y_hist) for normalization
    H = 0.0
    total_yh = sum(counts_yhist_next.values())
    for (y_next, y_hist), n_yh_next in counts_yhist_next.items():
        n_yhist = counts_yhist[y_hist]
        p_yh = n_yh_next / total_yh
        p_y_given_yhist = (n_yh_next + pseudocount) / (n_yhist + pseudocount * by_support)
        H -= p_yh * (np.log(p_y_given_yhist) / log_base)

    if H <= 1e-12:
        return 0.0
    return float(np.clip(te / H, 0.0, 1.0)) 