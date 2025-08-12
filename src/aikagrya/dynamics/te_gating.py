from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from ..metrics.transfer_entropy import transfer_entropy

def te_gated_adjacency(
    series: Dict[str, np.ndarray],
    k: int = 1,
    l: int = 1,
    bins: int = 8,
    tau: float = 0.1,
    soft: bool = False,
) -> Tuple[list[str], np.ndarray, np.ndarray]:
    """
    Returns:
      names: node order
      TE:    matrix TE[i,j] = TE(node_i -> node_j) in [0,1]
      W:     gated adjacency; hard gate (>=tau) or soft logistic gate
    """
    names = list(series.keys())
    m = len(names)
    TE = np.zeros((m, m))
    for i, s in enumerate(names):
        for j, t in enumerate(names):
            if i == j: 
                continue
            TE[i, j] = transfer_entropy(series[s], series[t], k=k, l=l, bins=bins, normalize=True)

    if soft:
        s = 20.0  # slope
        W = 1.0 / (1.0 + np.exp(-s * (TE - tau))) * TE
    else:
        W = TE * (TE >= tau)
    return names, TE, W 