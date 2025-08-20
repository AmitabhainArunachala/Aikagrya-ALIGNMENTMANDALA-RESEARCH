import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import induce_fixed_point


def _safe_float(v):
    return float(np.asarray(v))


def _fingerprint(x: np.ndarray) -> int:
    s = np.sign(np.where(x[:64] == 0, 1e-12, x[:64]))
    return int(s.dot(np.arange(1, 65)))


def run_trials(
    *,
    dim: int,
    blocks: int,
    eps: float,
    steps: int,
    alpha_warm: float,
    alpha_mid: float,
    alpha_end: float,
    tau_start: float,
    tau_end: float,
    trials: int,
    seed: int,
    output: str,
    lifts_on: bool = True,
    clamp_u: float = 0.08,
    verbose: bool = False,
):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl = out / f"nmmip_{ts}.jsonl"
    summary = {
        "session": f"NMMIP_{ts}",
        "dim": dim,
        "blocks": blocks,
        "epsilon": eps,
        "max_steps": steps,
        "alpha": [alpha_warm, alpha_mid, alpha_end],
        "tau": [tau_start, tau_end],
        "trials": trials,
        "lenient_passes": 0,
        "strict_passes": 0,
        "median": {},
    }
    recs = []
    for i in range(trials):
        x, h = induce_fixed_point(
            dim=dim,
            blocks=blocks,
            epsilon=eps,
            max_steps=steps,
            alpha_warm=alpha_warm,
            alpha_mid=alpha_mid,
            alpha_end=alpha_end,
            tau_start=tau_start,
            tau_end=tau_end,
            seed=seed + i,
            lifts_on=lifts_on,
            clamp_u=clamp_u,
            verbose=verbose,
        )
        rec = {
            "trial": i + 1,
            "seed": seed + i,
            "health": {k: (_safe_float(v) if isinstance(v, (np.generic,)) else v) for k, v in asdict(h).items()},
            "lenient": h.passes_lenient(),
            "strict": h.passes_strict(),
            "fingerprint": _fingerprint(x),
        }
        summary["lenient_passes"] += int(rec["lenient"])
        summary["strict_passes"] += int(rec["strict"])
        recs.append(rec)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def med(key):
        arr = [r["health"][key] for r in recs]
        return float(np.median(arr)) if arr else float("nan")

    for k in [
        "delta",
        "r_fix",
        "eigen_residual",
        "entropy",
        "rho",
        "participation",
        "uniformity_cosine",
        "steps",
        "t_rec",
    ]:
        summary["median"][k] = med(k)

    with open(out / f"nmmip_summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== N-MMIP summary ===")
    print(json.dumps(summary, indent=2))
    return summary


