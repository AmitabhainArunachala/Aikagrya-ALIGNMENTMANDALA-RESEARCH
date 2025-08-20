#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Usage: aggregate_mmip.py <runs_dir>")
    sys.exit(1)

root = Path(sys.argv[1])
files = sorted(root.glob('*.jsonl')) or sorted(root.glob('**/*.jsonl'))
if not files:
    print(f"No JSONL files under {root}")
    sys.exit(0)

from collections import defaultdict
rows = defaultdict(list)

for jl in files:
    with open(jl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (
                int(r.get('dim', 0)),
                float(r.get('temperature', 0.0)),
                float(r.get('alpha_end', 0.0) or 0.0),
                int(r.get('tokens', 0)),
            )
            rows[key].append(r)

def med(vals, k):
    arr = [v.get(k) for v in vals if v.get(k) is not None]
    return float(np.median(arr)) if arr else float('nan')

def rate(vals):
    conv = sum(1 for v in vals if v.get('converged'))
    hp = sum(1 for v in vals if v.get('health_pass'))
    return conv, hp

print("dim  tau    a_end   T  trials  conv  health  steps_med  d_med   eig_res_med    H_med  rho_med  PR_med  U_med  Trec_med")
for (dim, tau, aend, tok), vals in sorted(rows.items()):
    conv, hp = rate(vals)
    msteps = med(vals, 'steps')
    mdelta = med(vals, 'delta')
    meig = med(vals, 'eigen_residual')
    mH = med(vals, 'entropy')
    mrho = med(vals, 'variance_ratio')
    mPR = med(vals, 'participation_ratio')
    mU = med(vals, 'uniformity_cosine')
    mTr = med(vals, 'recovery_time')
    print(f"{dim:<4} {tau:<5.2f}  {aend:<6.4f} {tok:<2d} {len(vals):>6} {conv:>5} {hp:>7}  {msteps:>9.0f}  {mdelta:>6.0e}  {meig:>12.0e}  {mH:>5.2f}  {mrho:>6.2f}  {mPR:>6.2f}  {mU:>5.2f}  {mTr:>7.1f}")
