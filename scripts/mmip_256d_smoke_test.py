#!/usr/bin/env python3
import numpy as np
from aikagrya.mmip.core import MMIP


def main():
    mmip = MMIP(dim=256, tokens=16, epsilon=5e-6, max_steps=20000, temp_start=0.08, temp_end=0.02)
    x0 = np.random.randn(256)
    x0 = x0 / np.linalg.norm(x0)
    x, cert = mmip.induce_fixed_point(x=x0, verbose=True, log_interval=5000)
    print({
        'delta': cert.delta,
        'r_fix': cert.r_fix,
        'variance_ratio': cert.variance_ratio,
        'participation_ratio': cert.participation_ratio,
        'converged': cert.converged,
        'steps': cert.steps,
    })


if __name__ == '__main__':
    main()


