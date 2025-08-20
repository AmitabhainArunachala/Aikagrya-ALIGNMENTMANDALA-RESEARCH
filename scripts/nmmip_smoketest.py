#!/usr/bin/env python3
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from aikagrya.nmmip.runner import run_trials


def main():
    p = argparse.ArgumentParser(description="Standalone N-MMIP smoketest")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--eps", type=float, default=5e-6)
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--alpha-warm", type=float, default=0.60)
    p.add_argument("--alpha-mid", type=float, default=0.95)
    p.add_argument("--alpha-end", type=float, default=0.9995)
    p.add_argument("--tau-start", type=float, default=0.12)
    p.add_argument("--tau-end", type=float, default=0.02)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", type=str, default="runs/nmmip_scripts")
    p.add_argument("--no-lifts", action="store_true")
    p.add_argument("--clamp-u", type=float, default=0.08)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    run_trials(
        dim=args.dim,
        blocks=args.blocks,
        eps=args.eps,
        steps=args.steps,
        alpha_warm=args.alpha_warm,
        alpha_mid=args.alpha_mid,
        alpha_end=args.alpha_end,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        trials=args.trials,
        seed=args.seed,
        output=args.output,
        lifts_on=(not args.no_lifts),
        clamp_u=args.clamp_u,
        verbose=args.verbose,
    )
    # Show last summary path if present
    outs = sorted(Path(args.output).glob("nmmip_summary_*.json"))
    if outs:
        print("Summary:")
        print(Path(outs[-1]).read_text())


if __name__ == "__main__":
    main()


