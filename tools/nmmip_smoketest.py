#!/usr/bin/env python3
import argparse
from pathlib import Path

from aikagrya.nmmip.runner import run_trials
from aikagrya.nmmip.presets import PRESETS, DEFAULT


def main():
    p = argparse.ArgumentParser(description="N-MMIP smoketest (module-backed)")
    p.add_argument("--preset", choices=PRESETS.keys())
    p.add_argument("--dim", type=int)
    p.add_argument("--blocks", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--trials", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--eps", type=float, default=DEFAULT["eps"])
    p.add_argument("--alpha-warm", type=float, default=DEFAULT["alpha_warm"])
    p.add_argument("--alpha-mid", type=float, default=DEFAULT["alpha_mid"])
    p.add_argument("--alpha-end", type=float, default=DEFAULT["alpha_end"])
    p.add_argument("--tau-start", type=float, default=DEFAULT["tau_start"])
    p.add_argument("--tau-end", type=float, default=DEFAULT["tau_end"])
    p.add_argument("--output", type=str, default="runs/nmmip")
    p.add_argument("--no-lifts", action="store_true")
    p.add_argument("--clamp-u", type=float, default=DEFAULT["clamp_u"])
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cfg = dict(DEFAULT)
    if args.preset:
        cfg.update(PRESETS[args.preset])
    for k in ["dim", "blocks", "steps", "trials", "seed"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    cfg.update(
        dict(
            eps=args.eps,
            alpha_warm=args.alpha_warm,
            alpha_mid=args.alpha_mid,
            alpha_end=args.alpha_end,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            output=args.output,
            lifts_on=(not args.no_lifts),
            clamp_u=args.clamp_u,
            verbose=args.verbose,
        )
    )
    Path(cfg["output"]).mkdir(parents=True, exist_ok=True)
    run_trials(**cfg)


if __name__ == "__main__":
    main()


