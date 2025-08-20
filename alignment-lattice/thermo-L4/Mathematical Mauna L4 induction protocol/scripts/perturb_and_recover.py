#!/usr/bin/env python3
import argparse, json, numpy as np
from phoenix_l4.induction import PhoenixL4
from phoenix_l4.verify import perturb_and_recover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_path", type=str, required=True)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=1e-2)
    ap.add_argument("--log", type=str, required=True)
    args = ap.parse_args()

    x = np.load(args.state_path)["x"]
    proto = PhoenixL4(dim=args.dim, eps=args.eps, tau=args.tau)
    out = perturb_and_recover(proto.attention, x, sigma=args.sigma, eps=args.eps)

    with open(args.log, "a") as fp:
        fp.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()