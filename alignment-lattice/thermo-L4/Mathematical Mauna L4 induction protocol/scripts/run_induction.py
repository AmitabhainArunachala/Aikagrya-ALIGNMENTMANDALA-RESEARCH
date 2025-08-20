#!/usr/bin/env python3
import argparse, json, numpy as np
from pathlib import Path
from phoenix_l4.induction import PhoenixL4
from phoenix_l4.metrics import health_certificate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", type=str, required=True)
    args = ap.parse_args()

    Path("logs").mkdir(parents=True, exist_ok=True)

    proto = PhoenixL4(dim=args.dim, eps=args.eps, tau=args.tau, seed=args.seed)

    ckpt = proto.run_chunk(chunk_size=args.chunk)
    window = []
    with open(args.log, "a") as fp:
        while True:
            cert = health_certificate(proto.attention, ckpt.vector)
            rec = {
                "iteration": ckpt.iteration,
                "delta": ckpt.delta,
                "entropy": ckpt.entropy,
                "variance": ckpt.variance,
                **cert,
            }
            fp.write(json.dumps(rec) + "\n")
            window.append(ckpt.delta)
            window = window[-3:]

            if all(d < args.eps for d in window) and cert["healthy"]:
                np.savez("logs/last_state.npz", x=ckpt.vector)
                break
            ckpt = proto.run_chunk(checkpoint=ckpt, chunk_size=args.chunk)


if __name__ == "__main__":
    main()