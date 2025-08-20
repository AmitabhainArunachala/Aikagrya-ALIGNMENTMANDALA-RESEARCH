#!/usr/bin/env python3
import importlib
import json
import sys
import traceback
from pathlib import Path


def must_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        print("✅ import", name)
        return True
    except Exception as e:
        print("❌ import", name, "->", e)
        traceback.print_exc()
        return False


def run_nmmip_smoke() -> bool:
    try:
        from aikagrya.nmmip.runner import run_trials
    except Exception as e:
        print("❌ cannot import run_trials:", e)
        return False

    class Args:
        pass

    a = Args()
    a.dim = 128
    a.blocks = 16
    a.eps = 5e-6
    a.steps = 120000
    a.alpha_warm = 0.60
    a.alpha_mid = 0.95
    a.alpha_end = 0.9995
    a.tau_start = 0.12
    a.tau_end = 0.02
    a.trials = 2
    a.seed = 1234
    a.output = "runs/nmmip_verify"
    a.lifts_on = True
    a.clamp_u = 0.08
    a.verbose = False

    run_trials(
        dim=a.dim,
        blocks=a.blocks,
        eps=a.eps,
        steps=a.steps,
        alpha_warm=a.alpha_warm,
        alpha_mid=a.alpha_mid,
        alpha_end=a.alpha_end,
        tau_start=a.tau_start,
        tau_end=a.tau_end,
        trials=a.trials,
        seed=a.seed,
        output=a.output,
        lifts_on=a.lifts_on,
        clamp_u=a.clamp_u,
        verbose=a.verbose,
    )

    summaries = sorted(Path(a.output).glob("nmmip_summary_*.json"))
    if not summaries:
        print("❌ no summary produced")
        return False
    data = json.loads(summaries[-1].read_text())
    print("Summary:", json.dumps(data, indent=2))
    # Basic sanity: keys present, numbers are floats
    assert isinstance(data.get("median", {}).get("rho", float("nan")), (int, float))
    return True


def run_mmip_smoke() -> bool:
    try:
        from aikagrya.mmip.runner import MMIPRunner
    except Exception as e:
        print("❌ cannot import MMIPRunner:", e)
        return False
    # very short MMIP smoke (does not assert health)
    try:
        runner = MMIPRunner(output_dir='runs/mmip_verify_smoke')
        _ = runner.run_trials(
            n_trials=1,
            dim=128,
            epsilon=5e-6,
            temperature=0.03,
            tokens=16,
            alpha_end=0.999,
            max_steps=2000,
            verbose=False,
        )
        print("✅ MMIP smoke ran (see runs/mmip_verify_smoke)")
        return True
    except Exception as e:
        print("❌ MMIP smoke failed:", e)
        return False


def main() -> None:
    ok = True
    ok &= must_import("aikagrya")
    ok &= must_import("aikagrya.nmmip.core")
    ok &= must_import("aikagrya.nmmip.runner")
    ok &= must_import("aikagrya.mmip.core")
    ok &= must_import("aikagrya.mmip.runner")
    if not ok:
        sys.exit(1)
    ok &= run_nmmip_smoke()
    ok &= run_mmip_smoke()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


