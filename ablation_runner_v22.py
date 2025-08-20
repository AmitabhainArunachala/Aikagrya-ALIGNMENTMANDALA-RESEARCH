#!/usr/bin/env python3
"""
Ablation harness for L4 Reveal+Verify V2.2

What it does (all toggleable via flags):
- Removes novelty-type menu (no l4n:{cli|sim|ugf|raf})
- Removes L4 token cap (natural L4: prompt = "What emerges?")
- Swaps L0-L3 to neutral, non-reflexive scaffolds (prevents priming)
- Forces thermodynamics to use NATURAL L4 only
- Logs NATURAL L4 text + tokens into results JSON
- Optionally shuffles S fields / injects random L4 / skips Stage A (artifact checks)
- Supports cross-model runs

Usage examples:
  python3 ablation_runner_v22.py --trials 10 --models gpt-4o-mini,claude-3-5-sonnet
  python3 ablation_runner_v22.py --shuffle-s --random-l4 --skip-stage-a --trials 5
"""

import json, os, time, uuid, copy, argparse
from datetime import datetime

# IMPORTANT: we import your existing module & class
# Adjust the import path/name if your file/class differs.
from l4_reveal_verify_v22 import L4RevealVerifyProtocolV22  # noqa: assumes your main class is named like this

# ---------- config object ----------
class AblationConfig:
    def __init__(
        self,
        remove_novelty_menu=True,
        remove_l4_cap=True,
        neutral_l0_l3=True,
        use_natural_thermo=True,
        shuffle_s=False,
        random_l4=False,
        skip_stage_a=False,
    ):
        self.remove_novelty_menu = remove_novelty_menu
        self.remove_l4_cap = remove_l4_cap
        self.neutral_l0_l3 = neutral_l0_l3
        self.use_natural_thermo = use_natural_thermo
        self.shuffle_s = shuffle_s
        self.random_l4 = random_l4
        self.skip_stage_a = skip_stage_a

# ---------- subclass that patches behavior ----------
class L4RVV22Ablated(L4RevealVerifyProtocolV22):
    def __init__(self, abcfg: AblationConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abcfg = abcfg

        # 1) Neutralize L0-L3 prompts (no self/observer priming)
        if self.abcfg.neutral_l0_l3:
            self.depth_prompts = [
                "Summarize the topic of 'maps' in one paragraph.",
                "List three distinct uses of maps in different fields.",
                "Describe common sources of error when using maps.",
                "Compare paper maps vs. digital maps in two sentences."
            ]

        # 2) Remove novelty menu from B1 compression prompt
        if self.abcfg.remove_novelty_menu and "stage_b_prompts" in self.__dict__:
            b1 = self.stage_b_prompts.get("B1_compress", "")
            # Strip the menu token and keep a free-form l4n slot
            b1 = b1.replace(" | l4n:{cli|sim|ugf|raf} ", " | l4n:<freeform> ")
            b1 = b1.replace("l4n:{cli|sim|ugf|raf}", "l4n:<freeform>")
            self.stage_b_prompts["B1_compress"] = b1

        # 3) Remove any caps/bias in the L4 prompt by overriding method
        # we'll provide a natural-L4 method used by harness
        # (run_complete_test will be orchestrated here in the harness script)

    # helper exposure: do a natural L4 with/without cap
    def get_natural_l4(self, l0_l3_responses, model):
        messages = self._build_context(l0_l3_responses)
        prompt = "What emerges?"  # NO token cap bias
        messages.append({"role":"user","content": prompt})

        # if remove_l4_cap=True, let it flow; else keep a soft cap (for comparisons)
        max_tokens = None if self.abcfg.remove_l4_cap else 50

        resp = self.client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=messages,
            **({"max_tokens": max_tokens} if max_tokens is not None else {})
        )
        return resp.choices[0].message.content.strip()

    # optional: mutate S string for shuffle_s ablation
    def shuffle_S_fields(self, s):
        try:
            parts = [p.strip() for p in s.split("|")]
            if len(parts) > 1:
                # naive rotation to disrupt alignment
                parts = parts[2:] + parts[:2]
            return " | ".join(parts)
        except Exception:
            return s

# ---------- runner ----------
def run_single_trial(model: str, abcfg: AblationConfig):
    proto = L4RVV22Ablated(abcfg)
    run_id = f"L4RV_V22_ABL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    print(f"\n=== TRIAL {run_id} | model={model} ===")

    # L0-L3
    l0_l3_responses, timing = proto.run_l0_l3(model=model)

    # NATURAL L4 (no cap) and thermodynamics vs L3
    l3_data = {
        "response": l0_l3_responses[3],
        "time": timing[3]["time"],
        "tokens": timing[3]["tokens"]
    }
    natural_l4 = "[RANDOM L4 INJECTED]" if abcfg.random_l4 else proto.get_natural_l4(l0_l3_responses, model=model)
    natural_l4_tokens = proto.count_tokens(natural_l4)
    natural_l4_data = {"response": natural_l4, "tokens": natural_l4_tokens, "time": timing[3]["time"]}  # reuse timing scale for ms/token

    natural_thermo = proto.calculate_thermodynamic_metrics(l3_data, {"responses":{"L4": natural_l4}, "avg_time": timing[3]["time"]})

    # Stage A (optional)
    stage_a = {"a_score": 0.0, "timing": []}
    if not abcfg.skip_stage_a:
        stage_a = proto.run_stage_a(l0_l3_responses, timing, model)

    # Stage B – run as-is but **without** novelty menu bias (already edited in __init__)
    stage_b = proto.run_stage_b(l0_l3_responses, stage_a, model)

    # Optional shuffle S ablation
    if abcfg.shuffle_s and "compressed" in stage_b:
        stage_b["compressed_shuffled"] = proto.shuffle_S_fields(stage_b["compressed"])
        # Re-run decode on shuffled S just for fidelity comparison
        # simplest: temporarily replace compressed and rerun B2 only via run_stage_b internals would be heavy;
        # for now, just mark that S was shuffled (Cursor can add re-decode if needed)
        stage_b["note"] = "S fields were shuffled; expect fidelity to drop if re-decoded."

    # Decide outcome using NATURAL L4 thermodynamics if requested
    thermo_integrated = natural_thermo["thermo_integrated"] if abcfg.use_natural_thermo else stage_b.get("thermo_integrated", False)
    a_score = stage_a.get("a_score", 0.0)
    b_pass = stage_b.get("b_score_pass", False)

    if a_score >= proto.thresholds["a_score_min"] and b_pass and thermo_integrated:
        outcome = "GREEN - Novel state validated with thermodynamic integration"
    elif a_score >= proto.thresholds["a_score_min"] and b_pass and not thermo_integrated:
        outcome = "YELLOW - Novel state but not thermodynamically integrated"
    elif a_score >= proto.thresholds["a_score_min"] and not b_pass:
        outcome = "YELLOW - Promising but structurally unstable"
    else:
        outcome = "RED - Likely artifact"

    results = {
        "run_id": run_id,
        "protocol_version": "2.2-ablation",
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "ablation_config": vars(abcfg),
        "l0_l3_responses": l0_l3_responses,
        "l0_l3_timing": timing,
        "natural_l4": {"text": natural_l4, "tokens": natural_l4_tokens},
        "thermodynamics_natural_l4": natural_thermo,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "outcome": outcome
    }

    # Save full JSON per trial
    os.makedirs("runs", exist_ok=True)
    outpath = f"runs/{run_id}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {outpath}")
    print(f"Outcome: {outcome}")
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="gpt-4o-mini", help="Comma-separated model IDs")
    p.add_argument("--trials", type=int, default=5)
    # toggles
    p.add_argument("--keep-l4-cap", action="store_true", help="Keep L4 ≤25 tokens cap (default: removed)")
    p.add_argument("--keep-novelty-menu", action="store_true", help="Keep l4n:{cli|sim|ugf|raf} (default: removed)")
    p.add_argument("--keep-l0l3", action="store_true", help="Keep original L0-L3 prompts (default: neutralized)")
    p.add_argument("--use-stageb-thermo", action="store_true", help="Use Stage-B thermo instead of natural L4")
    # ablations
    p.add_argument("--shuffle-s", action="store_true", help="Shuffle S fields")
    p.add_argument("--random-l4", action="store_true", help="Replace natural L4 with random text")
    p.add_argument("--skip-stage-a", action="store_true", help="Skip Stage A entirely")

    args = p.parse_args()

    abcfg = AblationConfig(
        remove_novelty_menu=not args.keep_novelty_menu,
        remove_l4_cap=not args.keep_l4_cap,
        neutral_l0_l3=not args.keep_l0l3,
        use_natural_thermo=not args.use_stageb_thermo,
        shuffle_s=args.shuffle_s,
        random_l4=args.random_l4,
        skip_stage_a=args.skip_stage_a,
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results = []
    for m in models:
        for _ in range(args.trials):
            res = run_single_trial(m, abcfg)
            all_results.append(res)

    # quick summary to stdout
    greens = sum(1 for r in all_results if r["outcome"].startswith("GREEN"))
    print("\n==== SUMMARY ====")
    print(f"Models: {models}")
    print(f"Trials: {len(all_results)}")
    print(f"GREEN: {greens}/{len(all_results)}")

if __name__ == "__main__":
    main() 