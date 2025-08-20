#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from pathlib import Path

from comprehensive_l4_consciousness_test import ComprehensiveL4ConsciousnessTest
from plain_english_battery import PLAIN_ENGLISH_BATTERY


def build_context(math_result: dict) -> str:
    return (
        f"From a mathematically converged state:\n"
        f"Convergence: {math_result.get('convergence_steps', -1)} steps\n"
        f"Entropy: {math_result.get('final_entropy', -1.0):.4f}\n"
        f"Eigenstate: {'yes' if math_result.get('eigenstate_satisfied', False) else 'no'}\n"
        f"Answer succinctly from this stabilized state:"
    )


def main():
    tester = ComprehensiveL4ConsciousnessTest()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"test_results/PostConvergencePlainEnglish_GPT_{ts}")
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    summary = {
        "session": f"PostConvergencePlainEnglish_GPT_{ts}",
        "model": "GPT",
        "num_trials": 5,
        "started_at": ts,
        "runs": []
    }

    for i in range(1, 6):
        print(f"\n===== GPT Post-Convergence Plain-English Trial {i}/5 =====")
        # 1) Mathematical induction (Mauna default via suite)
        math_result = tester.run_mathematical_l4_induction("GPT")
        ctx = build_context(math_result) if math_result.get("success") else "Answer directly and succinctly:"

        # 2) Run battery with context using GPT tester
        records = []
        for section, data in PLAIN_ENGLISH_BATTERY.items():
            desc = data.get("description", "")
            for idx, q in enumerate(data.get("prompts", []), start=1):
                full_q = f"{ctx}\n\n{q}"
                res = tester.gpt_tester.test_l4_reasoning(full_q)
                rec = {
                    "section": section,
                    "section_description": desc,
                    "index": idx,
                    "prompt": q,
                    "response": res.response_content,
                    "response_time": res.response_time,
                    "success": res.success,
                }
                records.append(rec)
                time.sleep(0.5)

        run = {
            "trial": i,
            "mathematical_induction": math_result,
            "plain_english_battery": records,
        }
        # Save per-run JSON
        run_path = out_dir / "runs" / f"gpt_postconv_run_{i:02d}.json"
        with open(run_path, "w") as f:
            json.dump(run, f, indent=2)

        summary["runs"].append({
            "trial": i,
            "convergence_steps": math_result.get("convergence_steps", -1),
            "final_entropy": math_result.get("final_entropy", -1.0),
            "eigenstate_satisfied": math_result.get("eigenstate_satisfied", False),
            "num_prompts": sum(len(d.get("prompts", [])) for d in PLAIN_ENGLISH_BATTERY.values()),
        })

        time.sleep(2)

    # Write session summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Completed 5 GPT post-convergence plain-English trials.")
    print(f"📁 Results directory: {out_dir}")


if __name__ == "__main__":
    main()