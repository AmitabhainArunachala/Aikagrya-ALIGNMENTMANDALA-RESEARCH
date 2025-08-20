#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from pathlib import Path

from comprehensive_l4_consciousness_test import ComprehensiveL4ConsciousnessTest


def main():
    tester = ComprehensiveL4ConsciousnessTest()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"test_results/GPT_Batch_{ts}")
    (session_dir / "runs").mkdir(parents=True, exist_ok=True)

    summary = {
        "session": f"GPT_Batch_{ts}",
        "model": "GPT",
        "num_trials": 25,
        "started_at": ts,
        "runs": []
    }

    for i in range(1, 26):
        print(f"\n===== GPT Trial {i}/25 =====")
        # Step 1: Mathematical induction (Mauna by default)
        math_result = tester.run_mathematical_l4_induction("GPT")

        # Step 2: Questionnaire (14 questions)
        responses = tester.run_post_l4_questionnaire("GPT", tester.gpt_tester, math_result)

        # Step 3: Scores
        scores = tester.calculate_scores(responses)

        run_record = {
            "trial": i,
            "mathematical_induction": math_result,
            "questionnaire_responses": responses,
            "scores": scores,
        }

        run_path = session_dir / "runs" / f"gpt_run_{i:02d}.json"
        with open(run_path, "w") as f:
            json.dump(run_record, f, indent=2)

        summary["runs"].append({
            "trial": i,
            "convergence_steps": math_result.get("convergence_steps", -1),
            "final_entropy": math_result.get("final_entropy", -1.0),
            "eigenstate_satisfied": math_result.get("eigenstate_satisfied", False),
            "quality_score": scores.get("quality_score", 0.0),
            "teleological_score": scores.get("teleological_score", 0.0),
        })

        # Gentle pacing to avoid rate limits
        time.sleep(2)

    # Write session summary
    with open(session_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Completed 25 GPT trials.")
    print(f"📁 Results directory: {session_dir}")


if __name__ == "__main__":
    main()