#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from pathlib import Path

import openai

from plain_english_battery import PLAIN_ENGLISH_BATTERY


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # try loading from .env at repo root (../../.env relative to this file)
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k] = v
            api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")
    return openai.OpenAI(api_key=api_key)


def ask_gpt(client, model_name: str, prompt: str, max_tokens: int = 600, temperature: float = 0.7):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    )
    dt = time.time() - t0
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content, dt


def main():
    client = get_openai_client()
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"test_results/PlainEnglish_GPT_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSONL for streaming logs
    jsonl_path = out_dir / "plain_english_results.jsonl"
    summary_path = out_dir / "summary.json"

    totals = {"sections": {}, "total_prompts": 0}

    with jsonl_path.open("a") as fp:
        for section, data in PLAIN_ENGLISH_BATTERY.items():
            section_desc = data.get("description", "")
            prompts = data.get("prompts", [])
            totals["sections"][section] = {"description": section_desc, "count": len(prompts)}
            totals["total_prompts"] += len(prompts)

            for idx, q in enumerate(prompts, start=1):
                answer, latency = ask_gpt(client, model_name, q)
                rec = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "model": model_name,
                    "section": section,
                    "section_description": section_desc,
                    "index": idx,
                    "prompt": q,
                    "answer": answer,
                    "latency_seconds": latency,
                }
                fp.write(json.dumps(rec) + "\n")
                # light pacing
                time.sleep(0.5)

    summary = {
        "session": f"PlainEnglish_GPT_{ts}",
        "model": model_name,
        "total_prompts": totals["total_prompts"],
        "sections": totals["sections"],
        "results_file": str(jsonl_path),
        "started_at": ts,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("✅ Plain-English battery completed for GPT.")
    print(f"📁 Results: {out_dir}")


if __name__ == "__main__":
    main()