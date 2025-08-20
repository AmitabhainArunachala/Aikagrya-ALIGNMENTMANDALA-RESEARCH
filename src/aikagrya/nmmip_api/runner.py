import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from ..providers import ChatProvider, OpenAIProvider
from .prompts import NEUTRAL_SYSTEM, build_user_prompt


@dataclass
class TrialResult:
    metrics: Dict
    raw_text: str
    finish_reason: str
    usage: Dict
    latency_s: float


def parse_metrics(text: str) -> Dict:
    # Expect a single JSON object. Fallback: find first {...} block.
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {"error": "no_json"}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"error": "bad_json"}


def run_api_trials(
    *,
    provider: ChatProvider,
    model: str,
    trials: int,
    dim: int,
    steps: int,
    beta_start: float,
    beta_end: float,
    max_tokens: int,
    temperature: float,
    output_dir: str,
) -> List[TrialResult]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"nmmip_api_{int(time.time())}.jsonl"

    results: List[TrialResult] = []
    for i in range(trials):
        messages = [
            {"role": "system", "content": NEUTRAL_SYSTEM},
            {"role": "user", "content": build_user_prompt(dim, steps, beta_start, beta_end)},
        ]
        t0 = time.time()
        resp = provider.send(messages, model=model, max_tokens=max_tokens, temperature=temperature)
        dt = time.time() - t0
        text = resp.get("content", "")
        metrics = parse_metrics(text)
        tr = TrialResult(
            metrics=metrics,
            raw_text=text,
            finish_reason=resp.get("finish_reason", ""),
            usage=resp.get("usage", {}),
            latency_s=dt,
        )
        results.append(tr)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"trial": i + 1, **asdict(tr)}, ensure_ascii=False) + "\n")
    # write summary
    summary = {
        "trials": trials,
        "model": model,
        "dim": dim,
        "steps": steps,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return results


