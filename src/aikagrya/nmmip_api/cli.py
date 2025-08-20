import argparse

from ..providers import OpenAIProvider
from .runner import run_api_trials


def main():
    p = argparse.ArgumentParser(description="API-backed N-MMIP runner")
    p.add_argument("--provider", choices=["openai"], default="openai")
    p.add_argument("--model", required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--beta-start", type=float, default=0.6)
    p.add_argument("--beta-end", type=float, default=0.995)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--output", type=str, default="runs/nmmip_api_openai")
    args = p.parse_args()

    if args.provider == "openai":
        provider = OpenAIProvider()
    else:
        raise SystemExit("Unsupported provider")

    run_api_trials(
        provider=provider,
        model=args.model,
        trials=args.trials,
        dim=args.dim,
        steps=args.steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()


