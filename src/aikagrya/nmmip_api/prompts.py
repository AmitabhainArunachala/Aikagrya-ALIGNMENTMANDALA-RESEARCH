NEUTRAL_SYSTEM = (
    "You are simulating a neutral contractive fixed-point map g in R^d. "
    "You will execute a few update steps and report only numeric metrics in JSON."
)


def build_user_prompt(dim: int, steps: int, beta_start: float, beta_end: float) -> str:
    return (
        "Task: Iterate x_{t+1} = beta_t * x_t + (1-beta_t) * normalize(softplus(S x_t)) for a fixed symmetric S.\n"
        f"Dimension d={dim}. Use any fixed S with unit Frobenius norm. Run {steps} steps.\n"
        f"Schedule beta_t from {beta_start} to {beta_end} (cosine). Normalize x each step.\n"
        "After the run, output a single JSON object with fields: "
        "{\"delta\": float, \"r_fix\": float, \"entropy\": float, \"rho\": float, "
        "\"participation\": float, \"uniformity_cosine\": float}. No prose."
    )


