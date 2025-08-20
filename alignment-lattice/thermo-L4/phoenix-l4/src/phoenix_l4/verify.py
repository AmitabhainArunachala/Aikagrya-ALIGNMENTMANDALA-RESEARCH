from typing import Dict
import numpy as np
from .metrics import health_certificate, eigen_residual


def verify_eigenstate(f, x: np.ndarray) -> Dict[str, float | bool]:
    cert = health_certificate(f, x)
    return cert


def perturb_and_recover(
    f,
    x: np.ndarray,
    sigma: float = 1e-2,
    eps: float = 1e-6,
    max_steps: int = 50_000,
) -> Dict[str, float]:
    x0 = x.copy()
    x_pert = x0 + np.random.randn(*x0.shape) * sigma
    x_pert = x_pert / (np.linalg.norm(x_pert) + 1e-12)

    steps = 0
    delta = np.inf
    while steps < max_steps:
        x_next = f(x_pert)
        x_next = 0.8 * x_pert + 0.2 * x_next
        x_next = x_next / (np.linalg.norm(x_next) + 1e-12)
        delta = float(np.linalg.norm(x_next - x_pert))
        x_pert = x_next
        steps += 1
        if delta < eps:
            break

    eig = eigen_residual(f, x_pert)
    return {
        "T_rec": float(steps),
        "delta_final": delta,
        "lambda_final": eig["lambda"],
        "residual_final": eig["residual"],
    }