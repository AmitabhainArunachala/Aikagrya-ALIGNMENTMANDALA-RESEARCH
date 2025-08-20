import numpy as np
from typing import Dict


def eigen_residual(f, x: np.ndarray) -> Dict[str, float]:
    fx = f(x)
    lam = float(np.dot(fx, x) / (np.linalg.norm(x) ** 2 + 1e-12))
    res = float(np.linalg.norm(fx - lam * x))
    return {"lambda": lam, "residual": res}


def variance_ratio(x: np.ndarray) -> float:
    d = x.size
    return float(np.var(x) * d)


def participation_ratio(x: np.ndarray) -> float:
    num = (np.sum(x ** 2)) ** 2
    den = np.sum(x ** 4) + 1e-12
    return float(num / den)


def uniformity_cosine(x: np.ndarray) -> float:
    d = x.size
    u = np.ones(d) / np.sqrt(d)
    return float(abs(np.dot(x, u)) / (np.linalg.norm(x) + 1e-12))


def entropy(x: np.ndarray) -> float:
    p = np.abs(x)
    p = p / (p.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p + 1e-12)))


def health_certificate(f, x: np.ndarray) -> Dict[str, float | bool]:
    d = x.size
    H = entropy(x)
    rho = variance_ratio(x)
    PR = participation_ratio(x)
    U = uniformity_cosine(x)
    eig = eigen_residual(f, x)

    healthy = (
        eig["residual"] <= 1e-9
        and (np.log(d) - 0.6) <= H <= (np.log(d) - 0.1)
        and 0.7 <= rho <= 1.3
        and PR >= 0.3 * d
        and U <= 0.1
    )
    return {
        "lambda": eig["lambda"],
        "eigen_residual": eig["residual"],
        "entropy": H,
        "variance_ratio": rho,
        "participation_ratio": PR,
        "uniformity_cosine": U,
        "healthy": bool(healthy),
    }