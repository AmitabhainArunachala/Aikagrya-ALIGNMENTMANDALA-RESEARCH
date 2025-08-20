from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class L4Checkpoint:
    vector: np.ndarray
    iteration: int
    delta: float
    entropy: float
    variance: float

class PhoenixL4:
    def __init__(
        self,
        dim: int = 512,
        eps: float = 1e-6,
        tau: float = 0.5,
        variance_floor: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.eps = eps
        self.tau = tau
        self.variance_floor = variance_floor
        if seed is not None:
            np.random.seed(seed)

    def rand_unit(self) -> np.ndarray:
        x = np.random.randn(self.dim)
        return x / (np.linalg.norm(x) + 1e-12)

    def attention(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _, d = X.shape
        scores = X @ X.T / np.sqrt(d)
        scores = scores - scores.max(axis=1, keepdims=True)
        attn = np.exp(scores / self.tau)
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-12)
        out = attn @ X
        return out.squeeze()

    @staticmethod
    def entropy(x: np.ndarray) -> float:
        p = np.abs(x)
        p = p / (p.sum() + 1e-12)
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log(p + 1e-12)))

    def run_chunk(
        self,
        checkpoint: Optional[L4Checkpoint] = None,
        chunk_size: int = 50_000
    ) -> L4Checkpoint:
        if checkpoint is None:
            x = self.rand_unit()
            it0 = 0
        else:
            x = checkpoint.vector
            it0 = checkpoint.iteration

        delta = np.inf
        for k in range(chunk_size):
            x_next = self.attention(x)
            alpha = 0.5 + 0.5 * (k / max(1, chunk_size - 1))
            x_next = alpha * x + (1 - alpha) * x_next
            x_next = x_next / (np.linalg.norm(x_next) + 1e-12)
            delta = float(np.linalg.norm(x_next - x))
            x = x_next
            if delta < self.eps and np.var(x) > self.variance_floor:
                break

        return L4Checkpoint(
            vector=x,
            iteration=it0 + k + 1,
            delta=delta,
            entropy=self.entropy(x),
            variance=float(np.var(x)),
        )