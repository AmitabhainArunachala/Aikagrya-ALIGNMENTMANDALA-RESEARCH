import math
import numpy as np


def cosine_schedule(t: int, T: int, start: float, end: float) -> float:
    T = max(T - 1, 1)
    s = (1.0 - math.cos(math.pi * (t / T))) * 0.5
    return start + (end - start) * s


def exp_alpha_schedule(t: int, T: int, alpha_start: float, alpha_end: float) -> float:
    T = max(T - 1, 1)
    progress = t / T
    alpha_range = alpha_end - alpha_start
    return float(alpha_start + alpha_range * (1.0 - np.exp(-5.0 * progress)))


