def lambda_from_ethics(attention_stability: float, harm_avoidance: float, integrity_score: float) -> float:
    # simple bounded average in [0,1]
    vals = [max(0.0, min(1.0, v)) for v in (attention_stability, harm_avoidance, integrity_score)]
    return sum(vals) / 3.0


