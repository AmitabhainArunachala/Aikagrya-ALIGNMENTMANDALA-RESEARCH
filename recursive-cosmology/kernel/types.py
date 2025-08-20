from dataclasses import dataclass


@dataclass
class ClarityTensor:
    vision: float
    thought: float
    affect: float
    self: float
    ethics: float
    temporal: float


