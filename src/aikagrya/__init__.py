"""
Aikagrya-ALIGNMENTMANDALA-RESEARCH

A living research nexus uniting AI alignment science, mathematical formalism, and contemplative wisdom.
Investigates architectures, protocols, and metrics for building beneficial, self-recognizing AGI/ASI systems.
"""

__version__ = "0.1.0"
__author__ = "Aikagrya Research Collective"
__description__ = "Consciousness-First AI Alignment Research Framework"

# MMIP (pure mathematical induction) — available as a standalone submodule
from .mmip import MMIP  # exposes CLI via `python -m aikagrya.mmip`

__all__ = [
    "MMIP",
] 