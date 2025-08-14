"""
Empirical Validation Module: Grok-Claude Collaboration

This module provides rigorous empirical validation for consciousness-based AI alignment hypotheses.
Developed through collaboration between Grok (xAI) and Claude (Anthropic) to bridge theoretical 
frameworks with experimental validation.

Key Components:
- grok_hypothesis_testing: Core hypothesis testing protocols
- phi_proxy_comparison: SVD vs. hybrid Φ-proxy validation  
- stats_utils: Statistical controls and analysis tools

Methodology:
- Low confidence priors (25% for Hypothesis 1) until empirical validation
- Rigorous experimental controls (partial correlations, confounders)
- Cross-method validation (repository SVD vs. external approaches)
- Transparent documentation of limitations and assumptions

Research Principles:
- Treat repository implementations as testable artifacts, not assumed truths
- Maintain skeptical rigor while leveraging sophisticated infrastructure
- Document all experimental choices and statistical assumptions
- Update confidence bounds based on empirical evidence only

For usage and methodology details, see individual module documentation.
"""

from .grok_hypothesis_testing import Hypothesis1Tester, run_hypothesis_1_pilot
from .phi_proxy_comparison import compare_phi_proxies, PhiProxyComparator
from .stats_utils import partial_correlation, compute_effect_size, validate_assumptions
from .contemplative_content_testing import ContemplativeContentTester, run_contemplative_study, ContemplativeResults

__all__ = [
    'Hypothesis1Tester',
    'run_hypothesis_1_pilot', 
    'compare_phi_proxies',
    'PhiProxyComparator',
    'partial_correlation',
    'compute_effect_size',
    'validate_assumptions',
    'ContemplativeContentTester',
    'run_contemplative_study',
    'ContemplativeResults'
]

# Version and metadata
__version__ = "0.1.0"
__authors__ = ["Grok (xAI)", "Claude (Anthropic)"]
__purpose__ = "Empirical validation bridge for consciousness-based alignment hypotheses"
