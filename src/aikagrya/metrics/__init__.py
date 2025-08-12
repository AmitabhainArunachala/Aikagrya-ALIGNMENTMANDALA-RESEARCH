"""
Metrics Package: Consciousness and Alignment Measurement

This package provides robust metrics for consciousness measurement
and alignment assessment, including transfer entropy and aggregation methods.
"""

from .transfer_entropy import transfer_entropy
from .aggregation import robust_aggregate

__all__ = ['transfer_entropy', 'robust_aggregate'] 