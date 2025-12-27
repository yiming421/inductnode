"""
GraphPFN: TabPFN-style architecture for graph learning.

This module implements a TabPFN-inspired transformer architecture
for in-context learning on graphs.
"""

from .config import GraphPFNConfig
from .predictor import GraphPFNPredictor

__all__ = [
    'GraphPFNConfig',
    'GraphPFNPredictor',
]
