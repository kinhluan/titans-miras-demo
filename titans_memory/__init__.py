"""Titans surprise-based memory mechanism — educational demo."""

from titans_memory.surprise import SurpriseMetric
from titans_memory.memory import MemoryModule
from titans_memory.titans import TitansMemoryLayer
from titans_memory.data import generate_repeating_with_anomalies, generate_frequency_shift

__all__ = [
    "SurpriseMetric",
    "MemoryModule",
    "TitansMemoryLayer",
    "generate_repeating_with_anomalies",
    "generate_frequency_shift",
]
