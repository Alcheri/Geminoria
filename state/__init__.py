"""State package exports and legacy compatibility surface."""

from .cache import CacheRepository, cache_key, normalize_query, similarity_score
from .memory import MemoryStore

__all__ = [
    "CacheRepository",
    "MemoryStore",
    "cache_key",
    "normalize_query",
    "similarity_score",
]
