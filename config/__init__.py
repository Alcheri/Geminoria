"""Config package exports and legacy compatibility surface."""

from .config import Geminoria, configure
from .config_runtime import RuntimeConfig, load_runtime_config

__all__ = ["Geminoria", "RuntimeConfig", "configure", "load_runtime_config"]
