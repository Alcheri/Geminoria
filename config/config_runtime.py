"""Runtime configuration loader for Geminoria."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import supybot.conf as conf


@dataclass
class RuntimeConfig:
    api_key: str = ""
    model: str = "gemini-3-flash-preview"
    required_cap: str = "Geminoria"
    max_results: int = 5
    buffer_size: int = 50
    max_rounds: int = 3
    disable_ansi: bool = False
    redact_sensitive: bool = True
    log_sensitive: bool = False
    cooldown_seconds: int = 10
    max_concurrent_per_channel: int = 1
    max_reply_chars: int = 350
    progress_indicator_enabled: bool = True
    progress_indicator_delay_ms: int = 1200
    progress_indicator_style: str = "dots"
    progress_indicator_message: str = ""
    history_tools_channel_allowlist: list[str] | None = None
    search_last_channel_allowlist: list[str] | None = None
    search_urls_channel_allowlist: list[str] | None = None
    cache_enabled: bool = True
    cache_ttl_seconds: int = 172800
    cache_max_entries: int = 2000
    cache_min_query_length: int = 8
    cache_allow_fuzzy: bool = True
    cache_fuzzy_min_score: int = 92
    cache_prefix_hits: bool = True

    def __post_init__(self) -> None:
        self.history_tools_channel_allowlist = list(
            self.history_tools_channel_allowlist or []
        )
        self.search_last_channel_allowlist = list(
            self.search_last_channel_allowlist or []
        )
        self.search_urls_channel_allowlist = list(
            self.search_urls_channel_allowlist or []
        )
        self.progress_indicator_delay_ms = max(0, int(self.progress_indicator_delay_ms))
        style = str(self.progress_indicator_style or "").strip().lower()
        self.progress_indicator_style = style if style in ("dots", "plain") else "dots"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.as_dict())


def load_runtime_config() -> RuntimeConfig:
    cfg = RuntimeConfig()
    try:
        plugins = getattr(conf.supybot, "plugins", None)
        p = getattr(plugins, "Geminoria", None)
        if p is None:
            return cfg

        return RuntimeConfig(
            api_key=str(p.apiKey()),
            model=str(p.model()),
            required_cap=str(p.requiredCapability()),
            max_results=int(p.maxResults()),
            buffer_size=int(p.bufferSize()),
            max_rounds=int(p.maxToolRounds()),
            disable_ansi=bool(p.disableANSI()),
            redact_sensitive=bool(p.redactSensitiveData()),
            log_sensitive=bool(p.logSensitiveData()),
            cooldown_seconds=int(p.cooldownSeconds()),
            max_concurrent_per_channel=int(p.maxConcurrentPerChannel()),
            max_reply_chars=int(p.maxReplyChars()),
            progress_indicator_enabled=bool(p.progressIndicatorEnabled()),
            progress_indicator_delay_ms=int(p.progressIndicatorDelayMs()),
            progress_indicator_style=str(p.progressIndicatorStyle()),
            progress_indicator_message=str(p.progressIndicatorMessage()),
            history_tools_channel_allowlist=list(p.historyToolsChannelAllowlist()),
            search_last_channel_allowlist=list(p.searchLastChannelAllowlist()),
            search_urls_channel_allowlist=list(p.searchUrlsChannelAllowlist()),
            cache_enabled=bool(p.cacheEnabled()),
            cache_ttl_seconds=int(p.cacheTtlSeconds()),
            cache_max_entries=int(p.cacheMaxEntries()),
            cache_min_query_length=int(p.cacheMinQueryLength()),
            cache_allow_fuzzy=bool(p.cacheAllowFuzzy()),
            cache_fuzzy_min_score=int(p.cacheFuzzyMinScore()),
            cache_prefix_hits=bool(p.cachePrefixHits()),
        )
    except Exception:
        return cfg
