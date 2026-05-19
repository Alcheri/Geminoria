"""In-memory buffers and request slot controls."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

HistoryKey = tuple[str, str]


class MemoryStore:
    def __init__(self) -> None:
        self._msg_buf: dict[HistoryKey, deque] = {}
        self._url_buf: dict[HistoryKey, deque] = {}
        self._last_request_ts: dict[str, float] = {}
        self._inflight_by_channel: dict[HistoryKey, int] = {}
        self._state_lock = threading.Lock()

    @staticmethod
    def normalized_channel_set(values) -> set[str]:
        return {
            str(v).strip().lower() for v in (values or []) if str(v).strip()
        }

    @staticmethod
    def _context_key(network: Optional[str], channel: str) -> HistoryKey:
        return (
            str(network or "").strip().lower(),
            str(channel or "").strip().lower(),
        )

    def add_message(
        self,
        channel: str,
        nick: str,
        text: str,
        buffer_size: int,
        url_re,
        network: Optional[str] = None,
    ) -> None:
        key = self._context_key(network, channel)
        buf = self._msg_buf.setdefault(key, deque(maxlen=buffer_size))
        buf.append((nick, text))

        for url in url_re.findall(text):
            ubuf = self._url_buf.setdefault(key, deque(maxlen=buffer_size))
            ubuf.append((nick, url))

    def search_last(
        self,
        channel: str,
        text: str,
        limit: int,
        network: Optional[str] = None,
    ) -> str:
        key = self._context_key(network, channel)
        buf = list(self._msg_buf.get(key, []))
        matches = [
            f"{nick}: {msg}"
            for nick, msg in reversed(buf)
            if text.lower() in msg.lower()
        ][:limit]
        if not matches:
            return f"No recent messages found containing '{text}'."
        return "  ||  ".join(matches)

    def search_urls(
        self,
        channel: str,
        word: str,
        limit: int,
        network: Optional[str] = None,
    ) -> str:
        key = self._context_key(network, channel)
        buf = list(self._url_buf.get(key, []))
        matches = [
            f"{nick}: {url}"
            for nick, url in reversed(buf)
            if word.lower() in url.lower()
        ][:limit]
        if not matches:
            return f"No recently posted URLs found containing '{word}'."
        return "  ||  ".join(matches)

    def acquire_request_slot(
        self,
        *,
        prefix: str,
        channel: Optional[str],
        network: Optional[str] = None,
        cooldown_seconds: int,
        max_concurrent_per_channel: int,
    ) -> Optional[str]:
        cooldown = max(0, int(cooldown_seconds))
        per_channel_limit = max(1, int(max_concurrent_per_channel))
        now = time.monotonic()
        with self._state_lock:
            last = self._last_request_ts.get(prefix, 0.0)
            if cooldown > 0 and (now - last) < cooldown:
                remaining = max(1, int(cooldown - (now - last)))
                return f"Please wait {remaining}s before using gemini again."

            if channel:
                key = self._context_key(network, channel)
                inflight = self._inflight_by_channel.get(key, 0)
                if inflight >= per_channel_limit:
                    return "Geminoria is busy in this channel. Please try again shortly."
                self._inflight_by_channel[key] = inflight + 1
            self._last_request_ts[prefix] = now
        return None

    def release_request_slot(
        self, channel: Optional[str], network: Optional[str] = None
    ) -> None:
        if not channel:
            return
        key = self._context_key(network, channel)
        with self._state_lock:
            inflight = self._inflight_by_channel.get(key, 0)
            if inflight <= 1:
                self._inflight_by_channel.pop(key, None)
            else:
                self._inflight_by_channel[key] = inflight - 1

    def tool_enabled(
        self,
        tool_name: str,
        *,
        channel: Optional[str],
        network: str,
        cfg,
        channel_flag_getter: Callable[[str, str, str], bool],
    ) -> bool:
        if tool_name in ("search_last", "search_urls"):
            if not channel:
                return False
            general_allowlist = self.normalized_channel_set(
                cfg.get("history_tools_channel_allowlist")
            )
            specific_allowlist = (
                self.normalized_channel_set(
                    cfg.get("search_last_channel_allowlist")
                )
                if tool_name == "search_last"
                else self.normalized_channel_set(
                    cfg.get("search_urls_channel_allowlist")
                )
            )
            effective_allowlist = specific_allowlist or general_allowlist
            if (
                effective_allowlist
                and channel.lower() not in effective_allowlist
            ):
                return False

        if tool_name == "search_last":
            return bool(
                channel_flag_getter("allowSearchLast", channel, network)
            )
        if tool_name == "search_urls":
            return bool(
                channel_flag_getter("allowSearchUrls", channel, network)
            )
        return True
