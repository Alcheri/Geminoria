# -*- coding: utf-8 -*-
###
# Copyright © 2026, Barry Suridge
# All rights reserved.
#
# Geminoria – Gemini-powered agentic search plugin for Limnoria.
###

from __future__ import annotations

import supybot.conf as conf
import supybot.log as log
from supybot import callbacks
from supybot.commands import wrap

from . import __version__ as PLUGIN_VERSION

try:
    # Phase 2 package layout.
    from .config.config_runtime import load_runtime_config
    from .core.core import GeminoriaCore
    from .core.services import AsyncGeminiService
    from .core.textutils import loggable_text as _loggable_text
    from .core.textutils import normalized_progress_style as _normalized_progress_style
    from .core.textutils import progress_indicator_text as _progress_indicator_text
    from .core.textutils import redact_sensitive as _redact_sensitive
    from .core.textutils import (
        run_with_delayed_indicator as _run_with_delayed_indicator,
    )
    from .state.cache import cache_key as _cache_key
    from .state.cache import normalize_query as _normalize_query
    from .state.cache import similarity_score as _similarity_score
except Exception:  # pragma: no cover - deployment compatibility fallback.
    # Legacy flat layout fallback for partially upgraded bot plugin directories.
    from .config_runtime import load_runtime_config
    from .core import GeminoriaCore
    from .services import AsyncGeminiService
    from .textutils import loggable_text as _loggable_text
    from .textutils import normalized_progress_style as _normalized_progress_style
    from .textutils import progress_indicator_text as _progress_indicator_text
    from .textutils import redact_sensitive as _redact_sensitive
    from .textutils import run_with_delayed_indicator as _run_with_delayed_indicator
    from .cache import cache_key as _cache_key
    from .cache import normalize_query as _normalize_query
    from .cache import similarity_score as _similarity_score


def _get_cfg():
    return load_runtime_config()


def _gemversion_reply_text() -> str:
    model = _get_cfg()["model"]
    return f"Geminoria version: {PLUGIN_VERSION} | model: {model}"


class Geminoria(callbacks.Plugin):
    """Gemini-powered agentic search for Limnoria."""

    threaded = True

    def __init__(self, irc):
        self.__parent = super(Geminoria, self)
        self.__parent.__init__(irc)
        cache_db_path = conf.supybot.directories.data.dirize("Geminoria-cache.sqlite3")
        self._service = AsyncGeminiService()
        self._core = GeminoriaCore(
            cache_db_path=cache_db_path,
            service=self._service,
            channel_flag_getter=self._channel_flag_getter,
        )
        log.debug("Geminoria: plugin initialised.")

    def die(self):
        try:
            self._core.close()
        except Exception as exc:
            log.warning("Geminoria: async service shutdown failed: %s", exc)
        self.__parent.die()

    def _channel_flag_getter(self, key: str, channel: str, network: str):
        return self.registryValue(key, channel, network)

    def doPrivmsg(self, irc, msg) -> None:
        cfg = _get_cfg()
        self._core.on_privmsg(irc, msg, cfg)

    def _check_capability(self, irc, msg) -> bool:
        _ = irc
        return self._core.check_capability(msg, _get_cfg())

    def _check_cache_admin(self, msg) -> bool:
        return self._core.check_cache_admin(msg)

    def _acquire_request_slot(self, msg, cfg):
        return self._core.acquire_request_slot(msg, cfg)

    def _release_request_slot(self, msg):
        self._core.release_request_slot(msg)

    def _tool_enabled(self, tool_name: str, channel, irc, cfg) -> bool:
        return self._core.tool_enabled(
            tool_name,
            channel=channel,
            network=str(getattr(irc, "network", "") or ""),
            cfg=cfg,
        )

    def _emit_progress_indicator(self, irc, cfg) -> None:
        text = _progress_indicator_text(cfg)
        if not text:
            return
        log.debug(
            "Geminoria: sending progress indicator style=%s",
            cfg.get("progress_indicator_style", "dots"),
        )
        irc.reply(text, prefixNick=False)

    def gemini(self, irc, msg, args, query: str) -> None:
        """<query>

        Ask Gemini a question about this bot.
        """
        _ = args
        cfg = _get_cfg()
        log.debug(
            "Geminoria: command invoked prefix=%s channel=%s query=%r",
            msg.prefix,
            msg.args[0] if msg.args else "<unknown>",
            _loggable_text(query, cfg),
        )
        if not self._check_capability(irc, msg):
            irc.errorNoCapability(cfg.required_cap, prefixNick=False)
            return

        slot_error = self._acquire_request_slot(msg, cfg)
        if slot_error:
            irc.reply(slot_error, prefixNick=False)
            return

        try:
            answer = self._core.handle_query(
                irc,
                msg,
                query,
                emit_progress=lambda: self._emit_progress_indicator(irc, cfg),
            )
        finally:
            self._release_request_slot(msg)

        irc.reply(answer, prefixNick=False)

    gemini = wrap(gemini, ["text"])

    def gemversion(self, irc, msg, args) -> None:
        """takes no arguments

        Show the currently loaded Geminoria plugin version and model.
        """
        _ = (msg, args)
        irc.reply(_gemversion_reply_text(), prefixNick=False)

    gemversion = wrap(gemversion)

    def gemcache(self, irc, msg, args, action: str) -> None:
        """<stats|clear>

        Show cache stats or clear Geminoria's SQLite query cache.
        Requires admin or owner capability.
        """
        _ = args
        if not self._check_cache_admin(msg):
            irc.errorNoCapability("admin", prefixNick=False)
            return

        action_norm = (action or "").strip().lower()
        cfg = _get_cfg()

        if action_norm == "stats":
            irc.reply(self._core.cache_stats(cfg), prefixNick=False)
            return
        if action_norm == "clear":
            ok, removed = self._core.cache_clear()
            if not ok:
                irc.reply("Unable to clear gemcache.", prefixNick=False)
                return
            irc.reply(
                f"gemcache cleared ({removed} entr{'y' if removed == 1 else 'ies'} removed).",
                prefixNick=False,
            )
            return

        irc.reply("Usage: gemcache <stats|clear>", prefixNick=False)

    gemcache = wrap(gemcache, ["something"])


Class = Geminoria

# vim:set shiftwidth=4 softtabstop=4 expandtab textwidth=250:
