"""Core orchestration for Geminoria query handling."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

import supybot.conf as conf
import supybot.ircdb as ircdb
import supybot.log as log
import supybot.ircutils as ircutils

from google.genai import types as gtypes

from .. import __version__ as PLUGIN_VERSION
from ..config.config_runtime import RuntimeConfig, load_runtime_config
from ..state.cache import CacheRepository
from ..state.memory import MemoryStore
from .services import GeminiService
from .system import SYSTEM_INSTRUCTION, gen_config, make_tools
from .textutils import (
    clean_output,
    highlight_config_keys,
    loggable_args,
    loggable_text,
    redact_sensitive,
    sanitize_irc_text,
    truncate,
    run_with_delayed_indicator,
)

_QUERY_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_SCOPE_HINTS = frozenset(
    {
        "limnoria",
        "supybot",
        "geminoria",
        "plugin",
        "plugins",
        "config",
        "configuration",
        "setting",
        "settings",
        "capability",
        "capabilities",
        "anticapability",
        "anti",
        "command",
        "commands",
        "channel",
        "channels",
        "irc",
        "nick",
        "owner",
        "admin",
        "flood",
        "abuse",
    }
)
_OUT_OF_SCOPE_REPLY = (
    "Geminoria only answers Limnoria bot questions (config, commands, "
    "capabilities, and channel history tools)."
)


def gemversion_reply_text() -> str:
    model = load_runtime_config().model
    return f"Geminoria version: {PLUGIN_VERSION} | model: {model}"


def _walk_config(
    node: Any, path: str, word: str, results: list[tuple[str, bool]]
) -> None:
    node_name = getattr(node, "_name", None)
    is_leaf = bool(getattr(node, "_supplyDefault", False))
    if node_name and word in path.lower():
        results.append((path, is_leaf))
    children = getattr(node, "_children", {}) or {}
    for key in sorted(children.keys()):
        child = children[key]
        full = f"{path}.{key}"
        _walk_config(child, full, word, results)


def _format_config_matches(matches: list[str]) -> str:
    return "\n".join(f"- {m}" for m in matches)


def _partition_config_results(
    rows: list[tuple[str, bool]],
) -> tuple[list[str], list[str], list[str]]:
    deduped = list(dict.fromkeys(path for path, _ in rows))
    leaf_rows = [path for path, is_leaf in rows if is_leaf]
    leaf = list(dict.fromkeys(leaf_rows))
    leaf_set = set(leaf)
    parent = [path for path in deduped if path not in leaf_set]
    ordered = leaf + parent
    return leaf, parent, ordered


class GeminoriaCore:
    def __init__(
        self,
        *,
        cache_db_path: str,
        service: GeminiService,
        channel_flag_getter: Callable[[str, str, str], bool],
    ) -> None:
        self._service = service
        self._channel_flag_getter = channel_flag_getter
        self._memory = MemoryStore()
        self._cache = CacheRepository(cache_db_path)
        self._config_index: tuple[tuple[str, ...], tuple[str, ...]] = (tuple(), tuple())
        self._rebuild_config_index()

    def close(self) -> None:
        self._service.close()

    def load_cfg(self) -> RuntimeConfig:
        return load_runtime_config()

    def on_privmsg(self, irc, msg, cfg: RuntimeConfig) -> None:
        channel = msg.args[0]
        if not irc.isChannel(channel):
            return
        nick = msg.nick
        text = msg.args[1]
        self._memory.add_message(
            channel,
            nick,
            text,
            int(cfg.buffer_size),
            re.compile(r"https?://\S+", re.IGNORECASE),
        )

    def check_capability(self, msg, cfg: RuntimeConfig) -> bool:
        if not cfg.required_cap:
            return True
        try:
            allowed = ircdb.checkCapability(msg.prefix, cfg.required_cap)
            if not allowed:
                log.debug(
                    "Geminoria: capability denied prefix=%s required_cap=%s",
                    msg.prefix,
                    cfg.required_cap,
                )
            return allowed
        except Exception:
            return False

    def check_cache_admin(self, msg) -> bool:
        try:
            return bool(
                ircdb.checkCapability(msg.prefix, "admin")
                or ircdb.checkCapability(msg.prefix, "owner")
            )
        except Exception:
            return False

    def tool_enabled(
        self,
        tool_name: str,
        *,
        channel: Optional[str],
        network: str,
        cfg: RuntimeConfig,
    ) -> bool:
        return self._memory.tool_enabled(
            tool_name,
            channel=channel,
            network=network,
            cfg=cfg,
            channel_flag_getter=self._channel_flag_getter,
        )

    def acquire_request_slot(self, msg, cfg: RuntimeConfig) -> Optional[str]:
        channel = (
            msg.args[0]
            if msg.args and msg.args[0] and msg.args[0][0] in "#&+!"
            else None
        )
        return self._memory.acquire_request_slot(
            prefix=msg.prefix,
            channel=channel,
            cooldown_seconds=int(cfg.cooldown_seconds),
            max_concurrent_per_channel=int(cfg.max_concurrent_per_channel),
        )

    def release_request_slot(self, msg) -> None:
        channel = (
            msg.args[0]
            if msg.args and msg.args[0] and msg.args[0][0] in "#&+!"
            else None
        )
        self._memory.release_request_slot(channel)

    def cache_stats(self, cfg: RuntimeConfig) -> str:
        return self._cache.stats(cfg)

    def cache_clear(self) -> tuple[bool, int]:
        return self._cache.clear()

    def _build_config_index(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        rows: list[tuple[str, bool]] = []
        _walk_config(conf.supybot, "supybot", "", rows)
        leaf, parent, _ = _partition_config_results(rows)
        return tuple(leaf), tuple(parent)

    def _rebuild_config_index(self) -> None:
        started = time.monotonic()
        try:
            leaf, parent = self._build_config_index()
            self._config_index = (leaf, parent)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            log.debug(
                "Geminoria: built config index leaf=%s parent=%s ms=%s",
                len(leaf),
                len(parent),
                elapsed_ms,
            )
        except Exception as exc:
            log.warning("Geminoria: config index build failed: %s", exc)
            self._config_index = (tuple(), tuple())

    def _search_config_live(self, word: str) -> tuple[list[str], list[str], list[str]]:
        rows: list[tuple[str, bool]] = []
        _walk_config(conf.supybot, "supybot", word, rows)
        return _partition_config_results(rows)

    def _tool_search_config(self, word: str, limit: int) -> str:
        started = time.monotonic()
        query = (word or "").lower()
        leaf_index, parent_index = self._config_index
        leaf_matches = [full for full in leaf_index if query in full.lower()]
        parent_matches = [full for full in parent_index if query in full.lower()]
        ordered_matches = leaf_matches + parent_matches

        if not ordered_matches:
            try:
                live_leaf, live_parent, live_ordered = self._search_config_live(word)
                if live_ordered:
                    leaf_matches = live_leaf
                    parent_matches = live_parent
                    ordered_matches = live_ordered
                    self._rebuild_config_index()
            except Exception as exc:
                log.error("Geminoria: search_config error: %s", exc)

        log.debug(
            "Geminoria: search_config word=%r limit=%s matches=%s leaf_matches=%s parent_matches=%s ms=%s",
            word,
            limit,
            len(ordered_matches),
            len(leaf_matches),
            len(parent_matches),
            int((time.monotonic() - started) * 1000),
        )
        if not ordered_matches:
            return f"No config variables found containing '{word}'."
        return _format_config_matches(ordered_matches[:limit])

    def _tool_search_commands(self, irc, word: str, limit: int) -> str:
        results: list[str] = []
        try:
            for cb in irc.callbacks:
                name = cb.name()
                for cmd in cb.listCommands():
                    if word.lower() in cmd.lower():
                        results.append(f"{name}.{cmd}")
                    if len(results) >= limit:
                        break
                if len(results) >= limit:
                    break
        except Exception as exc:
            log.error("Geminoria: search_commands error: %s", exc)
        log.debug(
            "Geminoria: search_commands word=%r limit=%s matches=%s",
            word,
            limit,
            len(results),
        )
        if not results:
            return f"No commands found matching '{word}'."
        return "  ".join(results)

    def _execute_tool(
        self,
        *,
        irc,
        channel: Optional[str],
        fn: str,
        tool_args: dict[str, Any],
        limit: int,
        allow_search_last: bool,
        allow_search_urls: bool,
    ) -> str:
        if fn == "search_config":
            return self._tool_search_config(tool_args.get("word", ""), limit)
        if fn == "search_commands":
            return self._tool_search_commands(irc, tool_args.get("word", ""), limit)
        if fn == "search_last":
            if allow_search_last and channel:
                return self._memory.search_last(
                    channel, tool_args.get("text", ""), limit
                )
            return "search_last is disabled for this context."
        if fn == "search_urls":
            if allow_search_urls and channel:
                return self._memory.search_urls(
                    channel, tool_args.get("word", ""), limit
                )
            return "search_urls is disabled for this context."
        return f"Unknown tool: {fn}"

    def _is_in_scope_query(self, irc, query: str) -> bool:
        query_norm = str(query or "").strip().lower()
        if not query_norm:
            return False
        if "supybot." in query_norm:
            return True
        if re.search(r"@[a-z0-9][a-z0-9_-]*", query_norm):
            return True

        tokens = set(_QUERY_TOKEN_RE.findall(query_norm))
        if not tokens:
            return False
        if tokens & _SCOPE_HINTS:
            return True

        try:
            for cb in getattr(irc, "callbacks", []) or []:
                plugin_name = str(cb.name() or "").strip().lower()
                if plugin_name and plugin_name in tokens:
                    return True
                for cmd in cb.listCommands():
                    cmd_name = str(cmd or "").strip().lower()
                    if cmd_name and cmd_name in tokens:
                        return True
        except Exception:
            return False

        return False

    def _run_gemini(self, irc, msg, query: str, cfg: RuntimeConfig) -> str:
        api_key = cfg.api_key
        if not api_key:
            return "Geminoria: API client unavailable - check supybot.plugins.Geminoria.apiKey."

        model = cfg.model
        limit = cfg.max_results
        max_rounds = cfg.max_rounds
        channel = msg.args[0] if irc.isChannel(msg.args[0]) else None
        query_for_model = redact_sensitive(query) if cfg.redact_sensitive else query
        network = str(getattr(irc, "network", "") or "")
        allow_search_last = self.tool_enabled(
            "search_last", channel=channel, network=network, cfg=cfg
        )
        allow_search_urls = self.tool_enabled(
            "search_urls", channel=channel, network=network, cfg=cfg
        )

        log.debug(
            "Geminoria: starting query model=%s channel=%s rounds=%s limit=%s allow_last=%s allow_urls=%s query=%r",
            model,
            channel or "<private>",
            max_rounds,
            limit,
            allow_search_last,
            allow_search_urls,
            loggable_text(query, cfg),
        )

        tool = make_tools(
            cfg.max_results,
            allow_search_last=allow_search_last,
            allow_search_urls=allow_search_urls,
        )
        tool_cfg = gen_config(tools=[tool], systemInstruction=SYSTEM_INSTRUCTION)
        final_cfg = gen_config(systemInstruction=SYSTEM_INSTRUCTION)

        contents = [
            gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=query_for_model)],
            )
        ]
        collected_tool_results: list[str] = []

        for _round in range(max_rounds):
            round_number = _round + 1
            try:
                log.debug(
                    "Geminoria: generate_content round=%s/%s contents=%s",
                    round_number,
                    max_rounds,
                    len(contents),
                )
                response = self._service.generate_content(
                    api_key=api_key,
                    model=model,
                    contents=contents,
                    config=tool_cfg,
                )
            except Exception as exc:
                log.error("Geminoria: generate_content error: %s", exc)
                return f"Gemini error: {exc}"

            candidates = getattr(response, "candidates", None) or []
            log.debug(
                "Geminoria: round=%s candidates=%s response_text=%s",
                round_number,
                len(candidates),
                bool(getattr(response, "text", None)),
            )
            if not candidates:
                return clean_output(getattr(response, "text", None) or "")

            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = list(getattr(content, "parts", None) or [])

            function_calls = [
                function_call
                for p in parts
                if (function_call := getattr(p, "function_call", None)) is not None
            ]

            log.debug(
                "Geminoria: round=%s parts=%s function_calls=%s",
                round_number,
                len(parts),
                len(function_calls),
            )

            if not function_calls:
                text = getattr(response, "text", None) or "".join(
                    text_part
                    for p in parts
                    if (text_part := getattr(p, "text", None)) is not None
                )
                log.debug(
                    "Geminoria: returning text response round=%s text=%r",
                    round_number,
                    loggable_text(text, cfg),
                )
                return clean_output(text)

            if content is not None:
                contents.append(content)
            response_parts = []

            for fc in function_calls:
                fn = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                log.debug(
                    "Geminoria: executing tool round=%s name=%s args=%s",
                    round_number,
                    fn,
                    loggable_args(tool_args, cfg),
                )

                result = self._execute_tool(
                    irc=irc,
                    channel=channel,
                    fn=fn,
                    tool_args=tool_args,
                    limit=limit,
                    allow_search_last=allow_search_last,
                    allow_search_urls=allow_search_urls,
                )

                log.debug(
                    "Geminoria: tool result round=%s name=%s result=%r",
                    round_number,
                    fn,
                    loggable_text(result, cfg),
                )
                result_for_model = (
                    redact_sensitive(result) if cfg.redact_sensitive else result
                )
                collected_tool_results.append(f"{fn}: {result_for_model}")

                response_parts.append(
                    gtypes.Part(
                        function_response=gtypes.FunctionResponse(
                            name=fn,
                            response={"result": result_for_model},
                        )
                    )
                )

            contents.append(gtypes.Content(role="user", parts=response_parts))

        try:
            contents.append(
                gtypes.Content(
                    role="user",
                    parts=[
                        gtypes.Part(
                            text=(
                                "Using the tool results above, answer the original "
                                "question directly in plain text. Do not call any more "
                                "tools. If this is a configuration question, list the "
                                "exact config variable names first, then a brief "
                                "explanation. Keep the reply short for IRC "
                                "(one paragraph, max 2 sentences unless asked for detail)."
                            )
                        )
                    ],
                )
            )
            log.debug(
                "Geminoria: tool-call limit reached, requesting final text-only answer contents=%s",
                len(contents),
            )
            response = self._service.generate_content(
                api_key=api_key,
                model=model,
                contents=contents,
                config=final_cfg,
            )
            text = getattr(response, "text", None) or "".join(
                text_part
                for p in list(
                    getattr(
                        getattr(
                            (getattr(response, "candidates", None) or [None])[0],
                            "content",
                            None,
                        ),
                        "parts",
                        None,
                    )
                    or []
                )
                if (text_part := getattr(p, "text", None)) is not None
            )
            log.debug(
                "Geminoria: tool-call limit reached final text=%r",
                loggable_text(text, cfg),
            )
            if text:
                return clean_output(text)

            if collected_tool_results:
                fallback = " | ".join(collected_tool_results[-limit:])
                log.warning(
                    "Geminoria: final synthesis returned no text; using tool-result fallback."
                )
                return clean_output(fallback)

            return "No answer produced."
        except Exception as exc:
            log.error("Geminoria: final synthesis error: %s", exc)
            if collected_tool_results:
                fallback = " | ".join(collected_tool_results[-limit:])
                log.warning(
                    "Geminoria: final synthesis failed; using tool-result fallback."
                )
                return clean_output(fallback)
            return "No answer produced within the tool-call limit."

    def handle_query(
        self, irc, msg, query: str, *, emit_progress: Callable[[], None]
    ) -> str:
        started_total = time.monotonic()
        cfg = self.load_cfg()
        if not self._is_in_scope_query(irc, query):
            log.debug(
                "Geminoria: query rejected by scope gate query=%r",
                loggable_text(query, cfg),
            )
            return _OUT_OF_SCOPE_REPLY
        channel = msg.args[0] if msg.args and irc.isChannel(msg.args[0]) else None
        model = cfg.model
        network = str(getattr(irc, "network", "") or "")
        allow_search_last = self.tool_enabled(
            "search_last", channel=channel, network=network, cfg=cfg
        )
        allow_search_urls = self.tool_enabled(
            "search_urls", channel=channel, network=network, cfg=cfg
        )
        query_for_cache = redact_sensitive(query) if cfg.redact_sensitive else query

        cache_started = time.monotonic()
        answer = self._cache.lookup(
            cfg,
            network=network,
            channel=channel,
            model=model,
            allow_search_last=allow_search_last,
            allow_search_urls=allow_search_urls,
            query=query_for_cache,
        )
        cache_lookup_ms = int((time.monotonic() - cache_started) * 1000)
        cache_hit = answer is not None

        if answer is None:
            run_started = time.monotonic()
            if cfg.progress_indicator_enabled:
                try:
                    answer = run_with_delayed_indicator(
                        lambda: self._run_gemini(irc, msg, query, cfg),
                        emit_progress,
                        int(cfg.progress_indicator_delay_ms),
                    )
                except Exception as exc:
                    log.error("Geminoria: run-with-indicator error: %s", exc)
                    answer = f"Gemini error: {exc}"
            else:
                answer = self._run_gemini(irc, msg, query, cfg)
            run_gemini_ms = int((time.monotonic() - run_started) * 1000)
            response_for_cache = (
                redact_sensitive(answer) if cfg.redact_sensitive else answer
            )

            store_started = time.monotonic()
            self._cache.store(
                cfg,
                network=network,
                channel=channel,
                model=model,
                allow_search_last=allow_search_last,
                allow_search_urls=allow_search_urls,
                query=query_for_cache,
                response=response_for_cache,
            )
            cache_store_ms = int((time.monotonic() - store_started) * 1000)
            log.debug(
                "Geminoria: timings cache_hit=%s cache_lookup_ms=%s run_gemini_ms=%s cache_store_ms=%s total_ms=%s",
                cache_hit,
                cache_lookup_ms,
                run_gemini_ms,
                cache_store_ms,
                int((time.monotonic() - started_total) * 1000),
            )
        elif cfg.cache_prefix_hits:
            answer = f"[cached] {answer}"
            log.debug(
                "Geminoria: timings cache_hit=%s cache_lookup_ms=%s total_ms=%s",
                cache_hit,
                cache_lookup_ms,
                int((time.monotonic() - started_total) * 1000),
            )

        answer = sanitize_irc_text(answer)
        if cfg.disable_ansi:
            answer = ircutils.stripFormatting(answer)
        else:
            answer = highlight_config_keys(answer)
        max_reply_chars = int(cfg.max_reply_chars)
        if max_reply_chars > 0:
            answer = truncate(answer, max_reply_chars)
        log.debug("Geminoria: replying text=%r", loggable_text(answer, cfg))
        return answer
