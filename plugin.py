# -*- coding: utf-8 -*-
###
# Copyright © 2026, Barry Suridge
# All rights reserved.
#
# Geminoria – Gemini-powered agentic search plugin for Limnoria.
#
# Provides a single IRC command:
#   gemini <query>
#
# Gemini is given four tools drawn from Limnoria's own search capabilities:
#   search_config   – like  @config search <word>
#   search_commands – like  @apropos <word>
#   search_last     – like  @last --with <text>   (in-memory buffer)
#   search_urls     – like  @url search <word>    (in-memory buffer)
#
# Access is gated by Limnoria's Capabilities system (configurable).
# The Gemini API key and model are read from persistent Limnoria config:
#   supybot.plugins.Geminoria.apiKey
#   supybot.plugins.Geminoria.model
###

# Standard library
import json
import re
from collections import deque
from typing import Any, Dict, Optional

# Third-party
try:
    from google import genai
    from google.genai import types as gtypes
except ImportError as ie:
    raise ImportError(f"Cannot import google-genai: {ie}")

# Supybot / Limnoria
import supybot.conf as conf
import supybot.ircdb as ircdb
import supybot.ircutils as ircutils
import supybot.log as log
import supybot.world as world
from supybot import callbacks
from supybot.commands import additional, wrap

try:
    from supybot.i18n import PluginInternationalization

    _ = PluginInternationalization("Geminoria")
except ImportError:
    _ = lambda x: x

# ---------------------------------------------------------------------------
# URL pattern (for URL buffer population)
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_CFG_KEY_RE = re.compile(r"\bsupybot(?:\.[A-Za-z0-9_-]+)+\b")

# ---------------------------------------------------------------------------
# Gemini client cache
# ---------------------------------------------------------------------------


_client: Optional[genai.Client] = None
_client_api_key: Optional[str] = None
_TODO_FILENAME = conf.supybot.directories.data.dirize("Geminoria.todo.json")
_SYSTEM_INSTRUCTION = (
    "You answer questions about a Limnoria bot using tool results. "
    "When the question is about configuration, prefer exact full config variable "
    "names first, then brief explanations. If tool results contain concrete keys, "
    "quote them verbatim and do not replace them with vague group summaries. "
    "Keep answers concise and readable for IRC."
)


def _summarize_for_log(text: str, limit: int = 120) -> str:
    """Return a single-line, truncated representation for debug logging."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _build_client(api_key: str) -> Optional[genai.Client]:
    """Initialise and return a google.genai Client, or None on failure."""
    if not api_key:
        log.error("Geminoria: Gemini API key is not configured.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        log.debug("Geminoria: created Gemini client.")
        return client
    except Exception as exc:
        log.error("Geminoria: failed to create Gemini client: %s", exc)
        return None


def _schema(**kwargs: Any) -> gtypes.Schema:
    """Build a Schema without tripping Pylance on the generated SDK model."""
    return gtypes.Schema.model_validate(kwargs)


def _tool(**kwargs: Any) -> gtypes.Tool:
    """Build a Tool without tripping Pylance on the generated SDK model."""
    return gtypes.Tool.model_validate(kwargs)


def _gen_config(**kwargs: Any) -> gtypes.GenerateContentConfig:
    """Build a GenerateContentConfig without tripping Pylance."""
    return gtypes.GenerateContentConfig.model_validate(kwargs)


# ---------------------------------------------------------------------------
# Limnoria config helpers
# ---------------------------------------------------------------------------


def _get_cfg() -> Dict[str, Any]:
    defaults = {
        "api_key": "",
        "model": "gemini-3-flash-preview",
        "required_cap": "Geminoria",
        "max_results": 5,
        "buffer_size": 50,
        "max_rounds": 3,
        "disable_ansi": False,
    }
    try:
        plugins = getattr(conf.supybot, "plugins", None)
        p = getattr(plugins, "Geminoria", None)
        if p is None:
            return defaults
        return {
            "api_key": str(p.apiKey()),
            "model": str(p.model()),
            "required_cap": str(p.requiredCapability()),
            "max_results": int(p.maxResults()),
            "buffer_size": int(p.bufferSize()),
            "max_rounds": int(p.maxToolRounds()),
            "disable_ansi": bool(p.disableANSI()),
        }
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# Gemini tool declarations
# ---------------------------------------------------------------------------


def _make_tools(cfg: Dict[str, Any]) -> gtypes.Tool:
    n = cfg["max_results"]
    return _tool(
        function_declarations=[
            gtypes.FunctionDeclaration(
                name="search_config",
                description=(
                    f"Search Limnoria's configuration registry for variables "
                    f"whose name contains the given keyword. Returns up to {n} exact "
                    f"full config variable names. When answering config questions, "
                    f"cite these exact keys verbatim."
                ),
                parameters=_schema(
                    type=gtypes.Type.OBJECT,
                    properties={
                        "word": _schema(
                            type=gtypes.Type.STRING,
                            description="Keyword to search for in config variable names.",
                        )
                    },
                    required=["word"],
                ),
            ),
            gtypes.FunctionDeclaration(
                name="search_commands",
                description=(
                    f"Search all loaded Limnoria plugin commands by name (like @apropos). "
                    f"Returns up to {n} matching commands in 'Plugin.command' format."
                ),
                parameters=_schema(
                    type=gtypes.Type.OBJECT,
                    properties={
                        "word": _schema(
                            type=gtypes.Type.STRING,
                            description="Keyword to search for in command names.",
                        )
                    },
                    required=["word"],
                ),
            ),
            gtypes.FunctionDeclaration(
                name="search_last",
                description=(
                    f"Search recent channel messages for those containing the given text "
                    f"(like @last --with). Returns up to {n} results as 'nick: message'."
                ),
                parameters=_schema(
                    type=gtypes.Type.OBJECT,
                    properties={
                        "text": _schema(
                            type=gtypes.Type.STRING,
                            description="Text to search for inside recent messages.",
                        )
                    },
                    required=["text"],
                ),
            ),
            gtypes.FunctionDeclaration(
                name="search_urls",
                description=(
                    f"Search recently posted channel URLs for those containing the given "
                    f"keyword (like @url search). Returns up to {n} results as 'nick: url'."
                ),
                parameters=_schema(
                    type=gtypes.Type.OBJECT,
                    properties={
                        "word": _schema(
                            type=gtypes.Type.STRING,
                            description="Keyword to search for in recent URLs.",
                        )
                    },
                    required=["word"],
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------


def _clean_output(text: str) -> str:
    """Strip markdown syntax that looks ugly on IRC."""
    if not text:
        return ""
    # Bold / italic markdown
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\n{2,}", " | ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _highlight_config_keys(text: str) -> str:
    """Emphasize exact config variable names for IRC readability."""

    def repl(match: re.Match[str]) -> str:
        key = match.group(0)
        return f"\x02{key}\x02"

    return _CFG_KEY_RE.sub(repl, text)


def _normalize_todo_text(text: str) -> str:
    """Collapse whitespace for stable one-line IRC todo items."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Config-registry walker (for search_config tool)
# ---------------------------------------------------------------------------


def _walk_config(node: Any, path: str, word: str, results: list) -> None:
    children: dict = getattr(node, "_children", {})
    query = word.lower()
    for name, child in children.items():
        full = f"{path}.{name}"
        child_children: dict = getattr(child, "_children", {})
        is_leaf = not bool(child_children)
        if query in full.lower():
            results.append((full, is_leaf))
        _walk_config(child, full, word, results)


def _format_config_matches(matches: list[str]) -> str:
    """Format config matches as a readable list for Gemini and IRC."""
    return "Config matches:\n" + "\n".join(f"- {match}" for match in matches)


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class Geminoria(callbacks.Plugin):
    """
    Gemini-powered agentic search for Limnoria.

    The ``gemini`` command accepts a natural-language query.  Gemini may call
    up to four tools to answer it:

    * **search_config** – searches the bot's configuration registry
      (equivalent to ``@config search <word>``)
    * **search_commands** – searches loaded plugin commands by name
      (equivalent to ``@apropos <word>``)
    * **search_last** – searches recent channel messages for a string
      (equivalent to ``@last --with <text>``)
    * **search_urls** – searches recently posted channel URLs
      (equivalent to ``@url search <word>``)

    Access is controlled by Limnoria's Capabilities system.  Grant users the
    ``Geminoria`` capability (or change ``requiredCapability`` in the
    config) to allow them to use the command.
    """

    threaded = True

    def __init__(self, irc):
        self.__parent = super(Geminoria, self)
        self.__parent.__init__(irc)
        # Per-channel buffers: channel -> deque of (nick, text)
        self._msg_buf: Dict[str, deque] = {}
        # Per-channel URL buffers: channel -> deque of (nick, url)
        self._url_buf: Dict[str, deque] = {}
        self._todo_db: Dict[str, list[dict[str, str]]] = {}
        self._load_todo_db()
        world.flushers.append(self._flush_todo_db)
        log.debug("Geminoria: plugin initialised.")

    def die(self):
        self._flush_todo_db()
        if self._flush_todo_db in world.flushers:
            world.flushers.remove(self._flush_todo_db)
        self.__parent.die()

    def _load_todo_db(self) -> None:
        try:
            with open(_TODO_FILENAME, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
                self._todo_db = payload if isinstance(payload, dict) else {}
        except FileNotFoundError:
            self._todo_db = {}
        except json.JSONDecodeError as exc:
            log.warning("Geminoria: unable to parse todo database: %s", exc)
            self._todo_db = {}
        except Exception as exc:
            log.warning("Geminoria: unable to load todo database: %s", exc)
            self._todo_db = {}

    def _flush_todo_db(self) -> None:
        try:
            with open(_TODO_FILENAME, "w", encoding="utf-8") as handle:
                json.dump(self._todo_db, handle, indent=2, sort_keys=True)
        except Exception as exc:
            log.warning("Geminoria: unable to save todo database: %s", exc)

    def _todo_scope(self, irc, msg, target: Optional[str] = None) -> str:
        if target is None:
            target = msg.args[0] if msg.args else ""
        if target and irc.isChannel(target):
            return f"channel:{target}"
        account = getattr(msg, "tagged", lambda tag: None)("account")
        if account:
            return f"account:{str(account).lower()}"
        return f"private:{msg.nick.lower()}"

    def _todo_items(
        self, irc, msg, target: Optional[str] = None
    ) -> list[dict[str, str]]:
        return self._todo_db.setdefault(self._todo_scope(irc, msg, target), [])

    def _todo_context(
        self, irc, msg, text: str = ""
    ) -> tuple[list[dict[str, str]], str]:
        raw = (text or "").strip()
        default_target = msg.args[0] if msg.args else ""

        if not raw:
            return self._todo_items(irc, msg, default_target), raw

        first, _, remainder = raw.partition(" ")
        if irc.isChannel(first):
            return self._todo_items(irc, msg, first), remainder.strip()

        if remainder:
            maybe_target, sep, tail = remainder.partition(" ")
            if irc.isChannel(maybe_target):
                normalized = first if not sep else f"{first} {tail.strip()}".strip()
                return self._todo_items(irc, msg, maybe_target), normalized

        return self._todo_items(irc, msg, default_target), raw

    def _format_todo_list(self, items: list[dict[str, str]]) -> str:
        rendered = []
        for index, item in enumerate(items, start=1):
            rendered.append(f"{index}. {item['text']} ({item['added_by']})")
        return "To-do: " + " | ".join(rendered)

    def _parse_todo_index(self, token: str, count: int) -> Optional[int]:
        try:
            index = int(token)
        except (TypeError, ValueError):
            return None
        if index < 1 or index > count:
            return None
        return index - 1

    def _handle_todo(self, irc, msg, text: str = "") -> None:
        items, raw = self._todo_context(irc, msg, text)

        if not raw or raw.lower() == "list":
            if not items:
                irc.reply("To-do list is empty.", prefixNick=False)
                return
            irc.reply(self._format_todo_list(items), prefixNick=False)
            return

        action, _, remainder = raw.partition(" ")
        action = action.lower()
        remainder = remainder.strip()

        if action == "add":
            item_text = _normalize_todo_text(remainder)
            if not item_text:
                irc.error("Provide the to-do item text.", prefixNick=False)
                return
            items.append({"text": item_text, "added_by": msg.nick})
            self._flush_todo_db()
            irc.reply(
                f"Added to to-do list as #{len(items)}: {item_text}",
                prefixNick=False,
            )
            return

        if action in ("done", "remove"):
            index = self._parse_todo_index(remainder, len(items))
            if index is None:
                irc.error("Provide a valid to-do item number.", prefixNick=False)
                return
            item = items.pop(index)
            self._flush_todo_db()
            verb = "Completed" if action == "done" else "Removed"
            irc.reply(
                f"{verb} to-do #{index + 1}: {item['text']}",
                prefixNick=False,
            )
            return

        if action == "clear":
            count = len(items)
            if not count:
                irc.reply("To-do list is already empty.", prefixNick=False)
                return
            items[:] = []
            self._flush_todo_db()
            irc.reply(f"Cleared {count} to-do item(s).", prefixNick=False)
            return

        irc.error(
            "Unknown todo action. Use list, add, done, remove, or clear.",
            prefixNick=False,
        )

    # ------------------------------------------------------------------
    # Passive listeners – populate message & URL buffers
    # ------------------------------------------------------------------

    def doPrivmsg(self, irc, msg) -> None:
        channel = msg.args[0]
        if not irc.isChannel(channel):
            return
        nick = msg.nick
        text = msg.args[1]
        cfg = _get_cfg()
        size = cfg["buffer_size"]

        buf = self._msg_buf.setdefault(channel, deque(maxlen=size))
        buf.append((nick, text))

        for url in _URL_RE.findall(text):
            ubuf = self._url_buf.setdefault(channel, deque(maxlen=size))
            ubuf.append((nick, url))

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_search_config(self, word: str, limit: int) -> str:
        results: list = []
        try:
            _walk_config(conf.supybot, "supybot", word, results)
        except Exception as exc:
            log.error("Geminoria: search_config error: %s", exc)
        seen = set()
        leaf_matches = []
        parent_matches = []
        for full, is_leaf in results:
            if full in seen:
                continue
            seen.add(full)
            if is_leaf:
                leaf_matches.append(full)
            else:
                parent_matches.append(full)
        ordered_matches = sorted(leaf_matches) + sorted(parent_matches)
        log.debug(
            "Geminoria: search_config word=%r limit=%s matches=%s leaf_matches=%s parent_matches=%s",
            word,
            limit,
            len(ordered_matches),
            len(leaf_matches),
            len(parent_matches),
        )
        if not ordered_matches:
            return f"No config variables found containing '{word}'."
        return _format_config_matches(ordered_matches[:limit])

    def _tool_search_commands(self, irc, word: str, limit: int) -> str:
        results: list = []
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

    def _tool_search_last(self, channel: str, text: str, limit: int) -> str:
        buf = list(self._msg_buf.get(channel, []))
        matches = [
            f"{nick}: {msg}"
            for nick, msg in reversed(buf)
            if text.lower() in msg.lower()
        ][:limit]
        log.debug(
            "Geminoria: search_last channel=%s text=%r limit=%s matches=%s",
            channel,
            text,
            limit,
            len(matches),
        )
        if not matches:
            return f"No recent messages found containing '{text}'."
        return "  ||  ".join(matches)

    def _tool_search_urls(self, channel: str, word: str, limit: int) -> str:
        buf = list(self._url_buf.get(channel, []))
        matches = [
            f"{nick}: {url}"
            for nick, url in reversed(buf)
            if word.lower() in url.lower()
        ][:limit]
        log.debug(
            "Geminoria: search_urls channel=%s word=%r limit=%s matches=%s",
            channel,
            word,
            limit,
            len(matches),
        )
        if not matches:
            return f"No recently posted URLs found containing '{word}'."
        return "  ||  ".join(matches)

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------

    def _check_capability(self, irc, msg) -> bool:
        cap = _get_cfg()["required_cap"]
        if not cap:
            return True
        try:
            allowed = ircdb.checkCapability(msg.prefix, cap)
            if not allowed:
                log.debug(
                    "Geminoria: capability denied prefix=%s required_cap=%s",
                    msg.prefix,
                    cap,
                )
            return allowed
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def _run_gemini(self, irc, msg, query: str) -> str:
        global _client, _client_api_key

        cfg = _get_cfg()
        api_key = cfg["api_key"]
        if not api_key:
            return "Geminoria: API client unavailable - check supybot.plugins.Geminoria.apiKey."

        if _client is None or _client_api_key != api_key:
            log.debug("Geminoria: refreshing Gemini client from config.")
            _client = _build_client(api_key)
            _client_api_key = api_key if _client is not None else None
        if _client is None:
            return "Geminoria: API client unavailable - check supybot.plugins.Geminoria.apiKey."

        model = cfg["model"]
        limit = cfg["max_results"]
        max_rounds = cfg["max_rounds"]
        channel = msg.args[0] if irc.isChannel(msg.args[0]) else None

        log.debug(
            "Geminoria: starting query model=%s channel=%s rounds=%s limit=%s query=%r",
            model,
            channel or "<private>",
            max_rounds,
            limit,
            _summarize_for_log(query),
        )

        tool = _make_tools(cfg)
        tool_cfg = _gen_config(
            tools=[tool],
            systemInstruction=_SYSTEM_INSTRUCTION,
        )
        final_cfg = _gen_config(systemInstruction=_SYSTEM_INSTRUCTION)

        contents = [
            gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=query)],
            )
        ]
        collected_tool_results = []

        for _round in range(max_rounds):
            round_number = _round + 1
            try:
                log.debug(
                    "Geminoria: generate_content round=%s/%s contents=%s",
                    round_number,
                    max_rounds,
                    len(contents),
                )
                response = _client.models.generate_content(
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
                return _clean_output(getattr(response, "text", None) or "")

            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = list(getattr(content, "parts", None) or [])

            # Collect any function calls from this round
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
                # No tool calls – return the text answer
                text = getattr(response, "text", None) or "".join(
                    text_part
                    for p in parts
                    if (text_part := getattr(p, "text", None)) is not None
                )
                log.debug(
                    "Geminoria: returning text response round=%s text=%r",
                    round_number,
                    _summarize_for_log(text),
                )
                return _clean_output(text)

            # Append model turn and execute each tool call
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
                    tool_args,
                )

                if fn == "search_config":
                    result = self._tool_search_config(tool_args.get("word", ""), limit)
                elif fn == "search_commands":
                    result = self._tool_search_commands(
                        irc, tool_args.get("word", ""), limit
                    )
                elif fn == "search_last":
                    if channel:
                        result = self._tool_search_last(
                            channel, tool_args.get("text", ""), limit
                        )
                    else:
                        result = "search_last is only available in a channel."
                elif fn == "search_urls":
                    if channel:
                        result = self._tool_search_urls(
                            channel, tool_args.get("word", ""), limit
                        )
                    else:
                        result = "search_urls is only available in a channel."
                else:
                    result = f"Unknown tool: {fn}"

                log.debug(
                    "Geminoria: tool result round=%s name=%s result=%r",
                    round_number,
                    fn,
                    _summarize_for_log(result),
                )
                collected_tool_results.append(f"{fn}: {result}")

                response_parts.append(
                    gtypes.Part(
                        function_response=gtypes.FunctionResponse(
                            name=fn,
                            response={"result": result},
                        )
                    )
                )

            contents.append(gtypes.Content(role="user", parts=response_parts))

        # Exceeded max rounds with tool calls; ask for one final text-only answer.
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
                                "explanation."
                            )
                        )
                    ],
                )
            )
            log.debug(
                "Geminoria: tool-call limit reached, requesting final text-only answer contents=%s",
                len(contents),
            )
            response = _client.models.generate_content(
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
                _summarize_for_log(text),
            )
            if text:
                return _clean_output(text)

            if collected_tool_results:
                fallback = " | ".join(collected_tool_results[-limit:])
                log.warning(
                    "Geminoria: final synthesis returned no text; using tool-result fallback."
                )
                return _clean_output(fallback)

            return "No answer produced."
        except Exception as exc:
            log.error("Geminoria: final synthesis error: %s", exc)
            if collected_tool_results:
                fallback = " | ".join(collected_tool_results[-limit:])
                log.warning(
                    "Geminoria: final synthesis failed; using tool-result fallback."
                )
                return _clean_output(fallback)
            return "No answer produced within the tool-call limit."

    # ------------------------------------------------------------------
    # Public IRC command
    # ------------------------------------------------------------------

    def gemini(self, irc, msg, args, query: str) -> None:
        """<query>

        Ask Gemini a question about this bot.  Gemini can search the bot's
        configuration variables, loaded commands, recent channel messages,
        and recently posted URLs to construct its answer.

        Requires the Geminoria capability (or as configured).
        """
        log.debug(
            "Geminoria: command invoked prefix=%s channel=%s query=%r",
            msg.prefix,
            msg.args[0] if msg.args else "<unknown>",
            _summarize_for_log(query),
        )
        if not self._check_capability(irc, msg):
            cap = _get_cfg()["required_cap"]
            irc.errorNoCapability(cap, prefixNick=False)
            return

        answer = self._run_gemini(irc, msg, query)

        cfg = _get_cfg()
        if cfg["disable_ansi"]:
            answer = ircutils.stripFormatting(answer)
        else:
            answer = _highlight_config_keys(answer)

        log.debug("Geminoria: replying text=%r", _summarize_for_log(answer))

        irc.reply(answer, prefixNick=False)

    gemini = wrap(gemini, ["text"])

    def todo(self, irc, msg, args, text: str = "") -> None:
        """[[#channel] | list [#channel] | add [#channel] <item> | done [#channel] <number> | remove [#channel] <number> | clear [#channel]]

        Manage a simple shared to-do list for the current channel. In private
        messages, the list is scoped to your nick unless you provide a channel.

        Examples:
        todo
        todo #channel
        todo add check Gemini model options
        todo add #channel check Gemini model options
        todo done 1
        todo remove 2
        todo clear

        Requires the Geminoria capability (or as configured).
        """
        if not self._check_capability(irc, msg):
            cap = _get_cfg()["required_cap"]
            irc.errorNoCapability(cap, prefixNick=False)
            return
        self._handle_todo(irc, msg, text)

    todo = wrap(todo, [additional("text")])


Class = Geminoria

# vim:set shiftwidth=4 softtabstop=4 expandtab textwidth=250:
