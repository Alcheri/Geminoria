# -*- coding: utf-8 -*-
from supybot import conf, registry

try:
    from supybot.i18n import PluginInternationalization

    _ = PluginInternationalization("Geminoria")
except Exception:
    _ = lambda x: x


def configure(advanced):
    from supybot.questions import expect, anything, something, yn  # noqa: F401

    conf.registerPlugin("Geminoria", True)


Geminoria = conf.registerPlugin("Geminoria")

conf.registerGlobalValue(
    Geminoria,
    "apiKey",
    registry.String(
        "",
        _("Sets the API key for Gemini."),
        private=True,
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "model",
    registry.String(
        "gemini-3-flash-preview",
        _("Gemini model to use for generating responses."),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "requiredCapability",
    registry.String(
        "Geminoria",
        _(
            "Limnoria capability required to use the gemini command. "
            "This follows Limnoria's normal capability evaluation (including "
            "default-allow unless anti-capabilities are set). Set to '' to bypass "
            "explicit checks, or use values like 'admin' / 'owner'."
        ),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "maxResults",
    registry.Integer(
        5,
        _("Maximum number of results returned by each search tool."),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "bufferSize",
    registry.Integer(
        50,
        _(
            "Number of recent channel messages (and URLs) to keep in memory for searching."
        ),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "maxToolRounds",
    registry.Integer(
        3,
        _("Maximum agentic tool-call rounds before returning a partial answer."),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "disableANSI",
    registry.Boolean(
        False,
        _("Strip IRC colour/bold formatting from the final reply."),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "redactSensitiveData",
    registry.Boolean(
        True,
        _(
            "Redact token/password-like values from queries and tool results before "
            "sending them to Gemini."
        ),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "logSensitiveData",
    registry.Boolean(
        False,
        _(
            "Log raw query and tool payloads in debug logs. Leave disabled unless "
            "actively debugging in a trusted environment."
        ),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "cooldownSeconds",
    registry.Integer(
        10,
        _("Minimum seconds between gemini calls from the same user hostmask."),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "maxConcurrentPerChannel",
    registry.Integer(
        1,
        _("Maximum number of in-flight gemini calls allowed per channel."),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "maxReplyChars",
    registry.Integer(
        350,
        _("Maximum final reply length sent to IRC (characters)."),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "historyToolsChannelAllowlist",
    registry.SpaceSeparatedSetOfStrings(
        [],
        _(
            "Optional space-separated channel allowlist for history-based tools "
            "(search_last/search_urls). If empty, all channels are eligible."
        ),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "searchLastChannelAllowlist",
    registry.SpaceSeparatedSetOfStrings(
        [],
        _(
            "Optional space-separated channel allowlist for search_last only. "
            "If empty, search_last falls back to historyToolsChannelAllowlist."
        ),
    ),
)

conf.registerGlobalValue(
    Geminoria,
    "searchUrlsChannelAllowlist",
    registry.SpaceSeparatedSetOfStrings(
        [],
        _(
            "Optional space-separated channel allowlist for search_urls only. "
            "If empty, search_urls falls back to historyToolsChannelAllowlist."
        ),
    ),
)

conf.registerChannelValue(
    Geminoria,
    "allowSearchLast",
    registry.Boolean(
        True,
        _(
            "Allow Geminoria to use search_last (recent channel message search) "
            "in this channel."
        ),
    ),
)

conf.registerChannelValue(
    Geminoria,
    "allowSearchUrls",
    registry.Boolean(
        True,
        _(
            "Allow Geminoria to use search_urls (recent channel URL search) "
            "in this channel."
        ),
    ),
)

# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
