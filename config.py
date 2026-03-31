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
            "Set to '' to allow everyone, or 'admin' / 'owner' for those roles."
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

# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
