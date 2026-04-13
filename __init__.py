###
# Copyright © 2026, Barry Suridge
# All rights reserved.
#
###

"""
Geminoria : Gemini-powered search across Limnoria's config, commands,
recent messages, and URLs.  Access is gated by the Capabilities system.
"""

import sys

if sys.version_info < (3, 10):
    raise RuntimeError("This plugin requires Python 3.10 or newer.")

import supybot
from supybot import world

__version__ = "1.1.0-beta.3"
__author__ = supybot.Author("Barry Suridge", "Alcheri", "barry.suridge@gmail.com")
__contributors__ = {}
__url__ = "https://github.com/Alcheri/Geminoria "

from . import config
from . import plugin
from importlib import reload

reload(config)
reload(plugin)

if world.testing:
    try:
        from . import test
    except ImportError as e:
        missing_names = {"test", f"{__name__}.test"}
        if getattr(e, "name", None) not in missing_names:
            raise

Class = plugin.Class
configure = config.configure

# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
