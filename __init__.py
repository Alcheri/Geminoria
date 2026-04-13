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
from importlib import reload

__version__ = "1.1.0-beta.3"
__author__ = supybot.Author("Barry Suridge", "Alcheri", "barry.suridge@gmail.com")
__contributors__ = {}
__url__ = "https://github.com/Alcheri/Geminoria "

from .config import config as config_module
from . import plugin

reload(config_module)
reload(plugin)

if world.testing:
    try:
        from .tests import test as test_module
    except ImportError as e:
        _ = None
        missing_names = {"test", f"{__name__}.test", f"{__name__}.tests.test"}
        if getattr(e, "name", None) not in missing_names:
            raise
    else:
        reload(test_module)

Class = plugin.Class
configure = config_module.configure

# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
