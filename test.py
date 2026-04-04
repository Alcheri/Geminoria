###
# Copyright © MMXXV, Barry Suridge
# All rights reserved.
###

import unittest

from . import plugin


class GeminoriaSmokeTestCase(unittest.TestCase):
    def test_plugin_module_exports_class(self):
        self.assertTrue(hasattr(plugin, "Class"))


# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
