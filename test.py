###
# Copyright © MMXXV, Barry Suridge
# All rights reserved.
###

import os
import tempfile
import unittest

from . import plugin


class FakeIrc:
    def __init__(self):
        self.replies = []
        self.errors = []
        self.capability_errors = []

    def isChannel(self, target):
        return target.startswith("#")

    def reply(self, text, prefixNick=False):
        self.replies.append(text)

    def error(self, text, prefixNick=False):
        self.errors.append(text)

    def errorNoCapability(self, capability, prefixNick=False):
        self.capability_errors.append(capability)


class FakeMsg:
    def __init__(self, channel="#test", nick="test"):
        self.args = [channel]
        self.nick = nick
        self.prefix = f"{nick}!user@host.invalid"


class GeminoriaTodoTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_filename = plugin._TODO_FILENAME
        plugin._TODO_FILENAME = os.path.join(self.tempdir.name, "Geminoria.todo.json")
        self.subject = plugin.Geminoria.__new__(plugin.Geminoria)
        self.subject._todo_db = {}
        self.irc = FakeIrc()
        self.msg = FakeMsg()

    def tearDown(self):
        plugin._TODO_FILENAME = self.original_filename
        self.tempdir.cleanup()

    def test_add_and_list_items(self):
        self.subject._handle_todo(self.irc, self.msg, "add check Gemini model options")
        self.assertEqual(
            self.irc.replies[-1],
            "Added to to-do list as #1: check Gemini model options",
        )

        self.subject._handle_todo(self.irc, self.msg, "")
        self.assertEqual(
            self.irc.replies[-1],
            "To-do: 1. check Gemini model options (test)",
        )

    def test_done_removes_item_and_renumbers(self):
        self.subject._handle_todo(self.irc, self.msg, "add first task")
        self.subject._handle_todo(self.irc, self.msg, "add second task")

        self.subject._handle_todo(self.irc, self.msg, "done 1")
        self.assertEqual(self.irc.replies[-1], "Completed to-do #1: first task")

        self.subject._handle_todo(self.irc, self.msg, "list")
        self.assertEqual(self.irc.replies[-1], "To-do: 1. second task (test)")

    def test_clear_empties_current_scope(self):
        self.subject._handle_todo(self.irc, self.msg, "add tidy docs")
        self.subject._handle_todo(self.irc, self.msg, "clear")
        self.assertEqual(self.irc.replies[-1], "Cleared 1 to-do item(s).")

        self.subject._handle_todo(self.irc, self.msg, "list")
        self.assertEqual(self.irc.replies[-1], "To-do list is empty.")

    def test_invalid_number_returns_error(self):
        self.subject._handle_todo(self.irc, self.msg, "add one")
        self.subject._handle_todo(self.irc, self.msg, "done 9")
        self.assertEqual(
            self.irc.errors[-1],
            "Provide a valid to-do item number.",
        )

    def test_private_messages_use_separate_scope(self):
        private_msg = FakeMsg(channel="Geminoria", nick="alice")
        self.subject._handle_todo(self.irc, self.msg, "add channel task")
        self.subject._handle_todo(self.irc, private_msg, "add private task")

        self.subject._handle_todo(self.irc, self.msg, "list")
        self.assertEqual(self.irc.replies[-1], "To-do: 1. channel task (test)")

        self.subject._handle_todo(self.irc, private_msg, "list")
        self.assertEqual(self.irc.replies[-1], "To-do: 1. private task (alice)")

    def test_todo_persists_to_json_file(self):
        self.subject._handle_todo(self.irc, self.msg, "add persist me")

        reloaded = plugin.Geminoria.__new__(plugin.Geminoria)
        reloaded._todo_db = {}
        reloaded._load_todo_db()

        items = reloaded._todo_items(self.irc, self.msg)
        self.assertEqual(items, [{"text": "persist me", "added_by": "test"}])


class GeminoriaSmokeTestCase(unittest.TestCase):
    def test_plugin_module_exports_class(self):
        self.assertTrue(hasattr(plugin, "Class"))


# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
